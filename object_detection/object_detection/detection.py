import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from od_msg.srv import SrvDepthPosition
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .yolo import YoloModel


DEPTH_PATCH = 5
DETECTION_TIMEOUT = 8.0
COLOR_TARGET = (80, 220, 80)
COLOR_OTHER = (170, 170, 170)


class ObjectDetectionNode(Node):
    """Own RealSense subscriptions, YOLO inference and target 3-D service."""

    def __init__(self):
        super().__init__("object_detection_node")
        self.declare_parameter("target_set", "tools")
        self.target_set = (
            self.get_parameter("target_set")
            .get_parameter_value()
            .string_value
            .lower()
        )
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.color_frame = None
        self.depth_frame = None
        self.intrinsics = None
        self.detections = []
        self.requested_target = None
        self.inference_error = None

        io_group = ReentrantCallbackGroup()
        inference_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self._color_callback,
            10,
            callback_group=io_group,
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self._depth_callback,
            10,
            callback_group=io_group,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._camera_info_callback,
            10,
            callback_group=io_group,
        )
        self.annotated_publisher = self.create_publisher(
            Image, "/object_detection/annotated_image", 10
        )
        self.create_service(
            SrvDepthPosition,
            "/get_3d_position",
            self._handle_get_position,
            callback_group=io_group,
        )

        self.model = YoloModel(self.target_set)
        self.create_timer(
            0.10, self._infer_once, callback_group=inference_group
        )
        self.get_logger().info(
            f"Object detection ready ({self.target_set}). YOLO classes: "
            + ", ".join(self.model.class_names)
        )

    def _color_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        with self.lock:
            self.color_frame = frame

    def _depth_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        with self.lock:
            self.depth_frame = frame

    def _camera_info_callback(self, message):
        with self.lock:
            self.intrinsics = {
                "fx": message.k[0],
                "fy": message.k[4],
                "ppx": message.k[2],
                "ppy": message.k[5],
            }

    def _infer_once(self):
        with self.lock:
            frame = None if self.color_frame is None else self.color_frame.copy()
            target = self.requested_target
        if frame is None:
            return

        try:
            detections = self.model.predict_frame(frame)
            error = None
        except Exception as exception:
            detections = []
            error = str(exception)
            self.get_logger().error(f"YOLO inference failed: {exception}")

        visual = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["box"])
            selected = detection["name"] == target
            color = COLOR_TARGET if selected else COLOR_OTHER
            thickness = 3 if selected else 1
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                visual,
                f"{detection['name']} {detection['score']:.2f}",
                (x1, max(24, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                thickness,
                cv2.LINE_AA,
            )

        with self.condition:
            self.detections = detections
            self.inference_error = error
            self.condition.notify_all()

        message = self.bridge.cv2_to_imgmsg(visual, encoding="bgr8")
        self.annotated_publisher.publish(message)

    def _handle_get_position(self, request, response):
        target = request.target.lower().strip()
        if target not in self.model.class_names:
            self.get_logger().warn(f"Unsupported target: {target}")
            response.depth_position = [0.0, 0.0, 0.0]
            return response

        self.get_logger().info(f"Looking for target: {target}")
        deadline = time.monotonic() + DETECTION_TIMEOUT
        with self.condition:
            self.requested_target = target
            while rclpy.ok() and time.monotonic() < deadline:
                matches = [
                    item for item in self.detections if item["name"] == target
                ]
                if matches and self.depth_frame is not None and self.intrinsics:
                    detection = max(matches, key=lambda item: item["score"])
                    coords = self._detection_to_camera(detection)
                    if coords is not None:
                        response.depth_position = [float(value) for value in coords]
                        self.get_logger().info(
                            f"{target} camera position: "
                            f"({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})"
                        )
                        self.requested_target = None
                        return response
                self.condition.wait(timeout=0.15)
            self.requested_target = None

        self.get_logger().warn(f"No valid {target} detection")
        response.depth_position = [0.0, 0.0, 0.0]
        return response

    def _detection_to_camera(self, detection):
        x1, y1, x2, y2 = detection["box"]
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))
        depth = self._sample_depth(u, v, self.depth_frame)
        if depth is None:
            return None
        return (
            (u - self.intrinsics["ppx"]) * depth / self.intrinsics["fx"],
            (v - self.intrinsics["ppy"]) * depth / self.intrinsics["fy"],
            depth,
        )

    @staticmethod
    def _sample_depth(u, v, frame):
        height, width = frame.shape
        radius = DEPTH_PATCH // 2
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return None
        values = frame[y0:y1, x0:x1]
        values = values[values > 0]
        return float(np.median(values)) if values.size else None


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
