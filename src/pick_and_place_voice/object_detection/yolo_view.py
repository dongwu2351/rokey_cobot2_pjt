import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

PACKAGE_NAME = "pick_and_place_voice"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", "yolov8n_tools_0122.pt")

MIN_INTERVAL_SEC = 0.2  # cap inference rate (~5 Hz) so it doesn't saturate CPU


class YoloViewNode(Node):
    def __init__(self):
        super().__init__("yolo_view_node")
        self.bridge = CvBridge()
        self.model = YOLO(YOLO_MODEL_PATH)
        self.last_time = 0.0
        self.publisher = self.create_publisher(Image, "/yolo/detection_image", 10)
        self.subscription = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, 10
        )
        self.get_logger().info(
            "YoloViewNode initialized (ByteTrack). View with: "
            "ros2 run rqt_image_view rqt_image_view /yolo/detection_image"
        )

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_time < MIN_INTERVAL_SEC:
            return
        self.last_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )
        annotated = results[0].plot()

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloViewNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
