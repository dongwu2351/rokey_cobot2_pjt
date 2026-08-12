"""RGB-D from the realsense2_camera ROS driver instead of the device itself.

The D435i can only be opened once. On this cell the ROS driver already owns
it - the robot app refines its grasp and takes its inspection photographs
from those topics - so `RealSenseSource`, which opens the device directly
through pyrealsense2, cannot run at the same time. Killing the driver to
give the copilot a picture would take the robot's eyes away.

Subscribing to the topics the driver already publishes costs the camera
nothing and leaves every existing consumer untouched.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .realsense_source import CameraFrame

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
#: On this cell the driver's align module publishes nothing, so the aligned
#: topic stays silent. The raw depth stream is still the same scene from the
#: same camera, and the only consumer is a whole-frame quality statistic
#: (valid ratio, nearest decile) that never pairs a depth pixel with a colour
#: pixel - so it is used, and flagged as unaligned, rather than dropped.
RAW_DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"
#: realsense2_camera publishes z16 in millimetres; the direct SDK source
#: reports metres per unit, and the vision code multiplies by this.
DEPTH_SCALE = 0.001


class RosCameraSource:
    """Latest color (and aligned depth) frame from the running ROS driver."""

    def __init__(self, color_topic: str = COLOR_TOPIC,
                 depth_topic: str = DEPTH_TOPIC,
                 raw_depth_topic: str = RAW_DEPTH_TOPIC,
                 info_topic: str = INFO_TOPIC,
                 timeout: float = 8.0) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2(rclpy/sensor_msgs)를 불러오지 못했습니다. "
                "source /opt/ros/humble/setup.bash 후 실행하세요."
            ) from exc

        self._lock = threading.Lock()
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._depth_aligned = False
        self._stamp_ms: int = 0
        self._info: dict = {"depth_scale": DEPTH_SCALE}

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = rclpy.create_node("copilot_camera_source")
        # Sensor QoS (best effort): the driver publishes with it, and a
        # RELIABLE subscription would silently never match.
        self._node.create_subscription(
            Image, color_topic, self._color_cb, qos_profile_sensor_data)
        self._node.create_subscription(
            Image, depth_topic,
            lambda msg: self._depth_cb(msg, aligned=True), qos_profile_sensor_data)
        self._node.create_subscription(
            Image, raw_depth_topic,
            lambda msg: self._depth_cb(msg, aligned=False), qos_profile_sensor_data)
        self._node.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="copilot-camera")
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._color is not None:
                    return
            time.sleep(0.05)
        self.close()
        raise RuntimeError(
            f"{color_topic} 에서 영상이 오지 않습니다. realsense2_camera가 "
            "실행 중인지 확인하세요 (ros2 topic hz).")

    # ------------------------------------------------------------------
    def _color_cb(self, msg) -> None:
        image = self._to_array(msg)
        if image is None:
            return
        if msg.encoding == "rgb8":
            image = image[:, :, ::-1]          # the rest of the stack is BGR
        with self._lock:
            self._color = np.ascontiguousarray(image)
            self._stamp_ms = round(time.time() * 1000)

    def _depth_cb(self, msg, aligned: bool) -> None:
        image = self._to_array(msg)
        if image is None:
            return
        with self._lock:
            # Once an aligned frame has ever arrived, the raw stream is
            # ignored - never let the worse source overwrite the better one.
            if self._depth_aligned and not aligned:
                return
            self._depth = image
            self._depth_aligned = self._depth_aligned or aligned

    def _info_cb(self, msg) -> None:
        with self._lock:
            self._info = {
                "depth_scale": DEPTH_SCALE,
                "fx": msg.k[0], "fy": msg.k[4],
                "cx": msg.k[2], "cy": msg.k[5],
                "width": msg.width, "height": msg.height,
            }

    @staticmethod
    def _to_array(msg):
        """sensor_msgs/Image without cv_bridge (which pulls in a whole stack).

        Only the encodings the RealSense driver actually publishes."""
        channels = {"bgr8": 3, "rgb8": 3, "mono8": 1, "8UC1": 1,
                    "16UC1": 1, "mono16": 1}.get(msg.encoding)
        if channels is None:
            return None
        dtype = np.uint16 if msg.encoding in ("16UC1", "mono16") else np.uint8
        buffer = np.frombuffer(msg.data, dtype=dtype)
        expected = msg.height * msg.width * channels
        if buffer.size < expected:
            return None
        # step is in bytes and may be padded, so reshape by row then trim.
        row = msg.step // dtype().itemsize
        image = buffer[:msg.height * row].reshape(msg.height, row)
        image = image[:, :msg.width * channels]
        return image.reshape(msg.height, msg.width, channels).squeeze()

    # ------------------------------------------------------------------
    def read(self) -> CameraFrame | None:
        with self._lock:
            if self._color is None:
                return None
            info = dict(self._info)
            info["depth_aligned"] = self._depth_aligned
            return CameraFrame(self._color.copy(),
                               None if self._depth is None else self._depth.copy(),
                               self._stamp_ms, info)

    def close(self) -> None:
        # Join the spin thread: a daemon executor still spinning at
        # interpreter exit aborts the process with "terminate called without
        # an active exception", which reads as a crash in the log.
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
