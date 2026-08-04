import threading
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from od_msg.srv import SrvDepthPosition
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

import DR_init
from .onrobot import RG


ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
TARGET_SETS = {
    "tools": ("drill", "hammer", "pliers", "screwdriver", "wrench"),
    "fruits": ("apple", "banana", "kiwi", "orange", "pear"),
}

JOINT_VELOCITY, JOINT_ACC = 100, 100
LINEAR_MAX_VELOCITY, LINEAR_MAX_ACC = 100, 100
LINEAR_LIFT_VELOCITY, LINEAR_LIFT_ACC = 80, 80

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

APPROACH_HEIGHT = 100.0
PLACE_CLEARANCE = 3.0
PLACE_BLIND_DROP = 50.0
PLUNGE_MM = 35.0
DEPTH_PATCH = 5
RING_INNER_MM = 40.0
RING_OUTER_MM = 70.0
RING_MIN_SAMPLES = 8
PLACE_VIEW_JOINT = [0, 0, 90, 0, 90, 0]

BASE_FRAME = "base_link"
FLANGE_FRAME = "link_6"
COLOR_PICK = (80, 220, 80)
COLOR_PLACE = (80, 160, 255)
COLOR_TEXT = (255, 255, 255)


class State(Enum):
    WAIT_VOICE = "WAIT_VOICE"
    DETECTING = "DETECTING"
    PICKING = "PICKING"
    HOLDING = "HOLDING"
    MOVING_HOME = "MOVING_HOME"
    WAIT_PLACE_CLICK = "WAIT_PLACE_CLICK"
    PLACING = "PLACING"
    ERROR = "ERROR"


def resolve_calibration_path():
    package_path = Path(get_package_share_directory("robot_control"))
    source_file = Path(__file__).resolve()
    workspace = source_file.parents[2]
    candidates = [
        package_path / "resource" / "T_gripper2camera.npy",
        source_file.parents[1] / "resource" / "T_gripper2camera.npy",
        workspace / "corecode" / "Calibration_Tutorial" / "T_gripper2camera.npy",
        workspace / "pick_and_place_text" / "resource" / "T_gripper2camera.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"T_gripper2camera.npy not found. Checked:\n  - {checked}"
    )


class PickPlaceController(Node):
    """Coordinate voice, object detection, robot motion and the PLACE GUI."""

    def __init__(self):
        super().__init__("voice_pick_place_controller")
        self.declare_parameter("target_set", "tools")
        self.target_set = (
            self.get_parameter("target_set")
            .get_parameter_value()
            .string_value
            .lower()
        )
        if self.target_set not in TARGET_SETS:
            choices = ", ".join(TARGET_SETS)
            raise ValueError(
                f"Unknown target_set '{self.target_set}'. Choose: {choices}"
            )
        self.target_names = TARGET_SETS[self.target_set]
        self.bridge = CvBridge()
        self.data_lock = threading.RLock()
        self.job_lock = threading.RLock()
        self.robot_lock = threading.RLock()
        self.busy = False
        self.running = True
        self.dry_run = True

        self.state = State.WAIT_VOICE
        self.status = "Waiting for split services..."
        self.target_name = None
        self.pick = None
        self.place = None
        self.voice_future = None
        self.annotated_frame = None
        self.depth_frame = None
        self.intrinsics = None

        self.create_subscription(
            Image,
            "/object_detection/annotated_image",
            self._image_callback,
            10,
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self._depth_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._camera_info_callback,
            10,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.voice_client = self.create_client(Trigger, "/get_keyword")
        self.position_client = self.create_client(
            SrvDepthPosition, "/get_3d_position"
        )

        # Node.executor is an rclpy-managed weak-reference property. Keep the
        # controller's executor under a different name so it stays alive.
        self.ros_executor = SingleThreadedExecutor()
        self.ros_executor.add_node(self)
        self.executor_thread = threading.Thread(
            target=self.ros_executor.spin, daemon=True
        )
        self.executor_thread.start()

        self._wait_for_data(self.get_frame, "annotated YOLO image")
        self._wait_for_data(self.get_depth, "aligned depth image")
        self.intrinsics = self._wait_for_data(
            self.get_intrinsics, "camera intrinsics"
        )

        calibration_path = resolve_calibration_path()
        self.gripper2cam = np.load(calibration_path)
        self.get_logger().info(f"Calibration: {calibration_path}")
        self.flange2tcp = self._calibrate_flange_offset()

        try:
            self.gripper = RG(
                GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT
            )
            self.get_logger().info(f"Gripper connected: {GRIPPER_NAME}")
        except Exception as error:
            self.gripper = None
            self.get_logger().error(
                f"Gripper connection failed ({error}); grip commands are disabled."
            )

        self.status = "Press V for voice or 1-5 to select a target"

    # Camera callbacks -------------------------------------------------
    def _image_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        with self.data_lock:
            self.annotated_frame = frame

    def _depth_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        with self.data_lock:
            self.depth_frame = frame

    def _camera_info_callback(self, message):
        with self.data_lock:
            self.intrinsics = {
                "fx": message.k[0],
                "fy": message.k[4],
                "ppx": message.k[2],
                "ppy": message.k[5],
            }

    def get_frame(self):
        with self.data_lock:
            return self.annotated_frame

    def get_depth(self):
        with self.data_lock:
            return self.depth_frame

    def get_intrinsics(self):
        with self.data_lock:
            return self.intrinsics

    def _wait_for_data(self, getter, description, timeout=30.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            value = getter()
            if value is not None and not (
                isinstance(value, np.ndarray) and not value.any()
            ):
                return value
            self.get_logger().info(f"Waiting for {description}...")
            time.sleep(0.2)
        raise RuntimeError(f"Timed out waiting for {description}")

    # Voice and object-detection services ------------------------------
    def request_voice_if_ready(self):
        if self.voice_future is not None:
            self.status = "Voice input is already listening..."
            return
        if self.state != State.WAIT_VOICE or self.busy:
            self.status = f"Voice unavailable in {self.state.value}"
            return
        if not self.voice_client.service_is_ready():
            self.status = "Waiting for /get_keyword service..."
            return
        if not self.position_client.service_is_ready():
            self.status = "Waiting for /get_3d_position service..."
            return

        self.status = "Say 'Hello Rokey', then name a target"
        self.get_logger().info(
            "Voice input requested: say 'Hello Rokey', then name a target"
        )
        self.voice_future = self.voice_client.call_async(Trigger.Request())
        self.voice_future.add_done_callback(self._voice_done)

    def _voice_done(self, future):
        self.voice_future = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f"Voice service failed: {error}")
            self.status = "Voice failed; press V to retry"
            return

        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            self.get_logger().warn(f"Voice command rejected: {message}")
            self.status = "No valid voice command; press V to retry"
            return

        if self.state != State.WAIT_VOICE or self.busy:
            self.get_logger().warn(
                f"Voice result ignored while controller is in {self.state.value}"
            )
            self.status = f"Voice result ignored in {self.state.value}"
            return

        words = response.message.lower().split()
        target = next(
            (word for word in words if word in self.target_names), None
        )
        if target is None:
            self.get_logger().warn(f"No known target in: {response.message}")
            self.status = "Unknown target; press V and try again"
            return

        self.start_pick_for_target(target, source="voice")

    def handle_target_key(self, number):
        if not 1 <= number <= len(self.target_names):
            return
        target = self.target_names[number - 1]
        if self.voice_future is not None:
            self.status = "Voice input active; finish it before number selection"
            return
        if self.state != State.WAIT_VOICE or self.busy:
            self.status = f"{target} unavailable in {self.state.value}"
            return
        if not self.position_client.service_is_ready():
            self.status = "Waiting for /get_3d_position service..."
            return
        self.start_pick_for_target(target, source=f"key {number}")

    def start_pick_for_target(self, target, source):
        self.target_name = target
        self.state = State.DETECTING
        self.status = f"Detecting {target}..."
        self.get_logger().info(f"Target selected by {source}: {target}")
        if not self.start_job(self._job_detect_and_pick):
            self.target_name = None
            self.state = State.WAIT_VOICE
            self.status = "Target selection ignored while robot was busy"

    def request_camera_position(self, target, timeout=10.0):
        request = SrvDepthPosition.Request()
        request.target = target
        future = self.position_client.call_async(request)
        deadline = time.monotonic() + timeout
        while self.running and rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                response = future.result()
                if response is None:
                    return None
                values = np.asarray(response.depth_position, dtype=float)
                if values.size != 3 or not np.isfinite(values).all():
                    return None
                if np.allclose(values, 0.0):
                    return None
                return values
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for /get_3d_position")

    # Camera/base transforms ------------------------------------------
    @staticmethod
    def get_robot_pose_matrix(x, y, z, rx, ry, rz):
        rotation = Rotation.from_euler(
            "ZYZ", [rx, ry, rz], degrees=True
        ).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [x, y, z]
        return transform

    def current_base2cam(self):
        if not self.robot_lock.acquire(blocking=False):
            return None
        try:
            base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        finally:
            self.robot_lock.release()
        return base2gripper @ self.gripper2cam

    def tf_base2flange(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, FLANGE_FRAME, rclpy.time.Time()
            ).transform
        except Exception:
            return None
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ]
        ).as_matrix()
        matrix[:3, 3] = [
            transform.translation.x * 1000.0,
            transform.translation.y * 1000.0,
            transform.translation.z * 1000.0,
        ]
        return matrix

    def _calibrate_flange_offset(self):
        for _ in range(25):
            base2flange = self.tf_base2flange()
            if base2flange is not None:
                break
            time.sleep(0.2)
        else:
            self.get_logger().warn(
                f"No TF {BASE_FRAME}->{FLANGE_FRAME}; anchors may be stale while moving."
            )
            return None
        try:
            base2tcp = self.get_robot_pose_matrix(*get_current_posx()[0])
        except Exception as error:
            self.get_logger().warn(f"Could not calculate flange/TCP offset: {error}")
            return None
        return np.linalg.inv(base2flange) @ base2tcp

    def tf_base2cam(self):
        if self.flange2tcp is None:
            return None
        base2flange = self.tf_base2flange()
        if base2flange is None:
            return None
        return base2flange @ self.flange2tcp @ self.gripper2cam

    def camera_to_base(self, camera_position, base2cam):
        return (
            base2cam
            @ np.append(np.asarray(camera_position, dtype=float), 1.0)
        )[:3]

    def camera_to_pixel(self, camera_position):
        x, y, z = camera_position
        if z <= 0:
            return None
        u = self.intrinsics["fx"] * x / z + self.intrinsics["ppx"]
        v = self.intrinsics["fy"] * y / z + self.intrinsics["ppy"]
        return int(round(u)), int(round(v))

    def pixel_to_base(self, u, v, depth, base2cam):
        camera = np.array(
            [
                (u - self.intrinsics["ppx"]) * depth / self.intrinsics["fx"],
                (v - self.intrinsics["ppy"]) * depth / self.intrinsics["fy"],
                depth,
                1.0,
            ]
        )
        return (base2cam @ camera)[:3]

    def project_to_pixel(self, point_base, base2cam):
        point_camera = (
            np.linalg.inv(base2cam)
            @ np.append(np.asarray(point_base, dtype=float), 1.0)
        )
        if not np.isfinite(point_camera).all() or point_camera[2] <= 1.0:
            return None
        u = (
            self.intrinsics["fx"] * point_camera[0] / point_camera[2]
            + self.intrinsics["ppx"]
        )
        v = (
            self.intrinsics["fy"] * point_camera[1] / point_camera[2]
            + self.intrinsics["ppy"]
        )
        if not np.isfinite([u, v]).all():
            return None
        return int(round(u)), int(round(v))

    @staticmethod
    def sample_depth(u, v, depth_frame, patch=DEPTH_PATCH):
        height, width = depth_frame.shape
        radius = patch // 2
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return None
        values = depth_frame[y0:y1, x0:x1]
        values = values[values > 0]
        return float(np.median(values)) if values.size else None

    def estimate_floor_height(self, u, v, depth_frame, base2cam, center_depth):
        radius_inner = RING_INNER_MM * self.intrinsics["fx"] / center_depth
        radius_outer = RING_OUTER_MM * self.intrinsics["fx"] / center_depth
        heights = []
        for angle in range(0, 360, 15):
            cosine = np.cos(np.radians(angle))
            sine = np.sin(np.radians(angle))
            for radius in (
                radius_inner,
                (radius_inner + radius_outer) / 2.0,
                radius_outer,
            ):
                sample_u = int(u + radius * cosine)
                sample_v = int(v + radius * sine)
                depth = self.sample_depth(
                    sample_u, sample_v, depth_frame, patch=3
                )
                if depth is None:
                    continue
                heights.append(
                    self.pixel_to_base(
                        sample_u, sample_v, depth, base2cam
                    )[2]
                )
        if len(heights) < RING_MIN_SAMPLES:
            return None
        return float(np.median(heights))

    # State jobs -------------------------------------------------------
    def start_job(self, function):
        with self.job_lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(
            target=self._run_job, args=(function,), daemon=True
        ).start()
        return True

    def _run_job(self, function):
        try:
            with self.robot_lock:
                function()
        except Exception as error:
            self.get_logger().error(f"Robot job stopped: {error}")
            self.state = State.ERROR
            self.status = f"ERROR: {error}"
        finally:
            with self.job_lock:
                self.busy = False

    def _job_detect_and_pick(self):
        camera_position = self.request_camera_position(self.target_name)
        if camera_position is None:
            self.get_logger().warn(f"{self.target_name} not detected")
            self.state = State.WAIT_VOICE
            self.status = f"{self.target_name} not found; press V or 1-5"
            return

        base2cam = self.current_base2cam()
        if base2cam is None:
            raise RuntimeError("Robot pose unavailable for PICK")
        position = self.camera_to_base(camera_position, base2cam)
        if not np.isfinite(position).all() or position[2] < 0:
            raise RuntimeError(f"Invalid PICK coordinate: {position}")

        center = self.camera_to_pixel(camera_position)
        depth_frame = self.get_depth()
        floor_z = None
        if center is not None and depth_frame is not None:
            floor_z = self.estimate_floor_height(
                center[0],
                center[1],
                depth_frame,
                base2cam,
                camera_position[2],
            )
        object_height = (
            None if floor_z is None else float(position[2] - floor_z)
        )
        if object_height is not None and object_height <= 0:
            object_height = None

        orientation = list(get_current_posx()[0][3:])
        self.pick = {
            "uv": center,
            "pos": position,
            "height": object_height,
            "plunge": PLUNGE_MM,
            "orientation": orientation,
            "name": self.target_name,
        }
        self.state = State.PICKING
        self.status = f"Picking {self.target_name}..."
        self.pick_target()
        self.state = State.HOLDING
        self.status = f"Holding {self.target_name} - press H"

    def pick_target(self):
        position = self.pick["pos"]
        rx, ry, rz = self.pick["orientation"]
        plunge = self.pick["plunge"]
        self.grip(close=False)
        self.go_linear(
            "pick approach",
            position[0],
            position[1],
            position[2] + APPROACH_HEIGHT,
            rx,
            ry,
            rz,
        )
        self.go_linear(
            "pick descend",
            position[0],
            position[1],
            position[2] - plunge,
            rx,
            ry,
            rz,
        )
        self.grip(close=True)
        self.go_linear(
            "pick lift",
            position[0],
            position[1],
            position[2] + APPROACH_HEIGHT,
            rx,
            ry,
            rz,
            velocity=LINEAR_LIFT_VELOCITY,
            acceleration=LINEAR_LIFT_ACC,
        )

    def handle_home_key(self):
        if self.voice_future is not None:
            self.status = "Voice input active; finish it before HOME"
            return
        if self.busy:
            self.status = "Robot is busy; HOME command ignored"
            return
        if self.state == State.HOLDING:
            self.state = State.MOVING_HOME
            self.status = "Moving to PLACE camera view..."
            self.start_job(self._job_go_place_view)
        elif self.state == State.WAIT_VOICE:
            self.state = State.MOVING_HOME
            self.status = "Moving HOME before PICK..."
            self.start_job(self._job_go_home_before_pick)
        else:
            self.status = f"HOME unavailable in {self.state.value}"

    def _job_go_home_before_pick(self):
        self.go_joint("home before pick", PLACE_VIEW_JOINT)
        self.state = State.WAIT_VOICE
        self.status = "HOME ready - press V or 1-5"

    def _job_go_place_view(self):
        self.go_joint("place camera view", PLACE_VIEW_JOINT)
        self.state = State.WAIT_PLACE_CLICK
        self.status = "Click the PLACE point"

    def handle_gripper_key(self, close):
        if self.busy:
            self.status = "Robot is busy; gripper command ignored"
            return
        if self.state in (
            State.DETECTING,
            State.PICKING,
            State.MOVING_HOME,
            State.PLACING,
        ):
            self.status = f"Gripper unavailable in {self.state.value}"
            return
        previous_state = self.state
        action = "Closing" if close else "Opening"
        self.status = f"{action} gripper..."
        self.start_job(lambda: self._job_manual_grip(close, previous_state))

    def _job_manual_grip(self, close, previous_state):
        self.grip(close=close)
        if not close and previous_state in (
            State.HOLDING,
            State.WAIT_PLACE_CLICK,
        ):
            self.pick = None
            self.place = None
            self.target_name = None
            self.state = State.WAIT_VOICE
            self.status = "Gripper opened - press V or 1-5"
            return
        self.state = previous_state
        self.status = "Gripper closed" if close else "Gripper opened"

    def mouse_callback(self, event, x, y, flags, parameter):
        if (
            event != cv2.EVENT_LBUTTONDOWN
            or self.busy
            or self.state != State.WAIT_PLACE_CLICK
        ):
            return
        depth_frame = self.get_depth()
        if depth_frame is None:
            self.status = "No depth frame; click again"
            return
        depth = self.sample_depth(x, y, depth_frame)
        if depth is None:
            self.status = "No depth at that pixel; click again"
            return
        base2cam = self.current_base2cam()
        if base2cam is None:
            self.status = "Robot pose unavailable; click again"
            return
        position = self.pixel_to_base(x, y, depth, base2cam)
        if not np.isfinite(position).all() or position[2] < 0:
            self.status = "Invalid PLACE coordinate; click again"
            return
        self.place = {
            "uv": (x, y),
            "pos": position,
            "surface_z": float(position[2]),
        }
        self.state = State.PLACING
        self.status = "Placing object..."
        self.start_job(self._job_place)

    def _job_place(self):
        position = self.place["pos"]
        object_height = self.pick["height"]
        plunge = self.pick["plunge"]
        if object_height is None:
            place_z = self.place["surface_z"] + PLACE_BLIND_DROP
            self.get_logger().warn(
                f"Object height unknown; releasing {PLACE_BLIND_DROP} mm above surface"
            )
        else:
            place_z = (
                self.place["surface_z"]
                + object_height
                - plunge
                + PLACE_CLEARANCE
            )
        rx, ry, rz = get_current_posx()[0][3:]
        self.go_linear(
            "place approach",
            position[0],
            position[1],
            place_z + APPROACH_HEIGHT,
            rx,
            ry,
            rz,
        )
        self.go_linear(
            "place lower",
            position[0],
            position[1],
            place_z,
            rx,
            ry,
            rz,
        )
        self.grip(close=False)
        self.go_linear(
            "place retreat",
            position[0],
            position[1],
            place_z + APPROACH_HEIGHT,
            rx,
            ry,
            rz,
        )
        self.go_joint("home", PLACE_VIEW_JOINT)
        self.pick = None
        self.place = None
        self.target_name = None
        self.state = State.WAIT_VOICE
        self.status = "Done - press V or 1-5 for the next target"

    # Robot primitives -------------------------------------------------
    def go_linear(
        self,
        label,
        x,
        y,
        z,
        rx,
        ry,
        rz,
        velocity=LINEAR_MAX_VELOCITY,
        acceleration=LINEAR_MAX_ACC,
    ):
        target = [float(x), float(y), float(z), rx, ry, rz]
        self.get_logger().info(
            f"{label}: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})"
        )
        if self.dry_run:
            return
        result = movel(posx(target), vel=velocity, acc=acceleration)
        if result != 0:
            raise RuntimeError(f"{label} movel failed ({result})")

    def go_joint(self, label, joints):
        self.get_logger().info(f"{label}: {joints}")
        if self.dry_run:
            return
        result = movej(posj(joints), vel=JOINT_VELOCITY, acc=JOINT_ACC)
        if result != 0:
            raise RuntimeError(f"{label} movej failed ({result})")

    def grip(self, close):
        if self.gripper is None:
            self.get_logger().warn("Gripper unavailable; skipping grip command")
            return
        if self.dry_run:
            return
        if close:
            self.gripper.close_gripper()
        else:
            self.gripper.open_gripper()
        time.sleep(1.0)

    # GUI --------------------------------------------------------------
    @staticmethod
    def draw_marker(frame, point, color, label):
        cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 26, 2)
        cv2.circle(frame, point, 14, color, 2)
        cv2.putText(
            frame,
            label,
            (point[0] + 18, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def _anchored_pixel(self, entry, base2cam, frame_shape):
        if entry is None:
            return None
        if base2cam is None:
            return entry.get("uv")
        point = self.project_to_pixel(entry["pos"], base2cam)
        if point is None:
            return None
        height, width = frame_shape[:2]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return None
        return point

    def render(self):
        frame = self.get_frame()
        if frame is None:
            return
        visual = frame.copy()
        base2cam = self.tf_base2cam()
        pick_pixel = self._anchored_pixel(self.pick, base2cam, visual.shape)
        place_pixel = self._anchored_pixel(self.place, base2cam, visual.shape)
        if pick_pixel is not None:
            self.draw_marker(visual, pick_pixel, COLOR_PICK, "PICK")
        if place_pixel is not None:
            self.draw_marker(visual, place_pixel, COLOR_PLACE, "PLACE")

        cv2.rectangle(visual, (0, 0), (visual.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(
            visual,
            f"{self.state.value}: {self.status}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            visual,
            (0, visual.shape[0] - 53),
            (visual.shape[1], visual.shape[0]),
            (0, 0, 0),
            -1,
        )
        number_guide = "  ".join(
            f"{index}:{name}"
            for index, name in enumerate(self.target_names, start=1)
        )
        cv2.putText(
            visual,
            number_guide,
            (10, visual.shape[0] - 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            visual,
            "[V] voice  [H] home  [O] open  [C] close  [D] dry-run  [ESC] quit",
            (10, visual.shape[0] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        if self.dry_run:
            cv2.putText(
                visual,
                "DRY RUN",
                (visual.shape[1] - 112, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("Voice Pick and Place", visual)

    def shutdown(self):
        self.running = False
        self.ros_executor.shutdown()
        if self.executor_thread.is_alive():
            self.executor_thread.join(timeout=2.0)
        self.ros_executor.remove_node(self)
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    global get_current_posx, movej, movel, posj, posx
    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel
        from DR_common2 import posj, posx
    except ImportError as error:
        print(f"Error importing Doosan robot API: {error}")
        dsr_node.destroy_node()
        rclpy.shutdown()
        return

    controller = None
    try:
        cv2.namedWindow("Voice Pick and Place")
        controller = PickPlaceController()
        cv2.setMouseCallback(
            "Voice Pick and Place", controller.mouse_callback
        )
        print("=" * 72)
        print(" SPLIT: Voice/number -> YOLO PICK -> H -> anchored PLACE")
        print(" V: voice input")
        print(
            " "
            + " | ".join(
                f"{index}: {name}"
                for index, name in enumerate(
                    controller.target_names, start=1
                )
            )
        )
        print(" H: HOME | O: gripper open | C: gripper close")
        print(" D: dry-run | ESC: quit")
        print("=" * 72)

        while rclpy.ok():
            controller.render()
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                if controller.busy:
                    print("Robot is moving. Press ESC again after it finishes.")
                    continue
                break
            if key in (ord("v"), ord("V")):
                controller.request_voice_if_ready()
            elif ord("1") <= key <= ord("5"):
                controller.handle_target_key(key - ord("0"))
            elif key in (ord("h"), ord("H")):
                controller.handle_home_key()
            elif key in (ord("o"), ord("O")):
                controller.handle_gripper_key(close=False)
            elif key in (ord("c"), ord("C")):
                controller.handle_gripper_key(close=True)
            elif key in (ord("d"), ord("D")) and not controller.busy:
                controller.dry_run = not controller.dry_run
                print(f"DRY RUN: {'ON' if controller.dry_run else 'OFF'}")
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Controller failed: {error}")
    finally:
        cv2.destroyAllWindows()
        if controller is not None:
            controller.shutdown()
        dsr_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
