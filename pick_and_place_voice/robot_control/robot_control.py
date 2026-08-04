import os
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger

import DR_init
from dsr_msgs2.msg import SpeedlStream
from object_detection.realsense import ImgNode
from object_detection.yolo import YoloModel
from robot_control.onrobot import RG


ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
JOINT_VELOCITY, JOINT_ACC = 100, 100
LINEAR_MAX_VELOCITY, LINEAR_MAX_ACC = 100, 100
LINEAR_LIFT_VELOCITY, LINEAR_LIFT_ACC = 80, 80

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# PICK/PLACE geometry in millimeters.
APPROACH_HEIGHT = 100.0
PLACE_CLEARANCE = 3.0
PLACE_BLIND_DROP = 50.0
PLUNGE_MM = 35.0
DEPTH_PATCH = 5
RING_INNER_MM = 40.0
RING_OUTER_MM = 70.0
RING_MIN_SAMPLES = 8

# This must be taught as a collision-free camera view of the PLACE area.
PLACE_VIEW_JOINT = [0, 0, 90, 0, 90, 0]

BASE_FRAME = "base_link"
FLANGE_FRAME = "link_6"

COLOR_TARGET = (80, 220, 80)
COLOR_OTHER = (170, 170, 170)
COLOR_PICK = (80, 220, 80)
COLOR_PLACE = (80, 160, 255)
COLOR_TEXT = (255, 255, 255)


class State(Enum):
    WAIT_VOICE = "WAIT_VOICE"
    FOLLOWING = "FOLLOWING"
    GRASPING = "GRASPING"
    PICKING = "PICKING"
    HOLDING = "HOLDING"
    MOVING_HOME = "MOVING_HOME"
    WAIT_PLACE_CLICK = "WAIT_PLACE_CLICK"
    PLACING = "PLACING"
    ERROR = "ERROR"


# Following (visual servo hover) tuning via SpeedL (Cartesian velocity
# streaming), not ServoL: each tick we command a velocity proportional to
# the position error, capped at FOLLOW_MAX_LINEAR_SPEED, instead of
# re-issuing absolute poses. This decelerates naturally near the target and
# doesn't restart an accel profile every tick on sensor noise.
LOST_TRACK_TIMEOUT = 1.0
FOLLOW_MIN_INTERVAL_SEC = 0.05
FOLLOW_KP = 1.0  # (mm/s) commanded per (mm) of position error
FOLLOW_MAX_LINEAR_SPEED = 120.0  # mm/s cap - raise further only after watching for overshoot
FOLLOW_DEADBAND_MM = 3.0  # error below this -> command zero velocity
# Alpha-beta (g-h) filter on the raw vision target: estimates position AND
# velocity from noisy per-frame detections, instead of plain EMA (which only
# smooths position and always lags behind a moving target). ALPHA corrects
# position, BETA corrects the velocity estimate from the same residual.
FOLLOW_AB_ALPHA = 0.6
# BETA is divided by dt, so it amplifies measurement noise hard: at dt=0.05 a
# beta of 0.4 turns 2mm of box jitter into 16mm/s of phantom target velocity.
# With an eye-in-hand camera that phantom velocity moves the arm, which moves
# the camera, which feeds back - keep this low.
FOLLOW_AB_BETA = 0.15
FOLLOW_AB_DT_MIN, FOLLOW_AB_DT_MAX = 0.02, 0.2  # bound the 1/dt amplification
# Project the filtered estimate this far ahead using its velocity, so the
# hover target leads a moving object instead of chasing where it just was.
# Also what lets tracking "coast" for a moment on last known velocity when
# the object briefly leaves the frame (e.g. fast motion toward -Y).
FOLLOW_LEAD_TIME = 0.12
FOLLOW_MAX_TARGET_SPEED = 150.0  # mm/s sanity cap on the *estimated* target velocity
# The beta term is a differentiator, so its output is inherently noisy - a
# motionless object measured with 2mm of jitter still shows ~10mm/s average
# and 20mm/s peaks. Low-pass it before anything acts on it.
FOLLOW_VEL_SMOOTHING = 0.15
# Below this the target counts as stationary and its velocity is ignored
# entirely. Hysteresis (engage high, release lower) stops it flickering on and
# off around the threshold, which would itself look like twitching.
FOLLOW_MIN_TARGET_SPEED = 20.0
FOLLOW_RELEASE_TARGET_SPEED = 12.0
# Only partially compensate the target's motion. Full feedforward doubles down
# on a noisy estimate; half still removes most of the lag.
FOLLOW_FEEDFORWARD_GAIN = 0.5
# While coasting (no usable measurement) bleed the assumed target velocity
# away each tick, so a stale estimate fades out instead of flying the arm
# off at full speed for the whole LOST_TRACK_TIMEOUT window.
FOLLOW_COAST_DECAY = 0.95

# A detection box that touches the image border is CLIPPED: only part of the
# object is visible. Its centroid sits inboard of the object's true centre,
# so the measurement lags exactly when the object is leaving the frame - and
# that bogus "it slowed down" residual corrupts the velocity estimate right
# before we need it to coast. Rebuild the full box from the object's recent
# unclipped size, anchored on whichever edge is still inside the image.
FOLLOW_EDGE_MARGIN_PX = 6
FOLLOW_MIN_VISIBLE_FRACTION = 0.25  # below this, coast instead of trusting a stub
FOLLOW_BOX_SIZE_ALPHA = 0.3  # EMA on the unclipped box size used for rebuilding
# YOLO confidence floor for the tracking loop. Edge-clipped boxes are now
# reconstructed rather than relied on raw, so this no longer needs lowering
# (a lower floor mostly just admits noisier detections).
FOLLOW_CONF_THRESHOLD = 0.5

# Closed-loop INTERCEPTION after G. The arm aims at where the object will be
# when it has finished descending and crosses laterally to arrive at the same
# moment, so it sweeps in diagonally and snatches a moving object rather than
# hovering over it first and dropping straight down.
GRASP_MAX_XY_SPEED = 120.0  # mm/s lateral cap during the intercept
GRASP_DESCENT_SPEED = 60.0  # mm/s nominal descent; scaled down if XY can't keep up
GRASP_MIN_TIME_TO_CONTACT = 0.15  # s floor on the rendezvous horizon (bounds the gain)
GRASP_ALIGN_TOLERANCE = 12.0  # mm max miss distance still allowed to commit
# The blind final plunge takes time, and the object keeps moving through it;
# lead the committed grasp point by roughly how long that phase lasts.
GRASP_COMMIT_LEAD_TIME = 0.5
# Final stretch is committed blind: the gripper occludes the object and the
# depth camera is past its usable near range, so vision is worthless here.
GRASP_BLIND_MM = 25.0
GRASP_LOST_TIMEOUT = 0.6  # s without a usable fix before aborting the descent
GRASP_TIMEOUT = 20.0  # s hard cap on one descent attempt
GRASP_SETTLE_SEC = 0.3  # let SpeedL streaming actually stop before issuing movel
# Hard floor: however thin the object looks (or however bad a depth reading is),
# never drive the TCP more than this far below the surface detected around it.
# object_top - PLUNGE_MM alone goes straight through the table for anything
# thinner than PLUNGE_MM.
GRASP_MAX_BELOW_FLOOR_MM = 5.0
# Acceleration limit fed to SpeedL. The DRFL controller alarms
# ("Limit translation acceleration is very low") if this is too small;
# empirically anything near/above the low hundreds has been fine here.
FOLLOW_LINEAR_ACC, FOLLOW_ANGULAR_ACC = 400.0, 90.0

# Safe reachable box for the hover target, in base-frame millimeters.
# PLACEHOLDER, widened after seeing (427, 42, 383) get clamped away during
# testing - still not derived from the actual cell's real reachable
# workspace. Verify against the robot before trusting this at higher speed.
FOLLOW_X_MIN, FOLLOW_X_MAX = 150.0, 650.0
FOLLOW_Y_MIN, FOLLOW_Y_MAX = -300.0, 300.0
FOLLOW_Z_MIN, FOLLOW_Z_MAX = 100.0, 500.0


def resolve_calibration_path():
    package_path = Path(get_package_share_directory("pick_and_place_voice"))
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
    def __init__(self):
        super().__init__("voice_pick_place_controller")

        self.lock = threading.RLock()
        self.robot_lock = threading.RLock()
        self.detection_lock = threading.Lock()
        self.busy = False
        # Start safe: the operator must press D once to enable real motion.
        self.dry_run = True
        self.running = True
        # Lead/feedforward prediction. Toggle with P to compare live: with it
        # off the arm purely chases the measured position (calmer, laggier).
        self.predict_enabled = True

        self.state = State.WAIT_VOICE
        self.status = "Waiting for voice service..."
        self.target_name = None
        self.pick = None
        self.place = None
        self.voice_future = None
        self.target_requested_at = 0.0
        self.latest_detections = []
        self.detections_updated_at = 0.0
        self.inference_error = None
        self.last_tracked = None
        self.follow_orientation = None
        self.last_follow_send_at = 0.0
        self.target_estimate = None  # {"pos": np.array3, "vel": np.array3, "t": monotonic}
        self.last_measurement_at = 0.0  # last *real* vision fix, drives lost-track
        self.nominal_box = None  # [w, h] px of the target when fully visible
        self.velocity_filtered = None
        self.velocity_engaged = False
        self.grasp_started_at = 0.0
        self.grasp_floor_z = None
        self.abort_requested = False

        self.img_node = ImgNode()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.voice_client = self.create_client(Trigger, "/get_keyword")
        self.speedl_pub = self.create_publisher(
            SpeedlStream, f"/{ROBOT_ID}/speedl_stream", 10
        )

        # Camera, TF and service futures use a private executor. DSR_ROBOT2 keeps
        # its own global node/executor, avoiding "generator already executing".
        self.camera_executor = SingleThreadedExecutor()
        self.camera_executor.add_node(self.img_node)
        self.camera_executor.add_node(self)
        self.camera_thread = threading.Thread(
            target=self.camera_executor.spin, daemon=True
        )
        self.camera_thread.start()

        self.intrinsics = self._wait_for_camera_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        self._wait_for_camera_data(self.img_node.get_color_frame, "color frame")
        self._wait_for_camera_data(self.img_node.get_depth_frame, "depth frame")

        calibration_path = resolve_calibration_path()
        self.gripper2cam = np.load(calibration_path)
        self.get_logger().info(f"Calibration: {calibration_path}")

        self.flange2tcp = self._calibrate_flange_offset()
        self.yolo = YoloModel()
        self.get_logger().info(
            f"YOLO classes: {', '.join(self.yolo.class_names)}"
        )

        try:
            self.gripper = RG(
                GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT
            )
            self.get_logger().info(f"Gripper connected: {GRIPPER_NAME}")
        except Exception as error:
            self.gripper = None
            self.get_logger().error(
                f"Gripper connection failed ({error}); movement will run without grip."
            )

        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self.inference_thread.start()
        self.status = "Say 'Hello Rokey' and name a tool"

    def _wait_for_camera_data(self, getter, description, timeout=30.0):
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

    # ------------------------------------------------------------------
    # Voice command
    # ------------------------------------------------------------------
    def request_voice_if_ready(self):
        if self.state != State.WAIT_VOICE or self.voice_future is not None:
            return
        if not self.voice_client.service_is_ready():
            self.status = "Waiting for /get_keyword service..."
            return

        self.status = "Say 'Hello Rokey', then name a tool"
        self.voice_future = self.voice_client.call_async(Trigger.Request())
        self.voice_future.add_done_callback(self._voice_done)

    def _voice_done(self, future):
        self.voice_future = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f"Voice service failed: {error}")
            self.status = "Voice failed; retrying..."
            return

        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            self.get_logger().warn(f"Voice command rejected: {message}")
            self.status = "No valid voice command; retrying..."
            return

        words = response.message.lower().split()
        target = next(
            (word for word in words if word in self.yolo.class_names), None
        )
        if target is None:
            self.get_logger().warn(f"No known tool in: {response.message}")
            self.status = "Unknown tool; say the command again"
            return

        self._enter_following(target)

    def _enter_following(self, target):
        self.target_name = target
        self.target_requested_at = time.monotonic()
        self.last_tracked = None
        self.target_estimate = None
        self.velocity_filtered = None
        self.velocity_engaged = False
        self.last_measurement_at = time.monotonic()
        self.nominal_box = None
        self.follow_orientation = list(get_current_posx()[0][3:])
        self.state = State.FOLLOWING
        self.status = f"Following {target} - press G to grab"

    def handle_follow_key(self):
        """Manual trigger (no voice needed): lock onto whatever is currently
        detected with the highest score and start following it."""
        if self.busy:
            self.status = "Robot is busy; F ignored"
            return
        if self.state != State.WAIT_VOICE:
            self.status = f"F unavailable in {self.state.value}"
            return

        with self.detection_lock:
            detections = list(self.latest_detections)
        if not detections:
            self.status = "No object visible; show it to the camera and press F"
            return

        best = max(detections, key=lambda item: item["score"])
        self._enter_following(best["name"])

    # ------------------------------------------------------------------
    # YOLO inference and display data
    # ------------------------------------------------------------------
    def _inference_loop(self):
        while self.running and rclpy.ok():
            frame = self.img_node.get_color_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                detections = self.yolo.predict_frame(
                    frame, confidence_threshold=FOLLOW_CONF_THRESHOLD
                )
                with self.detection_lock:
                    self.latest_detections = detections
                    self.detections_updated_at = time.monotonic()
                    self.inference_error = None
            except Exception as error:
                with self.detection_lock:
                    self.inference_error = str(error)
                self.get_logger().error(f"YOLO inference failed: {error}")
                time.sleep(1.0)
            time.sleep(0.08)

    def _best_matching_detection(self, target):
        """Latest detection for `target`, or None. Non-blocking single check."""
        with self.detection_lock:
            detections = list(self.latest_detections)
            updated_at = self.detections_updated_at
        if updated_at < self.target_requested_at:
            return None
        matches = [d for d in detections if d["name"] == target]
        if not matches:
            return None
        return max(matches, key=lambda item: item["score"])

    # ------------------------------------------------------------------
    # Camera/base transforms and depth
    # ------------------------------------------------------------------
    @staticmethod
    def get_robot_pose_matrix(x, y, z, rx, ry, rz):
        rotation = Rotation.from_euler(
            "ZYZ", [rx, ry, rz], degrees=True
        ).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [x, y, z]
        return transform

    def _safe_current_posx(self):
        """[x, y, z, rx, ry, rz] of the current TCP pose, or None if the
        DRFL call glitches (seen transiently after a servo alarm)."""
        if not self.robot_lock.acquire(blocking=False):
            return None
        try:
            posx_data = get_current_posx()
            if not posx_data or not posx_data[0] or len(posx_data[0]) < 6:
                self.get_logger().warn(
                    f"get_current_posx() returned unusable data: {posx_data!r}"
                )
                return None
            return list(posx_data[0])
        except Exception as error:
            self.get_logger().warn(f"get_current_posx() failed: {error}")
            return None
        finally:
            self.robot_lock.release()

    def current_base2cam(self):
        current = self._safe_current_posx()
        if current is None:
            return None
        base2gripper = self.get_robot_pose_matrix(*current)
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
                f"No TF {BASE_FRAME}->{FLANGE_FRAME}; anchors will be stale while moving."
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

    def estimate_floor_height(
        self, u, v, depth_frame, base2cam, center_depth
    ):
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

    # ------------------------------------------------------------------
    # State jobs
    # ------------------------------------------------------------------
    def start_job(self, function):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        worker = threading.Thread(
            target=self._run_job, args=(function,), daemon=True
        )
        worker.start()
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
            with self.lock:
                self.busy = False

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _send_follow_velocity(self, linear_xyz):
        """Publish SpeedlStream directly (bypassing DSR_ROBOT2.speedl()'s
        Python wrapper - its client-side validation rejects vel[0]/vel[1]
        <= 0, which a P-controller legitimately produces at rest/negative
        error). Matches the pattern Doosan's own visual-servo example uses
        for ServolStream."""
        msg = SpeedlStream()
        msg.vel = [
            float(linear_xyz[0]),
            float(linear_xyz[1]),
            float(linear_xyz[2]),
            0.0,
            0.0,
            0.0,
        ]
        msg.acc = [FOLLOW_LINEAR_ACC, FOLLOW_ANGULAR_ACC]
        msg.time = 0.0
        self.speedl_pub.publish(msg)

    def _send_follow_stop(self):
        if self.dry_run:
            return
        self._send_follow_velocity([0.0, 0.0, 0.0])

    def _update_target_estimate(self, measurement, now):
        """Alpha-beta update from a fresh vision measurement."""
        if self.target_estimate is None:
            self.target_estimate = {
                "pos": measurement,
                "vel": np.zeros(3),
                "t": now,
            }
            return
        dt = self._clamp(
            now - self.target_estimate["t"], FOLLOW_AB_DT_MIN, FOLLOW_AB_DT_MAX
        )
        pos = self.target_estimate["pos"]
        vel = self.target_estimate["vel"]
        predicted = pos + vel * dt
        residual = measurement - predicted
        new_pos = predicted + FOLLOW_AB_ALPHA * residual
        new_vel = vel + (FOLLOW_AB_BETA / dt) * residual
        speed = float(np.linalg.norm(new_vel))
        if speed > FOLLOW_MAX_TARGET_SPEED:
            new_vel = new_vel * (FOLLOW_MAX_TARGET_SPEED / speed)
        self.target_estimate = {"pos": new_pos, "vel": new_vel, "t": now}
        if self.velocity_filtered is None:
            self.velocity_filtered = np.zeros(3)
        self.velocity_filtered = (
            FOLLOW_VEL_SMOOTHING * new_vel
            + (1.0 - FOLLOW_VEL_SMOOTHING) * self.velocity_filtered
        )

    def _target_velocity(self):
        """Target velocity as actually used for lead/feedforward: low-passed,
        and zero unless the object is convincingly moving. Feeding the raw
        differentiated estimate forward makes the arm chase its own
        measurement noise - and with an eye-in-hand camera that is a feedback
        loop, not just jitter."""
        if (
            self.target_estimate is None
            or self.velocity_filtered is None
            or not self.predict_enabled
        ):
            return np.zeros(3)
        speed = float(np.linalg.norm(self.velocity_filtered))
        if self.velocity_engaged:
            if speed < FOLLOW_RELEASE_TARGET_SPEED:
                self.velocity_engaged = False
        elif speed >= FOLLOW_MIN_TARGET_SPEED:
            self.velocity_engaged = True
        if not self.velocity_engaged:
            return np.zeros(3)
        return np.asarray(self.velocity_filtered, dtype=float)

    def _coast_target_estimate(self, now):
        """No usable measurement this tick - dead-reckon on last velocity,
        bleeding it off so a stale estimate fades instead of running away.
        Note this advances "t" but NOT last_measurement_at, which is what
        the lost-track timeout watches."""
        dt = max(now - self.target_estimate["t"], 1e-3)
        pos = self.target_estimate["pos"] + self.target_estimate["vel"] * dt
        self.target_estimate = {
            "pos": pos,
            "vel": self.target_estimate["vel"] * FOLLOW_COAST_DECAY,
            "t": now,
        }

    def _measure_target_center(self, box, width, height):
        """Map a possibly edge-clipped YOLO box to (full_center, visible_center,
        visible_fraction, clipped). full_center is the object's estimated true
        centre - outside the image when it is half out of frame - and is what
        the position estimate should use. visible_center always lies inside the
        image and is what depth must be sampled at."""
        x1, y1, x2, y2 = box
        visible_w = max(x2 - x1, 1.0)
        visible_h = max(y2 - y1, 1.0)
        margin = FOLLOW_EDGE_MARGIN_PX

        clip_left = x1 <= margin
        clip_right = x2 >= width - margin
        clip_top = y1 <= margin
        clip_bottom = y2 >= height - margin
        clipped = clip_left or clip_right or clip_top or clip_bottom

        visible_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if not clipped:
            size = np.array([visible_w, visible_h], dtype=float)
            if self.nominal_box is None:
                self.nominal_box = size
            else:
                self.nominal_box = (
                    FOLLOW_BOX_SIZE_ALPHA * size
                    + (1.0 - FOLLOW_BOX_SIZE_ALPHA) * self.nominal_box
                )
            return visible_center, visible_center, 1.0, False

        full_w, full_h = (
            (visible_w, visible_h)
            if self.nominal_box is None
            else (
                max(float(self.nominal_box[0]), visible_w),
                max(float(self.nominal_box[1]), visible_h),
            )
        )

        # Anchor on the edge still inside the frame and extend outward.
        if clip_left and not clip_right:
            fx1, fx2 = x2 - full_w, x2
        elif clip_right and not clip_left:
            fx1, fx2 = x1, x1 + full_w
        else:
            fx1, fx2 = x1, x2
        if clip_top and not clip_bottom:
            fy1, fy2 = y2 - full_h, y2
        elif clip_bottom and not clip_top:
            fy1, fy2 = y1, y1 + full_h
        else:
            fy1, fy2 = y1, y2

        full_center = ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)
        visible_fraction = float(
            (visible_w * visible_h) / max(full_w * full_h, 1e-6)
        )
        return full_center, visible_center, min(visible_fraction, 1.0), True

    def _coast_follow(self, now, note=" (coast)"):
        """Drive from the predicted estimate when there is no usable fix."""
        if self.target_estimate is None:
            return
        if now - self.last_measurement_at > LOST_TRACK_TIMEOUT:
            self.status = f"Lost track of {self.target_name} - show it to the camera"
            self._send_follow_stop()
            return
        current = self._safe_current_posx()
        if current is None:
            return
        self._coast_target_estimate(now)
        velocity = self._target_velocity()
        lead_pos = self.target_estimate["pos"] + velocity * FOLLOW_LEAD_TIME
        self.status = f"Following {self.target_name} (predicted) - press G to grab"
        self._drive_to(
            lead_pos,
            current,
            note=note,
            feedforward=velocity * FOLLOW_FEEDFORWARD_GAIN,
        )

    def _drive_to(self, lead_pos, current, note="", feedforward=None):
        """Rate-limited SpeedL step toward lead_pos: proportional correction
        plus velocity feedforward. Pure P-control leaves a steady-state lag of
        (target speed / gain) behind anything that keeps moving; adding the
        target's own velocity cancels that instead of forcing a huge gain."""
        now = time.monotonic()
        if now - self.last_follow_send_at < FOLLOW_MIN_INTERVAL_SEC:
            return
        self.last_follow_send_at = now

        raw_z = lead_pos[2] + APPROACH_HEIGHT
        target_x = self._clamp(lead_pos[0], FOLLOW_X_MIN, FOLLOW_X_MAX)
        target_y = self._clamp(lead_pos[1], FOLLOW_Y_MIN, FOLLOW_Y_MAX)
        target_z = self._clamp(raw_z, FOLLOW_Z_MIN, FOLLOW_Z_MAX)
        clamped = (
            target_x != lead_pos[0] or target_y != lead_pos[1] or target_z != raw_z
        )

        error = np.array([target_x, target_y, target_z]) - np.array(current[:3])
        error_norm = float(np.linalg.norm(error))
        velocity = np.zeros(3)
        if error_norm >= FOLLOW_DEADBAND_MM:
            velocity = FOLLOW_KP * error
        # Don't feed a runaway target's velocity forward past the safe box.
        if feedforward is not None and not clamped:
            velocity = velocity + np.asarray(feedforward, dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed > FOLLOW_MAX_LINEAR_SPEED:
            velocity = velocity * (FOLLOW_MAX_LINEAR_SPEED / speed)

        self.get_logger().info(
            f"follow{note}: target=({target_x:.1f},{target_y:.1f},{target_z:.1f}) "
            f"current=({current[0]:.1f},{current[1]:.1f},{current[2]:.1f}) "
            f"err={error_norm:.1f}mm vel=({velocity[0]:.1f},{velocity[1]:.1f},{velocity[2]:.1f})"
            + (" [CLAMPED]" if clamped else "")
        )
        if self.dry_run:
            return
        self._send_follow_velocity(velocity.tolist())

    def _acquire_measurement(self, now, current):
        """Try to fold one fresh vision fix into the target estimate.
        Returns (ok, note). Shared by hover-follow and the tracked descent so
        both see identical geometry."""
        detection = self._best_matching_detection(self.target_name)
        if detection is None:
            return False, " (coast)"

        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            return False, " (coast)"
        height, width = depth_frame.shape[:2]

        full_center, visible_center, visible_fraction, clipped = (
            self._measure_target_center(detection["box"], width, height)
        )
        if clipped and visible_fraction < FOLLOW_MIN_VISIBLE_FRACTION:
            # Too little of the object left for its centroid to mean anything.
            return False, " (coast/edge)"

        # Depth must come from a pixel that actually exists; the object's true
        # centre may be outside the frame when it is half out.
        depth = self.sample_depth(
            int(round(visible_center[0])), int(round(visible_center[1])), depth_frame
        )
        if depth is None:
            return False, " (coast)"

        base2cam = self.get_robot_pose_matrix(*current) @ self.gripper2cam
        raw_position = self.pixel_to_base(
            full_center[0], full_center[1], depth, base2cam
        )
        if not np.isfinite(raw_position).all() or raw_position[2] < 0:
            return False, " (coast)"

        self._update_target_estimate(np.asarray(raw_position[:3], dtype=float), now)
        self.last_measurement_at = now
        self.last_tracked = {
            "uv": (int(round(visible_center[0])), int(round(visible_center[1]))),
            "pos": self.target_estimate["pos"].copy(),
            "updated_at": now,
        }
        return True, (f" (clip {visible_fraction:.0%})" if clipped else "")

    def follow_tick(self):
        """One iteration of visual-servo hover tracking via SpeedL. Called
        from the main render loop every frame while state == FOLLOWING.
        An alpha-beta filter estimates the target's position AND velocity,
        so the commanded hover point leads a moving object by
        FOLLOW_LEAD_TIME instead of always chasing where it just was, and
        can keep coasting briefly on the last velocity if the object leaves
        the frame. Non-blocking."""
        now = time.monotonic()
        current = self._safe_current_posx()
        if current is None:
            return

        ok, note = self._acquire_measurement(now, current)
        if not ok:
            self._coast_follow(now, note=note)
            return

        self.status = f"Following {self.target_name} - press G to grab"
        velocity = self._target_velocity()
        lead_pos = self.target_estimate["pos"] + velocity * FOLLOW_LEAD_TIME
        self._drive_to(
            lead_pos,
            current,
            note=note,
            feedforward=velocity * FOLLOW_FEEDFORWARD_GAIN,
        )

    def handle_grasp_key(self):
        if self.state != State.FOLLOWING:
            self.status = f"GRAB unavailable in {self.state.value}"
            return
        if (
            self.last_tracked is None
            or time.monotonic() - self.last_tracked["updated_at"]
            > LOST_TRACK_TIMEOUT
        ):
            self.status = f"No fresh track of {self.target_name}; can't grab yet"
            return
        if self.busy:
            self.status = "Robot is busy; try G again"
            return

        center_u, center_v = self.last_tracked["uv"]
        depth_frame = self.img_node.get_depth_frame()
        base2cam = self.current_base2cam()
        floor_z = None
        if depth_frame is not None and base2cam is not None:
            depth = self.sample_depth(center_u, center_v, depth_frame)
            if depth is not None:
                floor_z = self.estimate_floor_height(
                    center_u, center_v, depth_frame, base2cam, depth
                )
        self.grasp_floor_z = floor_z
        self.grasp_started_at = time.monotonic()
        self.abort_requested = False

        # Open before descending so the fingers are clear on the way down.
        self.grip(close=False)
        self.state = State.GRASPING
        self.status = f"Descending onto {self.target_name} (tracking)..."

    def handle_abort_key(self):
        """Operator stop. Cuts the velocity stream immediately; a tracked
        descent picks the flag up on its next tick."""
        self.abort_requested = True
        self._send_follow_stop()
        if self.state == State.GRASPING:
            self.status = "Stopping..."
        else:
            self.status = "Stopped"

    def _abort_grasp(self, reason):
        self._send_follow_stop()
        self.state = State.FOLLOWING
        self.status = f"Grab aborted: {reason}"
        self.get_logger().warn(f"Grasp aborted: {reason}")

    def grasp_tick(self):
        """Closed-loop descent. Unlike the old freeze-and-plunge, this keeps
        folding in fresh vision while lowering Z, so the grasp still lands if
        the object drifts. Descent is gated on XY alignment, and the last
        GRASP_BLIND_MM is committed blind (gripper occludes the object and the
        depth camera is past its useful near range by then)."""
        now = time.monotonic()
        if self.abort_requested:
            self._abort_grasp("operator stop")
            return
        if now - self.grasp_started_at > GRASP_TIMEOUT:
            self._abort_grasp("timed out")
            return

        current = self._safe_current_posx()
        if current is None:
            return

        ok, note = self._acquire_measurement(now, current)
        if not ok:
            if now - self.last_measurement_at > GRASP_LOST_TIMEOUT:
                self._abort_grasp(f"lost {self.target_name}")
                return
            # Brief dropout: coast the estimate rather than freezing outright.
            if self.target_estimate is not None:
                self._coast_target_estimate(now)

        if self.target_estimate is None:
            self._abort_grasp("no target estimate")
            return

        estimate_pos = self.target_estimate["pos"]
        estimate_vel = self._target_velocity()

        # Object top surface minus the plunge, but never far below the surface
        # the object is sitting on - that is what stops a thin object (or a bad
        # depth reading) turning into a table collision.
        grasp_z = float(estimate_pos[2]) - PLUNGE_MM
        if self.grasp_floor_z is not None:
            grasp_z = max(
                grasp_z, float(self.grasp_floor_z) - GRASP_MAX_BELOW_FLOOR_MM
            )
        grasp_z = self._clamp(grasp_z, FOLLOW_Z_MIN, FOLLOW_Z_MAX)
        height_above = float(current[2]) - grasp_z

        # INTERCEPTION, not pursuit. Aim at where the object will be once we
        # have descended that far, and cross laterally at the speed that gets
        # us there at the same moment - so the arm sweeps in diagonally and
        # meets a moving object instead of hovering over where it used to be.
        # The schedule must close out at the COMMIT height, not at the object:
        # aiming at h=0 leaves a proportional slice of the lateral error still
        # open when the blind phase starts, which then reads as a miss.
        time_to_contact = max(
            (height_above - GRASP_BLIND_MM) / GRASP_DESCENT_SPEED,
            GRASP_MIN_TIME_TO_CONTACT,
        )
        rendezvous = estimate_pos[:2] + estimate_vel[:2] * time_to_contact
        true_x, true_y = float(rendezvous[0]), float(rendezvous[1])
        target_x = self._clamp(true_x, FOLLOW_X_MIN, FOLLOW_X_MAX)
        target_y = self._clamp(true_y, FOLLOW_Y_MIN, FOLLOW_Y_MAX)
        # If the rendezvous is outside the safe box we can only reach the
        # boundary. Error MUST still be judged against the real rendezvous,
        # otherwise the arm sits on the clamp, calls itself aligned, and
        # commits a grasp onto empty space.
        outside_workspace = target_x != true_x or target_y != true_y

        offset_xy = np.array([target_x, target_y]) - np.array(current[:2])
        miss_distance = float(
            np.linalg.norm(np.array([true_x, true_y]) - np.array(current[:2]))
        )

        if height_above <= GRASP_BLIND_MM:
            if outside_workspace or miss_distance >= GRASP_ALIGN_TOLERANCE:
                self._abort_grasp(
                    "target out of reach at commit"
                    if outside_workspace
                    else f"missed by {miss_distance:.0f}mm at commit"
                )
                return
            self._send_follow_stop()
            # The blind plunge itself takes time, during which the object keeps
            # moving - aim where it will be by the time the fingers arrive.
            commit_xy = estimate_pos[:2] + estimate_vel[:2] * GRASP_COMMIT_LEAD_TIME
            self.pick = {
                "uv": self.last_tracked["uv"],
                "pos": np.array(
                    [
                        self._clamp(float(commit_xy[0]), FOLLOW_X_MIN, FOLLOW_X_MAX),
                        self._clamp(float(commit_xy[1]), FOLLOW_Y_MIN, FOLLOW_Y_MAX),
                        float(estimate_pos[2]),
                    ]
                ),
                "grasp_z": grasp_z,
                "height": (
                    None
                    if self.grasp_floor_z is None
                    else float(estimate_pos[2]) - float(self.grasp_floor_z)
                ),
                "plunge": PLUNGE_MM,
                "orientation": list(self.follow_orientation),
                "name": self.target_name,
            }
            self.state = State.PICKING
            self.status = f"Snatching {self.target_name}..."
            if not self.start_job(self._job_commit_grasp):
                self._abort_grasp("robot busy at commit")
            return

        # Speed needed to cover the lateral gap within the descent window.
        # (offset / t) expands to P-control with gain 1/t plus exact velocity
        # feedforward, so the correction naturally sharpens near contact.
        required_speed = float(np.linalg.norm(offset_xy)) / time_to_contact
        if required_speed <= GRASP_MAX_XY_SPEED:
            velocity_xy = offset_xy / time_to_contact
            descent_scale = 1.0
        else:
            # Can't make the rendezvous in time: go flat out laterally and
            # stretch the schedule by descending slower, rather than arriving
            # at grasp height beside the object.
            velocity_xy = offset_xy * (GRASP_MAX_XY_SPEED / (required_speed * time_to_contact))
            descent_scale = GRASP_MAX_XY_SPEED / required_speed
        if outside_workspace:
            descent_scale = 0.0
        velocity_z = -GRASP_DESCENT_SPEED * descent_scale

        if outside_workspace:
            self.status = f"{self.target_name} is outside the safe area - holding"
        else:
            self.status = (
                f"Intercepting {self.target_name} "
                f"(miss {miss_distance:.0f}mm, {height_above:.0f}mm to go)"
            )

        # Same send cadence as the hover loop; the GUI loop itself runs faster.
        if now - self.last_follow_send_at < FOLLOW_MIN_INTERVAL_SEC:
            return
        self.last_follow_send_at = now

        self.get_logger().info(
            f"intercept{note}: miss={miss_distance:.1f}mm h={height_above:.1f}mm "
            f"ttc={time_to_contact:.2f}s descent={descent_scale:.0%} "
            f"vel=({velocity_xy[0]:.1f},{velocity_xy[1]:.1f},{velocity_z:.1f})"
        )
        if self.dry_run:
            return
        self._send_follow_velocity([velocity_xy[0], velocity_xy[1], velocity_z])

    def _job_commit_grasp(self):
        """Blind final plunge, close, lift. SpeedL streaming is already
        stopped; give the controller a moment to settle before switching
        back to queued MoveL commands."""
        time.sleep(GRASP_SETTLE_SEC)
        position = self.pick["pos"]
        grasp_z = self.pick["grasp_z"]
        rx, ry, rz = self.pick["orientation"]

        self.go_linear(
            "grasp commit", position[0], position[1], grasp_z, rx, ry, rz
        )
        self.grip(close=True)
        self.go_linear(
            "grasp lift",
            position[0],
            position[1],
            grasp_z + APPROACH_HEIGHT,
            rx,
            ry,
            rz,
            velocity=LINEAR_LIFT_VELOCITY,
            acceleration=LINEAR_LIFT_ACC,
        )
        self.state = State.HOLDING
        self.status = f"Holding {self.target_name} - press H"

    def handle_home_key(self):
        if self.busy:
            self.status = "Robot is busy; HOME command ignored"
            return

        if self.state == State.HOLDING:
            self.state = State.MOVING_HOME
            self.status = "Moving to PLACE camera view..."
            self.start_job(self._job_go_place_view)
            return

        if self.state == State.WAIT_VOICE:
            self.state = State.MOVING_HOME
            self.status = "Moving HOME before PICK..."
            self.start_job(self._job_go_home_before_pick)
            return

        self.status = f"HOME unavailable in {self.state.value}"

    def _job_go_home_before_pick(self):
        self.go_joint("home before pick", PLACE_VIEW_JOINT)
        self.state = State.WAIT_VOICE
        self.status = "HOME ready - say a tool name"

    def _job_go_place_view(self):
        self.go_joint("place camera view", PLACE_VIEW_JOINT)
        self.state = State.WAIT_PLACE_CLICK
        self.status = "Click the PLACE point"

    def handle_gripper_key(self, close):
        if self.busy:
            self.status = "Robot is busy; gripper command ignored"
            return
        if self.state in (
            State.FOLLOWING,
            State.GRASPING,
            State.PICKING,
            State.MOVING_HOME,
            State.PLACING,
        ):
            self.status = f"Gripper unavailable in {self.state.value}"
            return

        previous_state = self.state
        action = "Closing" if close else "Opening"
        self.status = f"{action} gripper..."
        self.start_job(
            lambda: self._job_manual_grip(close, previous_state)
        )

    def _job_manual_grip(self, close, previous_state):
        self.grip(close=close)
        if not close and previous_state in (
            State.HOLDING,
            State.WAIT_PLACE_CLICK,
        ):
            with self.lock:
                self.pick = None
                self.place = None
                self.target_name = None
            self.state = State.WAIT_VOICE
            self.status = "Gripper opened - waiting for voice command"
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

        depth_frame = self.img_node.get_depth_frame()
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

        current = get_current_posx()[0]
        rx, ry, rz = current[3:]
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

        with self.lock:
            self.pick = None
            self.place = None
            self.target_name = None
        self.state = State.WAIT_VOICE
        self.status = "Done - waiting for next voice command"

    # ------------------------------------------------------------------
    # Robot primitives
    # ------------------------------------------------------------------
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
        result = movej(
            posj(joints), vel=JOINT_VELOCITY, acc=JOINT_ACC
        )
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

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
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
            return entry["uv"]
        point = self.project_to_pixel(entry["pos"], base2cam)
        if point is None:
            return None
        height, width = frame_shape[:2]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return None
        return point

    def render(self):
        frame = self.img_node.get_color_frame()
        if frame is None:
            return
        visual = frame.copy()

        with self.detection_lock:
            detections = list(self.latest_detections)
            inference_error = self.inference_error

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["box"])
            selected = detection["name"] == self.target_name
            color = COLOR_TARGET if selected else COLOR_OTHER
            thickness = 3 if selected else 1
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                visual,
                f"{detection['name']} {detection['score']:.2f}",
                (x1, max(45, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                thickness,
                cv2.LINE_AA,
            )

        base2cam = self.tf_base2cam()
        pick_pixel = self._anchored_pixel(self.pick, base2cam, visual.shape)
        place_pixel = self._anchored_pixel(self.place, base2cam, visual.shape)
        if pick_pixel is not None:
            self.draw_marker(visual, pick_pixel, COLOR_PICK, "PICK")
        if place_pixel is not None:
            self.draw_marker(visual, place_pixel, COLOR_PLACE, "PLACE")
        if self.state in (State.FOLLOWING, State.GRASPING):
            tracked_pixel = self._anchored_pixel(
                self.last_tracked, base2cam, visual.shape
            )
            if tracked_pixel is not None:
                label = "TRACK" if self.state == State.FOLLOWING else "GRASP"
                self.draw_marker(visual, tracked_pixel, COLOR_TARGET, label)

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
        if self.state in (State.FOLLOWING, State.GRASPING) and self.target_estimate:
            raw = float(np.linalg.norm(self.target_estimate["vel"]))
            used = float(np.linalg.norm(self._target_velocity()))
            cv2.putText(
                visual,
                f"target vel: est {raw:5.0f} -> used {used:5.0f} mm/s"
                f"   predict {'ON' if self.predict_enabled else 'OFF'}",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

        help_text = (
            "[F] follow  [G] grab  [SPACE] STOP  [P] predict  [H] home  "
            "[O] open  [C] close  [D] dry-run  [ESC] quit"
        )
        cv2.putText(
            visual,
            help_text,
            (10, visual.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        if self.dry_run:
            cv2.putText(
                visual,
                "DRY RUN",
                (visual.shape[1] - 120, visual.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if inference_error:
            cv2.putText(
                visual,
                f"YOLO ERROR: {inference_error}",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("Voice Pick and Place", visual)

    def shutdown(self):
        # Never leave a velocity streaming into the controller on the way out.
        try:
            self._send_follow_stop()
        except Exception as error:
            self.get_logger().warn(f"Could not send stop on shutdown: {error}")
        self.running = False
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        self.camera_executor.shutdown()
        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        self.camera_executor.remove_node(self.img_node)
        self.camera_executor.remove_node(self)
        self.img_node.destroy_node()
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
        print(" Voice OR press F (locks onto best-scoring detection) -> follow -> G to grab")
        print(" F: follow (no voice needed) | G: snatch - intercepts on a diagonal")
        print(" SPACE: STOP (aborts a descent immediately)")
        print(" H: HOME | O: gripper open | C: gripper close | D: dry-run | ESC: quit")
        print("=" * 72)

        while rclpy.ok():
            controller.request_voice_if_ready()
            if controller.state == State.FOLLOWING:
                controller.follow_tick()
            elif controller.state == State.GRASPING:
                controller.grasp_tick()
            controller.render()
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                if controller.busy or controller.state == State.GRASPING:
                    controller.handle_abort_key()
                    print("Stopping motion. Press ESC again once it has stopped.")
                    continue
                break
            if key == 32:  # SPACE
                controller.handle_abort_key()
            elif key in (ord("f"), ord("F")):
                controller.handle_follow_key()
            elif key in (ord("g"), ord("G")):
                controller.handle_grasp_key()
            elif key in (ord("h"), ord("H")):
                controller.handle_home_key()
            elif key in (ord("o"), ord("O")):
                controller.handle_gripper_key(close=False)
            elif key in (ord("c"), ord("C")):
                controller.handle_gripper_key(close=True)
            elif key in (ord("d"), ord("D")) and not controller.busy:
                controller.dry_run = not controller.dry_run
                print(f"DRY RUN: {'ON' if controller.dry_run else 'OFF'}")
            elif key in (ord("p"), ord("P")):
                controller.predict_enabled = not controller.predict_enabled
                print(
                    f"PREDICTION: {'ON' if controller.predict_enabled else 'OFF'}"
                )
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
