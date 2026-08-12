import math
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
from scipy import ndimage
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger

import DR_init
from dsr_msgs2.msg import SpeedlStream
from sensor_msgs.msg import JointState as JointStateMsg
from dsr_msgs2.srv import SetRobotControl
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
# How far the camera stands off is not a fixed number: it is driven so the
# object keeps a steady apparent size. Standing further widens the field of
# view, so the object survives more motion before reaching an edge, but costs
# resolution - and the right trade depends on how big the thing is, which a
# constant cannot know. The ratio of measured to wanted pixel size needs no
# model of the object. Bounds live with the orbit constants (ORBIT_RADIUS_*),
# because that same standoff is the orbit radius.
FOLLOW_FRAME_FILL = 0.55  # object's long side, as a fraction of frame height
FOLLOW_HOVER_SMOOTHING = 0.08  # standoff should drift, not dart
FOLLOW_HOVER_INITIAL = 250.0  # until the first detection sizes the object
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
# A detection older than this is treated as no detection (coast instead).
# _inference_loop runs at roughly 0.08s + one YOLO pass, so this leaves
# headroom for a slow frame without letting a truly stale box get
# re-triangulated against the robot's *current* pose - with an eye-in-hand
# camera that reads pure ego-motion as target velocity and runs away.
FOLLOW_DETECTION_MAX_AGE = 0.25
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
FOLLOW_COAST_DECAY = 0.6
# How long the arm may keep moving on a guess after the object disappears.
# Coasting exists to ride out a dropped frame or two, not to commit to a
# direction: dead-reckoning for a full second sent the arm somewhere the object
# never went, and re-acquiring then snapped it back the other way. Past this it
# holds still, while the track itself stays alive (LOST_TRACK_TIMEOUT) so the
# id lock survives.
FOLLOW_COAST_MOTION_SEC = 0.25
# A measurement arriving after a gap this long is treated as a fresh fix rather
# than fed to the filter. The velocity term is residual/dt, so a large jump
# after a gap would be read as enormous speed and then fed forward - a lunge
# in whatever direction the object happened to reappear.
FOLLOW_RESEED_GAP_SEC = 0.4

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
# Following is locked to one tracked id, not to "whichever box of that class
# scores highest this frame" - with two of the same tool in view that would
# swap target silently, and a single low-scoring frame would do the same.
# If the id vanishes for longer than this the lock is dropped and re-taken,
# because ByteTrack retires an id once the object has been gone a while and
# gives the same object a new one when it comes back.
FOLLOW_ID_REACQUIRE_SEC = 1.5
# Losing the track while swung out to the side used to strand the arm there:
# past the timeout _coast_follow stopped calling the drive step, so the orbit
# elevation never wound back and the arm sat at a viewpoint from which it could
# no longer see the object to re-acquire it. Overhead is the widest view and
# the pose most likely to find the object again, so a lost track now walks the
# viewpoint back up - on the last known position, for a bounded time only.
ORBIT_RECOVER_SEC = 4.0

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
# Dead-man on every streamed velocity. speedl() applies a command for `time`
# seconds and then stops; at 0.0 it holds the last velocity forever, so a main
# loop that stalls (a blocking DRFL call, an exception) leaves the arm driving
# with nothing left running to stop it. We refresh at up to 20 Hz
# (FOLLOW_MIN_INTERVAL_SEC), so this is ~4x the send period: invisible in normal
# operation, but the arm coasts to a halt within it if the commands stop coming.
FOLLOW_COMMAND_TTL_SEC = 0.2

# Look-at wrist reorientation (L key, off by default). Rotates the wrist to
# keep the camera pointed at the target as it is lifted/tilted, on top of the
# unchanged X/Y/Z position tracking. Nothing here guards joint limits or wrist
# singularities - there is no such guard anywhere in this file - so the tilt is
# bounded to a cone around the orientation captured at follow-start and the
# angular speed is capped well under what the arm can do.
LOOKAT_MAX_TILT_DEG = 30.0  # max tilt away from the follow-start camera axis
LOOKAT_KP = 1.5  # (deg/s) commanded per (deg) of orientation error
LOOKAT_MAX_ANGULAR_SPEED = 10.0  # deg/s cap
# Low-pass on the angular command. The desired aim direction jumps around with
# every noisy detection, and feeding that straight through produced commands
# that reversed sign between consecutive ticks (ang y went -18 -> -0.4 -> +7 ->
# -23 deg/s in one run). The controller reads that jerk as a violation and
# drops into protective stop long before the speed cap alone would. Same idea
# and same weighting as FOLLOW_VEL_SMOOTHING on the linear side.
LOOKAT_SMOOTHING = 0.25
LOOKAT_MIN_TARGET_DIST_MM = 80.0  # closer than this, the look-at direction is noise
# Depth-based refinement of the detection box. The box centre is not the
# object: for anything elongated it orbits the true centroid as the object
# turns (~57mm on this hammer) and at roughly a fifth of the orientations it
# falls between the head and the handle, where the depth patch reads the table
# behind. Keeping only the pixels that are clearly nearer than the background
# gives the object's own centroid and its own depth instead.
# Background comes from the border ring of the box, which is background by
# construction, and the cut is placed a fraction of the object's own height
# below it rather than a fixed distance. A fixed 25mm kept only whatever stood
# proud by that much: with the hammer lying on the table its 40mm head cleared
# the bar and its 18mm handle did not, so the mask became the head alone and
# the target sat on the claw.
MASK_HEIGHT_RATIO = 0.35  # cut this far down from background toward the object
MASK_MIN_SEPARATION_MM = 8.0  # ...but never inside the depth noise
MASK_BORDER_PX = 3  # width of the ring sampled for background
MASK_MIN_BORDER_PX = 20  # too little border to judge background from
MASK_MIN_VALID_PX = 50  # too few depth returns to judge background at all
MASK_MIN_PIXELS = 30  # too small a mask to trust a centroid from
# With look-at on, the wrist can cover a whole cone from a standing position, so
# the arm does not have to chase the object to stay pointed at it. Below this
# fraction of the tilt budget the arm holds the posture it started following
# from and lets the wrist do the work; past it the arm blends in until, at the
# cone limit, it follows fully. Without this the arm walks itself out to the
# edge of the safe box on any lively motion and has nowhere left to go.
LOOKAT_ARM_ENGAGE_FRACTION = 0.6
# Consecutive failed pose reads before FOLLOWING gives up and stops the arm.
POSE_READ_FAILURE_LIMIT = 5
# When look-at is on, the hover target is shifted so the CAMERA sits over the
# object instead of the tool (see _lookat_position_offset). That is what lets
# the wrist aim straight at the target without fighting the position loop.
LOOKAT_DEADBAND_DEG = 1.0  # correction below this -> command zero
# Wrist singularity guard. On this arm axes 4 and 6 line up as J5 approaches 0,
# and there the Jacobian blows up: a small commanded rotation asks for an
# enormous J4/J6 rate. Cartesian caps do not help - they bound the tool, not
# the joints - so the arm whips round and the controller protective-stops.
# HOME sits at J5 = +90 and the measured orbit stayed within +44..+151, so this
# only bites when look-at has walked the wrist somewhere unusual.
# Angular command is faded out between these two, and refused below the floor.
LOOKAT_WRIST_SAFE_DEG = 30.0   # full angular authority above this |J5|
LOOKAT_WRIST_FLOOR_DEG = 15.0  # no angular command at or below this
# G assumes a top-down descent, so refuse it if look-at has left the wrist tilted.
GRASP_ORIENTATION_TOLERANCE_DEG = 5.0

# Orbit (B). Not a mode: the tracker keeps streaming SpeedL exactly as it does
# while following, and the orbit only moves WHERE the camera is asked to be -
# from straight overhead round to the side. All six joints resolve that the way
# the teach pendant resolves a TCP-frame jog, and the object stays tracked the
# whole way, because nothing about the control path changed.
#
# The envelope below is not guessed. cuRobo was used offline (orbit_probe.py)
# to sweep this arm: on the base side, 400mm standoff, elevation 30-75deg plans
# 100% of the time, keeps the wrist branch, and costs 22-30deg of joint travel
# per step. Orbiting the far side (+X) fails almost everywhere - the flange
# leaves the 900mm envelope even when the TCP looks fine.
ORBIT_AZIMUTH_DEG = 180.0          # 180 = swing round the base side
ORBIT_TOP_ELEVATION_DEG = 90.0     # straight overhead = ordinary following
# How far round it may go. Measured on this arm with a 400mm standoff and the
# object held at 400mm: elevation 0 (level with the object, looking straight in
# at its side) plans fine and even -10 works, but J5 falls with it - +79deg at
# elevation 30, +38 at 0, +26 at -10 - and below about 30deg of J5 the wrist
# guard starts taking authority away. 0 is the honest edge: it sees the side
# square-on and still leaves the wrist ~38deg of room.
ORBIT_MIN_ELEVATION_DEG = 0.0
# The viewpoint elevation is not just an orbit control - it decides whether a
# lifted object can be looked at from a sane arm posture at all. Held overhead,
# the flange climbs with the object and runs out of arm; but simply dropping to
# the lowest reachable angle is worse, because part of that range is only
# reachable by flipping the elbow over.
#
# Measured on this cell (400mm standoff, object at x=450 and x=600, taking the
# highest elevation that keeps BOTH J3 and J5 on the same side as HOME):
#
#   object height (mm):  150  250  350  450  550  650  750
#   highest same-branch:  80   80   60   50   30   30   10
#
# The earlier version of this table came from a reach-only sweep and read ~46deg
# at 500mm - right in the band where the elbow turns over (J3 went to -51deg
# against HOME's +90). That is what made the arm fold its elbow up instead of
# just angling the wrist. Reachable was never the whole question; same-branch
# is. BENCHMARK.md says exactly this about picking work points, and it applies
# to the elbow as much as the wrist.
# The 350-450mm rows are pulled down from what the coarse sweep allowed: at
# x=600, z=400 the elbow still turned over at 55deg. The table is sampled every
# 100mm and the interpolation between rows can land inside a flip band, so the
# entries either side of a known-bad point are lowered rather than trusting the
# midpoint.
ORBIT_AUTO_ELEVATION_TABLE = (
    (150.0, 80.0), (250.0, 75.0), (350.0, 45.0), (450.0, 40.0),
    (550.0, 30.0), (650.0, 30.0), (750.0, 10.0),
)
ORBIT_RATE_DEG_S = 22.0            # how fast the viewpoint walks round
# 22deg/s takes the sweep from overhead to side-on in under 3s. It is a rate of
# change of the TARGET, not of the arm: the position loop still chases it under
# its own speed and acceleration caps, so raising this makes the viewpoint lead
# further ahead rather than making the arm snap.
# Standoff band, which is also the orbit radius. Deliberately wide, and NOT
# narrowed toward 300mm: standing closer folds the wrist up. Measured J5 at
# elevation 30 was +44deg at a 280mm standoff, +55 at 300, +71 at 350 and 400 -
# so pulling the band in pushes the arm toward the very singularity the guard
# above exists to survive. Room to stand back is what keeps the wrist open.
ORBIT_RADIUS_MIN_MM, ORBIT_RADIUS_MAX_MM = 250.0, 500.0

# Safe box for the TCP target, base-frame millimetres. No longer a guess: the
# reachable set was swept with cuRobo (object on a grid, camera at a 400mm
# standoff, best viewpoint chosen per cell) and this is the box that contains
# it with a little margin. The old 100-500mm ceiling in Z was the reason the
# arm stopped following an object once it was lifted much past waist height -
# it was a placeholder, not a limit of the machine.
FOLLOW_X_MIN, FOLLOW_X_MAX = 120.0, 780.0
FOLLOW_Y_MIN, FOLLOW_Y_MAX = -450.0, 450.0
FOLLOW_Z_MIN, FOLLOW_Z_MAX = 80.0, 800.0


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
        # Wrist look-at reorientation. On by default, and not really optional
        # any more: the viewpoint elevation now moves with the object's height,
        # so the camera is put to the SIDE of a lifted object. Aiming is what
        # makes that a view of the object rather than a view of the floor
        # beside it. With look-at off the elevation is pinned overhead
        # (_reachable_elevation), where pointing straight down is correct - so
        # position and aim can never disagree.
        self.lookat_enabled = True
        self.lookat_reference_forward = None
        self.lookat_angular_filtered = None
        self.follow_home_xyz = None
        self.camera_distance = FOLLOW_HOVER_INITIAL
        self.orbit_active = False
        self.orbit_elevation = ORBIT_TOP_ELEVATION_DEG
        self.orbit_last_tick = None
        self.orbit_clamped = False
        self.pose_read_failures = 0

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
        self.target_track_id = None      # locked ByteTrack id, None until acquired
        self.target_id_seen_at = 0.0
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
        # J5 is watched every tick for the wrist singularity, so take it from
        # the state stream rather than paying for a DRFL round trip.
        self.measured_joints = None
        self.create_subscription(
            JointStateMsg, f"/{ROBOT_ID}/joint_states", self._joint_state_cb, 10
        )
        # Used by R to clear a safety stop without restarting the node.
        self.robot_control_client = self.create_client(
            SetRobotControl, f"/{ROBOT_ID}/system/set_robot_control"
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
        self.orbit_active = False
        self.orbit_elevation = ORBIT_TOP_ELEVATION_DEG
        self.orbit_last_tick = None
        self.target_requested_at = time.monotonic()
        self.last_tracked = None
        self.target_track_id = None
        self.target_id_seen_at = time.monotonic()
        self.target_estimate = None
        self.velocity_filtered = None
        self.velocity_engaged = False
        self.last_measurement_at = time.monotonic()
        self.nominal_box = None
        self.follow_orientation = list(get_current_posx()[0][3:])
        # Posture to hold while the wrist alone can keep the target in view.
        self.follow_home_xyz = np.asarray(
            get_current_posx()[0][:3], dtype=float
        )
        self.lookat_reference_forward = self._camera_forward(self.follow_orientation)
        self.lookat_angular_filtered = None
        self.camera_distance = FOLLOW_HOVER_INITIAL
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
        # Lock the instance the operator was actually looking at when they
        # pressed F, not just its class.
        if best.get("id") is not None:
            self.target_track_id = best["id"]
            self.target_id_seen_at = time.monotonic()

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
                detections = self.yolo.track_frame(
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

    def _best_matching_detection(self, target, now):
        """The detection being followed, or None. Non-blocking single check.

        Prefers the locked track id over the score. Score alone is not identity:
        it moves around frame to frame, so on any frame where a second hammer
        edges ahead the arm would quietly change its mind about which one it is
        chasing."""
        with self.detection_lock:
            detections = list(self.latest_detections)
            updated_at = self.detections_updated_at
        if updated_at < self.target_requested_at:
            return None
        if now - updated_at > FOLLOW_DETECTION_MAX_AGE:
            return None
        matches = [d for d in detections if d["name"] == target]
        if not matches:
            return None

        if self.target_track_id is not None:
            locked = [d for d in matches if d.get("id") == self.target_track_id]
            if locked:
                self.target_id_seen_at = now
                return locked[0]
            # Gone this frame. ByteTrack bridges short gaps itself, so hold the
            # lock briefly rather than grabbing whatever else is in frame.
            if now - self.target_id_seen_at < FOLLOW_ID_REACQUIRE_SEC:
                return None
            self.get_logger().warn(
                f"track id {self.target_track_id} lost for "
                f"{now - self.target_id_seen_at:.1f}s - re-acquiring"
            )
            self.target_track_id = None

        chosen = max(matches, key=lambda item: item["score"])
        if chosen.get("id") is not None:
            self.target_track_id = chosen["id"]
            self.target_id_seen_at = now
            self.get_logger().info(
                f"locked onto {target} id {self.target_track_id}"
            )
        return chosen

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

    # ------------------------------------------------------------------
    # Look-at wrist reorientation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vector):
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return None
        return np.asarray(vector, dtype=float) / norm

    def _camera_forward(self, orientation):
        """Camera optical axis (+Z) in base frame for a given [rx, ry, rz]."""
        gripper_rotation = self.get_robot_pose_matrix(0.0, 0.0, 0.0, *orientation)[
            :3, :3
        ]
        return self._normalize((gripper_rotation @ self.gripper2cam[:3, :3])[:, 2])

    def _on_pose_read_failed(self):
        """A tick with no usable robot pose. Stop rather than let the last
        streamed velocity keep running, and give up on FOLLOWING if the reads
        keep failing - past that point we are commanding an arm whose position
        we cannot see."""
        self.pose_read_failures += 1
        self._send_follow_stop()
        if self.pose_read_failures >= POSE_READ_FAILURE_LIMIT:
            self.status = (
                f"Lost robot pose {self.pose_read_failures}x - stopped; "
                "check the controller connection, then press F again"
            )
            self.get_logger().error(
                f"get_current_posx() failed {self.pose_read_failures} times in a "
                "row; leaving FOLLOWING with the arm stopped"
            )
            self.state = State.WAIT_VOICE
            self.target_name = None
            self.target_estimate = None
            self.velocity_filtered = None
            self.pose_read_failures = 0

    def _posture_blended_target(self, desired_tcp, object_pos):
        """Pull the hover target back toward the posture we started following
        from, by however much the wrist can cover on its own.

        Chasing the object with the whole arm burns the safe box for motion the
        wrist could have absorbed, and once the arm is pinned against a clamp it
        has no way to answer the next move - which is what "gets stuck as soon
        as it moves a bit" was. Letting the wrist take the first
        LOOKAT_ARM_ENGAGE_FRACTION of the cone keeps the arm parked in the
        posture it knows is good and still hands over smoothly when the target
        really does leave the wrist's reach."""
        if not self.lookat_enabled or self.follow_home_xyz is None:
            return desired_tcp
        if self.lookat_reference_forward is None:
            return desired_tcp

        # How far off the neutral aim would the object be if the arm just
        # stayed home? That is what decides whether the arm needs to move.
        home_pose = list(self.follow_home_xyz) + list(self.follow_orientation)
        home_base2cam = self.get_robot_pose_matrix(*home_pose) @ self.gripper2cam
        to_target = self._normalize(
            np.asarray(object_pos, dtype=float) - home_base2cam[:3, 3]
        )
        if to_target is None:
            return desired_tcp
        tilt = math.degrees(
            math.acos(
                self._clamp(
                    float(np.dot(self.lookat_reference_forward, to_target)), -1.0, 1.0
                )
            )
        )

        engage = LOOKAT_MAX_TILT_DEG * LOOKAT_ARM_ENGAGE_FRACTION
        if tilt <= engage:
            blend = 0.0
        elif tilt >= LOOKAT_MAX_TILT_DEG:
            blend = 1.0
        else:
            blend = (tilt - engage) / max(LOOKAT_MAX_TILT_DEG - engage, 1e-6)
        # Hold the arm back in X/Y only. Height is what buys the wide field of
        # view (FOLLOW_HOVER_HEIGHT), and standing tall costs no tilt budget, so
        # pinning Z to wherever the arm happened to be when F was pressed only
        # made look-at hover lower than plain following for no gain.
        desired_tcp = np.asarray(desired_tcp, dtype=float)
        held = self.follow_home_xyz + blend * (desired_tcp - self.follow_home_xyz)
        return np.array([held[0], held[1], desired_tcp[2]])

    def _lookat_desired_forward(self, current, target_pos):
        """Unit direction (base frame) the camera should point: straight at the
        target, bounded to a cone around the orientation captured at
        follow-start.

        A function of the target and the reference only - never of the current
        orientation. An earlier version held the wrist wherever it happened to
        be while the target was comfortably in view, which made the tilt a
        one-way integrator: it accumulated in whatever direction the target had
        pushed it and never unwound, so bringing the target back to the centre
        left the wrist still bent from the last excursion."""
        if self.lookat_reference_forward is None:
            return None

        base2cam = self.get_robot_pose_matrix(*current) @ self.gripper2cam
        offset = np.asarray(target_pos, dtype=float) - base2cam[:3, 3]
        if float(np.linalg.norm(offset)) < LOOKAT_MIN_TARGET_DIST_MM:
            return None
        forward = self._normalize(offset)
        if forward is None:
            return None

        reference = self.lookat_reference_forward
        tilt = math.degrees(
            math.acos(self._clamp(float(np.dot(reference, forward)), -1.0, 1.0))
        )
        if tilt <= LOOKAT_MAX_TILT_DEG:
            return forward
        # Slide along the cone surface rather than freezing at the limit, so a
        # target that keeps moving past it still tracks smoothly.
        cone_axis = self._normalize(np.cross(reference, forward))
        if cone_axis is None:
            return None
        return Rotation.from_rotvec(
            cone_axis * math.radians(LOOKAT_MAX_TILT_DEG)
        ).apply(reference)

    def _lookat_angular_velocity(self, current, desired_forward):
        """Bounded angular velocity (deg/s) that swings the camera's optical
        axis onto desired_forward.

        Deliberately only constrains the two pointing DOF. A full look-at
        basis would also pin roll about the optical axis to whatever "up"
        convention it picked, and that roll is both useless (it just rotates
        the image, the target stays centred either way) and expensive - it
        spends wrist travel toward joint limits and the wrist singularity,
        which nothing in this file guards against. Rotating the smallest
        amount that fixes the aim leaves roll wherever the arm already had it.

        The error is axis-angle rather than a ZYZ difference on purpose: ZYZ
        is singular at ry ~ 0, exactly where this rig sits pointing straight
        down, and there its components jump between frames for no physical
        reason."""
        current_rotation = self.get_robot_pose_matrix(*current)[:3, :3]
        current_forward = self._normalize(
            (current_rotation @ self.gripper2cam[:3, :3])[:, 2]
        )
        if current_forward is None:
            return np.zeros(3)

        angle = math.degrees(
            math.acos(
                self._clamp(float(np.dot(current_forward, desired_forward)), -1.0, 1.0)
            )
        )
        if angle < LOOKAT_DEADBAND_DEG:
            return np.zeros(3)
        axis = self._normalize(np.cross(current_forward, desired_forward))
        if axis is None:
            return np.zeros(3)

        angular = LOOKAT_KP * angle * axis
        speed = float(np.linalg.norm(angular))
        if speed > LOOKAT_MAX_ANGULAR_SPEED:
            angular = angular * (LOOKAT_MAX_ANGULAR_SPEED / speed)
        return angular

    def _tilt_from_follow_orientation(self):
        """Degrees between the current wrist orientation and the one captured
        at follow-start, or None if either is unavailable."""
        if self.follow_orientation is None:
            return None
        current = self._safe_current_posx()
        if current is None:
            return None
        current_rotation = self.get_robot_pose_matrix(*current)[:3, :3]
        reference_rotation = self.get_robot_pose_matrix(
            0.0, 0.0, 0.0, *self.follow_orientation
        )[:3, :3]
        error = Rotation.from_matrix(
            current_rotation @ reference_rotation.T
        ).as_rotvec(degrees=True)
        return float(np.linalg.norm(error))

    def _lookat_velocity_for(self, current, position):
        """Angular velocity to keep `position` in view, or None when look-at is
        off or there is nothing usable to aim at.

        Never raises: this runs inside the main render loop, and a stalled loop
        leaves the arm streaming its last velocity."""
        if not self.lookat_enabled or position is None:
            return None
        try:
            desired_forward = self._lookat_desired_forward(current, position)
            if desired_forward is None:
                raw = np.zeros(3)
            else:
                raw = self._lookat_angular_velocity(current, desired_forward)
            return self._smooth_angular(raw)
        except Exception as error:
            self.get_logger().warn(f"look-at failed, holding orientation: {error}")
            return None

    def _joint_state_cb(self, message):
        try:
            index = {name: i for i, name in enumerate(message.name)}
            self.measured_joints = np.array(
                [message.position[index[f"joint_{i}"]] for i in range(1, 7)]
            )
        except (KeyError, IndexError):
            pass

    def wrist_margin_deg(self):
        """How far J5 is from the singularity at 0, or None if unknown."""
        if self.measured_joints is None:
            return None
        return abs(float(np.degrees(self.measured_joints[4])))

    def _wrist_damping(self):
        """1.0 with room to spare, fading to 0 as the wrist folds toward the
        singularity. Fading rather than cutting: a hard stop on the angular
        command is itself a step input, which is the thing being avoided."""
        margin = self.wrist_margin_deg()
        if margin is None:
            return 1.0
        if margin >= LOOKAT_WRIST_SAFE_DEG:
            return 1.0
        if margin <= LOOKAT_WRIST_FLOOR_DEG:
            return 0.0
        span = LOOKAT_WRIST_SAFE_DEG - LOOKAT_WRIST_FLOOR_DEG
        return float((margin - LOOKAT_WRIST_FLOOR_DEG) / span)

    def _smooth_angular(self, raw):
        """Low-pass the angular command so a jittery aim direction cannot be
        handed to the controller as a sign-flipping step input."""
        if self.lookat_angular_filtered is None:
            self.lookat_angular_filtered = np.zeros(3)
        damping = self._wrist_damping()
        if damping < 1.0:
            self.get_logger().warn(
                f"wrist {self.wrist_margin_deg():.0f}deg from singularity - "
                f"angular authority {damping:.0%}",
                throttle_duration_sec=2.0,
            )
        self.lookat_angular_filtered = (
            LOOKAT_SMOOTHING * np.asarray(raw, dtype=float) * damping
            + (1.0 - LOOKAT_SMOOTHING) * self.lookat_angular_filtered
        )
        speed = float(np.linalg.norm(self.lookat_angular_filtered))
        if speed < LOOKAT_DEADBAND_DEG:
            return np.zeros(3)
        return self.lookat_angular_filtered.copy()

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

    def _send_follow_velocity(self, linear_xyz, angular_xyz=None):
        """Publish SpeedlStream directly (bypassing DSR_ROBOT2.speedl()'s
        Python wrapper - its client-side validation rejects vel[0]/vel[1]
        <= 0, which a P-controller legitimately produces at rest/negative
        error). Matches the pattern Doosan's own visual-servo example uses
        for ServolStream."""
        angular = np.zeros(3) if angular_xyz is None else np.asarray(angular_xyz)
        msg = SpeedlStream()
        msg.vel = [
            float(linear_xyz[0]),
            float(linear_xyz[1]),
            float(linear_xyz[2]),
            float(angular[0]),
            float(angular[1]),
            float(angular[2]),
        ]
        msg.acc = [FOLLOW_LINEAR_ACC, FOLLOW_ANGULAR_ACC]
        msg.time = FOLLOW_COMMAND_TTL_SEC
        self.speedl_pub.publish(msg)

    def _send_follow_stop(self):
        if self.dry_run:
            return
        self._send_follow_velocity([0.0, 0.0, 0.0])

    def _update_target_estimate(self, measurement, now):
        """Alpha-beta update from a fresh vision measurement."""
        if (
            self.target_estimate is not None
            and now - self.last_measurement_at > FOLLOW_RESEED_GAP_SEC
        ):
            # Long gap: start over at where the object actually is. Filtering
            # across the gap would turn the jump into velocity and lunge.
            self.target_estimate = None
            self.velocity_filtered = None
            self.velocity_engaged = False
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
        """No usable measurement this tick - carry on briefly, then hold.

        Note this advances "t" but NOT last_measurement_at, which is what the
        lost-track timeout watches."""
        dt = max(now - self.target_estimate["t"], 1e-3)
        if now - self.last_measurement_at > FOLLOW_COAST_MOTION_SEC:
            # Past the point where a guess is worth anything. Freeze the target
            # where it was last believed to be, so the arm settles instead of
            # travelling on toward somewhere nothing was ever seen.
            self.target_estimate = {
                "pos": self.target_estimate["pos"],
                "vel": np.zeros(3),
                "t": now,
            }
            self.velocity_filtered = np.zeros(3)
            self.velocity_engaged = False
            return
        pos = self.target_estimate["pos"] + self.target_estimate["vel"] * dt
        self.target_estimate = {
            "pos": pos,
            "vel": self.target_estimate["vel"] * FOLLOW_COAST_DECAY,
            "t": now,
        }
        # Bleed off the *fed-forward* velocity too. _target_velocity() returns
        # velocity_filtered, which is only rewritten on a fresh measurement, so
        # decaying the estimate alone left the arm being pushed at the last
        # measured speed for as long as the track stayed lost.
        if self.velocity_filtered is not None:
            self.velocity_filtered = self.velocity_filtered * FOLLOW_COAST_DECAY

    def _update_camera_distance(self, box, current, base2cam, target_pos, frame_shape):
        """Drive the standoff so the object holds a steady apparent size.

        Pure ratio: if it is showing up half as big as we want, the camera
        wants to be half as far away. That needs no idea of the object's real
        dimensions, so a screwdriver and a hammer each settle at their own
        sensible distance instead of sharing one hardcoded number."""
        box_pixels = max(box[2] - box[0], box[3] - box[1])
        if box_pixels <= 1.0:
            return
        wanted_pixels = FOLLOW_FRAME_FILL * frame_shape[0]
        if wanted_pixels <= 1.0:
            return

        camera_position = base2cam[:3, 3]
        distance = float(
            np.linalg.norm(np.asarray(target_pos, dtype=float) - camera_position)
        )
        if distance < LOOKAT_MIN_TARGET_DIST_MM:
            return
        wanted_distance = self._clamp(
            distance * (box_pixels / wanted_pixels),
            ORBIT_RADIUS_MIN_MM, ORBIT_RADIUS_MAX_MM,
        )
        # How far the CAMERA should stand off. This is the orbit radius too:
        # orbiting is the same standoff held at a different elevation, so both
        # follow and orbit work from this one number.
        self.camera_distance = (
            FOLLOW_HOVER_SMOOTHING * wanted_distance
            + (1.0 - FOLLOW_HOVER_SMOOTHING) * self.camera_distance
        )

    def _advance_orbit(self, now):
        """Walk the orbit elevation toward its target at a fixed rate.

        Deliberately a slow ramp rather than a jump: the elevation feeds the
        desired camera point, and the position loop chases that point, so
        moving it gently is what makes the arm sweep round smoothly instead of
        lunging at a new pose."""
        dt = 0.0 if self.orbit_last_tick is None else now - self.orbit_last_tick
        self.orbit_last_tick = now
        dt = self._clamp(dt, 0.0, 0.2)  # a stalled loop must not teleport it

        target = (
            ORBIT_MIN_ELEVATION_DEG if self.orbit_active else self._reachable_elevation()
        )
        # Standing against the safe-box clamp means the arm cannot actually get
        # to where the current elevation already asks. Winding it further would
        # just open a gap between what we command and where the arm can be.
        if self.orbit_clamped and target < self.orbit_elevation:
            return
        # Winding the viewpoint further round is what folds the wrist up. If it
        # is already near the singularity, stop asking for more.
        if self._wrist_damping() <= 0.0 and target < self.orbit_elevation:
            self.status = "Orbit paused - wrist near singularity"
            return
        step = ORBIT_RATE_DEG_S * dt
        if abs(target - self.orbit_elevation) <= step:
            self.orbit_elevation = float(target)
        else:
            self.orbit_elevation += step if target > self.orbit_elevation else -step

    def _reachable_elevation(self):
        """Highest viewpoint that can still reach the object at its height.

        Straight down for something on the table, sliding round toward level as
        it is lifted. This is the arm's reach talking, not a preference: with
        the camera held overhead the flange climbs with the object and leaves
        the envelope, which is why following used to stall as soon as the
        object went much above the table."""
        if self.target_estimate is None:
            return ORBIT_TOP_ELEVATION_DEG
        if not self.lookat_enabled:
            # Nothing is steering the camera, so the only viewpoint whose fixed
            # top-down orientation actually faces the object is straight above
            # it. Moving round without aiming would just point the camera at
            # the floor next to the object.
            return ORBIT_TOP_ELEVATION_DEG
        height = float(self.target_estimate["pos"][2])
        heights = [h for h, _ in ORBIT_AUTO_ELEVATION_TABLE]
        angles = [a for _, a in ORBIT_AUTO_ELEVATION_TABLE]
        # Interpolated, so the viewpoint slides rather than stepping as the
        # object is lifted; clamped at both ends by np.interp.
        return float(np.interp(height, heights, angles))

    def _desired_camera_position(self, object_pos):
        """Where the camera should be: on the arc around the object, at the
        current orbit elevation.

        At 90 deg this is straight overhead - the plain hover the tracker has
        always done - so following and orbiting are the same equation with a
        different angle, not two modes."""
        elevation = np.radians(self.orbit_elevation)
        azimuth = np.radians(ORBIT_AZIMUTH_DEG)
        direction = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])
        return np.asarray(object_pos, dtype=float) + self.camera_distance * direction

    def _tcp_for_camera(self, camera_target, current):
        """TCP pose that puts the camera at camera_target.

        Full 3D, unlike the old X/Y-only nudge: on the arc the camera sits
        beside and below the tool, not just off to one side, so the height has
        to come out of the same transform."""
        rotation = self.get_robot_pose_matrix(*current)[:3, :3]
        return np.asarray(camera_target, dtype=float) - rotation @ self.gripper2cam[:3, 3]

    def _depth_mask_measure(self, box, depth_frame):
        """(centroid_u, centroid_v, depth_mm, mask_fraction) for the object's
        own pixels inside `box`, or None when foreground and background are not
        separable - which is the honest answer for an object lying flat on the
        table, and where the caller falls back to the plain box centre."""
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        height, width = depth_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2 + 1), min(height, y2 + 1)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        crop = depth_frame[y1:y2, x1:x2]
        valid = crop[crop > 0]
        if valid.size < MASK_MIN_VALID_PX:
            return None

        border = np.ones(crop.shape, dtype=bool)
        if crop.shape[0] > 2 * MASK_BORDER_PX and crop.shape[1] > 2 * MASK_BORDER_PX:
            border[MASK_BORDER_PX:-MASK_BORDER_PX, MASK_BORDER_PX:-MASK_BORDER_PX] = False
        border_depths = crop[border & (crop > 0)]
        if border_depths.size < MASK_MIN_BORDER_PX:
            return None
        background = float(np.median(border_depths))
        nearest = float(np.percentile(valid, 5))
        height = max(background - nearest, 0.0)
        cut = max(MASK_MIN_SEPARATION_MM, MASK_HEIGHT_RATIO * height)
        near = (crop > 0) & (crop < background - cut)
        if int(near.sum()) < MASK_MIN_PIXELS:
            return None

        # Pixels with no depth return that touch the object are almost always
        # the object: this hammer's head is dark metal, it swallows the IR, and
        # keeping only pixels that answered drops the head and slides the
        # centroid down the handle. Grow the mask through the unknown region
        # instead - but only into blobs the object actually touches, so a
        # dropout patch of bare table does not get absorbed.
        unknown = crop <= 0
        components, count = ndimage.label(near | unknown)
        mask = near.copy()
        if count:
            touching = np.unique(components[near])
            mask = np.isin(components, touching[touching > 0])

        depths = crop[mask & (crop > 0)]
        if depths.size < MASK_MIN_PIXELS:
            return None
        pixels = int(mask.sum())
        rows, columns = np.nonzero(mask)
        return (
            float(columns.mean()) + x1,
            float(rows.mean()) + y1,
            float(np.median(depths)),
            pixels / float(mask.size),
        )

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
        lost_for = now - self.last_measurement_at
        if lost_for > LOST_TRACK_TIMEOUT:
            returning = (
                self.orbit_elevation < ORBIT_TOP_ELEVATION_DEG - 1.0
                and lost_for < LOST_TRACK_TIMEOUT + ORBIT_RECOVER_SEC
            )
            if not returning:
                self.status = (
                    f"Lost track of {self.target_name} - show it to the camera"
                )
                self._send_follow_stop()
                return
            # Wind the viewpoint back overhead so the object can come back into
            # view. The estimate is stale, so this drives to a geometric pose
            # off the last known position - never to a fresh measurement - and
            # only until it is back up or the window closes.
            self.orbit_active = False
            self.status = f"Lost {self.target_name} - returning overhead"
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
            angular_xyz=self._lookat_velocity_for(
                current, self.target_estimate["pos"]
            ),
        )

    def _drive_to(self, lead_pos, current, note="", feedforward=None, angular_xyz=None):
        """Rate-limited SpeedL step toward lead_pos: proportional correction
        plus velocity feedforward. Pure P-control leaves a steady-state lag of
        (target speed / gain) behind anything that keeps moving; adding the
        target's own velocity cancels that instead of forcing a huge gain."""
        now = time.monotonic()
        if now - self.last_follow_send_at < FOLLOW_MIN_INTERVAL_SEC:
            return
        self.last_follow_send_at = now

        # Where the object actually is. The posture blend has to see this, not
        # a camera-centred version of it: the ~85mm camera offset is about
        # 12deg of apparent tilt at a 400mm standoff, and feeding that in would
        # hand the arm work the wrist could still have absorbed.
        object_pos = np.asarray(lead_pos, dtype=float)
        self._advance_orbit(now)
        camera_target = self._desired_camera_position(object_pos)
        # Swing the look-at cone round with the orbit. The cone is a guard
        # against the wrist wandering somewhere silly, and it is measured from
        # a reference direction - but if that reference stays frozen at whatever
        # the arm held when F was pressed, then at 30deg elevation the camera
        # needs 60deg of tilt to face the object and the cone clamps it at 30.
        # The aim would stop short and the object would slide out of frame,
        # which is the one thing the orbit exists to avoid.
        # The reference is taken from the COMMANDED camera point, not from
        # where the object is measured to be. Both rotate with the orbit, but
        # the commanded point moves at a known 12deg/s while the measurement
        # jitters every frame - and the reference is what the +/-30deg cone is
        # measured against. Pinning it to the raw measurement would make the
        # cone follow the wrist instead of bounding it, which is how the guard
        # got lost in the first place.
        if self.lookat_enabled:
            aim = self._normalize(object_pos - camera_target)
            if aim is not None:
                self.lookat_reference_forward = aim
        # One equation for both behaviours: put the CAMERA on the arc around
        # the object at the current elevation, then ask what TCP pose puts it
        # there. At 90deg that is the overhead hover the tracker has always
        # done; walking the elevation down is the orbit. No mode switch, no
        # second kind of command - the same SpeedL stream the whole time.
        desired_tcp = self._tcp_for_camera(camera_target, current)
        desired_tcp = self._posture_blended_target(desired_tcp, object_pos)

        raw_z = desired_tcp[2]
        target_x = self._clamp(desired_tcp[0], FOLLOW_X_MIN, FOLLOW_X_MAX)
        target_y = self._clamp(desired_tcp[1], FOLLOW_Y_MIN, FOLLOW_Y_MAX)
        target_z = self._clamp(raw_z, FOLLOW_Z_MIN, FOLLOW_Z_MAX)
        clamped = (
            target_x != desired_tcp[0]
            or target_y != desired_tcp[1]
            or target_z != raw_z
        )
        # Tells the orbit to stop winding further out (see _advance_orbit).
        self.orbit_clamped = clamped

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

        angular_note = ""
        if angular_xyz is not None:
            angular_note = (
                f" ang=({angular_xyz[0]:.1f},{angular_xyz[1]:.1f},"
                f"{angular_xyz[2]:.1f})deg/s"
            )
        self.get_logger().info(
            f"follow{note}: target=({target_x:.1f},{target_y:.1f},{target_z:.1f}) "
            f"current=({current[0]:.1f},{current[1]:.1f},{current[2]:.1f}) "
            f"err={error_norm:.1f}mm vel=({velocity[0]:.1f},{velocity[1]:.1f},{velocity[2]:.1f})"
            + angular_note
            + (" [CLAMPED]" if clamped else "")
        )
        if self.dry_run:
            return
        self._send_follow_velocity(velocity.tolist(), angular_xyz)

    def _acquire_measurement(self, now, current):
        """Try to fold one fresh vision fix into the target estimate.
        Returns (ok, note). Shared by hover-follow and the tracked descent so
        both see identical geometry."""
        detection = self._best_matching_detection(self.target_name, now)
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

        # Prefer the object's own pixels over the box centre. Only when the box
        # is whole: for a clipped box the visible centroid is pulled toward the
        # part still in frame, and _measure_target_center's reconstruction of
        # the true centre is the better estimate there.
        refined = (
            None if clipped else self._depth_mask_measure(detection["box"], depth_frame)
        )
        if refined is not None:
            center_u, center_v, depth, mask_fraction = refined
            full_center = visible_center = (center_u, center_v)
        else:
            mask_fraction = None
            # Depth must come from a pixel that actually exists; the object's
            # true centre may be outside the frame when it is half out.
            depth = self.sample_depth(
                int(round(visible_center[0])),
                int(round(visible_center[1])),
                depth_frame,
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
        self._update_camera_distance(
            detection["box"], current, base2cam, self.target_estimate["pos"],
            depth_frame.shape,
        )
        self.last_measurement_at = now
        self.last_tracked = {
            "uv": (int(round(visible_center[0])), int(round(visible_center[1]))),
            "pos": self.target_estimate["pos"].copy(),
            "updated_at": now,
        }
        if clipped:
            return True, f" (clip {visible_fraction:.0%})"
        if mask_fraction is None:
            return True, " (box)"
        return True, f" (mask {mask_fraction:.0%})"

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
            # Returning silently would leave the last velocity streaming with
            # nothing driving it, which is how a transient DRFL read fault
            # turned into the arm coasting into the table.
            self._on_pose_read_failed()
            return
        self.pose_read_failures = 0

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
            angular_xyz=self._lookat_velocity_for(
                current, self.target_estimate["pos"]
            ),
        )

    def handle_grasp_key(self):
        if self.state != State.FOLLOWING:
            self.status = f"GRAB unavailable in {self.state.value}"
            return
        tilt = self._tilt_from_follow_orientation()
        if tilt is not None and tilt > GRASP_ORIENTATION_TOLERANCE_DEG:
            # The descent is a blind top-down plunge at follow_orientation;
            # starting it from a tilted wrist would drive the gripper somewhere
            # other than where the operator sees it pointing.
            self.status = (
                f"Tilted {tilt:.0f} deg from top-down; return to top-down before grabbing"
            )
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

    def handle_orbit_key(self):
        """Toggle the orbit. Nothing switches mode - this only changes the
        elevation the desired camera point is taken at, and the same SpeedL
        loop walks the arm round."""
        if self.state != State.FOLLOWING:
            self.status = f"ORBIT unavailable in {self.state.value}"
            return
        self.orbit_active = not self.orbit_active
        if self.orbit_active and not self.lookat_enabled:
            # Going round without re-aiming would just carry the object out of
            # frame: which face you see depends on where the camera is, but
            # keeping it in view is the wrist's job.
            self.lookat_enabled = True
            self.get_logger().info("orbit: look-at enabled (needed to stay aimed)")
        self.status = (
            f"Orbiting {self.target_name} - swinging to the side"
            if self.orbit_active
            else f"Returning overhead - following {self.target_name}"
        )

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

    def handle_recover_key(self):
        """R - clear a safety stop and put the arm back in a usable state.

        A flaky link drops the arm into SAFE_STOP (or trips a protective stop
        on a jerky command), and until this existed the only way out was to
        restart the node and re-home, which is far too slow to work with."""
        if self.busy:
            self.status = "Robot is busy; R ignored"
            return
        self.state = State.WAIT_VOICE
        self.status = "Recovering robot..."
        self.start_job(self._job_recover_robot)

    ROBOT_STATE_NAMES = {
        0: "INITIALIZING", 1: "STANDBY", 2: "MOVING", 3: "SAFE_OFF (servo off)",
        4: "TEACHING", 5: "SAFE_STOP", 6: "EMERGENCY_STOP", 7: "HOMING",
        8: "RECOVERY", 9: "SAFE_STOP2", 10: "SAFE_OFF2", 15: "NOT_READY",
    }

    def _robot_state(self):
        try:
            return get_robot_state()
        except Exception as error:
            self.get_logger().warn(f"get_robot_state() failed: {error}")
            return None

    def _state_name(self, state):
        return self.ROBOT_STATE_NAMES.get(state, f"state {state}")

    def _job_recover_robot(self):
        # Everything here blocks on a service round-trip, which is why it runs
        # as a job instead of inline in the render loop.
        self._send_follow_stop()
        self.pose_read_failures = 0
        self.target_estimate = None
        self.velocity_filtered = None
        self.lookat_angular_filtered = None

        state = self._robot_state()
        self.get_logger().info(f"recover: state = {self._state_name(state)}")
        if state in (STATE_STANDBY, STATE_MOVING):
            self.status = "Robot already OK - press H to home, then F to follow"
            return

        # The driver's set_robot_control service answers success=true no matter
        # what the controller actually did, so the only honest check is to read
        # the state back. Servo-on also takes a moment to take effect, hence
        # the retry rather than a single shot.
        for attempt in range(1, 4):
            if state == STATE_SAFE_STOP:
                sequence = [CONTROL_RESET_SAFET_STOP]
            elif state in (STATE_SAFE_OFF, STATE_SAFE_OFF2):
                sequence = [CONTROL_SERVO_ON]
            else:
                sequence = [CONTROL_RESET_SAFET_STOP, CONTROL_SERVO_ON]

            for control in sequence:
                self._call_robot_control(control)
            try:
                set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            except Exception as error:
                self.get_logger().warn(f"set_robot_mode() failed: {error}")

            time.sleep(0.6)
            state = self._robot_state()
            self.get_logger().info(
                f"recover: attempt {attempt} -> {self._state_name(state)}"
            )
            if state in (STATE_STANDBY, STATE_MOVING):
                self.status = "Robot recovered - press H to home, then F to follow"
                return

        # Still stuck. Servo-off with the light still red is normally a physical
        # condition or one the pendant holds authority over; nothing this node
        # can send will clear it.
        self.status = (
            f"Cannot clear {self._state_name(state)} from here - "
            "check the E-stop and release it on the teach pendant"
        )
        self.get_logger().error(
            f"recover: stuck in {self._state_name(state)}. If the robot light is "
            "still red, the safety condition is physical (E-stop engaged) or the "
            "teach pendant holds control authority - clear it there, or restart "
            "the dsr_bringup2 driver."
        )

    def _call_robot_control(self, control):
        if not self.robot_control_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("set_robot_control service unavailable")
            return False
        request = SetRobotControl.Request()
        request.robot_control = int(control)
        future = self.robot_control_client.call_async(request)
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            self.get_logger().warn(f"set_robot_control({control}) timed out")
            return False
        result = future.result()
        return bool(result and getattr(result, "success", False))

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
            selected = detection["name"] == self.target_name and (
                self.target_track_id is None
                or detection.get("id") == self.target_track_id
            )
            color = COLOR_TARGET if selected else COLOR_OTHER
            thickness = 3 if selected else 1
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                visual,
                f"{detection['name']} {detection['score']:.2f}"
                + (f" #{detection['id']}" if detection.get("id") is not None else ""),
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
                f"   predict {'ON' if self.predict_enabled else 'OFF'}"
                f"   look-at {'ON' if self.lookat_enabled else 'OFF'}"
                + (f"   id #{self.target_track_id}"
                   if self.target_track_id is not None else "   id --")
                + f"   view {self.orbit_elevation:.0f}deg"
                f"{' ORBIT' if self.orbit_active else ''}"
                + (f"   wrist {self.wrist_margin_deg():.0f}deg"
                   if self.wrist_margin_deg() is not None else ""),
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

        help_text = (
            "[F] follow  [G] grab  [B] orbit  [SPACE] STOP  [R] recover  "
            "[P] predict  [L] look-at  [H] home  [O] open  [C] close  "
            "[D] dry-run  [ESC] quit"
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

    global get_current_posx, get_current_posj, movej, movel, posj, posx
    global get_robot_state, set_robot_mode
    global STATE_SAFE_STOP, STATE_SAFE_OFF, STATE_SAFE_OFF2, STATE_STANDBY, STATE_MOVING
    global CONTROL_RESET_SAFET_STOP, CONTROL_SERVO_ON, ROBOT_MODE_AUTONOMOUS
    try:
        from DSR_ROBOT2 import get_current_posx, get_current_posj, movej, movel
        from DSR_ROBOT2 import get_robot_state, set_robot_mode
        from DR_common2 import posj, posx
        from DRFC import (
            STATE_SAFE_STOP,
            STATE_SAFE_OFF,
            STATE_SAFE_OFF2,
            STATE_STANDBY,
            STATE_MOVING,
            CONTROL_RESET_SAFET_STOP,
            CONTROL_SERVO_ON,
            ROBOT_MODE_AUTONOMOUS,
        )
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
        print(" B: orbit the tracked object to see another face (needs dry-run OFF)")
        print(" SPACE: STOP (aborts a descent immediately) | R: recover after a safety stop")
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
            elif key in (ord("b"), ord("B")):
                controller.handle_orbit_key()
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
            elif key in (ord("l"), ord("L")):
                controller.lookat_enabled = not controller.lookat_enabled
                print(
                    f"LOOK-AT: {'ON' if controller.lookat_enabled else 'OFF'}"
                    + ("" if controller.lookat_enabled
                       else "  (viewpoint pinned overhead while off)")
                )
            elif key in (ord("r"), ord("R")):
                controller.handle_recover_key()
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