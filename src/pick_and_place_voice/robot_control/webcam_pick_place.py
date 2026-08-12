"""Webcam-guided pick and place with live obstacle avoidance.

The pipeline this file implements, end to end:

 1. DETECT   three fixed, calibrated webcams find the target (YOLO per view)
             and triangulate it into the robot base frame.
 2. APPROACH a straight path from HOME to a hover point above the target is
             followed by streaming SpeedL velocities. The webcams watch for
             anything ENTERING the scene (background diff, robot arm masked
             out); an intruder is triangulated into a sphere and the remaining
             path is re-generated to skirt it - live, while moving. When the
             obstacle leaves, the path straightens again.
 3. REFINE   at the hover point the wrist D435i re-detects the same object;
             the bounding-box centre becomes the grasp target.
 4. GRASP    open gripper, descend to grasp height (guarded against the
             floor), close, lift.
 5. PLACE    move to the place camera view, operator clicks the spot in the
             D435i image, the object is set down there.
 6. HOME     back to the home pose, ready for the next run.

Safety: starts in DRY RUN (D toggles). SPACE stops streaming instantly and
aborts jobs at the next step boundary. Every streamed velocity carries a
0.2 s dead-man TTL, so a stalled loop coasts to a halt instead of driving on.

Run (after `colcon build --packages-select pick_and_place_voice`):

    ros2 run pick_and_place_voice webcam_pick_place [--live] [--auto]
        [--target hammer] [--place-base X,Y,Z]
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import DR_init
from dsr_msgs2.msg import SpeedlStream
from dsr_msgs2.srv import SetRobotControl
from object_detection.realsense import ImgNode
from realsense2_camera_msgs.msg import Extrinsics
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import String as StringMsg
from object_detection.webcam_rig import WebcamRig, KalmanCV, HandIntruderDetector
from object_detection import pointing
from object_detection.yolo import YoloModel
from robot_control.onrobot import RG
from robot_control import ar_hud

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

BASE_FRAME = "base_link"
FLANGE_FRAME = "link_6"
ARM_TF_FRAMES = ("link_1", "link_2", "link_3", "link_4", "link_5", "link_6")

DEFAULT_TARGET = "hammer"
HOME_JOINT = [0, 0, 90, 0, 90, 0]  # top-down camera view, same as the tracker

# Demo start: the approach begins here instead of at home, so the path is long
# enough for a detour around a hand to be visible. This was the verification
# pose; the operator can drive the arm anywhere better and press P to keep it.
FAR_START_MM = (260.0, -280.0, 380.0)
START_POSE_FILE = Path.home() / ".config" / "webcam_pnp" / "start_pose.json"

# Operator trim on the grasp height, millimetres above the computed value.
# Tuned live on the hammer: +6 gripped too high to be sure of the handle, so
# the default sits 7 mm lower (the fingers close nearer the table). Watch for
# the RG2 fingertip switch - if a "close" is ever ignored, raise this with ].
GRASP_LIFT_DEFAULT_MM = -1.0
GRASP_LIFT_STEP_MM = 2.0
GRASP_LIFT_RANGE_MM = (-10.0, 60.0)

# --- millimetre geometry ---------------------------------------------------
HOVER_STANDOFF_MM = 250.0     # hover this far above the detected object
PLUNGE_MM = 35.0              # how far below the object's top the TCP grasps
# Never command the fingertips below this height above the MEASURED floor.
# Commanding the old floor-5 drove the tips into the table (the wrist chain
# still carries a few mm of +z bias), which pushed the RG2's fingertip
# safety switch and latched safety circuit 1 - after which every "close"
# was silently ignored until a power cycle.
GRASP_ABOVE_FLOOR_MM = 3.0
# Grasp and place go BELOW the path-following box: the TCP is at the finger
# tips, and straddling a 25 mm handle puts them single-digit millimetres
# above the table. Clipping those moves to the box floor (as one run did,
# stopping 63 mm short) closes the gripper on air. This is the real hard
# floor, backed up by the floor-ring guard on the measured surface.
GRASP_Z_FLOOR_MM = -15.0
APPROACH_HEIGHT = 100.0       # lift after grasp / approach above place
PLACE_CLEARANCE = 3.0
PLACE_BLIND_DROP = 50.0
DEPTH_PATCH = 5
RING_INNER_MM, RING_OUTER_MM, RING_MIN_SAMPLES = 40.0, 70.0, 8

# Safe box for the TCP, base-frame mm (measured for this cell, see tracker).
BOX_X = (120.0, 780.0)
BOX_Y = (-450.0, 450.0)
BOX_Z = (80.0, 800.0)

# --- webcam detection ------------------------------------------------------
# The tools model was trained on close-up wrist-camera views; at the C270s'
# ~1.3 m standoff the raw frame yields ~0.05 confidence. Cropping each view
# down to the projected workspace and upscaling brings the object back to a
# size the model knows (measured: 0.21/0.27/0.33 across the three cameras),
# and the triangulation ray-gap gate rejects what low confidence lets through.
WEBCAM_CONF = 0.15
WEBCAM_CROP_MARGIN_PX = 20
WEBCAM_CROP_MAX_SCALE = 3.0
WEBCAM_IMGSZ = 960
DETECT_MAX_RAY_GAP_M = 0.040  # rays must agree this well to accept a fix
DETECT_SAMPLES = 6            # median over this many fixes
DETECT_SPREAD_MM = 40.0       # fixes must sit within this radius of another
DETECT_TIMEOUT_SEC = 20.0
# A pick target sits ON the table. A consistent triangulation well above it
# is a real 3D object that is not the target - concretely, the gripper
# hanging from the arm misread as a tool: its rays agree perfectly (it IS
# one object), so only its height gives it away. One run chased exactly
# that phantom to the edge of the workspace.
DETECT_TARGET_Z_MM = (-30.0, 120.0)
# Pointing is not fetching. A tool HELD UP in the other hand is a
# perfectly good thing to point at - and the table-height gate above
# dropped it from the 3D list entirely, so the beam had nothing to stop
# at, the mark fell through to the floor behind it, and the selection
# quietly picked whichever identical tool was still lying on the bench.
POINT_TARGET_Z_MAX_MM = 600.0
DETECT_TARGET_XY_MARGIN_MM = 50.0
# The workspace volume the crop covers, base-frame METERS: the reachable
# table area up to a hand's height above it.
WORKSPACE_CORNERS_M = [
    np.array([x, y, z])
    for x in (0.12, 0.78) for y in (-0.45, 0.45) for z in (0.0, 0.25)
]

# --- approach / path following --------------------------------------------
# Cruise speed is effectively min(KP * LOOKAHEAD, MAX): the carrot sits
# LOOKAHEAD ahead on the path, so KP*LOOKAHEAD is what gets commanded in
# open space. The first tuning (1.2 * 60 = 72 mm/s) read as "the robot
# ignores the replanning" - the path bent instantly but the arm strolled.
FOLLOW_KP = 2.0               # (mm/s) per mm of error toward the carrot
APPROACH_MAX_SPEED = 150.0    # mm/s
APPROACH_ACC = (700.0, 90.0)  # SpeedL linear/angular acc limits
# Slew limit on the COMMANDED velocity. A replan reverses the carrot
# direction in one tick, and handing the controller that sign-flip as a step
# input is what tripped a protective stop early on. 700 mm/s^2 bends a full
# reversal over ~0.4 s while still tracking a moving hand visibly.
APPROACH_SLEW_MM_S2 = 700.0
# Commanding real speed while the TCP does not move means the controller has
# stopped listening (protective stop, servo off). Detect it instead of
# streaming into a dead arm until the timeout.
STUCK_SPEED_MM_S = 15.0
STUCK_DISTANCE_MM = 2.0
STUCK_AFTER_SEC = 2.5
LOOKAHEAD_MM = 70.0           # carrot distance along the path
ARRIVE_TOLERANCE_MM = 6.0
SEND_INTERVAL_SEC = 0.05      # 20 Hz command rate
COMMAND_TTL_SEC = 0.2         # dead-man on every streamed velocity
APPROACH_TIMEOUT_SEC = 90.0
POSE_READ_FAILURE_LIMIT = 5

# --- obstacle handling -----------------------------------------------------
# The path clears the obstacle sphere by PLAN_MARGIN only ("hug it, don't
# flee it" - operator's requirement: within 2 cm of the obstacle). The
# corridor is re-checked with the smaller margin so a path planned just
# outside the sphere is not immediately re-declared blocked by measurement
# jitter (hysteresis).
PLAN_MARGIN_MM = 20.0
CHECK_MARGIN_MM = 5.0
# Straight runs stay two waypoints; a detour is sampled this densely and
# each violating sample is projected onto the inflated sphere's surface -
# which is what makes the avoidance a smooth wrap instead of a tent of
# corners.
PATH_SAMPLE_MM = 12.0
# The controller has a safety zone overhead (two live runs tripped it near
# z~435, pendant confirmed a space-limit violation). All planning stays
# below this; detours that would climb past it wrap horizontally instead.
PLAN_Z_MAX = 400.0
OBSTACLE_HOLD_SEC = 0.7       # keep an unseen obstacle alive this long
REPLAN_MIN_INTERVAL_SEC = 0.15
# While ARMED, move the goal only once the object has really moved this far -
# below it the fix is triangulation jitter and the drawn path would shiver.
ARMED_TARGET_MOVE_MM = 8.0
# A triangulated obstacle further than this outside the workspace box is not
# in the robot's world - people walking around the room register in two
# cameras occasionally, and nothing 2 m away should touch the planner.
OBSTACLE_WORKSPACE_MARGIN_MM = 250.0
# Track gating: a located hand this far from the current track is a
# DIFFERENT hand (the operator's other one, a bystander's), not the tracked
# one moving - with two hands in view the largest-blob pick alternates
# between them and the fused position teleports. Reject the jump; only if
# it persists does the track hand over to the new position.
OBSTACLE_JUMP_GATE_MM = 180.0
# 3, not more: at ~15 Hz each rejected frame is ~70 ms of blindness, and a
# hand that MOVED FAST (not a second hand) reappears far from the track -
# eight rejections added ~0.6 s of "obstacle vanished" after every quick
# arm motion. The jump is also measured against the KF's PREDICTED
# position, so a hand moving continuously fast never trips the gate at all.
OBSTACLE_JUMP_HANDOVER_COUNT = 3
TCP_FREEZE_MARGIN_MM = 25.0   # obstacle this close to the TCP: hold still
# The demo speed dial slows CRUISING, not DODGING: with everything scaled
# to a third, a hand approaching at normal speed simply outruns the
# robot's evasion. While an obstacle is in play the scale is floored here,
# so the arm ambles when clear and snaps awake to avoid.
OBSTACLE_REACT_MIN_SCALE = 0.85

# --- hand delivery (--hand-place) ------------------------------------------
# After the grasp the OPERATOR'S HAND becomes the place target: the arm
# hovers above the tracked hand (same MediaPipe triangulation that made it
# an obstacle on the way in), waits for it to hold still, lowers, opens.
DELIVER_HOVER_MM = 110.0        # TCP this far above the palm while tracking
DELIVER_CLEARANCE_MM = 50.0     # release: object bottom ~5cm above the palm
DELIVER_XY_TOL_MM = 15.0        # aligned when within this of the palm centre
DELIVER_Z_TOL_MM = 25.0
# "Hand is steady" comes from the palm's actual position history, NOT the
# Kalman velocity: the filter is tuned aggressive (q_acc 400) so triangulation
# jitter alone reads as ~40-80 mm/s and a genuinely still hand never passed a
# velocity gate - the first live run hovered forever without releasing.
DELIVER_STABLE_WINDOW_SEC = 0.7   # palm must stay put this long...
DELIVER_STABLE_RADIUS_MM = 30.0   # ...within this of its own median
# Judge spread at the 80th percentile of deviations-from-median, not the
# max: MediaPipe landmarks flick for single frames, and one 40mm spike in
# an otherwise rock-still window kept resetting a max-based gate forever.
DELIVER_STABLE_PERCENTILE = 80.0
DELIVER_HAND_SPEED_MM_S = 120.0   # loose backstop on the KF estimate
DELIVER_DWELL_SEC = 0.5         # aligned+steady this long -> commit
DELIVER_SPEED_SCALE = 0.8       # track the hand a touch slower than approach
DELIVER_TIMEOUT_SEC = 60.0      # no hand for this long -> click-place fallback
DELIVER_RELEASE_VEL = 60        # slow descent onto a person's hand

# --- free-wrist avoidance (--free-wrist, all six axes) ---------------------
# While TRAVELLING the tool may tilt away from straight-down: toward the
# direction of motion (it "leads" the move) and away from the obstacle
# (more physical clearance for the gripper body). The tilt budget ramps to
# zero near the goal so the arm ARRIVES pointing straight down, ready to
# grasp; a final movel snaps out whatever residual the ramp leaves.
FREEWRIST_TILT_MAX_DEG = 25.0
FREEWRIST_KP = 2.0                 # (deg/s) per deg of pointing error
FREEWRIST_MAX_ANG_SPEED = 30.0     # deg/s cap on the streamed rotation
FREEWRIST_SMOOTHING = 0.25         # low-pass; raw aim jitters with vision
FREEWRIST_RESTORE_DIST_MM = 140.0  # inside this range of the goal: go down
FREEWRIST_OBSTACLE_RANGE_MM = 160.0  # tilt-away influence past the surface
FREEWRIST_MOTION_WEIGHT = 0.7      # how strongly travel direction tilts
FREEWRIST_DEADBAND_DEG = 0.8
FREEWRIST_ARRIVE_TOL_DEG = 2.0     # residual tilt fixed by movel at arrival
# J5 singularity guard (axes 4 and 6 align as J5 -> 0): fade angular
# authority rather than cutting it, same as the tracker's look-at guard.
WRIST_SAFE_DEG = 30.0
WRIST_FLOOR_DEG = 15.0

# --- D435i refine / grasp --------------------------------------------------
REFINE_CONF = 0.5
REFINE_SETTLE_SEC = 0.6
REFINE_CENTER_TOL_MM = 10.0   # re-centre above the target until within this
REFINE_ATTEMPTS = 3
# Depth-mask segmentation inside the detection box (values straight from the
# proven tracker): pixels clearly nearer than the border-ring background are
# the object's own. The mask gives a grasp point ON the object (a hammer's
# bare box centre can fall between head and handle) and its long axis.
MASK_BORDER_PX = 3
MASK_MIN_BORDER_PX = 20
MASK_MIN_VALID_PX = 50
MASK_MIN_PIXELS = 30
MASK_HEIGHT_RATIO = 0.35
MASK_MIN_SEPARATION_MM = 8.0
# The RG2's fingers open along base +X at the home orientation (measured by
# unprojecting the two fingertips from cam0). To grasp an elongated tool the
# opening axis must be PERPENDICULAR to its long axis, so the wrist yaws by
# (long-axis angle + 90), wrapped to the nearest quarter turn.
FINGER_AXIS_HOME_DEG = 0.0    # opening axis angle in the base XY plane
GRASP_MAX_YAW_DEG = 90.0
LINEAR_VEL, LINEAR_ACC = 100, 100
LIFT_VEL, LIFT_ACC = 80, 80
JOINT_VEL, JOINT_ACC = 60, 60

COLOR_TARGET = (80, 220, 80)
COLOR_OBSTACLE = (60, 60, 255)
COLOR_PATH = (255, 200, 60)
COLOR_TCP = (0, 255, 255)
COLOR_PLACE = (80, 160, 255)
COLOR_TEXT = (255, 255, 255)

# Depth ends of the path ribbon: what is close to the camera burns bright and
# warm, what is far cools off. Same hue family as COLOR_PATH so the plain HUD
# and the AR HUD still look like the same program.
# Inspection: go and photograph the spot a finger is pointing at, for a
# vision model to judge. The height is a compromise - low enough that a
# fastener fills useful pixels, high enough that the frame still shows what
# it is attached to, which is what "am I doing this right?" needs.
#
# This is the height of the GRIPPER TIPS above the spot, NOT of the lens: a
# straight-down move drives the tips at the work, and the wrist camera sits
# 237 mm further back along the tool axis (measured, hand-eye calibration).
# 400 mm clears the conveyor (~100 mm tall) with three hands' width to spare;
# 300 mm also worked but looked like a dive from across the bench. Do not
# lower it to make the picture tighter - crop instead; the arm cannot back
# out of a collision, and the spot being on the table says nothing about what
# is standing next to it.
INSPECT_HEIGHT_MM = 400.0
INSPECT_HEIGHT_MIN_MM = 220.0     # live floor for - / = , never below this
INSPECT_HEIGHT_MAX_MM = 460.0
# Oblique inspection: look at the spot from where the FINGER is looking, so
# the picture shows the face of the work the operator is looking at, not its
# top. Tilt is measured from vertical; 0 is the proven straight-down shot,
# which is also where every fallback lands.
INSPECT_TILT_DEG = 45.0           # asked for when the finger allows it
INSPECT_TILT_MAX_DEG = 60.0       # past this the wrist fights its own limits
INSPECT_TILT_STEP_DEG = 5.0       # how finely the search backs off
INSPECT_STANDOFF_MM = 420.0       # camera-to-spot distance along the view
INSPECT_STANDOFF_MAX_MM = 620.0   # further than this and the part is a speck
INSPECT_STANDOFF_MIN_MM = 200.0   # closer than this and the frame is too tight
INSPECT_MIN_CAMERA_Z_MM = 150.0   # keep the wrist clear of the bench
# The TCP is the GRIPPER TIPS, 150-230 mm beyond the flange, so it hangs far
# below the lens. Judging it by the camera's floor rejected every legal
# oblique pose and quietly dropped the feature back to shooting straight
# down. The tips only have to clear the bench and whatever is lying on it.
INSPECT_MIN_TCP_Z_MM = 100.0
# M0609 joint travel, with margin. A pose the sphere check accepts can still
# need a joint past its stop - only the robot's own inverse kinematics knows.
JOINT_LIMITS_DEG = ((-360, 360), (-95, 95), (-135, 135),
                    (-360, 360), (-135, 135), (-360, 360))
JOINT_LIMIT_MARGIN_DEG = 5.0
#: Wrist singularity: J5 through zero makes J4/J6 spin to keep the tool
#: pointing, which is a protective stop waiting to happen. The streaming
#: controller already fades authority between 15 and 30 deg; a pose planned
#: from scratch has no excuse to sit inside that band at all.
J5_SINGULARITY_MARGIN_DEG = 18.0
#: Inverse kinematics is a service round trip; cap how many candidates get
#: one so a hopeless spot cannot stall the arm for a second.
INSPECT_IK_CHECKS = 8
#: How long a finished inspection stays re-shootable without pointing again.
INSPECT_RECALL_SEC = 90.0
#: After the shot the arm HOLDS at the viewing pose while the verdict is
#: worked out. A second look ("closer", "wider") then starts from where the
#: camera already is instead of going home and coming back - which is both
#: slow and, for the person watching, a confusing retreat before an answer.
INSPECT_HOLD_SEC = 30.0

# --- putting something down "anywhere free" --------------------------------
#: Keep this far from anything already on the bench, centre to centre. A tool
#: is up to ~200 mm long and the gripper needs room beside it, so touching
#: clearance is not enough - the arm has to be able to open, lower and let go.
FREE_SPOT_CLEARANCE_MM = 170.0
#: And this far from the person, whose hand is the one thing on the bench that
#: moves while the arm is deciding.
FREE_SPOT_HUMAN_CLEARANCE_MM = 320.0
#: Search grid over the reachable bench, and how far in from its edge to stay.
FREE_SPOT_STEP_MM = 45.0
FREE_SPOT_EDGE_MARGIN_MM = 90.0

# --- taking something back OFF a person's hand -----------------------------
#: The reverse of the handover, and the one manoeuvre in this system where
#: the gripper closes near fingers. Every number here is chosen to fail
#: towards "did not grip" rather than "gripped the wrong thing".
TAKE_HOVER_MM = 130.0            # park this far above the palm to look
#: Never let the fingertips go below this above the palm surface, whatever
#: the depth camera says the object's underside is. A palm is soft and the
#: reading is noisy; the object is what should be squeezed, never the hand.
TAKE_MIN_ABOVE_PALM_MM = 18.0
#: The hand must hold still like this before the arm descends, and keep
#: holding still while it does - the same median/percentile test the delivery
#: uses, because raw jitter never settles.
TAKE_STABLE_WINDOW_SEC = 0.8
TAKE_STABLE_RADIUS_MM = 25.0
TAKE_ABORT_MOVE_MM = 35.0        # palm moved this far mid-descent -> back off
TAKE_DESCEND_VEL, TAKE_DESCEND_ACC = 35, 35
TAKE_TIMEOUT_SEC = 45.0
# M0609 reaches 900 mm. This sphere is only a cheap PRE-FILTER to keep the
# candidate list short - inverse kinematics is the authority on what the arm
# can hold, so being too strict here just throws away poses the robot would
# happily take (a corner of the bench was coming out as "no view possible").
INSPECT_MAX_REACH_MM = 890.0
INSPECT_MIN_REACH_MM = 260.0
# Deliberately slow. A protective stop costs a pendant reset and the demo;
# an extra two seconds costs nothing. Also split into two legs so the wrist
# rotates while the tool is already parked over the spot.
INSPECT_VEL, INSPECT_ACC = 45, 45
INSPECT_TILT_VEL, INSPECT_TILT_ACC = 30, 30
INSPECT_SETTLE_SEC = 0.6          # let the arm stop ringing before the shot
INSPECT_CLEAR_TIMEOUT_SEC = 12.0  # waiting for the hand to leave
# Inside the project, not the home directory: these images are project data
# that the copilot reads back and that belongs with the rest of the work.
# INSPECT_DIR in the environment overrides it.
INSPECT_DIR = Path(os.environ.get(
    "INSPECT_DIR", str(Path.home() / "cobot2_ws_1" / "data" / "inspections")))
#: Height of the work surface in base coordinates. The pointing ray is
#: intersected with this plane, so it is where a finger "lands".
TABLE_Z_MM = 0.0
# The finger's ray is intersected with a horizontal plane to get a 3D point,
# and the table is only the right plane when the thing being pointed at lies
# ON the table. Point at the side of the conveyor and the ray passes over it
# and lands on the floor behind - which is what the camera then photographs.
# Raising this plane slides the mark back along the ray to the height of the
# surface actually being pointed at.
AIM_PLANE_STEP_MM = 10.0
AIM_PLANE_MAX_MM = 300.0

COLOR_OTHER = (150, 150, 150)      # a class you could switch to
COLOR_POINTED = (120, 255, 160)    # the class the finger settled on
COLOR_POINT_RAY = (0, 220, 255)
#: The latched spot once the hand is gone - same hue, calmer, so it reads as
#: "still selected" rather than "being aimed right now".
COLOR_LOCKED = (90, 170, 255)
COLOR_RAY_FADED = (55, 105, 150)
#: Committed by [I]: brighter and ringed, because from here until the shot
#: is taken this is where the arm is going regardless of what the hand does.
COLOR_LOCKED_FIRM = (150, 220, 255)
COLOR_PATH_NEAR = (255, 235, 150)
COLOR_PATH_FAR = (200, 90, 20)
COLOR_OBSTACLE_SHELL = (90, 90, 255)
PATH_DASH_SPEED = 90.0        # mm/s the dashes march toward the goal

# Speed dial presets on keys 1..N. Two steps turned out to be too coarse for
# demos - the useful range is between "dramatically slow" and full speed, so
# the middle is where the extra stops go.
SPEED_PRESETS = (
    (0.25, "CRAWL"),
    (0.45, "SLOW"),
    (0.70, "BRISK"),
    (1.00, "FULL"),
)

QUAD_W, QUAD_H = 640, 360     # 2x2 grid of 1280x720 sources, halved

WINDOW = "Webcam Pick and Place"


#: Filled in by main() from DSR_ROBOT2 when the driver provides them.
ikin = None
get_current_solution_space = None


def _speed_color(scale):
    """Cyan when held back, green at full speed - the demo dial at a glance."""
    if scale >= 0.95:
        return (80, 255, 80)
    if scale >= 0.6:
        return (80, 230, 200)
    return (0, 255, 255)


class State(Enum):
    IDLE = "IDLE"
    HOMING = "HOMING"
    DETECT = "DETECT"
    # Target found, path planned and replanning live, but the arm is parked at
    # the demo start until the operator presses G. Nothing about the approach
    # changes - this only splits "get ready" from "go", so the replanning can
    # be watched (wave a hand, see the path bend) before anything moves.
    ARMED = "ARMED"
    # Going to photograph a spot for the vision model, then coming home.
    INSPECT = "INSPECT"
    APPROACH = "APPROACH"
    REFINE = "REFINE"
    GRASP = "GRASP"
    TO_PLACE_VIEW = "TO_PLACE_VIEW"
    WAIT_PLACE_CLICK = "WAIT_PLACE_CLICK"
    PLACING = "PLACING"
    DELIVER_TRACK = "DELIVER_TRACK"
    DELIVER_RELEASE = "DELIVER_RELEASE"
    # Taking an object back off the operator's hand, then putting it down.
    TAKE_FROM_HAND = "TAKE_FROM_HAND"
    ERROR = "ERROR"


def load_wrist_depth_calibration():
    """T_flange_wrist_depth (mm) from the webcam calibration bundle, or None.

    Measured against the tutorial T_gripper2camera on the live table: the
    tutorial chain reads the tabletop at z = +26..32 mm (which is why grasps
    stopped a full hand above the object), the bundle's holdout-validated
    wrist calibration reads it at +5..11 mm."""
    path = (Path.home() / "webcam_calibration" / "results" / "transforms"
            / "production_transforms.yaml")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        matrix = np.asarray(
            data["transforms"]["T_flange_wrist_depth"]["matrix"], dtype=float
        )
        matrix[:3, 3] *= 1000.0  # bundle is in meters
        return matrix
    except Exception:
        return None


def resolve_calibration_path():
    package_path = Path(get_package_share_directory("pick_and_place_voice"))
    source_file = Path(__file__).resolve()
    workspace = source_file.parents[2]
    candidates = [
        package_path / "resource" / "T_gripper2camera.npy",
        source_file.parents[1] / "resource" / "T_gripper2camera.npy",
        workspace / "corecode" / "Calibration_Tutorial" / "T_gripper2camera.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "T_gripper2camera.npy not found: " + ", ".join(map(str, candidates))
    )


# ---------------------------------------------------------------------------
# Path geometry (all mm)
# ---------------------------------------------------------------------------
def _segment_point_distance(p0, p1, c):
    """Distance from point c to segment p0-p1, and the closest point."""
    p0, p1, c = (np.asarray(v, dtype=float) for v in (p0, p1, c))
    axis = p1 - p0
    length_sq = float(np.dot(axis, axis))
    if length_sq < 1e-9:
        return float(np.linalg.norm(c - p0)), p0
    t = float(np.clip(np.dot(c - p0, axis) / length_sq, 0.0, 1.0))
    closest = p0 + t * axis
    return float(np.linalg.norm(c - closest)), closest


def _segment_segment_distance(p0, p1, q0, q1):
    """Minimum distance between segments p0-p1 and q0-q1 (both clamped)."""
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(np.dot(d1, d1)), float(np.dot(d2, d2)), float(np.dot(d2, r))
    if a < 1e-9 and e < 1e-9:
        return float(np.linalg.norm(r))
    if a < 1e-9:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e < 1e-9:
            s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
                if denominator > 1e-9 else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))


def _capsule_distance(point, obstacle):
    """Distance from a point to the obstacle capsule's AXIS (subtract the
    radius for surface distance). obstacle = (end_a, end_b, radius)."""
    return _segment_point_distance(obstacle[0], obstacle[1], point)[0]


def path_passes_under(points, obstacle):
    """True if any waypoint sits in the prism directly BELOW the capsule's
    BODY (its interior span, not the end caps). The TCP fitting under a
    human arm proves nothing: the robot's wrist and forearm occupy the
    space above the TCP and would sweep through the person - so under the
    body is forbidden at ANY clearance. Past the fingertips or the elbow
    (the caps) lower ground is ordinary free space."""
    axis_a = np.asarray(obstacle[0], dtype=float)
    axis_b = np.asarray(obstacle[1], dtype=float)
    radius = float(obstacle[2])
    axis = axis_b - axis_a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        axis_u, length = None, 0.0
    else:
        axis_u = axis / length
    for point in points:
        point = np.asarray(point, dtype=float)
        if axis_u is None:
            projection, t = axis_a, 0.5
        else:
            t = float(np.dot(point - axis_a, axis_u)) / length
            if not 0.05 < t < 0.95:
                continue  # beyond the caps: not "under the arm"
            projection = axis_a + axis_u * (t * length)
        horizontal = math.hypot(
            point[0] - projection[0], point[1] - projection[1]
        )
        # DIRECTLY below only: the prism is the arm's own footprint, not
        # the wrap corridor. A point at the hug distance BESIDE a raised
        # arm sits at horizontal offset radius+margin - flagging that as
        # "under" rejected every legal side route and froze the robot the
        # moment the hand went up.
        if horizontal < radius * 0.75 and point[2] < projection[2] - 0.3 * radius:
            return True
    return False


def path_blocked(waypoints, obstacle, margin):
    """First index i where segment waypoints[i]..[i+1] enters the obstacle
    CAPSULE (hand-to-forearm segment, radius) inflated by margin, or None.

    The first waypoint is wherever the robot IS. If the obstacle pops up
    around it (or grows into it), no geometry can make that point clear -
    so the first segment counts as blocked only if it dives DEEPER in. A
    path that immediately climbs away is the best available move and must
    be judged passable, otherwise the planner loops forever."""
    if obstacle is None:
        return None
    axis_a, axis_b, radius = obstacle
    clearance = radius + margin
    start_distance = _capsule_distance(waypoints[0], obstacle)
    for i in range(len(waypoints) - 1):
        distance = _segment_segment_distance(
            waypoints[i], waypoints[i + 1], axis_a, axis_b
        )
        if i == 0 and start_distance < clearance:
            if distance < start_distance - 1.0:
                return 0
            continue
        if distance < clearance:
            return i
    return None


def camera_pose_looking_at(point_mm, tilt_deg, azimuth_rad, standoff_mm):
    """(camera position mm, tool orientation ZYZ deg) viewing `point_mm`.

    The wrist camera looks along the tool's +Z, so aiming it is choosing that
    axis. `tilt_deg` is the angle between the viewing direction and straight
    down: 0 reproduces the top-down shot exactly, larger values slide the
    camera around a cone whose apex is the spot and whose axis is vertical.

    Roll about the viewing axis is not pinned to anything meaningful, so the
    frame is built with its Y horizontal - the image stays level, and the
    wrist spends no travel on a rotation nobody asked for."""
    point = np.asarray(point_mm, dtype=float)
    tilt = math.radians(float(tilt_deg))
    # Unit vector from the spot TOWARDS the camera.
    up = np.array([
        math.sin(tilt) * math.cos(azimuth_rad),
        math.sin(tilt) * math.sin(azimuth_rad),
        math.cos(tilt),
    ])
    position = point + up * float(standoff_mm)
    z_axis = -up                                  # tool +Z looks at the spot
    world_up = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, world_up)
    norm = float(np.linalg.norm(y_axis))
    if norm < 1e-6:                               # straight down: any roll
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis = y_axis / norm
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    euler = Rotation.from_matrix(rotation).as_euler("ZYZ", degrees=True)
    return position, tuple(float(v) for v in euler)


def inspection_pose_is_safe(position_mm, min_z=INSPECT_MIN_CAMERA_Z_MM):
    """Can the arm actually stand there, and stay out of the bench?

    `min_z` differs for the two things being placed: the camera has to see
    over the work, the gripper tips only have to miss it."""
    x, y, z = (float(v) for v in position_mm)
    if not (BOX_X[0] <= x <= BOX_X[1] and BOX_Y[0] <= y <= BOX_Y[1]):
        return False, "outside the safe box"
    if z < min_z:
        return False, "too low over the bench"
    if z > PLAN_Z_MAX:
        return False, "above the controller's safety zone"
    reach = math.sqrt(x * x + y * y + z * z)
    if reach > INSPECT_MAX_REACH_MM:
        return False, f"out of reach ({reach:.0f}mm)"
    if reach < INSPECT_MIN_REACH_MM:
        return False, "too close to the base"
    return True, ""


def iter_inspection_views(point_mm, direction,
                          desired_tilt_deg=INSPECT_TILT_DEG,
                          standoff_mm=INSPECT_STANDOFF_MM):
    """Candidate camera poses, best first, ending at the vertical shot.

    A generator rather than a single answer because geometry alone cannot
    tell whether the ARM can hold a pose: a point inside the reach sphere can
    still need a joint past its limit or a wrist through a singularity. The
    caller walks these and asks the robot's own inverse kinematics, which is
    the only authority on the question."""
    point = np.asarray(point_mm, dtype=float)
    direction = np.asarray(direction, dtype=float)
    horizontal = math.hypot(direction[0], direction[1])
    azimuth = (math.atan2(-direction[1], -direction[0])
               if horizontal > 1e-6 else 0.0)
    finger_tilt = math.degrees(math.atan2(horizontal, max(-direction[2], 1e-6)))
    wanted = min(desired_tilt_deg, finger_tilt, INSPECT_TILT_MAX_DEG)

    tilt = wanted
    while tilt >= 0.0:
        distance = standoff_mm
        while distance >= INSPECT_STANDOFF_MIN_MM:
            position, orientation = camera_pose_looking_at(
                point, tilt, azimuth, distance)
            if inspection_pose_is_safe(position)[0]:
                note = ("as asked" if tilt >= wanted - 1e-6 else
                        f"flattened to {tilt:.0f}deg")
                yield position, orientation, tilt, distance, note
            distance -= 40.0
        tilt -= INSPECT_TILT_STEP_DEG
    # Straight down, the pose this system has always used, at whatever height
    # still reaches.
    height = INSPECT_HEIGHT_MM
    while height >= INSPECT_MIN_CAMERA_Z_MM:
        position = np.array([point[0], point[1], point[2] + height])
        if inspection_pose_is_safe(position)[0]:
            yield position, None, 0.0, height, (
                "vertical" if height >= INSPECT_HEIGHT_MM
                else f"vertical at {height:.0f}mm")
        height -= 25.0


def plan_inspection_view(point_mm, direction, desired_tilt_deg=INSPECT_TILT_DEG,
                         standoff_mm=INSPECT_STANDOFF_MM):
    """Best achievable oblique view of `point_mm`, given the finger's ray.

    Tries the operator's own viewing angle first, then walks the tilt down
    towards vertical - and, at each tilt, pulls the camera in towards the
    spot before giving up on that angle. Every step is still a legal pose,
    so "as oblique as the cell allows" is what comes out rather than a
    refusal. The last resort is the straight-down shot, which is the pose
    this system has always used.

    Returns (position, orientation, tilt_deg, standoff_mm, note)."""
    point = np.asarray(point_mm, dtype=float)
    direction = np.asarray(direction, dtype=float)
    horizontal = math.hypot(direction[0], direction[1])
    # Azimuth of the camera side: back up the finger's own ray.
    azimuth = (math.atan2(-direction[1], -direction[0])
               if horizontal > 1e-6 else 0.0)
    # The finger's own tilt is the most honest starting point: asking for a
    # 45 deg view when the finger is nearly vertical would show a face the
    # operator is not even looking at.
    finger_tilt = math.degrees(math.atan2(horizontal, max(-direction[2], 1e-6)))
    wanted = min(desired_tilt_deg, finger_tilt, INSPECT_TILT_MAX_DEG)

    tilt = wanted
    while tilt >= 0.0:
        distance = standoff_mm
        while distance >= INSPECT_STANDOFF_MIN_MM:
            position, orientation = camera_pose_looking_at(
                point, tilt, azimuth, distance)
            safe, _ = inspection_pose_is_safe(position)
            if safe:
                note = ("as asked" if tilt >= wanted - 1e-6 else
                        f"flattened to {tilt:.0f}deg to stay reachable")
                return position, orientation, tilt, distance, note
            distance -= 40.0
        tilt -= INSPECT_TILT_STEP_DEG
    # Nothing oblique fits: fall back to the proven vertical shot. It gets
    # the same scrutiny - a corner of the bench can be too far for the arm to
    # stand 300 mm above, and lowering the camera shortens the reach.
    # Orientation None means "keep the straight-down pose you already have".
    height = INSPECT_HEIGHT_MM
    while height >= INSPECT_MIN_CAMERA_Z_MM:
        position = np.array([point[0], point[1], point[2] + height])
        safe, _ = inspection_pose_is_safe(position)
        if safe:
            note = ("vertical fallback" if height >= INSPECT_HEIGHT_MM
                    else f"vertical at {height:.0f}mm to stay reachable")
            return position, None, 0.0, height, note
        height -= 25.0
    # The spot is real but the arm cannot stand anywhere that sees it.
    return None, None, 0.0, 0.0, "out of the robot's reach"


def find_free_spot(occupied, human_mm=None, prefer_near=None,
                   clearance_mm=FREE_SPOT_CLEARANCE_MM):
    """An empty patch of bench to put something down on, or None.

    "Anywhere free" still has to satisfy several things at once: clear of the
    tools already lying there, clear of the person, inside the reachable box
    with room for the gripper, and - all else equal - close to where the arm
    already is, because a long traverse across the bench for no reason is
    both slow and alarming to stand next to.

    Scored rather than first-found: the first free cell in a raster scan is
    always in a corner, which is the least useful place to leave a tool."""
    occupied = [np.asarray(p, dtype=float) for p in occupied]
    best = None
    x_lo, x_hi = BOX_X[0] + FREE_SPOT_EDGE_MARGIN_MM, BOX_X[1] - FREE_SPOT_EDGE_MARGIN_MM
    y_lo, y_hi = BOX_Y[0] + FREE_SPOT_EDGE_MARGIN_MM, BOX_Y[1] - FREE_SPOT_EDGE_MARGIN_MM
    x = x_lo
    while x <= x_hi:
        y = y_lo
        while y <= y_hi:
            candidate = np.array([x, y, TABLE_Z_MM])
            reach = math.hypot(x, y)
            if not (INSPECT_MIN_REACH_MM <= reach <= INSPECT_MAX_REACH_MM):
                y += FREE_SPOT_STEP_MM
                continue
            gap = min((float(np.linalg.norm(candidate[:2] - other[:2]))
                       for other in occupied), default=float("inf"))
            if gap < clearance_mm:
                y += FREE_SPOT_STEP_MM
                continue
            if human_mm is not None:
                human_gap = float(np.linalg.norm(
                    candidate[:2] - np.asarray(human_mm, dtype=float)[:2]))
                if human_gap < FREE_SPOT_HUMAN_CLEARANCE_MM:
                    y += FREE_SPOT_STEP_MM
                    continue
            else:
                human_gap = float("inf")
            travel = (0.0 if prefer_near is None else float(np.linalg.norm(
                candidate[:2] - np.asarray(prefer_near, dtype=float)[:2])))
            # Roomy beats tidy, but not at the cost of crossing the bench:
            # clearance is capped so extra emptiness stops paying, and travel
            # then decides between the remaining sensible spots.
            score = min(gap, clearance_mm * 2.0) - 0.35 * travel
            if best is None or score > best[0]:
                best = (score, candidate, gap, human_gap)
            y += FREE_SPOT_STEP_MM
        x += FREE_SPOT_STEP_MM
    if best is None:
        return None, {}
    return best[1], {"clearance_mm": round(best[2], 1),
                     "human_gap_mm": (None if human_mm is None
                                      else round(best[3], 1))}


def _clamp_box(point):
    return np.array([
        np.clip(point[0], *BOX_X),
        np.clip(point[1], *BOX_Y),
        np.clip(point[2], BOX_Z[0], min(BOX_Z[1], PLAN_Z_MAX)),
    ])


def plan_path(start, goal, obstacle):
    """Path from start to goal that hugs its way around the obstacle
    capsule (hand centre to forearm point, radius).

    No obstacle: the straight line, two waypoints, done - the optimal path.

    With an obstacle: the straight line is sampled every PATH_SAMPLE_MM and
    every sample inside the inflated capsule (radius + PLAN_MARGIN_MM, i.e.
    2 cm of clearance) is projected radially out from its axis onto the
    surface. Samples outside are untouched. The result is straight - wraps
    smoothly around the capsule - straight again: the minimum deviation
    that clears it.

    One push MODE for the whole detour, decided up front: radial from the
    axis normally; if any radial push would climb past PLAN_Z_MAX (the
    controller's safety zone), the whole detour wraps HORIZONTALLY at the
    samples' own heights instead. Mixing modes per sample once zigzagged a
    path straight through the obstacle."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if obstacle is None:
        return [start, goal]
    axis_a = np.asarray(obstacle[0], dtype=float)
    axis_b = np.asarray(obstacle[1], dtype=float)
    radius = float(obstacle[2])
    hug_radius = radius + PLAN_MARGIN_MM

    chord = goal - start
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        return [start, goal]
    direction = chord / length
    lateral = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    lateral_norm = float(np.linalg.norm(lateral))
    lateral = (
        lateral / lateral_norm if lateral_norm > 1e-6
        else np.array([1.0, 0.0, 0.0])
    )

    count = max(2, int(math.ceil(length / PATH_SAMPLE_MM)))
    samples = []
    deepest = None
    for i in range(count + 1):
        p = start + chord * (i / count)
        distance, closest = _segment_point_distance(axis_a, axis_b, p)
        samples.append((p, distance, closest))
        if distance < hug_radius and (deepest is None or distance < deepest[1]):
            deepest = (p, distance, closest)
    if deepest is None:
        return [start, goal]  # the line already clears the capsule

    # ONE wrap side for the whole detour (per-sample sides zigzag through
    # the obstacle), and it must NEVER point under the arm: the capsule is
    # a human limb hanging over the table - below it is not free space,
    # however short that detour looks. Build a preference-ordered list of
    # legal sides and take the first whose wrap actually clears.
    axis = axis_b - axis_a
    axis_norm = float(np.linalg.norm(axis))
    axis_u = axis / axis_norm if axis_norm > 1e-6 else None

    def _perpendicular(vector):
        vector = np.asarray(vector, dtype=float)
        if axis_u is not None:
            vector = vector - float(np.dot(vector, axis_u)) * axis_u
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-6 else None

    natural = _perpendicular(deepest[0] - deepest[2])
    if natural is None:
        natural = lateral
    over = _perpendicular(np.array([0.0, 0.0, 1.0]))
    horizontal = None
    if axis_u is not None and abs(axis_u[2]) < 0.99:
        horizontal = _perpendicular(
            np.cross(axis_u, np.array([0.0, 0.0, 1.0]))
        )
    if horizontal is None:
        horizontal = _perpendicular(np.array([natural[0], natural[1], 0.0]))
    if horizontal is None:
        horizontal = lateral
    if float(np.dot(horizontal, natural)) < 0.0:
        horizontal = -horizontal

    candidates = []
    for candidate in (natural, over, horizontal, -horizontal):
        if candidate is None:
            continue
        if candidate[2] < -0.15:
            continue  # under the arm: forbidden
        # No apex pre-filter: an over-the-top wrap whose crown pokes past
        # the ceiling gets flattened AT the ceiling by the builder, and
        # that flattened route often still clears - let the blocked-check
        # judge the result instead of rejecting the attempt.
        if any(float(np.linalg.norm(candidate - c)) < 1e-3 for c in candidates):
            continue
        candidates.append(candidate)
    if not candidates:
        candidates = [horizontal]

    def _wrapped(side):
        wrapped = []
        for p, distance, closest in samples:
            if distance < hug_radius:
                q = closest + side * hug_radius
                q[2] = min(q[2], PLAN_Z_MAX)
                p = _clamp_box(q)
            wrapped.append(p)
        wrapped[0] = start
        wrapped[-1] = goal
        # Elastic-band pass: round the sharp entry/exit corners, then push
        # anything that sagged back inside out to the surface - never to
        # the other side, never into the under-arm hemisphere.
        for _ in range(10):
            for i in range(1, len(wrapped) - 1):
                wrapped[i] = (
                    0.5 * wrapped[i] + 0.25 * (wrapped[i - 1] + wrapped[i + 1])
                )
                distance, closest = _segment_point_distance(
                    axis_a, axis_b, wrapped[i]
                )
                if distance < hug_radius:
                    push = (
                        (wrapped[i] - closest) / distance
                        if distance > 1e-6 else side
                    )
                    if float(np.dot(push, side)) < 0.0 or push[2] < -0.15:
                        push = side
                    q = closest + push * hug_radius
                    q[2] = min(q[2], PLAN_Z_MAX)
                    wrapped[i] = _clamp_box(q)
        return wrapped

    def _around_end(end, out_direction):
        """Route around one END of the capsule via an explicit waypoint,
        for the case no single-side wrap clears: the arm crosses the
        corridor, over is ceiling-blocked, under is forbidden - the only
        passage left is around the fingertips or past the elbow."""
        via = end + out_direction * (hug_radius + 30.0)
        via[2] = min(max(via[2], BOX_Z[0] + 20.0), PLAN_Z_MAX)
        via = _clamp_box(via)
        legs = []
        for leg_start, leg_end in ((start, via), (via, goal)):
            span = leg_end - leg_start
            leg_len = float(np.linalg.norm(span))
            steps = max(2, int(math.ceil(leg_len / PATH_SAMPLE_MM)))
            for i in range(steps):
                legs.append(leg_start + span * (i / steps))
        legs.append(goal.copy())
        # Same elastic smooth + radial reprojection; near an end cap the
        # radial directions rotate smoothly around the tip, so no fixed
        # side is needed - only the under-arm hemisphere stays barred.
        for _ in range(10):
            for i in range(1, len(legs) - 1):
                legs[i] = 0.5 * legs[i] + 0.25 * (legs[i - 1] + legs[i + 1])
                distance, closest = _segment_point_distance(
                    axis_a, axis_b, legs[i]
                )
                if distance < hug_radius:
                    push = (
                        (legs[i] - closest) / distance
                        if distance > 1e-6 else out_direction
                    )
                    if push[2] < -0.15:
                        push = np.array([push[0], push[1], -0.15])
                        push_norm = float(np.linalg.norm(push))
                        push = (
                            push / push_norm if push_norm > 1e-6
                            else out_direction
                        )
                    q = closest + push * hug_radius
                    q[2] = min(q[2], PLAN_Z_MAX)
                    legs[i] = _clamp_box(q)
        return legs

    points = None
    for side in candidates:
        trial = _wrapped(side)
        if (
            path_blocked(trial, obstacle, PLAN_MARGIN_MM - 2.0) is None
            and not path_passes_under(trial, obstacle)
        ):
            points = trial
            break
        if points is None:
            points = trial  # least-bad fallback; the freeze guards backstop
    if (
        path_blocked(points, obstacle, PLAN_MARGIN_MM - 2.0) is not None
        and axis_u is not None
    ):
        end_routes = []
        for end, direction in ((axis_a, -axis_u), (axis_b, axis_u)):
            trial = _around_end(end.copy(), direction.copy())
            if (
                path_blocked(trial, obstacle, PLAN_MARGIN_MM - 2.0) is None
                and not path_passes_under(trial, obstacle)
            ):
                length = sum(
                    float(np.linalg.norm(trial[i + 1] - trial[i]))
                    for i in range(len(trial) - 1)
                )
                end_routes.append((length, trial))
        if end_routes:
            end_routes.sort(key=lambda item: item[0])
            points = end_routes[0][1]
        # If nothing legal cleared, the returned path stays blocked and the
        # caller HOLDS: when the goal cannot be reached without passing
        # through or under the person's arm, not going is the only answer.

    # The first waypoint must be where the robot actually is, even if that
    # is inside the capsule (it appeared around the arm) - path_blocked's
    # first-segment rule handles escaping it.
    points[0] = start
    points[-1] = goal
    pruned = [points[0]]
    for p in points[1:]:
        if float(np.linalg.norm(p - pruned[-1])) > 1.0:
            pruned.append(p)
    if len(pruned) < 2:
        pruned.append(goal)
    return pruned


def path_arclength_lookup(waypoints):
    """(cumulative lengths, total) for carrot interpolation."""
    lengths = [0.0]
    for i in range(len(waypoints) - 1):
        lengths.append(
            lengths[-1]
            + float(np.linalg.norm(np.asarray(waypoints[i + 1]) - waypoints[i]))
        )
    return lengths, lengths[-1]


def project_progress(waypoints, current):
    """(segment index, arc length s) of the current position's projection
    onto the path."""
    best_i, best_s, best_distance = 0, 0.0, float("inf")
    lengths, _ = path_arclength_lookup(waypoints)
    for i in range(len(waypoints) - 1):
        distance, closest = _segment_point_distance(
            waypoints[i], waypoints[i + 1], current
        )
        if distance < best_distance:
            best_distance = distance
            best_i = i
            best_s = lengths[i] + float(
                np.linalg.norm(closest - np.asarray(waypoints[i]))
            )
    return best_i, best_s


def remaining_path(waypoints, current):
    """The path still ahead: current position, then every waypoint past the
    projection. Checking the FULL path instead would flag an obstacle sitting
    on a segment already walked - and replan for something behind the tool."""
    index, _ = project_progress(waypoints, current)
    return [np.asarray(current, dtype=float)] + [
        np.asarray(p, dtype=float) for p in waypoints[index + 1:]
    ]


def carrot_on_path(waypoints, current, lookahead):
    """Point `lookahead` mm beyond the current position's projection onto the
    path. Past the end it is the goal itself."""
    lengths, total = path_arclength_lookup(waypoints)
    _, best_s = project_progress(waypoints, current)
    target_s = min(best_s + lookahead, total)
    for i in range(len(waypoints) - 1):
        if target_s <= lengths[i + 1] or i == len(waypoints) - 2:
            span = max(lengths[i + 1] - lengths[i], 1e-9)
            t = np.clip((target_s - lengths[i]) / span, 0.0, 1.0)
            a = np.asarray(waypoints[i], dtype=float)
            b = np.asarray(waypoints[i + 1], dtype=float)
            return a + t * (b - a)
    return np.asarray(waypoints[-1], dtype=float)


def freewrist_desired_axis(position, velocity, obstacle, goal_distance):
    """Desired tool-Z direction (unit, base frame) while travelling.

    Straight down by default; tilted toward the motion direction and away
    from the obstacle capsule while far from the goal. The whole tilt
    budget scales with goal distance so the aim glides back to vertical on
    final approach instead of snapping there."""
    down = np.array([0.0, 0.0, -1.0])
    ramp = float(np.clip(
        (goal_distance - FREEWRIST_RESTORE_DIST_MM) / FREEWRIST_RESTORE_DIST_MM,
        0.0, 1.0,
    ))
    if ramp <= 0.0:
        return down
    tilt = np.zeros(3)
    speed = float(np.linalg.norm(velocity))
    if speed > 20.0:
        tilt += (
            np.asarray(velocity, dtype=float) / speed
            * min(speed / APPROACH_MAX_SPEED, 1.0)
            * FREEWRIST_MOTION_WEIGHT
        )
    if obstacle is not None:
        distance, closest = _segment_point_distance(
            obstacle[0], obstacle[1], position
        )
        surface = distance - float(obstacle[2])
        if distance > 1e-6 and surface < FREEWRIST_OBSTACLE_RANGE_MM:
            proximity = float(np.clip(
                1.0 - surface / FREEWRIST_OBSTACLE_RANGE_MM, 0.0, 1.0
            ))
            tilt += ((np.asarray(position, dtype=float) - closest) / distance
                     ) * proximity
    magnitude = float(np.linalg.norm(tilt))
    if magnitude < 1e-6:
        return down
    tilt /= magnitude
    axis = np.cross(down, tilt)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return down
    angle = math.radians(FREEWRIST_TILT_MAX_DEG) * ramp * min(magnitude, 1.0)
    return Rotation.from_rotvec(axis / axis_norm * angle).apply(down)


def freewrist_angular_command(tool_axis, desired_axis, filtered, damping):
    """(streamed angular velocity deg/s, new filter state).

    Axis-angle P-control on the two pointing DOF only - roll about the tool
    axis is left wherever the arm has it, exactly like the tracker's
    look-at (a pinned roll spends wrist travel for nothing). `damping`
    fades authority near the J5 singularity."""
    dot = float(np.clip(np.dot(tool_axis, desired_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    raw = np.zeros(3)
    if angle > FREEWRIST_DEADBAND_DEG:
        axis = np.cross(tool_axis, desired_axis)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-9:
            raw = FREEWRIST_KP * angle * (axis / axis_norm)
            speed = float(np.linalg.norm(raw))
            if speed > FREEWRIST_MAX_ANG_SPEED:
                raw *= FREEWRIST_MAX_ANG_SPEED / speed
    filtered = (
        FREEWRIST_SMOOTHING * raw * damping
        + (1.0 - FREEWRIST_SMOOTHING) * np.asarray(filtered, dtype=float)
    )
    command = (
        filtered.copy()
        if float(np.linalg.norm(filtered)) >= FREEWRIST_DEADBAND_DEG
        else np.zeros(3)
    )
    return command, filtered


# ---------------------------------------------------------------------------
class AlignedDepthNode(ImgNode):
    """ImgNode with software depth-to-color registration.

    The realsense-ros align filter (4.58.2) produces nothing on this machine:
    the uvc metadata nodes are absent, frames carry SYSTEM_TIME stamps, and
    the aligned topic stays silent while raw depth flows fine. So the
    registration is done here instead: deproject every raw depth pixel,
    transform depth->color with the driver's own extrinsics, and project into
    the color image. Forward splatting fills roughly half the color pixels
    (848x480 into 1280x720); every consumer samples depth through a patch
    median that tolerates those holes.
    """

    def __init__(self):
        super().__init__()
        self._raw_depth = None
        self._raw_stamp = 0.0
        self._depth_info = None
        self._extrinsics = None
        self._aligned = None
        self._aligned_from = -1.0
        self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw",
            self._raw_depth_callback, 10,
        )
        self.create_subscription(
            CameraInfo, "/camera/camera/depth/camera_info",
            self._depth_info_callback, 10,
        )
        # The extrinsics topic is latched (TRANSIENT_LOCAL); a volatile
        # subscriber that joins after the driver never hears it.
        from rclpy.qos import QoSProfile, DurabilityPolicy
        self.create_subscription(
            Extrinsics, "/camera/camera/extrinsics/depth_to_color",
            self._extrinsics_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    def _raw_depth_callback(self, msg):
        self._raw_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )
        self._raw_stamp = time.monotonic()

    def _depth_info_callback(self, msg):
        self._depth_info = {
            "fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5],
        }

    def _extrinsics_callback(self, msg):
        rotation = np.array(msg.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.array(msg.translation, dtype=np.float64) * 1000.0  # m -> mm
        self._extrinsics = (rotation, translation)

    def get_depth_frame(self):
        """Depth registered into color pixel coordinates (uint16 mm), or the
        cached result if the raw frame hasn't changed."""
        raw = self._raw_depth
        color_info = self.intrinsics
        if raw is None or self._depth_info is None or color_info is None \
                or self._extrinsics is None:
            return None
        if self._aligned is not None and self._aligned_from == self._raw_stamp:
            return self._aligned

        depth = raw.astype(np.float32)
        height, width = depth.shape
        info = self._depth_info
        us, vs = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        valid = depth > 0
        z = depth[valid]
        x = (us[valid] - info["ppx"]) / info["fx"] * z
        y = (vs[valid] - info["ppy"]) / info["fy"] * z
        points = np.vstack([x, y, z])
        rotation, translation = self._extrinsics
        transformed = rotation @ points + translation[:, None]
        zc = transformed[2]
        front = zc > 1.0
        uc = np.rint(
            color_info["fx"] * transformed[0][front] / zc[front] + color_info["ppx"]
        ).astype(np.int32)
        vc = np.rint(
            color_info["fy"] * transformed[1][front] / zc[front] + color_info["ppy"]
        ).astype(np.int32)
        inside = (uc >= 0) & (uc < 1280) & (vc >= 0) & (vc < 720)
        uc, vc = uc[inside], vc[inside]
        zc = zc[front][inside]

        aligned = np.full((720, 1280), np.inf, dtype=np.float32)
        # z-buffer: nearer surface wins where two depth pixels land together.
        np.minimum.at(aligned, (vc, uc), zc)
        aligned[~np.isfinite(aligned)] = 0.0
        self._aligned = aligned.astype(np.uint16)
        self._aligned_from = self._raw_stamp
        return self._aligned


class WebcamPickPlace(Node):
    def __init__(self, target=DEFAULT_TARGET, live=False, auto=False,
                 place_base=None, stop_after_approach=False,
                 fake_obstacle=False, far_start=False, free_wrist=False,
                 hand_place=False, speed_scale=1.0, fancy_hud=True,
                 start_pose=None, home_start=False,
                 grasp_lift=GRASP_LIFT_DEFAULT_MM, hold_start=True,
                 point_select=True, inspect_oblique=True):
        super().__init__("webcam_pick_place")
        # Staged-verification switches. stop_after_approach ends the run at
        # the hover point (no grasp). fake_obstacle injects a synthetic
        # obstacle sphere onto the path mid-approach - exercises the whole
        # replan/detour/straighten loop against the real arm without needing
        # a person to reach into the cell.
        self.stop_after_approach = stop_after_approach
        self.fake_obstacle_enabled = fake_obstacle
        self.fake_obstacle_state = None
        # Where the approach begins. HOME by default: the detour to a far
        # corner was only ever there to lengthen the path for a nicer
        # avoidance demo, and it costs more than it shows - from the far
        # corner the arm crosses the whole bench diagonally to reach anything
        # on the other side, sweeping low over whatever is lying in between.
        # --far-start (or an explicit --start-pose / a P-key pose) opts back
        # in when a long path really is the point.
        explicit = start_pose or self._load_start_pose()
        if home_start:
            self.start_pose = None
        elif explicit:
            self.start_pose = list(explicit)
        elif far_start or fake_obstacle:
            self.start_pose = list(FAR_START_MM)
        else:
            self.start_pose = None
        self.far_start = fake_obstacle or far_start
        # Grasp height trim, live-adjustable with [ and ].
        self.grasp_lift_mm = float(np.clip(grasp_lift, *GRASP_LIFT_RANGE_MM))
        # Two-step keyboard start: S gets ready (and shows the plan updating),
        # G releases the motion. Remote starts run straight through.
        self.point_select = point_select
        # Oblique inspection is opt-outable: the straight-down shot is the
        # proven one, and a demo should be able to fall back to it instantly.
        self.inspect_oblique = inspect_oblique
        # Live distance dial: +/- while looking at the result is faster than
        # arguing about millimetres in the abstract.
        self.inspect_standoff = INSPECT_STANDOFF_MM
        # Vertical shots have their own distance dial: the standoff above is
        # measured along a tilted line of sight and means nothing when the
        # wrist is square to the bench.
        self.inspect_height = INSPECT_HEIGHT_MM
        self.hold_default = hold_start
        self.hold_before_approach = False
        self.target_moved_at = 0.0
        self.target_moved_mm = 0.0
        self.target_stale_since = None
        # 6-axis avoidance: stream angular velocity during APPROACH so the
        # tool tilts with the motion / away from the obstacle. Opt-in; with
        # it off nothing about the proven behaviour changes.
        self.free_wrist = free_wrist
        self.freewrist_filtered = np.zeros(3)
        self.measured_joints = None
        # Deliver-to-hand: place the object on the operator's tracked hand
        # instead of at a clicked point.
        self.hand_place = hand_place
        # Launch-time defaults; a remote start command may override the
        # destination for ONE run (e.g. tidy-away to storage), after which
        # these are restored.
        self.default_hand_place = hand_place
        self.default_place_base = place_base
        # Where "정리해줘" puts things: the proven place spot from the E2E
        # runs. Override with --storage-base.
        self.storage_base = [260.0, -230.0, 0.0]
        # Demo dial: scales every commanded speed (streamed, movel, movej,
        # wrist rotation) uniformly. 1.0 = the tuned behaviour.
        self.speed_scale = float(np.clip(speed_scale, 0.1, 1.5))
        self.deliver_started_at = 0.0
        self.deliver_dwell_since = None
        self.deliver_palm_history = deque()  # (t, palm_mm) for steadiness
        self.deliver_commit_palm = None      # snapshot the release drops onto
        self.lock = threading.RLock()
        self.robot_lock = threading.RLock()
        self.dry_run = not live
        self.auto = auto
        self.place_base = place_base  # preset place position (mm) or None
        self.running = True
        self.busy = False
        self.abort_requested = False
        self.state = State.IDLE
        self.status = "Press S to start"
        self.target_name = target

        # Webcam-side results (written by the webcam thread)
        self.webcam_lock = threading.Lock()
        self.webcam_dets = {}          # cam -> {"box","score","stamp"}
        self.webcam_all_dets = {}      # cam -> {class: {"box","score","stamp"}}
        # Pointing: the operator's index finger chooses which class to fetch.
        self.pointing_ray = None       # (origin_mm, unit direction)
        self.pointing_end = None       # where the beam actually stops (mm)
        self.pointing_choice = None    # class the finger settled on
        self.pointing_scores = {}      # class -> angle/distance, for the HUD
        self.pointing_ambiguous = False
        self.point_smoother = pointing.PointingSmoother()
        self.pointing_spot = None      # where the ray meets the aim plane (mm)
        self.aim_plane_mm = TABLE_Z_MM  # height of that plane; , and . move it
        # The chosen spot OUTLIVES the gesture. A person points, lowers their
        # hand and then asks the question - and the arm has to wait for that
        # hand to leave before it can photograph anything anyway. Keeping the
        # mark on screen also makes the choice something you can check before
        # committing, instead of a thing that vanished.
        self.locked_spot = None
        self.locked_direction = None
        self.locked_ray = None         # (origin_mm, direction) for drawing
        self.locked_at = 0.0
        # Pressing I commits the mark. While the arm is on its way the hand
        # is moving OUT of the shot, and MediaPipe happily reads that motion
        # as a new gesture - re-aiming the mark to somewhere the robot is not
        # going. The display would then be lying at the exact moment it
        # matters most.
        self.point_frozen = False
        # What was last photographed. The MARK is cleared from the screen the
        # moment the shot is taken - a stale pin invites you to think it is
        # still selected - but the coordinates survive a little longer, so a
        # follow-up ("조금 더 가까이서 다시") has something to aim at after
        # the operator has already lowered their hand.
        self.last_inspect_point = None
        self.last_inspect_direction = None
        self.last_inspect_at = 0.0
        # Follow-up channel: while holding at the viewing pose, another
        # "inspect" (or "inspect_done") from the copilot lands here instead
        # of being refused for being busy.
        self._inspect_followup = None
        self._inspect_signal = threading.Event()
        self.last_inspection = None    # {"path","point","at"} for the LLM
        self._point_switch_at = 0.0
        self.webcam_overlay = {}       # cam -> list of blob (u,v,r_px)
        self.obstacle = None           # {"kf","radius_mm","last_seen","cams"}
        self.intruder_watch = False

        # Pipeline data
        self.detect_fixes = deque(maxlen=DETECT_SAMPLES)
        self.detect_started_at = 0.0
        self.target_base = None        # triangulated object position, mm
        self.hover_pose = None         # [x, y, z] mm
        self.grasp_orientation = None  # [rx, ry, rz] captured at home
        self.path = None               # list of np.array mm
        self.path_version = 0
        self.last_replan_at = 0.0
        self.approach_started_at = 0.0
        self.last_send_at = 0.0
        self.pose_read_failures = 0
        self.pick = None
        self.place = None
        self.frozen_for_obstacle = False
        self._d435_overlay = []
        self._d435_overlay_at = 0.0
        self.last_cmd_velocity = np.zeros(3)
        self.progress_anchor = None  # (time, position) for stuck detection
        self._render_posx = None
        self._render_posx_at = 0.0

        # --- overlay ---
        # One canvas per quadrant, allocated once: the buffers are the size of
        # the view and reallocating them at frame rate would show up in the
        # control loop, which shares this thread.
        self._hud = {
            name: ar_hud.HudCanvas(QUAD_W, QUAD_H, fancy=fancy_hud)
            for name in ("cam0", "cam1", "cam2", "d435")
        }
        self._chrome = ar_hud.HudCanvas(QUAD_W * 2, QUAD_H * 2,
                                        fancy=fancy_hud, bloom=False)
        self._hud_ms = 0.0
        self.fancy_hud = fancy_hud
        self.show_hud_cost = fancy_hud

        # --- hardware ---
        self.rig = WebcamRig()
        self.rig.start()
        self.get_logger().info("Webcam rig started")
        # Per-camera crop of the projected workspace (x1, y1, x2, y2, scale).
        self.webcam_crops = {}
        for name, camera in self.rig.cameras.items():
            points = [
                p for p in camera.project(WORKSPACE_CORNERS_M) if p is not None
            ]
            if len(points) < 4:
                self.webcam_crops[name] = (0, 0, 1280, 720, 1.0)
                continue
            us = [p[0] for p in points]
            vs = [p[1] for p in points]
            x1 = int(max(0, min(us) - WEBCAM_CROP_MARGIN_PX))
            x2 = int(min(1280, max(us) + WEBCAM_CROP_MARGIN_PX))
            y1 = int(max(0, min(vs) - WEBCAM_CROP_MARGIN_PX))
            y2 = int(min(720, max(vs) + WEBCAM_CROP_MARGIN_PX))
            scale = min(WEBCAM_CROP_MAX_SCALE, 1280.0 / max(1, x2 - x1))
            self.webcam_crops[name] = (x1, y1, x2, y2, scale)

        self.img_node = AlignedDepthNode()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )
        self.speedl_pub = self.create_publisher(
            SpeedlStream, f"/{ROBOT_ID}/speedl_stream", 10
        )
        self.robot_control_client = self.create_client(
            SetRobotControl, f"/{ROBOT_ID}/system/set_robot_control"
        )
        # J5 is watched for the wrist singularity guard; take it from the
        # state stream rather than paying for a DRFL round trip.
        self.create_subscription(
            JointStateMsg, f"/{ROBOT_ID}/joint_states", self._joint_state_cb, 10
        )
        # Remote command channel: an LLM chat (or anything else) can press
        # the same buttons the operator does. "start" == the S key,
        # "stop" == SPACE.
        self.create_subscription(
            StringMsg, "/webcam_pnp/command", self._command_cb, 10
        )
        self.status_pub = self.create_publisher(
            StringMsg, "/webcam_pnp/status", 10
        )
        # Machine-readable mirror of the status line for the skill bridge.
        # Inspection results go out on their own topic: the copilot needs the
        # image path, and stuffing a filename into the status line would make
        # every consumer parse prose.
        self.inspection_pub = self.create_publisher(
            StringMsg, "/webcam_pnp/inspection", 10)
        self.state_json_pub = self.create_publisher(
            StringMsg, "/webcam_pnp/state_json", 10
        )
        self.active_request_id = None

        self.camera_executor = SingleThreadedExecutor()
        self.camera_executor.add_node(self.img_node)
        self.camera_executor.add_node(self)
        self.camera_thread = threading.Thread(
            target=self.camera_executor.spin, daemon=True
        )
        self.camera_thread.start()

        self.intrinsics = self._wait_for(
            self.img_node.get_camera_intrinsic, "D435i intrinsics"
        )
        self._wait_for(self.img_node.get_color_frame, "D435i color")
        self._wait_for(self.img_node.get_depth_frame, "D435i depth")

        self.gripper2cam = np.load(resolve_calibration_path())
        self.flange2tcp = self._calibrate_flange_offset()
        self.flange2depth = load_wrist_depth_calibration()
        self.flange2color = None  # built once the driver extrinsics arrive
        if self.flange2depth is None or self.flange2tcp is None:
            self.get_logger().warn(
                "wrist-depth calibration unavailable - falling back to the "
                "tutorial T_gripper2camera (expect ~28mm of z bias)"
            )

        self.yolo = YoloModel()
        self.get_logger().info(f"YOLO classes: {', '.join(self.yolo.class_names)}")
        if target not in self.yolo.class_names:
            raise RuntimeError(f"target {target!r} not in YOLO classes")

        # Hands are the only thing that counts as an obstacle (MediaPipe).
        self.hand_detector = HandIntruderDetector(list(self.rig.cameras))
        self.get_logger().info("Hand intruder detector ready")

        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("Gripper connected")
        except Exception as error:
            self.gripper = None
            self.get_logger().error(f"Gripper connect failed: {error}")

        self.webcam_thread = threading.Thread(
            target=self._webcam_loop, daemon=True
        )
        self.webcam_thread.start()

    def _wait_for(self, getter, what, timeout=30.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            value = getter()
            if value is not None and not (
                isinstance(value, np.ndarray) and not value.any()
            ):
                return value
            time.sleep(0.2)
        raise RuntimeError(f"Timed out waiting for {what}")

    # ------------------------------------------------------------------
    # Robot pose helpers
    # ------------------------------------------------------------------
    @staticmethod
    def pose_matrix(x, y, z, rx, ry, rz):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "ZYZ", [rx, ry, rz], degrees=True
        ).as_matrix()
        transform[:3, 3] = [x, y, z]
        return transform

    def _safe_posx(self):
        if not self.robot_lock.acquire(blocking=False):
            return None
        try:
            data = get_current_posx()
            if not data or not data[0] or len(data[0]) < 6:
                return None
            return list(data[0])
        except Exception as error:
            self.get_logger().warn(f"get_current_posx failed: {error}")
            return None
        finally:
            self.robot_lock.release()

    def tf_base2flange(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, FLANGE_FRAME, rclpy.time.Time()
            ).transform
        except Exception:
            return None
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([
            transform.rotation.x, transform.rotation.y,
            transform.rotation.z, transform.rotation.w,
        ]).as_matrix()
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
            self.get_logger().warn("No TF base->flange; using identity offset")
            return None
        try:
            base2tcp = self.pose_matrix(*get_current_posx()[0])
        except Exception as error:
            self.get_logger().warn(f"flange/TCP offset failed: {error}")
            return None
        return np.linalg.inv(base2flange) @ base2tcp

    def _arm_points_m(self):
        """Arm skeleton in base frame, METERS, for webcam masking: base
        column, every link origin, and the TCP."""
        points = [np.array([0.0, 0.0, 0.05])]
        for frame in ARM_TF_FRAMES:
            try:
                transform = self.tf_buffer.lookup_transform(
                    BASE_FRAME, frame, rclpy.time.Time()
                ).transform
            except Exception:
                return None
            points.append(np.array([
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ]))
        base2flange = self.tf_base2flange()
        if base2flange is not None and self.flange2tcp is not None:
            tcp = (base2flange @ self.flange2tcp)[:3, 3] / 1000.0
            points.append(tcp)
        return points

    # ------------------------------------------------------------------
    # Webcam processing loop (own thread, ~8 Hz)
    # ------------------------------------------------------------------
    def _webcam_loop(self):
        while self.running and rclpy.ok():
            started = time.monotonic()
            frames = self.rig.frames()
            if len(frames) < 2:
                time.sleep(0.2)
                continue
            names = list(frames)
            images = []
            for name in names:
                x1, y1, x2, y2, scale = self.webcam_crops[name]
                crop = frames[name][0][y1:y2, x1:x2]
                if scale > 1.01:
                    crop = cv2.resize(
                        crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                images.append(crop)
            try:
                results = self.yolo.model(
                    images, verbose=False, conf=WEBCAM_CONF, imgsz=WEBCAM_IMGSZ
                )
            except Exception as error:
                self.get_logger().error(f"webcam YOLO failed: {error}")
                time.sleep(1.0)
                continue

            detections = {}
            everything = {}
            for name, result in zip(names, results):
                x1, y1, _, _, scale = self.webcam_crops[name]
                best = None
                per_class = {}
                for box, score, label in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                ):
                    # Crop coordinates back to the full frame.
                    full = [
                        box[0] / scale + x1, box[1] / scale + y1,
                        box[2] / scale + x1, box[3] / scale + y1,
                    ]
                    class_name = result.names[int(label)]
                    # Every class is kept now, not just the target: pointing
                    # at a tool can only select it if the tool was seen, and
                    # the overlay can only show what the operator could
                    # switch to if it knows where those things are.
                    previous = per_class.get(class_name)
                    if previous is None or score > previous["score"]:
                        per_class[class_name] = {"box": full,
                                                 "score": float(score)}
                    if class_name != self.target_name:
                        continue
                    if best is None or score > best["score"]:
                        best = {"box": full, "score": float(score)}
                if per_class:
                    stamp = time.monotonic()
                    for entry in per_class.values():
                        entry["stamp"] = stamp
                    everything[name] = per_class
                if best is not None:
                    best["stamp"] = time.monotonic()
                    detections[name] = best

            overlay = {}
            # Hands are needed for two different jobs now: as the obstacle to
            # dodge while moving, and as the finger that chooses a target
            # while standing still. Run the detector if either wants it.
            want_hands = self.intruder_watch or self.point_select
            intruders = {}
            if want_hands:
                intruders = self.hand_detector.detect(
                    {name: frames[name][0] for name in names}
                )
                for cam, data in intruders.items():
                    overlay[cam] = [
                        (blob[0], blob[1], blob[3]) for blob in data["blobs"]
                    ]
            if self.intruder_watch:
                self._update_obstacle(self.rig.locate_obstacle(intruders))
            else:
                with self.webcam_lock:
                    self.obstacle = None

            with self.webcam_lock:
                self.webcam_dets = detections
                self.webcam_all_dets = everything
                self.webcam_overlay = overlay
            if self.point_select:
                self._update_pointing(intruders, everything)

            elapsed = time.monotonic() - started
            # ~15 Hz target: obstacle latency feeds straight into how fresh
            # the capsule the planner dodges is.
            time.sleep(max(0.0, 0.065 - elapsed))

    def locate_all_objects(self, everything=None,
                           max_z=DETECT_TARGET_Z_MM[1]):
        """{class: position_mm} for every class two webcams agree on.

        The same triangulation the pick target goes through, applied to
        everything in view - which is what makes "point at that one" and
        "put it down somewhere free" possible: both need to know where the
        things on the bench actually are, not just the one being fetched."""
        if everything is None:
            with self.webcam_lock:
                everything = {cam: dict(per)
                              for cam, per in self.webcam_all_dets.items()}
        located = {}
        for class_name in {c for per in everything.values() for c in per}:
            picks = {}
            for cam, per_class in everything.items():
                if class_name not in per_class:
                    continue
                box = per_class[class_name]["box"]
                picks[cam] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            if len(picks) < 2:
                continue
            point, gap = self.rig.triangulate(picks)
            if point is None or gap is None or gap > DETECT_MAX_RAY_GAP_M:
                continue
            position = np.asarray(point) * 1000.0
            if not DETECT_TARGET_Z_MM[0] <= position[2] <= max_z:
                continue                    # out of the volume we care about
            located[class_name] = position
        return located

    def _update_pointing(self, intruders, everything):
        """Let the operator choose the target by pointing at it.

        Selection only - the finger never starts a motion, and it cannot
        change anything once one is under way (cycle_target refuses outside
        the standing-still states). Deciding what to fetch by pointing and
        deciding to fetch are deliberately two acts."""
        pointers = {cam: data["pointers"][0]
                    for cam, data in intruders.items() if data.get("pointers")}
        located = self.locate_all_objects(everything,
                                          max_z=POINT_TARGET_Z_MAX_MM)

        ray = pointing.pointer_ray(self.rig, pointers)
        chosen, info = (None, {})
        if ray is not None and located:
            chosen, info = pointing.select_pointed(located, ray[0], ray[1])
        now = time.monotonic()
        stable = self.point_smoother.update(chosen, now)

        spot = None
        end = None
        if ray is not None:
            spot = pointing.ray_plane_point(ray[0], ray[1],
                                            plane_z_mm=self.aim_plane_mm)
            if spot is not None:
                spot = _clamp_box(np.array([spot[0], spot[1],
                                            max(spot[2], BOX_Z[0])]))
                spot[2] = self.aim_plane_mm
                # A ray of light stops at the first thing it hits. The table
                # is the backstop; a tool standing between the finger and the
                # table is nearer, and drawing straight through it is what
                # made the beam look like an overlay instead of a pointer.
                end = spot
                origin, unit = ray
                reach = float(np.dot(spot - origin, unit))
                if stable is not None and stable in located:
                    hit = float(np.dot(located[stable] - origin, unit))
                    if 0.0 < hit < reach:
                        end = origin + unit * hit
                        # Pointing AT a thing means the thing, not the floor
                        # behind it. Snap the mark onto the object the finger
                        # settled on - the only case where the plane
                        # intersection was ever the right answer is an empty
                        # patch of bench, which is exactly when there is no
                        # object to snap to.
                        spot = _clamp_box(np.asarray(located[stable],
                                                     dtype=float).copy())
        if spot is not None and ray is not None and not self.point_frozen:
            # Re-aiming replaces the mark; a hand drifting a centimetre while
            # it is lowered must not.
            previous = self.locked_spot
            if (previous is None
                    or float(np.linalg.norm(spot - previous)) > 15.0):
                self.locked_spot = np.asarray(spot, dtype=float).copy()
                self.locked_direction = np.asarray(ray[1], dtype=float).copy()
                self.locked_ray = (np.asarray(ray[0], dtype=float).copy(),
                                   np.asarray(ray[1], dtype=float).copy(),
                                   np.asarray(end if end is not None else spot,
                                              dtype=float).copy())
                self.locked_at = now
        with self.webcam_lock:
            self.pointing_spot = spot
            self.pointing_end = end
            self.pointing_ray = ray
            self.pointing_choice = stable
            self.pointing_scores = info.get("scores", {})
            self.pointing_ambiguous = bool(info.get("ambiguous"))
        if (stable is not None and stable != self.target_name
                and now - self._point_switch_at > 0.6):
            self._point_switch_at = now
            # quiet: a finger wandering past during a run should not spam the
            # status line with "target locked".
            self.set_target(stable, quiet=True)

    def _update_obstacle(self, located):
        now = time.monotonic()
        with self.webcam_lock:
            if located is None:
                if (
                    self.obstacle is not None
                    and now - self.obstacle["last_seen"] > OBSTACLE_HOLD_SEC
                ):
                    self.obstacle = None
                return
            center_m, forearm_m, radius_m, cams = located
            center_mm = np.asarray(center_m, dtype=float) * 1000.0
            forearm_offset = (
                np.asarray(forearm_m, dtype=float) * 1000.0 - center_mm
            )
            margin = OBSTACLE_WORKSPACE_MARGIN_MM
            if not (
                BOX_X[0] - margin <= center_mm[0] <= BOX_X[1] + margin
                and BOX_Y[0] - margin <= center_mm[1] <= BOX_Y[1] + margin
                and center_mm[2] <= BOX_Z[1] + margin
            ):
                return
            if self.obstacle is None:
                self.obstacle = {
                    # q_acc high on purpose: a hand accelerates hard, and a
                    # sluggish filter reports where it WAS - the arm then
                    # dodges a ghost trailing the real hand.
                    "kf": KalmanCV(center_mm, q_acc=400.0, r_pos=15.0),
                    "forearm_offset": forearm_offset,
                    "radius_mm": radius_m * 1000.0,
                    "last_seen": now,
                    "last_predict": now,
                    "cams": cams,
                }
                return
            kf = self.obstacle["kf"]
            # Predict FIRST: the gate must compare against where a moving
            # hand is expected to be, not where it last was.
            kf.predict(max(1e-3, now - self.obstacle["last_predict"]))
            self.obstacle["last_predict"] = now
            jump = float(np.linalg.norm(center_mm - kf.pos))
            if (
                jump > OBSTACLE_JUMP_GATE_MM
                and now - self.obstacle["last_seen"] < 0.7
            ):
                self.obstacle["rejects"] = self.obstacle.get("rejects", 0) + 1
                if self.obstacle["rejects"] < OBSTACLE_JUMP_HANDOVER_COUNT:
                    return  # hold the current track; ignore the other hand
                # The far position keeps winning: it IS the hand now.
                self.obstacle = {
                    "kf": KalmanCV(center_mm, q_acc=400.0, r_pos=15.0),
                    "forearm_offset": forearm_offset,
                    "radius_mm": radius_m * 1000.0,
                    "last_seen": now,
                    "last_predict": now,
                    "cams": cams,
                }
                return
            self.obstacle["rejects"] = 0
            kf.update(center_mm)
            # Radius and forearm direction ride EMAs: one noisy frame must
            # not balloon the capsule or whip its axis around.
            self.obstacle["radius_mm"] = (
                0.3 * radius_m * 1000.0 + 0.7 * self.obstacle["radius_mm"]
            )
            self.obstacle["forearm_offset"] = (
                0.4 * forearm_offset + 0.6 * self.obstacle["forearm_offset"]
            )
            self.obstacle["last_seen"] = now
            self.obstacle["cams"] = cams

    def hand_target(self):
        """(palm_center_mm, palm_speed_mm_s) of the tracked hand, or None.
        Same MediaPipe capsule that plays obstacle on the way in - during
        delivery its centre is the goal instead."""
        with self.webcam_lock:
            if self.obstacle is None:
                return None
            return (
                self.obstacle["kf"].pos.copy(),
                float(np.linalg.norm(self.obstacle["kf"].vel)),
            )

    def current_obstacle(self):
        """Capsule (axis_end_a_mm, axis_end_b_mm, radius_mm) or None."""
        fake = self._fake_obstacle()
        if fake is not None:
            return fake
        with self.webcam_lock:
            if self.obstacle is None:
                return None
            center = self.obstacle["kf"].pos.copy()
            return (
                center,
                center + self.obstacle["forearm_offset"],
                float(self.obstacle["radius_mm"]),
            )

    def _fake_obstacle(self):
        """Synthetic obstacle for staged live verification: appears on the
        path midpoint 3 s into the approach, sits there for 6 s, vanishes.
        Runs through exactly the same replan machinery as a real one."""
        if not self.fake_obstacle_enabled or self.state != State.APPROACH:
            return None
        now = time.monotonic()
        if self.fake_obstacle_state is None:
            if now - self.approach_started_at < 0.5 or self.path is None:
                return None
            start = np.asarray(self.path[0], dtype=float)
            goal = np.asarray(self.path[-1], dtype=float)
            center = (start + goal) / 2.0
            self.fake_obstacle_state = {"center": center, "until": now + 5.0}
            self.get_logger().warn(
                f"FAKE obstacle up at ({center[0]:.0f},{center[1]:.0f},"
                f"{center[2]:.0f}) r=80 for 5s"
            )
        if now > self.fake_obstacle_state["until"]:
            return None
        center = self.fake_obstacle_state["center"]
        return center.copy(), center.copy(), 80.0

    def _command_cb(self, msg):
        """Remote channel pressing the operator's own buttons: an LLM chat
        or the copilot's skill bridge publishes "start" (== the S key) or
        "stop" (== SPACE). JSON form carries a request id to echo back."""
        raw = msg.data.strip()
        request_id = None
        destination = None
        payload = {}
        if raw.startswith("{"):
            try:
                import json as _json
                payload = _json.loads(raw)
                command = str(payload.get("command", "")).lower()
                request_id = payload.get("request_id")
                destination = payload.get("destination")
            except Exception:
                command = ""
        else:
            command = raw.lower()
        self.get_logger().info(
            f"remote command: {command!r} ({request_id}, dest={destination})"
        )
        if command == "start":
            # Per-run destination override, restored to launch defaults each
            # start: "storage" fetches then places at the storage spot
            # instead of the operator's hand.
            self.hand_place = self.default_hand_place
            self.place_base = self.default_place_base
            if destination == "storage":
                self.hand_place = False
                self.place_base = list(self.storage_base)
            elif destination == "hand":
                self.hand_place = True
            self.active_request_id = request_id
            self.abort_requested = False
            # Voice/UI requests run end to end: there is nobody at the window
            # to press G, and JARVIS already asked for confirmation.
            if self.state == State.ARMED:
                # Already parked with a live plan because someone pressed S.
                # The spoken request IS the release - refusing it here is how
                # "가져와" silently does nothing while the arm sits armed.
                self.hold_before_approach = False
                self.handle_go()
            elif self.state == State.DETECT:
                # Mid-search: adopt the request rather than reject it, so the
                # search that is already running finishes into the approach.
                self.hold_before_approach = False
                self.status = f"Looking for {self.target_name}..."
            else:
                self.handle_start(hold=False)
        elif command == "inspect" and self.state == State.INSPECT:
            # A second look while we are still parked at the first one.
            self._inspect_followup = {
                "request_id": request_id,
                "point": payload.get("point"),
                "standoff_mm": payload.get("standoff_mm"),
            }
            self._inspect_signal.set()
        elif command == "take_from_hand":
            self.handle_take_from_hand(request_id)
        elif command == "inspect_done":
            # The verdict is in and needs no more pictures - stop holding.
            self._inspect_followup = None
            self._inspect_signal.set()
        elif command == "inspect":
            # "Am I doing this right?" - photograph whatever the operator is
            # pointing at. The point may also be given explicitly so a caller
            # that already knows the spot does not need a finger.
            self.active_request_id = request_id
            point = payload.get("point") if raw.startswith("{") else None
            if not self.handle_inspect(
                    np.asarray(point, dtype=float) if point else None,
                    request_id=request_id,
                    standoff_mm=payload.get("standoff_mm")):
                self._publish_inspection(None, request_id,
                                         error=self.status)
        elif command == "stop":
            self.handle_abort()
        elif command == "home":
            self.abort_requested = False
            self.handle_home()
        elif command in ("open", "gripper_open"):
            self.handle_gripper(close=False)
        elif command in ("close", "gripper_close"):
            self.handle_gripper(close=True)

    def _publish_inspection(self, record, request_id, error=None):
        """Tell whoever asked where the picture landed (or why it did not)."""
        message = {
            "request_id": request_id,
            "ok": record is not None,
            "at": time.time(),
        }
        if record is not None:
            message.update(record)
        if error:
            message["error"] = error
        try:
            out = StringMsg()
            out.data = json.dumps(message)
            self.inspection_pub.publish(out)
        except Exception as failure:
            self.get_logger().warn(f"inspection publish failed: {failure}")

    def publish_status(self, snapshot):
        try:
            self.status_pub.publish(StringMsg(data=snapshot))
            import json as _json
            self.state_json_pub.publish(StringMsg(data=_json.dumps({
                "state": self.state.value,
                "status": self.status,
                "request_id": self.active_request_id,
                "delivered": "Delivered" in self.status,
            }, ensure_ascii=False)))
        except Exception:
            pass

    def _joint_state_cb(self, message):
        try:
            index = {name: i for i, name in enumerate(message.name)}
            self.measured_joints = np.array(
                [message.position[index[f"joint_{i}"]] for i in range(1, 7)]
            )
        except (KeyError, IndexError):
            pass

    def _wrist_damping(self):
        """1.0 with J5 far from its singularity at 0, fading to 0 at the
        floor - fading rather than cutting, a hard stop on the angular
        command is itself a step input."""
        if self.measured_joints is None:
            return 1.0
        margin = abs(float(np.degrees(self.measured_joints[4])))
        if margin >= WRIST_SAFE_DEG:
            return 1.0
        if margin <= WRIST_FLOOR_DEG:
            return 0.0
        return (margin - WRIST_FLOOR_DEG) / (WRIST_SAFE_DEG - WRIST_FLOOR_DEG)

    # ------------------------------------------------------------------
    # SpeedL streaming
    # ------------------------------------------------------------------
    def _send_velocity(self, linear, angular=None):
        rotation = np.zeros(3) if angular is None else np.asarray(angular)
        msg = SpeedlStream()
        msg.vel = [float(linear[0]), float(linear[1]), float(linear[2]),
                   float(rotation[0]), float(rotation[1]), float(rotation[2])]
        msg.acc = [APPROACH_ACC[0], APPROACH_ACC[1]]
        msg.time = COMMAND_TTL_SEC
        self.speedl_pub.publish(msg)

    def _send_stop(self):
        if not self.dry_run:
            self._send_velocity([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # State entry points
    # ------------------------------------------------------------------
    def handle_start(self, hold=None):
        """`hold` parks at the demo start until G. Remote (JARVIS) starts
        never hold - nobody is at the keyboard to release them."""
        if self.busy or self.state not in (State.IDLE, State.ERROR):
            self.status = f"START unavailable in {self.state.value}"
            return
        self.hold_before_approach = (
            self.hold_default if hold is None else bool(hold))
        self.clear_target_mark()
        self.abort_requested = False
        self.pick = None
        self.place = None
        self.target_base = None
        self.path = None
        self.detect_fixes.clear()
        self.state = State.HOMING
        self.status = "Moving HOME..."
        self.start_job(self._job_home_then_detect)

    def _job_home_then_detect(self):
        self.go_joint("home", HOME_JOINT)
        time.sleep(0.3)
        current = self._safe_posx()
        if current is None:
            raise RuntimeError("cannot read pose at home")
        self.grasp_orientation = list(current[3:])
        if self.start_pose is not None:
            # Begin the approach away from home so the path is long enough for
            # a detour around a hand to be visible and meaningful.
            self.go_linear(
                "demo start",
                np.asarray(self.start_pose, dtype=float),
                tuple(self.grasp_orientation),
            )
            time.sleep(0.3)
        self.detect_started_at = time.monotonic()
        self.detect_fixes.clear()
        self.state = State.DETECT
        self.status = f"Looking for {self.target_name} on the webcams..."

    def _webcam_fix(self, now):
        """One triangulated target fix, or (None, why).

        Shared by DETECT and ARMED: the armed hold has to re-run exactly the
        same gates, or the plan it shows would be built from a target the
        approach would never accept."""
        with self.webcam_lock:
            detections = dict(self.webcam_dets)
        fresh = {
            cam: det for cam, det in detections.items()
            if now - det["stamp"] < 0.6
        }
        if len(fresh) < 2:
            return None, (f"Waiting for {self.target_name} in 2+ webcams "
                          f"(seen in {len(fresh)})")
        pixels = {}
        for cam, det in fresh.items():
            x1, y1, x2, y2 = det["box"]
            pixels[cam] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        point_m, gap = self.rig.triangulate(pixels)
        if point_m is None or gap > DETECT_MAX_RAY_GAP_M:
            return None, (f"Rays disagree "
                          f"({0 if gap is None else gap * 1000:.0f}mm)")
        fix_mm = np.asarray(point_m) * 1000.0
        margin = DETECT_TARGET_XY_MARGIN_MM
        if not (
            BOX_X[0] - margin <= fix_mm[0] <= BOX_X[1] + margin
            and BOX_Y[0] - margin <= fix_mm[1] <= BOX_Y[1] + margin
            and DETECT_TARGET_Z_MM[0] <= fix_mm[2] <= DETECT_TARGET_Z_MM[1]
        ):
            return None, (
                f"Rejected off-table fix ({fix_mm[0]:.0f},{fix_mm[1]:.0f},"
                f"{fix_mm[2]:.0f}) - not the {self.target_name} on the table"
            )
        return fix_mm, f"gap {gap * 1000:.0f}mm"

    @staticmethod
    def _hover_for(target_mm):
        return _clamp_box(np.array([
            target_mm[0], target_mm[1], target_mm[2] + HOVER_STANDOFF_MM
        ]))

    def detect_tick(self):
        now = time.monotonic()
        if now - self.detect_started_at > DETECT_TIMEOUT_SEC:
            self.state = State.ERROR
            self.status = (
                f"No {self.target_name} triangulated in {DETECT_TIMEOUT_SEC:.0f}s"
                " - is it visible to 2+ webcams? Press S to retry"
            )
            return
        fix_mm, why = self._webcam_fix(now)
        if fix_mm is None:
            self.status = why
            return
        gap = why
        self.detect_fixes.append(fix_mm)
        self.status = (
            f"Triangulating {self.target_name}: "
            f"{len(self.detect_fixes)}/{DETECT_SAMPLES} fixes, {gap}"
        )
        if len(self.detect_fixes) < DETECT_SAMPLES:
            return
        fixes = np.array(self.detect_fixes)
        median = np.median(fixes, axis=0)
        spread = float(np.max(np.linalg.norm(fixes - median, axis=1)))
        if spread > DETECT_SPREAD_MM:
            self.detect_fixes.popleft()
            self.status = f"Fixes unstable (spread {spread:.0f}mm)"
            return

        self.target_base = median
        hover = self._hover_for(median)
        self.hover_pose = hover
        current = self._safe_posx()
        if current is None:
            self.state = State.ERROR
            self.status = "Robot pose unavailable"
            return
        self.path = plan_path(
            np.asarray(current[:3]), hover, self.current_obstacle()
        )
        self.path_version += 1
        self.intruder_watch = True
        self.approach_started_at = now
        self.last_cmd_velocity = np.zeros(3)
        self.progress_anchor = None
        self.fake_obstacle_state = None
        self.freewrist_filtered = np.zeros(3)
        self.get_logger().info(
            f"target {self.target_name} at base ({median[0]:.1f}, "
            f"{median[1]:.1f}, {median[2]:.1f})mm, hover at "
            f"({hover[0]:.1f}, {hover[1]:.1f}, {hover[2]:.1f})"
        )
        if self.hold_before_approach:
            self.state = State.ARMED
            self.status = f"Ready - press G to move to the {self.target_name}"
            return
        self.state = State.APPROACH
        self.status = f"Approaching {self.target_name}..."

    # ------------------------------------------------------------------
    # ARMED: standing still, but planning as if we were about to go
    # ------------------------------------------------------------------
    def armed_tick(self):
        """Keep the plan (and the overlay) live while the arm waits.

        BOTH ends of the plan stay live here: the goal keeps tracking the
        object (move the hammer, the path follows it) and the obstacle keeps
        replanning (put an arm in the way, the path bends). Freezing the goal
        at detection time made the hold look broken - the whole point of this
        state is to watch the plan react before anything moves."""
        if self.abort_requested:
            self.state = State.IDLE
            self.status = "Stopped"
            return
        current = self._safe_posx()
        if current is None:
            self.pose_read_failures += 1
            if self.pose_read_failures >= POSE_READ_FAILURE_LIMIT:
                self.state = State.ERROR
                self.status = "Lost robot pose - stopped"
            return
        self.pose_read_failures = 0
        now = time.monotonic()
        if now - self.last_replan_at < REPLAN_MIN_INTERVAL_SEC:
            return
        self.last_replan_at = now

        moved_mm = self._track_target_while_armed(now)
        obstacle = self.current_obstacle()
        self.path = plan_path(
            np.asarray(current[:3], dtype=float), self.hover_pose, obstacle
        )
        self.path_version += 1
        if self.path is None:
            self.status = "No safe way around your arm - waiting"
            return
        notes = []
        if obstacle is not None:
            notes.append("avoiding your arm")
        if now - self.target_moved_at < 1.5:
            notes.append(f"target moved {moved_mm or self.target_moved_mm:.0f}mm")
        elif self.target_stale_since and now - self.target_stale_since > 2.0:
            notes.append("target not visible - holding last position")
        self.status = (f"Ready - press G to move to the {self.target_name}"
                       + (f"  ({', '.join(notes)})" if notes else ""))

    def _track_target_while_armed(self, now):
        """Re-triangulate and move the goal if the object actually moved.

        A rolling median with the same spread gate as DETECT: raw per-frame
        fixes jitter by millimetres, and letting that jitter drive the goal
        would make the drawn path shiver instead of follow."""
        fix_mm, _ = self._webcam_fix(now)
        if fix_mm is None:
            if self.target_stale_since is None:
                self.target_stale_since = now
            return None
        self.target_stale_since = None
        self.detect_fixes.append(fix_mm)
        if len(self.detect_fixes) < DETECT_SAMPLES:
            return None
        fixes = np.array(self.detect_fixes)
        median = np.median(fixes, axis=0)
        spread = float(np.max(np.linalg.norm(fixes - median, axis=1)))
        if spread > DETECT_SPREAD_MM:
            self.detect_fixes.popleft()
            return None
        moved = float(np.linalg.norm(median - self.target_base))
        if moved < ARMED_TARGET_MOVE_MM:
            return None
        self.target_base = median
        self.hover_pose = self._hover_for(median)
        self.target_moved_at = now
        self.target_moved_mm = moved
        self.get_logger().info(
            f"armed: target moved {moved:.0f}mm -> "
            f"({median[0]:.0f}, {median[1]:.0f}, {median[2]:.0f})"
        )
        return moved

    def handle_go(self):
        """Release the ARMED hold and start the approach for real."""
        if self.state != State.ARMED:
            return
        if self.hover_pose is None:
            return
        self.abort_requested = False
        self.approach_started_at = time.monotonic()
        self.last_cmd_velocity = np.zeros(3)
        self.progress_anchor = None
        self.freewrist_filtered = np.zeros(3)
        self.state = State.APPROACH
        self.status = f"Approaching {self.target_name}..."
        print("GO: approach released")

    # ------------------------------------------------------------------
    # APPROACH: stream along the path, replanning around obstacles
    # ------------------------------------------------------------------
    def approach_tick(self):
        now = time.monotonic()
        if self.abort_requested:
            self._send_stop()
            self.state = State.IDLE
            self.status = "Stopped"
            return
        if now - self.approach_started_at > APPROACH_TIMEOUT_SEC:
            self._send_stop()
            self.state = State.ERROR
            self.status = "Approach timed out"
            return

        current = self._safe_posx()
        if current is None:
            self.pose_read_failures += 1
            self._send_stop()
            if self.pose_read_failures >= POSE_READ_FAILURE_LIMIT:
                self.state = State.ERROR
                self.status = "Lost robot pose - stopped"
            return
        self.pose_read_failures = 0
        position = np.asarray(current[:3], dtype=float)

        obstacle = self.current_obstacle()

        # Obstacle on top of the TCP: freeze rather than try to plan around
        # something already touching the tool. Same hold when it parks on
        # the TARGET - there is no path around a covered goal, and hugging
        # the sphere's surface next to a person's hand helps nobody.
        if obstacle is not None:
            radius = obstacle[2]
            if _capsule_distance(position, obstacle) < radius + TCP_FREEZE_MARGIN_MM:
                self._send_stop()
                self.frozen_for_obstacle = True
                self.last_cmd_velocity = np.zeros(3)
                self.progress_anchor = None
                self.status = "Obstacle at the tool - holding"
                return
            if _capsule_distance(
                np.asarray(self.hover_pose), obstacle
            ) < radius + CHECK_MARGIN_MM:
                self._send_stop()
                self.frozen_for_obstacle = True
                self.last_cmd_velocity = np.zeros(3)
                self.progress_anchor = None
                self.status = "Obstacle covers the target - waiting"
                return
        self.frozen_for_obstacle = False

        # Re-generate the path when the part still ahead is blocked, or when
        # the obstacle has gone and a detour is still being walked.
        need_replan = False
        if self.path is not None:
            ahead = remaining_path(self.path, position)
            if obstacle is not None and path_blocked(
                ahead, obstacle, CHECK_MARGIN_MM
            ) is not None:
                need_replan = True
            elif obstacle is None and len(self.path) > 2:
                need_replan = True
        if need_replan and now - self.last_replan_at > REPLAN_MIN_INTERVAL_SEC:
            self.path = plan_path(position, self.hover_pose, obstacle)
            self.path_version += 1
            self.last_replan_at = now
            self.get_logger().info(
                f"replanned (v{self.path_version}): {len(self.path)} wps "
                f"({self.path[0][0]:.0f},{self.path[0][1]:.0f},{self.path[0][2]:.0f})"
                f" -> ({self.path[-1][0]:.0f},{self.path[-1][1]:.0f},{self.path[-1][2]:.0f})"
                + ("" if obstacle is None else
                   f"  capsule ({obstacle[0][0]:.0f},{obstacle[0][1]:.0f},"
                   f"{obstacle[0][2]:.0f})-({obstacle[1][0]:.0f},"
                   f"{obstacle[1][1]:.0f},{obstacle[1][2]:.0f}) r={obstacle[2]:.0f}mm")
            )

        # After replanning, the path may STILL be blocked - meaning every
        # legal route (over, around either end) failed and only through/
        # under the person's arm remains. The only correct move is to wait.
        if obstacle is not None and self.path is not None:
            if path_blocked(
                remaining_path(self.path, position), obstacle, CHECK_MARGIN_MM
            ) is not None or path_passes_under(
                remaining_path(self.path, position), obstacle
            ):
                self._send_stop()
                self.last_cmd_velocity = np.zeros(3)
                self.progress_anchor = None
                self.status = "No safe way around your arm - waiting"
                return

        error_to_goal = float(np.linalg.norm(self.hover_pose - position))
        if error_to_goal < ARRIVE_TOLERANCE_MM:
            self._send_stop()
            self.intruder_watch = False
            if self.stop_after_approach:
                self.state = State.IDLE
                self.status = f"Arrived (verification stop), err {error_to_goal:.1f}mm"
                self.get_logger().info(
                    f"ARRIVED at hover, error {error_to_goal:.1f}mm - stopping "
                    "here (--stop-after-approach)"
                )
                return
            self.state = State.REFINE
            self.status = "Arrived above target - refining with D435i..."
            self.start_job(self._job_refine_grasp)
            return

        carrot = carrot_on_path(self.path, position, LOOKAHEAD_MM)
        velocity = FOLLOW_KP * (carrot - position)
        speed = float(np.linalg.norm(velocity))
        if speed > APPROACH_MAX_SPEED:
            velocity *= APPROACH_MAX_SPEED / speed
        reaction_scale = self.speed_scale
        if obstacle is not None:
            reaction_scale = max(reaction_scale, OBSTACLE_REACT_MIN_SCALE)
        velocity *= reaction_scale

        if now - self.last_send_at < SEND_INTERVAL_SEC:
            return
        interval = min(now - self.last_send_at, 0.2)
        self.last_send_at = now

        # Slew-limit the command so a replan bends the motion instead of
        # snapping it (see APPROACH_SLEW_MM_S2).
        delta = velocity - self.last_cmd_velocity
        max_delta = APPROACH_SLEW_MM_S2 * interval
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            velocity = self.last_cmd_velocity + delta * (max_delta / delta_norm)
        self.last_cmd_velocity = velocity.copy()

        # Commanding motion but not moving: the controller has stopped
        # listening (protective stop / servo off). Give up loudly.
        commanded = float(np.linalg.norm(velocity))
        if commanded > STUCK_SPEED_MM_S and not self.dry_run:
            if self.progress_anchor is None:
                self.progress_anchor = (now, position.copy())
            else:
                anchor_t, anchor_p = self.progress_anchor
                moved = float(np.linalg.norm(position - anchor_p))
                if moved > STUCK_DISTANCE_MM:
                    self.progress_anchor = (now, position.copy())
                elif now - anchor_t > STUCK_AFTER_SEC:
                    self._send_stop()
                    self.state = State.ERROR
                    self.status = (
                        "Robot not responding to velocity commands - "
                        "likely protective stop. Press R to recover."
                    )
                    self.get_logger().error(
                        f"stuck: commanded {commanded:.0f}mm/s but moved "
                        f"{moved:.1f}mm in {now - anchor_t:.1f}s"
                    )
                    return
        else:
            self.progress_anchor = None
        self.status = (
            f"Approach: {error_to_goal:.0f}mm to go"
            + ("" if obstacle is None else "  [avoiding obstacle]")
        )
        angular = None
        if self.free_wrist:
            tool_axis = self.pose_matrix(*current)[:3, 2]
            desired_axis = freewrist_desired_axis(
                position, velocity, obstacle, error_to_goal
            )
            angular, self.freewrist_filtered = freewrist_angular_command(
                tool_axis, desired_axis, self.freewrist_filtered,
                self._wrist_damping(),
            )
            if angular is not None:
                angular = np.asarray(angular) * reaction_scale

        self.get_logger().info(
            f"approach: pos=({position[0]:.0f},{position[1]:.0f},{position[2]:.0f}) "
            f"goal {error_to_goal:.0f}mm "
            f"vel=({velocity[0]:.0f},{velocity[1]:.0f},{velocity[2]:.0f})"
            + ("" if angular is None else
               f" ang=({angular[0]:.0f},{angular[1]:.0f},{angular[2]:.0f})deg/s")
            + ("" if obstacle is None else
               f" obs=({obstacle[0][0]:.0f},{obstacle[0][1]:.0f},{obstacle[0][2]:.0f})"
               f" r={obstacle[2]:.0f}"),
            throttle_duration_sec=0.5,
        )
        if self.dry_run:
            return
        self._send_velocity(velocity, angular)

    # ------------------------------------------------------------------
    # REFINE + GRASP (blocking job)
    # ------------------------------------------------------------------
    def _bbox_depth(self, box, depth_frame):
        """Object-top depth for a box: 25th percentile of the central half.
        Robust to the box centre landing between a hammer's head and handle
        (where a single-pixel read would return the table)."""
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        w, h = x2 - x1, y2 - y1
        cx1, cy1 = x1 + w // 4, y1 + h // 4
        cx2, cy2 = x2 - w // 4, y2 - h // 4
        height, width = depth_frame.shape[:2]
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(width, cx2), min(height, cy2)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            return None
        crop = depth_frame[cy1:cy2, cx1:cx2]
        valid = crop[crop > 0]
        if valid.size < 20:
            return None
        return float(np.percentile(valid, 25))

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

    def pixel_to_base(self, u, v, depth, base2cam):
        camera = np.array([
            (u - self.intrinsics["ppx"]) * depth / self.intrinsics["fx"],
            (v - self.intrinsics["ppy"]) * depth / self.intrinsics["fy"],
            depth,
            1.0,
        ])
        return (base2cam @ camera)[:3]

    def project_to_pixel(self, point_base, base2cam):
        point_camera = (
            np.linalg.inv(base2cam)
            @ np.append(np.asarray(point_base, dtype=float), 1.0)
        )
        if not np.isfinite(point_camera).all() or point_camera[2] <= 1.0:
            return None
        u = self.intrinsics["fx"] * point_camera[0] / point_camera[2] + self.intrinsics["ppx"]
        v = self.intrinsics["fy"] * point_camera[1] / point_camera[2] + self.intrinsics["ppy"]
        if not np.isfinite([u, v]).all():
            return None
        return int(round(u)), int(round(v))

    def estimate_floor_height(self, u, v, depth_frame, base2cam, center_depth):
        radius_inner = RING_INNER_MM * self.intrinsics["fx"] / center_depth
        radius_outer = RING_OUTER_MM * self.intrinsics["fx"] / center_depth
        heights = []
        for angle in range(0, 360, 15):
            cosine, sine = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            for radius in (radius_inner, (radius_inner + radius_outer) / 2,
                           radius_outer):
                su, sv = int(u + radius * cosine), int(v + radius * sine)
                depth = self.sample_depth(su, sv, depth_frame, patch=3)
                if depth is None:
                    continue
                heights.append(self.pixel_to_base(su, sv, depth, base2cam)[2])
        if len(heights) < RING_MIN_SAMPLES:
            return None
        # Low percentile, not the median: for a tool whose head/handle spill
        # into the 40-70 mm ring, half the samples are OBJECT, and a median
        # splits the difference (measured 26.6 mm for a table at ~0) - which
        # then lifts the grasp depth until the fingers only graze the top.
        # The lowest fifth of the ring is table for anything that doesn't
        # fill it entirely.
        return float(np.percentile(heights, 20))

    def _grasp_mask_geometry(self, box, depth_frame):
        """(centroid_u, centroid_v, axis_angle_image_deg) of the object's own
        depth pixels inside `box`, or None when foreground and background
        cannot be separated (flat object - caller falls back to box centre).
        axis_angle is the long axis of the mask in image coordinates."""
        x1, y1, x2, y2 = (int(round(v)) for v in box)
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
        cut = max(MASK_MIN_SEPARATION_MM,
                  MASK_HEIGHT_RATIO * max(background - nearest, 0.0))
        mask = (crop > 0) & (crop < background - cut)
        if int(mask.sum()) < MASK_MIN_PIXELS:
            return None
        rows, cols = np.nonzero(mask)
        points = np.column_stack([cols, rows]).astype(np.float32)
        (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = cv2.minAreaRect(points)
        # minAreaRect's angle refers to the first side; make it the LONG one.
        if rect_w < rect_h:
            rect_angle += 90.0
        centroid_u = float(cols.mean()) + x1
        centroid_v = float(rows.mean()) + y1
        return centroid_u, centroid_v, rect_angle

    def _grasp_yaw(self, center_uv, axis_angle_deg, depth, base2cam):
        """Wrist yaw (deg, about base Z) that puts the finger-opening axis
        perpendicular to the object's long axis, from the mask's image-space
        angle carried into the base frame by unprojecting two points on it."""
        angle = math.radians(axis_angle_deg)
        du, dv = math.cos(angle) * 40.0, math.sin(angle) * 40.0
        p1 = self.pixel_to_base(center_uv[0] + du, center_uv[1] + dv, depth, base2cam)
        p2 = self.pixel_to_base(center_uv[0] - du, center_uv[1] - dv, depth, base2cam)
        axis = p1[:2] - p2[:2]
        if float(np.linalg.norm(axis)) < 1e-3:
            return 0.0
        long_angle = math.degrees(math.atan2(axis[1], axis[0]))
        yaw = long_angle + 90.0 - FINGER_AXIS_HOME_DEG
        while yaw > 90.0:
            yaw -= 180.0
        while yaw < -90.0:
            yaw += 180.0
        return float(np.clip(yaw, -GRASP_MAX_YAW_DEG, GRASP_MAX_YAW_DEG))

    def pose_is_reachable(self, position, orientation):
        """Ask the robot, not our geometry, whether it can hold this pose.

        Returns (ok, why). Falls back to "yes" when the IK service is not
        available (dry run, no driver): refusing every pose because we could
        not ask would break the feature offline, and movel still fails
        safely on its own if the pose is truly impossible."""
        if ikin is None or get_current_solution_space is None or self.dry_run:
            return True, "ik unavailable"
        try:
            solution_space = get_current_solution_space()
            if isinstance(solution_space, (list, tuple)):
                solution_space = solution_space[0]
            joints = ikin(
                posx([float(position[0]), float(position[1]),
                      float(position[2]), float(orientation[0]),
                      float(orientation[1]), float(orientation[2])]),
                int(solution_space))
        except Exception as error:
            self.get_logger().warn(f"ikin unavailable ({error}); "
                                   "falling back to the geometric check")
            return True, "ik unavailable"
        if joints is None:
            return False, "no inverse-kinematics solution"
        try:
            angles = [float(v) for v in list(joints)[:6]]
        except Exception:
            return False, "unreadable ik result"
        if len(angles) < 6 or not all(math.isfinite(v) for v in angles):
            return False, "unreadable ik result"
        # An all-zero answer is what this service returns when it fails.
        if all(abs(v) < 1e-9 for v in angles):
            return False, "no inverse-kinematics solution"
        for index, (angle, (low, high)) in enumerate(
                zip(angles, JOINT_LIMITS_DEG)):
            if not (low + JOINT_LIMIT_MARGIN_DEG <= angle
                    <= high - JOINT_LIMIT_MARGIN_DEG):
                return False, f"J{index + 1} at {angle:.0f}deg is past its limit"
        if abs(angles[4]) < J5_SINGULARITY_MARGIN_DEG:
            return False, (f"J5 at {angles[4]:.0f}deg sits in the wrist "
                           "singularity")
        return True, ""

    def _tcp_from_camera(self, camera_matrix):
        """TCP pose (4x4) that puts the wrist COLOR camera at `camera_matrix`.

        The inverse of _wrist_cam_matrix. Aiming needs it because the camera
        is NOT at the TCP: it sits on the flange, tens of millimetres off to
        the side. Pointing the TCP at a spot leaves the camera looking past
        it - invisible while shooting straight down, because the offset is
        then mostly along the line of sight, and glaring the moment the wrist
        tilts."""
        if (self.flange2depth is not None and self.flange2tcp is not None
                and self.flange2color is not None):
            return camera_matrix @ np.linalg.inv(self.flange2color) \
                @ self.flange2tcp
        return camera_matrix @ np.linalg.inv(self.gripper2cam)

    def _wrist_cam_matrix(self, current):
        """base -> wrist COLOR camera for the given TCP posx, preferring the
        bundle's holdout-validated flange->depth calibration composed with
        the driver's own depth->color extrinsics."""
        base2tcp = self.pose_matrix(*current)
        if self.flange2depth is not None and self.flange2tcp is not None:
            if self.flange2color is None and self.img_node._extrinsics is not None:
                rotation, translation = self.img_node._extrinsics
                color_from_depth = np.eye(4)
                color_from_depth[:3, :3] = rotation
                color_from_depth[:3, 3] = translation
                self.flange2color = self.flange2depth @ np.linalg.inv(
                    color_from_depth
                )
            if self.flange2color is not None:
                return (
                    base2tcp @ np.linalg.inv(self.flange2tcp) @ self.flange2color
                )
        return base2tcp @ self.gripper2cam

    def _d435_find_target(self):
        """One D435i detection pass. Returns dict or None."""
        frame = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()
        current = self._safe_posx()
        if frame is None or depth_frame is None or current is None:
            return None
        base2cam = self._wrist_cam_matrix(current)

        detections = self.yolo.predict_frame(frame, confidence_threshold=REFINE_CONF)
        matches = [d for d in detections if d["name"] == self.target_name]
        if not matches:
            return None
        # The instance we triangulated from the webcams, not just the
        # highest-scoring one of that class.
        expected = None
        if self.target_base is not None:
            expected = self.project_to_pixel(self.target_base, base2cam)
        if expected is not None and len(matches) > 1:
            def pixel_distance(det):
                x1, y1, x2, y2 = det["box"]
                return math.hypot(
                    (x1 + x2) / 2 - expected[0], (y1 + y2) / 2 - expected[1]
                )
            best = min(matches, key=pixel_distance)
        else:
            best = max(matches, key=lambda det: det["score"])

        x1, y1, x2, y2 = best["box"]
        center_u = int(round((x1 + x2) / 2))
        center_v = int(round((y1 + y2) / 2))
        # Prefer the object's own pixels: the bare box centre of an L-shaped
        # tool can fall between head and handle (one run closed the fingers
        # on exactly that gap edge and spun the hammer instead of holding
        # it). The mask also carries the long axis for the wrist yaw.
        geometry = self._grasp_mask_geometry(best["box"], depth_frame)
        if geometry is not None:
            center_u = int(round(geometry[0]))
            center_v = int(round(geometry[1]))
        depth = self._bbox_depth(best["box"], depth_frame)
        if depth is None:
            depth = self.sample_depth(center_u, center_v, depth_frame)
        if depth is None:
            return None
        position = self.pixel_to_base(center_u, center_v, depth, base2cam)
        if not np.isfinite(position).all():
            return None
        yaw = 0.0
        if geometry is not None:
            yaw = self._grasp_yaw(
                (geometry[0], geometry[1]), geometry[2], depth, base2cam
            )
        floor_z = self.estimate_floor_height(
            center_u, center_v, depth_frame, base2cam, depth
        )
        return {
            "uv": (center_u, center_v),
            "pos": np.asarray(position, dtype=float),
            "floor_z": floor_z,
            "box": best["box"],
            "score": best["score"],
            "yaw": yaw,
        }

    def _job_refine_grasp(self):
        time.sleep(REFINE_SETTLE_SEC)
        rx, ry, rz = self.grasp_orientation
        hover_z = float(self.hover_pose[2])

        # Free-wrist travel may arrive with a few degrees of residual tilt
        # (the ramp converges but never exactly): square the tool up before
        # anything downstream assumes top-down.
        current = self._safe_posx()
        if current is not None:
            error = Rotation.from_matrix(
                self.pose_matrix(*current)[:3, :3]
                @ self.pose_matrix(0, 0, 0, rx, ry, rz)[:3, :3].T
            ).as_rotvec(degrees=True)
            if float(np.linalg.norm(error)) > FREEWRIST_ARRIVE_TOL_DEG:
                self.go_linear(
                    "orientation restore",
                    np.asarray(current[:3]),
                    (rx, ry, rz),
                )
                time.sleep(0.2)

        found = None
        for attempt in range(REFINE_ATTEMPTS):
            if self.abort_requested:
                self.state = State.IDLE
                self.status = "Stopped"
                return
            found = self._d435_find_target()
            if found is None:
                time.sleep(0.4)
                continue
            current = self._safe_posx()
            if current is None:
                found = None
                time.sleep(0.4)
                continue
            offset = math.hypot(
                found["pos"][0] - current[0], found["pos"][1] - current[1]
            )
            self.get_logger().info(
                f"refine {attempt}: target ({found['pos'][0]:.1f}, "
                f"{found['pos'][1]:.1f}, {found['pos'][2]:.1f}) "
                f"offset {offset:.1f}mm score {found['score']:.2f}"
            )
            if offset <= REFINE_CENTER_TOL_MM or attempt == REFINE_ATTEMPTS - 1:
                break
            # Re-centre the camera straight above the box centre and look
            # again - the closer look is the more accurate one.
            self.state = State.REFINE
            self.status = f"Centering above {self.target_name} ({offset:.0f}mm off)"
            self.go_linear(
                "refine centre",
                _clamp_box(np.array([found["pos"][0], found["pos"][1], hover_z])),
                (rx, ry, rz),
            )
            time.sleep(0.3)

        if found is None:
            self.state = State.ERROR
            self.status = f"D435i cannot find {self.target_name} - press S to retry"
            return

        object_top = float(found["pos"][2])
        floor_z = found["floor_z"]
        grasp_z = object_top - PLUNGE_MM
        if floor_z is not None:
            grasp_z = max(grasp_z, floor_z + GRASP_ABOVE_FLOOR_MM)
        # Operator trim, applied AFTER the floor guard: on a thin handle the
        # floor clamp is what actually sets the height, so nudging PLUNGE_MM
        # would do nothing. Adjustable live with [ and ] because the right
        # value depends on the object and the day's depth reading.
        grasp_z += self.grasp_lift_mm
        grasp_z = float(np.clip(grasp_z, GRASP_Z_FLOOR_MM, BOX_Z[1]))
        # Yaw the wrist so the fingers close ACROSS the object's long axis.
        # Prepending a base-Z rotation to a ZYZ orientation is exactly adding
        # to its first angle.
        yaw = float(found.get("yaw", 0.0))
        grasp_orientation = (rx + yaw, ry, rz)
        self.get_logger().info(
            f"grasp: yaw {yaw:+.0f}deg, z {grasp_z:.1f} "
            f"(top {object_top:.1f}, "
            f"floor {floor_z if floor_z is None else round(floor_z, 1)}, "
            f"lift +{self.grasp_lift_mm:.0f})"
        )
        self.pick = {
            "pos": found["pos"].copy(),
            "grasp_z": grasp_z,
            "height": None if floor_z is None else object_top - floor_z,
            # TCP height above the measured pick surface while gripping.
            # Reusing this at PLACE (over the measured place surface) makes
            # any constant camera z-bias cancel out of the release height.
            "grip_above_floor": None if floor_z is None else grasp_z - floor_z,
            "orientation": grasp_orientation,
        }

        self.state = State.GRASP
        self.status = f"Grasping {self.target_name}..."
        # Open first, descend with the fingers already clear.
        self.grip(close=False)
        if self.abort_requested:
            self.state = State.IDLE
            self.status = "Stopped"
            return
        self.go_linear(
            "grasp descend",
            np.array([found["pos"][0], found["pos"][1], grasp_z]),
            grasp_orientation,
        )
        self.grip(close=True)
        self.go_linear(
            "grasp lift",
            np.array([found["pos"][0], found["pos"][1], grasp_z + APPROACH_HEIGHT]),
            grasp_orientation,
            velocity=LIFT_VEL,
            acceleration=LIFT_ACC,
        )

        if self.hand_place:
            # The operator's hand is the destination. Re-arm the webcam hand
            # tracking (it was the obstacle detector during approach) and
            # hand control back to the render loop's deliver_tick.
            self.intruder_watch = True
            self.deliver_started_at = time.monotonic()
            self.deliver_dwell_since = None
            self.deliver_palm_history.clear()
            self.last_cmd_velocity = np.zeros(3)
            self.state = State.DELIVER_TRACK
            self.status = "Hold out your open hand..."
            return

        self.state = State.TO_PLACE_VIEW
        self.status = "Moving to PLACE view..."
        self.go_joint("place view", HOME_JOINT)
        if self.place_base is not None:
            self.place = {
                "pos": np.asarray(self.place_base, dtype=float),
                "surface_z": float(self.place_base[2]),
                "uv": None,
            }
            self.state = State.PLACING
            self.status = "Placing at preset position..."
            self._do_place()
        else:
            self.state = State.WAIT_PLACE_CLICK
            self.status = "Click the PLACE point in the D435i view"

    # ------------------------------------------------------------------
    # DELIVER: place the object on the operator's hand
    # ------------------------------------------------------------------
    def deliver_tick(self):
        """Track the operator's hand and hover above it; when the palm holds
        still and the TCP is aligned, commit the release. Streaming, like
        the approach - the hand may wander and the arm follows it."""
        now = time.monotonic()
        if self.abort_requested:
            self._send_stop()
            self.state = State.IDLE
            self.status = "Stopped (still holding the object - press O to open)"
            return
        current = self._safe_posx()
        if current is None:
            self.pose_read_failures += 1
            self._send_stop()
            if self.pose_read_failures >= POSE_READ_FAILURE_LIMIT:
                self.state = State.ERROR
                self.status = "Lost robot pose - stopped"
            return
        self.pose_read_failures = 0
        position = np.asarray(current[:3], dtype=float)

        hand = self.hand_target()
        if hand is None:
            self._send_stop()
            self.last_cmd_velocity = np.zeros(3)
            self.deliver_dwell_since = None
            self.deliver_palm_history.clear()
            if now - self.deliver_started_at > DELIVER_TIMEOUT_SEC:
                # Nobody's hand showed up: fall back to the click flow so
                # the object is never stranded in the gripper.
                self.intruder_watch = False
                self.state = State.TO_PLACE_VIEW
                self.status = "No hand seen - falling back to click place"
                self.start_job(self._job_fallback_place_view)
                return
            self.status = "Hold out your open hand (webcams must see it)"
            return

        palm, palm_speed = hand
        # Steadiness from where the palm actually WAS, not the KF velocity.
        self.deliver_palm_history.append((now, palm.copy()))
        while (
            self.deliver_palm_history
            and now - self.deliver_palm_history[0][0]
            > DELIVER_STABLE_WINDOW_SEC + 0.3
        ):
            self.deliver_palm_history.popleft()
        window = [
            entry for entry in self.deliver_palm_history
            if now - entry[0] <= DELIVER_STABLE_WINDOW_SEC
        ]
        spread = None
        palm_smooth = palm
        if len(window) >= 4:
            points = np.array([entry[1] for entry in window])
            median = np.median(points, axis=0)
            palm_smooth = median
            deviations = np.linalg.norm(points - median, axis=1)
            spread = float(np.percentile(deviations, DELIVER_STABLE_PERCENTILE))
        palm_steady = (
            spread is not None
            and now - window[0][0] >= DELIVER_STABLE_WINDOW_SEC * 0.8
            and spread < DELIVER_STABLE_RADIUS_MM
        )

        # Track (and later release above) the MEDIAN palm, not the latest
        # jittering fix - a steadier target also steadies the alignment gate.
        target = _clamp_box(np.array([
            palm_smooth[0], palm_smooth[1], palm_smooth[2] + DELIVER_HOVER_MM
        ]))
        error = target - position
        xy_error = float(np.linalg.norm(error[:2]))
        z_error = abs(float(error[2]))

        velocity = FOLLOW_KP * error
        speed = float(np.linalg.norm(velocity))
        cap = APPROACH_MAX_SPEED * DELIVER_SPEED_SCALE
        if speed > cap:
            velocity *= cap / speed
        velocity *= self.speed_scale
        if now - self.last_send_at >= SEND_INTERVAL_SEC:
            interval = min(now - self.last_send_at, 0.2)
            self.last_send_at = now
            delta = velocity - self.last_cmd_velocity
            max_delta = APPROACH_SLEW_MM_S2 * interval
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_delta:
                velocity = self.last_cmd_velocity + delta * (max_delta / delta_norm)
            self.last_cmd_velocity = velocity.copy()
            if not self.dry_run:
                self._send_velocity(velocity)

        aligned = (
            xy_error < DELIVER_XY_TOL_MM
            and z_error < DELIVER_Z_TOL_MM
            and palm_steady
            and palm_speed < DELIVER_HAND_SPEED_MM_S
        )
        self.get_logger().info(
            f"deliver: xy {xy_error:.0f} z {z_error:.0f} "
            f"spread80 {'--' if spread is None else f'{spread:.0f}'}mm "
            f"kfvel {palm_speed:.0f} steady={palm_steady}",
            throttle_duration_sec=1.0,
        )
        if aligned:
            if self.deliver_dwell_since is None:
                self.deliver_dwell_since = now
            elif now - self.deliver_dwell_since >= DELIVER_DWELL_SEC:
                self._send_stop()
                # Snapshot the palm NOW. The release job must not re-read the
                # live track: turning the tracker off at commit made the job
                # find "no hand" and bounce back to tracking, forever.
                self.deliver_commit_palm = palm_smooth.copy()
                self.state = State.DELIVER_RELEASE
                self.status = "Placing it on your hand..."
                if not self.start_job(self._job_deliver_release):
                    self.state = State.DELIVER_TRACK
                return
            self.status = (
                f"Hand steady - releasing in "
                f"{DELIVER_DWELL_SEC - (now - self.deliver_dwell_since):.1f}s"
            )
        else:
            self.deliver_dwell_since = None
            self.status = (
                f"Tracking your hand ({xy_error:.0f}mm off"
                + ("" if palm_steady else ", hold your hand still")
                + ")"
            )

    def _job_fallback_place_view(self):
        self.go_joint("place view", HOME_JOINT)
        self.state = State.WAIT_PLACE_CLICK
        self.status = "Click the PLACE point in the D435i view"

    def _job_deliver_release(self):
        time.sleep(0.3)
        palm = self.deliver_commit_palm
        rx, ry, rz = self.pick["orientation"]
        grip_above = self.pick.get("grip_above_floor")
        if grip_above is None:
            grip_above = 40.0
        release_z = float(palm[2]) + grip_above + DELIVER_CLEARANCE_MM
        release_z = max(release_z, float(palm[2]) + 15.0)
        target = _clamp_box(np.array([palm[0], palm[1], release_z]))
        self.get_logger().info(
            f"deliver: palm ({palm[0]:.0f},{palm[1]:.0f},{palm[2]:.0f}) "
            f"release z {release_z:.0f}"
        )
        self.go_linear(
            "deliver descend", target, (rx, ry, rz),
            velocity=DELIVER_RELEASE_VEL, acceleration=LIFT_ACC,
        )
        self.grip(close=False)
        self.go_linear(
            "deliver retreat",
            np.array([target[0], target[1], target[2] + APPROACH_HEIGHT]),
            (rx, ry, rz),
        )
        self.go_joint("home", HOME_JOINT)
        self.intruder_watch = False
        self.pick = None
        self.place = None
        self.state = State.IDLE
        self.status = "Delivered! Press S to run again"

    # ------------------------------------------------------------------
    # PLACE
    # ------------------------------------------------------------------
    def on_place_click(self, u, v):
        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            self.status = "No depth frame; click again"
            return
        depth = self.sample_depth(u, v, depth_frame)
        if depth is None:
            self.status = "No depth at that pixel; click again"
            return
        current = self._safe_posx()
        if current is None:
            self.status = "Robot pose unavailable; click again"
            return
        base2cam = self._wrist_cam_matrix(current)
        position = self.pixel_to_base(u, v, depth, base2cam)
        if not np.isfinite(position).all() or position[2] < -10:
            self.status = "Invalid PLACE coordinate; click again"
            return
        self.place = {
            "pos": np.asarray(position, dtype=float),
            "surface_z": float(position[2]),
            "uv": (u, v),
        }
        self.state = State.PLACING
        self.status = "Placing object..."
        self.start_job(self._do_place)

    def _do_place(self):
        position = self.place["pos"]
        rx, ry, rz = self.pick["orientation"]
        grip_above = self.pick.get("grip_above_floor")
        if grip_above is None:
            place_z = self.place["surface_z"] + PLACE_BLIND_DROP
        else:
            # Same TCP height above the place surface as above the pick
            # surface: the object touches down exactly as it was lifted, and
            # a constant measurement bias cancels between the two surfaces.
            place_z = self.place["surface_z"] + grip_above + PLACE_CLEARANCE
        place_z = float(np.clip(place_z, GRASP_Z_FLOOR_MM, BOX_Z[1]))
        target_xy = _clamp_box(np.array([position[0], position[1], place_z]))

        self.go_linear(
            "place approach",
            np.array([target_xy[0], target_xy[1], place_z + APPROACH_HEIGHT]),
            (rx, ry, rz),
        )
        self.go_linear(
            "place lower",
            np.array([target_xy[0], target_xy[1], place_z]),
            (rx, ry, rz),
        )
        self.grip(close=False)
        self.go_linear(
            "place retreat",
            np.array([target_xy[0], target_xy[1], place_z + APPROACH_HEIGHT]),
            (rx, ry, rz),
        )
        self.go_joint("home", HOME_JOINT)
        self.pick = None
        self.place = None
        self.state = State.IDLE
        self.status = "Done - press S to run again"

    # ------------------------------------------------------------------
    # Jobs and primitives
    # ------------------------------------------------------------------
    def start_job(self, function):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self._run_job, args=(function,), daemon=True).start()
        return True

    def _run_job(self, function):
        try:
            with self.robot_lock:
                function()
        except Exception as error:
            self.get_logger().error(f"Job failed: {error}")
            self._send_stop()
            self.state = State.ERROR
            self.status = f"ERROR: {error}"
        finally:
            with self.lock:
                self.busy = False

    def go_linear(self, label, xyz, orientation, velocity=LINEAR_VEL,
                  acceleration=LINEAR_ACC):
        if self.abort_requested:
            raise RuntimeError(f"aborted before {label}")
        target = [float(xyz[0]), float(xyz[1]), float(xyz[2]),
                  float(orientation[0]), float(orientation[1]), float(orientation[2])]
        self.get_logger().info(
            f"{label}: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})"
        )
        if self.dry_run:
            time.sleep(0.3)
            return
        result = movel(
            posx(target),
            vel=max(5, velocity * self.speed_scale),
            acc=max(20, acceleration * self.speed_scale),
        )
        if result != 0:
            raise RuntimeError(f"{label} movel failed ({result})")

    def go_joint(self, label, joints):
        if self.abort_requested:
            raise RuntimeError(f"aborted before {label}")
        self.get_logger().info(f"{label}: {joints}")
        if self.dry_run:
            time.sleep(0.3)
            return
        result = movej(
            posj(joints),
            vel=max(5, JOINT_VEL * self.speed_scale),
            acc=max(10, JOINT_ACC * self.speed_scale),
        )
        if result != 0:
            raise RuntimeError(f"{label} movej failed ({result})")

    def grip(self, close):
        if self.gripper is None:
            self.get_logger().warn("Gripper unavailable; skipping")
            return
        if self.dry_run:
            return
        if close:
            self.gripper.close_gripper()
        else:
            self.gripper.open_gripper()
        time.sleep(1.0)
        # The RG2 silently ignores commands while a fingertip safety switch
        # is latched - the width readback is the only honest confirmation.
        try:
            width = self.gripper.get_width()
            status = self.gripper.get_status()
            self.get_logger().info(
                f"gripper {'close' if close else 'open'}: width {width:.1f}mm"
                + (" GRIP-DETECTED" if status and status[1] else "")
            )
            if status and (status[3] or status[5]):
                self.get_logger().error(
                    "RG2 safety circuit latched - it will ignore all motion "
                    "until the tool power is cycled!"
                )
        except Exception as error:
            self.get_logger().warn(f"gripper status read failed: {error}")

    def handle_abort(self):
        self.abort_requested = True
        self._send_stop()
        if self.state in (State.DETECT, State.ARMED, State.APPROACH):
            self.state = State.IDLE
        self.status = "Stopped"

    # -- recovery (copied behaviour from the tracker) -------------------
    ROBOT_STATE_NAMES = {
        0: "INITIALIZING", 1: "STANDBY", 2: "MOVING", 3: "SAFE_OFF",
        4: "TEACHING", 5: "SAFE_STOP", 6: "EMERGENCY_STOP", 7: "HOMING",
        8: "RECOVERY", 9: "SAFE_STOP2", 10: "SAFE_OFF2", 15: "NOT_READY",
    }

    def handle_recover(self):
        if self.busy:
            self.status = "Robot is busy; R ignored"
            return
        self.abort_requested = False
        self.state = State.IDLE
        self.status = "Recovering robot..."
        self.start_job(self._job_recover)

    def _job_recover(self):
        self._send_stop()
        try:
            state = get_robot_state()
        except Exception:
            state = None
        self.get_logger().info(
            f"recover: state={self.ROBOT_STATE_NAMES.get(state, state)}"
        )
        if state in (STATE_STANDBY, STATE_MOVING):
            self.status = "Robot OK - press S to start"
            return
        for attempt in range(3):
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
            except Exception:
                pass
            time.sleep(0.6)
            try:
                state = get_robot_state()
            except Exception:
                state = None
            if state in (STATE_STANDBY, STATE_MOVING):
                self.status = "Robot recovered - press S to start"
                return
        self.status = (
            f"Cannot clear {self.ROBOT_STATE_NAMES.get(state, state)} - "
            "check E-stop / teach pendant"
        )

    def _call_robot_control(self, control):
        if not self.robot_control_client.wait_for_service(timeout_sec=2.0):
            return False
        request = SetRobotControl.Request()
        request.robot_control = int(control)
        future = self.robot_control_client.call_async(request)
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        result = future.result() if future.done() else None
        return bool(result and getattr(result, "success", False))

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _scaler(self, frame):
        """Pixel mapper from a source frame into the quadrant."""
        scale_x = QUAD_W / frame.shape[1]
        scale_y = QUAD_H / frame.shape[0]

        def scale(pixels):
            if pixels is None:
                return None
            if isinstance(pixels, list):        # project() always returns a list
                return [None if p is None else (p[0] * scale_x, p[1] * scale_y)
                        for p in pixels]
            return (pixels[0] * scale_x, pixels[1] * scale_y)

        return scale, scale_x, scale_y

    def _draw_webcam_quadrant(self, name, frame, current):
        view = cv2.resize(frame, (QUAD_W, QUAD_H))
        scale, scale_x, scale_y = self._scaler(frame)
        camera = self.rig.cameras[name]
        hud = self._hud[name].begin(view)
        phase = -(time.monotonic() * PATH_DASH_SPEED) % 1e6

        with self.webcam_lock:
            detection = self.webcam_dets.get(name)
            others = dict(self.webcam_all_dets.get(name, {}))
            blobs = list(self.webcam_overlay.get(name, []))
            ray = self.pointing_ray
            spot = self.pointing_spot
            locked = self.locked_spot
            locked_ray = self.locked_ray
            live_end = self.pointing_end
            locked_age = time.monotonic() - self.locked_at
            frozen = self.point_frozen
            pointed = self.pointing_choice
            scores = dict(self.pointing_scores)
            ambiguous = self.pointing_ambiguous
            obstacle = None
            if self.obstacle is not None:
                center = self.obstacle["kf"].pos / 1000.0
                obstacle = (
                    center,
                    center + self.obstacle["forearm_offset"] / 1000.0,
                    self.obstacle["radius_mm"] / 1000.0,
                )

        # Planned path first so the arm and the hand shell draw over it.
        if self.path is not None and len(self.path) >= 2:
            dense_mm = ar_hud.resample_polyline(self.path, step_mm=16.0)
            points_m = dense_mm / 1000.0
            pixels = scale(camera.project(points_m))
            depths = ar_hud.depths_in_camera(camera.T_cam_base, points_m)
            arcs = ar_hud.arc_lengths(dense_mm)
            ground_mm = dense_mm.copy()
            ground_mm[:, 2] = ar_hud.TABLE_Z_MM
            ground_px = scale(camera.project(ground_mm / 1000.0))
            # The hand is a tracked volume, so the stretch of path it hides
            # from THIS camera is computable - drawn as a ghost, the plan
            # passes behind the operator's arm instead of over it.
            occluded = None
            if obstacle is not None and hud.fancy:
                occluded = ar_hud.occluded_by_capsule(
                    camera.T_base_cam[:3, 3], points_m,
                    obstacle[0], obstacle[1], obstacle[2])
            hud.shadow(ground_px, track_color=(60, 48, 18))
            hud.drop_lines(pixels, ground_px, (52, 42, 16))
            hud.ribbon(pixels, depths, COLOR_PATH_NEAR, COLOR_PATH_FAR,
                       phase_mm=phase, arc_mm=arcs, occluded=occluded)
            goal = pixels[-1] if pixels else None
            if goal is not None:
                hidden_goal = occluded is not None and bool(occluded[-1])
                ink = tuple(int(c * 0.34) for c in COLOR_PATH_NEAR) \
                    if hidden_goal else COLOR_PATH_NEAR
                hud.circle(goal, 9, ink, 1)
                if not hidden_goal:
                    hud.circle(goal, 3, ink, -1)

        # Every other class the cameras can see, drawn faintly: these are what
        # a finger (or the T key) can switch to, and an operator cannot choose
        # what the window never showed them.
        for class_name, other in others.items():
            if class_name == self.target_name:
                continue
            ox1, oy1, ox2, oy2 = other["box"]
            obox = (ox1 * scale_x, oy1 * scale_y, ox2 * scale_x, oy2 * scale_y)
            angle = scores.get(class_name, {}).get("angle_deg")
            hud.brackets(obox, COLOR_OTHER, thickness=1)
            label = class_name.upper()
            if angle is not None and angle < 90.0:
                label += f"  {angle:.0f}deg"
            hud.text(label, (obox[0], max(14, obox[1] - 5)), COLOR_OTHER,
                     size=12)
        if detection:
            x1, y1, x2, y2 = detection["box"]
            box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            chosen_by_finger = pointed == self.target_name
            colour = COLOR_POINTED if chosen_by_finger else COLOR_TARGET
            hud.brackets(box, colour, thickness=3 if chosen_by_finger else 2)
            label = f"{self.target_name.upper()} {detection['score']:.2f}"
            if chosen_by_finger:
                label += "  <- POINTED"
            hud.text(label, (box[0], max(14, box[1] - 5)), colour, size=14)
        # The finger's line as this camera sees it, so the operator can tell
        # whether the rig read the gesture the way they meant it. Once the
        # hand is gone the same line stays, faded: during an inspection it is
        # the only thing on screen that says WHERE the robot is looking and
        # FROM WHICH SIDE.
        draw_ray = ((ray[0], ray[1], live_end) if ray is not None
                    else locked_ray)
        if draw_ray is not None and draw_ray[2] is not None:
            origin, direction, stop = draw_ray
            faded = ray is None
            ends = scale(camera.project(
                [np.asarray(origin) / 1000.0, np.asarray(stop) / 1000.0]))
            hud.line(ends[0], ends[1],
                     COLOR_RAY_FADED if faded else COLOR_POINT_RAY,
                     1 if faded else 2)
            if ends[0] is not None:
                hud.circle(ends[0], 5 if faded else 7,
                           COLOR_RAY_FADED if faded else COLOR_POINT_RAY, 2)
            if spot is not None:
                # Where the finger lands on the table right now.
                landing = scale(camera.project([np.asarray(spot) / 1000.0]))[0]
                if landing is not None:
                    hud.circle(landing, 13, COLOR_POINT_RAY, 2)
                    hud.marker(landing, COLOR_POINT_RAY, 18)
        if locked is not None:
            # The standing mark: what [I] will photograph and what the
            # copilot means by "이거". Drawn whether or not a hand is in
            # view, because the decision outlives the gesture.
            pin = scale(camera.project([np.asarray(locked) / 1000.0]))[0]
            if pin is not None:
                live = spot is not None and float(np.linalg.norm(
                    np.asarray(spot) - np.asarray(locked))) < 20.0
                colour = (COLOR_LOCKED_FIRM if frozen else
                          COLOR_POINT_RAY if live else COLOR_LOCKED)
                hud.circle(pin, 20, colour, 3 if frozen else 2)
                hud.circle(pin, 4, colour, -1)
                if frozen:
                    hud.circle(pin, 27, colour, 1)
                hud.text("SHOOTING THIS" if frozen else
                         "TARGET" if live else f"TARGET  {locked_age:.0f}s",
                         (pin[0] + 30, pin[1] - 8), colour, size=13)
        for (u, v, radius_px) in blobs:
            hud.circle((u * scale_x, v * scale_y),
                       max(4, radius_px * scale_x), COLOR_OBSTACLE, 2)
        if obstacle is not None:
            start, end, radius = obstacle
            if hud.fancy:
                # Real capsule shell: rings along the forearm axis, dimmed
                # with depth, instead of two flat circles and a fat line.
                rings = ar_hud.capsule_rings(start, end, radius)
                ring_px = [scale(camera.project(ring)) for ring in rings]
                ring_depth = [camera.depth_of(ring.mean(axis=0)) for ring in rings]
                hud.rings(ring_px, COLOR_OBSTACLE_SHELL, ring_depth)
                for spine in ar_hud.capsule_spines(start, end, radius):
                    hud.polyline(scale(camera.project(spine)),
                                 COLOR_OBSTACLE_SHELL, 1)
            else:
                ends = scale(camera.project([start, end]))
                radii_px = []
                for point, pixel in zip((start, end), ends):
                    depth = camera.depth_of(point)
                    if pixel is None or depth <= 0.05:
                        radii_px.append(None)
                        continue
                    radius_px = radius * camera.K[0, 0] / depth
                    radii_px.append(radius_px)
                    hud.circle(pixel, max(4, radius_px * scale_x),
                               COLOR_OBSTACLE, 2)
                if all(p is not None for p in ends):
                    hud.line(ends[0], ends[1], COLOR_OBSTACLE, max(
                        2, int(np.mean([r for r in radii_px if r is not None]
                                       or [2]) * scale_x)))
            head = scale(camera.project([start]))[0]
            if head is not None:
                hud.text("HAND", (head[0] + 8, head[1]), COLOR_OBSTACLE,
                         size=13)

        # Live TCP frame - also the end-to-end calibration sanity check.
        if current is not None:
            pixel = scale(camera.project([np.asarray(current[:3]) / 1000.0]))[0]
            if hud.fancy and len(current) >= 6:
                pose = self.pose_matrix(*current[:6])
                pose[:3, 3] /= 1000.0
                for origin, tip, color in ar_hud.axis_segments(pose, 0.07):
                    ends = scale(camera.project([origin, tip]))
                    hud.line(ends[0], ends[1], color, 2)
            if pixel is not None:
                hud.marker(pixel, COLOR_TCP, 14)
        hud.text(name.upper(), (8, QUAD_H - 10), COLOR_TEXT, size=15)
        return hud.composite(view)

    def _draw_d435_quadrant(self):
        frame = self.img_node.get_color_frame()
        if frame is None:
            return np.zeros((QUAD_H, QUAD_W, 3), dtype=np.uint8)
        view = cv2.resize(frame, (QUAD_W, QUAD_H))
        hud = self._hud["d435"].begin(view)
        scale_x = QUAD_W / frame.shape[1]
        scale_y = QUAD_H / frame.shape[0]
        if self.state in (State.REFINE, State.GRASP, State.WAIT_PLACE_CLICK):
            # Throttled: an extra YOLO pass per rendered frame would fight
            # the webcam batch for the GPU and stall the control loop.
            now = time.monotonic()
            if now - self._d435_overlay_at > 0.4:
                try:
                    self._d435_overlay = self.yolo.predict_frame(
                        frame, confidence_threshold=REFINE_CONF
                    )
                except Exception:
                    self._d435_overlay = []
                self._d435_overlay_at = now
            for det in self._d435_overlay:
                x1, y1, x2, y2 = det["box"]
                is_target = det["name"] == self.target_name
                color = COLOR_TARGET if is_target else (160, 160, 160)
                box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                if is_target:
                    hud.brackets(box, color)
                    hud.text(f"{det['name'].upper()} {det.get('score', 0.0):.2f}",
                             (box[0], max(14, box[1] - 5)), color, size=14)
                else:
                    cv2.rectangle(view, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), color, 1)
        if self.pick is not None:
            pass  # pick marker only meaningful in base frame
        waiting = self.state == State.WAIT_PLACE_CLICK
        hud.text("D435i" + ("  <- CLICK PLACE" if waiting else ""),
                 (8, QUAD_H - 10),
                 COLOR_PLACE if waiting else COLOR_TEXT, size=15)
        return hud.composite(view)

    def render(self):
        frames = self.rig.frames()
        # One pose read for all three quadrants, cached across frames: the
        # render loop must not hammer the DRFL service at frame rate.
        now = time.monotonic()
        if not self.busy and now - self._render_posx_at > 0.2:
            self._render_posx = self._safe_posx()
            self._render_posx_at = now
        current = self._render_posx
        quadrants = []
        for name in ("cam0", "cam1", "cam2"):
            if name in frames:
                quadrants.append(
                    self._draw_webcam_quadrant(name, frames[name][0], current)
                )
            else:
                quadrants.append(np.zeros((QUAD_H, QUAD_W, 3), dtype=np.uint8))
        quadrants.append(self._draw_d435_quadrant())
        top = np.hstack(quadrants[:2])
        bottom = np.hstack(quadrants[2:])
        canvas = np.vstack([top, bottom])

        self._draw_chrome(canvas)
        self._hud_ms = 0.9 * self._hud_ms + 0.1 * (
            (time.monotonic() - now) * 1000.0)
        cv2.imshow(WINDOW, canvas)

    #: Accent per state, so the status strip carries the phase at a glance.
    STATE_ACCENT = {
        State.IDLE: (150, 150, 150),
        State.HOMING: (255, 200, 60),
        State.DETECT: (255, 200, 60),
        State.ARMED: (120, 255, 255),
        State.INSPECT: (255, 180, 255),
        State.APPROACH: (255, 200, 60),
        State.REFINE: (0, 255, 255),
        State.GRASP: (80, 220, 80),
        State.TO_PLACE_VIEW: (80, 160, 255),
        State.WAIT_PLACE_CLICK: (80, 160, 255),
        State.PLACING: (80, 160, 255),
        State.DELIVER_TRACK: (0, 255, 255),
        State.DELIVER_RELEASE: (80, 220, 80),
        State.TAKE_FROM_HAND: (255, 140, 200),
        State.ERROR: (60, 60, 255),
    }

    def _draw_chrome(self, canvas):
        """Status strip and key legend around the 2x2 grid."""
        height, width = canvas.shape[:2]
        chrome = self._chrome
        chrome.begin(canvas)
        accent = self.STATE_ACCENT.get(self.state, COLOR_TEXT)
        cv2.rectangle(canvas, (0, 0), (width, 34), (0, 0, 0), -1)
        cv2.rectangle(canvas, (0, height - 26), (width, height), (0, 0, 0), -1)
        if chrome.fancy:
            # Thin accent rule under the strip instead of a hard black edge.
            cv2.line(canvas, (0, 34), (width, 34), accent, 1, cv2.LINE_AA)
        chrome.text(self.state.value, (12, 25), accent, size=19)
        offset = 12 + max(90, len(self.state.value) * 13)
        chrome.text(self.status, (offset, 24), COLOR_TEXT, size=15)
        chrome.text(
            "[S] arm  [G] go  [SPACE] stop  [R] recover  [D] dry-run  "
            "[O] open  [C] close  [H] home  [T] target  [1-4] speed  "
            "[F] hud  [X] point  [I] look  [M] take back  [V] view  [0] clear  "
            "[,/.] aim height  [ESC] quit",
            (12, height - 8), (190, 190, 190), size=14)
        if self.state == State.ARMED:
            # The one thing the operator must know in this state, said big.
            chrome.text("PRESS  G  TO GO", (width // 2 - 120, height - 44),
                        (120, 255, 255), size=30)
        with self.webcam_lock:
            pointed = self.pointing_choice
            ambiguous = self.pointing_ambiguous
        chrome.text(f"TARGET {self.target_name}"
                    + ("  (pointed)" if pointed == self.target_name else ""),
                    (width - 470, 25),
                    COLOR_POINTED if pointed == self.target_name
                    else (120, 255, 180), size=17)
        if ambiguous:
            chrome.text("two tools too close to call - move your hand",
                        (width // 2 - 200, 55), (0, 200, 255), size=14)
        start = ("HOME" if self.start_pose is None else
                 f"{self.start_pose[0]:.0f},{self.start_pose[1]:.0f},"
                 f"{self.start_pose[2]:.0f}")
        shot = (f"TILT {self.inspect_standoff:.0f}mm" if self.inspect_oblique
                else f"DOWN {self.inspect_height:.0f}mm")
        chrome.text(f"START {start}   GRASP {self.grasp_lift_mm:+.0f}mm"
                    f"   AIM z{self.aim_plane_mm:+.0f}mm   LOOK {shot}",
                    (12, 55), (150, 190, 210),
                    size=14)
        if self.aim_plane_mm > TABLE_Z_MM + 0.5:
            # Off the table: say so, or a mark that looks fine from above
            # will quietly be pointing at a plane nobody remembers setting.
            chrome.text(f"AIM PLANE {self.aim_plane_mm:.0f}mm",
                        (width - 470, 55), (255, 200, 90), size=15)
        if self.dry_run:
            chrome.text("DRY RUN", (width - 240, height - 8), (0, 255, 255),
                        size=16)
        label = self.speed_label()
        chrome.text(f"{label} x{self.speed_scale:.2f}", (width - 190, 25),
                    _speed_color(self.speed_scale), size=17)
        # Four-segment gauge: which stop the dial is on, at a glance.
        gauge_x, gauge_y = width - 190, 30
        for index, (value, _) in enumerate(SPEED_PRESETS):
            filled = self.speed_scale >= value - 0.02
            cv2.rectangle(
                canvas,
                (gauge_x + index * 22, gauge_y),
                (gauge_x + index * 22 + 17, gauge_y + 4),
                _speed_color(value) if filled else (60, 60, 60), -1)
        if self.show_hud_cost:
            chrome.text(f"hud {self._hud_ms:.1f}ms", (width - 150, height - 8),
                        (140, 140, 140), size=13)
        composed = chrome.composite(canvas)
        if composed is not canvas:
            np.copyto(canvas, composed)

    def mouse_callback(self, event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONDOWN or self.busy:
            return
        if self.state != State.WAIT_PLACE_CLICK:
            return
        # D435i lives in the bottom-right quadrant.
        if x < QUAD_W or y < QUAD_H:
            return
        u = int((x - QUAD_W) * (1280 / QUAD_W))
        v = int((y - QUAD_H) * (720 / QUAD_H))
        self.on_place_click(u, v)

    def handle_home(self):
        if self.busy:
            self.status = "Robot busy; H ignored"
            return
        if self.state in (State.IDLE, State.ERROR, State.WAIT_PLACE_CLICK):
            previous = self.state
            self.state = State.HOMING

            def job():
                self.go_joint("manual home", HOME_JOINT)
                self.state = State.IDLE if previous != State.WAIT_PLACE_CLICK else previous
                self.status = "At HOME"
            self.clear_target_mark()
        self.start_job(job)

    def handle_gripper(self, close):
        if self.busy:
            self.status = "Robot busy; gripper key ignored"
            return
        if self.state in (State.APPROACH, State.REFINE, State.GRASP, State.PLACING):
            self.status = f"Gripper key unavailable in {self.state.value}"
            return
        self.start_job(lambda: self.grip(close))

    @staticmethod
    def _load_start_pose():
        """Remembered demo start, or None. A bad file must never block a run."""
        try:
            if not START_POSE_FILE.is_file():
                return None
            data = json.loads(START_POSE_FILE.read_text())
            pose = [float(v) for v in data["start_pose_mm"]]
            return pose if len(pose) == 3 else None
        except Exception:
            return None

    def capture_start_pose(self):
        """Remember where the arm is standing now as the demo start.

        The useful starting point is the one the operator finds by driving the
        arm until the detour looks good from the cameras - so let them keep it
        rather than describing coordinates."""
        current = self._safe_posx()
        if current is None:
            self.status = "Cannot read pose - start position NOT saved"
            print("start pose: pose read failed, nothing saved")
            return
        pose = [round(float(v), 1) for v in current[:3]]
        inside = (BOX_X[0] <= pose[0] <= BOX_X[1]
                  and BOX_Y[0] <= pose[1] <= BOX_Y[1]
                  and BOX_Z[0] <= pose[2] <= min(BOX_Z[1], PLAN_Z_MAX))
        if not inside:
            self.status = f"Start pose {pose} outside the safe box - not saved"
            print(f"start pose: {pose} outside the safe box, not saved")
            return
        self.start_pose = pose
        try:
            START_POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
            START_POSE_FILE.write_text(json.dumps(
                {"start_pose_mm": pose}, indent=2))
            where = str(START_POSE_FILE)
        except Exception as error:
            where = f"(not saved to disk: {error})"
        self.status = f"Start pose saved: {pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f}"
        print(f"start pose: {pose} -> {where}")

    def nudge_grasp_lift(self, delta):
        self.grasp_lift_mm = float(np.clip(
            self.grasp_lift_mm + delta, *GRASP_LIFT_RANGE_MM))
        self.status = f"Grasp lift {self.grasp_lift_mm:+.0f} mm"
        print(f"grasp lift: {self.grasp_lift_mm:+.1f} mm "
              f"(applies to the NEXT grasp)")

    def nudge_inspect_distance(self, delta):
        """Back the shot off or bring it in, in whichever mode is active.

        Vertical shots move the gripper TIPS, so the floor here is a real
        clearance over the work, not a framing preference."""
        if self.inspect_oblique:
            self.inspect_standoff = float(np.clip(
                self.inspect_standoff + delta * 2.0,
                INSPECT_STANDOFF_MIN_MM, INSPECT_STANDOFF_MAX_MM))
            self.status = f"Inspect standoff {self.inspect_standoff:.0f}mm"
        else:
            self.inspect_height = float(np.clip(
                self.inspect_height + delta * 2.0,
                INSPECT_HEIGHT_MIN_MM, INSPECT_HEIGHT_MAX_MM))
            self.status = (f"Inspect height {self.inspect_height:.0f}mm "
                           f"(lens ~{self.inspect_height + 237:.0f}mm)")
        print(self.status)

    def nudge_aim_plane(self, delta):
        """Slide the pointed mark up or down the finger's own ray.

        Also re-aims a mark that is already locked: the operator locks it,
        sees it sitting on the floor behind the conveyor, and fixes it - so
        the correction has to reach the mark they are looking at, not only
        the next one."""
        self.aim_plane_mm = float(np.clip(
            self.aim_plane_mm + delta, TABLE_Z_MM, AIM_PLANE_MAX_MM))
        with self.webcam_lock:
            ray = self.locked_ray
        if ray is not None and self.locked_spot is not None:
            origin, unit = ray[0], ray[1]
            moved = pointing.ray_plane_point(origin, unit,
                                             plane_z_mm=self.aim_plane_mm)
            if moved is not None:
                moved = _clamp_box(np.array([moved[0], moved[1],
                                             max(moved[2], BOX_Z[0])]))
                moved[2] = self.aim_plane_mm
                self.locked_spot = moved
                with self.webcam_lock:
                    self.locked_ray = (origin, unit, moved.copy())
        self.status = f"Aim plane {self.aim_plane_mm:.0f} mm"
        print(f"aim plane: {self.aim_plane_mm:.0f} mm above the table")

    def speed_label(self):
        """Name of the nearest preset, so the HUD reads 'SLOW' not just x0.45.

        --speed-scale can name any value; the closest preset is the honest
        label for it."""
        value, name = min(SPEED_PRESETS,
                          key=lambda preset: abs(preset[0] - self.speed_scale))
        return name if abs(value - self.speed_scale) < 0.03 else "CUSTOM"

    def handle_inspect(self, point_mm=None, request_id=None, direction=None,
                       standoff_mm=None):
        """Photograph the spot the finger is pointing at, then come home.

        The picture is for a vision model to answer "am I doing this right?",
        so the frame has to be taken from where the wrist camera can actually
        resolve the work - not from the observation pose, which is too far
        away and at the wrong angle."""
        if self.busy or self.state not in (State.IDLE, State.ERROR):
            self.status = f"INSPECT unavailable in {self.state.value}"
            return False
        if point_mm is None or direction is None:
            with self.webcam_lock:
                if point_mm is None and self.pointing_spot is not None:
                    point_mm = np.asarray(self.pointing_spot,
                                          dtype=float).copy()
                if direction is None and self.pointing_ray is not None:
                    direction = np.asarray(self.pointing_ray[1],
                                           dtype=float).copy()
            # Hand already lowered: use the mark it left behind, including
            # the direction it was pointing from, so the oblique view still
            # looks from where the operator was looking.
            if point_mm is None and self.locked_spot is not None:
                point_mm = self.locked_spot.copy()
            if direction is None and self.locked_direction is not None:
                direction = self.locked_direction.copy()
            # Nothing on screen either: fall back to what was just
            # photographed, so "다시, 더 가까이" works without pointing again.
            recent = (time.monotonic() - self.last_inspect_at
                      < INSPECT_RECALL_SEC)
            if point_mm is None and recent and self.last_inspect_point is not None:
                point_mm = self.last_inspect_point.copy()
                if direction is None and self.last_inspect_direction is not None:
                    direction = self.last_inspect_direction.copy()
        if point_mm is None:
            self.status = "Point at the spot first (index finger, 2+ cameras)"
            return False
        self.abort_requested = False
        # Commit exactly what is on screen right now.
        point_mm = np.asarray(point_mm, dtype=float)
        self.locked_spot = point_mm.copy()
        if direction is not None:
            self.locked_direction = np.asarray(direction, dtype=float).copy()
            with self.webcam_lock:
                live_end = self.pointing_end
                live_ray = self.pointing_ray
            if live_ray is not None:
                self.locked_ray = (
                    np.asarray(live_ray[0], dtype=float).copy(),
                    np.asarray(live_ray[1], dtype=float).copy(),
                    np.asarray(live_end if live_end is not None else point_mm,
                               dtype=float).copy())
        self.locked_at = time.monotonic()
        self.last_inspect_point = point_mm.copy()
        self.last_inspect_direction = (
            None if direction is None
            else np.asarray(direction, dtype=float).copy())
        self.last_inspect_at = self.locked_at
        self.point_frozen = True
        self.start_job(lambda: self._job_inspect(point_mm, request_id,
                                                 direction, standoff_mm))
        return True

    def _job_inspect(self, point_mm, request_id=None, direction=None,
                     standoff_mm=None):
        """Go and photograph a spot, HOLD there for the verdict, come home.

        Holding is the point of the loop: the model asks for a closer or a
        wider look often enough that driving home between attempts wastes
        both time and the operator's patience - and a robot that retreats
        before answering reads as a robot that gave up."""
        try:
            self.state = State.INSPECT
            if not self._wait_for_clear_hand(point_mm, request_id):
                return
            self.go_joint("home", HOME_JOINT)
            time.sleep(0.2)
            while True:
                shot = self._inspect_once(point_mm, request_id, direction,
                                          standoff_mm)
                if shot is None:
                    return
                self._inspect_signal.clear()
                self._publish_inspection(
                    self.last_inspection if shot["saved"] else None,
                    request_id,
                    error=None if shot["saved"] else "no camera frame")
                self.status = (f"Captured {shot['name']} - looking at it..."
                               if shot["saved"] else
                               "Captured nothing - waiting")
                heard = self._inspect_signal.wait(INSPECT_HOLD_SEC)
                follow_up, self._inspect_followup = self._inspect_followup, None
                if self.abort_requested:
                    return
                if not (heard and follow_up is not None):
                    return                      # verdict in, or nobody asked
                if follow_up.get("point"):
                    point_mm = np.asarray(follow_up["point"], dtype=float)
                standoff_mm = follow_up.get("standoff_mm") or standoff_mm
                request_id = follow_up.get("request_id") or request_id
                self.status = "Taking another look..."
                self.get_logger().info(
                    f"inspect: second look at {standoff_mm or 0:.0f}mm")
        finally:
            # However this ends - captured, cancelled, out of reach - the arm
            # comes home, the mark comes off the screen, and the next gesture
            # starts fresh. The coordinates live on in last_inspect_* so a
            # late follow-up still has something to aim at.
            self.clear_target_mark()
            self.intruder_watch = False
            try:
                self.go_joint("home", HOME_JOINT)
            except Exception as error:
                self.get_logger().warn(f"inspect: home failed ({error})")
            if self.state == State.INSPECT:
                self.state = State.IDLE
                if not self.status.startswith(("Stopped", "Hand stayed")):
                    self.status = "Inspection finished - S to run, I to look"

    def _inspect_once(self, point_mm, request_id, direction, standoff_mm):
        """Plan a view, drive there slowly, take one picture.

        Returns {"saved", "name"} or None when it could not be done - the
        caller has already been told why through the inspection topic."""
        current = self._safe_posx()
        if current is None:
            self.state = State.ERROR
            self.status = "Robot pose unavailable"
            self._publish_inspection(None, request_id,
                                     error="robot pose unavailable")
            return None
        orientation = tuple(current[3:])
        above = _clamp_box(np.array([point_mm[0], point_mm[1],
                                     point_mm[2] + self.inspect_height]))
        # Put the LENS over the spot, not the TCP. The camera sits ~85 mm to
        # the side of the tool centre, so aiming the TCP leaves the subject a
        # fifth of the frame off centre - measured, not guessed, on the shots
        # from this bench. The oblique path already did this; the vertical
        # shortcut did not, which is why every top-down photograph came out
        # with the red mark down and to the right.
        probe = (float(above[0]), float(above[1]), float(above[2]),
                 *orientation)
        try:
            lens = self._wrist_cam_matrix(probe)[:2, 3]
            centred = _clamp_box(np.array([
                above[0] + (point_mm[0] - lens[0]),
                above[1] + (point_mm[1] - lens[1]),
                above[2]]))
            if inspection_pose_is_safe(centred, min_z=INSPECT_MIN_TCP_Z_MM)[0]:
                above = centred
            else:
                self.get_logger().info(
                    "inspect: lens centring would leave the workspace - "
                    "shooting from the uncentred pose")
        except Exception as failure:
            self.get_logger().warn(f"inspect: lens centring failed ({failure})")
        position, aimed, tilt, standoff, note = (above, None, 0.0,
                                                 self.inspect_height, "vertical")
        if self.inspect_oblique and direction is not None:
            # Geometry proposes, inverse kinematics disposes: a pose inside
            # the reach sphere can still need a joint past its stop or drive
            # the wrist through a singularity, and finding that out by
            # commanding it is how a demo ends in a protective stop.
            chosen = None
            checks = 0
            for candidate in iter_inspection_views(
                    point_mm, direction,
                    standoff_mm=float(np.clip(
                        standoff_mm or self.inspect_standoff,
                        INSPECT_STANDOFF_MIN_MM, INSPECT_STANDOFF_MAX_MM))):
                cam_pos, cam_aim, cand_tilt, cand_dist, cand_note = candidate
                if cam_aim is None:
                    tcp_pos, tcp_aim = cam_pos, orientation
                else:
                    tcp = self._tcp_from_camera(self.pose_matrix(
                        cam_pos[0], cam_pos[1], cam_pos[2], *cam_aim))
                    tcp_pos = tcp[:3, 3].copy()
                    tcp_aim = tuple(float(v) for v in Rotation.from_matrix(
                        tcp[:3, :3]).as_euler("ZYZ", degrees=True))
                    if not inspection_pose_is_safe(
                            tcp_pos, min_z=INSPECT_MIN_TCP_Z_MM)[0]:
                        continue
                checks += 1
                if checks > INSPECT_IK_CHECKS:
                    break
                reachable, why = self.pose_is_reachable(tcp_pos, tcp_aim)
                if not reachable:
                    self.get_logger().info(
                        f"inspect: {cand_tilt:.0f}deg/{cand_dist:.0f}mm "
                        f"rejected - {why}")
                    continue
                chosen = (tcp_pos, None if cam_aim is None else tcp_aim,
                          cand_tilt, cand_dist, cand_note)
                break
            if chosen is None:
                self.status = "The arm cannot reach a view of that spot"
                self._publish_inspection(None, request_id,
                                         error="no reachable view")
                return None
            position, aimed, tilt, standoff, note = chosen

        self.status = (f"Looking at ({point_mm[0]:.0f}, {point_mm[1]:.0f}) "
                       f"from {tilt:.0f}deg ({note})...")
        self.get_logger().info(
            f"inspect view: tilt {tilt:.0f}deg standoff {standoff:.0f}mm "
            f"pos ({position[0]:.0f},{position[1]:.0f},{position[2]:.0f}) "
            f"[{note}]")
        # Two legs, both slow. Straight up over the spot first, then tilt
        # into place: a single move that translates AND rotates makes the
        # wrist race the arm, and a fast wrist near the work is exactly what
        # trips a protective stop.
        staging = _clamp_box(np.array([
            point_mm[0], point_mm[1],
            max(position[2], point_mm[2] + self.inspect_height)]))
        self.go_linear("inspect approach", staging, orientation,
                       velocity=INSPECT_VEL, acceleration=INSPECT_ACC)
        if aimed is not None:
            time.sleep(0.15)
            self.go_linear("inspect aim", position, aimed,
                           velocity=INSPECT_TILT_VEL,
                           acceleration=INSPECT_TILT_ACC)
        elif float(np.linalg.norm(np.asarray(position) - staging)) > 2.0:
            self.go_linear("inspect settle", position, orientation,
                           velocity=INSPECT_TILT_VEL,
                           acceleration=INSPECT_TILT_ACC)
        time.sleep(INSPECT_SETTLE_SEC)

        frame = self.img_node.get_color_frame()
        self.last_inspection = None
        if frame is None:
            self.status = "No wrist camera frame - nothing captured"
            return {"saved": False, "name": ""}
        INSPECT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        saved = INSPECT_DIR / f"inspect_{stamp}.jpg"
        cv2.imwrite(str(saved), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pose = self._safe_posx()
        # A vision model cannot know WHICH of the things in frame the question
        # is about - and the frame usually holds several. We do know: the
        # finger's 3D spot projects to one pixel. Mark a copy and tell the
        # model to look there. The clean original is kept alongside, because a
        # drawn-on image is evidence that has been edited.
        marked = None
        pixel = None
        if pose is not None and self.intrinsics is not None:
            try:
                pixel = self.project_to_pixel(point_mm,
                                              self._wrist_cam_matrix(pose))
            except Exception as failure:
                self.get_logger().warn(f"mark projection failed: {failure}")
            if pixel is not None:
                height, width = frame.shape[:2]
                if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    overlay = frame.copy()
                    cv2.circle(overlay, pixel, 42, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.drawMarker(overlay, pixel, (0, 0, 255),
                                   cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
                    cv2.putText(overlay, "HERE", (pixel[0] + 50, pixel[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    marked = INSPECT_DIR / f"inspect_{stamp}_marked.jpg"
                    cv2.imwrite(str(marked), overlay,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    self.get_logger().warn(
                        f"pointed spot fell outside the frame at {pixel}")
                    pixel = None
        (INSPECT_DIR / f"inspect_{stamp}.json").write_text(json.dumps({
            "image": str(saved),
            "marked_image": None if marked is None else str(marked),
            "pointed_pixel": None if pixel is None else list(pixel),
            "pointed_at_mm": [round(float(v), 1) for v in point_mm],
            "camera_pose": None if pose is None else
                           [round(float(v), 1) for v in pose],
            "tilt_deg": round(float(tilt), 1),
            "standoff_mm": round(float(standoff), 1),
            "view_note": note,
            "captured_at": time.time(),
        }, indent=2))
        self.last_inspection = {
            "path": str(saved),
            "marked_path": None if marked is None else str(marked),
            "pointed_pixel": None if pixel is None else list(pixel),
            "tilt_deg": round(float(tilt), 1),
            "standoff_mm": round(float(standoff), 1),
            "point": [float(v) for v in point_mm],
            "at": time.time(),
        }
        self.get_logger().info(f"inspection saved: {saved}")
        return {"saved": True, "name": saved.name}

    def clear_target_mark(self):
        """Drop the pointing mark and its beam.

        The mark belongs to the look-at-this flow. Leaving it on screen while
        a fetch runs puts two different "targets" in front of the operator -
        the circle on the bench and the class being picked - and they are not
        the same thing."""
        self.point_frozen = False
        self.locked_spot = None
        self.locked_direction = None
        self.locked_ray = None
        self.point_smoother.selection = None
        with self.webcam_lock:
            self.pointing_spot = None
            self.pointing_end = None
            self.pointing_ray = None
            self.pointing_choice = None

    def handle_take_from_hand(self, request_id=None):
        """Take what the operator is holding and put it down somewhere free."""
        if self.busy or self.state not in (State.IDLE, State.ERROR):
            self.status = f"TAKE unavailable in {self.state.value}"
            return False
        self.abort_requested = False
        self.clear_target_mark()
        self.active_request_id = request_id
        self.start_job(lambda: self._job_take_from_hand(request_id))
        return True

    def _steady_palm(self, deadline):
        """Wait for a hand that is holding still, and return where it is.

        Median plus a high percentile of the spread, exactly as the handover
        does: a raw position never settles (vision jitter is 40-80 mm/s even
        on a hand resting on a bench), and the maximum deviation is destroyed
        by a single spike."""
        history = deque()
        while time.monotonic() < deadline:
            if self.abort_requested:
                return None
            target = self.hand_target()
            now = time.monotonic()
            if target is None:
                history.clear()
                self.status = "Hold the object out on your open hand..."
                time.sleep(0.05)
                continue
            history.append((now, target[0].copy()))
            while history and now - history[0][0] > TAKE_STABLE_WINDOW_SEC:
                history.popleft()
            if (len(history) < 5
                    or now - history[0][0] < TAKE_STABLE_WINDOW_SEC * 0.8):
                time.sleep(0.05)
                continue
            points = np.array([entry[1] for entry in history])
            median = np.median(points, axis=0)
            spread = float(np.percentile(
                np.linalg.norm(points - median, axis=1),
                DELIVER_STABLE_PERCENTILE))
            if spread < TAKE_STABLE_RADIUS_MM:
                return median
            self.status = f"Hold still... ({spread:.0f}mm)"
            time.sleep(0.05)
        return None

    def _job_take_from_hand(self, request_id=None):
        self.state = State.TAKE_FROM_HAND
        self.intruder_watch = True          # the hand tracker feeds both jobs
        try:
            self.go_joint("home", HOME_JOINT)
            current = self._safe_posx()
            if current is None:
                raise RuntimeError("cannot read pose")
            orientation = tuple(current[3:])
            self.grip(close=False)          # open before approaching a person

            palm = self._steady_palm(time.monotonic() + TAKE_TIMEOUT_SEC)
            if palm is None:
                self.state = State.IDLE
                self.status = ("Stopped" if self.abort_requested else
                               "No steady hand to take from")
                return
            hover = _clamp_box(np.array([palm[0], palm[1],
                                         palm[2] + TAKE_HOVER_MM]))
            self.status = "Coming to take it - keep your hand still"
            self.go_linear("take hover", hover, orientation,
                           velocity=LINEAR_VEL * 0.6,
                           acceleration=LINEAR_ACC * 0.6)

            # Look from directly above: the object is on the palm, so the
            # palm is the "floor" and the object's top is what we measure.
            found = self._d435_find_target()
            grasp_z = None
            if found is not None:
                grasp_z = float(found["pos"][2]) - PLUNGE_MM
                self.get_logger().info(
                    f"take: object top {found['pos'][2]:.1f} "
                    f"palm {palm[2]:.1f}")
            floor = palm[2] + TAKE_MIN_ABOVE_PALM_MM
            if grasp_z is None:
                # Nothing recognised on the hand - grip just above the palm
                # rather than guessing a height from thin air.
                grasp_z = floor + 10.0
                self.status = "Cannot see what it is - taking it gently"
            grasp_z = max(grasp_z, floor)

            settled = self._steady_palm(time.monotonic() + 8.0)
            if settled is None:
                self.state = State.IDLE
                self.status = ("Stopped" if self.abort_requested
                               else "Hand moved - not taking it")
                return
            if float(np.linalg.norm(settled - palm)) > TAKE_ABORT_MOVE_MM:
                palm = settled          # it moved but is steady again
                hover = _clamp_box(np.array([palm[0], palm[1],
                                             palm[2] + TAKE_HOVER_MM]))
                self.go_linear("take re-hover", hover, orientation,
                               velocity=LINEAR_VEL * 0.6,
                               acceleration=LINEAR_ACC * 0.6)
                grasp_z = max(grasp_z, palm[2] + TAKE_MIN_ABOVE_PALM_MM)

            target = _clamp_box(np.array([palm[0], palm[1], grasp_z]))
            self.status = "Taking it..."
            self.go_linear("take descend", target, orientation,
                           velocity=TAKE_DESCEND_VEL,
                           acceleration=TAKE_DESCEND_ACC)
            self.grip(close=True)
            time.sleep(0.3)
            self.go_linear("take lift", hover, orientation,
                           velocity=TAKE_DESCEND_VEL,
                           acceleration=TAKE_DESCEND_ACC)

            width = None
            try:
                width = float(self.gripper.get_width())
            except Exception as error:
                self.get_logger().warn(f"gripper width read failed: {error}")
            self.get_logger().info(
                f"take: gripper width {width if width is None else round(width, 1)}mm")
            if width is not None and width < 12.0:
                self.grip(close=False)
                self.go_joint("home", HOME_JOINT)
                self.state = State.IDLE
                self.status = "Nothing in the gripper - try again"
                return

            # Put it down somewhere free, away from the person and the tools.
            occupied = list(self.locate_all_objects().values())
            spot, info = find_free_spot(
                occupied, human_mm=palm,
                prefer_near=np.asarray(hover[:2], dtype=float))
            if spot is None:
                self.status = "No free space on the bench - holding it"
                self.state = State.IDLE
                return
            self.get_logger().info(
                f"take: putting it at ({spot[0]:.0f},{spot[1]:.0f}) {info}")
            self.intruder_watch = False
            # _do_place lowers to "the same TCP height above the place
            # surface as above the pick surface". Coming off a hand there is
            # no measured pick surface, so state the height explicitly: the
            # bench is the surface and the object hangs below the fingers by
            # however far we reached past the palm.
            self.pick = {
                "pos": palm.copy(),
                "grasp_z": grasp_z,
                "orientation": orientation,
                "grip_above_floor": max(6.0, grasp_z - palm[2]),
            }
            self.place = {"pos": spot.copy(), "surface_z": TABLE_Z_MM}
            self.status = (f"Putting it down at "
                           f"({spot[0]:.0f}, {spot[1]:.0f})")
            self._do_place()
        finally:
            self.intruder_watch = False
            if self.state == State.TAKE_FROM_HAND:
                self.state = State.IDLE

    def _wait_for_clear_hand(self, point_mm, request_id):
        """The operator's hand is ON the spot they just pointed at. Wait for
        it to leave rather than planning around it: this is a short, blocking,
        straight-line move with no obstacle following."""
        self.intruder_watch = True
        self.status = "Move your hand away..."
        deadline = time.monotonic() + INSPECT_CLEAR_TIMEOUT_SEC
        clear_since = None
        while time.monotonic() < deadline:
            if self.abort_requested:
                self.status = "Stopped"
                self._publish_inspection(None, request_id, error="stopped")
                return False
            obstacle = self.current_obstacle()
            near = obstacle is not None and _capsule_distance(
                point_mm, obstacle) < obstacle[2] + 150.0
            if near:
                clear_since = None
            elif clear_since is None:
                clear_since = time.monotonic()
            elif time.monotonic() - clear_since > 0.4:
                self.intruder_watch = False
                return True
            time.sleep(0.1)
        self.status = "Hand stayed over the spot - inspection cancelled"
        self._publish_inspection(None, request_id,
                                 error="hand stayed over the spot")
        return False

    def can_change_target(self):
        """Changing the target mid-approach would leave the arm travelling to
        a hover pose computed for a different object."""
        return not self.busy and self.state in (
            State.IDLE, State.ERROR, State.DETECT, State.ARMED)

    def cycle_target(self, step=1):
        classes = list(self.yolo.class_names)
        if not classes:
            return
        index = (classes.index(self.target_name) + step) % len(classes) \
            if self.target_name in classes else 0
        self.set_target(classes[index])

    def set_target(self, name, quiet=False):
        """Switch which class the run will fetch, without restarting.

        Detection state is cleared so the new target is triangulated from
        scratch rather than inheriting the old one's fixes."""
        if name == self.target_name or name not in self.yolo.class_names:
            return
        if not self.can_change_target():
            if not quiet:
                self.status = f"Target locked while {self.state.value}"
            return
        self.target_name = name
        self.detect_fixes.clear()
        self.target_base = None
        self.hover_pose = None
        self.path = None
        self.target_stale_since = None
        if self.state in (State.DETECT, State.ARMED):
            # Already hunting: restart the search on the new class rather than
            # standing in ARMED with a plan aimed at the previous object.
            self.detect_started_at = time.monotonic()
            self.state = State.DETECT
        self.status = f"Target: {self.target_name}"
        print(f"TARGET: {self.target_name}")

    def set_speed_preset(self, index):
        if not 0 <= index < len(SPEED_PRESETS):
            return
        value, name = SPEED_PRESETS[index]
        self.speed_scale = value
        self.status = f"SPEED preset {index + 1}: {name} x{value:.2f}"
        print(f"SPEED x{value:.2f} ({name})")

    def set_fancy_hud(self, enabled):
        self.fancy_hud = bool(enabled)
        self.show_hud_cost = self.fancy_hud
        for canvas in self._hud.values():
            canvas.fancy = self.fancy_hud
        self._chrome.fancy = self.fancy_hud
        print(f"HUD: {'AR' if self.fancy_hud else 'plain'}")

    def shutdown(self):
        try:
            self._send_stop()
        except Exception:
            pass
        self.running = False
        self.rig.stop()
        self.camera_executor.shutdown()
        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="start with dry-run OFF (real motion)")
    parser.add_argument("--auto", action="store_true",
                        help="start the sequence immediately")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--place-base", default=None,
                        help="X,Y,Z base mm - place here instead of clicking")
    parser.add_argument("--stop-after-approach", action="store_true",
                        help="verification: stop at the hover point, no grasp")
    parser.add_argument("--fake-obstacle", action="store_true",
                        help="verification: inject a synthetic obstacle mid-path")
    parser.add_argument("--far-start", action="store_true",
                        help="begin at the far corner of the workspace for a "
                             "long, showy path - note it crosses the bench "
                             "diagonally to reach the other side")
    parser.add_argument("--free-wrist", action="store_true",
                        help="use all six axes while travelling: tool tilts "
                             "with the motion / away from the obstacle, and "
                             "returns to straight-down at the goal")
    parser.add_argument("--hand-place", action="store_true",
                        help="after grasping, deliver the object onto the "
                             "operator's tracked hand instead of a clicked "
                             "point (falls back to click if no hand appears)")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="scale every robot speed uniformly (demo dial, "
                             "e.g. 0.33 for a 3x slower, more dramatic run)")
    parser.add_argument("--storage-base", default=None,
                        help="X,Y,Z base mm - where remote 'tidy away' "
                             "requests place the object (default 260,-230,0)")
    parser.add_argument("--plain-hud", action="store_true",
                        help="draw the old flat overlay (no depth ribbon, "
                             "bloom or capsule shell). F toggles at runtime")
    parser.add_argument("--start-pose", default=None,
                        help="X,Y,Z base mm - begin the approach here instead "
                             "of at home (default: remembered pose, else "
                             f"{FAR_START_MM[0]:.0f},{FAR_START_MM[1]:.0f},"
                             f"{FAR_START_MM[2]:.0f}). P saves the live pose")
    parser.add_argument("--home-start", action="store_true",
                        help="(default) start the approach from home "
                             "(0,0,90,0,90,0) and go straight to the target")
    parser.add_argument("--grasp-lift", type=float,
                        default=GRASP_LIFT_DEFAULT_MM,
                        help="mm to raise every grasp above the computed "
                             "height (live: [ and ])")
    parser.add_argument("--inspect-vertical", action="store_true",
                        help="(default) photograph straight down, wrist "
                             "square to the bench (V toggles at runtime)")
    parser.add_argument("--inspect-oblique", action="store_true",
                        help="photograph along the finger's line of sight "
                             "instead of straight down")
    parser.add_argument("--no-point", action="store_true",
                        help="do not let a pointing finger change the target "
                             "(X toggles it at runtime)")
    parser.add_argument("--no-hold", action="store_true",
                        help="S runs straight through instead of parking at "
                             "the start until G (voice/UI starts always do)")
    known, ros_args = parser.parse_known_args()

    place_base = None
    if known.place_base:
        place_base = [float(v) for v in known.place_base.split(",")]
        if len(place_base) != 3:
            raise SystemExit("--place-base needs X,Y,Z")

    start_pose = None
    if known.start_pose:
        start_pose = [float(v) for v in known.start_pose.split(",")]
        if len(start_pose) != 3:
            raise SystemExit("--start-pose needs X,Y,Z")

    rclpy.init(args=ros_args)
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    dsr_node = rclpy.create_node("webcam_pnp_dsr", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    global get_current_posx, movej, movel, posj, posx
    global get_robot_state, set_robot_mode
    global ikin, get_current_solution_space
    global STATE_SAFE_STOP, STATE_SAFE_OFF, STATE_SAFE_OFF2
    global STATE_STANDBY, STATE_MOVING
    global CONTROL_RESET_SAFET_STOP, CONTROL_SERVO_ON, ROBOT_MODE_AUTONOMOUS
    try:
        from DSR_ROBOT2 import (
            get_current_posx, movej, movel, get_robot_state, set_robot_mode,
        )
        # Inverse kinematics: used to ask the controller whether a planned
        # inspection pose is actually holdable before commanding it. Optional
        # - older driver builds may not export it.
        try:
            from DSR_ROBOT2 import ikin, get_current_solution_space
        except ImportError:
            ikin = get_current_solution_space = None
        from DR_common2 import posj, posx
        from DRFC import (
            STATE_SAFE_STOP, STATE_SAFE_OFF, STATE_SAFE_OFF2,
            STATE_STANDBY, STATE_MOVING,
            CONTROL_RESET_SAFET_STOP, CONTROL_SERVO_ON, ROBOT_MODE_AUTONOMOUS,
        )
    except ImportError as error:
        print(f"Error importing Doosan robot API: {error}")
        dsr_node.destroy_node()
        rclpy.shutdown()
        return

    app = None
    try:
        cv2.namedWindow(WINDOW)
        app = WebcamPickPlace(
            target=known.target, live=known.live, auto=known.auto,
            place_base=place_base,
            stop_after_approach=known.stop_after_approach,
            fake_obstacle=known.fake_obstacle,
            far_start=known.far_start,
            free_wrist=known.free_wrist,
            hand_place=known.hand_place,
            speed_scale=known.speed_scale,
            fancy_hud=not known.plain_hud,
            start_pose=start_pose,
            home_start=known.home_start,
            grasp_lift=known.grasp_lift,
            hold_start=not known.no_hold,
            point_select=not known.no_point,
            # Straight down is the default: it is the pose the wrist holds
            # most rigidly, it never argues with a joint limit, and it is the
            # one that has never lost a shot. The oblique view is opt-in.
            inspect_oblique=known.inspect_oblique and not known.inspect_vertical,
        )
        if known.storage_base:
            app.storage_base = [float(v) for v in known.storage_base.split(",")]
        cv2.setMouseCallback(WINDOW, app.mouse_callback)
        print("=" * 72)
        print(f" Target: {known.target} | dry-run {'OFF - LIVE' if known.live else 'ON'}")
        if known.no_hold:
            print(" S: start (runs straight through)  SPACE: stop")
        else:
            print(" S: arm (go to start, keep replanning)  G: go  SPACE: stop")
        print(" R: recover  D: dry-run  H: home")
        print(" O/C: gripper open/close  F: overlay style  ESC: quit")
        print(" Speed: " + "  ".join(
            f"{index + 1}={name} x{value:.2f}"
            for index, (value, name) in enumerate(SPEED_PRESETS)))
        if app.start_pose is None:
            print(" Start: HOME (short path)")
        else:
            print(f" Start: {app.start_pose[0]:.0f},{app.start_pose[1]:.0f},"
                  f"{app.start_pose[2]:.0f} mm"
                  + ("  [remembered]" if START_POSE_FILE.is_file()
                     and not known.start_pose else "")
                  + "   P: save the arm's current position here")
        print(f" Grasp lift: {app.grasp_lift_mm:+.0f} mm   [ / ]: -/+ "
              f"{GRASP_LIFT_STEP_MM:.0f} mm")
        if app.point_select:
            print(" Point at a tool with your index finger to select it "
                  "(X turns it off)")
            print(" I: send the wrist camera to photograph the spot you are "
                  f"pointing at -> {INSPECT_DIR}")
            print(f" Aim plane: {app.aim_plane_mm:.0f} mm above the table "
                  f"( , / . move it by {AIM_PLANE_STEP_MM:.0f} mm, / resets)")
            print("   raise it to point at something standing UP - the side "
                  "of the conveyor, a part on a jig - not the table top")
            print(" M: take what is on your hand and put it down somewhere free")
            print(f" View: {'along your finger' if app.inspect_oblique else 'straight down'}"
                  "   V: switch")
        print(f" Target: {app.target_name}   T: next class ("
              + " > ".join(app.yolo.class_names) + ")")
        print("=" * 72)
        if known.auto:
            app.handle_start(hold=False)     # --auto means run, not wait

        last_logged = None
        last_render = 0.0
        last_status_beat = 0.0
        while rclpy.ok():
            if app.state == State.DETECT:
                app.detect_tick()
            elif app.state == State.ARMED:
                app.armed_tick()
            elif app.state == State.APPROACH:
                app.approach_tick()
            elif app.state == State.DELIVER_TRACK:
                app.deliver_tick()
            # Status lines are the only view into non-logging phases; echo
            # every change to the log so a headless run can be diagnosed.
            snapshot = f"[{app.state.value}] {app.status}"
            if snapshot != last_logged:
                app.get_logger().info(snapshot)
                app.publish_status(snapshot)
                last_logged = snapshot
                last_status_beat = time.monotonic()
            elif time.monotonic() - last_status_beat > 1.0:
                # Heartbeat re-publish: the skill bridge treats silence as a
                # dead robot app, and a long unchanged phase is not silence.
                app.publish_status(snapshot)
                last_status_beat = time.monotonic()
            # While the arm is streaming, the GUI is capped to ~15 fps so a
            # slow render can never starve the control tick: a rendering
            # hiccup longer than the SpeedL dead-man reads as a stutter in
            # the motion. Idle states render every loop as before.
            now_render = time.monotonic()
            streaming = app.state in (State.APPROACH, State.DELIVER_TRACK)
            if not streaming or now_render - last_render >= 0.066:
                app.render()
                last_render = now_render
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                if app.busy or app.state == State.APPROACH:
                    app.handle_abort()
                    print("Stopping motion; ESC again to quit.")
                    continue
                break
            if key == 32:
                app.handle_abort()
            elif key in (ord("s"), ord("S")):
                # Keyboard start always uses the launch-time destination;
                # remote overrides apply to their own run only.
                app.hand_place = app.default_hand_place
                app.place_base = app.default_place_base
                app.abort_requested = False
                app.handle_start()
            elif key in (ord("r"), ord("R")):
                app.handle_recover()
            elif key in (ord("d"), ord("D")) and not app.busy:
                app.dry_run = not app.dry_run
                print(f"DRY RUN: {'ON' if app.dry_run else 'OFF'}")
            elif key in (ord("h"), ord("H")):
                app.handle_home()
            elif key in (ord("o"), ord("O")):
                app.handle_gripper(close=False)
            elif key in (ord("c"), ord("C")):
                app.handle_gripper(close=True)
            elif key in (ord("f"), ord("F")):
                # Live A/B for the overlay - the two looks share call sites,
                # so this is just a flag flip on every canvas.
                app.set_fancy_hud(not app.fancy_hud)
            elif key in (ord("g"), ord("G")):
                app.handle_go()
            elif key in (ord("m"), ord("M")):
                # Take back what is on the operator's hand.
                app.handle_take_from_hand()
            elif key in (ord("i"), ord("I")):
                app.handle_inspect()
            elif key == ord(","):
                app.nudge_aim_plane(-AIM_PLANE_STEP_MM)
            elif key == ord("."):
                app.nudge_aim_plane(AIM_PLANE_STEP_MM)
            elif key == ord("/"):
                app.nudge_aim_plane(TABLE_Z_MM - app.aim_plane_mm)
            elif key == ord("0"):
                app.clear_target_mark()
                app.status = "Target spot cleared"
                print(app.status)
            elif key in (ord("-"), ord("_")):
                app.nudge_inspect_distance(+20.0)
            elif key in (ord("="), ord("+")):
                app.nudge_inspect_distance(-20.0)
            elif key in (ord("v"), ord("V")):
                app.inspect_oblique = not app.inspect_oblique
                app.status = ("Inspect view: along your finger"
                              if app.inspect_oblique else
                              "Inspect view: straight down")
                print(app.status)
            elif key in (ord("x"), ord("X")):
                app.point_select = not app.point_select
                if not app.point_select:
                    with app.webcam_lock:
                        app.pointing_ray = None
                        app.pointing_choice = None
                        app.pointing_scores = {}
                app.status = ("Point-to-select ON" if app.point_select
                              else "Point-to-select OFF")
                print(app.status)
            elif key in (ord("t"), ord("T")):
                # Shift+T steps backwards through the classes.
                app.cycle_target(-1 if key == ord("T") else 1)
            elif key in (ord("p"), ord("P")) and not app.busy:
                # Drive the arm where the detour looks best, then keep it.
                app.capture_start_pose()
            elif key == ord("["):
                app.nudge_grasp_lift(-GRASP_LIFT_STEP_MM)
            elif key == ord("]"):
                app.nudge_grasp_lift(GRASP_LIFT_STEP_MM)
            elif ord("1") <= key <= ord("0") + len(SPEED_PRESETS):
                # Demo presets. Take effect immediately, even mid-approach;
                # obstacle reactions stay floored at 0.85 regardless.
                app.set_speed_preset(key - ord("1"))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if app is not None:
            app.shutdown()
        dsr_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
