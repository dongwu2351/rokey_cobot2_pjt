"""Fixed webcam triad: capture, calibration, triangulation, intruder detection.

Three Logitech C270s look at the M0609 workspace from fixed mounts, calibrated
eye-to-hand (see ~/webcam_calibration). This module owns everything about them:

    WebcamRig     device discovery by stable USB path, capture threads,
                  intrinsics/extrinsics, ray casting, N-view triangulation,
                  projection for overlays, background-diff intruder detection
                  with the robot arm masked out
    KalmanCV      constant-velocity Kalman filter for a moving obstacle
                  (taken from dum_E_project/tools/obstacle_tracker.py)

Everything here works in METERS in the robot base frame, matching the
calibration bundle. Callers that live in Doosan millimetres convert at the
boundary.

The base frame is the CONTROLLER's base frame (the extrinsics were solved
against `posx` flange poses), which for masking and triangulation purposes is
treated as identical to the URDF base_link - verified by projecting the live
TCP into each camera and checking it lands on the gripper.
"""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_CALIB_ROOT = Path.home() / "webcam_calibration"
CAM_NAMES = ("cam0", "cam1", "cam2")
FRAME_SIZE = (1280, 720)
# The C270 delivers 720p only as MJPG; raw YUYV tops out far lower.
FOURCC = cv2.VideoWriter_fourcc(*"MJPG")

# --- intruder detection tuning -------------------------------------------
# absdiff threshold on grayscale. High enough to ignore soft shadows; an arm
# or a tool is far brighter/darker than its background.
DIFF_THRESHOLD = 40
BLUR_KERNEL = 5
MORPH_KERNEL = 7
MIN_BLOB_AREA_PX = 1200  # smaller foreground blobs are noise/shadow scraps
MAX_BLOBS_PER_CAM = 3
# Radius of the cylinder swept around each robot-arm segment when masking the
# arm out of the foreground. Generous on purpose: better to hide a strip of
# real obstacle next to the arm than to chase the arm as a phantom obstacle.
ARM_MASK_RADIUS_M = 0.13
# Rays from two cameras must pass within this of each other to count as the
# same physical object.
RAY_MATCH_TOLERANCE_M = 0.10
# Obstacle radius estimated from blob size, clamped to something sane for a
# hand before margins are added on top. Deliberately tight: the operator
# wants the arm to skim past a hand, not flee it - the planner's own 2 cm
# margin is the actual clearance.
OBSTACLE_RADIUS_MIN_M = 0.04
OBSTACLE_RADIUS_MAX_M = 0.13


class KalmanCV:
    """Constant-velocity Kalman filter. State = [x,y,z,vx,vy,vz].

    Observations are noisy positions only; velocity comes out of the filter.
    That velocity is what lets a planner predict where the obstacle will be
    instead of reacting to where it was."""

    def __init__(self, p0, q_acc=0.15, r_pos=0.015):
        self.x = np.r_[np.asarray(p0, float), np.zeros(3)]
        self.P = np.diag([0.02] * 3 + [0.05] * 3)
        self.q_acc = q_acc
        self.R = np.eye(3) * r_pos ** 2

    def predict(self, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.q_acc ** 2
        G = np.r_[np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt]
        Q = (G @ G.T) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        y = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    @property
    def pos(self):
        return self.x[:3].copy()

    @property
    def vel(self):
        return self.x[3:].copy()


class _Camera:
    """One webcam: calibration + capture thread holding the freshest frame."""

    def __init__(self, name, device_path, camera_matrix, distortion, T_base_cam):
        self.name = name
        self.device_path = device_path
        self.K = np.asarray(camera_matrix, dtype=float)
        self.dist = np.asarray(distortion, dtype=float)
        self.T_base_cam = np.asarray(T_base_cam, dtype=float)
        self.T_cam_base = np.linalg.inv(self.T_base_cam)
        # projectPoints wants rvec/tvec of the base->cam mapping.
        self._rvec, _ = cv2.Rodrigues(self.T_cam_base[:3, :3])
        self._tvec = self.T_cam_base[:3, 3].reshape(3, 1)

        self._capture = None
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._running = False
        self._thread = None
        self.frame_failures = 0

    def start(self):
        capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"{self.name}: cannot open {self.device_path}")
        capture.set(cv2.CAP_PROP_FOURCC, FOURCC)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (int(width), int(height)) != FRAME_SIZE:
            capture.release()
            raise RuntimeError(
                f"{self.name}: got {int(width)}x{int(height)}, "
                f"needs {FRAME_SIZE[0]}x{FRAME_SIZE[1]} (calibration resolution)"
            )
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"cap-{self.name}", daemon=True
        )
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                self.frame_failures += 1
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()

    def latest(self):
        with self._lock:
            return self._frame, self._stamp

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()

    # -- geometry -----------------------------------------------------
    def ray(self, u, v):
        """(origin, direction) of the viewing ray in base frame, meters."""
        undistorted = cv2.undistortPoints(
            np.array([[[float(u), float(v)]]]), self.K, self.dist
        )[0, 0]
        direction_cam = np.array([undistorted[0], undistorted[1], 1.0])
        direction = self.T_base_cam[:3, :3] @ direction_cam
        norm = np.linalg.norm(direction)
        return self.T_base_cam[:3, 3].copy(), direction / norm

    def project(self, points_base):
        """Base-frame points (meters, Nx3) -> pixel coords (Nx2) or None each
        if behind the camera."""
        points = np.atleast_2d(np.asarray(points_base, dtype=float))
        in_cam = (self.T_cam_base[:3, :3] @ points.T).T + self.T_cam_base[:3, 3]
        pixels, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3), self._rvec, self._tvec, self.K, self.dist
        )
        pixels = pixels.reshape(-1, 2)
        result = []
        for pixel, cam_point in zip(pixels, in_cam):
            if cam_point[2] <= 0.05 or not np.isfinite(pixel).all():
                result.append(None)
            else:
                result.append((float(pixel[0]), float(pixel[1])))
        return result

    def depth_of(self, point_base):
        """Distance along the optical axis from this camera to a base point."""
        cam_point = self.T_cam_base[:3, :3] @ np.asarray(point_base) + self.T_cam_base[:3, 3]
        return float(cam_point[2])


def _load_transforms(calib_root):
    path = Path(calib_root) / "results" / "transforms" / "production_transforms.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data.get("translation_unit") != "m":
        raise RuntimeError(f"{path}: expected translation_unit m")
    return {
        name: np.asarray(entry["matrix"], dtype=float)
        for name, entry in data["transforms"].items()
    }


def _load_device_map(calib_root):
    path = Path(calib_root) / "config" / "camera_device_map.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    frame_to_cam = {
        f"{name}_optical": name for name in CAM_NAMES
    }
    devices = {}
    for entry in data.get("fixed_external_candidates", []):
        cam = frame_to_cam.get(entry.get("assigned_frame"))
        if cam:
            devices[cam] = entry["stable_device_path"]
    missing = [name for name in CAM_NAMES if name not in devices]
    if missing:
        raise RuntimeError(f"{path}: no device path for {missing}")
    return devices


def _load_intrinsics(calib_root, cam):
    path = Path(calib_root) / "results" / "intrinsics" / cam / "intrinsics.json"
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["camera_matrix"], data["distortion"]


class WebcamRig:
    def __init__(self, calib_root=DEFAULT_CALIB_ROOT):
        calib_root = Path(calib_root)
        transforms = _load_transforms(calib_root)
        devices = _load_device_map(calib_root)
        self.cameras = {}
        for name in CAM_NAMES:
            camera_matrix, distortion = _load_intrinsics(calib_root, name)
            self.cameras[name] = _Camera(
                name,
                devices[name],
                camera_matrix,
                distortion,
                transforms[f"T_base_{name}"],
            )
        self._background = {}
        # Arm skeleton at the moment the background was captured. The parked
        # arm is part of the reference image, so once it moves away it leaves
        # a "ghost" difference where it used to be - mask that silhouette
        # permanently, alongside the live arm.
        self.ghost_points = None

    def start(self):
        for camera in self.cameras.values():
            camera.start()

    def stop(self):
        for camera in self.cameras.values():
            camera.stop()

    def frames(self):
        """{cam: (frame, stamp)} for cams that have delivered a frame."""
        out = {}
        for name, camera in self.cameras.items():
            frame, stamp = camera.latest()
            if frame is not None:
                out[name] = (frame, stamp)
        return out

    # -- triangulation ------------------------------------------------
    def triangulate(self, pixel_by_cam):
        """Least-squares intersection of the viewing rays through the given
        pixels, >= 2 cameras.

        Returns (point_base_m, worst_gap_m): the point minimising the summed
        squared distance to every ray, and the largest distance from it to any
        contributing ray - the honesty metric. A big gap means the pixels do
        not look at the same physical point (mismatched detections), and the
        caller must reject the fix rather than average nonsense."""
        rays = [
            self.cameras[cam].ray(u, v) for cam, (u, v) in pixel_by_cam.items()
        ]
        if len(rays) < 2:
            return None, None
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for origin, direction in rays:
            projector = np.eye(3) - np.outer(direction, direction)
            A += projector
            b += projector @ origin
        point = np.linalg.lstsq(A, b, rcond=None)[0]
        worst = 0.0
        for origin, direction in rays:
            offset = point - origin
            perpendicular = offset - direction * float(np.dot(offset, direction))
            worst = max(worst, float(np.linalg.norm(perpendicular)))
        return point, worst

    # -- background / intruders ---------------------------------------
    def capture_background(self):
        """Reference frame per camera. Whatever is in the scene now - table,
        hammer, the robot arm at its current pose - becomes 'background';
        only what ENTERS afterwards shows as foreground. The arm moving later
        is handled by the arm mask, not by the reference."""
        for name, camera in self.cameras.items():
            frame, _ = camera.latest()
            if frame is None:
                raise RuntimeError(f"{name}: no frame to capture background from")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._background[name] = cv2.GaussianBlur(
                gray, (BLUR_KERNEL, BLUR_KERNEL), 0
            )

    @property
    def has_background(self):
        return len(self._background) == len(self.cameras)

    def _arm_mask(self, camera, shape, arm_points_base):
        """Boolean mask (True = arm) from projecting the arm skeleton."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if arm_points_base is None or len(arm_points_base) < 2:
            return mask
        pixels = camera.project(arm_points_base)
        fx = camera.K[0, 0]
        for i in range(len(arm_points_base) - 1):
            a, b = pixels[i], pixels[i + 1]
            if a is None or b is None:
                continue
            depth = max(
                0.3,
                (camera.depth_of(arm_points_base[i])
                 + camera.depth_of(arm_points_base[i + 1])) / 2.0,
            )
            thickness = max(8, int(round(2.0 * ARM_MASK_RADIUS_M * fx / depth)))
            cv2.line(
                mask,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                255,
                thickness,
            )
        return mask

    def detect_intruders(self, arm_points_base=None):
        """Foreground blobs per camera, arm masked out.

        Returns {cam: {"blobs": [(u, v, area_px, radius_px)], "mask": mask,
        "arm_mask": arm}} for every camera with a background reference. Blobs
        are sorted largest first."""
        results = {}
        for name, camera in self.cameras.items():
            background = self._background.get(name)
            frame, _ = camera.latest()
            if background is None or frame is None:
                continue
            gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (BLUR_KERNEL, BLUR_KERNEL),
                0,
            )
            diff = cv2.absdiff(gray, background)
            _, fg = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
            )
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

            arm = self._arm_mask(camera, frame.shape, arm_points_base)
            if self.ghost_points is not None:
                ghost = self._arm_mask(camera, frame.shape, self.ghost_points)
                arm = cv2.bitwise_or(arm, ghost)
            fg[arm > 0] = 0

            contours, _ = cv2.findContours(
                fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            blobs = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_BLOB_AREA_PX:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                u = moments["m10"] / moments["m00"]
                v = moments["m01"] / moments["m00"]
                radius_px = float(np.sqrt(area / np.pi))
                blobs.append((u, v, area, radius_px))
            blobs.sort(key=lambda blob: blob[2], reverse=True)
            results[name] = {
                "blobs": blobs[:MAX_BLOBS_PER_CAM],
                "mask": fg,
                "arm_mask": arm,
            }
        return results

    def locate_obstacle(self, intruders):
        """One obstacle sphere from the per-camera blobs, or None.

        Takes the largest blob from every camera that has one and finds the
        subset whose rays agree. Two cameras minimum: a single camera cannot
        place anything in 3D, and a blob only one camera sees (the robot seen
        from an angle the mask missed, a shadow) is exactly what must NOT
        become a phantom obstacle.

        Returns (center_m, radius_m, cams_used)."""
        candidates = {
            cam: data["blobs"][0]
            for cam, data in intruders.items()
            if data["blobs"]
        }
        if len(candidates) < 2:
            return None

        best = None
        cams = list(candidates)
        # Try every pair, keep the tightest agreement, then fold in a third
        # camera if its ray also passes near the pair's point.
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                pair = {
                    cams[i]: candidates[cams[i]][:2],
                    cams[j]: candidates[cams[j]][:2],
                }
                point, gap = self.triangulate(pair)
                if point is None or gap > RAY_MATCH_TOLERANCE_M:
                    continue
                if best is None or gap < best[1]:
                    best = (point, gap, [cams[i], cams[j]])
        if best is None:
            return None

        point, _, used = best
        for cam in cams:
            if cam in used:
                continue
            trial = {name: candidates[name][:2] for name in used + [cam]}
            refined, gap = self.triangulate(trial)
            if refined is not None and gap <= RAY_MATCH_TOLERANCE_M:
                point, used = refined, used + [cam]

        radii = []
        for cam in used:
            camera = self.cameras[cam]
            depth = camera.depth_of(point)
            if depth <= 0.05:
                continue
            radius_px = candidates[cam][3]
            radii.append(radius_px * depth / camera.K[0, 0])
        # Mean, not max: each camera sees the hand's silhouette from its own
        # angle and the widest one (fingers spread toward that camera) is an
        # overestimate of the solid hand.
        radius = float(np.clip(
            float(np.mean(radii)) if radii else OBSTACLE_RADIUS_MIN_M,
            OBSTACLE_RADIUS_MIN_M,
            OBSTACLE_RADIUS_MAX_M,
        ))

        # Second end of the capsule: the forearm points from the same
        # cameras. An extrapolated point is noisier than a landmark
        # centroid, so the gate is looser, and on failure the capsule
        # degenerates to the hand sphere.
        forearm = point
        if all(len(candidates[cam]) >= 6 for cam in used):
            trial = {cam: candidates[cam][4:6] for cam in used}
            refined, gap = self.triangulate(trial)
            if refined is not None and gap <= RAY_MATCH_TOLERANCE_M * 1.6:
                forearm = refined
        return point, forearm, radius, used


class HandIntruderDetector:
    """Hands as the one kind of obstacle, via MediaPipe.

    Replaces the background-diff detector: no reference frame to manage, no
    ghost of the parked arm, and - decisively - the robot arm itself can
    never register as an intruder, because only things that look like a
    human hand do.

    One Hands instance per camera: the tracker carries state between frames
    of ONE video stream, and feeding it three interleaved viewpoints would
    make it 'track' a hand teleporting between them.

    Output matches detect_intruders' blob format, so locate_obstacle
    consumes it unchanged: {cam: {"blobs": [(u, v, area_px, radius_px)]}}.
    """

    # The landmark span already covers palm plus spread fingers, which
    # overstates the solid part of a hand; no extra inflation on top of it.
    # (Was 1.5 to cover the forearm too - measured too timid in practice:
    # the sphere grew until the arm stopped for a hand a spa away.)
    RADIUS_INFLATION = 1.0
    # Detection runs on a downscaled copy: at 1.3 m a hand still spans
    # ~90 px of a 640-wide frame, plenty for MediaPipe, and a third of the
    # full-resolution cost. Landmarks come back normalised, so coordinates
    # scale straight back to the full frame.
    DETECT_WIDTH = 640

    def __init__(self, camera_names):
        import mediapipe as mp
        self._hands = {
            name: mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.35,
                # Loose on purpose: a fast-moving hand motion-blurs on a
                # C270 and the tracker drops it for several frames at 0.4.
                min_tracking_confidence=0.25,
            )
            for name in camera_names
        }

    # How far past the wrist the forearm reference point sits, in units of
    # the hand's own landmark span. The hand sphere alone leaves the
    # OUTSTRETCHED FOREARM unmodelled - the arm then dodges a fist-sized
    # ball at the palm and skims along the arm behind it.
    FOREARM_REACH = 1.6

    #: Index finger landmarks: knuckle and tip. The line between them is the
    #: direction a person means when they point - the fingertip alone gives a
    #: position but no direction, and the wrist-to-tip line is dragged off
    #: target by how the hand happens to be rotated.
    INDEX_MCP, INDEX_TIP = 5, 8
    MIDDLE_TIP, RING_TIP = 12, 16
    #: A pointing hand has the index clearly more extended than the fingers
    #: curled beside it. Without this test an open palm "points" at whatever
    #: happens to lie along the index, and objects get selected by accident.
    POINT_EXTENSION_RATIO = 1.25

    def detect(self, frames_by_cam):
        """{cam: {"blobs": [...], "pointers": [...]}} on full-frame pixels.

        `blobs` is (u, v, area_px, radius_px, fu, fv), largest hand first;
        (fu, fv) is a point projected down the forearm behind the wrist, so
        the caller can build a hand-to-forearm capsule instead of a lone
        sphere.

        `pointers` is (mcp_u, mcp_v, tip_u, tip_v, is_pointing) for the same
        hands in the same order - two points on the index finger, which two
        cameras turn into a 3D ray (see pointing.py)."""
        results = {}
        for name, frame in frames_by_cam.items():
            hands = self._hands.get(name)
            if hands is None or frame is None:
                continue
            height, width = frame.shape[:2]
            scale = self.DETECT_WIDTH / float(width)
            small = cv2.resize(
                frame, (self.DETECT_WIDTH, int(round(height * scale)))
            )
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            processed = hands.process(rgb)
            blobs = []
            pointers = []
            for landmarks in (processed.multi_hand_landmarks or []):
                xs = np.array([lm.x for lm in landmarks.landmark]) * width
                ys = np.array([lm.y for lm in landmarks.landmark]) * height
                u, v = float(xs.mean()), float(ys.mean())
                span = max(xs.max() - xs.min(), ys.max() - ys.min())
                radius_px = float(span / 2.0 * self.RADIUS_INFLATION)
                # Forearm direction: from the middle-finger knuckle (9)
                # through the wrist (0), extended past the wrist.
                wrist = np.array([xs[0], ys[0]])
                knuckle = np.array([xs[9], ys[9]])
                direction = wrist - knuckle
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    forearm = wrist + direction / norm * span * self.FOREARM_REACH
                else:
                    forearm = wrist
                blobs.append((
                    u, v, np.pi * radius_px ** 2, radius_px,
                    float(forearm[0]), float(forearm[1]),
                ))

                # Pointing: index extended past the neighbouring fingertips.
                # Measured from the wrist so it does not depend on how far
                # away the hand is.
                def reach(index):
                    return float(np.hypot(xs[index] - xs[0], ys[index] - ys[0]))
                index_reach = reach(self.INDEX_TIP)
                folded = max(reach(self.MIDDLE_TIP), reach(self.RING_TIP))
                pointers.append((
                    float(xs[self.INDEX_MCP]), float(ys[self.INDEX_MCP]),
                    float(xs[self.INDEX_TIP]), float(ys[self.INDEX_TIP]),
                    bool(index_reach > folded * self.POINT_EXTENSION_RATIO),
                ))
            if blobs:
                order = sorted(range(len(blobs)),
                               key=lambda i: blobs[i][2], reverse=True)
                results[name] = {
                    "blobs": [blobs[i] for i in order],
                    "pointers": [pointers[i] for i in order],
                }
        return results

    def close(self):
        for hands in self._hands.values():
            hands.close()
