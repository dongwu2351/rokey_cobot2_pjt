"""cuRobo orbit planning for the tracker.

Kept out of robot_control.py and imported lazily, so the tracker still starts
on a machine without cuRobo - the orbit key just reports that it is missing.

Everything here is in METRES and RADIANS (cuRobo's units). The tracker works
in millimetres and the Doosan API in degrees, so the caller converts at the
boundary. Getting that wrong is silent and produces a 1000x goal.
"""
import numpy as np

# Camera standoff and the arc it sweeps. Measured feasible on this arm:
# orbiting on the base side at 400mm keeps every viewpoint reachable, the
# wrist branch unchanged, and viewpoint-to-viewpoint travel at 22-30 deg.
# Orbiting on the far side (+X) fails almost everywhere - the flange ends up
# outside the 900mm envelope even when the TCP looks comfortably inside.
ORBIT_AZIMUTH_DEG = 180.0
# Roll about the optical axis only spins the picture, so it is free to pick -
# but it must be sampled finely enough to land near the previous viewpoint's
# roll. At 45deg steps J6 swung 50-80deg between viewpoints purely from the
# sampling; at 15deg that disappears.
ROLLS_DEG = tuple(range(0, 360, 15))

# BENCHMARK.md: reach is a flange property. The TCP sits 238mm further along
# the tool, so a TCP well inside the envelope can hang off a flange that is not.
TCP_AHEAD_OF_FLANGE_M = 0.238
REACH_LIMIT_M = 0.900


def look_at_rotation(camera_pos, target_pos, roll_deg):
    """Rotation whose +Z (optical axis) points from the camera at the target,
    spun by roll_deg about that axis."""
    forward = np.asarray(target_pos, dtype=float) - np.asarray(camera_pos, dtype=float)
    forward = forward / np.linalg.norm(forward)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, forward))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    right = np.cross(seed, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    basis = np.column_stack([right, up, forward])
    angle = np.radians(roll_deg)
    spin = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return basis @ spin


def matrix_to_quat(matrix):
    """[w, x, y, z] from a rotation matrix (cuRobo's order)."""
    trace = np.trace(matrix)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array([0.25 * s,
                         (matrix[2, 1] - matrix[1, 2]) / s,
                         (matrix[0, 2] - matrix[2, 0]) / s,
                         (matrix[1, 0] - matrix[0, 1]) / s])
    if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        return np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                         (matrix[0, 1] + matrix[1, 0]) / s,
                         (matrix[0, 2] + matrix[2, 0]) / s])
    if matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        return np.array([(matrix[0, 2] - matrix[2, 0]) / s,
                         (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                         (matrix[1, 2] + matrix[2, 1]) / s])
    s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
    return np.array([(matrix[1, 0] - matrix[0, 1]) / s,
                     (matrix[0, 2] + matrix[2, 0]) / s,
                     (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])


def resample(traj, n):
    src, dst = np.linspace(0, 1, len(traj)), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, traj[:, j]) for j in range(traj.shape[1])], 1)


class OrbitPlanner:
    """Owns the cuRobo instance. Build once, plan many times.

    use_cuda_graph is OFF on purpose. The tracker runs YOLO on the GPU from a
    separate thread the whole time, and CUDA graph capture breaks if any other
    thread touches CUDA during it - down to tensor creation. step2_stream.py,
    the script that was validated on this arm, disables it for the same reason.
    """

    ROBOT_CFG = "/home/rokey/cobot2_ws/dum_E_project/tools/m0609.yml"
    TOOL = "tcp"
    JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
    # The table, as a finite box. Treating the floor as an infinite z=0 plane
    # reports the robot's own base_link spheres (they reach to z=-0.095, at
    # x~0, nowhere near the table) as digging in by 66mm.
    TABLE_CENTRE = np.array([0.55, 0.0, -0.025])
    TABLE_HALF = np.array([0.40, 0.50, 0.025])

    def __init__(self, gripper2cam_m, hz):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Cuboid, Scene
        from curobo.types import JointState

        self._torch = torch
        self._JointState = JointState
        self.cam2gripper = np.linalg.inv(np.asarray(gripper2cam_m, dtype=float))

        floor = Cuboid(name="table", dims=[0.8, 1.0, 0.05],
                       pose=[0.55, 0.0, -0.025, 1, 0, 0, 0])
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=self.ROBOT_CFG, scene_model=Scene(cuboid=[floor]),
            collision_cache={"cuboid": 32, "mesh": 8},
            # All the rolls for one viewpoint go in as a goal SET, so cuRobo
            # picks a reachable one itself. Planning them one at a time meant
            # 4 viewpoints x 24 rolls = 96 calls, which with CUDA graphs off
            # and YOLO sharing the GPU took minutes.
            max_goalset=len(ROLLS_DEG),
            use_cuda_graph=False, interpolation_dt=1.0 / hz))
        self.planner.warmup(enable_graph=True)

    def _js(self, q):
        t = self._torch.tensor(np.atleast_2d(q), dtype=self._torch.float32,
                               device="cuda")
        return self._JointState.from_position(t, joint_names=self.JOINT_NAMES)

    def _plan_to(self, q_from, base2tcp_list):
        """Plan to whichever of these TCP poses cuRobo can reach.

        They are all the same viewpoint at different rolls about the optical
        axis, and roll only spins the picture - so any of them will do. Handing
        the whole set to one call lets cuRobo choose, instead of us paying for
        a full plan per candidate."""
        from curobo.types import GoalToolPose
        torch = self._torch
        poses = list(base2tcp_list)
        position = np.stack([b[:3, 3] for b in poses]).reshape(1, 1, 1, len(poses), 3)
        quaternion = np.stack(
            [matrix_to_quat(b[:3, :3]) for b in poses]
        ).reshape(1, 1, 1, len(poses), 4)
        result = self.planner.plan_pose(
            GoalToolPose(
                tool_frames=[self.TOOL],
                position=torch.tensor(position, dtype=torch.float32, device="cuda"),
                quaternion=torch.tensor(quaternion, dtype=torch.float32,
                                        device="cuda"),
            ),
            self._js(q_from), max_attempts=5, enable_graph_attempt=1)
        if result is None or not bool(result.success.flatten()[0]):
            return None
        traj = np.atleast_2d(
            result.get_interpolated_plan().position.squeeze().detach().cpu().numpy())
        # Never assume which end is the start: plan_cspace hands trajectories
        # back reversed, and streaming that makes the first command a ~500deg/s
        # jump, which is a protective stop.
        if np.abs(traj[-1] - q_from).max() < np.abs(traj[0] - q_from).max():
            traj = traj[::-1].copy()
        return traj

    def clearance_m(self, traj):
        """Smallest gap from any robot sphere to the table, over the path.
        Negative means it digs in. success=True does not imply this is
        positive - only a swept check does."""
        state = self.planner.compute_kinematics(self._js(traj))
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(len(traj), -1, 4)
        centres, radii = spheres[..., :3], spheres[..., 3]
        delta = np.abs(centres - self.TABLE_CENTRE) - self.TABLE_HALF
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
        inside = np.minimum(delta.max(axis=-1), 0.0)
        gap = outside + inside - radii
        gap[radii <= 0] = np.inf
        return float(gap.min())

    def viewpoint_tcp(self, object_m, radius_m, elevation_deg, roll_deg):
        e, a = np.radians(elevation_deg), np.radians(ORBIT_AZIMUTH_DEG)
        camera = np.asarray(object_m, dtype=float) + radius_m * np.array(
            [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        base2cam = np.eye(4)
        base2cam[:3, :3] = look_at_rotation(camera, object_m, roll_deg)
        base2cam[:3, 3] = camera
        return base2cam @ self.cam2gripper, camera

    def plan_orbit(self, start_q_rad, object_m, radius_m, elevations_deg,
                   box_min_m, box_max_m, log=None):
        """Chain the viewpoints from the arm's current pose.

        Chained, not each-from-home: what an orbit costs is the
        viewpoint-to-viewpoint step, and planning each leg from home hides
        exactly the reconfigurations that matter.

        Returns (trajectory, notes) or (None, reason).
        """
        notes, q = [], np.asarray(start_q_rad, dtype=float)
        legs = []
        for elevation in elevations_deg:
            # Screen the rolls with cheap geometry first - workspace box and
            # flange reach - so the planner is only ever asked about candidates
            # that could work at all.
            candidates, camera = [], None
            for roll in ROLLS_DEG:
                base2tcp, cam = self.viewpoint_tcp(
                    object_m, radius_m, elevation, roll)
                target = base2tcp[:3, 3]
                if np.any(target < box_min_m) or np.any(target > box_max_m):
                    continue
                flange = target - TCP_AHEAD_OF_FLANGE_M * base2tcp[:3, 2]
                if float(np.linalg.norm(flange)) > REACH_LIMIT_M:
                    continue
                candidates.append(base2tcp)
                camera = cam   # same point for every roll; only the spin differs
            if not candidates:
                return None, f"no viewpoint within reach at {elevation:.0f} deg"

            traj = self._plan_to(q, candidates)
            if traj is None:
                return None, f"no path to the {elevation:.0f} deg viewpoint"

            legs.append(traj)
            swing = float(np.abs(np.degrees(traj[-1] - traj[0])).max())
            note = (f"elev {elevation:.0f}deg {len(candidates)} rolls offered, "
                    f"swing {swing:.1f}deg cam {np.round(camera, 3)}")
            notes.append(note)
            if log:
                log(note)
            q = traj[-1]
        return legs, notes
