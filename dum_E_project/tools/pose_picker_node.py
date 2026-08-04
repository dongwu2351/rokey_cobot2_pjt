#!/usr/bin/env python3
"""시작·목표를 RViz 에서 직접 집어서 경로를 확인하는 도구.

RViz 에 마커 두 개가 뜬다. 화살표를 끌면 위치가, 링을 돌리면 자세가 바뀐다.

    파랑 = 시작 (로봇이 그 자세로 즉시 이동해서 보여준다)
    초록 = 목표

마커를 놓는 순간 다시 계획해서 경로를 그리고, 아래 세 가지를 판정해 준다.
이 셋이 지금까지 우리를 물먹인 것들이라 자동으로 확인하게 만들어 뒀다.

  1. 도달 가능한가 — TCP 좌표가 아니라 **플랜지** 기준으로 따진다.
     TCP 는 플랜지에서 238mm 앞이라 TCP(0.70,0.12) 는 리치 안처럼 보여도
     플랜지 기준 0.786m 로 한계 근접이다.

  2. 손목 분기가 같은가 — 시작과 목표의 J5 부호가 다르면 손목이 180도 뒤집히면서
     TCP 가 큰 원호를 그린다. 경로가 직선거리의 4.7배까지 늘어난 원인이 이거였다.
     seed 를 48개로 늘려도, 그래프 플래너를 꺼도 해결되지 않는다. 애초에 그 지점에
     같은 분기의 IK 해가 없기 때문이다. **사람 옆에서는 위험하므로 미리 걸러야 한다.**

  3. 우회 배율 — 경로길이 / 직선거리. 1~2배가 정상, 3배 넘으면 뭔가 잘못된 것.

토픽
    /curobo/play        (Empty)  경로를 따라 로봇을 한 번 움직여 본다
    /curobo/reset       (Empty)  시작 자세로 되돌린다
    /curobo/swap        (Empty)  시작 <-> 목표 맞바꾸기
    /curobo/set_start   (Point)  시작을 좌표로 직접 지정 (마커 대신 숫자로)
    /curobo/set_goal    (Point)  목표를 좌표로 직접 지정

실행:  ros2 launch pose_picker.launch.py
"""
import threading
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, PoseStamped
from interactive_markers import InteractiveMarkerServer
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Empty
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    Marker, MarkerArray)

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, Sphere
from curobo.types import GoalToolPose, JointState, Pose

from curobo_hybrid_demo import (CACHE, GOALS_XYZ, MOVER_BOX, MOVER_SPD, MOVER_X, MOVER_Z,
                                OBS_NOISE, PERCEPTION_HZ, PREDICT_MARGIN, PREDICT_T,
                                R_MOVER, js)
from curobo_rviz_demo import BASE, CFG, HOME, JOINTS, R_OBS, STATIC_OBS, TOOL, floor_cuboids
from obstacle_tracker import KalmanCV, RandomMover

REACH = 0.90          # M0609 사양 리치 [m] (플랜지 기준)
REACH_WARN = 0.75     # 이 이상이면 한계 근접 경고
DETOUR_WARN = 3.0     # 우회 배율 경고선
ANIM_HZ = 30.0


class PosePicker(Node):
    def __init__(self):
        super().__init__("pose_picker")
        self.declare_parameter("n_static", 1)
        self.declare_parameter("mover_y", 0.0)
        self.declare_parameter("rate", 2.0)
        # 기본은 움직인다 — 장애물이 지나갈 때마다 경로가 어떻게 휘는지 보는 게
        # 이 도구의 주 용도다. 같은 시작·목표 조합을 '비교' 하고 싶을 때만
        # moving:=false 로 고정한다 (움직이면 매번 답이 달라져 비교가 안 된다).
        self.declare_parameter("moving", True)
        n = int(self.get_parameter("n_static").value)
        self.static_obs = STATIC_OBS[:max(0, min(n, len(STATIC_OBS)))]
        self.mover_y = float(self.get_parameter("mover_y").value)
        rate = float(self.get_parameter("rate").value)
        self.moving = bool(self.get_parameter("moving").value)

        # 움직일 때는 하이브리드 데모와 같은 경로를 쓴다:
        #   무작위 이동 -> 노이즈 낀 관측 -> 칼만 -> PREDICT_T 초 앞 예측
        # 실기에서 카메라+YOLO 가 들어올 자리가 observe() 하나뿐이도록 맞춰 둔 것이다.
        self.mover = RandomMover(center=[MOVER_X, self.mover_y, MOVER_Z],
                                 box=MOVER_BOX, speed=MOVER_SPD, seed=0)
        self.kf = None
        self.t0 = time.time()

        self.lock = threading.Lock()
        self.start_xyz = np.array(GOALS_XYZ[0], dtype=float)
        self.goal_xyz = np.array(GOALS_XYZ[1], dtype=float)
        self.start_quat = None      # 홈 자세의 툴 방향으로 초기화한다
        self.goal_quat = None
        self.dirty = True
        self.start_dirty = True
        self.q_start = HOME.copy()
        self.traj = None
        self.play_i = None          # 재생 중이면 인덱스

        self.get_logger().info("cuRobo 로딩...")
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=CFG, scene_model=self._scene(), collision_cache=CACHE,
            use_cuda_graph=True, interpolation_dt=0.02))
        self.planner.warmup(enable_graph=True)
        home_q = self.planner.compute_kinematics(js(HOME)) \
            .tool_poses.quaternion.reshape(4).detach().cpu().numpy()
        self.start_quat = home_q.copy()   # [w,x,y,z]
        self.goal_quat = home_q.copy()

        self.pub_js = self.create_publisher(JointStateMsg, "/joint_states", 10)
        self.pub_path = self.create_publisher(Path, "/curobo/global_path", 1)
        self.pub_obs = self.create_publisher(MarkerArray, "/curobo/obstacles", 1)
        self.pub_info = self.create_publisher(MarkerArray, "/curobo/pick_info", 1)

        misc = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Empty, "/curobo/play", self._on_play, 1, callback_group=misc)
        self.create_subscription(Empty, "/curobo/reset", self._on_reset, 1, callback_group=misc)
        self.create_subscription(Empty, "/curobo/swap", self._on_swap, 1, callback_group=misc)
        self.create_subscription(Point, "/curobo/set_start", self._on_set_start, 1,
                                 callback_group=misc)
        self.create_subscription(Point, "/curobo/set_goal", self._on_set_goal, 1,
                                 callback_group=misc)

        self.srv = InteractiveMarkerServer(self, "curobo/pose_picker")
        self._make_marker("start", self.start_xyz, self.start_quat, (0.2, 0.5, 1.0))
        self._make_marker("goal", self.goal_xyz, self.goal_quat, (0.2, 1.0, 0.3))
        self.srv.applyChanges()

        # 계획(GPU)과 재생/발행은 따로 돈다. 계획이 340ms 씩 걸려서
        # 같은 그룹에 두면 재생이 뚝뚝 끊긴다.
        self.create_timer(1.0 / rate, self.solve_tick,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / ANIM_HZ, self.anim_tick, callback_group=misc)
        # 칼만은 CPU 만 쓰므로 GPU 그룹에 둘 필요가 없다
        self.create_timer(1.0 / PERCEPTION_HZ, self.perception_tick, callback_group=misc)

        self.get_logger().info(
            f"준비 완료 — 주황 장애물 {'움직임' if self.moving else '고정'} "
            f"(moving:={str(self.moving).lower()})\n"
            "  RViz 에서 파랑(시작)/초록(목표) 마커를 끌어보세요.\n"
            "  재생:   ros2 topic pub --once /curobo/play  std_msgs/msg/Empty {}\n"
            "  되돌림: ros2 topic pub --once /curobo/reset std_msgs/msg/Empty {}\n"
            "  맞바꿈: ros2 topic pub --once /curobo/swap  std_msgs/msg/Empty {}")

    # ------------------------------------------------------------------ scene
    def _obstacles(self):
        return list(self.static_obs) + [self._mover_xyz()]

    def _mover_xyz(self):
        """계획/표시에 쓸 움직이는 장애물 위치."""
        if not self.moving or self.kf is None:
            return np.array([MOVER_X, self.mover_y, MOVER_Z])
        return self.kf.predict_ahead(PREDICT_T)[0]

    def perception_tick(self):
        """카메라 흉내 — 노이즈 낀 위치만 얻어서 칼만에 넣는다. CPU 만 쓴다."""
        if not self.moving:
            return
        dt = 1.0 / PERCEPTION_HZ
        self.mover.step(time.time() - self.t0, dt)
        z = self.mover.observe(OBS_NOISE)
        with self.lock:
            if self.kf is None:
                self.kf = KalmanCV(z)
            else:
                self.kf.predict(dt)
                self.kf.update(z)
            self.dirty = True          # 장애물이 움직였으니 다시 계획한다

    def _sync_world(self):
        """장애물 위치만 갱신한다. 씬 재생성은 96ms 라 쓸 수 없다.

        GPU 를 건드리므로 반드시 solve_tick 과 같은 콜백 그룹에서 불러야 한다.
        (다른 스레드에서 부르면 CUDA graph 캡처가 깨진다)
        """
        if not self.moving or self.kf is None:
            return
        p = self._mover_xyz()
        name = f"obs_{len(self.static_obs)}"
        try:
            self.planner.scene_collision_checker.update_obstacle_pose(
                name, Pose.from_list([float(p[0]), float(p[1]), float(p[2]),
                                      1.0, 0.0, 0.0, 0.0]))
        except Exception as e:
            if not getattr(self, "_warned_sync", False):
                self._warned_sync = True
                self.get_logger().warn(f"장애물 위치 갱신 실패({name}): {e}")

    def _radius(self, i, n):
        return (R_MOVER + PREDICT_MARGIN) if i == n - 1 else R_OBS

    def _scene(self):
        pts = self._obstacles()
        sph = [Sphere(name=f"obs_{i}", radius=self._radius(i, len(pts)),
                      pose=[*p.tolist(), 1, 0, 0, 0]) for i, p in enumerate(pts)]
        return Scene(cuboid=floor_cuboids() + list(Scene(sphere=sph).get_obb_world().cuboid))

    # --------------------------------------------------------- interactive UI
    def _make_marker(self, name, xyz, quat, rgb):
        im = InteractiveMarker()
        im.header.frame_id = BASE
        im.name = name
        im.description = "시작" if name == "start" else "목표"
        im.scale = 0.18
        im.pose.position.x, im.pose.position.y, im.pose.position.z = map(float, xyz)
        im.pose.orientation.w = float(quat[0])
        im.pose.orientation.x = float(quat[1])
        im.pose.orientation.y = float(quat[2])
        im.pose.orientation.z = float(quat[3])

        # 가운데 구 — 잡아끌 몸통
        vis = InteractiveMarkerControl()
        vis.always_visible = True
        vis.interaction_mode = InteractiveMarkerControl.MOVE_3D
        sph = Marker()
        sph.type = Marker.SPHERE
        sph.scale.x = sph.scale.y = sph.scale.z = 0.055
        sph.color.r, sph.color.g, sph.color.b = rgb
        sph.color.a = 0.9
        vis.markers.append(sph)
        im.controls.append(vis)

        # 축별 이동 화살표 + 회전 링 (6-DOF)
        axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 0.0, 1.0), "z": (0.0, 1.0, 0.0)}
        for ax, (qx, qy, qz) in axes.items():
            for mode, suffix in ((InteractiveMarkerControl.MOVE_AXIS, "move"),
                                 (InteractiveMarkerControl.ROTATE_AXIS, "rot")):
                c = InteractiveMarkerControl()
                c.orientation.w = 1.0
                c.orientation.x, c.orientation.y, c.orientation.z = qx, qy, qz
                _norm(c.orientation)
                c.name = f"{suffix}_{ax}"
                c.interaction_mode = mode
                im.controls.append(c)

        self.srv.insert(im, feedback_callback=self._on_feedback)

    def _on_feedback(self, fb):
        if fb.event_type != fb.POSE_UPDATE:
            return
        p, o = fb.pose.position, fb.pose.orientation
        with self.lock:
            if fb.marker_name == "start":
                self.start_xyz = np.array([p.x, p.y, p.z])
                self.start_quat = np.array([o.w, o.x, o.y, o.z])
                self.start_dirty = True
            else:
                self.goal_xyz = np.array([p.x, p.y, p.z])
                self.goal_quat = np.array([o.w, o.x, o.y, o.z])
            self.dirty = True
            self.play_i = None      # 마커를 건드리면 재생 중단

    def _move_marker(self, name, xyz, quat):
        """토픽으로 좌표를 지정했을 때 RViz 마커도 같이 옮긴다."""
        from geometry_msgs.msg import Pose
        ps = Pose()
        ps.position.x, ps.position.y, ps.position.z = map(float, xyz)
        ps.orientation.w = float(quat[0])
        ps.orientation.x, ps.orientation.y, ps.orientation.z = map(float, quat[1:])
        self.srv.setPose(name, ps)
        self.srv.applyChanges()

    # ------------------------------------------------------------------ input
    def _on_set_start(self, msg):
        with self.lock:
            self.start_xyz = np.array([msg.x, msg.y, msg.z])
            self.start_dirty = self.dirty = True
            q = self.start_quat.copy()
        self._move_marker("start", self.start_xyz, q)

    def _on_set_goal(self, msg):
        with self.lock:
            self.goal_xyz = np.array([msg.x, msg.y, msg.z])
            self.dirty = True
            q = self.goal_quat.copy()
        self._move_marker("goal", self.goal_xyz, q)

    def _on_swap(self, _msg):
        with self.lock:
            self.start_xyz, self.goal_xyz = self.goal_xyz.copy(), self.start_xyz.copy()
            self.start_quat, self.goal_quat = self.goal_quat.copy(), self.start_quat.copy()
            self.start_dirty = self.dirty = True
            s, g, sq, gq = self.start_xyz, self.goal_xyz, self.start_quat, self.goal_quat
        self._move_marker("start", s, sq)
        self._move_marker("goal", g, gq)
        self.get_logger().info("시작 <-> 목표 맞바꿈")

    def _on_play(self, _msg):
        with self.lock:
            if self.traj is None:
                self.get_logger().warn("재생할 경로가 없다 (계획 실패 상태)")
                return
            self.play_i = 0
        self.get_logger().info("재생 시작")

    def _on_reset(self, _msg):
        with self.lock:
            self.play_i = None
        self.get_logger().info("시작 자세로 되돌림")

    # ------------------------------------------------------------------- plan
    def solve_tick(self):
        self._sync_world()          # GPU. solve_tick 과 같은 그룹이어야 한다
        with self.lock:
            if not self.dirty:
                return
            self.dirty = False
            s_xyz, s_q = self.start_xyz.copy(), self.start_quat.copy()
            g_xyz, g_q = self.goal_xyz.copy(), self.goal_quat.copy()
            need_start = self.start_dirty
            self.start_dirty = False

        notes = []
        # --- 1. 도달 가능성은 플랜지 기준으로 본다 ------------------------------
        for nm, xyz in (("시작", s_xyz), ("목표", g_xyz)):
            d = self._flange_dist(xyz)
            if d > REACH:
                notes.append(f"{nm} 리치 초과 (플랜지 {d:.3f}m > {REACH})")
            elif d > REACH_WARN:
                notes.append(f"{nm} 한계 근접 (플랜지 {d:.3f}m)")

        # --- 2. 시작 자세 IK (홈에서 그 지점까지 풀어서 관절값을 얻는다) ---------
        if need_start or self.q_start is None:
            q_s = self._solve(HOME, s_xyz, s_q)
            if q_s is None:
                self._report(None, ["시작 지점에 도달할 수 없다 (IK 해 없음 또는 충돌)"] + notes)
                return
            with self.lock:
                self.q_start = q_s[-1]
        q_start = self.q_start.copy()

        # --- 3. 시작 -> 목표 -------------------------------------------------
        t0 = time.time()
        traj = self._solve(q_start, g_xyz, g_q)
        dt = time.time() - t0
        if traj is None:
            with self.lock:
                self.traj = None
            self._publish_js(q_start)
            self._report(None, ["목표까지 경로가 없다 (장애물이 막았거나 IK 해 없음)"] + notes)
            return

        # --- 4. 손목 분기 확인 — 오늘 우리를 4.7배 우회로 몰았던 것 ---------------
        j5_s, j5_g = np.rad2deg(q_start[4]), np.rad2deg(traj[-1][4])
        flipped = (j5_s * j5_g) < 0
        if flipped:
            notes.append(f"손목 분기 반대 (J5 {j5_s:+.0f}° -> {j5_g:+.0f}°) — TCP 가 크게 휘둘린다")

        xyz = self._fk(traj)
        L = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        d = float(np.linalg.norm(g_xyz - xyz[0]))
        ratio = L / d if d > 1e-6 else 0.0
        if ratio > DETOUR_WARN:
            notes.append(f"우회 {ratio:.1f}배 — 비정상적으로 길다")

        with self.lock:
            self.traj = traj
        self._publish_js(q_start)
        self._publish_path(xyz)
        self._report(dict(n=len(traj), ms=dt * 1000, straight=d, length=L, ratio=ratio,
                          j5_s=j5_s, j5_g=j5_g, zmax=float(xyz[:, 2].max()), flipped=flipped),
                     notes)

    def _solve(self, q_from, xyz, quat):
        res = self.planner.plan_pose(
            GoalToolPose(
                tool_frames=[TOOL],
                position=torch.tensor(np.asarray(xyz).reshape(1, 1, 1, 1, 3),
                                      dtype=torch.float32, device="cuda"),
                quaternion=torch.tensor(np.asarray(quat).reshape(1, 1, 1, 1, 4),
                                        dtype=torch.float32, device="cuda")),
            js(q_from), max_attempts=5, enable_graph_attempt=1)
        if res is None or not bool(res.success.flatten()[0]):
            return None
        tr = res.get_interpolated_plan().position.squeeze().detach().cpu().numpy()
        return tr[None, :] if tr.ndim == 1 else tr

    def _fk(self, traj):
        st = self.planner.compute_kinematics(js(traj))
        return st.tool_poses.position.reshape(len(traj), -1)[:, :3].detach().cpu().numpy()

    @staticmethod
    def _flange_dist(tcp_xyz):
        """TCP 가 아니라 플랜지까지의 거리. 리치는 플랜지 기준이다.

        정확히는 툴 방향에 따라 다르지만, 툴이 아래를 보는 기본 자세에서는
        플랜지가 TCP 보다 238mm 위에 있다고 보면 충분하다.
        """
        return float(np.linalg.norm([tcp_xyz[0], tcp_xyz[1], tcp_xyz[2] + 0.23813]))

    # ---------------------------------------------------------------- publish
    def anim_tick(self):
        with self.lock:
            traj, i = self.traj, self.play_i
            q = self.q_start.copy() if self.q_start is not None else HOME.copy()
            if traj is not None and i is not None:
                if i >= len(traj):
                    self.play_i = None
                else:
                    q = traj[i]
                    self.play_i = i + 1
        self._publish_js(q)
        self._publish_scene()

    def _publish_js(self, q):
        m = JointStateMsg()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = JOINTS
        m.position = [float(v) for v in q]
        self.pub_js.publish(m)

    def _hdr(self):
        h = Path().header
        h.frame_id = BASE
        h.stamp = self.get_clock().now().to_msg()
        return h

    def _publish_path(self, xyz):
        path = Path(header=self._hdr())
        for p in xyz:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def _publish_scene(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        pts = self._obstacles()
        for i, p in enumerate(pts):
            m = Marker()
            m.header.frame_id, m.header.stamp = BASE, now
            m.ns, m.id, m.type, m.action = "obs", i, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2.0 * self._radius(i, len(pts))
            mover = i == len(pts) - 1
            m.color.r, m.color.g, m.color.b = (1.0, 0.5, 0.1) if mover else (0.85, 0.2, 0.2)
            m.color.a = 0.7
            ma.markers.append(m)
        self.pub_obs.publish(ma)

    def _report(self, res, notes):
        """터미널 로그 + RViz 안내문."""
        if res is None:
            line = "계획 실패"
        else:
            line = (f"{res['n']:3d}점 {res['ms']:4.0f}ms | 직선 {res['straight']:.3f} "
                    f"경로 {res['length']:.3f} = {res['ratio']:.1f}배 | "
                    f"J5 {res['j5_s']:+.0f}°->{res['j5_g']:+.0f}° | 최고z {res['zmax']:.2f}")
        if notes:
            self.get_logger().warn(line + "\n   ! " + "\n   ! ".join(notes),
                                   throttle_duration_sec=1.0)
        else:
            self.get_logger().info(line + "   OK", throttle_duration_sec=1.0)

        m = Marker()
        m.header.frame_id, m.header.stamp = BASE, self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "info", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = 0.0, 0.0, 0.95
        m.pose.orientation.w = 1.0
        m.scale.z = 0.045
        bad = bool(notes)
        m.color.r, m.color.g, m.color.b = (1.0, 0.45, 0.1) if bad else (0.6, 1.0, 0.6)
        m.color.a = 0.95
        m.text = line + ("\n" + "\n".join("! " + n for n in notes) if notes else "")
        self.pub_info.publish(MarkerArray(markers=[m]))


def _norm(q):
    n = np.sqrt(q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)
    q.w, q.x, q.y, q.z = q.w / n, q.x / n, q.y / n, q.z / n


def main():
    rclpy.init()
    node = PosePicker()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
