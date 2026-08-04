#!/usr/bin/env python3
"""
cuRobo 경로계획 RViz 데모 — 궤적 버퍼 구조.

계획과 실행을 분리한다. 이게 실기에서 쓸 구조와 같은 모양이다.

    [5Hz]   재계획   cuRobo plan_pose (182ms) -> 궤적 버퍼 교체
    [30Hz]  실행     버퍼를 시간축으로 보간해서 joint_states 발행
                     + 장애물/경로 마커

핵심 두 가지
  - 계획이 도는 동안에도 실행은 30Hz 로 계속 돈다 (다른 스레드).
  - 계획 시작점을 "지금 위치"가 아니라 "계획이 끝날 시점의 예상 위치"로 잡는다.
    안 그러면 새 궤적이 도착했을 때 로봇이 뒤로 튄다. (지연 보상)

발행 토픽
    /joint_states       로봇 자세
    /curobo/obstacles   장애물 + 작업대
    /curobo/tcp_path    계획된 TCP 경로 (라임 선)
    /curobo/goal        목표 지점

실행:  ros2 launch curobo_demo.launch.py
"""
import threading
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState as JointStateMsg
from visualization_msgs.msg import Marker, MarkerArray

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene, Sphere
from curobo.types import GoalToolPose, JointState

CFG = "/home/rokey/cobot2_ws/dum_E_project/tools/m0609.yml"
TOOL, BASE = "tcp", "base_link"   # 계획 목표는 플랜지가 아니라 실제 TCP
JOINTS = [f"joint_{i}" for i in range(1, 7)]
HOME = np.deg2rad([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])

DT = 0.02          # 궤적 보간 간격 (cuRobo interpolation_dt 와 맞춤)
ANIM_HZ = 30.0
PLAN_HZ = 5.0      # executing 플래그가 중복 계획을 막는다. 대기시간만 줄임
PLAN_LATENCY = 0.55  # 계획 소요시간 초기 추정. 실측으로 갱신된다.

TABLE = dict(name="table", dims=[0.9, 1.2, 0.05], pose=[0.60, 0.0, -0.025, 1, 0, 0, 0])

# ---------------------------------------------------------------------------
# 작업대 상판 = 바닥. z < 0 은 물리적으로 불가능하다.
#
# 그런데 베이스 밑까지 통짜로 깔면 base_link 구(x -0.122~0.107, z -0.061~0.074)와
# 겹쳐서 시작 자세부터 불가능 판정이 난다. 로봇은 작업대에 볼트로 고정돼 있으니
# 그 접촉은 충돌이 아니다.
#   -> 베이스 발자국만 구멍을 내고 그 바깥을 4장의 판으로 덮는다.
# ---------------------------------------------------------------------------
FLOOR_T = 0.20                      # 판 두께 (아래로 파고드는 것 방지)
KEEPOUT_X = (-0.20, 0.20)           # 베이스 발자국 (여유 포함)
KEEPOUT_Y = (-0.45, 0.25)
FLOOR_EXT = 1.1                     # 바닥 바깥 경계


def floor_cuboids():
    """윗면이 z=0 인 바닥 판 4장. 베이스 발자국만 비워 둔다."""
    zc = -FLOOR_T / 2.0
    kx0, kx1 = KEEPOUT_X
    ky0, ky1 = KEEPOUT_Y
    E = FLOOR_EXT
    boxes = [
        ("floor_front", (kx1, E), (-E, E)),
        ("floor_back", (-E, kx0), (-E, E)),
        ("floor_left", (kx0, kx1), (ky1, E)),
        ("floor_right", (kx0, kx1), (-E, ky0)),
    ]
    out = []
    for name, (x0, x1), (y0, y1) in boxes:
        out.append(Cuboid(
            name=name,
            dims=[x1 - x0, y1 - y0, FLOOR_T],
            pose=[(x0 + x1) / 2, (y0 + y1) / 2, zc, 1, 0, 0, 0]))
    return out

# 팔이 홈 자세에 있을 때와 겹치지 않는 위치에 둔다.
# 홈에서 TCP 는 대략 (0.37, 0.0, 0.19) 이므로 그보다 바깥·위쪽.
R_OBS = 0.05

# 작업대 위에 놓인 물체. 앞(x=0.40)과 뒤(x=0.70) 사이를 가로막아
# 로봇이 '들어올려서 넘어가는' 동작을 하게 만든다.
#
# 왜 이 배치인가 (측정으로 확인):
#   목표를 y 로 크게 벌리면(예: [0.60,-0.28,0.15]) 그 지점에는 홈과 같은
#   손목 분기의 IK 해가 아예 없다. J5 를 +로 제한하면 계획 자체가 실패한다.
#   손목이 180도 뒤집히면서 TCP(플랜지에서 238mm)가 큰 원호를 그려
#   경로가 직선거리의 4.7배까지 늘어난다. seed 를 48개로 늘려도, 그래프
#   플래너를 꺼도, 경유점을 넣어도 4.4~4.7배에서 변하지 않았다.
#   앞뒤 왕복은 y=0 근처라 손목 분기가 유지되고 1.5배로 자연스럽다.
STATIC_OBS = np.array([[0.48, 0.00, 0.10]])
# 들어올려 넘어가는 높이(z≈0.19)를 y 방향으로 가로지른다.
MOVER_X, MOVER_Z, MOVER_AMP = 0.48, 0.24, 0.22
MOVER_SPEED = 0.25                   # rad/s. 낮출수록 천천히 왕복

# 목표는 장애물 주변 껍질에서 무작위로 뽑는다 (작업공간 좌표 직접 지정).
# 관절값을 무작위로 뽑아 FK 하는 방식은 팔을 뻗은 자세가 거의 안 나와서 못 쓴다.
GOAL_RING = dict(r_min=0.17, r_max=0.30, z_min=0.10, z_max=0.30,
                 ang=2.0, x_min=0.42)
N_GOALS = 8


class TrajBuffer:
    """궤적 + 재생 위치. 계획 스레드가 쓰고 실행 스레드가 읽는다."""

    def __init__(self, q0):
        self.lock = threading.Lock()
        self.traj = np.repeat(q0[None, :], 2, axis=0)
        self.t = 0.0

    def sample(self, t):
        """t[초] 지점의 관절값 (선형보간)."""
        with self.lock:
            tr = self.traj
        u = np.clip(t / DT, 0.0, len(tr) - 1.0)
        i = int(np.floor(u))
        j = min(i + 1, len(tr) - 1)
        a = u - i
        return tr[i] * (1 - a) + tr[j] * a

    def duration(self):
        with self.lock:
            return (len(self.traj) - 1) * DT

    def replace(self, traj, t0: float = 0.0):
        with self.lock:
            self.traj = traj
            self.t = t0


class CuroboRvizDemo(Node):
    def __init__(self):
        super().__init__("curobo_rviz_demo")
        self.declare_parameter("moving", False)      # 동적 장애물
        self.declare_parameter("continuous", False)  # 실행 중 재계획
        self.declare_parameter("scene_mode", "obb")  # obb(빠름) | mesh(정확)
        self.declare_parameter("n_static", 3)
        self.declare_parameter("speed", 0.7)   # 궤적 재생 배속
        self.moving = bool(self.get_parameter("moving").value)
        self.continuous = bool(self.get_parameter("continuous").value)
        self.scene_mode = str(self.get_parameter("scene_mode").value)
        n = int(self.get_parameter("n_static").value)
        self.speed = float(self.get_parameter("speed").value)
        self.static_obs = STATIC_OBS[:max(0, min(n, len(STATIC_OBS)))]

        self.get_logger().info("cuRobo 로딩...")
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=CFG, scene_model=self._scene(0.0),
            collision_cache={"cuboid": 64, "mesh": 64},
            use_cuda_graph=True, interpolation_dt=DT,
        ))
        self.planner.warmup(enable_graph=True)
        self.get_logger().info("준비 완료")

        self.wall = 0.0
        self.buf = TrajBuffer(HOME.copy())
        self.path_xyz = None
        self.executing = False
        self.pending = None      # 미리 계획해 둔 다음 궤적
        self._n_ok = self._n_block = 0
        self._scene_cache = None
        self.plan_latency = PLAN_LATENCY   # 실측으로 갱신된다
        self.goal = None
        self.get_logger().info("목표 검증 중...")
        self.build_goal_pool()
        self.new_goal()

        self.pub_js = self.create_publisher(JointStateMsg, "/joint_states", 10)
        self.pub_obs = self.create_publisher(MarkerArray, "/curobo/obstacles", 1)
        self.pub_path = self.create_publisher(Path, "/curobo/tcp_path", 1)
        self.pub_goal = self.create_publisher(Marker, "/curobo/goal", 1)

        # /joint_states 에 발행자가 둘이면 RViz 가 두 자세를 번갈아 받아서
        # 로봇이 초당 수십 번 깜빡인다. sphere_check.launch.py 를 같이 띄우면
        # 거기 joint_state_publisher_gui 가 홈 자세를 계속 쏘기 때문에 겹친다.
        self.create_timer(3.0, self._check_publishers)

        # 계획은 오래 걸리므로 실행 타이머와 다른 콜백 그룹에 둔다.
        self.create_timer(1.0 / ANIM_HZ, self.animate,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / PLAN_HZ, self.replan,
                          callback_group=MutuallyExclusiveCallbackGroup())

    def _check_publishers(self):
        n = self.count_publishers("/joint_states")
        if n > 1:
            self.get_logger().error(
                f"/joint_states 발행자가 {n}개다! 로봇이 두 자세 사이에서 깜빡인다.\n"
                f"   -> sphere_check.launch.py 등 다른 런치가 같이 떠 있는지 확인하고 끄세요\n"
                f"      (joint_state_publisher_gui 가 홈 자세를 계속 발행한다)")

    # ------------------------------------------------------------------ scene
    def _obstacles(self, t):
        pts = list(self.static_obs)
        if self.moving:
            pts.append(np.array([MOVER_X, MOVER_AMP * np.sin(t * MOVER_SPEED), MOVER_Z]))
        return pts

    def _scene_cached(self, t):
        """정적 장애물이면 씬을 한 번만 만들고 재사용한다.

        get_mesh_world() 는 구마다 trimesh icosphere 를 새로 만든다.
        이걸 매 재계획마다 호출하면 계획 시간이 645ms -> 1800ms 로 계속 늘어난다
        (씬 재업로드가 누적된다).
        """
        if not self.moving:
            if self._scene_cache is None:
                self._scene_cache = self._scene(0.0)
                self.planner.update_world(self._scene_cache)
            return None                    # 갱신 불필요
        return self._scene(t)

    def _scene(self, t):
        """장애물만. 작업대는 충돌 대상에서 제외한다.

        로봇이 작업대에 볼트로 고정돼 있어서 base_link 와의 접촉은 충돌이 아닌데,
        그걸 일반적으로 구분할 방법이 없다. 작업대는 화면에만 그린다.

        ★ get_mesh_world() 필수: cuRobo 의 충돌월드는 cuboid/mesh/voxel 만 받는다.
          Scene(sphere=...) 을 그대로 넘기면 구 장애물이 조용히 전부 무시된다.
        """
        sph = [Sphere(name=f"obs_{i}", radius=R_OBS, pose=[*p.tolist(), 1, 0, 0, 0])
               for i, p in enumerate(self._obstacles(t))]

        # ★ 구는 그대로 넘기면 충돌월드에서 조용히 사라진다. 변환이 필수다.
        #   그런데 씬 전체에 get_mesh_world() 를 걸면 바닥 cuboid 까지 메시가 돼서
        #   엄청나게 느려진다 (warmup 96s / plan 2104ms).
        #   바닥은 cuboid 로 두고 구만 변환한다. 실측:
        #       전체 mesh          warmup 96.6s  plan 2104ms
        #       바닥cuboid+구mesh  warmup 31.5s  plan  747ms
        #       전부 cuboid(obb)   warmup 18.5s  plan  340ms   <- 기본값
        #   obb 는 구를 외접 정육면체로 바꾸므로 모서리가 0.73r 만큼 더 튀어나온다.
        #   장애물은 어차피 안전여유로 부풀려 쓰는 값이라 이 보수성은 감수할 만하다.
        if self.scene_mode == "obb":
            conv = Scene(sphere=sph).get_obb_world()
            return Scene(cuboid=floor_cuboids() + list(conv.cuboid))
        conv = Scene(sphere=sph).get_mesh_world()
        return Scene(cuboid=floor_cuboids(), mesh=list(conv.mesh))

    # ------------------------------------------------------------------- FK
    def _fk_batch(self, qs):
        """(N,6) 관절값 -> (N,3) TCP 위치. GPU 왕복 1회."""
        q = torch.tensor(np.asarray(qs, dtype=np.float32), device="cuda")
        st = self.planner.compute_kinematics(JointState.from_position(q))
        return st.tool_poses.position.reshape(len(qs), -1)[:, :3].detach().cpu().numpy()

    def build_goal_pool(self, n_target=N_GOALS, max_try=40):
        """장애물 주변에서 목표를 무작위로 뽑되, 실제로 계획 가능한 것만 남긴다.

        검증 없이 쓰면 장애물 안이나 도달 불가 지점이 섞여서
        "계획 실패" 로그만 잔뜩 남는다. 시작할 때 한 번 걸러 둔다.
        """
        start = JointState.from_position(
            torch.tensor(HOME[None, :], dtype=torch.float32, device="cuda"))
        self.quat = self.planner.compute_kinematics(start) \
            .tool_poses.quaternion.reshape(1, 1, 1, 1, 4).clone()

        c = STATIC_OBS[0]
        rng = np.random.default_rng(5)
        g_cfg = GOAL_RING
        pool, tried = [], 0
        while len(pool) < n_target and tried < max_try:
            tried += 1
            a = rng.uniform(-g_cfg["ang"], g_cfg["ang"])
            r = rng.uniform(g_cfg["r_min"], g_cfg["r_max"])
            g = np.array([c[0] + r * np.cos(a),
                          c[1] + r * np.sin(a),
                          rng.uniform(g_cfg["z_min"], g_cfg["z_max"])])
            if g[0] < g_cfg["x_min"]:
                continue
            # 정적 장애물 표면에서 충분히 떨어졌는지 싸게 먼저 거른다
            if np.linalg.norm(g - c) < R_OBS + 0.09:
                continue
            # 움직이는 장애물이 쓸고 지나가는 영역과도 떨어뜨린다.
            # 안 그러면 그 구간에서 목표가 장애물 안에 들어가 계획이 계속 실패한다.
            if self.moving:
                d_sweep = np.hypot(g[0] - MOVER_X, g[2] - MOVER_Z)
                if d_sweep < R_OBS + 0.05:
                    continue

            pos = torch.tensor(g.reshape(1, 1, 1, 1, 3),
                               dtype=torch.float32, device="cuda")
            res = self.planner.plan_pose(
                GoalToolPose(tool_frames=[TOOL], position=pos, quaternion=self.quat),
                start, max_attempts=5, enable_graph_attempt=1)
            if res is not None and bool(res.success.flatten()[0]):
                pool.append((pos, g))
        # 가까운 목표끼리 이어지도록 정렬한다 (greedy nearest).
        # 안 하면 y=-0.28 <-> +0.28 를 오가며 매번 홈 근처를 지나가서
        # "홈으로 돌아갔다 나온다" 처럼 보인다.
        if pool:
            ordered = [pool.pop(0)]
            while pool:
                last = ordered[-1][1]
                j = int(np.argmin([np.linalg.norm(g - last) for _, g in pool]))
                ordered.append(pool.pop(j))
            pool = ordered
        self.pool = pool
        self.pool_i = 0
        self.get_logger().info(f"유효 목표 {len(pool)}개 ({tried}회 시도)")

    def new_goal(self):
        if not self.pool:
            return
        _, xyz = self.pool[self.pool_i % len(self.pool)]
        self.goal_xyz = xyz

    # ----------------------------------------------------------------- plan
    def replan(self):
        """continuous=False : 실행 중엔 '다음' 궤적을 미리 계획 (파이프라이닝)
           continuous=True  : 같은 목표로 '현재' 궤적을 계속 갈아끼움 (동적 회피)
        """
        sc = self._scene_cached(self.wall)
        if sc is not None:
            self.planner.update_world(sc)

        if self.continuous:
            self._replan_continuous()
            return

        if self.pending is not None:
            return                      # 이미 다음 것이 준비돼 있다
        if self.executing:
            q_from = self.buf.sample(self.buf.duration())
        else:
            q_from = self.buf.sample(self.buf.t + self.plan_latency * self.speed)
        traj, xyz, elapsed = self._plan_from(q_from)
        if traj is None:
            return
        self.pool_i += 1
        self.pending = (traj, xyz, elapsed)
        if not self.executing:
            self._start_pending()

    def _replan_continuous(self):
        """이동 중에도 현재 궤적을 새로 계획한 것으로 교체한다.

        장애물이 움직이면 이미 세운 궤적이 무효가 되므로, 같은 목표를 향해
        '지금 위치' 에서 다시 계획해 갈아끼운다. 지연 보상을 넣지 않으면
        새 궤적이 도착했을 때 로봇이 뒤로 튄다.
        """
        lead = self.plan_latency * self.speed
        q_from = self.buf.sample(self.buf.t + lead)

        # 목표에 충분히 가까워졌으면 다음 목표로
        if np.linalg.norm(self._fk_batch(q_from[None, :])[0] - self.goal_xyz) < 0.04:
            self.pool_i += 1
            self.new_goal()

        traj, xyz, elapsed = self._plan_from(q_from, advance=False)
        if traj is None:
            return
        # 새 궤적의 0 지점 = 계획 시작 + lead 의 자세.
        # 실제로는 elapsed*speed 만큼 흘렀으니 그 차이에서 시작한다.
        self.buf.replace(traj, t0=max(0.0, elapsed * self.speed - lead))
        self.path_xyz = self._fk_batch(traj[::2])
        self.executing = True
        self._n_ok += 1
        if self._n_ok % 5 == 1:
            self.get_logger().info(
                f"재계획 #{self._n_ok}  {len(traj):3d}점  {elapsed*1000:4.0f}ms  "
                f"목표 {np.round(self.goal_xyz,2)}  (막힘 {self._n_block}회)")

    def _plan_from(self, q_from, advance=True):
        """q_from 에서 현재 목표까지 계획. (traj, goal_xyz, elapsed) 반환."""
        start = JointState.from_position(
            torch.tensor(q_from[None, :], dtype=torch.float32, device="cuda"))
        pos, xyz = self.pool[self.pool_i % len(self.pool)]
        g = GoalToolPose(tool_frames=[TOOL], position=pos, quaternion=self.quat)

        t0 = time.time()
        res = self.planner.plan_pose(g, start, max_attempts=5, enable_graph_attempt=1)
        elapsed = time.time() - t0
        self.plan_latency = 0.7 * self.plan_latency + 0.3 * elapsed

        if res is None or not bool(res.success.flatten()[0]):
            self._n_block += 1
            self.get_logger().warn(
                f"장애물이 경로를 막음 -> 그 자리에서 대기 (Hold)  "
                f"[{elapsed*1000:.0f}ms, 누적 {self._n_block}회]",
                throttle_duration_sec=2.0)
            if advance:
                self.pool_i += 1
            return None, None, elapsed

        traj = res.get_interpolated_plan().position.squeeze().detach().cpu().numpy()
        if traj.ndim == 1:
            traj = traj[None, :]
        return np.asarray(traj, dtype=float), xyz, elapsed

    def _start_pending(self):
        """준비된 궤적을 버퍼에 걸고 실행을 시작한다."""
        traj, xyz, elapsed = self.pending
        self.pending = None
        # 실행 중 이어붙이는 경우엔 처음부터, 정지 상태에서 시작하는 경우엔
        # 계획하는 동안 흘러간 만큼 앞에서 시작해야 튀지 않는다.
        t0 = 0.0 if self.executing else max(
            0.0, elapsed * self.speed - self.plan_latency * self.speed)
        self.buf.replace(traj, t0=t0)
        self.goal_xyz = xyz
        self.path_xyz = self._fk_batch(traj[::2])
        self.executing = True
        self.get_logger().info(
            f"실행 시작  {len(traj)}점  계획 {elapsed*1000:4.0f}ms  "
            f"목표 {np.round(xyz,3)}")

    # ------------------------------------------------------------- animate
    def animate(self):
        dt = 1.0 / ANIM_HZ
        self.wall += dt
        self.buf.t += dt * self.speed

        q = self.buf.sample(self.buf.t)
        if not self.continuous and self.executing and self.buf.t >= self.buf.duration():
            if self.pending is not None:
                self._start_pending()    # 끊김 없이 이어서 실행
            else:
                self.executing = False   # 아직 준비 안 됨 -> 여기서 대기

        now = self.get_clock().now().to_msg()

        js = JointStateMsg()
        js.header.stamp = now
        js.name = JOINTS
        js.position = [float(v) for v in q]
        self.pub_js.publish(js)

        ma = MarkerArray()
        m = Marker()
        m.header.frame_id, m.header.stamp = BASE, now
        m.ns, m.id, m.type, m.action = "table", 0, Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = TABLE["pose"][:3]
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = TABLE["dims"]
        m.color.r = m.color.g = 0.45
        m.color.b, m.color.a = 0.50, 0.5
        ma.markers.append(m)

        obs = self._obstacles(self.wall)
        for i, p in enumerate(obs):
            mk = Marker()
            mk.header.frame_id, mk.header.stamp = BASE, now
            # planner 가 실제로 쓰는 형상을 그린다. obb 모드인데 구를 그리면
            # "화면과 다른 걸 피한다" 는 오해가 생긴다.
            box = self.scene_mode == "obb"
            mk.ns, mk.id, mk.action = "obs", i, Marker.ADD
            mk.type = Marker.CUBE if box else Marker.SPHERE
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, p)
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 2.0 * R_OBS
            is_mover = self.moving and i == len(obs) - 1
            mk.color.r, mk.color.g, mk.color.b = (
                (1.0, 0.5, 0.1) if is_mover else (0.85, 0.2, 0.2))
            mk.color.a = 0.6
            ma.markers.append(mk)
        self.pub_obs.publish(ma)

        if self.path_xyz is not None:
            path = Path()
            path.header.frame_id, path.header.stamp = BASE, now
            for p in self.path_xyz:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.pub_path.publish(path)

        gm = Marker()
        gm.header.frame_id, gm.header.stamp = BASE, now
        gm.ns, gm.id, gm.type, gm.action = "goal", 0, Marker.SPHERE, Marker.ADD
        gm.pose.position.x, gm.pose.position.y, gm.pose.position.z = map(float, self.goal_xyz)
        gm.pose.orientation.w = 1.0
        gm.scale.x = gm.scale.y = gm.scale.z = 0.06
        gm.color.r, gm.color.g, gm.color.b, gm.color.a = 0.2, 1.0, 0.3, 0.95
        self.pub_goal.publish(gm)


def main():
    rclpy.init()
    node = CuroboRvizDemo()
    ex = MultiThreadedExecutor(num_threads=3)
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
            pass  # launch 가 이미 context 를 내린 경우


if __name__ == "__main__":
    main()
