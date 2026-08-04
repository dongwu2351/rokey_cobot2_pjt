#!/usr/bin/env python3
"""
하이브리드 데모 — 미리보기 → 승인 → 실시간 회피 실행.

기획서의 CP5(AR 경로 표시) + CP6(승인) + CP8(동적 회피) 를 한 흐름으로 묶은 것.

    [PREVIEW]  목표 고정. 로봇은 정지.
               주황 장애물이 움직이면 -> MotionGen 이 3Hz 로 다시 계획
               -> 화면의 경로가 계속 바뀐다. 사람이 이걸 보고 판단한다.
                    |
                    |  ros2 topic pub --once /curobo/approve std_msgs/msg/Empty {}
                    v
    [EXECUTE]  승인 시점의 전역 경로를 고정하고 출발.
               가는 도중에도 주황이 움직이므로 MPC 가 50Hz 로 실시간 보정.
               -> 원래 경로를 따라가되 그때그때 휘어서 피한다.

왜 둘을 합치나
    MotionGen 단독 : 전역을 보지만 매번 새로 짜서 경로가 통째로 튄다
    MPC 단독       : 연속적이고 736Hz 지만 국소최적에 갇힌다 (측정: 254mm 에서 정체)
    하이브리드     : MotionGen 이 "어느 쪽으로 돌지" 를 정하고
                     MPC 가 그 길을 따라가며 실시간으로 휜다 (측정: 18.8mm 도달)

토픽
    /joint_states          로봇 자세
    /curobo/obstacles      장애물 + 작업대
    /curobo/global_path    전역 경로 (흰 선) — 승인 대상
    /curobo/tcp_path       MPC horizon (라임 선) — 실시간으로 꿈틀거림
    /curobo/goal           목표
    /curobo/approve        승인 입력 (std_msgs/Empty)
"""
import threading
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, Pose as PoseMsg, PoseStamped
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Empty
from interactive_markers import InteractiveMarkerServer
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    Marker, MarkerArray)

from curobo.model_predictive_control import (ModelPredictiveControl,
                                             ModelPredictiveControlCfg)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, Sphere
from curobo.types import GoalToolPose, JointState, Pose

# 기존 데모에서 씬 정의를 그대로 가져온다 (바닥 판, 장애물 배치, 로봇 설정)
from curobo_rviz_demo import (BASE, CFG, HOME, JOINTS, MOVER_X, MOVER_Z,
                              R_OBS, STATIC_OBS, TOOL, floor_cuboids)
from obstacle_tracker import KalmanCV, RandomMover

CACHE = {"cuboid": 64, "mesh": 64}
R_MOVER = 0.025       # 움직이는 물체의 실제 반경 [m] (음료캔 정도)

# --- 장애물 미래 위치 예측 ---------------------------------------------------
# cuRobo 의 월드는 "계획하는 순간의 정지 스냅샷" 이다. 현재 위치만 주면
# 로봇은 장애물이 움직인다는 걸 모르고 반응만 한다.
# 속도를 알면 "로봇이 거기 도달할 시점의 위치" 를 대신 넣을 수 있고,
# 그러면 미리 비켜간다. 속도는 실기에서 칼만필터가 준다.
# 예측 구간을 얼마로 잡을 것인가 — 측정으로 정했다.
#
# 등속 칼만의 0.8초 예측오차는 평균 74mm / 95% 192mm 였다. 물체가 3초마다
# 방향을 바꾸는데 등속 모델은 그걸 예측할 수 없기 때문이다. 여유 35mm 로는
# 턱없이 부족했다 (= 예측 위치를 믿고 비켜갔는데 실제로는 딴 데 있었다).
#
#   예측구간   평균오차   95%오차
#     0.0초     14mm      33mm
#     0.2초     26mm      69mm
#     0.3초     33mm      88mm      <- 채택
#     0.8초     74mm     192mm      <- 이전 설정
#
# 얼마나 앞을 봐야 하는가? "로봇이 실제로 비켜나는 데 걸리는 시간" 이다.
# 그건 MPC 지평 = MPC_HORIZON(16) x action_dt(0.02) = 0.32초.
# 그보다 먼 미래를 봐야 할 이유가 없다. 0.3초로 맞춘다.
PREDICT_T = 0.3
PREDICT_MARGIN = 0.035  # 예측오차 여유 [m]. 0.3초 기준 평균오차(33mm)에 해당한다.
                        # 95%(88mm)까지 덮으려면 0.09 로. 안전은 주로 50Hz
                        # 재계획에서 나오지 이 상수에서 나오는 게 아니다.

# 충돌 비용이 '언제부터' 발생하는지 = "안전거리 우선" 손잡이.
# cuRobo 비용함수에 clearance 항이 따로 없어서 이 값을 키우는 게 여유 확보 수단이다.
# 논문에선 0.025 에서 성공률 +27% 였으나, 우리 씬(장애물이 빡빡)에서는
# 완주 9->4회, 이탈 6->9회로 오히려 손해였다. 기하에 따라 다르므로 파라미터로 둔다.
ACTIVATION_DIST_DEFAULT = 0.01
PERCEPTION_HZ = 30.0  # 카메라 관측 주기 (YOLO 흉내)
OBS_NOISE = 0.012     # 관측 노이즈 [m]
MOVER_BOX = np.array([0.02, 0.30, 0.06])   # 무작위 이동 범위 (중심 기준)
MOVER_SPD = 0.12      # 이동 속도 [m/s]
CTRL_HZ = 50.0        # MPC action_dt 가 0.02s 이므로 50Hz 가 실시간 배속
PREVIEW_HZ = 3.0      # 미리보기 재계획 주기
MPC_HORIZON = 16      # MPC action_horizon (실측). 시드 텐서는 [1, 16, 6] 이어야 한다
LOOKAHEAD = 10        # 전역 경로에서 몇 점 앞을 MPC 목표로 줄 것인가
                      # 크게 잡으면 MPC 가 관절공간 지름길을 타서 팔이 크게 휘둘린다
MAX_DEVIATION = 0.35  # 전역 경로에서 이만큼(rad) 벗어나면 되돌리기 시도
RESYNC_TRIES = 3      # 재시드 몇 번까지 시도하고 포기할지
GRACE_TICKS = 30      # 재시드 후 효과를 기다리는 틱 수 (50Hz -> 0.6초)

# 목표는 고정된 몇 개를 순환한다. 미리보기 중엔 안 바뀐다.
# 장애물 앞(x=0.40) <-> 뒤(x=0.70) 왕복. Pick&Place 의 기본 모양이다.
# y 를 크게 벌리면 손목이 뒤집혀야 해서 경로가 4.7배로 늘어난다 (측정).
# 앞뒤 왕복은 1.5배로 자연스럽다.
# x=0.70 은 플랜지까지 0.786m 라 M0609 한계에 가깝다. 안쪽으로 당겼다.
GOALS_XYZ = np.array([
    [0.33, 0.00, 0.10],     # 앞 (장애물 이쪽)
    [0.60, 0.00, 0.10],     # 뒤 (장애물 저쪽)
])


def js(q, device="cuda"):
    q = torch.tensor(np.atleast_2d(q), dtype=torch.float32, device=device)
    z = torch.zeros_like(q)
    s = JointState.from_position(q, joint_names=JOINTS)
    s.velocity, s.acceleration, s.jerk = z.clone(), z.clone(), z.clone()
    return s


class HybridDemo(Node):
    def __init__(self):
        super().__init__("curobo_hybrid_demo")
        self.declare_parameter("moving", True)
        self.declare_parameter("n_static", 1)
        self.declare_parameter("auto_approve", 0.0)   # >0 이면 N초 뒤 자동 승인
        self.declare_parameter("activation_dist", ACTIVATION_DIST_DEFAULT)
        # 전역 경로를 MPC 시드로 재주입하는 '추가' 주기 (제어틱, 0=주기재시드 없음).
        #
        # 재시드 1회 비용이 87ms 다 (update_seed_trajectory 는 실행버퍼를 비우므로
        # optimize_action_sequence 로 재구성해야 하고, 그게 full solve 다).
        # 10Hz(=5틱) 로 하면 100ms 예산의 87% 를 먹어 제어루프가 다시 굶는다.
        # 그래서 기본은 '이벤트 기반' 으로 간다:
        #   - 실행 시작 시 1회   (어느 쪽으로 우회할지를 전역계획에서 상속)
        #   - 경로 이탈 감지 시   (MPC 가 길을 잃었을 때 되돌림)
        self.declare_parameter("reseed_every", 0)
        # 승인 대기 중에도 장애물이 다가오면 비켜설 것인가.
        # false 면 승인 전까지 완전히 정지 (어디까지가 '승인 후 동작' 인지 명확해짐).
        self.declare_parameter("hold_yield", True)
        self.moving = bool(self.get_parameter("moving").value)
        n = int(self.get_parameter("n_static").value)
        self.static_obs = STATIC_OBS[:max(0, min(n, len(STATIC_OBS)))]
        self.auto_approve = float(self.get_parameter("auto_approve").value)
        act = float(self.get_parameter("activation_dist").value)
        self.reseed_every = int(self.get_parameter("reseed_every").value)
        self.hold_yield = bool(self.get_parameter("hold_yield").value)

        self.t0 = time.time()
        self.lock = threading.Lock()
        # cuRobo 는 use_cuda_graph=True 로 도는데, CUDA graph 는 캡처/재생 중에
        # 다른 CUDA 작업이 끼어들면 깨진다("Offset increment outside graph capture").
        # MotionGen(미리보기)과 MPC(제어)가 다른 스레드에서 동시에 GPU 를 쓰므로
        # 반드시 직렬화해야 한다.
        self.gpu = threading.RLock()   # 재진입 허용 (_fk 가 잠금 안에서 불릴 수 있음)
        self.state = "PREVIEW"
        self.q = HOME.copy()
        self.js_state = js(HOME)
        self.global_path = None
        self.horizon_xyz = None
        self.path_xyz = None
        # 목표는 이제 고정 상수가 아니다. RViz 마커로 끌어서 바꿀 수 있다.
        self.goals = [np.array(g, dtype=float) for g in GOALS_XYZ]
        self.q_hold = HOME.copy()   # 대기 중 유지할 자세
        self._floor = floor_cuboids()   # 안 변하므로 한 번만

        # 인지 파이프라인 흉내: 무작위로 움직이는 장애물 + 칼만필터.
        # 진자운동을 해석적으로 미분해 속도를 얻는 건 반칙이라, 실제처럼
        # "노이즈 낀 위치만 관측하고 속도는 추정" 하는 구조로 바꾼다.
        self.mover = RandomMover([MOVER_X, 0.0, MOVER_Z], MOVER_BOX,
                                 speed=MOVER_SPD, change_every=3.0, seed=7)
        self.kf = KalmanCV(self.mover.p)
        self._preview_since = 0.0
        self._need_reseed = False
        self._resync_count = 0
        self._grace = 0
        self._teleport_req = False   # 'A 로 순간이동' 눌림

        self.get_logger().info("MotionGen 로딩...")
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=CFG, scene_model=self._scene(0.0), collision_cache=CACHE,
            use_cuda_graph=True, interpolation_dt=0.02,
            optimizer_collision_activation_distance=act))
        self.planner.warmup(enable_graph=True)

        self.get_logger().info("MPC 로딩...")
        self.mpc = ModelPredictiveControl(ModelPredictiveControlCfg.create(
            robot=CFG, scene_model=self._scene(0.0), collision_cache=CACHE,
            use_cuda_graph=True,
            optimizer_collision_activation_distance=act))
        self.mpc.setup(self.js_state)
        # ★ 이걸 안 켜면 update_goal_state 가 조용히 무시된다.
        self.mpc.enable_joint_position_tracking()
        self.mpc.disable_tool_pose_tracking()
        self.mpc.update_goal_state(js(HOME))
        self.mpc.cold_start_solve(self.js_state)

        self.quat = self.planner.compute_kinematics(js(HOME)) \
            .tool_poses.quaternion.reshape(1, 1, 1, 1, 4).clone()
        self.goal_xyz = self.goals[1].copy()   # 목표는 항상 B (A 는 출발 위치)
        self._preview_since = self.wall   # 로딩 시간을 미리보기에 포함시키지 않는다
        self.get_logger().info("준비 완료 — 미리보기 시작")

        self.pub_js = self.create_publisher(JointStateMsg, "/joint_states", 10)
        self.pub_obs = self.create_publisher(MarkerArray, "/curobo/obstacles", 1)
        self.pub_gpath = self.create_publisher(Path, "/curobo/global_path", 1)
        self.pub_path = self.create_publisher(Path, "/curobo/tcp_path", 1)
        self.pub_goal = self.create_publisher(Marker, "/curobo/goal", 1)
        # ★ cuRobo(MotionGen + MPC)를 쓰는 콜백은 반드시 같은 그룹에 둔다.
        #   CUDA graph 캡처 중에는 다른 스레드의 CUDA 연산이 하나라도 끼면 깨진다.
        #   (텐서 생성, .cpu() 전송까지 포함이라 수동 락으로는 빈틈을 못 막는다)
        gpu_group = MutuallyExclusiveCallbackGroup()
        misc = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Empty, "/curobo/approve", self._on_approve, 1,
                                 callback_group=misc)
        self.create_subscription(Point, "/curobo/set_a", lambda m: self._set_goal(0, m), 1,
                                 callback_group=misc)
        self.create_subscription(Point, "/curobo/set_b", lambda m: self._set_goal(1, m), 1,
                                 callback_group=misc)
        self.create_subscription(Empty, "/curobo/teleport", self._on_teleport, 1,
                                 callback_group=misc)

        # 목표 두 지점을 RViz 에서 끌어서 바꾼다. 로봇은 A -> B -> A ... 를 왕복한다.
        self.srv = InteractiveMarkerServer(self, "curobo/goals")
        for i, nm in enumerate(("A", "B")):
            self._make_goal_marker(nm, self.goals[i])
        self._make_approve_button()
        self._make_teleport_button()
        self.srv.applyChanges()
        self.create_timer(1.0 / CTRL_HZ, self.control_tick, callback_group=gpu_group)
        # 시각화는 제어와 분리한다. MPC 가 느려져도 장애물 마커는 계속 부드럽게 돈다.
        self.create_timer(1.0 / 30.0, self._publish, callback_group=misc)
        self.create_timer(1.0 / PREVIEW_HZ, self.preview_tick, callback_group=gpu_group)
        self.create_timer(3.0, self._check_publishers, callback_group=misc)
        # 인지(칼만)와 시각화는 순수 numpy 라 GPU 와 무관하다. 따로 돌아도 안전하다.
        self.create_timer(1.0 / PERCEPTION_HZ, self.perception_tick, callback_group=misc)

    @property
    def wall(self):
        """실제 경과 시간. 고정 스텝으로 누적하면 타이머가 밀릴 때
        장애물 움직임이 뚝뚝 끊긴다."""
        return time.time() - self.t0

    # ------------------------------------------------------------------ scene
    def perception_tick(self):
        """카메라 관측 -> 칼만 갱신. 실기의 YOLO+트래커 자리."""
        dt = 1.0 / PERCEPTION_HZ
        self.mover.step(self.wall, dt)
        z = self.mover.observe(OBS_NOISE)     # 노이즈 낀 위치만 얻는다
        self.kf.predict(dt)
        self.kf.update(z)

    def _mover_planning(self):
        """계획에 넣을 (위치, 반경).

        위치는 칼만이 준 속도로 PREDICT_T 초 앞을 본 값 -> 미리 비켜간다.
        반경은 상수다. update_obstacle_pose() 는 위치만 바꿀 수 있어서
        불확실성에 따라 반경을 키우는 방식은 쓸 수 없다
        (초기 공분산이 크면 반경 0.66m 로 굳어 모든 경로가 막힌다).
        대신 예측오차를 감안한 고정 여유를 미리 더해 둔다.
        """
        p, _ = self.kf.predict_ahead(PREDICT_T)
        return p, R_MOVER + PREDICT_MARGIN

    def _obstacles(self, t, predict=0.0):
        """predict>0 이면 칼만 예측 위치, 아니면 현재 추정 위치."""
        pts = list(self.static_obs)
        if self.moving:
            pts.append(self._mover_planning()[0] if predict > 0 else self.kf.pos)
        return pts

    def _scene(self, t):
        """구는 그대로 넘기면 충돌월드에서 사라진다. obb 로 변환해야 한다.
        바닥은 cuboid 로 유지 — 전체를 mesh 로 만들면 5배 느려진다."""
        # 계획에는 미래 위치를 쓴다. 예측이 틀릴 여지만큼 반경도 키운다.
        pts = self._obstacles(t, predict=PREDICT_T)
        sph = []
        for i, p in enumerate(pts):
            mover = self.moving and i == len(pts) - 1
            r = self._mover_planning()[1] if mover else R_OBS
            sph.append(Sphere(name=f"obs_{i}", radius=r, pose=[*p.tolist(), 1, 0, 0, 0]))
        conv = Scene(sphere=sph).get_obb_world()
        return Scene(cuboid=self._floor + list(conv.cuboid))

    def _sync_world(self, solver):
        """움직이는 장애물의 위치만 갱신한다.

        Scene 을 통째로 다시 만들어 load_collision_model 하면
        get_obb_world() 에만 96ms 가 든다 (50Hz 는 애초에 불가능).
        위치만 바뀌므로 이름으로 pose 만 바꾸면 된다.
        """
        if not self.moving:
            return
        p, _ = self._mover_planning()
        name = f"obs_{len(self.static_obs)}"      # 마지막이 움직이는 장애물
        pose = Pose.from_list([float(p[0]), float(p[1]), float(p[2]),
                               1.0, 0.0, 0.0, 0.0])
        try:
            solver.scene_collision_checker.update_obstacle_pose(name, pose)
        except Exception as e:                    # 이름이 안 맞으면 한 번만 알린다
            if not getattr(self, "_warned_sync", False):
                self._warned_sync = True
                self.get_logger().warn(f"장애물 위치 갱신 실패({name}): {e}")

    def _seed_from_path(self, G, i):
        """전역 경로에서 MPC_HORIZON 개를 잘라 [1, H, 6] 시드 텐서로 만든다.

        길이가 모자라면 마지막 자세로 채운다. shape 이 매번 같아야
        CUDA graph 가 재캡처되지 않는다.
        """
        seg = G[i: i + MPC_HORIZON]
        if len(seg) < MPC_HORIZON:
            pad = np.repeat(G[-1:], MPC_HORIZON - len(seg), axis=0)
            seg = np.vstack([seg, pad]) if len(seg) else pad
        return torch.tensor(seg[None], dtype=torch.float32, device="cuda")

    def _reseed(self, G, i):
        """전역 경로를 MPC 최적화의 초기 추정값으로 주입한다.

        호출 순서가 중요하다. update_seed_trajectory() 는 실행 버퍼의
        joint_state 궤적을 None 으로 만들면서 has_valid_next_command() 는
        True 로 남겨두기 때문에, 곧바로 optimize_next_action() 을 부르면
        None 을 참조하며 죽는다. optimize_action_sequence() 로 재구성해야 한다.
        (실측: 재구성 포함 87ms — 그래서 매 사이클 못 한다)
        """
        self.mpc.update_seed_trajectory(self._seed_from_path(G, i))
        self.mpc.optimize_action_sequence(self.js_state)

    def _fk(self, qs):
        """주의: self.gpu 를 잡은 상태에서 부르면 데드락. 밖에서만 호출할 것."""
        with self.gpu:
            st = self.planner.compute_kinematics(js(qs))
        return st.tool_poses.position.reshape(len(np.atleast_2d(qs)), -1)[:, :3] \
            .detach().cpu().numpy()

    def _check_publishers(self):
        """발행자가 둘이면 RViz 가 두 자세를 번갈아 받아 화면이 뒤엉킨다.
        이전 런치가 살아있는 게 원인이고, pkill -f rviz2 만으로는 안 죽는다."""
        n = self.count_publishers("/joint_states")
        if n > 1:
            self.get_logger().error(
                f"\n{'='*62}\n"
                f"  /joint_states 발행자가 {n}개입니다 — 화면이 뒤엉킵니다.\n"
                f"  이전 런치를 정리하세요:   bash {__file__.rsplit('/',1)[0]}/stop_demos.sh\n"
                f"{'='*62}",
                throttle_duration_sec=10.0)

    # --------------------------------------------------------------- preview
    def _plan(self, q_from, xyz):
        """q_from 에서 xyz 까지 계획. 실패하면 None. GPU 그룹에서만 부를 것."""
        pos = torch.tensor(np.asarray(xyz).reshape(1, 1, 1, 1, 3),
                           dtype=torch.float32, device="cuda")
        with self.gpu:
            self._sync_world(self.planner)
            res = self.planner.plan_pose(
                GoalToolPose(tool_frames=[TOOL], position=pos, quaternion=self.quat),
                js(q_from), max_attempts=5, enable_graph_attempt=1)
        if res is None or not bool(res.success.flatten()[0]):
            return None
        tr = res.get_interpolated_plan().position.squeeze().detach().cpu().numpy()
        return tr[None, :] if tr.ndim == 1 else tr

    def preview_tick(self):
        """PREVIEW 상태에서만 전역 경로를 다시 계획한다.

        목표는 고정이고 장애물만 움직이므로, 경로가 어떻게 바뀌는지가 그대로 보인다.
        사람은 이걸 보고 승인할지 판단한다.
        """
        if self.state != "PREVIEW":
            return
        # A 가 옮겨졌으면 먼저 로봇을 거기로 보낸다 (출발 위치를 따라간다)
        if self._teleport_req:
            self._do_teleport()
        t0 = time.time()
        traj = self._plan(self.q, self.goal_xyz)
        dt = time.time() - t0

        if traj is None:
            self.get_logger().warn(
                f"지금은 경로 없음 (장애물이 막음) [{dt*1000:.0f}ms]",
                throttle_duration_sec=2.0)
            return
        with self.lock:
            self.global_path = np.asarray(traj, dtype=float)
        self.path_xyz = self._fk(traj[::2])
        self.get_logger().info(
            f"미리보기 갱신  {len(traj)}점  {dt*1000:4.0f}ms  "
            f"— 승인: ros2 topic pub --once /curobo/approve std_msgs/msg/Empty {{}}",
            throttle_duration_sec=4.0)

        if self.auto_approve > 0 and self.wall - self._preview_since > self.auto_approve:
            self._on_approve(None)

    def _make_teleport_button(self):
        """로봇을 목표 A 자세로 즉시 옮긴다 = '출발 위치' 를 정하는 수단.

        원래 이 데모는 A <-> B 왕복만 했고 '로봇을 여기서 출발시켜라' 가 없었다.
        A 를 끌어 놓고 이 버튼을 누르면 그 자세에서 시작할 수 있다.
        """
        im = InteractiveMarker()
        im.header.frame_id = BASE
        im.name = "teleport"
        im.scale = 0.3
        im.pose.position.x, im.pose.position.y, im.pose.position.z = 0.0, 0.0, 0.70
        im.pose.orientation.w = 1.0

        c = InteractiveMarkerControl()
        c.always_visible = True
        c.interaction_mode = InteractiveMarkerControl.BUTTON
        c.name = "teleport_btn"
        box = Marker()
        box.type = Marker.CUBE
        box.scale.x, box.scale.y, box.scale.z = 0.30, 0.13, 0.012
        box.color.r, box.color.g, box.color.b, box.color.a = 0.35, 0.65, 1.0, 0.85
        c.markers.append(box)
        txt = Marker()
        txt.type = Marker.TEXT_VIEW_FACING
        txt.pose.position.z = 0.03
        txt.scale.z = 0.05
        txt.color.r = txt.color.g = txt.color.b = 1.0
        txt.color.a = 1.0
        txt.text = "A 로 되돌아가기"
        c.markers.append(txt)
        im.controls.append(c)
        self.srv.insert(im, feedback_callback=self._on_button)

    def _on_teleport(self, _msg):
        """실제 이동은 preview_tick(GPU 그룹)에서 한다.

        여기서 planner 를 부르면 다른 스레드에서 CUDA graph 를 건드리게 되어
        캡처가 깨진다. 플래그만 세우고 넘긴다.
        """
        if self.state != "PREVIEW":
            self.get_logger().warn("이동 중에는 순간이동하지 않는다 — 도착하거나 목표를 바꾼 뒤에")
            return
        self._teleport_req = True

    def _do_teleport(self):
        """목표 A 자세로 로봇을 옮긴다. GPU 그룹에서만 부를 것."""
        self._teleport_req = False
        target = self.goals[0].copy()
        traj = self._plan(self.q, target)
        if traj is None:
            self.get_logger().warn(f"A {np.round(target,3)} 로는 갈 수 없다 — 위치를 옮겨보세요")
            return
        q_new = np.asarray(traj[-1], dtype=float)
        self.q = q_new
        self.q_hold = q_new.copy()
        self.js_state = js(q_new)
        self.mpc.update_goal_state(js(q_new))
        self.mpc.cold_start_solve(self.js_state)
        self.horizon_xyz = None
        self.path_xyz = None
        with self.lock:
            self.global_path = None
        self.goal_xyz = self.goals[1].copy()      # 목표는 언제나 B
        self._preview_since = self.wall
        self.get_logger().info(f"출발 위치 A {np.round(target,3)} 로 이동 — 목표는 B")

    # --------------------------------------------------------- 승인 버튼
    def _make_approve_button(self):
        """RViz 안에서 클릭으로 승인. 터미널을 오가지 않아도 된다.

        다른 터미널에서 `ros2 topic pub` 로 승인하려면 그 터미널도
        ROS_DOMAIN_ID=84 여야 한다 (`source env84.sh`). 도메인이 다르면
        메시지가 아예 도달하지 않는데 오류도 안 난다 — 그래서 버튼을 만들었다.
        """
        self._approve_ui = None
        self._refresh_approve_button(force=True)

    def _refresh_approve_button(self, force=False):
        """상태가 바뀔 때만 다시 그린다 (매 틱 insert 하면 낭비고 깜빡인다)."""
        waiting = self.state == "PREVIEW"
        if not force and waiting == self._approve_ui:
            return
        self._approve_ui = waiting

        im = InteractiveMarker()
        im.header.frame_id = BASE
        im.name = "approve"
        im.scale = 0.3
        im.pose.position.x, im.pose.position.y, im.pose.position.z = 0.0, 0.0, 0.80
        im.pose.orientation.w = 1.0

        c = InteractiveMarkerControl()
        c.always_visible = True
        c.interaction_mode = InteractiveMarkerControl.BUTTON
        c.name = "approve_btn"

        box = Marker()
        box.type = Marker.CUBE
        box.scale.x, box.scale.y, box.scale.z = 0.30, 0.13, 0.012
        box.color.r, box.color.g, box.color.b = (1.0, 0.85, 0.1) if waiting else (0.3, 0.3, 0.34)
        box.color.a = 0.92 if waiting else 0.45
        c.markers.append(box)

        txt = Marker()
        txt.type = Marker.TEXT_VIEW_FACING
        txt.pose.position.z = 0.03
        txt.scale.z = 0.055
        txt.color.r = txt.color.g = txt.color.b = 0.05 if waiting else 0.8
        txt.color.a = 1.0
        txt.text = "클릭해서 승인" if waiting else "실행 중"
        c.markers.append(txt)

        im.controls.append(c)
        self.srv.insert(im, feedback_callback=self._on_button)
        self.srv.applyChanges()

    def _on_button(self, fb):
        if fb.event_type != fb.BUTTON_CLICK:
            return
        if fb.marker_name == "teleport":
            self._on_teleport(None)
        else:
            self._on_approve(None)

    # ------------------------------------------------------------ 목표 마커
    def _make_goal_marker(self, name, xyz):
        im = InteractiveMarker()
        im.header.frame_id = BASE
        im.name = name
        im.description = "출발 A" if name == "A" else "목표 B"
        im.scale = 0.16
        im.pose.position.x, im.pose.position.y, im.pose.position.z = map(float, xyz)
        im.pose.orientation.w = 1.0

        vis = InteractiveMarkerControl()
        vis.always_visible = True
        vis.interaction_mode = InteractiveMarkerControl.MOVE_3D
        sph = Marker()
        sph.type = Marker.SPHERE
        # 중립색·반투명. 밝은 구는 '현재 목표'(노랑=승인대기 / 초록=실행중) 하나뿐이어야
        # 어디로 가는 중인지가 한눈에 보인다. 손잡이가 초록이면 목표와 헷갈린다.
        sph.scale.x = sph.scale.y = sph.scale.z = 0.05
        sph.color.r, sph.color.g, sph.color.b = 0.85, 0.85, 0.92
        sph.color.a = 0.40
        vis.markers.append(sph)
        im.controls.append(vis)

        # 툴 방향은 홈 자세로 고정이라 위치만 옮기면 된다 (회전 링 없음)
        for qx, qy, qz in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
            c = InteractiveMarkerControl()
            c.orientation.w, c.orientation.x = 1.0, qx
            c.orientation.y, c.orientation.z = qy, qz
            n = float(np.sqrt(1.0 + qx * qx + qy * qy + qz * qz))
            c.orientation.w /= n
            c.orientation.x /= n
            c.orientation.y /= n
            c.orientation.z /= n
            c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            c.name = f"move_{qx}{qy}{qz}"
            im.controls.append(c)

        self.srv.insert(im, feedback_callback=self._on_goal_feedback)

    def _on_goal_feedback(self, fb):
        if fb.event_type != fb.POSE_UPDATE:
            return
        i = 0 if fb.marker_name == "A" else 1
        p = fb.pose.position
        self._apply_goal(i, np.array([p.x, p.y, p.z]))

    def _set_goal(self, i, msg):
        """토픽으로 목표를 지정. RViz 마커도 같이 옮긴다."""
        xyz = np.array([msg.x, msg.y, msg.z])
        self._apply_goal(i, xyz)
        pm = PoseMsg()
        pm.position.x, pm.position.y, pm.position.z = map(float, xyz)
        pm.orientation.w = 1.0
        self.srv.setPose("A" if i == 0 else "B", pm)
        self.srv.applyChanges()

    def _apply_goal(self, i, xyz):
        """마커가 움직였다.

            A = 출발 위치 — 로봇이 그 자세로 따라간다 (순간이동)
            B = 목표       — 경로가 그쪽으로 다시 그려진다

        ★ 실행 중에 둘 중 뭐가 바뀌든 미리보기로 되돌린다.
          사람이 승인한 것은 '그때 보여준 그 경로' 지 '아무 데나 가도 된다' 가 아니다.
          승인 없이 새 목적지로 계속 가면 승인 게이트가 무의미해진다.
        """
        with self.lock:
            self.goals[i] = xyz
            was_exec = self.state == "EXECUTE"
            self.state = "PREVIEW"
            self.global_path = None
            if i == 1:
                self.goal_xyz = xyz.copy()
        if i == 0:
            self._teleport_req = True       # 실제 이동은 preview_tick(GPU)에서
        self.horizon_xyz = None
        self.path_xyz = None                # 옛 목표로 가던 선을 지운다
        self.q_hold = self.q.copy()
        self._preview_since = self.wall
        if was_exec:
            self.get_logger().warn(
                f"{'출발 A' if i == 0 else '목표 B'} 가 이동 중에 바뀌었다 "
                f"— 실행을 멈추고 다시 승인받는다")

    # --------------------------------------------------------- 승인 버튼
    def _make_approve_button(self):
        """RViz 안에서 클릭으로 승인. 터미널을 오가지 않아도 된다.

        다른 터미널에서 `ros2 topic pub` 로 승인하려면 그 터미널도
        ROS_DOMAIN_ID=84 여야 한다 (`source env84.sh`). 도메인이 다르면
        메시지가 아예 도달하지 않는데 오류도 안 난다 — 그래서 버튼을 만들었다.
        """
        self._approve_ui = None
        self._refresh_approve_button(force=True)

    def _refresh_approve_button(self, force=False):
        """상태가 바뀔 때만 다시 그린다 (매 틱 insert 하면 낭비고 깜빡인다)."""
        waiting = self.state == "PREVIEW"
        if not force and waiting == self._approve_ui:
            return
        self._approve_ui = waiting

        im = InteractiveMarker()
        im.header.frame_id = BASE
        im.name = "approve"
        im.scale = 0.3
        im.pose.position.x, im.pose.position.y, im.pose.position.z = 0.0, 0.0, 0.80
        im.pose.orientation.w = 1.0

        c = InteractiveMarkerControl()
        c.always_visible = True
        c.interaction_mode = InteractiveMarkerControl.BUTTON
        c.name = "approve_btn"

        box = Marker()
        box.type = Marker.CUBE
        box.scale.x, box.scale.y, box.scale.z = 0.30, 0.13, 0.012
        box.color.r, box.color.g, box.color.b = (1.0, 0.85, 0.1) if waiting else (0.3, 0.3, 0.34)
        box.color.a = 0.92 if waiting else 0.45
        c.markers.append(box)

        txt = Marker()
        txt.type = Marker.TEXT_VIEW_FACING
        txt.pose.position.z = 0.03
        txt.scale.z = 0.055
        txt.color.r = txt.color.g = txt.color.b = 0.05 if waiting else 0.8
        txt.color.a = 1.0
        txt.text = "클릭해서 승인" if waiting else "실행 중"
        c.markers.append(txt)

        im.controls.append(c)
        self.srv.insert(im, feedback_callback=self._on_button)
        self.srv.applyChanges()

    def _on_button(self, fb):
        if fb.event_type != fb.BUTTON_CLICK:
            return
        if fb.marker_name == "teleport":
            self._on_teleport(None)
        else:
            self._on_approve(None)

    # ------------------------------------------------------------ 목표 마커
    def _make_goal_marker(self, name, xyz):
        im = InteractiveMarker()
        im.header.frame_id = BASE
        im.name = name
        im.description = "출발 A" if name == "A" else "목표 B"
        im.scale = 0.16
        im.pose.position.x, im.pose.position.y, im.pose.position.z = map(float, xyz)
        im.pose.orientation.w = 1.0

        vis = InteractiveMarkerControl()
        vis.always_visible = True
        vis.interaction_mode = InteractiveMarkerControl.MOVE_3D
        sph = Marker()
        sph.type = Marker.SPHERE
        # 중립색·반투명. 밝은 구는 '현재 목표'(노랑=승인대기 / 초록=실행중) 하나뿐이어야
        # 어디로 가는 중인지가 한눈에 보인다. 손잡이가 초록이면 목표와 헷갈린다.
        sph.scale.x = sph.scale.y = sph.scale.z = 0.05
        sph.color.r, sph.color.g, sph.color.b = 0.85, 0.85, 0.92
        sph.color.a = 0.40
        vis.markers.append(sph)
        im.controls.append(vis)

        # 툴 방향은 홈 자세로 고정이라 위치만 옮기면 된다 (회전 링 없음)
        for qx, qy, qz in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
            c = InteractiveMarkerControl()
            c.orientation.w, c.orientation.x = 1.0, qx
            c.orientation.y, c.orientation.z = qy, qz
            n = float(np.sqrt(1.0 + qx * qx + qy * qy + qz * qz))
            c.orientation.w /= n
            c.orientation.x /= n
            c.orientation.y /= n
            c.orientation.z /= n
            c.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            c.name = f"move_{qx}{qy}{qz}"
            im.controls.append(c)

        self.srv.insert(im, feedback_callback=self._on_goal_feedback)

    def _on_goal_feedback(self, fb):
        if fb.event_type != fb.POSE_UPDATE:
            return
        i = 0 if fb.marker_name == "A" else 1
        p = fb.pose.position
        self._apply_goal(i, np.array([p.x, p.y, p.z]))

    def _set_goal(self, i, msg):
        """토픽으로 목표를 지정. RViz 마커도 같이 옮긴다."""
        xyz = np.array([msg.x, msg.y, msg.z])
        self._apply_goal(i, xyz)
        pm = PoseMsg()
        pm.position.x, pm.position.y, pm.position.z = map(float, xyz)
        pm.orientation.w = 1.0
        self.srv.setPose("A" if i == 0 else "B", pm)
        self.srv.applyChanges()

    def _on_approve(self, _msg):
        with self.lock:
            if self.state != "PREVIEW" or self.global_path is None:
                return
            self.state = "EXECUTE"
            self.path_idx = 0
            self._need_reseed = True     # 출발 시 전역 경로를 시드로 주입
            self._resync_count = 0       # 실행마다 초기화 (누적되면 안 된다)
            self._grace = 0
        self.get_logger().info("★ 승인됨 — 실행 시작 (이동 중에도 실시간 회피)")

    # --------------------------------------------------------------- control
    def control_tick(self):
        if self.state != "EXECUTE":
            if not self.hold_yield:
                return          # 승인 전엔 완전히 정지
            # 대기 중 MPC 는 제자리 유지라 고빈도가 필요 없다.
            # 50Hz 로 GPU 락을 잡으면 MotionGen(미리보기)이 굶어서
            # 계획이 380ms -> 1600ms 로 늘어난다.
            self._hold_tick = getattr(self, "_hold_tick", 0) + 1
            if self._hold_tick % 10:
                return          # 50Hz 중 9/10 은 즉시 반환 -> preview_tick 이 굶지 않는다
            # 승인 대기 중에도 가만히 굳어 있으면 안 된다.
            # 목표를 "지금 자세 유지" 로 주면 MPC 는 제자리에 머물다가,
            # 장애물이 다가오면 충돌 비용 때문에 스스로 비켜난다.
            # 위험이 지나가면 다시 원래 자세로 돌아온다.
            with self.gpu:
                self._sync_world(self.mpc)
                self.mpc.update_goal_state(js(self.q_hold))
                res = self.mpc.optimize_next_action(self.js_state)
            if res.next_action is not None:
                self.js_state = res.next_action.clone()
                self.q = self.js_state.position.detach().cpu().numpy().ravel()
            return

        with self.lock:
            G = self.global_path
        q_now = self.js_state.position.detach().cpu().numpy().ravel()
        d_all = np.linalg.norm(G - q_now, axis=1)
        i_close = int(d_all.argmin())
        dev_now = float(d_all[i_close])       # 진짜 이탈거리 (가장 가까운 점까지)
        self.path_idx = max(self.path_idx, i_close)   # 진행은 뒤로 안 감
        tgt = G[min(self.path_idx + LOOKAHEAD, len(G) - 1)]
        self._exec_tick = getattr(self, "_exec_tick", 0) + 1
        # 전역 경로를 MPC 최적화의 '초기 추정값' 으로 다시 넣는다.
        #
        # MPC 는 L-BFGS(경사하강)라 시작점이 결과를 거의 결정한다. 목표점 하나만
        # 주면 초기추정이 직선이 되어 장애물을 관통하려다 밀리고 뒤틀린다.
        # 전역 경로 구간을 시드로 주면 "어느 쪽으로 우회할지" 가 상속되고,
        # 최적화는 "얼마나 휠지" 만 실시간으로 정한다.
        #
        # 매 사이클 재시드하면 reinitialize() 가 직전 해를 버려서 warm start
        # 이점을 잃는다. 그래서 주기적으로만 넣는다 (PG-MPPI: 로컬 100Hz / prior 10Hz).
        periodic = self.reseed_every > 0 and self._exec_tick % self.reseed_every == 1
        with self.gpu:
            self._sync_world(self.mpc)
            if periodic or self._need_reseed:
                self._need_reseed = False
                self._reseed(G, self.path_idx)
            self.mpc.update_goal_state(js(tgt))
            res = self.mpc.optimize_next_action(self.js_state)
        if res.next_action is not None:
            self.js_state = res.next_action.clone()
            self.q = self.js_state.position.detach().cpu().numpy().ravel()
        # 전역 경로에서 너무 벗어났으면 MPC 가 길을 잃은 것이다.
        # 그대로 두면 팔이 뒤틀린 자세로 굳고, 다음 미리보기도 거기서 시작해
        # 경로가 계속 이상해진다. 끊고 다시 계획하는 게 맞다.
        # 이탈 판정.
        #
        # 주의: MPC 가 장애물을 피하려고 휘는 것 자체는 '정상'이다. 우리가 원하는
        # 동작이다. 이탈은 그 휨이 되돌아오지 못할 때만 문제다. 그래서
        #   1) 이탈이 '연속'일 때만 센다 (한 번 붙었다 떨어지는 건 무시)
        #   2) 재시드 직후엔 유예를 준다 — 재시드는 87ms 걸리는데 틱은 20ms 라
        #      바로 다음 틱에 효과가 나타날 수 없다. 유예 없이 세면
        #      "재시드 -> 아직 그대로 -> 중단" 으로 항상 즉시 중단된다.
        if dev_now <= MAX_DEVIATION:
            self._resync_count = 0          # 돌아왔으면 리셋
            self._grace = 0
        else:
            if self._grace > 0:             # 방금 재시드했다. 효과를 기다린다
                self._grace -= 1
                return
            self._resync_count += 1
            if self._resync_count < RESYNC_TRIES:
                self._need_reseed = True    # 다음 틱에 재시드
                self._grace = GRACE_TICKS
                self.get_logger().warn(
                    f"이탈 {dev_now:.2f} rad — 전역 경로로 재시드 "
                    f"({self._resync_count}/{RESYNC_TRIES})",
                    throttle_duration_sec=2.0)
                return
            self.get_logger().warn(
                f"전역 경로에서 {dev_now:.2f} rad 이탈 — 재시드 {RESYNC_TRIES}회 실패, "
                f"실행 중단하고 재계획")
            self.q_hold = self.q.copy()
            self.horizon_xyz = None
            with self.lock:
                self.state = "PREVIEW"
                self.global_path = None
            self._preview_since = self.wall
            return

        # horizon 시각화용 FK 는 매 틱 할 필요가 없다 (화면은 30Hz 로 별도 발행)
        self._tick = getattr(self, "_tick", 0) + 1
        if res.robot_state_sequence is not None and self._tick % 5 == 0:
            hq = res.robot_state_sequence.joint_state.position
            xyz = self._fk(hq.reshape(-1, 6).detach().cpu().numpy())
            self.horizon_xyz = xyz
            self._tcp_now = xyz[0]
        p = getattr(self, "_tcp_now", self.goal_xyz + 1.0)
        if np.linalg.norm(p - self.goal_xyz) < 0.03 or \
                self.path_idx >= len(G) - 2:
            # 왕복하지 않는다. B 에 도착하면 거기 서 있고,
            # 사람이 A(출발)나 B(목표)를 옮겨서 다음 동작을 정한다.
            self.get_logger().info(
                f"B 도착 (오차 {np.linalg.norm(p-self.goal_xyz)*1000:.0f}mm) "
                f"— A 를 옮기면 그 위치에서 다시 출발, B 를 옮기면 새 목표")
            self.horizon_xyz = None
            with self.lock:
                self.state = "PREVIEW"
                self.global_path = None
            self._preview_since = self.wall


    # --------------------------------------------------------------- publish
    def _marker(self, i, ns, mtype, xyz, scale, rgba, now):
        m = Marker()
        m.header.frame_id, m.header.stamp = BASE, now
        m.ns, m.id, m.type, m.action = ns, i, mtype, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _path_msg(self, xyz, now):
        p = Path()
        p.header.frame_id, p.header.stamp = BASE, now
        for q in xyz:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, q)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        return p

    def _publish(self):
        now = self.get_clock().now().to_msg()
        self._refresh_approve_button()   # 상태 안 바뀌었으면 즉시 리턴한다

        msg = JointStateMsg()
        msg.header.stamp = now
        msg.name = JOINTS
        msg.position = [float(v) for v in self.q]
        self.pub_js.publish(msg)

        ma = MarkerArray()
        # 장애물은 하나당 마커 하나만 그린다.
        # 예전엔 "현재 위치" 와 "예측 영역" 을 겹쳐 그렸는데 두 개가 따로 놀아
        # 오히려 헷갈리고 지저분했다.
        # 위치는 칼만이 준 현재 추정, 크기는 planner 가 실제로 쓰는 반경.
        obs = self._obstacles(self.wall)
        for i, p in enumerate(obs):
            mover = self.moving and i == len(obs) - 1
            d = 2.0 * (self._mover_planning()[1] if mover else R_OBS)
            col = (1.0, 0.5, 0.1, 0.75) if mover else (0.85, 0.2, 0.2, 0.6)
            ma.markers.append(self._marker(i, "obs", Marker.CUBE, p, (d, d, d), col, now))
        self.pub_obs.publish(ma)

        if self.path_xyz is not None:
            self.pub_gpath.publish(self._path_msg(self.path_xyz, now))
        if self.horizon_xyz is not None:
            self.pub_path.publish(self._path_msg(self.horizon_xyz, now))

        # 승인 대기 중이면 목표를 크게/노랗게 깜빡여 상태를 구분한다
        waiting = self.state == "PREVIEW"
        col = (1.0, 0.9, 0.1, 0.95) if waiting else (0.2, 1.0, 0.3, 0.95)
        s = 0.075 if waiting else 0.06
        self.pub_goal.publish(
            self._marker(0, "goal", Marker.SPHERE, self.goal_xyz, (s, s, s), col, now))


def main():
    rclpy.init()
    node = HybridDemo()
    ex = MultiThreadedExecutor(num_threads=4)
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
