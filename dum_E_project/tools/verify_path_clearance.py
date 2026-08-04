#!/usr/bin/env python3
"""
cuRobo 가 만든 경로가 정말로 장애물을 피하는지 숫자로 검증한다.

RViz 에서는 얇은 선이 반투명 구의 앞을 지나가는지 뚫고 가는지 눈으로 구분이 안 된다.
여기서는 궤적의 모든 시점에 대해 로봇 구 91개(그리퍼 8개 포함) vs 장애물 구의
최소거리를 잰다.

  음수  -> 실제로 파고든 것 (버그)
  양수  -> 통과. 값이 곧 여유거리.

배치: 로봇과 목표 사이에 장애물 벽을 세워서 반드시 우회하게 만든다.

실행:  ~/curobo_env/bin/python verify_path_clearance.py
"""
import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene, Sphere
from curobo.types import GoalToolPose, JointState

CFG = "/home/rokey/cobot2_ws/dum_E_project/tools/m0609.yml"
HOME = np.deg2rad([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
R_OBS = 0.07

TABLE = dict(name="table", dims=[1.0, 1.2, 0.05], pose=[0.65, 0.0, -0.025, 1, 0, 0, 0])

# 홈(TCP 약 0.37,0,0.19)과 목표 구역(x 0.60~0.75) 사이를 가로막는 벽
OBS = np.array([
    [0.50, -0.22, 0.28],
    [0.50, 0.00, 0.30],
    [0.50, 0.22, 0.28],
    [0.52, 0.00, 0.50],
])

# 목표: 장애물 너머 작업대 위
GOALS_XYZ = np.array([
    [0.66, -0.18, 0.16],
    [0.70, 0.00, 0.14],
    [0.66, 0.18, 0.16],
    [0.62, -0.10, 0.28],
    [0.62, 0.10, 0.28],
])


def main():
    # ★ get_mesh_world() 없이 Scene(sphere=...) 을 넘기면 구 장애물이 전부 무시된다.
    #   cuRobo 의 충돌월드는 cuboid / mesh / voxel 만 받는다.
    # 작업대는 충돌 대상에서 뺀다 (로봇이 거기 고정돼 있어 구분이 불가능).
    scene = Scene(
        sphere=[Sphere(name=f"o{i}", radius=R_OBS, pose=[*p.tolist(), 1, 0, 0, 0])
                for i, p in enumerate(OBS)],
    ).get_mesh_world()
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=CFG, scene_model=scene,
        collision_cache={"cuboid": 32, "mesh": 64},
        use_cuda_graph=True, interpolation_dt=0.02,
    ))
    planner.warmup(enable_graph=True)

    start = JointState.from_position(
        torch.tensor(HOME[None, :], dtype=torch.float32, device="cuda"))

    # 홈 자세의 방향을 그대로 목표 방향으로 쓴다 (툴이 아래를 향한 상태).
    st_home = planner.compute_kinematics(start)
    quat = st_home.tool_poses.quaternion.reshape(1, 1, 1, 1, 4).clone()
    p_home = st_home.tool_poses.position.reshape(-1)[:3].detach().cpu().numpy()
    print(f"홈 TCP(link_6): {np.round(p_home, 3)}\n")

    worst_overall = np.inf
    n_ok = 0

    for i, gxyz in enumerate(GOALS_XYZ):
        pos = torch.tensor(gxyz.reshape(1, 1, 1, 1, 3),
                           dtype=torch.float32, device="cuda")
        res = planner.plan_pose(
            GoalToolPose(tool_frames=["tcp"], position=pos, quaternion=quat),
            start, max_attempts=5, enable_graph_attempt=1)
        if res is None or not bool(res.success.flatten()[0]):
            print(f"  목표 {i} {np.round(gxyz,2)} : 계획 실패")
            continue

        traj = res.get_interpolated_plan().position.squeeze().detach().cpu().numpy()
        if traj.ndim == 1:
            traj = traj[None, :]

        # 궤적 전체를 한 번에 FK -> 로봇 구 위치 (그리퍼 포함)
        st = planner.compute_kinematics(JointState.from_position(
            torch.tensor(traj, dtype=torch.float32, device="cuda")))
        sp = st.robot_spheres.detach().cpu().numpy().reshape(len(traj), -1, 4)
        c, r = sp[..., :3], sp[..., 3]

        d = np.linalg.norm(c[:, :, None, :] - OBS[None, None, :, :], axis=-1)
        d = d - r[:, :, None] - R_OBS
        d[r <= 0] = np.inf                     # attached_object 미사용 슬롯 제외

        worst = float(d.min())
        worst_overall = min(worst_overall, worst)
        n_ok += 1
        ti, si, oi = np.unravel_index(np.argmin(d), d.shape)
        flag = "   <-- 파고듦!" if worst < 0 else ""
        print(f"  목표 {i} {np.round(gxyz,2)} : {len(traj):3d}점  "
              f"최소여유 {worst*1000:7.1f} mm  (t={ti*0.02:.2f}s, 구#{si}, 장애물#{oi})"
              f"{flag}")

    print(f"\n성공한 궤적 {n_ok}/{len(GOALS_XYZ)}")
    if n_ok:
        print(f"전체 최소여유 : {worst_overall*1000:.1f} mm")
        print("음수가 없으면 cuRobo 가 그리퍼까지 포함해서 장애물을 피하고 있다는 뜻이다.")


if __name__ == "__main__":
    main()
