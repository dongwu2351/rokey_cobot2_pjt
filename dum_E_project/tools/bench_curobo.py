#!/usr/bin/env python3
"""
cuRobo MotionPlanner 성능 측정 — 결정 게이트.

RTX 4060 에서 M0609 경로계획이 몇 ms 나오는지가 CP8(동적 회피)·CP9(움직이는 통)
일정을 좌우한다.

  < 50ms    -> 20Hz 재계획. 연속 회피 가능
  50~150ms  -> 7~20Hz. 0.1m/s 장애물에 충분
  > 200ms   -> MoveIt 과 차이 없음. 아키텍처 재검토

같이 재는 것:
  - update_world (장애물 갱신) 비용 : 30Hz 루프에서 매 사이클 호출한다
  - warm start 효과                 : 직전 궤적을 seed 로 쓰면 연속성이 생기는가
  - VRAM                            : YOLO 와 8GB 를 나눠 써야 한다

실행:  ~/curobo_env/bin/python bench_curobo.py
"""
import time

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, Sphere, Cuboid
from curobo.types import GoalToolPose, JointState

CFG = "/home/rokey/cobot2_ws/dum_E_project/tools/m0609.yml"
TOOL = "tcp"


def vram(tag=""):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    print(f"  VRAM {tag:22s} allocated {a:7.1f} MB | reserved {r:7.1f} MB")


def make_scene(n_obstacles: int) -> Scene:
    """작업대 + 구형 장애물 n 개.

    ★ 함정: 작업대를 로봇 베이스 밑까지 깔면 base_link 구와 충돌해서
      시작 자세부터 불가능 판정이 나고 plan_pose 가 통째로 None 을 반환한다.
      실제로 로봇은 작업대 위에 볼트로 고정돼 있으니 그 부분은 충돌이 아니다.
      -> 작업대를 로봇 앞쪽으로만 두거나, base_link 충돌을 따로 처리해야 한다.
    """
    table = Cuboid(name="table", dims=[0.8, 1.0, 0.05],
                   pose=[0.55, 0.0, -0.025, 1, 0, 0, 0])  # 로봇 앞, 윗면 z=0
    rng = np.random.default_rng(0)
    spheres = []
    for i in range(n_obstacles):
        p = rng.uniform([0.30, -0.35, 0.15], [0.65, 0.35, 0.55])
        spheres.append(Sphere(name=f"obs_{i}", radius=0.06,
                              pose=[*p.tolist(), 1, 0, 0, 0]))
    # ★ get_mesh_world() 필수. 안 하면 sphere 장애물이 조용히 전부 무시된다.
    #   (cuRobo 충돌월드는 cuboid/mesh/voxel 만 받는다)
    return Scene(cuboid=[table], sphere=spheres).get_mesh_world()


def stats(name, ts):
    t = np.array(ts) * 1000.0
    print(f"  {name:26s} 중앙값 {np.median(t):7.1f} ms | "
          f"평균 {t.mean():7.1f} | p95 {np.percentile(t,95):7.1f} | "
          f"최대 {t.max():7.1f}")


def main():
    print("=" * 74)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram("시작")

    N_OBS = 5
    t0 = time.time()
    cfg = MotionPlannerCfg.create(
        robot=CFG,
        scene_model=make_scene(N_OBS),
        # 30Hz 루프에서 장애물을 갱신하려면 캐시를 미리 잡아둬야 한다
        collision_cache={"cuboid": 32, "mesh": 64},
        use_cuda_graph=True,
        interpolation_dt=0.02,      # 50Hz 궤적 샘플링
    )
    planner = MotionPlanner(cfg)
    print(f"\nMotionPlanner 생성 : {time.time()-t0:.1f}s   (장애물 {N_OBS}개)")
    vram("생성 후")

    t0 = time.time()
    planner.warmup(enable_graph=True)
    print(f"warmup             : {time.time()-t0:.1f}s")
    vram("warmup 후")

    # ---- 목표 자세: 현재 자세에서 FK 로 만들어 도달 가능성 보장 ----
    home = np.deg2rad([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
    rng = np.random.default_rng(1)

    start = JointState.from_position(
        torch.tensor(home[None, :], dtype=torch.float32, device="cuda"))

    # GoalToolPose 는 5D [B, H, L, G, 3] 을 요구한다 (배치/호라이즌/링크/goalset/xyz)
    goals = []
    for _ in range(30):
        q = home + rng.uniform(-0.6, 0.6, 6)
        st = planner.compute_kinematics(
            JointState.from_position(
                torch.tensor(q[None, :], dtype=torch.float32, device="cuda")))
        goals.append((st.tool_poses.position.reshape(1, 1, 1, 1, 3).clone(),
                      st.tool_poses.quaternion.reshape(1, 1, 1, 1, 4).clone()))

    # ---- plan_pose 반복 ----
    # 두 가지 설정을 비교한다.
    #   빠른 모드 : 실행 중 재계획용. 그래프 플래너 없이 궤적최적화만 1회.
    #   안전 모드 : 최초 계획용. 실패하면 재시도하고 그래프 플래너까지 동원.
    def run(label, attempts, graph, pl=None, n_warm=8):
        """cuRobo 0.8 은 런타임 JIT 이라 첫 수 회는 컴파일 비용이 섞인다.
        n_warm 회를 먼저 돌려 버리고 그 뒤부터 잰다."""
        pl = pl or planner
        for pos, quat in goals[:n_warm]:
            pl.plan_pose(GoalToolPose(tool_frames=[TOOL], position=pos,
                                      quaternion=quat),
                         start, max_attempts=attempts, enable_graph_attempt=graph)
        torch.cuda.synchronize()

        ts, ok, err = [], 0, []
        for pos, quat in goals:
            g = GoalToolPose(tool_frames=[TOOL], position=pos, quaternion=quat)
            torch.cuda.synchronize()
            t0 = time.time()
            res = pl.plan_pose(g, start, max_attempts=attempts,
                               enable_graph_attempt=graph)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
            if res is not None and bool(res.success.flatten()[0]):
                ok += 1
                err.append(float(res.position_error.flatten()[0]))
        stats(label, ts)
        e = f" | 오차 {np.median(err)*1000:.2f} mm" if err else ""
        print(f"  {'  -> 성공률':26s} {ok}/{len(goals)} ({ok/len(goals)*100:3.0f}%){e}")
        return float(np.median(ts) * 1000)

    print("\n[1] plan_pose  (워밍업 8회 후 30회 측정)")
    run("빠른 모드 (재계획용)", 1, 0)
    run("안전 모드 (최초계획용)", 5, 1)

    # ---- update_world 비용 (30Hz 루프의 핵심) ----
    print("\n[2] update_world  (장애물 위치 갱신 — 30Hz 루프에서 매번 호출)")
    ts = []
    for i in range(50):
        sc = make_scene(N_OBS)
        torch.cuda.synchronize()
        t0 = time.time()
        planner.update_world(sc)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    stats("update_world", ts)

    # ---- 장애물 개수 스케일링 ----
    print("\n[3] 장애물 개수별 plan_pose 중앙값 (빠른 모드)")
    for n in (0, 5, 10, 20):
        planner.update_world(make_scene(n))
        for pos, quat in goals[:5]:      # 씬 바뀔 때마다 워밍업
            planner.plan_pose(GoalToolPose(tool_frames=[TOOL], position=pos,
                                           quaternion=quat),
                              start, max_attempts=1, enable_graph_attempt=0)
        ts = []
        for pos, quat in goals[:15]:
            g = GoalToolPose(tool_frames=[TOOL], position=pos, quaternion=quat)
            torch.cuda.synchronize()
            t0 = time.time()
            planner.plan_pose(g, start, max_attempts=1, enable_graph_attempt=0)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        print(f"  장애물 {n:2d}개 : {np.median(np.array(ts)*1000):7.1f} ms")
    vram("최종")
    planner.destroy()

    # ---- seed 개수 튜닝: 속도 vs 성공률 ----
    # 재계획 주기를 올리려면 여기가 제일 큰 손잡이다.
    print("\n[4] trajopt seed 개수별 (빠른 모드) — 속도/성공률 트레이드오프")
    for ns in (2, 4, 8, 16):
        cfg2 = MotionPlannerCfg.create(
            robot=CFG, scene_model=make_scene(N_OBS),
            collision_cache={"cuboid": 32, "mesh": 64},
            use_cuda_graph=True, interpolation_dt=0.02,
            num_trajopt_seeds=ns,
        )
        p2 = MotionPlanner(cfg2)
        p2.warmup(enable_graph=True)
        print(f"\n  --- num_trajopt_seeds = {ns} ---")
        run(f"  seeds={ns}", 1, 0, pl=p2)
        p2.destroy()

    print("=" * 74)


if __name__ == "__main__":
    main()
