#!/usr/bin/env python3
"""
경로 탐침 — 로봇을 손으로 움직이며 그 자세에서의 계획 경로를 실시간으로 본다.

데모는 로봇을 자동으로 움직이므로 "이 자세에서는 어떤 경로가 나오나" 를 볼 수 없다.
이 노드는 반대로 동작한다:

    joint_state_publisher_gui 로 사람이 관절을 돌린다
        -> 이 노드가 그 자세에서 목표까지 계획해서 경로를 그린다

경로가 크게 우회하는 자세를 손으로 찾아낼 수 있다. 계획 품질을 눈으로 디버깅하는 용도.

  /joint_states        구독 (joint_state_publisher_gui 가 발행)
  /curobo/global_path  발행 — 현재 자세에서 목표까지의 경로
  /curobo/obstacles    발행
  /curobo/goal         발행
  /curobo/next_goal    구독 — 목표를 다음 것으로 넘김 (std_msgs/Empty)

실행:
  ros2 launch path_probe.launch.py
  ros2 topic pub --once /curobo/next_goal std_msgs/msg/Empty {}
"""
import threading
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, Sphere
from curobo.types import GoalToolPose, JointState

from curobo_hybrid_demo import (CACHE, GOALS_XYZ, MOVER_X, MOVER_Z, PREDICT_MARGIN,
                                R_MOVER, js)
from curobo_rviz_demo import BASE, CFG, HOME, JOINTS, R_OBS, STATIC_OBS, TOOL, floor_cuboids


class PathProbe(Node):
    def __init__(self):
        super().__init__("path_probe")
        self.declare_parameter("rate", 3.0)
        self.declare_parameter("n_static", 1)
        self.declare_parameter("mover_y", 0.0)      # 움직이는 장애물을 여기 고정해 둔다
        self.declare_parameter("num_seeds", 0)      # 0 이면 cuRobo 기본값
        rate = float(self.get_parameter("rate").value)
        n = int(self.get_parameter("n_static").value)
        self.static_obs = STATIC_OBS[:max(0, min(n, len(STATIC_OBS)))]
        self.mover_y = float(self.get_parameter("mover_y").value)
        ns = int(self.get_parameter("num_seeds").value)

        self.q = HOME.copy()
        self.goal_i = 0
        self.lock = threading.Lock()

        self.get_logger().info("cuRobo 로딩...")
        kw = dict(robot=CFG, scene_model=self._scene(), collision_cache=CACHE,
                  use_cuda_graph=True, interpolation_dt=0.02)
        if ns > 0:
            kw["num_trajopt_seeds"] = ns
        self.planner = MotionPlanner(MotionPlannerCfg.create(**kw))
        self.planner.warmup(enable_graph=True)
        self.quat = self.planner.compute_kinematics(js(HOME)) \
            .tool_poses.quaternion.reshape(1, 1, 1, 1, 4).clone()
        self.get_logger().info(
            "준비 완료 — joint_state_publisher_gui 로 로봇을 움직여보세요.\n"
            "  목표 바꾸기: ros2 topic pub --once /curobo/next_goal std_msgs/msg/Empty {}")

        self.pub_path = self.create_publisher(Path, "/curobo/global_path", 1)
        self.pub_obs = self.create_publisher(MarkerArray, "/curobo/obstacles", 1)
        self.pub_goal = self.create_publisher(Marker, "/curobo/goal", 1)
        self.create_subscription(JointStateMsg, "/joint_states", self._on_js, 10)
        self.create_subscription(Empty, "/curobo/next_goal", self._next_goal, 1)
        self.create_timer(1.0 / rate, self.tick)

    # ------------------------------------------------------------------ scene
    def _obstacles(self):
        return list(self.static_obs) + [np.array([MOVER_X, self.mover_y, MOVER_Z])]

    def _radius(self, i, n):
        return (R_MOVER + PREDICT_MARGIN) if i == n - 1 else R_OBS

    def _scene(self):
        pts = self._obstacles()
        sph = [Sphere(name=f"obs_{i}", radius=self._radius(i, len(pts)),
                      pose=[*p.tolist(), 1, 0, 0, 0]) for i, p in enumerate(pts)]
        return Scene(cuboid=floor_cuboids() + list(Scene(sphere=sph).get_obb_world().cuboid))

    # ------------------------------------------------------------------ input
    def _on_js(self, msg):
        """사람이 슬라이더로 돌린 자세를 받는다."""
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = np.array([msg.position[idx[j]] for j in JOINTS], dtype=float)
        except (KeyError, IndexError):
            return
        with self.lock:
            self.q = q

    def _next_goal(self, _msg):
        self.goal_i = (self.goal_i + 1) % len(GOALS_XYZ)
        self.get_logger().info(f"목표 -> {np.round(GOALS_XYZ[self.goal_i], 3)}")

    # ------------------------------------------------------------------- plan
    def tick(self):
        with self.lock:
            q = self.q.copy()
        g = GOALS_XYZ[self.goal_i]

        t0 = time.time()
        res = self.planner.plan_pose(
            GoalToolPose(
                tool_frames=[TOOL],
                position=torch.tensor(g.reshape(1, 1, 1, 1, 3),
                                      dtype=torch.float32, device="cuda"),
                quaternion=self.quat),
            js(q), max_attempts=5, enable_graph_attempt=1)
        dt = time.time() - t0

        self._publish_scene(g)
        if res is None or not bool(res.success.flatten()[0]):
            self.get_logger().warn(f"이 자세에서는 경로 없음 [{dt*1000:.0f}ms]",
                                   throttle_duration_sec=1.5)
            self.pub_path.publish(Path(header=self._hdr()))   # 선 지우기
            return

        traj = res.get_interpolated_plan().position.squeeze().detach().cpu().numpy()
        if traj.ndim == 1:
            traj = traj[None, :]
        st = self.planner.compute_kinematics(js(traj))
        xyz = st.tool_poses.position.reshape(len(traj), -1)[:, :3].detach().cpu().numpy()

        # 직선거리 대비 경로길이 = 우회 정도. 1.0~2.0 이 정상, 3배 넘으면 크게 돈 것.
        L = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        d = float(np.linalg.norm(g - xyz[0]))
        ratio = L / d if d > 1e-6 else 0.0
        self.get_logger().info(
            f"경로 {len(traj):3d}점 {dt*1000:4.0f}ms | 직선 {d:.3f} 경로 {L:.3f} "
            f"= {ratio:4.1f}x" + ("   <-- 크게 우회" if ratio > 3.0 else ""),
            throttle_duration_sec=0.8)

        path = Path(header=self._hdr())
        for p in xyz:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    # ---------------------------------------------------------------- publish
    def _hdr(self):
        p = Path().header
        p.frame_id = BASE
        p.stamp = self.get_clock().now().to_msg()
        return p

    def _publish_scene(self, g):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        pts = self._obstacles()
        for i, p in enumerate(pts):
            m = Marker()
            m.header.frame_id, m.header.stamp = BASE, now
            m.ns, m.id, m.type, m.action = "obs", i, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
            m.pose.orientation.w = 1.0
            d = 2.0 * self._radius(i, len(pts))
            m.scale.x = m.scale.y = m.scale.z = d
            mover = i == len(pts) - 1
            m.color.r, m.color.g, m.color.b = (1.0, 0.5, 0.1) if mover else (0.85, 0.2, 0.2)
            m.color.a = 0.7
            ma.markers.append(m)
        self.pub_obs.publish(ma)

        gm = Marker()
        gm.header.frame_id, gm.header.stamp = BASE, now
        gm.ns, gm.id, gm.type, gm.action = "goal", 0, Marker.SPHERE, Marker.ADD
        gm.pose.position.x, gm.pose.position.y, gm.pose.position.z = map(float, g)
        gm.pose.orientation.w = 1.0
        gm.scale.x = gm.scale.y = gm.scale.z = 0.06
        gm.color.r, gm.color.g, gm.color.b, gm.color.a = 0.2, 1.0, 0.3, 0.95
        self.pub_goal.publish(gm)


def main():
    rclpy.init()
    node = PathProbe()
    try:
        rclpy.spin(node)
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
