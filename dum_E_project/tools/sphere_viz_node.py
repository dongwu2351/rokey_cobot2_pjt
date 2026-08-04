#!/usr/bin/env python3
"""
collision_spheres.yaml 을 RViz 에 MarkerArray 로 띄운다.

구를 각 링크 프레임(header.frame_id = link 이름)에 그대로 발행하므로
TF 만 살아 있으면 관절을 움직여도 구가 알아서 따라간다.

실행 (시스템 python3 로 충분 — cuRobo 환경 불필요):
    ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=virtual model:=m0609
    python3 sphere_viz_node.py --ros-args -p spheres_yaml:=./collision_spheres.yaml

RViz 에서 Add -> MarkerArray -> Topic: /collision_spheres
"""
import sys
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray


class SphereViz(Node):
    def __init__(self):
        super().__init__("sphere_viz")
        default = str(Path(__file__).parent / "collision_spheres.yaml")
        self.declare_parameter("spheres_yaml", default)
        self.declare_parameter("alpha", 0.35)

        path = Path(self.get_parameter("spheres_yaml").value)
        if not path.exists():
            self.get_logger().error(f"파일 없음: {path}")
            sys.exit(1)

        data = yaml.safe_load(path.read_text())
        # 두 가지 형식을 다 받는다.
        #   collision_spheres.yaml  -> 링크 구만 (그리퍼 없음)
        #   m0609.yml               -> 그리퍼까지 합쳐진 최종본  <- 이쪽을 써야 한다
        if "robot_cfg" in data:
            self.spheres = data["robot_cfg"]["kinematics"]["collision_spheres"]
        else:
            self.spheres = data["collision_spheres"]
        self.alpha = float(self.get_parameter("alpha").value)

        n = sum(len(v) for v in self.spheres.values())
        self.get_logger().info(f"{len(self.spheres)}개 링크, 구 {n}개 로드: {path}")

        self.pub = self.create_publisher(MarkerArray, "/collision_spheres", 1)
        self.create_timer(0.5, self.publish_markers)

    def publish_markers(self):
        ma = MarkerArray()
        # stamp 를 0 으로 두면 TF 가 "가장 최근" 변환을 쓴다.
        # 현재 시각을 찍으면 그 시점 TF 가 아직 안 와서 RViz 가 마커를 버린다
        # (Fixed Frame 인 base_link 만 변환 없이 그려져서 그것만 보이는 증상).
        now = Time().to_msg()
        # 링크마다 다른 색 -> 어느 구가 어느 링크인지 눈으로 구분
        palette = [
            (0.20, 0.85, 0.35), (0.25, 0.60, 0.95), (0.95, 0.75, 0.20),
            (0.90, 0.35, 0.40), (0.65, 0.45, 0.90), (0.30, 0.85, 0.85),
            (0.95, 0.55, 0.25),
        ]
        mid = 0
        for li, (link, lst) in enumerate(self.spheres.items()):
            r, g, b = palette[li % len(palette)]
            for s in lst:
                m = Marker()
                m.header.frame_id = link
                m.header.stamp = now
                m.ns = link
                m.id = mid
                mid += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                # 관절이 움직일 때마다 RViz 가 최신 TF 로 다시 배치하게 한다
                m.frame_locked = True
                cx, cy, cz = s["center"]
                m.pose.position.x = float(cx)
                m.pose.position.y = float(cy)
                m.pose.position.z = float(cz)
                m.pose.orientation.w = 1.0
                d = 2.0 * float(s["radius"])
                m.scale.x = m.scale.y = m.scale.z = d
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, self.alpha
                ma.markers.append(m)
        self.pub.publish(ma)


def main():
    rclpy.init()
    node = SphereViz()
    try:
        rclpy.spin(node)
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
