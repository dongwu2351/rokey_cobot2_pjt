#!/usr/bin/env python3
"""
RG2 그리퍼를 RViz 에 그린다.

dsr_description2 에는 Robotiq 2F-85 만 있고 RG2 모델이 없어서, OnRobot 공식
데이터시트 치수로 직접 그린다. link_6 프레임에 붙이므로 TF 만 있으면 따라간다.

  전체 길이(마운트면->핑거팁) 213,  본체 132(높이) x 75(폭) x 36(깊이),
  최대 개구 110(안쪽)/124(바깥),  손가락 두께 24            [mm]

실측 flange->TCP = 238.13mm = 툴체인저 25 + RG2 213 이라, link_6 로컬 z 로
  z 0~25     툴 체인저
  z 25~157   RG2 본체
  z 157~238  손가락
  z 238      TCP

파라미터
  opening   손가락 벌림 [m] 0~0.11 (기본 0.06)
  yaw       그리퍼가 링크 z 축을 중심으로 얼마나 돌아 달렸는지 [rad]
            (실기 장착 방향에 맞춰서 조정할 것)

실행:  python3 gripper_viz_node.py
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray

LINK = "link_6"

TOOLCHANGER_LEN = 0.025
BODY_LEN = 0.132
BODY_W, BODY_D = 0.075, 0.036
FINGER_LEN = 0.081
FINGER_T, FINGER_D = 0.014, 0.024
TCP_Z = 0.23813


def yaw_quat(yaw):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))  # w,x,y,z


class GripperViz(Node):
    def __init__(self):
        super().__init__("gripper_viz")
        self.declare_parameter("opening", 0.06)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("alpha", 0.9)
        self.pub = self.create_publisher(MarkerArray, "/gripper_viz", 1)
        self.create_timer(0.5, self.tick)
        self.get_logger().info("RG2 그리퍼 마커 발행 -> /gripper_viz")

    def _m(self, i, mtype, xyz, scale, rgba, yaw):
        m = Marker()
        m.header.frame_id = LINK
        m.header.stamp = Time().to_msg()      # 0 = TF 최신값 사용
        m.ns, m.id = "rg2", i
        m.type, m.action = mtype, Marker.ADD
        m.frame_locked = True                 # 관절이 움직여도 따라붙게
        m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
        w, x, y, z = yaw_quat(yaw)
        m.pose.orientation.w = w
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = x, y, z
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def tick(self):
        opening = float(self.get_parameter("opening").value)
        yaw = float(self.get_parameter("yaw").value)
        a = float(self.get_parameter("alpha").value)
        metal = (0.30, 0.32, 0.36, a)
        dark = (0.15, 0.15, 0.17, a)

        z_body = TOOLCHANGER_LEN + BODY_LEN / 2.0
        z_fing = TOOLCHANGER_LEN + BODY_LEN + FINGER_LEN / 2.0
        # 손가락 중심 간격 = 개구 + 손가락 두께
        half = (opening + FINGER_T) / 2.0
        c, s = math.cos(yaw), math.sin(yaw)

        ma = MarkerArray()
        ma.markers.append(self._m(
            0, Marker.CYLINDER, (0.0, 0.0, TOOLCHANGER_LEN / 2.0),
            (0.064, 0.064, TOOLCHANGER_LEN), metal, yaw))
        ma.markers.append(self._m(
            1, Marker.CUBE, (0.0, 0.0, z_body),
            (BODY_W, BODY_D, BODY_LEN), metal, yaw))
        for k, sign in enumerate((-1.0, 1.0)):
            ma.markers.append(self._m(
                2 + k, Marker.CUBE,
                (sign * half * c, sign * half * s, z_fing),
                (FINGER_T, FINGER_D, FINGER_LEN), dark, yaw))
        # TCP 표시
        ma.markers.append(self._m(
            4, Marker.SPHERE, (0.0, 0.0, TCP_Z),
            (0.018, 0.018, 0.018), (1.0, 0.85, 0.1, 1.0), 0.0))
        self.pub.publish(ma)


def main():
    rclpy.init()
    node = GripperViz()
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
