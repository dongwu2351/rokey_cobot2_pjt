#!/usr/bin/env python3
"""
로봇에 설정된 TCP / 툴(무게) 을 읽어온다.

cuRobo robot_cfg 의 tool_frames 와 attached_object 오프셋을 실제 값으로 맞추려면
컨트롤러에 뭐가 설정돼 있는지부터 알아야 한다.

주의: Doosan 은 mm / deg 단위, cuRobo·ROS 는 m / rad 단위다.

실행 (로봇 연결 필요):
    source /opt/ros/humble/setup.bash
    source ~/cobot_ws/install/setup.bash
    python3 query_robot_tool.py
"""
import numpy as np
import rclpy

import DR_init

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main():
    rclpy.init()
    node = rclpy.create_node("query_robot_tool", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (  # noqa: E402  (노드 준비 후에 import 해야 함)
        get_current_posx,
        get_current_posj,
        get_tcp,
        get_tool,
        get_tool_force,
    )

    print("=" * 60)

    try:
        print(f"활성 TCP  이름 : {get_tcp()}")
    except Exception as e:  # noqa: BLE001
        print(f"get_tcp 실패: {e}")

    try:
        print(f"활성 TOOL 이름 : {get_tool()}   (무게·무게중심·관성)")
    except Exception as e:  # noqa: BLE001
        print(f"get_tool 실패: {e}")

    try:
        posj = get_current_posj()
        posx = np.asarray(get_current_posx()[0], dtype=float)
        si = np.r_[posx[:3] / 1000.0, np.deg2rad(posx[3:])]
        print(f"\n현재 관절 [deg] : {np.round(posj, 2)}")
        print(f"현재 TCP  [mm,deg]: {np.round(posx, 2)}")
        print(f"현재 TCP  [m,rad] : {np.round(si, 4)}")
    except Exception as e:  # noqa: BLE001
        print(f"자세 읽기 실패: {e}")

    try:
        print(f"\n외력(tool) : {np.round(get_tool_force(), 2)}")
        print("  -> 아무것도 안 잡은 상태에서 0 근처가 아니면 툴 무게 설정이 틀린 것")
    except Exception as e:  # noqa: BLE001
        print(f"get_tool_force 실패: {e}")

    print("=" * 60)
    print("""
다음으로 필요한 값:
  1. 위 TCP 이름에 해당하는 x,y,z,rx,ry,rz  (펜던트에서 확인)
     -> cuRobo 의 flange->TCP 오프셋이 된다
  2. 툴 무게 [kg] 와 무게중심 [mm]
     -> 로봇 컨트롤러의 중력보상/힘추정에 쓰인다 (cuRobo 는 안 씀)
  3. RG2 외형 치수 (플랜지에서 손가락 끝까지 길이, 폭)
     -> 그리퍼 충돌 구를 만들어야 cuRobo 가 그리퍼 충돌을 감지한다
""")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
