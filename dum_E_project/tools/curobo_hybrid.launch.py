#!/usr/bin/env python3
"""하이브리드 데모: 미리보기 -> 승인 -> 실시간 회피 실행.

  ros2 launch curobo_hybrid.launch.py

  승인:  ros2 topic pub --once /curobo/approve std_msgs/msg/Empty {}
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            SetEnvironmentVariable)
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).parent
VENV_PY = "/home/rokey/curobo_env/bin/python"


def generate_launch_description():
    xacro_file = (Path(get_package_share_directory("dsr_description2"))
                  / "xacro" / "m0609.urdf.xacro")
    robot_description = Command(
        ["xacro ", str(xacro_file), " color:=white", " model:=m0609"])

    return LaunchDescription([
        # 실기기 tracking test 와 섞이지 않도록 도메인을 분리한다.
        # 다른 터미널에서 ros2 topic 을 쓰려면 `source env84.sh` 먼저.
        SetEnvironmentVariable("ROS_DOMAIN_ID", "84"),
        DeclareLaunchArgument("moving", default_value="true"),
        DeclareLaunchArgument("n_static", default_value="1"),
        DeclareLaunchArgument("hold_yield", default_value="true",
                              description="승인 대기 중에도 장애물이 오면 비켜설지. false=완전정지"),
        DeclareLaunchArgument("reseed_every", default_value="0",
                              description="전역경로를 MPC 시드로 재주입하는 주기(틱). 0=끔, 5=10Hz"),
        DeclareLaunchArgument("activation_dist", default_value="0.01",
                              description="충돌비용 활성거리[m]. 키우면 안전거리 우선"),
        DeclareLaunchArgument("auto_approve", default_value="0.0",
                              description="0 이면 수동 승인, >0 이면 N초 뒤 자동"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", str(HERE / "curobo_hybrid.rviz")]),
        ExecuteProcess(
            cmd=[VENV_PY, str(HERE / "curobo_hybrid_demo.py"), "--ros-args",
                 "-p", ["moving:=", LaunchConfiguration("moving")],
                 "-p", ["n_static:=", LaunchConfiguration("n_static")],
                 "-p", ["auto_approve:=", LaunchConfiguration("auto_approve")],
                 "-p", ["reseed_every:=", LaunchConfiguration("reseed_every")],
                 "-p", ["hold_yield:=", LaunchConfiguration("hold_yield")],
                 "-p", ["activation_dist:=", LaunchConfiguration("activation_dist")]],
            output="screen"),
        # 로봇 충돌 구 91개 (그리퍼 8개 포함). RViz 에서 켜고 끌 수 있다.
        ExecuteProcess(
            cmd=["python3", str(HERE / "sphere_viz_node.py"), "--ros-args",
                 "-p", f"spheres_yaml:={HERE / 'm0609.yml'}", "-p", "alpha:=0.22"],
            output="screen"),
        ExecuteProcess(
            cmd=["python3", str(HERE / "gripper_viz_node.py"),
                 "--ros-args", "-p", "opening:=0.06"],
            output="screen"),
    ])
