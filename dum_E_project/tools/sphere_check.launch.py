#!/usr/bin/env python3
"""
collision sphere 육안 검증용 최소 구성.

Doosan 드라이버나 로봇 연결이 전혀 필요 없다. URDF -> TF -> 구 마커만 띄운다.
슬라이더로 관절을 돌려가며 구가 팔을 제대로 덮는지 확인하는 것이 목적.

  source /opt/ros/humble/setup.bash
  source ~/cobot_ws/install/setup.bash        # dsr_description2 (메시) 때문에 필요
  ros2 launch sphere_check.launch.py
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import Command
from launch_ros.actions import Node

HERE = Path(__file__).parent


def generate_launch_description():
    xacro_file = (
        Path(get_package_share_directory("dsr_description2"))
        / "xacro"
        / "m0609.urdf.xacro"
    )
    robot_description = Command(
        ["xacro ", str(xacro_file), " color:=white", " model:=m0609"]
    )

    return LaunchDescription([
        # 실기기 tracking test 와 섞이지 않도록 도메인을 분리한다.
        # 다른 터미널에서 ros2 topic 을 쓰려면 `source env84.sh` 먼저.
        SetEnvironmentVariable("ROS_DOMAIN_ID", "84"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        # 슬라이더로 J1~J6 를 직접 돌려본다
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(HERE / "sphere_check.rviz")],
        ),
        # launch_ros 의 Node 는 package 없이 절대경로만 주면 조용히 안 뜨는 경우가 있다.
        # ExecuteProcess 로 명시적으로 실행한다.
        ExecuteProcess(
            cmd=[
                "python3", str(HERE / "sphere_viz_node.py"),
                "--ros-args",
                "-p", f"spheres_yaml:={HERE / 'm0609.yml'}",
                "-p", "alpha:=0.35",
            ],
            output="screen",
        ),
            ExecuteProcess(
            cmd=["python3", str(HERE / "gripper_viz_node.py"),
                 "--ros-args", "-p", "opening:=0.06", "-p", "yaw:=0.0"],
            output="screen",
        ),
    ])
