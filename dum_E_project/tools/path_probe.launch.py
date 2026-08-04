#!/usr/bin/env python3
"""경로 탐침 — 슬라이더로 로봇을 움직이며 그 자세에서의 계획 경로를 본다.

  ros2 launch path_probe.launch.py
  ros2 topic pub --once /curobo/next_goal std_msgs/msg/Empty {}
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
        DeclareLaunchArgument("mover_y", default_value="0.0",
                              description="움직이는 장애물을 이 y 에 고정"),
        DeclareLaunchArgument("n_static", default_value="1"),
        DeclareLaunchArgument("num_seeds", default_value="0",
                              description="trajopt seed 수. 0=cuRobo 기본"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        # ★ 사람이 여기서 관절을 돌린다. 이 데모에선 이게 유일한 joint_states 발행자다.
        Node(package="joint_state_publisher_gui",
             executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", str(HERE / "curobo_hybrid.rviz")]),
        ExecuteProcess(
            cmd=[VENV_PY, str(HERE / "path_probe_node.py"), "--ros-args",
                 "-p", ["mover_y:=", LaunchConfiguration("mover_y")],
                 "-p", ["n_static:=", LaunchConfiguration("n_static")],
                 "-p", ["num_seeds:=", LaunchConfiguration("num_seeds")]],
            output="screen"),
        ExecuteProcess(
            cmd=["python3", str(HERE / "gripper_viz_node.py"),
                 "--ros-args", "-p", "opening:=0.06"], output="screen"),
        ExecuteProcess(
            cmd=["python3", str(HERE / "sphere_viz_node.py"), "--ros-args",
                 "-p", f"spheres_yaml:={HERE / 'm0609.yml'}", "-p", "alpha:=0.20"],
            output="screen"),
    ])
