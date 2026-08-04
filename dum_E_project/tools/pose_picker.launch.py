#!/usr/bin/env python3
"""시작·목표를 RViz 마커로 직접 집어서 경로를 확인한다.

  ros2 launch pose_picker.launch.py

RViz 왼쪽 위 툴바에서 'Interact'(손가락 아이콘)를 고른 다음
파랑(시작)/초록(목표) 마커를 끌면 된다.

  재생:   ros2 topic pub --once /curobo/play  std_msgs/msg/Empty {}
  맞바꿈: ros2 topic pub --once /curobo/swap  std_msgs/msg/Empty {}
  좌표지정: ros2 topic pub --once /curobo/set_goal geometry_msgs/msg/Point \
              "{x: 0.5, y: 0.1, z: 0.2}"
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
        DeclareLaunchArgument("n_static", default_value="1"),
        DeclareLaunchArgument("moving", default_value="true",
                              description="주황 장애물을 움직일지. false=고정(조합 비교용)"),
        DeclareLaunchArgument("mover_y", default_value="0.0",
                              description="moving=false 일 때 장애물을 고정할 y. "
                                          "moving=true 면 이 y 를 중심으로 왕복한다"),
        DeclareLaunchArgument("rate", default_value="2.0",
                              description="재계획 주기[Hz]. 계획이 340ms 라 2~3이 적당"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", str(HERE / "pose_picker.rviz")]),
        ExecuteProcess(
            cmd=[VENV_PY, str(HERE / "pose_picker_node.py"), "--ros-args",
                 "-p", ["n_static:=", LaunchConfiguration("n_static")],
                 "-p", ["mover_y:=", LaunchConfiguration("mover_y")],
                 "-p", ["moving:=", LaunchConfiguration("moving")],
                 "-p", ["rate:=", LaunchConfiguration("rate")]],
            output="screen"),
        ExecuteProcess(
            cmd=["python3", str(HERE / "sphere_viz_node.py"), "--ros-args",
                 "-p", f"spheres_yaml:={HERE / 'm0609.yml'}", "-p", "alpha:=0.22"],
            output="screen"),
        ExecuteProcess(
            cmd=["python3", str(HERE / "gripper_viz_node.py"),
                 "--ros-args", "-p", "opening:=0.06"],
            output="screen"),
    ])
