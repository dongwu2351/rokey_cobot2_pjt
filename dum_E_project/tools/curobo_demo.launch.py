#!/usr/bin/env python3
"""
cuRobo 경로계획 RViz 데모.

  source /opt/ros/humble/setup.bash
  source ~/cobot_ws/install/setup.bash
  ros2 launch curobo_demo.launch.py

RViz 에 보이는 것
  회색 판    작업대
  빨간 구    정적 장애물
  주황 구    움직이는 장애물  <- 이게 지나갈 때 경로가 바뀌는지 보세요
  라임 선    계획된 TCP 경로
  라임 구    목표 지점
  반투명 구  로봇 collision sphere (91개)
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
    xacro_file = (
        Path(get_package_share_directory("dsr_description2"))
        / "xacro" / "m0609.urdf.xacro"
    )
    robot_description = Command(
        ["xacro ", str(xacro_file), " color:=white", " model:=m0609"]
    )

    return LaunchDescription([
        # 실기기 tracking test 와 섞이지 않도록 도메인을 분리한다.
        # 다른 터미널에서 ros2 topic 을 쓰려면 `source env84.sh` 먼저.
        SetEnvironmentVariable("ROS_DOMAIN_ID", "84"),
        DeclareLaunchArgument("moving", default_value="false",
                              description="움직이는 장애물 사용 (정적 테스트가 먼저)"),
        DeclareLaunchArgument("continuous", default_value="false",
                              description="실행 중 재계획 (false=한 번 계획하고 끝까지 실행)"),
        DeclareLaunchArgument("scene_mode", default_value="obb",
                              description="obb(6배 빠름) 또는 mesh(정확)"),
        DeclareLaunchArgument("speed", default_value="1.5",
                              description="궤적 재생 배속"),
        DeclareLaunchArgument("n_static", default_value="1",
                              description="정적 장애물 개수"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(HERE / "curobo_demo.rviz")],
        ),
        # cuRobo 는 venv 안에만 있으므로 venv 의 python 으로 직접 실행한다.
        ExecuteProcess(
            cmd=[VENV_PY, str(HERE / "curobo_rviz_demo.py"),
                 "--ros-args",
                 "-p", ["moving:=", LaunchConfiguration("moving")],
                 "-p", ["continuous:=", LaunchConfiguration("continuous")],
                 "-p", ["speed:=", LaunchConfiguration("speed")],
                 "-p", ["scene_mode:=", LaunchConfiguration("scene_mode")],
                 "-p", ["n_static:=", LaunchConfiguration("n_static")]],
            output="screen",
        ),
        # 로봇 collision sphere 도 같이 (시스템 python 으로 충분)
        ExecuteProcess(
            cmd=["python3", str(HERE / "sphere_viz_node.py"),
                 "--ros-args",
                 "-p", f"spheres_yaml:={HERE / 'm0609.yml'}",
                 "-p", "alpha:=0.20"],
            output="screen",
        ),
            ExecuteProcess(
            cmd=["python3", str(HERE / "gripper_viz_node.py"),
                 "--ros-args", "-p", "opening:=0.06", "-p", "yaw:=0.0"],
            output="screen",
        ),
    ])
