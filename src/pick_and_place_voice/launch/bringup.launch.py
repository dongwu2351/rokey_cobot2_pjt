"""Bring up everything the tracking controller depends on: the RealSense
camera and the Doosan driver.

    ros2 launch pick_and_place_voice bringup.launch.py

The controller itself is deliberately NOT started here - it opens an OpenCV
window and reads keystrokes, so it wants its own terminal:

    ros2 run pick_and_place_voice robot_control

Override the defaults if the cell differs, e.g.
    ros2 launch pick_and_place_voice bringup.launch.py host:=192.168.1.101 rviz:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    model = LaunchConfiguration("model")
    mode = LaunchConfiguration("mode")
    rviz = LaunchConfiguration("rviz")

    declared = [
        DeclareLaunchArgument("host", default_value="192.168.1.100"),
        DeclareLaunchArgument("port", default_value="12345"),
        DeclareLaunchArgument("model", default_value="m0609"),
        DeclareLaunchArgument("mode", default_value="real"),
        # dsr_bringup2 ships only the *_rviz launch file; its own `gui` arg is
        # what actually decides whether RViz opens.
        DeclareLaunchArgument(
            "rviz", default_value="false", description="open RViz too"
        ),
        # rs_launch.py defaults these to "0,0,0", which lets the driver pick -
        # and it picks small. Name them explicitly so the picture is the same
        # size every run instead of depending on the driver's choice.
        DeclareLaunchArgument(
            "color_profile",
            default_value="1280x720x30",
            description="RGB stream WxHxFPS",
        ),
        DeclareLaunchArgument(
            "depth_profile",
            default_value="848x480x30",
            description="depth stream WxHxFPS (aligned up to the colour size)",
        ),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        # aligned depth is required: the controller samples the depth frame at
        # colour-image pixel coordinates.
        launch_arguments={
            "align_depth.enable": "true",
            "enable_sync": "true",
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
        }.items(),
    )

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("dsr_bringup2"),
                    "launch",
                    "dsr_bringup2_rviz.launch.py",
                ]
            )
        ),
        launch_arguments={
            "host": host,
            "port": port,
            "model": model,
            "mode": mode,
            "gui": rviz,
        }.items(),
    )

    return LaunchDescription(declared + [camera, driver])
