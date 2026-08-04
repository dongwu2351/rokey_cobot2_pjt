from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start only the Host-side nodes when YOLO runs in Docker."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_set",
                default_value="fruits",
                description="Voice/robot target set: tools or fruits",
            ),
            Node(
                package="voice_processing",
                executable="get_keyword",
                name="get_keyword",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {"target_set": LaunchConfiguration("target_set")}
                ],
            ),
            Node(
                package="robot_control",
                executable="robot_control",
                name="voice_pick_place",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {"target_set": LaunchConfiguration("target_set")}
                ],
            ),
        ]
    )
