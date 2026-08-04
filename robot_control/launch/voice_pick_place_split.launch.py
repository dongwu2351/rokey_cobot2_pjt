from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_set",
                default_value="tools",
                description="Detection/voice target set: tools or fruits",
            ),
            Node(
                package="voice_processing",
                executable="get_keyword",
                name="get_keyword",
                output="screen",
                parameters=[
                    {"target_set": LaunchConfiguration("target_set")}
                ],
            ),
            Node(
                package="object_detection",
                executable="object_detection",
                name="object_detection",
                output="screen",
                parameters=[
                    {"target_set": LaunchConfiguration("target_set")}
                ],
            ),
            Node(
                package="robot_control",
                executable="robot_control",
                name="voice_pick_place",
                output="screen",
                parameters=[
                    {"target_set": LaunchConfiguration("target_set")}
                ],
            ),
        ]
    )
