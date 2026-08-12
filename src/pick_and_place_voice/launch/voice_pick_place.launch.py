from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="pick_and_place_voice",
                executable="object_detection",
                name="object_detection",
                output="screen",
            ),
            Node(
                package="pick_and_place_voice",
                executable="get_keyword",
                name="get_keyword",
                output="screen",
            ),
            Node(
                package="pick_and_place_voice",
                executable="robot_control",
                name="voice_pick_place",
                output="screen",
            ),
        ]
    )
