#!/usr/bin/env bash

set -euo pipefail

container_name="object_detection"

if ! docker inspect "${container_name}" >/dev/null 2>&1; then
    echo "Container '${container_name}' does not exist." >&2
    exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "${container_name}")" != "true" ]; then
    docker start "${container_name}" >/dev/null
fi

exec docker exec -it \
    -e ROS_DOMAIN_ID=85 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    "${container_name}" \
    bash -lc '
        source /opt/ros/humble/setup.bash
        source /home/ros2_ws/install/setup.bash
        echo "Docker YOLO: domain=${ROS_DOMAIN_ID}, rmw=${RMW_IMPLEMENTATION}"
        exec ros2 run object_detection object_detection \
            --ros-args -p target_set:=fruits
    '
