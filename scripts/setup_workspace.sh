#!/usr/bin/env bash
# Set up both workspaces from scratch on a fresh Ubuntu 22.04 + ROS 2 Humble box.
#
#   bash scripts/setup_workspace.sh
#
# Idempotent: safe to re-run. Does NOT touch the robot or move anything.
set -euo pipefail

DRIVER_WS="${DRIVER_WS:-$HOME/cobot_ws}"
DRIVER_REPO="${DRIVER_REPO:-https://github.com/ROKEY-SPARK/doosan-robot2_2026}"
DRIVER_COMMIT="${DRIVER_COMMIT:-a8fdcdc}"
APP_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
say "Checking prerequisites"
[ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble not found at /opt/ros/humble.
Install it first: https://docs.ros.org/en/humble/Installation.html"
command -v colcon >/dev/null || die "colcon missing: sudo apt install python3-colcon-common-extensions"
echo "OK: ROS 2 Humble, colcon, $(python3 --version)"

# ---------------------------------------------------------------- apt deps
say "Installing ROS / system packages (sudo)"
sudo apt-get update
sudo apt-get install -y \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-xacro \
    ros-humble-rqt-image-view \
    python3-colcon-common-extensions \
    python3-rosdep \
    portaudio19-dev \
    git

# ---------------------------------------------------------------- python deps
say "Installing Python packages (pinned - see requirements.txt)"
pip3 install --user -r "$APP_WS/requirements.txt"

# ---------------------------------------------------------------- driver ws
say "Setting up the Doosan driver workspace at $DRIVER_WS"
mkdir -p "$DRIVER_WS/src"
if [ ! -d "$DRIVER_WS/src/doosan-robot2/.git" ]; then
    git clone "$DRIVER_REPO" "$DRIVER_WS/src/doosan-robot2"
fi
git -C "$DRIVER_WS/src/doosan-robot2" fetch --all --tags --quiet || true
git -C "$DRIVER_WS/src/doosan-robot2" checkout --quiet "$DRIVER_COMMIT" \
    || echo "WARNING: could not pin to $DRIVER_COMMIT; staying on $(git -C "$DRIVER_WS/src/doosan-robot2" rev-parse --short HEAD)"

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init || true
fi
rosdep update || true
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
rosdep install --from-paths "$DRIVER_WS/src" --ignore-src -r -y || \
    echo "WARNING: rosdep reported problems; continuing"

say "Building the driver workspace (this takes a while)"
( cd "$DRIVER_WS" && colcon build --symlink-install )

# ---------------------------------------------------------------- app ws
say "Building the application workspace at $APP_WS"
# shellcheck disable=SC1091
source "$DRIVER_WS/install/setup.bash"
( cd "$APP_WS" && colcon build --symlink-install )

# ---------------------------------------------------------------- env file
ENV_FILE="$APP_WS/pick_and_place_voice/resource/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$APP_WS/pick_and_place_voice/resource/.env.example" "$ENV_FILE"
    echo "Created $ENV_FILE from the template - put a real OPENAI_API_KEY in it"
    echo "if you want the voice path. Keyboard tracking works without it."
fi

cat <<EOF

$(printf '\033[1;32m')Setup complete.$(printf '\033[0m')

Source both workspaces in every new terminal, driver FIRST:

    source $DRIVER_WS/install/setup.bash
    source $APP_WS/install/setup.bash

Then, in two terminals:

    # 1) camera + robot driver
    ros2 launch pick_and_place_voice bringup.launch.py

    # 2) tracking / snatching controller (needs the GUI window focused)
    ros2 run pick_and_place_voice robot_control

$(printf '\033[1;33m')Before running on real hardware, read the "Per-cell calibration"
section of README.md - the hand-eye calibration and the workspace
limits in robot_control.py are specific to one physical setup.$(printf '\033[0m')
EOF
