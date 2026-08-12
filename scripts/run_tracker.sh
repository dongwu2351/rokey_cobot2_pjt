#!/usr/bin/env bash
# Start the tracker with the interpreter that can see cuRobo.
#
#   bash ~/cobot2_ws/scripts/run_tracker.sh
#
# Why not `ros2 run pick_and_place_voice robot_control`?
# The installed console script carries a hardcoded `#!/usr/bin/python3`, so it
# runs under system python no matter what venv is active - and cuRobo lives
# only in ~/curobo_env. Tracking works fine that way; the orbit key (B) does
# not, because `import curobo` fails.
#
# cuRobo is deliberately NOT installed into system python: it drags in numpy 2
# and friends, which break ROS's cv_bridge. The venv keeps them apart.
set -euo pipefail

WS="$HOME/cobot2_ws"
VENV_PYTHON="$HOME/curobo_env/bin/python"
ENTRY="$WS/install/pick_and_place_voice/lib/pick_and_place_voice/robot_control"

[ -x "$VENV_PYTHON" ] || {
    echo "cuRobo venv missing at $VENV_PYTHON" >&2
    echo "run: bash $WS/dum_E_project/tools/install_curobo.sh" >&2
    exit 1
}
[ -f "$ENTRY" ] || {
    echo "not built: $ENTRY" >&2
    echo "run: cd $WS && colcon build --packages-select pick_and_place_voice" >&2
    exit 1
}

# ROS setup scripts reference unset variables; -u would abort on them.
set +u
source /opt/ros/humble/setup.bash
source "$HOME/cobot_ws/install/setup.bash"     # driver underlay, must come first
source "$WS/install/setup.bash"
set -u

echo "tracker: $VENV_PYTHON   ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
exec "$VENV_PYTHON" "$ENTRY" "$@"
