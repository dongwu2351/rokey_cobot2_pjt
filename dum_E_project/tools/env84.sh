# DUM-E RViz 데모용 환경. 실기기 tracking test(기본 도메인)와 분리한다.
#   source /home/rokey/cobot2_ws/dum_E_project/tools/env84.sh
set +u
source /opt/ros/humble/setup.bash
[ -f "$HOME/cobot_ws/install/setup.bash" ] && source "$HOME/cobot_ws/install/setup.bash"
set -u 2>/dev/null || true
export ROS_DOMAIN_ID=84
echo "ROS_DOMAIN_ID=84 (DUM-E RViz 데모 전용 / 실기기와 분리됨)"
