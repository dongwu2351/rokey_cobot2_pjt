#!/bin/bash
# 워크스페이스 표준화 마무리 — 로봇 앱과 코파일럿을 멈춘 뒤 실행할 것
set -e
cd ~/cobot2_ws_1
echo "[1/4] ROS 패키지를 src/ 로 이동"
for p in pick_and_place_voice od_msg; do
    [ -d "$p" ] && mv "$p" src/ && echo "     src/ <- $p"
done
echo "[2/4] 예전 빌드 산출물 제거 (심볼릭 링크가 옛 경로를 가리킴)"
rm -rf build install log
echo "[3/4] 재빌드"
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash 2>/dev/null || true
colcon build --symlink-install
echo "[4/4] 확인"
source install/setup.bash
colcon list
echo
echo "완료. 이제 평소대로 실행하면 됩니다:"
echo "  bash ~/cobot2_ws_1/scripts/run_webcam_pnp.sh --live --free-wrist --hand-place"
