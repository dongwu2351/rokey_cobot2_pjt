#!/bin/bash
# Webcam pick&place 실행 스크립트
#
#   bash ~/cobot2_ws_1/scripts/run_webcam_pnp.sh [옵션...]
#
# 자주 쓰는 조합:
#   (데모)  bash run_webcam_pnp.sh --live --free-wrist --hand-place --far-start --speed-scale 0.25
#   (최종)  bash run_webcam_pnp.sh --live --free-wrist --hand-place
#   (안전)  bash run_webcam_pnp.sh                # dry-run: 로봇 안 움직임
#
# 옵션:
#   --live              실제 모션 (없으면 dry-run)
#   --hand-place        파지 후 손 위에 배달 (없으면 D435i 화면 클릭으로 놓기)
#   --free-wrist        이동 중 6축 손목 활용
#   --far-start         접근 전 반대편으로 이동 (회피 시연용 긴 경로)
#   --speed-scale 0.25  전체 속도 배율 (생략 = 1.0)
#   --target hammer     타겟 클래스 (기본 hammer)
#
# 창 단축키: S 시작 | SPACE 정지 | 1 슬로우(x0.25) | 2 풀스피드 | R 복구
#            O/C 그리퍼 열기/닫기 | H 홈 | D dry-run 토글 | ESC 종료
#
# 사전 조건: DSR 드라이버 + realsense가 떠 있어야 함 (아래 주석 참고)
#   드라이버:  ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.1.100 port:=12345 model:=m0609 name:=dsr01
#   카메라:    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true enable_sync:=false rgb_camera.color_profile:=1280x720x30 depth_module.depth_profile:=848x480x30
#   (aligned_depth 토픽이 안 나와도 됨 - 앱이 자체 정합함. raw depth만 나오면 OK)
set -e
source /opt/ros/humble/setup.bash
source "$HOME/cobot_ws/install/setup.bash"       # DSR 드라이버 언더레이
source "$HOME/cobot2_ws_1/install/setup.bash"
exec ros2 run pick_and_place_voice webcam_pick_place "$@"
