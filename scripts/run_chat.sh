#!/bin/bash
# LLM 채팅으로 로봇에게 해머 배달 시키기
#
#   bash ~/cobot2_ws_1/scripts/run_chat.sh
#
# 메인 앱(run_webcam_pnp.sh)이 떠 있는 상태에서 별도 터미널로 실행.
# "해머 좀 챙겨줘" -> 로봇 출발 (S와 동일) / "멈춰" -> 정지 (SPACE와 동일)
# OpenAI 크레딧이 있으면 GPT-4o 자유 대화, 없으면 키워드 오프라인 모드로 동작.
set -e
source /opt/ros/humble/setup.bash
source "$HOME/cobot_ws/install/setup.bash"
source "$HOME/cobot2_ws_1/install/setup.bash"
exec ros2 run pick_and_place_voice chat_commander "$@"
