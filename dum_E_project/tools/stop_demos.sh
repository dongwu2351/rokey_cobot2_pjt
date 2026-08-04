#!/usr/bin/env bash
# 데모 관련 프로세스 정리.
#
# rviz2 만 죽이면 데모 노드가 살아남아 /joint_states 를 계속 발행하고,
# 다음 런치와 겹쳐서 로봇이 두 자세 사이에서 깜빡인다.
# 런치 파일 자체도 죽여야 자식이 되살아나지 않는다.
#
# ★ 실기기 tracking test 를 죽이지 않는 것이 이 스크립트의 최우선 조건이다.
#   rviz2 / robot_state_publisher / joint_state_publisher 는 실기기 쪽에서도
#   쓰는 이름이라 이름만 보고 죽이면 실기기 세션까지 같이 꺼진다.
#   -> /proc/<pid>/environ 에서 ROS_DOMAIN_ID=84 인 것만 죽인다.
#      (환경변수는 프로세스 시작 시점에 고정되므로 신뢰할 수 있다)
#   데모 전용 스크립트는 이름이 유일하므로 도메인 확인 없이 죽여도 안전하다.
SELF=$$
DOMAIN=84

in_domain() {   # $1=pid — 도메인 84 로 뜬 프로세스인가?
    local envfile="/proc/$1/environ"
    [ -r "$envfile" ] || return 1
    tr '\0' '\n' < "$envfile" 2>/dev/null | grep -qx "ROS_DOMAIN_ID=$DOMAIN"
}

# 이 프로젝트에만 있는 이름 — 도메인 확인 없이 죽여도 된다
OURS="curobo_hybrid_demo.py curobo_rviz_demo.py pose_picker_node.py path_probe_node.py
      sphere_viz_node.py gripper_viz_node.py
      curobo_hybrid.launch curobo_demo.launch sphere_check.launch
      path_probe.launch pose_picker.launch"
# 실기기와 이름을 공유하는 것 — 도메인 84 인 것만 죽인다
SHARED="rviz2 robot_state_publisher joint_state_publisher"

kill_matching() {   # $1=패턴목록  $2=check|nocheck  $3=시그널
    for pat in $1; do
        for pid in $(pgrep -f -- "$pat" 2>/dev/null); do
            [ "$pid" = "$SELF" ] && continue
            if [ "$2" = "check" ]; then
                in_domain "$pid" || continue
            fi
            kill "-$3" "$pid" 2>/dev/null
        done
    done
}

kill_matching "$OURS"   nocheck TERM
kill_matching "$SHARED" check   TERM
sleep 1
kill_matching "$OURS"   nocheck KILL
kill_matching "$SHARED" check   KILL

echo "정리 완료 (도메인 $DOMAIN 만 — 실기기 세션은 건드리지 않음)"
