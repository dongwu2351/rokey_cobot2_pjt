#!/bin/bash
# JARVIS 코파일럿 (로컬 웹 UI) — 로봇 스킬 연결본
#
#   bash ~/cobot2_ws_1/scripts/run_jarvis.sh
#
# 전제: 로봇 앱이 이미 떠 있어야 실제로 움직입니다.
#   bash ~/cobot2_ws_1/scripts/run_webcam_pnp.sh --live --free-wrist --hand-place
#
# 사용자 UI:  http://127.0.0.1:8765
# 진단  UI:  http://127.0.0.1:8765/debug
# 대기 상태에서는 'jarvis'를 먼저 입력해야 요청이 전달됩니다.
#
# 명령 예: "jarvis 해머 가져와줘" -> "응" | "해머 정리해줘" | "홈으로 돌아가"
#          "그리퍼 열어줘" | 동작 중 "멈춰"(즉시 정지)
#
# --robot ros  : 실제 로봇 앱과 연결 (기본)
# --robot mock : 로봇 없이 UI/대화만 시험
#
# 음성으로 쓰려면:
#   bash run_jarvis.sh --voice
# 마이크는 자동 탐지한다. 특정 장치를 쓰려면:
#   JARVIS_MIC=13 bash run_jarvis.sh --voice
# (앱 기본값 --device-index 12는 이 PC에서 HDMI 출력이라 소리가 안 들어온다)
set -e
source /opt/ros/humble/setup.bash
source "$HOME/cobot_ws/install/setup.bash" 2>/dev/null || true
source "$HOME/cobot2_ws_1/install/setup.bash"

PROJECT="$HOME/cobot2_ws_1/DUME_COPILOT_ROBOT_SKILL_HANDOFF_20260809/dume_team_handoff/integrated_dum_e_project"
MANUAL="$HOME/cobot2_ws_1/DUME_COPILOT_ROBOT_SKILL_HANDOFF_20260809/dume_team_handoff/dum_E_project/assembly_manuals/mini_conveyor_module/assembly.yaml"

# 포트를 이미 쓰는 코파일럿이 있으면 명령을 가로채므로 먼저 정리한다.
# run_integrated.py(부모)와 unified_copilot.app(자식)을 모두 내리고,
# 포트가 실제로 해제될 때까지 기다린다 - 바로 띄우면 bind 실패로 UI가
# 죽은 채 프로세스만 살아 있는 상태가 된다.
PORT="${JARVIS_PORT:-8765}"
if pgrep -f 'run_integrated.py copilot' > /dev/null 2>&1; then
    echo "JARVIS> 이미 실행 중인 코파일럿을 종료합니다 (다른 터미널의 창이 닫힐 수 있음)"
fi
pkill -INT -f 'run_integrated.py copilot' 2>/dev/null || true
pkill -INT -f 'unified_copilot.app' 2>/dev/null || true
for _ in $(seq 1 20); do
    ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN || break
    sleep 0.5
done
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    pkill -KILL -f 'unified_copilot.app' 2>/dev/null || true
    sleep 1
fi

cd "$PROJECT"

# 음성을 켠 경우에만 마이크를 정한다. 사용자가 --device-index를 직접 준
# 경우에는 건드리지 않는다.
VOICE_ARGS=(--no-voice)
case " $* " in *" --voice "*) VOICE_ARGS=() ;; esac

# 이 PC의 시스템 기본 입력이 '웹캠 마이크'(로봇 셀을 보는 C270, 1.3m 거리)로
# 고정돼 있고 pactl로 바꿔도 GNOME이 되돌린다. 웹캠 마이크는 작업장 소음을
# 끊임없이 담아 VAD가 문장 끝을 못 찾는다. 코파일럿 프로세스에만
# PULSE_SOURCE로 웹캠이 아닌 입력(노트북 내장 마이크)을 강제한다.
if [ ${#VOICE_ARGS[@]} -eq 0 ] && [ -z "${PULSE_SOURCE:-}" ]; then
    # 웹캠 마이크 제외 + '활성 포트가 실제로 존재하는' 입력만 후보로.
    # (이 노트북의 첫 non-webcam 소스는 아무것도 안 꽂힌 헤드셋 잭이라
    #  볼륨을 아무리 올려도 무음이다 - 포트가 not available 이면 걸러야 한다)
    SRC=$(pactl list sources 2>/dev/null | python3 -c "
import sys, re
blocks = sys.stdin.read().split('Source #')
best = None
for b in blocks[1:]:
    name = re.search(r'Name: (\S+)', b)
    if not name or 'alsa_input' not in name.group(1):
        continue
    if re.search(r'WEBCAM|webcam', name.group(1)):
        continue
    active = re.search(r'Active Port: (.+)', b)
    port_dead = False
    if active:
        port_line = re.search(re.escape(active.group(1).strip().split(' ')[0]) + r'.*not available', b)
        port_dead = bool(re.search(r'Active Port.*', b)) and bool(port_line)
    # 포트 목록에서 active port가 not available로 표시되면 제외
    if active:
        pname = active.group(1).strip()
        for line in b.splitlines():
            if pname.split(':')[0] in line and 'not available' in line:
                port_dead = True
    if not port_dead:
        best = name.group(1)
        break
print(best or '')
")
    if [ -n "$SRC" ]; then
        # 이 소스는 시스템 설정 화면에 안 보여서 볼륨이 15% 같은 값으로
        # 방치되기 쉽다(-48dB면 사실상 무음). 항상 쓸 수 있게 맞춰 둔다.
        pactl set-source-mute "$SRC" 0 2>/dev/null || true
        pactl set-source-volume "$SRC" 100% 2>/dev/null || true

        # 에코 캔슬레이션: 노트북 스피커로 나간 JARVIS 목소리를 내장 마이크가
        # 되들으면 자기 말에 자기가 대답하는 루프가 된다. AEC 소스로 듣고
        # AEC 싱크로 말해야 상쇄가 걸린다 (barge-in도 유지됨).
        DEFAULT_SINK=$(pactl get-default-sink 2>/dev/null)
        pactl unload-module module-echo-cancel 2>/dev/null || true
        if pactl load-module module-echo-cancel aec_method=webrtc \
              source_master="$SRC" sink_master="$DEFAULT_SINK" \
              source_name=jarvis_ec_src sink_name=jarvis_ec_sink \
              >/dev/null 2>&1; then
            export PULSE_SOURCE=jarvis_ec_src
            export PULSE_SINK=jarvis_ec_sink
            pactl set-source-volume jarvis_ec_src 100% 2>/dev/null || true
            pactl set-sink-volume jarvis_ec_sink 100% 2>/dev/null || true
            echo "JARVIS> 마이크: $SRC + 에코캔슬 (jarvis_ec_src/sink)"
        else
            export PULSE_SOURCE="$SRC"
            echo "JARVIS> 마이크 소스 강제: $SRC (AEC 모듈 없음 - 에코 주의)"
        fi
    fi
fi

# 사용자가 --device-index를 직접 준 경우: 입력 채널이 있는 장치인지 먼저 본다.
# 출력 전용 장치(예: HDMI)를 주면 PyAudio가 "-9998 Invalid number of
# channels"로 죽는데, 그 메시지만 보고 원인을 알기는 어렵다.
if [ ${#VOICE_ARGS[@]} -eq 0 ] && [[ " $* " == *" --device-index "* ]]; then
    REQ_MIC=""
    prev=""
    for a in "$@"; do
        [ "$prev" = "--device-index" ] && REQ_MIC="$a"
        prev="$a"
    done
    python3 - "$REQ_MIC" <<'PY' || exit 1
import sys
idx = sys.argv[1]
try:
    import pyaudio
except Exception:
    sys.exit(0)                      # 확인 불가면 통과시킨다
p = pyaudio.PyAudio()
try:
    info = p.get_device_info_by_index(int(idx))
except Exception:
    print(f"JARVIS> 마이크 장치 {idx}번이 존재하지 않습니다.", file=sys.stderr)
    p.terminate(); sys.exit(1)
if info.get("maxInputChannels", 0) < 1:
    print(f"JARVIS> 장치 {idx}번 '{info['name']}'은(는) 입력 채널이 없습니다 "
          f"(출력 전용). 음성이 동작하지 않습니다.", file=sys.stderr)
    ins = [f"{i}:{p.get_device_info_by_index(i)['name']}"
           for i in range(p.get_device_count())
           if p.get_device_info_by_index(i).get("maxInputChannels", 0) > 0]
    print("JARVIS> 사용 가능한 입력 장치 -> " + ", ".join(ins), file=sys.stderr)
    print("JARVIS> --device-index 를 빼면 자동으로 선택합니다.", file=sys.stderr)
    p.terminate(); sys.exit(1)
p.terminate()
PY
fi

if [ ${#VOICE_ARGS[@]} -eq 0 ] && [[ " $* " != *" --device-index "* ]]; then
    MIC="${JARVIS_MIC:-$(python3 - <<'PY'
# 실제로 입력 채널이 있고 16kHz로 열리는 장치를 고른다. pulse/default 우선.
try:
    import pyaudio
    p = pyaudio.PyAudio()
    best = None
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("maxInputChannels", 0) < 1:
            continue
        name = d["name"].lower()
        rank = 0 if name.startswith("pulse") else 1 if name.startswith("default") else 2
        if best is None or rank < best[0]:
            best = (rank, i)
    p.terminate()
    print(best[1] if best else "")
except Exception:
    print("")
PY
)}"
    if [ -n "$MIC" ]; then
        echo "JARVIS> 마이크 장치 자동 선택: index $MIC"
        set -- "$@" --device-index "$MIC"
    else
        echo "JARVIS> 입력 장치를 찾지 못했습니다. JARVIS_MIC=<번호>로 지정하세요." >&2
    fi
fi

exec python3 run_integrated.py copilot \
  "${VOICE_ARGS[@]}" --camera "${JARVIS_CAMERA:-ros}" --robot ros --no-open-ui \
  --manual "$MANUAL" "$@"
