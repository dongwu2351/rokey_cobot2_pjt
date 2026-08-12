# ROKEY 저지연 음성 명령 파이프라인

각 명령의 수신 구간에서 하나의 마이크 스트림을 사용해 다음 순서로 처리합니다.

```text
WAIT_WAKE -> WAIT_SPEECH/VAD -> STT -> FAST_RULE 또는 LLM -> SAFETY VALIDATION
```

- 웨이크워드: 언제 명령 수신을 시작할지 결정
- VAD: 고정 5초 녹음 대신 발화 시작과 종료를 검출
- STT: 완성된 메모리 WAV를 `gpt-transcribe`로 전사
- 빠른 경로: STT 이후 명확한 `STOP`, `FETCH`, `MOVE`를 추가 LLM 호출 없이 해석
- LLM 경로: 비정형·문맥형 명령만 Responses API Structured Outputs로 해석
- 안전 검증: 허용된 스킬·물체·목적지만 실행 가능 상태로 반환

STT와 LLM은 같은 OpenAI 클라이언트/HTTP 연결 풀을 재사용해 복잡한 명령의 추가
연결 설정 지연을 줄입니다.

현재 버전은 검증된 JSON 명령까지 생성하며 실제 로봇 모션은 호출하지 않습니다. 로봇
제어기는 최신 `robot_state`와 비전 문맥을 제공하고, 최상위
`PipelineResult.executable`이 `true`일 때만 별도 어댑터에서 연결하세요.
동작 명령은 최신 문맥, 비전 스냅샷 식별자, 유일하게 확정된 실제 물체 ID가 모두
없으면 Router와 최상위 실행 게이트에서 차단됩니다. `STOP`만 문맥 없이 통과할 수
있습니다.
연속 모드에서는 웨이크 대기 전 정적 딕셔너리를 재사용하지 말고
`run_forever(..., context_provider=...)` 콜백으로 해석 직전 상태를 공급하세요.

녹음이 끝나면 마이크 스트림을 STT 전에 닫습니다. 따라서 원격 STT·LLM과 결과
callback 동안 입력 큐가 쌓이지 않지만, 그 구간의 음성은 의도적으로 듣지 않는
half-duplex 방식입니다. callback은 로봇 작업 큐에 넣고 즉시 반환해야 다음 웨이크
수신이 빨리 재개됩니다.
`run_once()`에도 닫힌 마이크를 전달해야 하며, 이미 열린 스트림은 ownership이
불명확하므로 오디오·클라우드 처리 전에 `ERROR`로 거부됩니다. `run_forever()`는
진입 시 열린 스트림을 먼저 정리합니다.

## 설치

Ubuntu에서 PortAudio 개발 패키지와 격리된 Python 환경을 준비합니다.

```bash
sudo apt-get install portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r VoiceProcessing/requirements.txt
cp VoiceProcessing/.env.example VoiceProcessing/.env
```

`VoiceProcessing/.env`의 `OPENAI_API_KEY`를 설정합니다. 이 파일은 Git에서 제외됩니다.

openWakeWord는 사용자 정의 TFLite 모델 외에 음성 특징 추출용 보조 모델이 필요합니다.
설치 후 한 번만 다음 명령으로 받아 둡니다. 런타임 중에는 다운로드하지 않습니다.

```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
```

보조 Silero VAD 자산이 없으면 파이프라인은 적응형 에너지 VAD로 동작하며 경고를
출력합니다. 현장 정확도를 위해서는 위의 일회성 다운로드를 권장합니다.

## 실행

웨이크워드 `hello rokey`를 기다린 뒤 한 번 처리합니다.

```bash
python -m VoiceProcessing
```

웨이크워드 없이 VAD 녹음을 바로 시작하려면 다음과 같이 실행합니다.

```bash
python -m VoiceProcessing --no-wake
```

웨이크워드 기반으로 계속 처리하려면:

```bash
python -m VoiceProcessing --continuous
```

개인정보 보호를 위해 콘솔 출력에서 원문 전사는 기본적으로 가려집니다. 개발 중
확인이 필요할 때만 `--show-transcript`를 추가하세요.
현재 구현은 오디오를 OpenAI STT로 보내며, LLM 경로에서는 전사문과 구조화된 비전
문맥도 전송합니다. `attributes`에 개인정보나 비밀 값을 넣지 마세요. 완전한 로컬
처리가 필요하면 `STT.transcribe()`와 `LLMCommandParser` 인터페이스를 로컬 모델
어댑터로 교체해야 합니다.
기본 CLI에는 실제 로봇/비전 `context_provider`가 연결되어 있지 않으므로 모션 명령은
안전하게 `CONTEXT_REQUIRED`로 차단됩니다. 아래 비전 문맥 연결 예시를 적용해야
`executable: true`가 될 수 있습니다.

모호한 명령에 대한 `CLARIFY` 질문을 AI 음성으로 재생하려면:

```bash
python -m VoiceProcessing --no-wake --show-transcript --speak-clarifications
```

사용자에게 들리는 음성이 AI 생성 음성임을 명확히 고지해야 합니다. TTS만 별도로
확인할 때는 다음 명령을 사용합니다.

```bash
python -m VoiceProcessing.TTS \
  "어느 나사를 말씀하시는 건가요?" \
  --disclosure \
  --play
```

마이크 장치를 확인하는 수동 도구:

```bash
python VoiceProcessing/mic_test.py --list-devices
python VoiceProcessing/mic_test.py --device-index 0
```

기본 입력이 16 kHz를 지원하지 않으면 `MIC_SAMPLE_RATE=48000` 또는
`--sample-rate 48000`을 사용합니다. 이 경우 엔드포인트 VAD는 에너지 기반 대체
모델을 사용하고, 웨이크워드 입력만 내부에서 16 kHz로 변환합니다.

## 주요 설정

| 설정 | 기본값 | 의미 |
|---|---:|---|
| 오디오 프레임 | 30 ms | VAD 판정 단위 |
| 웨이크워드 입력 | 80 ms | openWakeWord 추론 단위 |
| 발화 시작 제한 | 2,000 ms | 웨이크워드 뒤 말이 없을 때 종료 |
| 종료 침묵 | 600 ms | 연속 침묵을 발화 끝으로 판정 |
| 최대 발화 | 10,000 ms | 초과 시 `TRUNCATED`, STT·명령 실행 차단 |
| STT 모델 | `gpt-transcribe` | `.env`의 `STT_MODEL`로 변경 |
| 명령 모델 | `gpt-5.6-terra` | 지연·정확도 균형; `COMMAND_MODEL`로 변경 |
| 추론 강도 | `none` | 단순 구조화 지연 최소화; 평가 후 조정 |
| 비전 문맥 TTL | 3,000 ms | 초과 시 `STALE_CONTEXT`; `COMMAND_MAX_SNAPSHOT_AGE_MS`로 변경 |
| 미래 시각 허용 | 250 ms | 카메라/호스트 시계 오차 상한 |

명령 중간의 짧은 쉼에서 너무 일찍 끊기면 `--end-silence-ms 750`처럼 늘립니다.
주변 소음으로 발화가 끝나지 않으면 Silero VAD를 준비하고 `--vad-threshold`를
0.5에서 0.05 단위로 조정하세요.

음성 전사 지연을 줄이려면 `faster-whisper` 로컬 백엔드를 사용할 수 있습니다.
첫 실행에서 선택한 모델을 한 번 내려받으며, CPU 환경에서는 `base/int8`부터
시작하는 것이 적절합니다.

```bash
export STT_BACKEND=local
export LOCAL_STT_MODEL=base
export LOCAL_STT_DEVICE=cpu
export LOCAL_STT_COMPUTE_TYPE=int8
```

로컬 모델 오류 시 클라우드 STT로 대체하려면 `STT_BACKEND=hybrid`를 사용합니다.

## LLM 모델과 프롬프트

현재 기본 설정은 다음과 같습니다.

```text
STT:     gpt-transcribe
명령 LLM: gpt-5.6-terra
TTS:     gpt-4o-mini-tts
```

`assistive_cli --realtime`은 `SituatedCommandParser`를 사용하고,
`VoiceProcessing` 기본 음성 명령 경로는 `CommandRouter`를 사용합니다. 두 경로 모두
`COMMAND_MODEL` 값을 공유하지만 시스템 프롬프트는 각각의 역할에 맞게 별도로 정의되어
있습니다. 비정형 공구·문맥·공간 선택은 `SituatedCommandParser`의
`SITUATED_INSTRUCTIONS`가 담당합니다.

프롬프트는 코드에 직접 넣거나 외부 파일로 지정할 수 있습니다.

```bash
export COMMAND_MODEL=gpt-5.6-terra
export COMMAND_INSTRUCTIONS_FILE=/home/taehwan/corecode/VoiceProcessing/prompts/assistive_system.txt
```

또는 Python에서 일회성으로 주입할 수 있습니다.

```python
parser = SituatedCommandParser(
    model="gpt-5.6-terra",
    instructions="""
당신은 조립 보조 로봇의 음성 명령 해석기다.
항상 구조화된 PerceptionPlan만 반환하고, 모호하면 한국어로 되물어라.
""",
)
```

프롬프트를 바꾸더라도 좌표·물체 ID·실행 가능 여부를 LLM이 생성하도록 허용하면 안
됩니다. LLM은 의도와 시각 질의만 제안하고, 실제 BB·ID·안전 검증은 로컬 코드가
담당해야 합니다.

## 명령 출력

최신 비전 문맥과 함께 `해머 가져와`를 처리한 전체 출력의 핵심 예시는 다음과
같습니다. 기본 콘솔 출력처럼 원문은 가린 상태입니다.

```json
{
  "state": "COMMAND_READY",
  "executable": true,
  "safety_context_fresh": true,
  "context_valid_until_ms": 1785823203000,
  "transcript": null,
  "recording": {
    "duration_ms": 960,
    "speech_ms": 420,
    "stop_reason": "end_silence",
    "speech_detected": true
  },
  "command": {
    "schema_version": "1.0",
    "decision": "READY",
    "actions": [
      {
        "intent": "FETCH",
        "object": "hammer",
        "destination": null,
        "object_query": "해머",
        "resolved_object_id": "hammer_01"
      }
    ],
    "ambiguity": "NONE",
    "clarification_question": null,
    "route": "FAST_RULE",
    "raw_utterance": "[REDACTED]",
    "grounding_query": null,
    "snapshot_revision": "camera-1842",
    "snapshot_timestamp_ms": 1785823200000,
    "latency_ms": 0.021,
    "error_code": null
  },
  "timings": {
    "wake_wait_ms": 812.4,
    "recording_ms": 960.2,
    "stt_ms": 1740.0,
    "command_ms": 0.021,
    "total_ms": 3512.621
  },
  "error": null
}
```

다음 조건에서는 모션을 실행하지 않습니다.

- 부정 명령 또는 지원하지 않는 스킬
- 대상·목적지가 없거나 여러 후보가 존재
- LLM 시간 초과, API 오류 또는 스키마 오류
- 허용 목록 밖의 물체·목적지
- `STOP`과 다른 동작이 한 결과에 혼합됨
- 한 명령의 여러 액션이 같은 `resolved_object_id`를 중복 사용함
- 비전 스냅샷 revision·timestamp 또는 확정된 실제 물체 ID가 없음
- 스냅샷이 TTL보다 오래됐거나 허용 범위보다 미래 시각임

`STOP`의 의도 해석은 STT 이후 로컬에서 최우선 처리되지만, 음성 수집·원격 STT를
거치며 half-duplex 구간에는 새 음성을 듣지 않습니다. 따라서 음성 STOP은 편의 기능일
뿐 안전정지 수단이 아닙니다. 일반 운전 정지와 산업용 비상정지는 이 경로와 별개의
독립 안전 입력 및 하드웨어 E-stop을 반드시 유지해야 합니다.

## 비전 문맥 연결

현재 보이는 물체를 구조화해 전달하면 동일 이름의 물체가 0개 또는 여러 개인 경우
자동으로 `CLARIFY`가 됩니다.

```python
import time

from VoiceProcessing.command_router import CommandRouter

context = {
    "visible_objects": [
        {
            "id": "hammer_01",
            "canonical_name": "hammer",
            "snapshot_revision": "camera-1842",
            "location": "table",
        }
    ],
    # recent_object는 실제 ID가 아니라 canonical name입니다.
    "recent_object": "hammer",
    "robot_state": "ready",
    "snapshot_revision": "camera-1842",
    "snapshot_timestamp_ms": int(time.time() * 1000),
}
result = CommandRouter().parse_command("해머 가져와", context)
```

`snapshot_timestamp_ms`는 파이프라인 호스트와 같은 Unix epoch 밀리초 기준이어야
합니다. 기본 TTL은 3초입니다. 복잡한 LLM 추론이 이 시간을 넘기면 모션은
`STALE_CONTEXT`로 차단되며, TTL을 늘리기 전에 실제 작업 공간에서 안전성을
평가하세요. 같은 프로세스의 `PipelineResult.executable`은 Linux에서 suspend를
포함하는 `CLOCK_BOOTTIME`과 wall clock을 함께 확인하며, 한 번 만료되면 시스템
시각이 뒤로 조정돼도 다시 살아나지 않습니다.

연속 파이프라인을 로봇 어댑터에 연결할 때는 문맥을 콜백으로 즉시 조회합니다.

```python
def current_context():
    # 한 번의 원자적 조회로 불변 프레임 전체를 가져옵니다.
    snapshot = latest_camera_snapshot()
    return {
        "robot_state": robot_state(),
        # 각 VisibleObject.snapshot_revision도 snapshot.revision과 같아야 합니다.
        "visible_objects": snapshot.visible_objects,
        "snapshot_revision": snapshot.revision,
        "snapshot_timestamp_ms": snapshot.timestamp_ms,
    }

def handle(result):
    if not result.executable:
        return  # 되묻기/오류 처리는 여기서 분리
    # callback을 오래 막지 말고, 검증된 ID와 revision을 함께 작업 큐에 넣습니다.
    robot_adapter.enqueue(
        actions=result.command.actions,
        snapshot_revision=result.command.snapshot_revision,
        snapshot_timestamp_ms=result.command.snapshot_timestamp_ms,
        context_valid_until_ms=result.context_valid_until_ms,
    )

pipeline.run_forever(handle, context_provider=current_context)
```

`visible_objects`, revision, timestamp를 각각 따로 조회하면 카메라 프레임 전환 시 서로
다른 시점의 값이 섞일 수 있습니다. `latest_camera_snapshot()`은 이 세 값을 하나의
불변 객체로 원자적으로 반환해야 하며, Router는 각 물체의 `snapshot_revision`이
최상위 revision과 다른 문맥을 거부합니다.

작업 큐의 worker도 모션 직전에 `context_valid_until_ms`와 현재 Unix epoch 밀리초를
비교하고, snapshot revision 및 `robot_state`를 다시 검사한 뒤에만 실행해야 합니다.
enqueue 시점의 `result.executable` 값만 저장해 나중에 신뢰하면 안 됩니다.

어댑터는 `action.object`로 물체를 다시 검색하면 안 됩니다. 반드시
`resolved_object_id`를 사용하고, 실행 직전에 같은 snapshot revision인지 원자적으로
확인하거나 해당 ID를 안정적으로 추적해야 합니다. 직렬화된 결과를 큐에 오래 보관할
수 있으므로 `context_valid_until_ms`도 실행 시각과 다시 비교하고, `robot_state`도
실행 직전에 다시 검사하세요.

오픈 보카블러리 묘사는 실행 `actions`와 분리된 최상위 `grounding_query`에 유지되고
`VISION_GROUNDING_REQUIRED`로 반환됩니다. Grounding DINO/OWL-ViT 같은 비전
그라운더가 실제 물체 ID를 확정한 후 같은 안전 검증기를 다시 통과시키는 방식으로
확장할 수 있습니다.

## 비정형 작업 보조와 제로샷 웹캠 grounding

`assistive_cli`는 고정 공구 목록에 없는 공구·조립품을 자유 묘사로 해석하고,
Grounding DINO로 현재 웹캠 프레임의 BB를 찾는 별도 perception 계층입니다.
이 계층의 결과는 물체 제안일 뿐이며 `executable`은 항상 `false`입니다. 깊이,
캘리브레이션, 로봇 상태와 최신 스냅샷 검증을 통과하기 전에는 모션에 연결하지 마세요.

비전 의존성은 음성 환경과 분리되어 있습니다. NVIDIA 드라이버와 CUDA가 정상인지 먼저
확인한 뒤 해당 PyTorch CUDA 빌드를 준비하고 나머지 패키지를 설치하세요. CPU에서도
실행할 수 있지만 Grounding DINO 응답이 느릴 수 있습니다.

```bash
python -m pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r VoiceProcessing/requirements-vision.txt
```

텍스트로 먼저 BB까지 시험합니다. 최초 실행은 모델을 내려받으므로 시간이 걸립니다.

```bash
python -m VoiceProcessing.assistive_cli \
  --text "파란 L자 조립 지그 가져와" \
  --camera-index 0 \
  --show \
  --save-frame /tmp/assistive_result.jpg
```

마이크와 되묻기 TTS를 함께 사용하려면:

```bash
python -m VoiceProcessing.assistive_cli \
  --no-wake \
  --continuous \
  --realtime \
  --camera-index 0 \
  --show \
  --speak \
  --disclose-ai-voice
```

반응속도를 측정하거나 실제로 반복 사용할 때는 `--continuous`를 유지하세요. 이 모드는
Grounding DINO, 카메라, OpenAI 클라이언트와 물체 메모리를 한 프로세스에서 재사용합니다.
프로그램을 명령마다 다시 실행하면 약 8초의 모델 로드 및 첫 CUDA 워밍업 비용이 매번
발생합니다. 종료는 `Ctrl+C`, 영상 창에서는 `q`입니다.

YOLOE-26s-seg를 기본 고속 탐지기로 시험하려면 다음을 사용합니다.

```bash
python -m VoiceProcessing.assistive_cli \
  --grounding-backend yoloe \
  --yoloe-model yoloe-26s-seg.pt \
  --no-wake --continuous --realtime --camera-index 0 --show
```

YOLOE가 찾지 못한 경우 Grounding DINO로 재탐색하려면 `--grounding-backend cascade`를
사용합니다. `yoloe-26s-seg.pt`와 텍스트 프롬프트 인코더는 첫 실행 시 자동으로
준비됩니다.

### 웹캠 시각 메모리(실험 기능)

성공적으로 Grounding된 물체의 현재 프레임 crop은 기본적으로
`VoiceProcessing/data/visual_memory.sqlite3`와
`VoiceProcessing/data/visual_samples/`에 저장됩니다. 이 메모리는 오래된 좌표를
저장하지 않고 RGB 외관 특징만 저장합니다. 다음에 같은 이름의 후보가 여러 개이면
현재 프레임의 후보 crop과 비교해 가장 비슷한 후보를 우선 선택합니다. 따라서 현재
단계에서는 Grounding DINO를 완전히 생략하지 않고, 오인식·되묻기를 줄이는 실험용
단계입니다. 로봇 실행 전에는 항상 최신 프레임으로 다시 검증해야 합니다.
사용 여부는 결과 JSON의 `visual_memory_hit`와 `visual_similarity` 필드로 확인할 수
있습니다. 첫 등록은 `false`, 같은 물체의 후속 호출에서 임계값을 넘으면 `true`가
됩니다.

`--realtime`은 카메라를 별도 프로세스의 1프레임 공유메모리 버퍼로 실행하고, UI를
30FPS로 유지합니다. 음성/VAD·STT·LLM·Grounding DINO·TTS는 worker에서 처리되므로
이 단계들이 실행되는 동안에도 영상 창과 BB 클릭이 멈추지 않습니다. DINO 결과 사이의
BB는 Lucas–Kanade optical flow로 갱신합니다. 현재 내장 웹캠의 저조도 자동 노출에서는
카메라 입력 약 18~20FPS, UI 약 30FPS가 정상입니다.
기준 물체 선택은 기본적으로 음성으로만 진행합니다. 후보가 여러 개면 시스템이
왼쪽/오른쪽/위/아래 중 하나를 묻고, 사용자가 말한 위치를 장면 BB에 매핑합니다.
`--debug-click-focus`를 지정한 경우에만 개발용 클릭 선택 대기 모드를 사용합니다.

처리 흐름은 다음과 같습니다.

```text
비정형 음성 -> 구조화된 관계 질의 -> 기억 조회 -> 최신 프레임 zero-shot grounding
            -> 유일 후보: BB/ID 반환 및 개념 기억
            -> 0개/여러 개/기준 불명: 한국어 질문 생성 및 TTS
```

되묻기가 발생하면 원래 perception plan을 pending 상태로 유지합니다. 다음 발화가
`"왼쪽 것"`, `"파란 손잡이 달린 것"`처럼 짧아도 직전 FETCH/MOVE 요청과 결합해 다시
grounding하며, 성공하거나 STOP이 들어오면 pending 상태를 지웁니다.

확인된 개념은 기본적으로 `VoiceProcessing/data/object_memory.sqlite3`에 영어 비전 이름,
사용자가 실제로 부른 한국어 표현을 포함한 별칭,
grounding prompt와 비공간 속성만 기록합니다. 위치나 로봇 좌표는 저장하지 않습니다.
따라서 사용자가 같은 이름을 다시 말하면 단순 명령은 LLM을 거치지 않는 메모리 빠른
경로를 사용할 수 있지만, BB는 물체 이동에 대비해 최신 프레임에서 다시 검출합니다.

`"이 나사 말고 조금 더 긴 나사"`는 기준 물체, 제외 조건, 길이 비교와
`NEAREST_GREATER` 선택 정책으로 구조화됩니다. `--continuous --show` 모드에서는 기준이
없을 때 후보 BB와 함께 TTS로 가리켜 달라고 요청하며, 사용자가 영상 창의 기준 BB를
클릭하면 해당 ID를 focus로 유지합니다. 이후 `"그것보다 조금 더 긴 것"` 같은 짧은
후속 발화를 원래 계획과 결합합니다. 외부 포인팅 추적기를 사용할 때는 동일하게
`AssistiveCommandProcessor.set_focus(object_id)`를 호출하면 됩니다. `length_mm` 같은
보정된 측정값이 없으면 BB 픽셀 크기로 화면상 후보만 제안하고
`geometry_verified: false`를 유지합니다.

이전 `(objects, destinations)` 튜플 API인 `ExtractKeyword.extract_keyword()`는 객체
ID와 스냅샷 정보를 표현할 수 없어 항상 경고와 함께 `None`을 반환합니다. 기존 코드는
`parse_command()` 또는 `PipelineResult` 계약으로 이전해야 합니다.

## 테스트

기본 테스트는 마이크, 모델 다운로드, OpenAI API를 사용하지 않습니다.

```bash
python -m unittest discover -s VoiceProcessing/tests -v
```

실제 장치에서는 별도로 웨이크워드 오탐/미탐, VAD 종료 지연, STT 정확도,
명령 정확도와 각 단계의 p50/p95 지연을 녹음 샘플로 측정하세요.
특히 STT `keywords`는 전문용어 정확도를 높일 수도 있지만 말하지 않은 단어를
유도할 수도 있으므로, 키워드 사용 전후를 같은 한국어 WAV 세트로 비교해야 합니다.
USB/ALSA 장치가 반복 open/close에 안정적인지, 웨이크 직후 쉬지 않고 말해도 첫
음절이 보존되는지, 추론 중 말한 소리가 다음 세션에서 오래된 웨이크로 재생되지 않는지도
현장에서 확인하세요.
### 멀티카메라 실시간 모드

작업대에 카메라를 여러 대 설치한 경우 `--camera-indices`에 장치 번호를
콤마로 지정합니다. 각 카메라는 별도 캡처 프로세스로 유지되므로 UI 프레임은
추론 지연과 분리되고, 명령이 들어오면 모든 프레임을 탐지해 가장 높은 점수의
후보를 선택합니다.

```bash
python -m VoiceProcessing.assistive_cli \
  --camera-indices 4,6 \
  --camera-width 640 --camera-height 480 \
  --grounding-backend yoloe \
  --yoloe-model yoloe-26s-seg.pt \
  --yoloe-device 0 --yoloe-image-size 384 \
  --no-wake --continuous --realtime --show
```

`--camera-indices`는 `--realtime` 모드에서 사용하며, 두 카메라의 추론 시간은
순차적으로 합산됩니다. 화면 표시 FPS와 명령 응답 시간은 별도로 측정하세요.
