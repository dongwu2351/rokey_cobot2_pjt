# VoiceProcessing → Robot Control 인수인계 문서

## 1. 목적

이 프로젝트는 사용자의 한국어 음성 발화를 다음 단계로 처리하는 협동로봇 어시스턴트 파이프라인이다.

```text
마이크 음성
  ↓
웨이크워드/VAD
  ↓
Whisper STT
  ↓
대화 라우터·대화 기억
  ↓
명령이면 물체 grounding
  ↓
YOLOE bbox/segmentation
  ↓
카메라 좌표·깊이·캘리브레이션(다음 단계)
  ↓
로봇 제어 어댑터(별도)
```

현재 코드는 인식·대화·물체 검출까지 담당하며, 실제 두산 M0609 모션을 직접 호출하지 않는다. 로봇 제어 AI는 `executable=true`와 최신 좌표·안전 검증을 통과한 결과만 받아야 한다.

## 2. 구성 요소와 역할

### 음성 입력

- `MicController.py`: PyAudio 마이크 스트림
- `vad.py`: 음성 시작·종료 검출, 고정 녹음 대신 endpoint recording
- `wakeup_word.py`: `hello_rokey_8332_32.tflite` 웨이크워드
- `voice_pipeline.py`: 웨이크워드 → VAD → STT의 기존 음성 파이프라인
- `streaming_preview.py`: 발화 중 부분 STT 미리보기
- `speculative_intent.py`: 동일한 부분 전사가 반복될 때 `INTENT_STABLE` 이벤트 생성

### STT

- `STT.py`
- `STT_BACKEND=cloud`: OpenAI 전사
- `STT_BACKEND=local`: faster-whisper
- 한국어 언어 설정은 `ko`
- 최종 STT 결과가 로봇 명령의 기준이며, `PARTIAL_TRANSCRIPT`는 임시 추정이다.

### 대화·기억

- `conversation_router.py`: 자유 대화와 로봇 명령을 분기
- `conversation_memory.py`: 텍스트 대화 영구 저장
- `conversation_memory.sqlite3`: 최근 대화 복원용 DB
- `situated_parser.py`: 로봇 명령을 `PerceptionPlan`으로 구조화
- `object_memory.py`: 물체 canonical name, 별칭, grounding prompt 저장
- `visual_memory.py`: 물체 crop/시각 특징 저장. 좌표는 저장하지 않는다.

주요 라우트:

```text
ROBOT_COMMAND       물체 탐색·로봇 작업 후보
CHAT                일반 대화
TOOL_QUESTION       공구 사용법 질문
ADVICE              작업 조언
DESCRIBE            이미지 기반 물체 설명
CAPABILITY_PLAN     등록된 기능 조합 계획
CLARIFY             정보 부족으로 되묻기
STOP                즉시 정지 요청
```

### 비전

- `grounding.py`: Grounding DINO, YOLOE, cascade backend
- 기본 주력 모델: `yoloe-26s-seg.pt`
- YOLOE는 텍스트 prompt 기반 zero-shot 물체 검출과 segmentation을 담당한다.
- 결과는 `label`, `score`, `bbox`, segmentation attributes다.
- 일반 대화에는 이미지를 보내지 않는다.
- `DESCRIBE` 요청일 때만 현재 카메라 프레임을 비전 LLM에 전달한다.

### 실행 런타임

- `assistive_processor.py`: 문장 → 계획 → grounding → `AssistiveResult`
- `realtime_runtime.py`: 카메라 worker, 음성 worker, TTS, UI
- `assistive_cli.py`: 실행 CLI 및 camera index/backend 선택
- `TTS.py`: 일반 WAV 또는 PCM streaming TTS

## 3. 현재 출력 계약

정상적으로 물체를 찾은 경우:

```json
{
  "state": "GROUNDED",
  "plan": {
    "decision": "GROUND",
    "intent": "FETCH",
    "target_category": "hammer",
    "target_description": "hammer",
    "source_object_expression": "망치"
  },
  "detections": [
    {
      "id": "object_0001",
      "label": "hammer",
      "score": 0.82,
      "bbox": {"x1": 386.7, "y1": 123.9, "x2": 850.6, "y2": 717.8},
      "attributes": {}
    }
  ],
  "selected_object_id": "object_0001",
  "conversation_route": "ROBOT_COMMAND",
  "camera_index": 0,
  "geometry_verified": false,
  "spatial_estimate": null,
  "grasp_point": null,
  "executable": false,
  "utterance": "망치 가져와"
}
```

현재 `bbox`는 이미지 픽셀 좌표다. 아직 로봇 좌표가 아니므로 `geometry_verified=false`이면 로봇을 움직이면 안 된다.

대화 결과 예:

```json
{
  "state": "CONVERSATION",
  "conversation_route": "TOOL_QUESTION",
  "reply": "드라이버는 나사를 조이거나 풀 때 사용하는 공구입니다.",
  "robot_action_allowed": false,
  "executable": false
}
```

## 4. 로봇 제어 AI가 구현해야 하는 실행 게이트

로봇 제어 측은 다음 조건을 모두 만족할 때만 동작을 허용해야 한다.

```text
1. state == GROUNDED
2. plan.intent가 허용된 작업(FETCH/MOVE/PLACE)
3. selected_object_id가 존재
4. geometry_verified == true
5. spatial_estimate가 robot_base 좌표계임
6. snapshot이 최신임
7. 장애물·사람·그리퍼 상태 확인
8. 사용자가 요구한 작업과 실제 계획이 일치
```

`robot_action_allowed`나 `executable`이 false이면 perception 결과일 뿐이며 로봇 API를 호출하지 않는다. `STOP`은 별도 안전 경로로 즉시 처리한다.

## 5. 향후 3D 좌표 계약

캘리브레이션과 RealSense/stereo 계산이 연결되면 다음 구조를 사용한다.

```json
{
  "geometry_verified": true,
  "source_camera_index": 0,
  "spatial_estimate": {
    "position": {"x": 0.42, "y": -0.18, "z": 0.76},
    "frame": "robot_base",
    "method": "DEPTH",
    "confidence": 0.91
  },
  "grasp_point": {"x": 0.43, "y": -0.17, "z": 0.78}
}
```

단위는 meter다. bbox 중심을 그대로 파지점으로 사용하면 안 된다. 최종 파지는 D435i 깊이와 segmentation/mask, 물체 방향, 그리퍼 폭을 함께 사용해야 한다.

## 6. 카메라 전략

현재 목표 구성:

```text
C270/작업대 카메라
  - 어느 물체인지 결정
  - 작업대에서 대략적인 영역 결정
  - 빠른 YOLOE 검출

RealSense D435i/로봇 손목
  - 접근 후 깊이 재측정
  - 실제 3D 위치·자세·파지점 검증
  - 최종 grasp planning
```

두 RGB 카메라를 stereo로 사용할 경우 내부/외부 캘리브레이션, 프레임 동기화, `camera → robot_base` 변환이 필요하다. 단순히 높은 confidence의 카메라 한 대를 선택하는 것만으로는 3D 좌표가 생성되지 않는다.

## 7. 실행 예시

노트북 웹캠 단일 카메라:

```bash
cd /home/taehwan/corecode
source .venv/bin/activate

MPLCONFIGDIR=/tmp/mpl python -m VoiceProcessing.assistive_cli \
  --camera-index 0 \
  --camera-width 640 --camera-height 480 \
  --grounding-backend yoloe \
  --yoloe-model /home/taehwan/corecode/yoloe-26s-seg.pt \
  --yoloe-device 0 --yoloe-image-size 384 \
  --no-wake --continuous --realtime --show
```

멀티카메라(현재 캡처/후보 선택용):

```bash
--camera-indices 4,6
```

대화/TTS:

```bash
--speak --stream-tts
```

부분 전사와 안정 의도 이벤트:

```bash
--partial-preview
```

`PARTIAL_TRANSCRIPT`는 임시 추정이며 최종 STT가 권위 있는 결과다. `INTENT_STABLE`도 선행 준비 이벤트일 뿐, 최종 확정 전 로봇 실행을 허용하지 않는다.

## 8. 전달해야 할 파일

### 반드시 전달

```text
VoiceProcessing/assistive_models.py
VoiceProcessing/assistive_processor.py
VoiceProcessing/assistive_cli.py
VoiceProcessing/realtime_runtime.py
VoiceProcessing/grounding.py
VoiceProcessing/situated_parser.py
VoiceProcessing/conversation_router.py
VoiceProcessing/conversation_memory.py
VoiceProcessing/object_memory.py
VoiceProcessing/visual_memory.py
VoiceProcessing/voice_pipeline.py
VoiceProcessing/MicController.py
VoiceProcessing/vad.py
VoiceProcessing/STT.py
VoiceProcessing/TTS.py
VoiceProcessing/wakeup_word.py
VoiceProcessing/speculative_intent.py
VoiceProcessing/streaming_preview.py
VoiceProcessing/command_models.py
VoiceProcessing/command_router.py
VoiceProcessing/requirements.txt
VoiceProcessing/requirements-vision.txt
VoiceProcessing/.env.example
```

### 모델 파일 또는 다운로드 안내로 전달

```text
yoloe-26s-seg.pt
hello_rokey_8332_32.tflite
mobileclip2_b.ts (YOLOE text prompt backend가 요구하는 경우)
```

대용량 모델은 Git에 직접 넣기보다 다운로드 URL, SHA256, 저장 경로를 함께 전달하는 것이 좋다.

### 로봇 제어 팀에 별도로 전달할 것

```text
ROBOT_CONTROL_HANDOFF.md (이 문서)
카메라 캘리브레이션 YAML/JSON
카메라 index와 실제 카메라 이름 매핑
robot_base 좌표계 정의
M0609 TCP/그리퍼(RG2) 좌표계 정의
안전 정지·속도·접근 제한 규칙
```

## 9. 전달하지 말아야 할 파일

```text
VoiceProcessing/.env
OPENAI_API_KEY가 포함된 파일
object_memory.sqlite3 (필요 시 별도 선별 전달)
conversation_memory.sqlite3 (개인 대화가 포함될 수 있음)
visual_samples/의 개인 이미지
GPU/가상환경 캐시
```

`.env`는 `.env.example`만 전달하고 실제 API 키는 절대 포함하지 않는다.

## 10. 로봇 제어 팀의 첫 통합 작업

첫 단계에서는 모션을 호출하지 않고 다음 dry-run adapter만 구현한다.

```python
def handle_perception(result: dict) -> None:
    if result["state"] != "GROUNDED":
        return
    if not result["geometry_verified"]:
        print("geometry is not verified; do not move robot")
        return
    if not result["executable"]:
        return
    # 다음 단계에서 M0609/RG2 command adapter 연결
```

이후 RealSense 깊이와 캘리브레이션을 연결하고, 사람·장애물·그리퍼 상태를 검사한 뒤 실제 모션 어댑터를 붙인다.

## 11. 알려진 제한

- 부분 전사는 빠른 임시 결과이며 오인식할 수 있다.
- 클라우드 STT/LLM/TTS는 네트워크 지연이 있다.
- C270 bbox만으로는 3D 좌표를 만들 수 없다.
- 일반 대화는 카메라를 사용하지 않는다.
- `DESCRIBE`만 비전 LLM에 이미지가 전달된다.
- 현재 `CAPABILITY_PLAN`은 계획/응답 단계이며 실제 하드웨어 어댑터가 없다.
- 실제 로봇 실행은 캘리브레이션·깊이·안전 게이트 구현 후에만 허용한다.
