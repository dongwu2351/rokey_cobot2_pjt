# DUM-E 시스템 전체 구조 — 무엇 위에, 어떻게 흐르는가

작업자 옆에서 정밀 조립을 돕는 협동로봇 코파일럿. 말로 지시하고, 손가락으로
가리키고, 로봇이 도구를 건네주고 작업 상태를 봐준다.

이 문서는 **전체 지도**다. 개별 기능의 상세는 다음 문서에 있다.

- `docs/HANDOVER_AND_POINTING.md` — 손에서 받기 · 손가락 레이저 타겟팅
- `docs/GRASP_YAW.md` — 물체 방향에 맞춘 손목 회전
- `docs/PROJECT_TECHNICAL_REPORT.md` — 알고리즘 수식 · 시행착오 전체 기록

---

## 1. 하드웨어

| 구성 | 사양 | 역할 |
|---|---|---|
| 로봇 | Doosan **M0609** 6축 협동로봇 | 작업 수행 |
| 그리퍼 | OnRobot **RG2** (개폐 0~110 mm) | 파지 |
| 손목 카메라 | Intel RealSense **D435i** (RGB-D) | 정밀정렬 · 검사 촬영 |
| 환경 카메라 | 웹캠 **3대** (캘리브레이션 완료) | 3D 위치 추정 · 손 추적 |
| 호스트 | Ubuntu 22.04 | — |

**카메라를 두 층으로 나눈 것**이 이 시스템의 기본 구도다.
환경 카메라 3대는 **넓게 보고**(어디에 무엇이 있는지, 사람 팔이 어디로 움직이는지),
손목 카메라는 **가까이 본다**(정확히 어디를 어떤 각도로 물지).
넓은 시야로 접근하고, 도착해서 정밀도를 확보한다.

---

## 2. 소프트웨어 베이스

| 계층 | 사용 기술 |
|---|---|
| 미들웨어 | **ROS 2 Humble** (rclpy, tf2) |
| 로봇 드라이버 | **DSR_ROBOT2 / DRFL** (`movel`, `movej`, `SpeedlStream`, `ikin`) |
| 비전 | **OpenCV 4.11**, **Ultralytics YOLO 8.4**, **MediaPipe 0.10** (Hands) |
| 수치 | **NumPy 1.26**, **SciPy 1.15** (회전 변환, 최소제곱) |
| 웹 UI | **FastAPI 0.141** + uvicorn (홀로그램 UI, 진단 UI) |
| LLM | **OpenAI 1.98** — 대화 `gpt-5.6-terra`, 사진 판정 `gpt-5.6-sol`, 음성 `gpt-realtime-2.1` / `gpt-transcribe` / `gpt-4o-mini-tts` |
| 상태 저장 | **SQLite** (대화 기억, 조립 진행 상태, 확인 대기) |

**모션 플래너로 cuRobo·MoveIt 같은 무거운 스택을 쓰지 않는다.** 기하학적 경로
계획(캡슐 장애물 투영 + 탄성 밴드)과 속도 스트리밍 추종만으로 실시간 회피를
구현했다. GPU가 필요 없고 지연이 작다.

---

## 3. 프로세스 구조 — 왜 둘로 나눴는가

```
┌───────────────────────────────────┐        ┌──────────────────────────────┐
│  JARVIS 코파일럿                  │        │  로봇 앱                     │
│  (unified_copilot.app)            │        │  (webcam_pick_place)         │
│                                   │        │                              │
│  · 자연어 이해 (정규식 + LLM)     │ ROS 2  │  · 상태기계 15상태           │
│  · 조립 매뉴얼 상태 추적          │ 토픽   │  · 웹캠 3대 삼각측량         │
│  · 사진 판정 (GPT Vision)         │◄──────►│  · 손 추적 · 경로계획        │
│  · 웹 UI (FastAPI, 8765)          │        │  · 파지 · 그리퍼 제어        │
│  · 음성 (별도 프로세스로 격리)    │        │  · AR HUD 렌더링             │
└───────────────────────────────────┘        └──────────────────────────────┘
        ▲                                            ▲
        │ OpenAI API                                 │ DRFL
        ▼                                            ▼
   [클라우드 LLM]                              [로봇 컨트롤러]
```

**분리한 이유는 안전이다.** 로봇의 실시간 제어 루프와 안전 정지 판단이 LLM의
네트워크 지연이나 오류에 영향을 받으면 안 된다. 코파일럿이 죽어도 로봇 앱은
계속 돌고, 키보드로 조작할 수 있다. 반대로 로봇 앱이 죽으면 코파일럿은
"로봇 앱이 실행 중이지 않습니다"라고 정직하게 답한다.

통신은 토픽 4개뿐이다.

| 토픽 | 방향 | 내용 |
|---|---|---|
| `/webcam_pnp/command` | 코파일럿 → 로봇 | `start` / `stop` / `inspect` / `take_from_hand` / `inspect_done` (JSON) |
| `/webcam_pnp/status` | 로봇 → 코파일럿 | `[STATE] 사람이 읽을 메시지` |
| `/webcam_pnp/state_json` | 로봇 → 코파일럿 | 상태 · 진행률 · request_id |
| `/webcam_pnp/inspection` | 로봇 → 코파일럿 | 촬영 결과(사진 경로, 촬영 자세, 가리킨 좌표) |

---

## 4. 전체 데이터 흐름

```
 [말] ──► 호출어 관문 ──► 의도 분류(정규식) ──┬──► 대화 LLM
                                              ├──► 매뉴얼 상태기계
                                              └──► 로봇 스킬 ──► ROS 토픽
                                                                    │
 [웹캠 3대] ─► YOLO + MediaPipe ─► 삼각측량 ─► 3D 세계 모델 ◄───────┘
                                                    │
                                          경로계획(회피) ─► SpeedL 스트리밍
                                                    │
 [손목 D435i] ─► 정밀정렬 · 파지각 계산 ─────────────┘
                                                    │
                                              그리퍼 · 전달
```

**핵심은 "3D 세계 모델"이 중앙에 있다는 점이다.** 픽셀이 아니라 밀리미터 단위
좌표로 세계를 표현하고, 모든 판단(무엇을 가리켰나 · 어디가 비었나 · 손을 어떻게
피하나)이 그 위에서 이루어진다.

---

## 5. 로봇 앱 — 15개 상태의 흐름

```
IDLE ─(S / "가져와")─► HOMING ─► DETECT ─► [ARMED] ─(G)─► APPROACH
                                                              │
                                         REFINE ◄─────────────┘
                                            │
                                         GRASP
                                            │
                    ┌───────────────────────┴──────────────────┐
                    ▼                                          ▼
            DELIVER_TRACK (손 추종)                    TO_PLACE_VIEW
                    │                                          │
            DELIVER_RELEASE                            PLACING → IDLE

  별도 진입:  INSPECT (사진 촬영)   TAKE_FROM_HAND (손에서 받기)
  이상 시  :  ERROR ─(R)─► IDLE
```

각 단계에서 일어나는 일:

| 상태 | 하는 일 |
|---|---|
| **DETECT** | 웹캠 3대로 목표 물체를 삼각측량. worst-gap 40 mm 초과면 폐기 |
| **ARMED** | 계획을 세운 채 대기. 목표가 움직이면 경로도 따라 갱신 |
| **APPROACH** | 경로를 SpeedL 속도 스트림으로 추종. 손이 들어오면 실시간 재계획 |
| **REFINE** | 손목 카메라로 파지점(무게중심)과 파지각(긴 축 ⊥) 계산 |
| **GRASP** | 측정된 높이로 하강 → 그리퍼 닫기 → 폭 확인으로 실제 파지 검증 |
| **DELIVER_TRACK** | 손바닥을 추종. 안정되면 릴리즈 |
| **INSPECT** | 가리킨 지점 위 400 mm 수직, 렌즈 중앙 정렬 후 촬영 |

---

## 6. 코파일럿 — 능력 구성

```
CentralTurnManager  ("유일한 의미론적 진입점")
        └── UnifiedCopilotEngine.handle()
              ├── CONVERSATION  일상 대화 (LLM + 최근 12턴 기억)
              ├── ASSEMBLY      단계 안내 · 상태 판정 (매뉴얼 YAML 기반)
              ├── MANUAL        PDF → 조립서 자동 생성
              ├── SYSTEM        화면 · 작업 시작/종료
              ├── ROBOT_SKILL   가져오기 · 검사 · 손에서 받기
              ├── CONFIRMATION  "응/아니" 확인 처리
              ├── VISION        물체 설명
              └── SEARCH        웹 검색
```

**규칙과 LLM을 의도적으로 분리했다.**

| | 담당 | 특성 |
|---|---|---|
| 정규식 · 템플릿 | 의도 분류, 단계 안내, 확인 질문, 로봇 명령 | 항상 동일, 테스트 가능 |
| LLM | 자유 대화, 사진 판정, 음성 | 매번 다름, 유연 |

**상태를 바꾸는 것(단계 완료 기록, 로봇 이동)은 전부 규칙 쪽**이다. LLM이
자유롭게 문장을 만들면 "넘어갈까요?"와 "넘어갔습니다"가 날마다 달라지고,
그러면 실제 기록과 말이 어긋난다.

---

## 7. 조립 매뉴얼 파이프라인

```
PDF 조립서 ──► 페이지 분석(LLM) ──► 단계 추출 ──► assembly.yaml
                                                      │
                              ┌───────────────────────┤
                              ▼                       ▼
                     단계별 지시문 안내         참고 사진 표시
                              │                       │
                              └──► 작업자 확인 / 사진 판정 ──► 진행 상태 저장
```

`AssemblyStateTracker`가 **현재 단계 / 완료 단계 / 사용자 확인 단계 /
영상 검증 단계**를 구분해 SQLite에 저장한다. 프로그램을 껐다 켜도 이어진다.

단계를 완료로 넘기는 경로는 셋뿐이다.

1. 작업자가 명시적으로 선언 ("4단계 완료했어")
2. 로봇 검사 결과가 `CORRECT` — 자동 기록 후 다음 단계 안내
3. 서로 일치하는 영상 관찰 2회 (`ASSEMBLY_TRANSITION_CONSENSUS=2`)

칭찬 한마디로 단계가 넘어가지 않게 한 것이 설계 의도다.

---

## 8. 인식 모델 학습 파이프라인

`tools/` 아래에 데이터셋 제작 도구 일체가 있다.

```
capture_dataset.py    4카메라 동시 촬영 (버튼 즉시 촬영)
        ↓ 1,000장
autolabel_dataset.py  기존 모델 + 교차뷰 기하 투영으로 자동 라벨
        ↓             (한 카메라에서 검출 → 3D 복원 → 나머지 뷰에 투영)
review_labels.py      사람이 검수 (라벨링 노동 606장 → 166장으로 감소)
        ↓
YOLOv8 학습           mAP50 0.652 → 0.859 (사람 검증 val 기준)
        ↓             wrench 클래스 0.394 → 0.774
test_model_live.py    실시간 검증
```

부수 도구: `check_calibration.py`(카메라 정합도 진단),
`remap_classes.py`(클래스 재매핑), `watch_training.py`(학습 모니터)

---

## 9. 디렉터리 지도

```
cobot2_ws_1/
├── pick_and_place_voice/          ROS 2 패키지 (로봇 앱)
│   ├── robot_control/
│   │   ├── webcam_pick_place.py   메인 앱 (4,756줄) — 상태기계·계획·파지
│   │   ├── ar_hud.py              AR 오버레이 렌더러 (548줄)
│   │   └── onrobot.py             RG2 그리퍼 드라이버
│   ├── object_detection/
│   │   ├── webcam_rig.py          3카메라 리그·삼각측량·MediaPipe (609줄)
│   │   ├── pointing.py            손가락 광선·각도 선택 (161줄)
│   │   └── yolo.py                YOLO 래퍼
│   └── resource/                  학습된 가중치 (tools_v5_4class.pt)
│
├── DUME_COPILOT.../integrated_dum_e_project/    코파일럿
│   ├── unified_copilot/
│   │   ├── app.py                 메인 루프 (920줄)
│   │   ├── engine.py              능력 라우팅 (1,245줄)
│   │   ├── intents.py             의도 분류 정규식
│   │   └── ui/                    웹 UI (HTML/CSS/JS)
│   ├── robot_skills/              로봇 스킬 + ROS 브리지
│   ├── assembly_copilot/          조립 상태 추적·비전 판정
│   ├── manual_generator/          PDF → 조립서 생성
│   └── assembly_manuals/          생성된 조립서 (YAML + 사진)
│
├── tools/                         데이터셋 제작·검증 도구
├── docs/                          기술 문서
└── scripts/                       실행 스크립트
```

---

## 10. 실행 순서

```bash
# 1) 로봇 드라이버
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
     mode:=real host:=192.168.1.100 port:=12345 model:=m0609 name:=dsr01

# 2) 손목 카메라
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
     enable_sync:=false rgb_camera.color_profile:=1280x720x30 \
     depth_module.depth_profile:=848x480x30

# 3) 로봇 앱
bash ~/cobot2_ws_1/scripts/run_webcam_pnp.sh --live --free-wrist --hand-place

# 4) JARVIS 코파일럿 (--voice 를 붙이면 음성, 없으면 채팅)
bash ~/cobot2_ws_1/scripts/run_jarvis.sh
```

---

## 11. 이 시스템을 관통하는 설계 원칙

**1. 측정한 것만 믿는다.** 삼각측량마다 광선 간 최대 간격(worst gap)을 함께
계산하고, 기준을 넘으면 결과를 버린다. 깊이 분리에 실패하면 추측하지 않고
안전한 기본값으로 되돌아간다.

**2. 애매하면 고르지 않는다.** 두 공구가 4° 이내로 붙어 있으면 "어느 쪽이냐"고
되묻는다. 사진으로 판단이 안 되면 `UNCERTAIN`을 반환한다. 확신에 찬 오답이
작업자에게 가장 해롭다.

**3. 판단 근거를 화면에 드러낸다.** 각 공구의 각도, 경로 계획, 파지 높이,
조준 평면 높이가 모두 HUD에 뜬다. "왜 저걸 골랐지?"에 답할 수 없는 시스템은
신뢰할 수도, 디버깅할 수도 없다.

**4. 사람 근처에서는 느리게, 그리고 검증하며.** 손 위에서는 속도 60%,
하강 35 mm/s, 손이 35 mm 움직이면 중단. 그리퍼 폭을 읽어 실제로 잡았는지
확인한 뒤에야 다음 동작으로 넘어간다.

**5. 상태 변경은 규칙이, 표현은 LLM이.** 로봇을 움직이고 진행 상태를 바꾸는
결정은 결정론적 코드가 내린다. LLM은 설명하고 판정할 뿐 실행 권한이 없다.
