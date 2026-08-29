> **팀 프로젝트입니다.** 원본 저장소: https://github.com/yoon-taehwan/rokey_cobot2_pjt
>
> 담당 파트: 실시간 장애물 회피 및 손 제스처 기반 타겟팅·트래킹

# DUM-E — 정밀 조립 보조 협동로봇 시스템

작업자 옆에서 정밀 조립을 돕는 협동로봇 코파일럿.
**말로 지시하고, 손가락으로 가리키면, 로봇이 공구를 건네주고 작업 상태를 봐준다.**

Doosan M0609 + OnRobot RG2 + RealSense D435i + 캘리브레이션된 웹캠 3대 위에서
비전 인식, 동적 장애물 회피, 음성·언어 인터페이스, 작업용 UI를 하나로 통합했다.

---

## 제출 폴더 구조

```
submission/
├── src/                    ROS2 패키지 — 새로 개발한 소스코드
│   ├── pick_and_place_voice/   로봇 제어 앱 (상태기계·비전·경로계획·파지)
│   └── od_msg/                 커스텀 서비스 정의
│
├── models/                 AI 모델 파일 — 학습된 가중치와 보정값
│   ├── tools_v5_4class.pt      공구 인식 YOLOv8 (4클래스, mAP50 0.859)
│   ├── class_name_tool_v5.json 클래스 정의
│   └── T_gripper2camera.npy    손목 카메라 핸드아이 보정 행렬
│
├── api/                    외부 API 스크립트 · 설정
│   ├── unified_copilot/        대화 엔진 · 의도 분류 · 웹 UI
│   ├── robot_skills/           로봇 스킬 · ROS 브리지
│   ├── assembly_copilot/       조립 상태 추적 · 사진 판정
│   ├── manual_generator/       PDF → 조립서 자동 생성
│   ├── llm/                    음성 인식·합성·대화 라우팅
│   ├── config/webcam_calibration/  웹캠 3대 외부 파라미터 (YAML/JSON)
│   └── .env.example            API 키·모델 설정 양식
│
├── scripts/                실행 스크립트
├── tools/                  데이터셋 제작·검증 도구
├── docs/                   기술 문서
└── README.md
```

---

## 설치 및 실행

### 사전 요구

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS | ROS 2 Humble |
| 로봇 드라이버 | `dsr_bringup2` (별도 워크스페이스 `~/cobot_ws`) |
| 카메라 드라이버 | `ros-humble-realsense2-camera` |
| Python | 3.10 — `opencv-python`, `ultralytics`, `mediapipe`, `numpy`, `scipy`, `fastapi`, `uvicorn`, `openai`, `python-dotenv` |

### 1. 워크스페이스 구성

```bash
mkdir -p ~/cobot2_ws_1 && cd ~/cobot2_ws_1
cp -r <제출폴더>/src .
cp -r <제출폴더>/{scripts,tools,docs} .

# 모델 가중치와 보정값을 패키지 리소스로 배치
cp <제출폴더>/models/* src/pick_and_place_voice/resource/

# 웹캠 캘리브레이션 결과 배치
mkdir -p ~/webcam_calibration
cp -r <제출폴더>/api/config/webcam_calibration/* ~/webcam_calibration/
```

### 2. 빌드

```bash
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash     # 로봇 드라이버 언더레이 (순서 중요)
cd ~/cobot2_ws_1 && colcon build --symlink-install
source install/setup.bash
```

### 3. API 키 설정

```bash
cp <제출폴더>/api/.env.example api/llm/VoiceProcessing/.env
# 파일을 열어 OPENAI_API_KEY 를 실제 값으로 채운다
```

### 4. 실행 — 터미널 4개

```bash
# 1) 로봇 드라이버
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
     mode:=real host:=192.168.1.100 port:=12345 model:=m0609 name:=dsr01

# 2) 손목 카메라
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
     enable_sync:=false rgb_camera.color_profile:=1280x720x30 \
     depth_module.depth_profile:=848x480x30

# 3) 로봇 앱
bash scripts/run_webcam_pnp.sh --live --free-wrist --hand-place

# 4) JARVIS 코파일럿 (--voice 를 붙이면 음성 모드)
bash scripts/run_jarvis.sh
```

브라우저에서 `http://127.0.0.1:8765` 접속. 첫 입력에 **"자비스"** 를 붙여 깨운다.

---

## 주요 기능

| 기능 | 사용법 |
|---|---|
| **공구 전달** | "해머 가져와" → 확인 → 손을 내밀면 손 위에 놓아줌 |
| **손가락 지목** | 공구를 가리키면 광선과 함께 대상 확정 → "저거 가져와" |
| **작업 검사** | 작업 지점을 가리키며 "여기 와서 이거 보고 잘하고 있는지 확인해줘"<br>→ 로봇이 그 지점 위로 이동·촬영 → 조립서와 비교 판정 → 단계 자동 진행 |
| **손에서 받기** | 손바닥에 물건을 올리고 "이거 가져다 놔" → 빈 자리를 찾아 정리 |
| **조립 안내** | "작업 시작하자", "3단계 설명해줘", "참고 사진 보여줘" |
| **조립서 생성** | `downloaded_manuals/`에 PDF를 넣고 "이 조립서를 분석해줘" |
| **동적 회피** | 접근 경로에 손이 들어오면 실시간으로 경로를 다시 그림 |

키보드 조작(로봇 앱 창): `S` 준비 · `G` 출발 · `SPACE` 정지 · `R` 복구 · `I` 검사 촬영 ·
`M` 손에서 받기 · `T` 대상 전환 · `1~4` 속도 · `,` `.` 조준 평면 · `-` `=` 촬영 거리

---

## 시스템 구조

```
┌─────────────────────────────┐   ROS 2 토픽   ┌──────────────────────────┐
│  JARVIS 코파일럿 (api/)     │◄──────────────►│  로봇 앱 (src/)          │
│  자연어 · LLM 판정 · 웹 UI  │                │  비전 · 계획 · 제어      │
└─────────────────────────────┘                └──────────────────────────┘
         ▲                                              ▲
         │ OpenAI API                                   │ DRFL
    [클라우드 LLM]                                 [로봇 컨트롤러]
```

**두 프로세스로 나눈 이유는 안전이다.** 로봇의 실시간 제어와 안전 정지 판단이
LLM의 네트워크 지연이나 오류에 영향받지 않는다. 코파일럿이 죽어도 로봇 앱은
키보드로 계속 조작할 수 있다.

통신 토픽 4개: `/webcam_pnp/command` · `/status` · `/state_json` · `/inspection`

---

## 핵심 기술

**웹캠 3대 삼각측량** — 픽셀당 3D 광선을 만들어 최소제곱으로 교점을 구하고,
광선 간 최대 오차(worst gap)를 함께 계산해 기준을 넘으면 결과를 **버린다**.

**손가락 레이저 타겟팅** — 검지 두 관절(MCP·TIP)을 삼각측량해 3D 광선을 만들고,
거리가 아닌 **각도**(22° 콘)로 대상을 고른다. 1·2등 차가 4° 미만이면 되묻는다.

**동적 장애물 회피** — 사람 팔을 캡슐로 모델링해 경로를 밖으로 밀어내고 탄성
밴드로 다듬은 뒤, SpeedL 속도 스트림으로 추종한다. 손이 움직이면 재계획한다.
GPU 기반 플래너(cuRobo) 없이 구현했다.

**방향 인식 파지** — 깊이 마스크로 물체 픽셀만 분리해 무게중심(파지점)과 긴
축(파지각)을 구하고, 손목을 긴 축에 수직으로 돌려 잡는다.

**시각 판정** — 손목 카메라 사진과 조립서 참고 사진을 함께 비전 모델에 넘겨
비교 판정하고, 판단이 안 되면 "확실하지 않다"고 답한다.

---

## 성능

| 항목 | 값 |
|---|---|
| 공구 인식 mAP50 | 0.859 (4클래스) |
| 검사 촬영 조준 오차 | 0 mm (프레임 정중앙) |
| 삼각측량 신뢰 게이트 | 광선 간 최대 40 mm |
| 포인팅 선택 콘 | 22° |
| 빈자리 탐색 검증 | 3,876조합, 안전 위반 0건 |

---

## 문서

| 파일 | 내용 |
|---|---|
| `docs/SYSTEM_OVERVIEW.md` | 시스템 전체 구조와 데이터 흐름 |
| `docs/PROJECT_STRUCTURE.md` | 폴더·모듈 구성 |
| `docs/HANDOVER_AND_POINTING.md` | 손에서 받기 · 손가락 타겟팅 상세 |
| `docs/GRASP_YAW.md` | 물체 방향 기반 파지각 계산 |
| `docs/PROJECT_TECHNICAL_REPORT.md` | 알고리즘 · 수식 · 시행착오 전체 기록 |
| `docs/SELF_EVALUATION.md` | 자체 평가 및 개선 방향 |

---

## 주의

- `.env`(API 키)는 제출물에 포함하지 않았다. `api/.env.example`을 복사해 채울 것
- 웹캠 3대와 손목 카메라의 캘리브레이션 값은 **이 셀의 물리 배치에 종속**된다.
  다른 환경에서는 `tools/check_calibration.py`로 검증 후 재보정이 필요하다
- 로봇 동작은 실물 안전을 전제로 한다. 첫 실행은 `--live` 없이(드라이런) 확인할 것
