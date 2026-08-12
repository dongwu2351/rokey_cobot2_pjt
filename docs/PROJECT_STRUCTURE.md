# 프로젝트 구조

## 워크스페이스 전체

```
cobot2_ws_1/
│
├── src/                                    ROS 2 패키지 (빌드 대상)
│   ├── pick_and_place_voice/               로봇 제어 앱
│   │   ├── robot_control/                  상태기계 · 경로계획 · 파지
│   │   ├── object_detection/               비전 · 삼각측량 · 포인팅
│   │   ├── voice_processing/               음성 입출력
│   │   ├── resource/                       학습 가중치 · 캘리브레이션
│   │   └── launch/                         런치 파일
│   └── od_msg/srv/                         커스텀 서비스 정의
│
├── DUME_COPILOT_.../integrated_dum_e_project/   JARVIS 코파일럿
│   ├── unified_copilot/                    대화 엔진 · 의도 분류 · 웹 UI
│   ├── robot_skills/                       로봇 스킬 · ROS 브리지
│   ├── assembly_copilot/                   조립 상태 추적 · 비전 판정
│   ├── manual_generator/                   PDF → 조립서 생성
│   ├── assembly_manuals/                   생성된 조립서 (YAML + 사진)
│   └── copilot_data/                       대화 기억 · 진행 상태 (SQLite)
│
├── tools/                                  데이터셋 제작 · 검증 도구
├── docs/                                   기술 문서
├── scripts/                                실행 스크립트
├── data/                                   실행 산출물 (검사 사진 · 백업)
├── _archive/                               초기 실습 코드 (빌드 제외)
└── build/  install/  log/                  colcon 생성물
```

---

## 핵심 모듈

### 로봇 앱 — `src/pick_and_place_voice/`

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `robot_control/webcam_pick_place.py` | 4,756 | 메인 앱 — 15상태 기계, 경로계획, 파지, 검사 |
| `robot_control/ar_hud.py` | 548 | AR 오버레이 렌더링 |
| `robot_control/onrobot.py` | — | RG2 그리퍼 드라이버 |
| `object_detection/webcam_rig.py` | 609 | 웹캠 3대 리그 · 삼각측량 · MediaPipe |
| `object_detection/pointing.py` | 161 | 손가락 광선 · 각도 기반 선택 |
| `object_detection/yolo.py` | — | YOLO 추론 래퍼 |
| `resource/tools_v5_4class.pt` | 6.1 MB | 공구 인식 모델 (4클래스) |

### 코파일럿 — `unified_copilot/` · `robot_skills/`

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `unified_copilot/app.py` | 920 | 메인 루프 · 카메라 · UI · 음성 |
| `unified_copilot/engine.py` | 1,245 | 능력 라우팅 (8개 도메인) |
| `unified_copilot/intents.py` | — | 의도 분류 정규식 |
| `unified_copilot/ui_server.py` | — | FastAPI 웹 UI (포트 8765) |
| `robot_skills/ros_bridge.py` | — | ROS 토픽 브리지 |
| `robot_skills/inspect_step.py` | — | 검사 스킬 |
| `robot_skills/inspect_vision.py` | — | 사진 판정 (GPT Vision) |
| `robot_skills/take_from_hand.py` | — | 손에서 받기 스킬 |

---

## 실행 구조

```
┌─────────────────────────────┐   ROS 2 토픽   ┌──────────────────────────┐
│  JARVIS 코파일럿            │◄──────────────►│  로봇 앱                 │
│  python -m unified_copilot  │                │  webcam_pick_place       │
│                             │                │                          │
│  자연어 · LLM 판정 · UI     │                │  비전 · 계획 · 제어      │
└─────────────────────────────┘                └──────────────────────────┘
         ▲                                              ▲
         │ OpenAI API                                   │ DRFL
    [클라우드 LLM]                                 [로봇 컨트롤러]
```

**통신 토픽**

| 토픽 | 방향 | 내용 |
|---|---|---|
| `/webcam_pnp/command` | 코파일럿 → 로봇 | start · stop · inspect · take_from_hand |
| `/webcam_pnp/status` | 로봇 → 코파일럿 | 상태 문자열 |
| `/webcam_pnp/state_json` | 로봇 → 코파일럿 | 상태 · 진행률 · request_id |
| `/webcam_pnp/inspection` | 로봇 → 코파일럿 | 촬영 결과 (사진 경로 · 자세 · 좌표) |

---

## 실행 순서

```bash
# 1. 로봇 드라이버
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
     mode:=real host:=192.168.1.100 port:=12345 model:=m0609 name:=dsr01

# 2. 손목 카메라
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
     enable_sync:=false rgb_camera.color_profile:=1280x720x30 \
     depth_module.depth_profile:=848x480x30

# 3. 로봇 앱
bash scripts/run_webcam_pnp.sh --live --free-wrist --hand-place

# 4. JARVIS 코파일럿
bash scripts/run_jarvis.sh          # 채팅 모드 (--voice 로 음성)
```

---

## 도구 · 문서

**`tools/`** — 데이터셋 파이프라인

| 파일 | 역할 |
|---|---|
| `capture_dataset.py` | 4카메라 동시 촬영 |
| `autolabel_dataset.py` | 교차뷰 기하 투영 자동 라벨링 |
| `review_labels.py` | 라벨 검수 |
| `test_model_live.py` | 실시간 모델 검증 |
| `check_calibration.py` | 카메라 정합도 진단 |
| `watch_training.py` | 학습 모니터 |

**`docs/`** — 기술 문서

| 파일 | 내용 |
|---|---|
| `SYSTEM_OVERVIEW.md` | 시스템 전체 지도 |
| `PROJECT_STRUCTURE.md` | 이 문서 |
| `HANDOVER_AND_POINTING.md` | 손에서 받기 · 손가락 타겟팅 |
| `GRASP_YAW.md` | 물체 방향 기반 파지각 |
| `PROJECT_TECHNICAL_REPORT.md` | 알고리즘 · 수식 · 시행착오 전체 |
