# fetch_object 물리 스킬 — 로봇 기능 명세 (핸드오프 §3 회신)

물리 노드는 `~/cobot2_ws_1`의 `pick_and_place_voice` 패키지
`webcam_pick_place` 앱이며 **별도 프로세스**로 실행된다. 코파일럿은
`robot_skills.ros_bridge`를 통해 시작/정지 버튼만 누르고 상태를 구독한다.
LLM/코파일럿은 관절값·TCP 좌표·속도를 절대 생성하지 않는다.

## 실행 환경
- ROS 2 Humble, `ROS_DOMAIN_ID` 미설정(기본 0)
- Doosan: id `dsr01`, model `m0609`, namespace `/dsr01`, host 192.168.1.100
- 그리퍼: OnRobot RG2 (Modbus TCP 192.168.1.1:502)
- 단위: 로봇 mm/ZYZ deg(posx), 캘리브레이션 번들 m
- 프레임: 컨트롤러 base(= 웹캠 extrinsics 기준). `~/webcam_calibration/
  results/transforms/production_transforms.yaml`의 `T_base_cam{0,1,2}`,
  `T_flange_wrist_depth`(손목 D435i, holdout 검증본) 사용

## 카메라 4대
| 카메라 | 역할 |
|---|---|
| cam0/1/2 (C270, USB 경로 고정) | 대상 삼각측량 + MediaPipe 손(장애물/전달 목표) |
| 손목 D435i | 파지 직전 정밀 정렬(bbox+깊이 마스크), 전달 아님 |

## 제어권
- 접근/손 추적: SpeedL 스트리밍(TTL 0.2s 데드맨, 슬루 700mm/s²,
  장애물=손+전완 캡슐을 반경+2cm로 회피, z≤400 안전영역, 팔 아래 통과 금지)
- 파지/릴리즈: movel/movej (스트림 정지 확인 후 전환, 동시 발행 없음)
- ServoJ는 사용하지 않음(스펙의 MOVING_SERVOJ 상태는 SpeedL 접근에 대응)

## 판정
- 파지 성공: RG2 폭 readback (물체 파지 ≈ 25~30mm, 빈손 ≈ 10.7mm,
  무반응 = 안전회로 래치 → 전원 재인가 필요)
- 손 전달 완료: 손 위치 중앙값 0.7s 안정 + 정렬 후 손바닥 위 ~5cm 릴리즈,
  이후 `Delivered` 상태

## 토픽 계약 (브리지가 사용)
- 구독(코파일럿→로봇) `/webcam_pnp/command` std_msgs/String:
  `"start"` | `"stop"` | `{"command":"start","request_id":...}`
- 발행(로봇→코파일럿) `/webcam_pnp/status` String `"[STATE] message"`,
  `/webcam_pnp/state_json` String JSON
  `{"state","status","request_id","delivered"}` — 1초 하트비트 포함

## 실행/정지
```bash
# 물리 노드 (별도 터미널; dry-run이 기본, --live만 실기)
bash ~/cobot2_ws_1/scripts/run_webcam_pnp.sh --live --free-wrist --hand-place
```
- 즉시 정지: 창에서 SPACE, 또는 `/webcam_pnp/command`에 `"stop"`
  (스킬 매니저 cancel이 이 정지를 발행) → 스트림 TTL 0.2s 내 정지
- 안전 상태: SAFE_OFF2(안전영역 위반)는 펜던트에서만 해제
- 드라이런: `--live` 없이 실행하면 동일 파이프라인이 로그만 남김

## 의존성
- underlay: `~/cobot_ws` (dsr_bringup2/dsr_msgs2), `~/cobot2_ws_1` install
- pip: ultralytics, mediapipe, pyrealsense2(드라이버는 ROS 노드), pymodbus
- realsense-ros는 align 없이 raw depth만으로 동작(SW 정합 내장)
