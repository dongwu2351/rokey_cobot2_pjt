# JARVIS UI 디자인 브리프 (Claude에 그대로 붙여넣는 프롬프트)

아래 `---` 사이 전체를 복사해서 Claude에 붙여넣으면 됩니다.
`[내가 원하는 방향]` 부분만 취향에 맞게 채우세요.

---

## 의뢰 내용

산업용 협동로봇 작업장에서 쓰는 **JARVIS 워크스테이션 UI**를 새로 디자인해줘.
로컬 웹앱(FastAPI + 바닐라 JS, 브라우저 전체화면)이고, 실제로 사람 옆에서 로봇이
움직이는 동안 보는 화면이라 "예쁘지만 정보가 즉시 읽히는" 게 핵심이야.

산출물: `index.html` + `style.css` + `app.js` 3개 파일 (완성본 코드).

### 제품이 하는 일

두산 M0609 협동로봇 + 카메라 4대(고정 웹캠 3대 + 손목 RealSense D435i)로 두 가지를 한다.

1. **작업 코파일럿** — 사용자가 미니 컨베이어 모듈 같은 제품을 조립하는 동안, 손목
   카메라 영상을 AI가 보고 "지금 몇 단계인지 / 제대로 했는지"를 판정하고 다음 단계를
   안내한다. (총 8단계짜리 조립 매뉴얼)
2. **물리적 심부름** — "해머 가져와줘"라고 하면 로봇이 웹캠으로 해머를 찾아 삼각측량하고,
   사람 팔을 실시간으로 피해 접근해서 집은 뒤, 사용자가 내민 손바닥 위에 올려준다.
   "멈춰" 한마디면 즉시 정지.

사용자는 두 손으로 조립 작업을 하고 있어서 화면을 **1~2초 힐끗 보는 게 전부**다.
그 순간에 "지금 로봇이 뭘 하고 있고, 내가 뭘 해야 하는지"가 읽혀야 한다.

### 디자인 방향

[내가 원하는 방향 — 예: "현재의 SF 홀로그램 톤을 유지하되 더 세련되게" 또는
"차분한 산업용 계기판 느낌으로 완전히 새로" 또는 "Braun/Teenage Engineering 같은
절제된 하드웨어 UI" 등]

지금 화면은 시안 네온(#66f6ff) + 다크 배경의 아이언맨 JARVIS 톤이고,
파티클 배경 · 스캔라인 · 홀로그램 얼굴(눈/입이 말할 때 움직임) · 회전 궤도링이 있다.
이 컨셉을 유지할지 갈아엎을지는 위 방향에 따라 판단해줘.

---

## 화면 구성 — 두 개의 모드

이 UI는 하나의 페이지가 **두 가지 모드** 사이를 전환한다. 지금은 1번만 제대로 있고,
**2번을 이번에 제대로 설계해야 한다.**

### 모드 1 — 어시스턴트 (대기/대화 상태)

로봇이 놀고 있고 사용자가 말을 거는 상태. 지금 구현돼 있는 화면.

- 중앙에 큰 아이덴티티 오브젝트(현재는 홀로그램 얼굴 + 궤도링)
- 상태 라벨 / 큰 제목 / 한 줄 상태 메시지
- 하단에 대화 로그 + 텍스트 입력창
- 어시스턴트는 4가지 상태를 가진다:
  - `DORMANT` — 잠들어 있음. "JARVIS"라고 불러야 깨어남 (웨이크워드 게이트)
  - `AWAKE` — 듣고 있음
  - `THINKING` — 요청 처리 중
  - `ERROR` — 연결 문제
- 상태별로 시각적 차이가 분명해야 한다 (지금은 dormant면 얼굴이 어두워지고 궤도가 멈춤)

### 모드 2 — 작업 분석 (★ 이번에 새로 설계할 화면)

사용자가 "작업 시작" 하거나 로봇 심부름이 실행되면 전환되는 화면.
어시스턴트 오브젝트는 작아져서 구석으로 물러나고, **작업 정보가 주인공**이 된다.

이 화면이 동시에 보여줘야 하는 정보 5가지:

**① 조립 진행 상태**
- 제품명 / 현재 단계 번호와 전체 개수 (예: 3 / 8) / 단계 제목 / 상세 지시문(길다, 2~4줄)
- 완료된 단계 목록 (진행 바 혹은 스텝 인디케이터로 — 8단계를 한눈에)
- 상태: `IN_PROGRESS` / `WAITING` / `COMPLETE`

**② 라이브 카메라 + AI 시각 판정**
- 손목 RealSense 영상이 실시간으로 흐른다 (JPEG 폴링, 약 3fps)
- 그 위에 AI 판정 결과가 얹힌다. 판정은 5종:
  - `CORRECT` — 잘 하고 있음
  - `NEEDS_CORRECTION` — 잘못 조립됨 (어디가 틀렸는지 설명 있음)
  - `UNCERTAIN` — 판단 불가 (필수 체크가 안 보임)
  - `WRONG_STEP` — 다른 단계를 하고 있음
  - `NOT_VISIBLE` — 작업물이 화면에 안 보임
- 판정은 여러 개의 **체크 항목**으로 구성된다. 각 항목은 `true` / `false` / `unknown`.
  (예: "타이밍 벨트가 양쪽 풀리에 모두 걸려 있는가" → true)
  체크 항목별 결과가 리스트로 보여야 하고, `false`인 항목이 눈에 띄어야 한다.
- 판정에는 시간이 걸린다(수 초~1분). **분석 중 상태**의 표현이 필요하다.

**③ 로봇 작업 진행 (심부름 실행 중일 때)**
로봇이 물리적으로 움직이는 동안, 어디까지 왔는지가 진행 타임라인으로 보여야 한다.
실제 상태 시퀀스:

```
ACCEPTED → SEARCHING_TARGET → TARGET_LOCKED → PLANNING_PREGRASP
→ MOVING_SERVOJ(접근·회피) → ALIGNING_SPEEDL(정밀정렬) → GRASPING
→ VERIFYING_GRASP → TRACKING_HAND(손 추적) → WAITING_FOR_HANDOVER
→ RELEASING → RETREATING → SUCCEEDED
```
종료 상태: `SUCCEEDED` / `CANCELLED` / `FAILED` / `SAFETY_STOPPED`
진행률(0~1)과 한국어 설명 문장이 함께 온다.

**중요**: 어떤 상태는 **사용자에게 행동을 요구**한다. 이건 화면에서 가장 크게 보여야 한다.
- `TRACKING_HAND` / `WAITING_FOR_HANDOVER` → **"손을 내밀어 주세요"**
- 로봇이 팔에 막혀 대기 중 → **"팔을 치워 주세요"**
- `SAFETY_STOPPED` → **"안전 정지됨"** + 복구 안내

**④ 즉시 정지**
로봇이 움직이는 동안에는 **항상 보이는 정지 버튼**이 필요하다.
(사용자가 급할 때 화면에서 찾느라 헤매면 안 되는 요소. 다만 조립 중 오조작도 곤란하니
정지 버튼 스타일링에 대한 판단은 맡길게.)

**⑤ 대화 로그**
로봇 진행 상황이 어시스턴트 말풍선으로 계속 흘러들어온다.
("사람 팔을 피해 접근하고 있습니다" → "손을 추적하고 있습니다" → "전달했습니다")
작업 화면에서도 이 로그가 보여야 하지만, ①~④를 가리면 안 된다.

---

## 데이터 계약 (실제 API — 이대로 동작함)

**폴링**: `GET /api/state` — 250ms 간격. 아래 JSON이 그대로 온다.

```json
{
  "assistant": {
    "mode": "DORMANT|AWAKE|THINKING|ERROR",
    "message": "온라인 상태 · 말씀하세요",
    "wake_word": "jarvis",
    "api_available": false,
    "voice_available": false
  },
  "assembly": {
    "active": true,
    "product": "미니 컨베이어 모듈 (모터 구동부 + 벨트 조립)",
    "step": 1,
    "total": 8,
    "title": "1단계 - 부품 준비 (레이아웃 확인)",
    "instruction": "작업대에 모터, 다공판 2개, 타이밍벨트... (200자 내외 긴 문장)",
    "status": "IN_PROGRESS",
    "completed_steps": []
  },
  "robot": {
    "available": true,
    "mode": "ros|mock|off",
    "busy": false,
    "state": "TRACKING_HAND",
    "message": "손을 추적하고 있습니다. 손을 내밀어 주세요.",
    "progress": 0.85,
    "query": "해머",
    "error_code": null,
    "last_outcome": "SUCCEEDED",
    "last_outcome_message": "해머을(를) 손 위에 전달했습니다."
  },
  "conversation": [
    {"role": "user|assistant", "text": "해머 가져와줘", "at": 1786322501.59}
  ],
  "system": {
    "camera": "online|starting",
    "started_at": 1786322000.0,
    "last_frame_at": 1786322510.2,
    "last_request_at": null,
    "last_reply_at": null
  }
}
```

**영상**: `GET /api/frame.jpg` — 캐시 무효화용 쿼리 붙여서 약 300ms 간격 폴링.
프레임이 없으면 204. (즉 "카메라 오프라인" 상태 처리 필요)

**입력**: `POST /api/input` body `{"text": "..."}`
응답: `{"accepted": true, "queued": true}` 또는
`{"accepted": false, "reason": "wake_word_required"}` ← 잠든 상태에서 웨이크워드 없이
입력한 경우. 이때 사용자에게 "먼저 JARVIS라고 불러주세요"를 알려야 한다.

**대기 전환**: `POST /api/sleep`

### 아직 API에 없는 값 (설계에 필요하면 제안해줘)

AI 시각 판정(②)의 상세 데이터는 아직 `/api/state`에 실려 있지 않다.
어떤 필드가 있으면 좋을지 JSON 스키마로 제안해주면 백엔드에 추가할게.
예상 형태:
```json
"vision": {
  "verdict": "CORRECT|NEEDS_CORRECTION|UNCERTAIN|WRONG_STEP|NOT_VISIBLE",
  "analyzing": false,
  "checks": [{"id": "belt_on_pulleys", "description": "타이밍 벨트가 양쪽 풀리에 걸림", "result": true}],
  "summary": "3단계 조립이 정상적으로 완료됐습니다.",
  "reference_image": "/api/reference.jpg"
}
```

---

## 기술 제약

- **의존성 없음**: 외부 CDN·프레임워크 금지. 바닐라 HTML/CSS/JS만. (오프라인 작업장에서 돌아감)
- **웹폰트 금지**: 폰트 CDN 접근 불가. 시스템 폰트 스택으로 해결하거나 CSS로 표현.
  한글이 주 언어 → Pretendard/Noto Sans KR 우선, 없으면 system-ui 폴백.
  숫자가 정렬되는 곳은 `font-variant-numeric: tabular-nums`.
- **다크 환경 고정**: 작업장 조명 아래 모니터. 다크 테마 단일로 간다 (라이트 모드 불필요).
- **해상도**: 주 타깃은 데스크톱 와이드(1920×1080). 태블릿(900px 이하)에서는 세로로 쌓이게.
- **`prefers-reduced-motion` 존중**.
- 폴링은 250ms이므로 **DOM을 매번 통째로 갈아엎지 말 것** (대화 로그는 개수 변화 시에만 재렌더 등).
- 캔버스 애니메이션을 쓴다면 GPU를 과하게 먹지 않게 — 같은 PC에서 YOLO 추론이 돌고 있다.

## 현재 파일 위치 (참고용 — 교체 대상)

```
unified_copilot/ui/
├── index.html   49줄  — 마크업
├── style.css    5줄(압축) — 시안 네온 다크 테마
├── app.js       12줄(압축) — 폴링·렌더·파티클 배경
└── debug.html   진단용 (raw JSON + 카메라, 이건 그대로 둬도 됨)
```

현재 레이아웃: 좌측 어시스턴트 스테이지 + 우측 작업 패널(작업 시작 시 슬라이드 인) +
하단 전체폭 대화창. 작업 모드에서 어시스턴트가 0.73배로 축소되며 그리드 비율이 바뀐다.

## 원하는 결과

1. 먼저 **디자인 방향 한 문단** + 컬러/타이포/레이아웃 토큰 정리
2. 두 모드의 **레이아웃 구조 설명** (특히 모드 2를 어떻게 구성했는지와 그 이유)
3. `index.html`, `style.css`, `app.js` 전체 코드
4. 백엔드에 추가로 필요한 API 필드가 있다면 그 스키마

상태가 많으니, 모든 상태가 실제로 어떻게 보이는지 확인할 수 있게
**데모 모드**(`?demo=1` 같은 걸로 상태를 순환시키며 보여주는)가 있으면 좋겠다.

---

## (참고) 이 브리프 자체에 대한 메모 — 프롬프트에 포함 안 해도 됨

- 로봇 상태 17종은 `robot_skills/models.py`의 `SKILL_STATES`와 일치
- 조립 판정 5종은 `assembly_copilot`의 VLM 판정 결과
- `robot.mode`가 `mock`이면 로봇 없이 도는 테스트 모드 — UI에 표시해두면 시연 때 헷갈리지 않음
