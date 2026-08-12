#!/usr/bin/env python3
"""음성 -> 검출 프롬프트. 말한 것을 영어 시각 질의로 바꿔 토픽에 실어 보낸다.

    "망치 가져와"  ->  /dum_e/prompt = "hammer"
                       /dum_e/intent = "FETCH"

★ 이 노드는 카메라를 **절대 열지 않는다.**
  V4L2 장치는 한 프로세스만 열 수 있다. 카메라는 perceive.py 가 쥐고 있고,
  그쪽이 YOLOE 도 직접 돌린다. 여기가 카메라를 건드리면 둘 중 하나가 죽는다.
  (VoiceProcessing 의 assistive_cli/realtime_runtime 은 카메라를 열므로 쓰지 않는다.)

★ 그래서 VoiceProcessing 에서 쓰는 것은 두 조각뿐이다
    voice_pipeline   마이크 -> 웨이크워드 -> VAD -> STT (문장)
    situated_parser  문장 -> PerceptionPlan (영어 시각 질의 + intent)
  물체 검출·grounding·좌표는 전부 vision 쪽 담당이다.

★ 왜 영어로 바꾸나 — YOLOE 의 텍스트 인코더가 영어로 학습됐다. "망치" 를 그대로
  넣으면 못 찾는다. LLM 이 "망치"->"hammer", "파란 L자 지그"->"blue L-shaped jig"
  로 바꿔주는 것이 이 계층의 핵심 가치다.

실행
    python3 voice_prompt_node.py --text "망치 가져와"      # 마이크 없이 한 번
    python3 voice_prompt_node.py --no-wake                 # 웨이크워드 없이 계속 듣기
    python3 voice_prompt_node.py                           # "hello rokey" 부르면 듣기
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))       # VoiceProcessing 패키지를 찾게

from VoiceProcessing.assistive_models import PerceptionDecision
from VoiceProcessing.command_models import Intent
from VoiceProcessing.situated_parser import SituatedCommandParser


PROMPT_TOPIC = "/dum_e/prompt"
INTENT_TOPIC = "/dum_e/intent"
SAY_TOPIC = "/dum_e/say"        # 말한 내용을 다른 노드도 볼 수 있게 (로그·자막용)
FOUND_TOPIC = "/dum_e/found"    # 인지가 대상을 3D 로 확실히 잡았다
TARGET_TOPIC = "/dum_e/target"  # 인지가 보는 물체의 베이스 좌표
SET_B_TOPIC = "/curobo/set_b"   # 목표 B 지정 (펜던트와 같은 경로)
GOAL_TOPIC = "/curobo/goal"     # 계획기가 실제로 들고 있는 목표 (검증용)
TRACK_TOPIC = "/curobo/track"   # 움직이는 물체를 따라갈 때만 (2차용)
APPROVE_TOPIC = "/curobo/approve"

# ★ 물체 중심 바로 위 이만큼에서 멈춘다 [m].
#   바닥 윗면이 z=0 이고 테이프 중심은 z≈0.015 다. 거기로 TCP 를 보내면 그리퍼가
#   바닥을 뚫어야 하므로 계획이 실패한다. 그리고 계획이 실패하면 계획기는
#   **옛 경로를 지우지 않고 그냥 돌아간다**(curobo_hybrid_demo.py:864) —
#   그 상태로 승인하면 옛 목표(B)로 가버린다. 실제로 그렇게 됐다.
#   1차 목표는 '그 위치까지 가기' 이므로 물체 위에 서는 것이 맞다.
#   집는 것은 3차에서 D455 로 다시 재고 내려간다.
APPROACH_Z = 0.10
GOAL_TOLERANCE = 0.03           # 계획기 목표가 이보다 어긋나면 승인하지 않는다

# 검출기에 넘길 수 있는 작업만 통과시킨다. STOP 은 여기로 보내면 안 된다 —
# 정지는 LLM 왕복(수 초)을 태우면 안 되는 안전 경로다. 별도 채널로 처리한다.
GROUNDABLE = (Intent.FETCH, Intent.MOVE, Intent.PLACE)

# 되묻기 답. ★ 부정을 먼저 본다 — "아니 그거 말고" 를 긍정으로 읽으면
#   사람이 거절했는데 로봇이 움직인다. 애매하면 안 가는 쪽이 안전한 방향이다.
# 부정은 **부분 일치**로 본다 — "아니 그거 말고 흰색" 처럼 명령에 섞여 와도
# 일단 멈추는 쪽이 안전한 방향이다.
NO = re.compile(r"(아니|아냐|아뇨|안\s*돼|취소|그만|하지\s*마|멈춰|정지|싫|no|stop)", re.I)

# ★ 긍정은 **발화 전체**가 긍정어로만 이뤄졌을 때만 인정한다.
#   부분 일치로 하면 "흰색 테이프 가져와" 의 '가' 를 긍정으로 읽어서,
#   사람이 **다른 물체를 말했는데 원래 물체로 출발한다.** 실제로 그랬다.
#   "어제", "음냐" 같은 오인식도 부분 일치에서는 전부 승인이 된다.
_YES_TOKEN = (
    r"(?:응+|웅+|어+|음+|ㅇ+|네+|넹|예+|그래+|그럼|그렇게|맞아+|맞아요|맞습니다|"
    r"좋아+|좋아요|좋습니다|알겠어+|알겠습니다|알았어+|오케이?|ok|okay|yes|yep|"
    r"고고|진행|시작|가자|가져와줘|가져와|가져다줘|해줘|해|부탁해|부탁|콜)"
)
YES_FULL = re.compile(rf"^{_YES_TOKEN}(?:[\s,.!?~]*{_YES_TOKEN})*[\s,.!?~]*$", re.I)

# ★ 사람이 많은 곳에서는 "응" 만으로 실기를 움직이면 안 된다.
#   로봇이 "가져갈까요?" 하고 물은 직후, 옆에서 다른 대화를 하던 사람이 우연히
#   "응" 이라고 말할 수 있다. 우리 시스템에는 화자 검증이 없다 — 웨이크워드는
#   명령 시작만 보호하고, 열려 있는 확인 창은 보호하지 못한다.
#   --crowded 를 켜면 동작을 뜻하는 말이 반드시 들어가야 승인한다.
_YES_STRICT_TOKEN = (
    # 어미를 받아야 한다 — "진행" 만 허용하면 실제로 사람이 말하는 "진행해" 가
    # 안 걸린다 (테스트로 잡았다).
    r"(?:(?:실행|진행|시작|출발)(?:해|해줘|하자|해요|합니다)?"
    r"|가져와줘|가져와|가져다줘|해줘|고고|오케이|okay|ok)"
)
YES_STRICT = re.compile(
    rf"^(?:{_YES_TOKEN}[\s,.!?~]*)*{_YES_STRICT_TOKEN}[\s,.!?~]*$", re.I)

FOUND_TIMEOUT_S = 20.0          # 이 시간 안에 못 찾으면 포기하고 추적을 끈다


def eul(word):
    """받침에 맞는 조사. TTS 가 '테이프을(를)' 이라고 읽으면 알아듣기 어렵다."""
    word = (word or "").strip()
    if not word:
        return "을"
    ch = word[-1]
    if "가" <= ch <= "힣":
        return "을" if (ord(ch) - 0xAC00) % 28 else "를"
    return "을"


class Voice:
    """음성 출력 창구. 실제 합성·재생·취소는 SpeechManager 가 전담한다.

    ★ say() 는 **부르는 쪽을 막지 않는다.** 예전에는 여기서 TTS 왕복(4.9초)을
      그대로 기다렸고, 그게 ROS 콜백 스레드였다. 되묻는 동안 좌표와 계획기
      목표가 갱신되지 않아 승인 검증이 낡은 값을 볼 수 있었다.
    """

    def __init__(self, enabled=True):
        from speech_manager import (PRIO_CONFIRM, PRIO_SAFETY,  # noqa: F401
                                    PRIO_STATUS, SpeechManager)
        self.mgr = SpeechManager(enabled=enabled)
        self.PRIO_SAFETY, self.PRIO_CONFIRM = PRIO_SAFETY, PRIO_CONFIRM
        self.PRIO_STATUS = PRIO_STATUS

    def say(self, text, pub=None, *, priority=None):
        if pub is not None:
            pub.send(pub.say, text)
        self.mgr.say(text, priority=self.PRIO_STATUS if priority is None
                     else priority)

    def prepare(self, text):
        """미리 합성해 둔다. 인지가 물체를 찾는 동안 질문을 만들어 놓는 용도."""
        self.mgr.prepare(text)

    def cancel(self, reason=""):
        self.mgr.cancel(reason)

    def new_task(self):
        """작업이 바뀌었다 — 이전 작업의 음성은 더 이상 재생하지 않는다."""
        self.mgr.cancel()

    def close(self):
        self.mgr.close()


class PromptPublisher:
    """ROS 발행. --no-ros 로 끄면 화면 출력만 한다 (파서만 시험할 때)."""

    def __init__(self, enabled=True):
        self.node = None
        self.prompt = self.intent = self.say = None
        if not enabled:
            return
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from std_msgs.msg import String
        rclpy.init()
        self.rclpy, self.String = rclpy, String
        self.node = Node("dum_e_voice_prompt")
        # ★ latched (transient_local) — 마지막 값이 늦게 붙는 구독자에게도 간다.
        #   말을 먼저 하고 perceive 를 나중에 띄워도 대상이 유지된다. 보통 순서로
        #   쓰면 DDS 디스커버리(1초 안팎) 전에 발행한 첫 명령이 그냥 사라진다.
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prompt = self.node.create_publisher(String, PROMPT_TOPIC, latched)
        self.intent = self.node.create_publisher(String, INTENT_TOPIC, latched)
        self.say = self.node.create_publisher(String, SAY_TOPIC, 10)

        from std_msgs.msg import Bool, Empty
        from geometry_msgs.msg import Point, PointStamped
        from visualization_msgs.msg import Marker
        self.Bool, self.Empty, self.Point = Bool, Empty, Point
        self.track = self.node.create_publisher(Bool, TRACK_TOPIC, 1)
        self.approve = self.node.create_publisher(Empty, APPROVE_TOPIC, 1)
        # ★ 정지 물체는 '추적' 이 아니라 '목표 지정' 이다.
        #   펜던트가 B 를 찍는 것과 **똑같은 경로**를 쓴다. 그래야 B 와 추적이
        #   서로 끄고 켜며 싸우지 않는다 (_apply_goal 은 B 를 직접 찍으면
        #   추적을 끈다 — 둘은 애초에 배타적으로 설계돼 있다).
        self.set_b = self.node.create_publisher(Point, SET_B_TOPIC, 1)

        # 계획기가 실제로 어디를 목표로 잡았는지 되읽는다. 승인 직전에 이걸로
        # 검증한다 — 안 맞으면 승인하지 않는다.
        self.last_target = None      # 인지가 본 물체 (베이스 좌표)
        self.last_goal = None        # 계획기가 들고 있는 목표
        self.on_target = None        # 좌표를 기다리던 쪽이 있으면 알려 준다
        self.node.create_subscription(
            PointStamped, TARGET_TOPIC, self._target_cb, 10)
        self.node.create_subscription(
            Marker, GOAL_TOPIC,
            lambda m: setattr(self, "last_goal",
                              (m.pose.position.x, m.pose.position.y,
                               m.pose.position.z)), 1)

        # 인지가 대상을 잡았다는 알림. on_found 는 나중에 꽂는다.
        self.on_found = None
        self.node.create_subscription(String, FOUND_TOPIC, self._found_cb, 10)
        # ★ 전용 스레드에서 spin 한다. 메인 스레드는 run_once() 안에서 마이크를
        #   기다리며 몇 초씩 막혀 있으므로, 거기서 spin_once 를 부를 틈이 없다.
        self._spin = threading.Thread(target=self._spin_loop, daemon=True)
        self._stop = threading.Event()
        self._spin.start()

    def send_goal_b(self, xyz):
        """목표 B 를 이 좌표로 옮긴다. 계획기가 PREVIEW 로 돌아가 다시 계획한다."""
        if self.node is None:
            return
        m = self.Point()
        m.x, m.y, m.z = (float(v) for v in xyz)
        self.set_b.publish(m)

    def _target_cb(self, m):
        self.last_target = (m.point.x, m.point.y, m.point.z)
        if self.on_target is not None:
            self.on_target()

    def _found_cb(self, m):
        if self.on_found is not None:
            self.on_found(m.data.strip())

    def _spin_loop(self):
        while not self._stop.is_set():
            try:
                self.rclpy.spin_once(self.node, timeout_sec=0.2)
            except Exception:                       # noqa: BLE001
                return

    def send_track(self, on: bool):
        """추적 on/off. 켜면 curobo 가 /dum_e/target 을 목표로 미리보기를 그린다."""
        if self.node is None:
            return
        self.track.publish(self.Bool(data=bool(on)))

    def send_approve(self):
        if self.node is None:
            return
        self.approve.publish(self.Empty())

    def send(self, pub, text):
        if self.node is None:
            return
        pub.publish(self.String(data=text))

    def close(self, settle=0.5):
        if self.node is None:
            return
        # 발행 직후 바로 종료하면 아직 안 나간 메시지가 버려진다 (--text 모드).
        time.sleep(settle)
        self._stop.set()
        self._spin.join(timeout=1.0)
        self.node.destroy_node()
        try:
            self.rclpy.shutdown()
        except Exception:
            pass


def plan_to_prompt(plan):
    """PerceptionPlan -> (검출 프롬프트, intent, 사람에게 할 말, 발행해도 되나).

    ★ target_description 을 먼저 쓴다. category 는 'screw' 처럼 뭉뚱그려지지만
      description 은 'long silver screw' 라서 YOLOE 가 훨씬 잘 찾는다.
    """
    if plan.decision == PerceptionDecision.STOP:
        return None, "STOP", "정지합니다.", False
    if plan.decision == PerceptionDecision.CLARIFY:
        return None, plan.intent.value, plan.clarification_question, False
    if plan.decision != PerceptionDecision.GROUND:
        return None, plan.intent.value, None, False
    if plan.intent not in GROUNDABLE:
        return None, plan.intent.value, None, False
    query = plan.grounding_query
    if not query:
        return None, plan.intent.value, None, False
    return query.strip(), plan.intent.value, None, True


class Session:
    """말 -> 찾기 -> 되묻기 -> 승인 -> 이동. 한 번에 하나의 작업만 진행한다.

        IDLE      명령을 기다린다
        SEARCHING 프롬프트를 보냈고 인지가 찾아주기를 기다린다 (미리보기 중, 로봇 정지)
        CONFIRM   "찾았습니다. 가져갈까요?" 를 묻고 대답을 기다린다

    ★ 왜 track 을 먼저 켜고 나중에 승인하나
      curobo 는 track 이 켜져야 미리보기 경로를 그린다. 승인은 그 경로를 보고
      하는 것이므로 순서가 이렇게 된다. track 만으로는 로봇이 움직이지 않는다 —
      approve 가 있어야 출발한다. 되묻는 동안 사람은 이미 경로를 보고 있는 셈이다.
    """

    def __init__(self, pub, voice, *, go=True, approach_z=APPROACH_Z, track=False,
                 crowded=False):
        self.pub, self.voice, self.go = pub, voice, go
        self.approach_z = approach_z
        self.crowded = crowded
        self.track = track          # 움직이는 물체를 따라갈 때만 (2차용)
        self.goal = None            # 계획기에 보낸 목표 (승인 전 검증 기준)
        self.state = "IDLE"
        self.prompt = None          # 검출기에 보낸 영어 질의
        self.spoken = None          # 사용자가 실제로 말한 한국어 이름
        self.since = 0.0
        self._await_target = None   # found 는 왔는데 좌표가 아직일 때의 대기 표시
        self.t_command = self.t_confirm = None      # 구간 시간 측정용
        # ★ 직전에 되물은 계획. 다음 발화를 **그 질문의 답**으로 읽게 하는 열쇠다.
        #   이게 없으면 "가지고와" -> "무엇을?" -> "초록색 테이프" -> "그걸로 뭘?"
        #   처럼 매번 처음부터 다시 묻는다 (파서가 문맥을 못 보므로).
        self.pending = None

    # ── 인지 콜백 (다른 스레드에서 들어온다) ─────────────────────────
    def on_found(self, label):
        # 조용히 버리지 않는다 — 되묻기가 안 나오는 원인을 찾을 수 있는 유일한 지점이다.
        if label != self.prompt:
            print(f"  [found 무시] '{label}' 은 지금 찾는 대상('{self.prompt}')이 아님",
                  flush=True)
            return
        if self.state != "SEARCHING":
            print(f"  [found 무시] 상태가 {self.state} — 되묻기는 SEARCHING 에서만",
                  flush=True)
            return
        # ★ 여기서 기다리면 안 된다. 이 콜백은 ROS spin 스레드에서 돈다 —
        #   막으면 그동안 /dum_e/target 과 /curobo/goal 이 하나도 갱신되지 않아
        #   승인 직전 검증이 낡은 값을 보게 된다. 좌표가 아직이면 표시만 해 두고
        #   즉시 돌아간다. on_target 이 도착하는 순간 이어서 진행한다.
        if self.pub.last_target is None:
            self._await_target = time.time()
            print("  [found] 좌표를 기다립니다", flush=True)
            return
        self._begin_confirm()

    def on_target(self):
        """좌표가 도착했다. found 를 먼저 받고 기다리던 중이면 이어서 진행한다."""
        if self._await_target is None or self.state != "SEARCHING":
            return
        self._await_target = None
        self._begin_confirm()

    def _begin_confirm(self):
        """찾은 좌표를 목표로 밀어 넣고 사람에게 되묻는다. 막히지 않는다."""
        p = self.pub.last_target
        if p is None:
            return
        self.goal = (p[0], p[1], p[2] + self.approach_z)
        self.pub.send_goal_b(self.goal)
        print(f"  -> {SET_B_TOPIC} = ({self.goal[0]:+.3f}, {self.goal[1]:+.3f}, "
              f"{self.goal[2]:+.3f})   물체 {p[2]:.3f} 위 {self.approach_z*100:.0f}cm",
              flush=True)
        self.state = "CONFIRM"
        self.since = time.time()
        self.t_confirm = time.monotonic()
        # 명령을 알아들은 시점에 미리 합성해 뒀다면 여기서 즉시 재생된다.
        self.voice.say(self.confirm_text(), self.pub, priority=self.voice.PRIO_CONFIRM)

    def confirm_text(self):
        name = self.spoken or self.prompt
        return (f"{name}{eul(name)} 찾았습니다. "
                f"{self.approach_z*100:.0f}센티미터 위로 갈까요?")

    # ── 시간 초과 (메인 루프가 주기적으로 부른다) ───────────────────
    def tick(self):
        # found 는 왔는데 좌표가 안 오는 경우. 콜백에서 기다리지 않고 여기서 판정한다.
        if self._await_target is not None and time.time() - self._await_target > 2.0:
            self._await_target = None
            print("  [found] 좌표가 2초 안에 안 옴 — 포기", flush=True)
            self.voice.say("위치를 확인하지 못했습니다. 다시 말씀해 주세요.", self.pub,
                           priority=self.voice.PRIO_SAFETY)
            self.cancel()
            return
        if self.state == "SEARCHING" and time.time() - self.since > FOUND_TIMEOUT_S:
            name = self.spoken or self.prompt
            self.voice.say(f"{name}{eul(name)} 찾지 못했습니다. 다시 말씀해 주세요.",
                           self.pub)
            self.cancel()

    def cancel(self):
        self.pub.send_track(False)
        self.state, self.prompt, self.spoken = "IDLE", None, None
        self.pending = None

    # ── 되묻기에 대한 대답 ──────────────────────────────────────────
    def answer(self, text):
        """(처리했나). 긴 문장만 새 명령으로 넘긴다."""
        if self.state != "CONFIRM":
            return False
        text = text.strip()
        if NO.search(text):
            self.voice.say("취소했습니다.", self.pub)
            self.cancel()
            return True
        yes = YES_STRICT.match(text) if self.crowded else YES_FULL.match(text)
        if self.crowded and not yes and YES_FULL.match(text):
            # 긍정이긴 한데 사람 많은 곳에서 쓰기엔 약하다. 무시하지 말고 알려 준다.
            print(f"  [승인 보류] '{text}' — 사람이 많은 모드입니다", flush=True)
            self.voice.say("진행할까요? 진행해 라고 말씀해 주세요.", self.pub,
                           priority=self.voice.PRIO_CONFIRM)
            return True
        if not yes:
            # 긍정도 부정도 아니다. 새 명령일 수도, 그냥 오인식일 수도 있다.
            # 여기서 판단하지 않고 파서에게 넘긴다 — 진짜 명령이면 GROUND 가
            # 나오고, 아니면 handle() 이 **되묻기를 유지한 채 다시 묻는다.**
            # (예전엔 파서의 "어떤 작업을 도와드릴까요?" 가 그대로 나가서
            #  사람 입장에선 대답이 통째로 무시당한 것으로 보였다)
            print(f"  [대답 아님] '{text}' — 새 명령인지 확인합니다", flush=True)
            return False
        # 긍정이다 — 다만 바로 출발시키지 않는다.
        if True:
            # ★ 승인 전에 계획기의 목표가 정말 그 물체인지 되읽어 확인한다.
            #   예전엔 그냥 approve 를 쐈다. 그래서 계획기가 아직 옛 목표(펜던트로
            #   찍어 둔 B)를 들고 있어도 그대로 출발했고, 사람은 "승인했더니
            #   엉뚱한 데로 간다" 만 보게 됐다. 어긋나면 안 가는 쪽이 안전하다.
            g, want = self.pub.last_goal, self.goal
            if want is not None and g is not None:
                off = max(abs(a - b) for a, b in zip(g, want))
                if off > GOAL_TOLERANCE:
                    print(f"  [승인 거부] 계획기 목표 ({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}) 가 "
                          f"물체 ({want[0]:+.3f},{want[1]:+.3f},{want[2]:+.3f}) 와 "
                          f"{off*100:.0f}cm 어긋남", flush=True)
                    self.voice.say("목표가 아직 물체로 맞춰지지 않았습니다. "
                                   "다시 말씀해 주세요.", self.pub)
                    self.cancel()
                    return True
            self.pub.send_approve()
            print(f"  -> {APPROVE_TOPIC}  (사람이 음성으로 승인)", flush=True)
            self.voice.say("이동합니다.", self.pub, priority=self.voice.PRIO_SAFETY)
            self._report_timing()
            self.state = "IDLE"
            return True

    def _report_timing(self):
        """한 작업이 끝날 때 구간 시간을 한 줄로. 어디가 느린지 추측하지 않기 위해."""
        now = time.monotonic()
        parts = []
        if self.t_command is not None and self.t_confirm is not None:
            parts.append(f"명령→찾음 {self.t_confirm - self.t_command:5.2f}s")
        if self.t_confirm is not None:
            parts.append(f"찾음→승인 {now - self.t_confirm:5.2f}s")
        lat = self.voice.mgr.last_latency_ms()
        if lat is not None:
            parts.append(f"질문 재생까지 {lat:4.0f}ms")
        s = self.voice.mgr.stats
        parts.append(f"음성캐시 적중 {s['cache_hit']}/{s['cache_hit'] + s['cache_miss']}")
        print("  [시간] " + "   ".join(parts), flush=True)
        return False

    # ── 새 명령 ────────────────────────────────────────────────────
    def command(self, prompt, intent, spoken):
        # ★ 상태를 **프롬프트보다 먼저** 바꾼다.
        #   인지는 프롬프트를 받은 다음 프레임(30ms)에 찾아서 /dum_e/found 를 쏜다.
        #   그 콜백은 다른 스레드로 들어오는데, 그때 state 가 아직 IDLE 이면
        #   on_found 가 조기 return 해서 **되묻기가 통째로 사라진다.**
        #   (물체가 이미 화면에 보일수록 더 확실하게 진다 — 빠를수록 깨지는 경합이다)
        self.prompt, self.spoken = prompt, spoken
        self.goal = None
        self._await_target = None
        self.voice.new_task()          # 이전 작업의 음성은 더 이상 재생하지 않는다
        self.state, self.since = "SEARCHING", time.time()
        self.t_command = time.monotonic()
        # ★ 선합성 — 물어볼 문장은 **지금 이미 정해져 있다.**
        #   인지가 물체를 찾는 1초 남짓 동안 미리 만들어 두면, 정작 물어볼 때는
        #   캐시에서 즉시 나온다. 이게 체감 지연을 가장 크게 줄인다.
        self.voice.prepare(self.confirm_text())
        if self.go:
            # 정지 물체는 추적을 쓰지 않는다. 목표는 물체를 **찾은 뒤** on_found 에서
            # /curobo/set_b 로 한 번 지정한다 — 그게 펜던트가 B 를 찍는 것과 같은
            # 경로이고, 계획기 안에서 B 와 추적이 서로 끄고 켜며 싸우지 않는다.
            self.pub.send_track(bool(self.track))
        # 프롬프트를 **맨 마지막에** 보낸다 — 받는 쪽이 이걸 트리거로 쓰므로,
        # 이 시점엔 계획기 준비도 상태 전환도 이미 끝나 있어야 한다.
        self.pub.send(self.pub.intent, intent)
        self.pub.send(self.pub.prompt, prompt)
        print(f"  -> {PROMPT_TOPIC} = '{prompt}'  ({intent})", flush=True)


def handle(text, parser, pub, voice, session, *, verbose=True):
    # 되묻는 중이면 먼저 대답으로 해석한다. 예/아니오가 아니면 새 명령으로 흘린다.
    was_confirm = session is not None and session.state == "CONFIRM"
    if session is not None and session.answer(text):
        return None

    # 직전에 되물었으면 그 계획을 같이 넘긴다 — 파서가 "이 발화는 그 질문의 답"
    # 으로 읽고 대상 범주와 intent 를 유지한 채 새 특징만 결합한다.
    pending = session.pending if session is not None else None
    try:
        plan = parser.parse(text, pending_plan=pending)
    except Exception as exc:                        # noqa: BLE001
        # ★ 여기서 죽으면 안 된다. 실제로 죽었다 —
        #   "흰색 테이프 안 보여?" 같은 질문에 모델이 decision=GROUND 에
        #   조작 intent 가 아닌 값을 붙여 보냈고, 스키마 검증(의도된 안전장치)이
        #   ValueError 를 냈다. 그게 그대로 올라와 노드가 통째로 종료됐다.
        #   스키마가 거부하는 것은 **정상 동작**이다. 거부를 처리하는 게 우리 몫이다.
        print(f"  [해석 실패] {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        if was_confirm:
            voice.say("네 또는 아니오로 답해 주세요.", pub,
                      priority=voice.PRIO_CONFIRM)
        else:
            voice.say("다시 말씀해 주세요.", pub)
        return None
    prompt, intent, say, ok = plan_to_prompt(plan)
    if session is not None:
        # 또 되물으면 그 계획을 물고 간다. 아니면 문맥을 놓아 준다 —
        # 안 놓으면 다음 명령이 옛 질문에 오염된다.
        session.pending = plan if plan.decision == PerceptionDecision.CLARIFY else None
    if verbose:
        print(json.dumps({
            "utterance": text,
            "decision": plan.decision.value,
            "intent": intent,
            "target_category": plan.target_category,
            "target_description": plan.target_description,
            "alternatives": list(plan.visual_query_alternatives),
            "spatial_relation": plan.spatial_relation.value if plan.spatial_relation else None,
            "prompt": prompt,
            "llm": parser.last_used_llm,
        }, ensure_ascii=False, indent=2), flush=True)

    if plan.decision == PerceptionDecision.STOP:
        # 추적을 끄면 목표가 사라져 로봇이 선다. 다만 이건 **부드러운 정지**다 —
        # 말 -> STT -> LLM 왕복이 수 초 걸릴 수 있으므로 진짜 비상정지가 아니다.
        # 비상정지는 이 경로에 태우면 안 된다 (하드웨어 E-stop / 전용 저지연 채널).
        if session is not None:
            session.cancel()
        else:
            pub.send_track(False)
        voice.say("정지합니다.", pub)
        return plan

    # ★ 되묻던 중에 온 발화가 새 명령이 아니면, 되묻기를 **버리지 않고** 다시 묻는다.
    #   여기가 "대답했는데 무엇을 도와드릴까요를 반복" 하던 지점이다.
    if was_confirm and plan.decision != PerceptionDecision.GROUND:
        session.pending = None       # 파서의 되묻기 문맥은 물지 않는다
        name = session.spoken or session.prompt
        print(f"  [되묻기 유지] '{text}' 는 명령이 아님 — 다시 묻습니다", flush=True)
        voice.say("네 또는 아니오로 답해 주세요.", pub,
                  priority=voice.PRIO_CONFIRM)      # 고정 문구라 캐시에서 즉시 나온다
        return plan

    if say:                                   # 되묻기 질문 (정보가 부족할 때)
        voice.say(say, pub)
        return plan

    if ok and session is not None:
        session.command(prompt, intent, plan.source_object_expression)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="마이크 대신 이 문장을 한 번 처리한다")
    ap.add_argument("--no-ros", action="store_true", help="발행하지 않고 화면만")
    ap.add_argument("--no-go", dest="go", action="store_false",
                    help="미리보기(/curobo/track)를 켜지 않는다 — 인지만 시험할 때")
    ap.add_argument("--approach-z", type=float, default=APPROACH_Z,
                    help=f"물체 중심 위 몇 m 에 설 것인가 (기본 {APPROACH_Z})")
    ap.add_argument("--track", action="store_true",
                    help="움직이는 물체를 따라간다 (2차용). 정지 물체에는 쓰지 말 것")
    ap.add_argument("--crowded", action="store_true",
                    help="사람이 많은 곳: '응' 만으로는 승인하지 않고 '진행해' 를 요구한다")
    ap.add_argument("--no-speak", action="store_true", help="TTS 를 끈다")
    ap.add_argument("--no-disclose", dest="disclose", action="store_false",
                    help="시작할 때 'AI 합성 음성' 고지를 생략한다")
    ap.add_argument("--no-wake", action="store_true", help="웨이크워드 없이 바로 듣기")
    ap.add_argument("--device-index", type=int)
    ap.add_argument("--sample-rate", type=int)
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--start-timeout-ms", type=int, default=10_000)
    ap.add_argument("--end-silence-ms", type=int, default=450)
    ap.add_argument("--max-record-ms", type=int, default=12_000)
    a = ap.parse_args()

    parser = SituatedCommandParser()
    pub = PromptPublisher(enabled=not a.no_ros)
    voice = Voice(enabled=not a.no_speak)
    session = (Session(pub, voice, go=a.go, approach_z=a.approach_z, track=a.track,
                       crowded=a.crowded)
               if not a.no_ros else None)
    if session is not None:
        pub.on_found = session.on_found
        pub.on_target = session.on_target
        # ★ 시작할 때 대상을 비운다.
        #   프롬프트는 latched(transient_local) 라 마지막 값이 살아남는다.
        #   그 덕에 순서 상관없이 붙일 수 있지만, 껐다 켜도 옛 물체가 그대로
        #   선택돼 있어서 "내가 시키지도 않았는데 잡고 있다" 가 된다.
        #   새 세션은 빈 상태로 시작하는 게 사람의 기대와 맞는다.
        pub.send(pub.prompt, "")
        pub.send_track(False)
        print("  대상 초기화 (이전 세션의 물체를 놓았습니다)", flush=True)

    try:
        if a.text:
            handle(a.text, parser, pub, voice, session)
            return

        from VoiceProcessing.voice_pipeline import build_pipeline

        pipeline = build_pipeline(SimpleNamespace(
            no_wake=a.no_wake, continuous=True,
            device_index=a.device_index, sample_rate=a.sample_rate,
            vad_threshold=a.vad_threshold,
            start_timeout_ms=a.start_timeout_ms,
            max_record_ms=a.max_record_ms,
            end_silence_ms=a.end_silence_ms,
        ))
        # 구버전 명령 라우터는 쓰지 않는다 — 의미 해석은 situated_parser 몫이다.
        pipeline.router = SimpleNamespace(
            parse_command=lambda utterance, context: None)

        if a.disclose and not a.no_speak:
            # .env.example 에 적힌 요구사항 — 합성 음성임을 사람에게 알린다.
            from VoiceProcessing.TTS import AI_VOICE_DISCLOSURE
            voice.say(AI_VOICE_DISCLOSURE, pub)

        print("듣는 중" + ("" if a.no_wake else " (\"hello rokey\" 라고 부르세요)"))
        while True:
            if session is not None:
                session.tick()
            state = session.state if session is not None else "IDLE"
            print(f"\n[LISTENING] ({state}) ...", flush=True)
            result = pipeline.run_once(wait_for_wake=not a.no_wake)
            text = (result.transcript or "").strip()
            if not text:
                continue
            # ★ 사람이 말했으면 로봇은 입을 다문다. 질문이 다 끝나기를 기다렸다가
            #   답해야 한다면 그건 대화가 아니라 순서 지키기다.
            voice.cancel("사용자 발화")
            print(f"[들림] {text}", flush=True)
            try:
                handle(text, parser, pub, voice, session)
            except Exception as exc:                # noqa: BLE001
                # 한 턴이 실패해도 다음 턴은 받아야 한다. 음성 인터페이스가
                # 죽으면 사람은 로봇에게 정지조차 말할 수 없게 된다.
                import traceback
                print(f"[턴 실패] {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
    except KeyboardInterrupt:
        pass
    finally:
        voice.close()
        pub.close()


if __name__ == "__main__":
    main()
