from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import re
import threading
import time
from typing import Any

from VoiceProcessing.conversation_router import ConversationRoute

from .engine import CopilotReply, UnifiedCopilotEngine
from .intents import UnifiedIntent


@dataclass
class TaskRecord:
    task_id: str
    turn_id: int
    kind: str
    utterance: str
    status: str = "QUEUED"
    created_at: float = field(default_factory=time.time)
    result: str | None = None


class TaskRegistry:
    """Authoritative task state; dialogue models never invent busy status."""

    def __init__(self) -> None:
        self._items: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()

    def add(self, turn_id: int, index: int, kind: str, utterance: str) -> TaskRecord:
        item = TaskRecord(f"turn-{turn_id}-{index}", turn_id, kind, utterance)
        with self._lock:
            self._items[item.task_id] = item
        return item

    def set_status(self, task_id: str, status: str, result: str | None = None) -> None:
        with self._lock:
            item = self._items.get(task_id)
            if item is not None:
                item.status = status
                item.result = result

    def active(self) -> list[TaskRecord]:
        with self._lock:
            return [item for item in self._items.values()
                    if item.status in {"QUEUED", "RUNNING"}]

    def status(self, task_id: str) -> str | None:
        with self._lock:
            item = self._items.get(task_id)
            return item.status if item is not None else None

    def cancel_active(self) -> int:
        count = 0
        with self._lock:
            for item in self._items.values():
                if item.status in {"QUEUED", "RUNNING"}:
                    item.status = "CANCELLED"
                    count += 1
        return count


class CentralTurnManager:
    """The sole semantic entry point for voice and terminal turns."""

    _STEP = re.compile(r"([0-9]+|[일이삼사오육칠팔구십]+)\s*단계", re.I)
    _SHOW = re.compile(
        r"(?:사진|이미지|참고\s*(?:사진|이미지|자료)|화면).*?"
        r"(?:띄워|보여|열어|표시)|"
        r"(?:띄워|보여|열어|표시).*?"
        r"(?:사진|이미지|참고\s*(?:사진|이미지|자료)|화면)", re.I)
    _EXPLAIN = re.compile(r"설명|어떻게|방법|과정|뭘\s*해야|무엇을\s*해야", re.I)
    _ASSEMBLY_CHECK = re.compile(
        r"(?:단계.*(?:검수|확인)|조립.*(?:검수|확인)|"
        r"(?:이렇게|이대로|이\s*상태).*?(?:맞|괜찮|조이|확인)|"
        r"몇\s*단계|잘\s*(?:됐|했|만들))", re.I)
    _ASSEMBLY_ACTION = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계.*?"
        r"(?:페이지|이미지|사진|자료|설명|검수|확인|보여|띄워|수정|바꿔)", re.I)
    _SEARCH = re.compile(r"맛집|식당|카페|날씨|뉴스|가격|웹\s*검색|인터넷", re.I)
    _STATUS = re.compile(
        r"(?:뭘|무엇|어떤).*?(?:처리|진행)\s*중|요청.*?상태|"
        r"(?:처리|진행)\s*중인\s*요청|요청.*?(?:뭐|무엇|알려)", re.I)
    _CANCEL = re.compile(r"(?:요청|검사|분석|처리).*?(?:취소|그만|중단|멈춰)", re.I)
    _MANUAL_FOLLOWUP = re.compile(r"^(?:그거|그걸|그것|이거|이걸)?\s*만들어\s*줘[.!?]?$", re.I)
    _VISUAL_DIALOGUE = re.compile(
        r"(?:확인|판단|화면).*?(?:어려|불확실).*?(?:질문|물어)|"
        r"(?:나한테|저한테).*?(?:질문|물어).*?(?:단계|판단|유추)", re.I)

    def __init__(self, engine: UnifiedCopilotEngine) -> None:
        self.engine = engine
        self.tasks = TaskRegistry()
        self.focus_step: int | None = None
        self.focus_domain: str = "conversation"
        self.pending_proposal: str | None = None
        self._turn = 0
        self._lock = threading.RLock()

    def classify(self, text: str) -> UnifiedIntent:
        normalized = self._resolve_context(text)
        return self.engine.classify(normalized)

    def handle(self, text: str, *, frames, current_frame,
               timestamp_ms: int) -> CopilotReply:
        with self._lock:
            self._turn += 1
            turn_id = self._turn

        if self._STATUS.search(text):
            return self._task_status_reply()
        if self._VISUAL_DIALOGUE.search(text):
            self.engine.interactive_visual_clarification = True
            return CopilotReply(
                "알겠습니다. 카메라만으로 필수 조건을 확인하지 못하면, "
                "확인하기 쉬운 항목을 한 번에 하나씩 질문하겠습니다.", "SYSTEM")
        if self._CANCEL.search(text):
            cancelled = self.tasks.cancel_active()
            if not self._has_new_request_after_cancel(text):
                return CopilotReply(
                    f"진행 중이거나 대기 중인 요청 {cancelled}개를 취소했습니다."
                    if cancelled else "현재 취소할 요청이 없습니다.", "SYSTEM")

        normalized = self._resolve_context(text)
        parts = self._independent_parts(normalized)
        if len(parts) == 1:
            record = self.tasks.add(turn_id, 1, self._kind(parts[0]), parts[0])
            return self._execute(record, parts[0], frames, current_frame, timestamp_ms)

        records = [self.tasks.add(turn_id, index, self._kind(part), part)
                   for index, part in enumerate(parts, 1)]
        replies: list[CopilotReply | None] = [None] * len(parts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(parts))) as pool:
            future_map = {
                pool.submit(self._execute, record, part, frames, current_frame,
                            timestamp_ms): index
                for index, (record, part) in enumerate(zip(records, parts))
            }
            for future in concurrent.futures.as_completed(future_map):
                index = future_map[future]
                replies[index] = future.result()
        completed = [reply for reply in replies if reply is not None]
        return CopilotReply(
            "\n".join(reply.text for reply in completed),
            "MULTI",
            stop=any(reply.stop for reply in completed),
            action=next((reply.action for reply in completed if reply.action), None),
            reference_image=next(
                (reply.reference_image for reply in completed if reply.reference_image), None),
            reference_label=next(
                (reply.reference_label for reply in completed if reply.reference_label), None),
        )

    def _execute(self, record: TaskRecord, text: str, frames, current_frame,
                 timestamp_ms: int) -> CopilotReply:
        self.tasks.set_status(record.task_id, "RUNNING")
        try:
            intent = self.engine.classify(text)
            if (intent.domain == "ASSEMBLY" and not self.engine.work_session_active
                    and not self._explicit_assembly_request(text)):
                reply = self._conversation(text)
            elif intent.domain == "CONVERSATION":
                reply = self._conversation(text)
            else:
                reply = self.engine.handle(
                    text, frames=frames, current_frame=current_frame,
                    timestamp_ms=timestamp_ms)
            if self.tasks.status(record.task_id) == "CANCELLED":
                return CopilotReply("이 요청은 취소되어 결과를 반영하지 않습니다.", "SYSTEM")
            self._remember_focus(text, intent, reply)
            self.tasks.set_status(record.task_id, "COMPLETED", reply.text)
            return reply
        except Exception:
            self.tasks.set_status(record.task_id, "FAILED")
            raise

    def _conversation(self, text: str) -> CopilotReply:
        state = self.engine.tracker.snapshot()
        reply = self.engine.conversation.respond(
            text, route=ConversationRoute.CHAT,
            context={
                "focus_domain": self.focus_domain,
                "focus_step": self.focus_step,
                "work_active": self.engine.work_session_active,
                "active_step": state.current_step_id,
                "actual_active_tasks": [item.task_id for item in self.tasks.active()],
            })
        self.engine.conversation.remember("user", text, "CONVERSATION")
        return CopilotReply(reply, "CONVERSATION")

    def _resolve_context(self, text: str) -> str:
        value = text.strip()
        visual_answer = self._pending_visual_answer(value)
        if visual_answer == "ACCEPT":
            return "네"
        if visual_answer == "REJECT":
            return "아니요"
        value = re.sub(
            r"(?:그리고\s+)?프린터\s+(?=오늘\s*(?:날씨|기온))", "그리고 ", value)
        if (self.focus_domain == "assembly"
                and re.search(r"(?:현재\s*)?메뉴(?:를|가|는)?\s*(?:확인|보여|띄워)", value)
                and "메뉴얼" not in value):
            # Korean STT frequently drops the final syllable of 메뉴얼.
            # Limit correction to an active assembly context so restaurant/app
            # menu requests are not rewritten.
            value = value.replace("메뉴", "매뉴얼", 1)
        if self.pending_proposal == "GENERATE_MANUAL" and self._MANUAL_FOLLOWUP.fullmatch(value):
            return "다운로드한 PDF 조립서를 분석해서 매뉴얼로 만들어 줘"
        mentions = list(self._STEP.finditer(value))
        if mentions:
            # In "1단계 말고 6단계", the last affirmative mention is target.
            self.focus_step = self._step_number(mentions[-1].group(1))
        elif self.focus_step is not None and (self._SHOW.search(value) or self._EXPLAIN.search(value)):
            value = f"{self.focus_step}단계 {value}"
        return value

    def _pending_visual_answer(self, text: str) -> str | None:
        # Screen-control corrections are new commands, not yes/no evidence for
        # an older visual clarification.
        if re.search(
                r"(?:실제|현재|작업대|카메라).*?"
                r"(?:화면|이미지|영상).*?(?:보여|띄워|띄우|열어|표시)|"
                r"(?:화면|이미지).*?(?:보여|띄워|띄우).*?(?:달라|라고)",
                text, re.I):
            return None
        engine = getattr(self, "engine", None)
        db = getattr(engine, "db", None)
        if db is None:
            return None
        pending = db.pending()
        if pending is None or pending["operation_type"] != "VISUAL_CLARIFICATION":
            return None
        negative = re.search(
            r"아니|않|안\s|못|느슨|틀려|다르|아닌|없어|아직", text, re.I)
        if negative:
            return "REJECT"
        positive = re.search(
            r"(?:네|응|예)|맞(?:아|습니다|아요)|그렇|보여|"
            r"팽팽|걸었|연결했|고정했|조였|체결했|장착했", text, re.I)
        if positive is None:
            return None
        description = str(pending["payload"].get("current", {}).get("description", ""))
        # Explicit yes is sufficient; otherwise require a shared object/state
        # word so a new work report is not mistaken for the previous answer.
        if re.match(r"^(?:네|응|예)(?:\b|[,\s])", text.strip(), re.I):
            return "ACCEPT"
        if re.search(r"^(?:그|이)\s*상태(?:가|는)?\s*맞", text.strip(), re.I):
            return "ACCEPT"
        relation_terms = {
            "팽팽", "평행", "비틀", "높이", "나란히", "맞닿", "밀착",
            "왼쪽", "오른쪽", "위쪽", "아래쪽", "중앙", "장력",
        }
        if any(term in description and term in text for term in relation_terms):
            return "ACCEPT"
        return None

    def _remember_focus(self, text: str, intent: UnifiedIntent, reply: CopilotReply) -> None:
        mentions = list(self._STEP.finditer(text))
        if mentions and intent.domain == "ASSEMBLY":
            self.focus_step = self._step_number(mentions[-1].group(1))
            self.focus_domain = "assembly"
        elif intent.domain == "MANUAL":
            self.focus_domain = "manual"
            if re.search(r"(?:가능|할\s*수|만들어\s*줄)", text):
                self.pending_proposal = "GENERATE_MANUAL"
        elif intent.domain == "CONVERSATION":
            self.focus_domain = "conversation"
        if intent.domain == "MANUAL" and "매뉴얼" in reply.text:
            self.pending_proposal = None

    def _independent_parts(self, text: str) -> list[str]:
        # Split only when two independently executable domains are present.
        if not ((self._ASSEMBLY_CHECK.search(text) or self._ASSEMBLY_ACTION.search(text))
                and self._SEARCH.search(text)):
            return [text]
        match = re.search(
            r"(?:해\s*주고|해주고|하고|그리고|(?:보여|띄워)\s*주면서)", text)
        if match is None:
            return [text]
        left, right = text[:match.start()].strip(), text[match.end():].strip()
        if left and re.search(r"(?:페이지|이미지|사진|자료)$", left):
            left += " 보여 줘"
        return [part for part in (left, right) if part]

    def _kind(self, text: str) -> str:
        if self._ASSEMBLY_CHECK.search(text):
            return "VISION"
        if self._SEARCH.search(text):
            return "SEARCH"
        return self.engine.classify(text).domain

    def _task_status_reply(self) -> CopilotReply:
        active = self.tasks.active()
        if not active:
            return CopilotReply("현재 처리 중이거나 대기 중인 요청은 없습니다.", "SYSTEM")
        labels = ", ".join(f"{item.task_id} {item.kind.lower()}({item.status.lower()})"
                           for item in active)
        return CopilotReply(f"현재 요청 상태는 {labels}입니다.", "SYSTEM")

    @staticmethod
    def _has_new_request_after_cancel(text: str) -> bool:
        return bool(re.search(r"(?:지금|대신|그리고).*?(?:확인|찾아|보여|설명|검수)", text))

    def _explicit_assembly_request(self, text: str) -> bool:
        return bool(re.search(
            r"(?:작업|조립)\s*(?:시작|재개)|\d+\s*단계|"
            r"카메라|RealSense|리얼센스|이렇게|이대로|현재\s*조립|작업\s*상태",
            text, re.I))

    @staticmethod
    def _step_number(token: str) -> int:
        if token.isdigit():
            return int(token)
        digits = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
                  "육": 6, "칠": 7, "팔": 8, "구": 9}
        if "십" in token:
            tens, ones = token.split("십", 1)
            return digits.get(tens, 1) * 10 + digits.get(ones, 0)
        return digits.get(token, 0)
