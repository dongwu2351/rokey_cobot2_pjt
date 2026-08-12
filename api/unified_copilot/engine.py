from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from VoiceProcessing.conversation_router import ConversationRoute, ConversationRouter

from assembly_copilot.assessment_validator import AssessmentValidator
from assembly_copilot.copilot import AssemblyCopilot
from assembly_copilot.manual_retriever import ManualRetriever
from assembly_copilot.models import AssemblyObservation
from assembly_copilot.multimodal_assessor import MultimodalAssessor
from assembly_copilot.question_router import AssemblyQuestionRouter
from assembly_copilot.state_tracker import AssemblyStateTracker

from .intents import UnifiedIntentRouter, extract_step_number
from .manual_service import ManualGenerationService
from .memory import CopilotStateDB


@dataclass(frozen=True)
class CopilotReply:
    text: str
    domain: str
    stop: bool = False
    action: str | None = None
    reference_image: str | None = None
    reference_label: str | None = None


class UnifiedCopilotEngine:
    def __init__(self, manual, *, data_dir: Path, downloaded_dir: Path,
                 manuals_dir: Path, vision_model: str | None = None,
                 conversation_client: Any | None = None,
                 assessor_client: Any | None = None,
                 manual_service: ManualGenerationService | None = None) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.router = UnifiedIntentRouter()
        self.db = CopilotStateDB(data_dir / "copilot.sqlite3")
        self.conversation = ConversationRouter(
            client=conversation_client,
            memory_path=data_dir / "conversation.sqlite3")
        self.tracker = AssemblyStateTracker(manual)
        saved_state = self.db.load_runtime_state(f"assembly:{manual.manual_id}")
        if saved_state:
            self.tracker.restore(saved_state.get("current_step_id"),
                                 saved_state.get("completed_steps", []),
                                 saved_state.get("user_confirmed_steps", []),
                                 saved_state.get("verified_completed_steps", []),
                                 saved_state.get("progress_update_source", "RESTORED"))
        self.local_copilot = AssemblyCopilot(manual)
        self.question_router = AssemblyQuestionRouter()
        self.retriever = ManualRetriever(manual)
        self.assessor = MultimodalAssessor(model=vision_model, client=assessor_client)
        self.validator = AssessmentValidator(manual)
        self.transition_consensus_required = max(
            1, int(os.getenv("ASSEMBLY_TRANSITION_CONSENSUS", "2")))
        self._transition_candidate: tuple[str, str] | None = None
        self._transition_hits = 0
        self.manuals = manual_service or ManualGenerationService(downloaded_dir, manuals_dir)
        # A loaded manual does not mean the user has entered an assembly task.
        self.work_session_active = False
        self.interactive_visual_clarification = (
            os.getenv("DUME_INTERACTIVE_VISUAL_CLARIFICATION", "true").lower()
            in {"1", "true", "yes", "on"})
        self._visual_question_history: dict[int, set[str]] = {}
        # Physical robot skills. Injected by the runtime (app.py); when None
        # every fetch request is answered honestly as unavailable. The engine
        # only ever produces semantic requests and confirmations - motion,
        # cancellation and safety live entirely in the skill manager and the
        # physical node behind it.
        self.robot_skills = None
        # One follow-up job, run when the current one succeeds. "이거 갖다
        # 놓고 저거 가져와" is two physical jobs and the manager runs exactly
        # one at a time on purpose - so the second waits here rather than
        # being dropped or racing the first.
        self.queued_robot_request = None
        self.queued_robot_reply = ""

    def _robot_skill_active(self) -> bool:
        manager = getattr(self, "robot_skills", None)
        return bool(manager is not None and manager.busy)

    def classify(self, text: str):
        """Expose the cheap gate so the runtime can isolate long-running jobs."""
        return self.router.route(
            text, has_pending_confirmation=self.db.pending() is not None,
            robot_skill_active=self._robot_skill_active())

    def handle(self, text: str, *, frames, current_frame,
               timestamp_ms: int) -> CopilotReply:
        db = getattr(self, "db", None)
        pending = db.pending() if db is not None else None
        intent = self.router.route(
            text, has_pending_confirmation=pending is not None,
            robot_skill_active=self._robot_skill_active())
        self.db.event("USER_UTTERANCE", {"text": text, "domain": intent.domain,
                                          "intent": intent.intent})
        # ConversationRouter.route() records CONVERSATION turns itself. Other
        # capability lanes must also feed the shared dialogue memory so follow-up
        # phrases such as "내 위치", "그 근처", and "그걸로" retain meaning.
        if intent.domain != "CONVERSATION":
            self.conversation.remember("user", text, intent.domain)
        if intent.domain == "SYSTEM":
            if intent.intent == "CLARIFY_STOP_SCOPE":
                return CopilotReply(
                    "어떤 것을 종료할지 지정해 주세요. 작업만 멈추려면 ‘작업 일시정지’, "
                    "화면만 닫으려면 ‘화면 끄기’, 프로그램 전체를 끝내려면 "
                    "‘코파일럿 종료’라고 말씀해 주세요.", "SYSTEM")
            if intent.intent == "SAVE_PROGRESS_AND_CLARIFY_STOP":
                number = extract_step_number(text)
                if number is None:
                    return CopilotReply("저장할 단계 번호를 확인하지 못했습니다.", "SYSTEM")
                state = self._save_progress(number)
                if isinstance(state, CopilotReply):
                    return state
                return CopilotReply(
                    f"사용자 확인 기준으로 {number}단계까지 저장했습니다. 작업만 멈추려면 "
                    "‘작업 일시정지’, 화면만 닫으려면 ‘화면 끄기’, 프로그램 전체를 "
                    "끝내려면 ‘코파일럿 종료’라고 말씀해 주세요.", "SYSTEM")
            if intent.intent == "PAUSE_WORK":
                self.work_session_active = False
                self._persist_assembly_state()
                return CopilotReply(
                    "조립 작업을 일시정지하고 현재 진행 상태를 저장했습니다. "
                    "다시 시작할 때는 ‘작업 재개’라고 말씀해 주세요.", "SYSTEM")
            if intent.intent == "STOP_COPILOT":
                self._persist_assembly_state()
                return CopilotReply(
                    "현재 진행 상태를 저장하고 음성 코파일럿을 종료합니다.",
                    "SYSTEM", stop=True)
            if intent.intent == "SAVE_PROGRESS_AND_DISPLAY_OFF":
                number = extract_step_number(text)
                if number is None:
                    return CopilotReply("저장할 단계 번호를 확인하지 못했습니다.", "SYSTEM")
                state = self._save_progress(number)
                if isinstance(state, CopilotReply):
                    return state
                return CopilotReply(
                    f"{number}단계까지 완료한 것으로 저장했습니다. 다음은 "
                    f"{state.current_step_id or '전체 완료'}이며 작업 화면을 닫습니다.",
                    "SYSTEM", action="DISPLAY_OFF")
            if intent.intent == "SAVE_PROGRESS_AND_STOP":
                number = extract_step_number(text)
                if number is None:
                    return CopilotReply("저장할 단계 번호를 확인하지 못했습니다.", "SYSTEM")
                state = self._save_progress(number)
                if isinstance(state, CopilotReply):
                    return state
                return CopilotReply(
                    f"{number}단계까지 완료한 것으로 저장하고 작업 화면과 통합 "
                    f"코파일럿을 종료합니다. 다음 시작 단계는 "
                    f"{state.current_step_id or '전체 완료'}입니다.",
                    "SYSTEM", stop=True)
            if intent.intent == "DISPLAY_ON":
                return CopilotReply("작업 화면을 표시합니다.", "SYSTEM",
                                    action="DISPLAY_ON")
            if intent.intent == "DISPLAY_LIVE":
                return CopilotReply(
                    "참고 사진을 내리고 현재 RealSense 작업대 카메라 "
                    "화면을 표시합니다.", "SYSTEM", action="DISPLAY_ON")
            if intent.intent == "DISPLAY_OFF":
                return CopilotReply("작업 화면을 닫았습니다. 음성 코파일럿은 계속 동작합니다.",
                                    "SYSTEM", action="DISPLAY_OFF")
            if intent.intent == "DISPLAY_FEEDBACK":
                return CopilotReply(
                    "맞습니다. 참고 사진을 표시해 달라는 요청이 아닌데 "
                    "잘못 해석했습니다. 추가 화면 변경 없이 실시간 작업 "
                    "화면을 그대로 유지하겠습니다.", "SYSTEM")
            if intent.intent == "STATE_FEEDBACK":
                current = self.tracker.step
                current_label = (f"{current.order}단계" if current is not None
                                 else "전체 완료 상태")
                return CopilotReply(
                    "맞습니다. 완료 여부를 묻는 질문을 완료 명령으로 "
                    f"잘못 해석했습니다. 이 발화로는 상태를 더 변경하지 않고, "
                    f"현재 기록은 {current_label}입니다. 다음 현재 화면 확인 "
                    "요청은 VLM 검사로 처리하겠습니다.", "SYSTEM")
            if intent.intent == "VISION_FEEDBACK":
                return CopilotReply(
                    "맞습니다. 현재 작업을 잘하고 있는지 묻는 질문에는 "
                    "유사한 단계를 나열하지 않고, 현재 추적 단계의 필수 "
                    "조건별로 맞는 점과 부족한 점만 설명하겠습니다.", "SYSTEM")
            if intent.intent == "CAMERA_FEEDBACK":
                if frames:
                    return CopilotReply(
                        f"현재 요청에 RealSense 최근 프레임 {len(frames)}장이 "
                        "실제로 전달되고 있습니다. 화면 정보가 없다는 이전 "
                        "응답은 잘못된 대화 분류 결과였으며, 참고 사진을 "
                        "표시할 이유도 없습니다. 작업 상태 확인 요청은 이 "
                        "프레임을 VLM에 전달하도록 처리합니다.", "SYSTEM")
                return CopilotReply(
                    "현재 요청 시점에는 RealSense 최근 프레임이 전달되지 "
                    "않았습니다. 카메라 프레임 수신 상태를 확인해야 합니다.",
                    "SYSTEM")
            if intent.intent == "SHOW_REFERENCE":
                number = extract_step_number(text)
                target = (self._step_by_order(number) if number is not None
                          else self.tracker.step)
                reference = self._reference_image_for_step(target)
                if reference is None:
                    return CopilotReply(
                        "요청한 단계에 표시할 JPG 또는 PNG 참고 이미지가 없습니다. "
                        "실시간 작업 화면만 표시합니다.", "SYSTEM", action="DISPLAY_ON")
                return CopilotReply(
                    f"{target.order}단계, {target.title}의 참고 사진을 표시합니다. "
                    f"현재 추적 단계는 {self.tracker.step.order}단계로 유지합니다.",
                    "SYSTEM", action="DISPLAY_ON", reference_image=reference,
                    reference_label=f"{target.order}단계 참고 사진")
            if intent.intent == "START_WORK":
                return self._start_work(text)
            if intent.intent == "SAVE_AND_STOP":
                state = self.tracker.snapshot()
                self.db.save_runtime_state(
                    f"assembly:{self.tracker.manual.manual_id}",
                    {"current_step_id": state.current_step_id,
                     "completed_steps": list(state.completed_steps)})
                return CopilotReply(
                    f"현재 작업 단계({state.current_step_id or '완료'})를 저장하고 "
                    "작업 화면과 통합 코파일럿을 종료합니다.", "SYSTEM", stop=True)
            return CopilotReply(
                "시스템 제어 요청을 정확히 구분하지 못했습니다. 작업만 멈추려면 "
                "‘작업 일시정지’, 화면만 닫으려면 ‘화면 끄기’, 전체 종료는 "
                "‘코파일럿 종료’라고 말씀해 주세요.", "SYSTEM")
        if intent.domain == "ROBOT_SKILL":
            return self._robot_skill(intent)
        if intent.domain == "CONFIRMATION":
            return self._confirmation(intent.intent)
        if intent.domain == "MANUAL":
            return self._manual_request()
        if intent.domain == "ASSEMBLY":
            if intent.intent == "SHOW_LOADED_MANUAL":
                return self._show_loaded_manual(text)
            if intent.intent == "USER_PROGRESS_UPDATE":
                return self._user_progress_update(text)
            if intent.intent == "ACTIVE_STEP_UPDATE":
                return self._active_step_update(text)
            return self._assembly(text, frames, timestamp_ms)
        if intent.domain == "VISION":
            return self._describe(text, current_frame)
        if intent.domain == "SEARCH":
            return self._web_search(text)
        return self._chat(text, frames, current_frame, timestamp_ms)

    def _show_loaded_manual(self, text: str) -> CopilotReply:
        mentions = re.findall(r"([0-9]+|[일이삼사오육칠팔구십]+)\s*단계", text)
        number = extract_step_number(f"{mentions[-1]}단계") if mentions else None
        target = self._step_by_order(number) if number is not None else self.tracker.step
        steps = self.tracker.manual.steps
        if target is None:
            return CopilotReply(
                f"현재 {self.tracker.manual.product} 매뉴얼이 로드되어 있고, "
                f"총 {len(steps)}단계입니다.", "ASSEMBLY")
        reference = self._reference_image_for_step(target)
        summary = (
            f"현재 {self.tracker.manual.product} 매뉴얼이 로드되어 있습니다. "
            f"{target.order}단계는 {target.title}이며, {target.instruction}")
        if reference is None:
            return CopilotReply(summary + " 표시할 참고 이미지는 없습니다.", "ASSEMBLY")
        source_note = (
            "이 YAML 매뉴얼에는 원본 PDF 페이지 번호가 연결되어 있지 않아, "
            f"{target.order}단계 PDF 페이지 대신 단계 참고 사진 "
            f"{Path(reference).name}을 표시합니다. "
            if "페이지" in text else "")
        return CopilotReply(
            summary + " " + source_note + "참고 이미지를 화면 오른쪽 REFERENCE 패널에 표시합니다.",
            "ASSEMBLY",
            action="DISPLAY_ON", reference_image=reference,
            reference_label=f"{target.order}단계 참고 사진 · {Path(reference).name}")

    def _save_progress(self, number: int):
        try:
            state = self.tracker.complete_through(number)
        except ValueError as exc:
            return CopilotReply(str(exc), "SYSTEM")
        self._persist_assembly_state()
        return state

    def _user_progress_update(self, text: str) -> CopilotReply:
        number = extract_step_number(text)
        if number is None:
            return CopilotReply(
                "완료로 기록할 단계 번호를 확인하지 못했습니다.", "ASSEMBLY")
        # An explicit "N단계까지 완료로 저장" command supersedes an
        # older visual question or a completion-scope proposal. Leaving that
        # operation waiting made every subsequent utterance repeat the prompt.
        if (hasattr(self.db, "pending") and hasattr(self.db, "resolve_pending")
                and self.db.pending() is not None):
            self.db.resolve_pending("SUPERSEDED_BY_USER_PROGRESS")
        state = self._save_progress(number)
        if isinstance(state, CopilotReply):
            return state
        self.work_session_active = True
        target = self.tracker.step
        next_label = (f"현재 작업 위치를 {target.order}단계로 변경했습니다."
                      if target is not None else "매뉴얼의 마지막 단계까지 기록했습니다.")
        return CopilotReply(
            f"사용자 확인 기준으로 {number}단계까지 완료로 기록했습니다. "
            f"{next_label} 카메라로 검증된 완료 기록과는 별도로 보관합니다.",
            "ASSEMBLY", action="DISPLAY_ON",
            reference_image=self._current_reference_image())

    def _active_step_update(self, text: str) -> CopilotReply:
        number = extract_step_number(text)
        if number is None:
            bare = re.search(
                r"([0-9]+|[일이삼사오육칠팔구십]+)\s*(?:단계)?(?:로|으로)", text)
            if bare is not None:
                number = extract_step_number(f"{bare.group(1)}단계")
        target = self._step_by_order(number)
        if number is None or target is None:
            return CopilotReply("변경할 작업 단계 번호를 확인하지 못했습니다.", "ASSEMBLY")
        # Declaring step N as the current work position implies that the
        # preceding steps are operator-confirmed, not vision-verified.
        previous_orders = [step.order for step in self.tracker.manual.steps
                           if step.order < number]
        if previous_orders:
            self.tracker.complete_through(max(previous_orders))
        self.tracker.select_step_number(number)
        self.work_session_active = True
        self._persist_assembly_state()
        return CopilotReply(
            f"사용자 확인 기준으로 현재 작업 위치를 {number}단계로 변경했습니다. "
            f"{number}단계 이전 완료 기록은 미검증 상태로 보관합니다. "
            f"{target.instruction}", "ASSEMBLY", action="DISPLAY_ON",
            reference_image=self._reference_image_for_step(target),
            reference_label=f"{number}단계 참고 사진")

    def _start_work(self, text: str) -> CopilotReply:
        self.work_session_active = True
        completed_match = re.search(
            r"([0-9]+|[일이삼사오육칠팔구십]+)\s*단계\s*(?:까지)?[^.!?]*?"
            r"(?:완료|끝냈|마쳤|했(?:고|어|습니다))", text, re.I)
        completed_number = (extract_step_number(completed_match.group(0))
                            if completed_match else None)
        if completed_number is not None:
            saved = self._save_progress(completed_number)
            if isinstance(saved, CopilotReply):
                return saved

        all_mentions = list(re.finditer(
            r"([0-9]+|[일이삼사오육칠팔구십]+)\s*단계", text, re.I))
        # In a compound utterance ("1단계 완료했고 3단계부터 시작"),
        # the final step mention is the requested work pointer; earlier mentions
        # describe progress evidence.
        target_match = all_mentions[-1] if all_mentions else None
        number = (extract_step_number(target_match.group(0))
                  if target_match is not None else None)
        if number is not None:
            target = self._step_by_order(number)
            if target is None:
                return CopilotReply(f"매뉴얼에 {number}단계가 없어요.", "ASSEMBLY")
            current = self.tracker.step
            if current is not None and current.id != target.id:
                self.db.set_pending("SELECT_ASSEMBLY_STEP", {
                    "step_order": number, "source": "USER_CONFIRMED"})
                skipped = [
                    step.order for step in self.tracker.manual.steps
                    if completed_number is not None
                    and completed_number < step.order < number
                ]
                completion_note = (
                    f"{completed_number}단계 완료는 기록했습니다. "
                    if completed_number is not None else "")
                skip_note = (
                    f"{', '.join(map(str, skipped))}단계 완료 기록은 없습니다. "
                    if skipped else "")
                return CopilotReply(
                    f"{completion_note}{skip_note}지금 작업 위치는 {current.order}단계예요. "
                    f"{number}단계부터 진행하도록 작업 위치를 바꿀까요?", "ASSEMBLY")
            try:
                self.tracker.select_step_number(number)
            except ValueError as exc:
                return CopilotReply(str(exc), "SYSTEM")
        step = self.tracker.step
        if step is None:
            return CopilotReply("현재 매뉴얼의 모든 단계가 완료된 상태입니다.", "ASSEMBLY")
        return CopilotReply(
            f"{step.title} 작업을 시작합니다. {step.instruction}",
            "ASSEMBLY", action="DISPLAY_ON",
            reference_image=self._current_reference_image())

    def _current_reference_image(self) -> str | None:
        return self._reference_image_for_step(self.tracker.step)

    def _step_by_order(self, number: int | None):
        return next((step for step in self.tracker.manual.steps
                     if step.order == number), None)

    def _step_mapping_reply(self, question) -> CopilotReply:
        claimed = next(
            (step for step in self.tracker.manual.steps
             if step.id == question.claimed_step_id), None)
        if claimed is None:
            return CopilotReply("확인할 단계를 매뉴얼에서 찾지 못했습니다.",
                                "ASSEMBLY")
        semantic_text = re.sub(
            r"[0-9일이삼사오육칠팔구십]+\s*단계", "", question.text.lower())
        tokens = {
            token for token in re.findall(r"[0-9a-zA-Z가-힣]+", semantic_text)
            if len(token) >= 2
            and token not in {"단계", "작업", "것으로", "알고", "있는데",
                              "왜", "설명해주니", "맞습니까"}
        }
        scored = []
        for step in self.tracker.manual.steps:
            title = re.sub(r"\s+", "", step.title.lower())
            body = re.sub(r"\s+", "", f"{step.title} {step.instruction}".lower())
            score = sum(2 for token in tokens if token in title)
            score += sum(1 for token in tokens if token in body)
            scored.append((score, step))
        best_score, best = max(scored, key=lambda item: item[0])
        if best_score > 0 and best.id != claimed.id:
            return CopilotReply(
                f"로드된 assembly.yaml 기준으로 {claimed.order}단계는 "
                f"‘{claimed.title}’입니다. 말씀하신 작업 내용은 "
                f"{best.order}단계 ‘{best.title}’에 더 가깝습니다. "
                "사용자의 단계 기억에 맞춰 매뉴얼 번호를 바꾸지 않고 "
                "로드된 YAML을 기준으로 안내합니다.", "ASSEMBLY")
        return CopilotReply(
            f"로드된 assembly.yaml에서 {claimed.order}단계는 "
            f"‘{claimed.title}’입니다. {claimed.instruction}", "ASSEMBLY")

    @staticmethod
    def _reference_image_for_step(step) -> str | None:
        if step is None:
            return None
        raster_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        # The final referenced page normally shows the completed state and is
        # the same representative image used by Vision comparison.
        for value in reversed(step.references.images):
            path = Path(value)
            if path.suffix.lower() in raster_extensions and path.is_file():
                return str(path)
        return None

    def after_inspection(self, verdict: str, spoken: str = "") -> str:
        """Carry the verdict into the manual instead of stopping at a remark.

        The operator asked the robot to go and look; making them then announce
        "4단계 완료했어" is asking them to repeat what they just watched.

        CORRECT records the step and reads out the next one. Anything else
        asks - because "잘 안 보인다" and "볼트가 비스듬하다" are not evidence
        of completion, and only the person standing there can settle it. One
        word moves it either way.
        """
        current = self.tracker.step
        if current is None:
            return ""
        following = next((item for item in self.tracker.manual.steps
                          if item.order == current.order + 1), None)

        if str(verdict).upper() == "CORRECT":
            state = self._save_progress(current.order)
            if isinstance(state, CopilotReply):     # tracker refused
                return ""
            after = self.tracker.step
            if after is None:
                return (f"{current.order}단계는 완료로 기록했습니다. "
                        "모든 조립 단계가 끝났습니다.")
            return (f"{current.order}단계를 완료로 기록했습니다. "
                    f"다음은 {after.order}단계, {after.title}입니다. "
                    f"{after.instruction}")

        # Not a clean pass - ask, and leave the manual untouched until told.
        if not hasattr(getattr(self, "db", None), "set_pending"):
            return ""
        self.db.set_pending("COMPLETE_THROUGH_STEP", {
            "step_order": current.order,
            "previous_current_order": current.order,
        })
        tail = (f" 맞으면 {following.order}단계로 넘어가겠습니다."
                if following is not None else "")
        return (f"{current.order}단계는 완료된 상태인가요?{tail}")

    def _web_search(self, text: str) -> CopilotReply:
        if os.getenv("ENABLE_WEB_SEARCH", "true").lower() not in {"1", "true", "yes", "on"}:
            return CopilotReply("웹 검색 기능이 설정에서 꺼져 있습니다.", "SEARCH")
        model = os.getenv("WEB_SEARCH_MODEL", self.conversation.model)
        try:
            search_input = {
                "current_request": text,
                "recent_conversation": list(self.conversation.history),
                "active_manual": self.tracker.manual.manual_id,
                "instruction": (
                    "'내 위치', '그 근처', '아까 말한 곳', '그 메뉴'는 최근 대화에서 "
                    "가장 최근의 명시적 주소·지역·선호를 찾아 해석하세요."
                ),
            }
            response = self.conversation.client.responses.create(
                model=model,
                instructions=("웹 검색 결과를 근거로 한국어로 간결하게 답하세요. 최근 대화에 "
                              "주소나 지역이 있으면 후속 발화의 '내 위치'에 반드시 적용하세요. "
                              "맛집 요청에는 일반적인 음식 종류를 설명하지 말고 실제 주변 "
                              "업소명과 종류, 위치 판단에 유용한 정보를 제시하세요. 최신 정보의 "
                              "기준 날짜를 밝히고, 최근 대화에도 지역이 없을 때만 되물으세요."),
                input=json.dumps(search_input, ensure_ascii=False),
                tools=[{"type": "web_search"}],
                tool_choice="required",
                max_output_tokens=500,
                store=False,
                timeout=max(self.conversation.timeout_seconds, 30),
            )
        except Exception as exc:
            return CopilotReply(f"웹 검색에 실패했습니다: {exc}", "SEARCH")
        answer = getattr(response, "output_text", None) or "웹 검색 결과에서 답을 만들지 못했습니다."
        self.conversation.remember("assistant", answer, "SEARCH")
        return CopilotReply(answer, "SEARCH")

    def handle_general_search(self, text: str) -> CopilotReply:
        """General-current-information lane, separate from assembly handling."""
        self.db.event("GENERAL_SEARCH", {"text": text})
        self.conversation.remember("user", text, "SEARCH")
        return self._web_search(text)

    def _manual_request(self) -> CopilotReply:
        statuses = self.manuals.scan()
        if not statuses:
            return CopilotReply("downloaded_manuals 폴더에 PDF가 없습니다.", "MANUAL")
        new = [item for item in statuses if item.is_new]
        old = [item for item in statuses if not item.is_new]
        if new:
            paths = [item.path for item in new]
            try:
                targets = self.manuals.generate(paths)
            except Exception as exc:
                return CopilotReply(f"새 PDF 매뉴얼 생성에 실패했습니다: {exc}", "MANUAL")
            names = ", ".join(path.name for path in targets)
            suffix = (f" 이전에 처리한 PDF {len(old)}개는 다시 분석하지 않았습니다."
                      if old else "")
            return CopilotReply(f"새 매뉴얼을 생성했습니다: {names}.{suffix}", "MANUAL")
        self.db.set_pending("REGENERATE_PDFS", {
            "paths": [str(item.path) for item in old],
            "hashes": [item.sha256 for item in old],
        })
        products = ", ".join(sorted({item.existing_product or "알 수 없음" for item in old}))
        return CopilotReply(
            f"모든 PDF가 이전에 처리된 파일이며 기존 제품은 {products}입니다. "
            "기존 제품 폴더를 백업하고 다시 생성할까요?", "MANUAL")

    def _robot_skill(self, intent) -> CopilotReply:
        """Semantic side of physical fetch: build the request, ask for
        confirmation, and forward stop. Never moves anything itself."""
        manager = getattr(self, "robot_skills", None)
        if intent.intent == "STOP":
            if manager is not None and manager.cancel_active():
                return CopilotReply(
                    "로봇을 즉시 정지했습니다. 계속하려면 다시 요청해 주세요.",
                    "ROBOT_SKILL")
            return CopilotReply("지금 움직이는 로봇 작업이 없습니다.", "ROBOT_SKILL")
        if manager is None:
            return CopilotReply(
                "로봇 기능이 아직 연결되어 있지 않아 물건을 가져다드릴 수 없습니다.",
                "ROBOT_SKILL")
        if manager.busy:
            snapshot = manager.snapshot()
            return CopilotReply(
                f"지금 '{snapshot.get('query') or '다른 물건'}' 작업을 수행 중입니다. "
                "끝난 뒤 다시 요청해 주세요. 멈추려면 '멈춰'라고 말씀해 주세요.",
                "ROBOT_SKILL")
        # "내 손 위에 있는 거 갖다 놔" - the reverse handover. No
        # confirmation: the operator is holding the thing out, which is
        # already a deliberate act, and asking "정말요?" while they stand
        # there with their arm extended is the wrong kind of careful.
        if intent.intent == "TAKE_THEN_FETCH":
            from robot_skills import FetchOptions, FetchRequest, FetchTarget
            take = FetchRequest(
                target=FetchTarget(query="손 위의 물건"),
                options=FetchOptions(dry_run=False,
                                     require_confirmation=False,
                                     timeout_seconds=90.0),
                skill="take_from_hand")
            accepted, message = manager.submit(take)
            if not accepted:
                return CopilotReply(message, "ROBOT_SKILL")
            # If they named the second tool ("...놓고 저 드라이버 가져와"),
            # use the name; otherwise the finger is the answer.
            from robot_skills import resolve_object_class
            named = resolve_object_class(intent.text)
            self.queued_robot_request = FetchRequest(
                target=FetchTarget(query="가리키신 물건" if named is None
                                   else named,
                                   class_name=named,
                                   attributes={"utterance": intent.text}),
                options=FetchOptions(dry_run=False,
                                     require_confirmation=False),
                destination_type="user_handover")
            self.queued_robot_reply = "이제 가리키신 물건을 가져다드릴게요."
            return CopilotReply(
                "먼저 손 위의 물건을 받아서 빈 곳에 내려놓고, 그다음 "
                "가리키신 물건을 가져다드리겠습니다. 손을 펴고 계세요.",
                "ROBOT_SKILL")
        if intent.intent == "TAKE_FROM_HAND":
            from robot_skills import FetchOptions, FetchRequest, FetchTarget
            request = FetchRequest(
                target=FetchTarget(query="손 위의 물건"),
                options=FetchOptions(dry_run=False,
                                     require_confirmation=False,
                                     timeout_seconds=90.0),
                skill="take_from_hand")
            accepted, message = manager.submit(request)
            return CopilotReply(
                "손 위의 물건을 받아서 빈 곳에 내려놓겠습니다. 손을 펴고 "
                "가만히 계세요." if accepted else message, "ROBOT_SKILL")
        # "저거 가져와" while pointing: the finger already chose the object,
        # so the app's current target is the answer - no class name needed.
        if intent.intent == "FETCH_POINTED":
            from robot_skills import FetchOptions, FetchRequest, FetchTarget
            self.db.set_pending("ROBOT_FETCH_OBJECT", {
                "query": intent.text, "class_name": None,
                "label": "가리키신 물건", "destination": "user_handover"},
                ttl_s=120)
            return CopilotReply(
                "가리키신 물건을 손 위에 올려드릴까요? 진행하려면 '응', "
                "아니면 '취소'라고 말씀해 주세요.", "ROBOT_SKILL")
        # "이거 맞아?" - go and photograph what the finger is pointing at.
        # No confirmation round: the arm only travels to look and comes back,
        # nothing is gripped or moved, and making someone say "응" first would
        # break the rhythm of asking about your own work while doing it.
        if intent.intent == "INSPECT_STEP":
            from robot_skills import FetchOptions, FetchRequest, FetchTarget
            # "더 가까이서 봐줘" changes only how far the camera stands off;
            # the spot and the angle stay whatever the finger chose.
            attributes = {}
            if self.router.INSPECT_CLOSER.search(intent.text):
                attributes["standoff_mm"] = 260.0
            elif self.router.INSPECT_FARTHER.search(intent.text):
                attributes["standoff_mm"] = 560.0
            request = FetchRequest(
                target=FetchTarget(query=intent.text, attributes=attributes),
                options=FetchOptions(dry_run=False,
                                     require_confirmation=False,
                                     timeout_seconds=60.0),
                skill="inspect_step")
            accepted, message = manager.submit(request)
            if not accepted:
                return CopilotReply(message, "ROBOT_SKILL")
            how = ("조금 더 가까이서 " if attributes.get("standoff_mm", 0) < 400
                   else "조금 더 넓게 " if attributes else "")
            return CopilotReply(
                f"가리키신 곳을 {how}보고 판단해 볼게요. 손을 잠깐 치워 주세요.",
                "ROBOT_SKILL")
        # Quick operator-style commands: no confirmation, same exclusive
        # manager (they can never overlap a running fetch).
        if intent.intent in ("HOME", "GRIPPER_OPEN", "GRIPPER_CLOSE"):
            from robot_skills import FetchOptions, FetchRequest, FetchTarget
            skill_name = {"HOME": "robot_home",
                          "GRIPPER_OPEN": "gripper_open",
                          "GRIPPER_CLOSE": "gripper_close"}[intent.intent]
            label = {"HOME": "홈 복귀", "GRIPPER_OPEN": "그리퍼 열기",
                     "GRIPPER_CLOSE": "그리퍼 닫기"}[intent.intent]
            request = FetchRequest(
                target=FetchTarget(query=label),
                options=FetchOptions(dry_run=False,
                                     require_confirmation=False,
                                     timeout_seconds=40.0),
                skill=skill_name)
            accepted, message = manager.submit(request)
            return CopilotReply(
                f"{label}를 실행합니다." if accepted else message,
                "ROBOT_SKILL")
        from robot_skills import resolve_object_class
        class_name = resolve_object_class(intent.text)
        if class_name is None:
            return CopilotReply(
                "어떤 공구인지 정확히 알려주세요. 지금은 해머, 드라이버, 렌치, "
                "펜치, 드릴을 다룰 수 있습니다.", "ROBOT_SKILL")
        korean = {"hammer": "해머", "screwdriver": "드라이버",
                  "wrench": "렌치", "pliers": "펜치", "drill": "드릴"}
        tidy = intent.intent == "TIDY_OBJECT"
        # Confirmation-first: the pending row carries the full semantic
        # request, and a later "응" applies to THIS request only.
        self.db.set_pending("ROBOT_FETCH_OBJECT", {
            "query": intent.text, "class_name": class_name,
            "label": korean.get(class_name, class_name),
            "destination": "fixed_storage" if tidy else "user_handover"},
            ttl_s=120)
        if tidy:
            return CopilotReply(
                f"{korean.get(class_name, class_name)}를 집어서 보관 위치에 "
                "정리해 둘까요? 진행하려면 '응', 아니면 '취소'라고 말씀해 주세요.",
                "ROBOT_SKILL")
        return CopilotReply(
            f"{korean.get(class_name, class_name)}를 가져다드릴까요? 손 위에 "
            "올려드립니다. 진행하려면 '응', 아니면 '취소'라고 말씀해 주세요.",
            "ROBOT_SKILL")

    def _confirmation(self, decision: str) -> CopilotReply:
        pending = self.db.resolve_pending("ACCEPTED" if decision == "ACCEPT" else "REJECTED")
        if pending is None:
            return CopilotReply("현재 확인을 기다리는 작업이 없습니다.", "CONFIRMATION")
        if pending["operation_type"] == "ROBOT_FETCH_OBJECT":
            if decision == "REJECT":
                return CopilotReply("알겠습니다. 로봇 작업을 취소했습니다.",
                                    "ROBOT_SKILL")
            manager = getattr(self, "robot_skills", None)
            if manager is None:
                return CopilotReply(
                    "로봇 기능이 연결되어 있지 않아 실행할 수 없습니다.",
                    "ROBOT_SKILL")
            from robot_skills import FetchOptions, FetchRequest, FetchTarget
            payload = pending["payload"]
            destination = payload.get("destination", "user_handover")
            request = FetchRequest(
                target=FetchTarget(
                    query=payload.get("label") or payload["query"],
                    class_name=payload["class_name"],
                    attributes={"utterance": payload["query"]}),
                options=FetchOptions(dry_run=False,
                                     require_confirmation=True),
                destination_type=destination)
            accepted, message = manager.submit(request)
            if not accepted:
                return CopilotReply(message, "ROBOT_SKILL")
            if destination == "fixed_storage":
                return CopilotReply(
                    "네, 정리하겠습니다. 집어서 보관 위치에 놓아둘게요. "
                    "언제든 '멈춰'라고 말씀하시면 즉시 정지합니다.",
                    "ROBOT_SKILL")
            return CopilotReply(
                "네, 가져오겠습니다. 사람 팔은 피해서 접근하고, 손을 내밀고 "
                "계시면 그 위에 올려드릴게요. 언제든 '멈춰'라고 말씀하시면 즉시 "
                "정지합니다.", "ROBOT_SKILL")
        if pending["operation_type"] == "SELECT_ASSEMBLY_STEP":
            current = self.tracker.step
            if decision == "REJECT":
                return CopilotReply(
                    f"알겠습니다. 현재 {current.order}단계를 그대로 유지할게요.",
                    "CONFIRMATION")
            number = int(pending["payload"]["step_order"])
            self.tracker.select_step_number(
                number, source=pending["payload"].get("source"))
            self.work_session_active = True
            self._persist_assembly_state()
            step = self.tracker.step
            return CopilotReply(
                f"좋아요. 현재 작업 상태를 {number}단계로 바꿨어요. "
                f"{step.instruction}", "ASSEMBLY", action="DISPLAY_ON",
                reference_image=self._current_reference_image(),
                reference_label=f"{number}단계 참고 사진")
        if pending["operation_type"] == "VISUAL_CLARIFICATION":
            payload = pending["payload"]
            number = int(payload["step_order"])
            current = dict(payload["current"])
            answer = decision == "ACCEPT"
            camera_result = str(current.get("camera_result", "unknown"))
            evidence = {
                "step_order": number,
                "check_id": current["id"],
                "description": current["description"],
                "answer": answer,
                "camera_result": camera_result,
            }
            self.db.event("USER_PROVIDED_EVIDENCE", evidence)
            answers = [*payload.get("answers", []), evidence]
            if answer and camera_result == "false":
                self.db.event("VISUAL_EVIDENCE_CONFLICT", evidence)
                self._save_visual_assistance("CONFLICT", number, answers)
                return CopilotReply(
                    "사용자 답변과 카메라 관측이 서로 다릅니다. 임의로 단계를 "
                    "정하지 않겠습니다. 해당 부위가 보이도록 각도를 바꿔 다시 확인해 주세요.",
                    "ASSEMBLY")
            if not answer:
                self._save_visual_assistance("UNCERTAIN", number, answers)
                return CopilotReply(
                    f"알겠습니다. 답변상 {number}단계의 필수 조건이 아직 충족되지 "
                    "않았습니다. 해당 조건을 완료하거나 다른 각도에서 다시 확인해 주세요.",
                    "ASSEMBLY")
            remaining = list(payload.get("remaining", []))
            if remaining:
                next_check = remaining.pop(0)
                self._visual_question_history.setdefault(number, set()).add(
                    str(next_check["id"]))
                self.db.set_pending("VISUAL_CLARIFICATION", {
                    "step_order": number,
                    "current": next_check,
                    "remaining": remaining,
                    "answers": answers,
                    "asked_ids": [*payload.get("asked_ids", []), current["id"]],
                    "resolution": payload.get("resolution", "SELECT_STEP"),
                })
                return CopilotReply(
                    f"답변을 기록했습니다. 다음 한 가지만 확인할게요. "
                    f"{next_check['description']} 상태가 맞나요?", "ASSEMBLY")
            self._save_visual_assistance("USER_ASSISTED", number, answers)
            resolution = str(payload.get("resolution", "SELECT_STEP"))
            operation = ("CONFIRM_USER_ASSISTED_COMPLETION"
                         if resolution == "COMPLETE_STEP"
                         else "SELECT_ASSEMBLY_STEP")
            self.db.set_pending(operation, {
                "step_order": number, "source": "USER_ASSISTED"})
            next_step = self._step_by_order(number + 1)
            if resolution == "COMPLETE_STEP":
                next_label = (f"{next_step.order}단계로 넘어갈까요?"
                              if next_step is not None else "전체 작업 완료로 기록할까요?")
                return CopilotReply(
                    f"카메라 관측과 {len(answers)}개의 사용자 답변을 함께 보면 "
                    f"{number}단계 완료 상태로 판단할 수 있습니다. {next_label}",
                    "ASSEMBLY")
            return CopilotReply(
                f"카메라 관측과 {len(answers)}개의 사용자 답변을 함께 보면 현재 모습은 "
                f"{number}단계일 가능성이 높습니다. "
                f"카메라 검증 완료와는 별도로, 현재 작업 위치를 {number}단계로 바꿀까요?",
                "ASSEMBLY")
        if pending["operation_type"] == "CONFIRM_USER_ASSISTED_COMPLETION":
            number = int(pending["payload"]["step_order"])
            if decision == "REJECT":
                return CopilotReply(
                    f"알겠습니다. {number}단계 완료로 기록하지 않고 현재 위치를 유지합니다.",
                    "ASSEMBLY")
            self.tracker.complete_through(
                number, source="USER_ASSISTED")
            self.work_session_active = True
            self._persist_assembly_state()
            target = self.tracker.step
            if target is None:
                return CopilotReply(
                    f"사용자 보조 확인 기준으로 {number}단계 완료를 기록했습니다. "
                    "모든 조립 단계가 완료되었습니다.", "ASSEMBLY", action="DISPLAY_ON")
            reference = self._reference_image_for_step(target)
            return CopilotReply(
                f"사용자 보조 확인 기준으로 {number}단계 완료를 기록했습니다. "
                f"다음은 {target.order}단계, {target.title}입니다. {target.instruction}",
                "ASSEMBLY", action="DISPLAY_ON", reference_image=reference,
                reference_label=(f"{target.order}단계 참고 사진" if reference else None))
        if pending["operation_type"] == "COMPLETE_THROUGH_STEP":
            number = int(pending["payload"]["step_order"])
            if decision == "REJECT":
                return CopilotReply(
                    f"알겠습니다. {number}단계까지 완료로 저장하지 않습니다.",
                    "ASSEMBLY")
            state = self._save_progress(number)
            if isinstance(state, CopilotReply):
                return state
            self.work_session_active = True
            target = self.tracker.step
            if target is None:
                return CopilotReply(
                    f"사용자 확인 기준으로 {number}단계까지 완료로 저장했습니다. "
                    "모든 조립 단계가 완료되었습니다.", "ASSEMBLY",
                    action="DISPLAY_ON")
            reference = self._reference_image_for_step(target)
            return CopilotReply(
                f"사용자 확인 기준으로 {number}단계까지 완료로 저장했습니다. "
                f"다음은 {target.order}단계, {target.title}입니다. "
                f"{target.instruction}", "ASSEMBLY", action="DISPLAY_ON",
                reference_image=reference,
                reference_label=(f"{target.order}단계 참고 사진" if reference else None))
        if decision == "REJECT":
            return CopilotReply("기존 매뉴얼을 유지하고 재생성을 취소했습니다.", "CONFIRMATION")
        payload = pending["payload"]
        paths = [Path(value) for value in payload.get("paths", [])]
        current_hashes = [item.sha256 for item in self.manuals.scan() if item.path in paths]
        if sorted(current_hashes) != sorted(payload.get("hashes", [])):
            return CopilotReply("확인 대기 중 PDF 내용이 변경되어 재생성을 취소했습니다.",
                                "CONFIRMATION")
        try:
            targets = self.manuals.generate(paths, overwrite=True)
        except Exception as exc:
            return CopilotReply(f"매뉴얼 재생성에 실패했습니다: {exc}", "MANUAL")
        return CopilotReply(
            "기존 폴더를 백업하고 다시 생성했습니다: "
            + ", ".join(path.name for path in targets), "MANUAL")

    def _assembly(self, text: str, frames, timestamp_ms: int) -> CopilotReply:
        self.work_session_active = True
        state = self.tracker.snapshot()
        question = self.question_router.route(text)
        db = getattr(self, "db", None)
        pending = db.pending() if db is not None else None
        if (question.intent == "NEXT_STEP" and pending is not None
                and pending["operation_type"] == "VISUAL_CLARIFICATION"):
            current = pending["payload"]["current"]
            return CopilotReply(
                "현재 단계 보조 판정이 아직 끝나지 않았습니다. 먼저 이 항목을 "
                f"확인해 주세요. {current['description']} 상태가 맞나요?", "ASSEMBLY")
        if question.intent in {"EXPLAIN_TARGET_STEP", "SHOW_TARGET_REFERENCE",
                               "EXPLAIN_AND_SHOW_TARGET"}:
            target = next(
                (item for item in self.tracker.manual.steps
                 if item.id == question.claimed_step_id), None)
            if target is None:
                return CopilotReply("요청하신 단계를 매뉴얼에서 찾지 못했어요.", "ASSEMBLY")
            current = self.tracker.step
            suffix = (f" 지금 진행 상태는 {current.order}단계로 그대로 둘게요."
                      if current is not None and current.id != target.id else "")
            reference = (self._reference_image_for_step(target)
                         if question.intent in {"SHOW_TARGET_REFERENCE",
                                                "EXPLAIN_AND_SHOW_TARGET"} else None)
            return CopilotReply(
                f"{target.order}단계는 {target.title} 작업이에요. "
                f"{target.instruction}{suffix}", "ASSEMBLY",
                action="DISPLAY_ON" if reference else None,
                reference_image=reference,
                reference_label=(f"{target.order}단계 참고 사진 · {Path(reference).name}"
                                 if reference else None))
        if question.intent == "CHALLENGE_STEP_MAPPING":
            return self._step_mapping_reply(question)
        if question.intent == "RETURN_TO_STEP":
            target = next(
                (item for item in self.tracker.manual.steps
                 if item.id == question.claimed_step_id), None)
            if target is None:
                return CopilotReply("돌아갈 단계를 매뉴얼에서 찾지 못했습니다.",
                                    "ASSEMBLY")
            previous = self.tracker.step
            try:
                self.tracker.reopen_step(target.order)
            except ValueError as exc:
                return CopilotReply(str(exc), "ASSEMBLY")
            self.work_session_active = True
            self._persist_assembly_state()
            reference = self._reference_image_for_step(target)
            previous_label = (f"{previous.order}단계" if previous is not None else "완료 상태")
            return CopilotReply(
                f"현재 작업 위치를 {previous_label}에서 {target.order}단계로 "
                f"되돌렸습니다. {target.order}단계 이후의 완료 기록은 "
                f"취소했습니다. {target.instruction}", "ASSEMBLY", action="DISPLAY_ON",
                reference_image=reference,
                reference_label=(f"{target.order}단계 참고 사진" if reference else None))
        if question.intent == "SELECT_STEP":
            target = next(
                (item for item in self.tracker.manual.steps
                 if item.id == question.claimed_step_id), None)
            if target is None:
                return CopilotReply("변경할 단계를 매뉴얼에서 찾지 못했어요.", "ASSEMBLY")
            current = self.tracker.step
            if current is not None and current.id == target.id:
                return CopilotReply(
                    f"이미 {target.order}단계를 진행 중이에요. {target.instruction}",
                    "ASSEMBLY")
            self.db.set_pending("SELECT_ASSEMBLY_STEP", {"step_order": target.order})
            current_label = (f"{current.order}단계" if current is not None else "완료 상태")
            return CopilotReply(
                f"현재 기록은 {current_label}예요. 완료 기록은 건드리지 않고 "
                f"작업 위치만 {target.order}단계로 바꿀까요?", "ASSEMBLY")
        if question.intent == "COMPLETE_STEP":
            current = self.tracker.step
            if (question.claimed_step_id is not None and current is not None
                    and question.claimed_step_id != current.id):
                claimed = next(
                    (item for item in self.tracker.manual.steps
                     if item.id == question.claimed_step_id), None)
                label = (f"{claimed.order}단계" if claimed is not None
                         else question.claimed_step_id)
                if claimed is not None and hasattr(getattr(self, "db", None), "set_pending"):
                    self.db.set_pending("COMPLETE_THROUGH_STEP", {
                        "step_order": claimed.order,
                        "previous_current_order": current.order,
                    })
                return CopilotReply(
                    f"현재 기록은 {current.order}단계입니다. "
                    f"중간 단계도 사용자 확인 완료로 포함해 {label}까지 "
                    "완료로 저장할까요?", "ASSEMBLY")
            before = self.tracker.step.id if self.tracker.step else None
            self.tracker.confirm_current_step()
            after = self.tracker.step
            answer = (f"{before} 완료를 기록했습니다. 다음은 {after.title}입니다."
                      if after else "모든 조립 단계를 완료했습니다.")
            return CopilotReply(answer, "ASSEMBLY")
        if question.intent in {"IDENTIFY_STEP", "CHECK_PROGRESS", "CHECK_CLAIMED_STEP"}:
            if not frames:
                return CopilotReply("최근 RealSense 프레임이 없어 작업 상태를 판단할 수 없습니다.",
                                    "ASSEMBLY")
            observation = AssemblyObservation(timestamp_ms, visibility="VISIBLE")
            steps = self.retriever.retrieve(question, state)
            raw = self.assessor.assess(question, state, observation, steps, frames)
            # A correctness question is scoped to the selected/current step.
            # Even if the VLM recognizes an earlier-looking state, report that
            # as missing requirements of the requested step instead of changing
            # the answer into an unsolicited "most similar step" search.
            if (question.intent in {"CHECK_PROGRESS", "CHECK_CLAIMED_STEP"}
                    and len(steps) == 1
                    and raw.observed_step_id != steps[0].id):
                raw = raw.model_copy(update={
                    "assessment": "UNCERTAIN",
                    "observed_step_id": steps[0].id,
                    "confidence": 0.0,
                    "instruction": (
                        f"{steps[0].order}단계 기준으로 확인 중이며, "
                        "현재 화면에서 필수 조건을 모두 확인하지 못했습니다."),
                    "checks": [],
                })
            if os.getenv("DUME_VERBOSE_LOGS", "false").lower() in {"1", "true", "yes", "on"}:
                print(
                    f"[Vision 판정] model={self.assessor.model}, "
                    f"assessment={raw.assessment}, candidate={raw.observed_step_id}, "
                    f"confidence={raw.confidence:.3f}, visible={raw.visible}, "
                    f"needs_better_view={raw.needs_better_view}", flush=True)
            checked = self.validator.validate(raw, state, observation)
            return self._apply_assessment(checked, question=question)
        local = self.local_copilot.answer(text, state)
        return CopilotReply(local.text, "ASSEMBLY")

    def _assessment_message(self, checked, question=None) -> str:
        if checked.observed_step_id is None:
            return checked.instruction
        step = next(
            (item for item in self.tracker.manual.steps
             if item.id == checked.observed_step_id), None)
        if step is None:
            return checked.instruction
        if question is not None and question.intent == "CHECK_CLAIMED_STEP":
            claimed = next(
                (item for item in self.tracker.manual.steps
                 if item.id == question.claimed_step_id), step)
            label = f"{claimed.order}단계 기준 검수"
        elif question is not None and question.intent == "CHECK_PROGRESS":
            current = self.tracker.step
            label = (f"현재 {current.order}단계 기준 검수"
                     if current is not None else "현재 작업 기준 검수")
        else:
            label = ("가장 유사한 단계" if not checked.accepted
                     else "판정 단계")
        return (f"{label}: {step.order}단계, {step.title} "
                f"(신뢰도 {checked.confidence * 100:.0f}%). "
                f"{checked.instruction}")

    def _apply_assessment(self, checked, question=None) -> CopilotReply:
        message = self._assessment_message(checked, question=question)
        if hasattr(self.db, "event"):
            self.db.event("ASSEMBLY_OBSERVATION", {
                "assessment": checked.assessment,
                "observed_step_id": checked.observed_step_id,
                "accepted": checked.accepted,
                "validation_notes": list(checked.validation_notes),
            })
        if not checked.accepted or checked.observed_step_id is None:
            self._reset_transition_consensus()
            clarification = self._visual_clarification_question(checked, question=question)
            if clarification is not None:
                return CopilotReply(message + " " + clarification, "ASSEMBLY")
            return CopilotReply(message, "ASSEMBLY")
        observed = next(
            (item for item in self.tracker.manual.steps
             if item.id == checked.observed_step_id), None)
        if observed is None:
            self._reset_transition_consensus()
            return CopilotReply(message, "ASSEMBLY")
        if checked.assessment not in {"CORRECT", "STEP_COMPLETE"}:
            self._reset_transition_consensus()
            return CopilotReply(message + " 현재 추적 단계는 변경하지 않습니다.", "ASSEMBLY")
        if (question is not None
                and question.intent in {"CHECK_PROGRESS", "CHECK_CLAIMED_STEP"}
                and not re.search(r"완료|끝|\b다\s*했", question.text, re.I)):
            self._reset_transition_consensus()
            return CopilotReply(
                message + " 검수 요청이므로 작업 단계 기록은 변경하지 않습니다.",
                "ASSEMBLY")
        current = self.tracker.step
        # CORRECT for the already tracked step is a status report, not a state
        # transition. It should never manufacture progress.
        if checked.assessment == "CORRECT" and current is not None and current.id == observed.id:
            self._reset_transition_consensus()
            return CopilotReply(message + " 현재 추적 단계를 유지합니다.", "ASSEMBLY")
        hits, required = self._record_transition_vote(
            checked.assessment, observed.id)
        if hits < required:
            return CopilotReply(
                f"{message} 상태 변경 후보로 기록했습니다({hits}/{required}회 일치). "
                "다음 관측에서도 같을 때 단계에 반영합니다.", "ASSEMBLY")
        self._reset_transition_consensus()
        if checked.assessment == "STEP_COMPLETE":
            self.tracker.complete_through(observed.order, verified=True)
            transition = f"{observed.order}단계 완료를 확인했습니다."
        else:
            self.tracker.select_step_number(observed.order)
            transition = f"{observed.order}단계 작업 상태로 맞췄습니다."
        self._persist_assembly_state()
        target = self.tracker.step
        if target is None:
            return CopilotReply(
                f"{message} {transition} 모든 조립 단계가 완료되었습니다.",
                "ASSEMBLY", action="DISPLAY_ON")
        return CopilotReply(
            f"{message} {transition} 이제 {target.order}단계 작업을 시작합니다. "
            f"{target.instruction}",
            "ASSEMBLY", action="DISPLAY_ON",
            reference_image=self._current_reference_image())

    def _visual_clarification_question(self, checked, question=None) -> str | None:
        if not getattr(self, "interactive_visual_clarification", False):
            return None
        step = next(
            (item for item in self.tracker.manual.steps
             if item.id == checked.observed_step_id), None)
        if step is None:
            return None
        results = {item.check_id: item.result for item in checked.checks}
        missing = [
            item for item in step.visual_checks
            if item.get("kind", "required") == "required"
            and bool(item.get("required", True))
            and results.get(str(item.get("id"))) != "true"
        ]
        if not missing:
            return None
        missing.sort(key=self._visual_question_priority)
        questions = [{
            "id": str(item.get("id")),
            "description": str(item.get("description") or item.get("id")),
            "camera_result": results.get(str(item.get("id")), "unknown"),
        } for item in missing]
        check, remaining = questions[0], questions[1:]
        request_text = str(getattr(question, "text", ""))
        completion_requested = bool(re.search(
            r"다음|완료|끝|걸었|연결했|고정했|조였|체결했|장착했|설치했|확인했",
            request_text, re.I))
        self.db.set_pending("VISUAL_CLARIFICATION", {
            "step_order": step.order,
            "current": check,
            "remaining": remaining,
            "answers": [],
            "asked_ids": [],
            "resolution": "COMPLETE_STEP" if completion_requested else "SELECT_STEP",
        })
        self.db.event("VISUAL_CLARIFICATION_ASKED", {
            "step_order": step.order, "check_id": check["id"]})
        return f"한 가지만 확인할게요. {check['description']} 상태가 맞나요?"

    @staticmethod
    def _visual_question_priority(check: dict) -> tuple[int, int, str]:
        description = str(check.get("description", ""))
        safety_words = ("안전", "고정", "체결", "장력", "빠짐", "풀림")
        safety = 0 if any(word in description for word in safety_words) else 1
        relation = 0 if "relation" in str(check.get("id", "")) else 1
        return safety, relation, str(check.get("id", ""))

    def _save_visual_assistance(self, status: str, step_order: int,
                                answers: list[dict]) -> None:
        self.db.save_runtime_state(
            f"visual_assistance:{self.tracker.manual.manual_id}",
            {"status": status, "step_order": step_order, "answers": answers})

    def _record_transition_vote(self, assessment: str, step_id: str) -> tuple[int, int]:
        candidate = (assessment, step_id)
        if getattr(self, "_transition_candidate", None) == candidate:
            self._transition_hits = getattr(self, "_transition_hits", 0) + 1
        else:
            self._transition_candidate = candidate
            self._transition_hits = 1
        return self._transition_hits, getattr(self, "transition_consensus_required", 2)

    def _reset_transition_consensus(self) -> None:
        self._transition_candidate = None
        self._transition_hits = 0

    def _persist_assembly_state(self) -> None:
        state = self.tracker.snapshot()
        self.db.save_runtime_state(
            f"assembly:{self.tracker.manual.manual_id}",
            {"current_step_id": state.current_step_id,
             "completed_steps": list(state.completed_steps),
             "user_confirmed_steps": list(state.user_confirmed_steps),
             "verified_completed_steps": list(state.verified_completed_steps),
             "progress_update_source": state.progress_update_source})

    def _describe(self, text: str, frame) -> CopilotReply:
        reply = self.conversation.respond(
            text, route=ConversationRoute.DESCRIBE,
            context={"active_manual": self.tracker.manual.manual_id}, image=frame)
        return CopilotReply(reply, "VISION")

    def _chat(self, text: str, frames, current_frame, timestamp_ms: int) -> CopilotReply:
        state = self.tracker.snapshot()
        step = self.tracker.step
        decision = self.conversation.route(
            text,
            capabilities=("assembly_start_or_resume", "assembly_identify_step",
                          "assembly_check_progress",
                          "assembly_next_instruction", "assembly_show_reference",
                          "manual_generate_from_pdf", "describe_scene", "web_search",
                          "speak"),
            context={"active_manual": self.tracker.manual.manual_id,
                     "active_product": self.tracker.manual.product,
                     "current_step": state.current_step_id,
                     "current_step_title": step.title if step else None,
                     "completed_steps": list(state.completed_steps),
                     "assembly_status": state.status})
        capabilities = {item.capability for item in decision.steps}
        if decision.capability:
            capabilities.add(decision.capability)
        if (decision.route == ConversationRoute.CLARIFY
                or "ask_clarification" in capabilities
                or (capabilities and decision.confidence < 0.6)):
            question = self._semantic_clarification(decision, capabilities)
            self.conversation.remember("assistant", question, "CLARIFY")
            return CopilotReply(question, "CONVERSATION")
        if "assembly_start_or_resume" in capabilities:
            return self._start_work(text)
        if capabilities & {"assembly_identify_step", "assembly_check_progress"}:
            return self._assembly(text, frames, timestamp_ms)
        if "assembly_next_instruction" in capabilities:
            state = self.tracker.snapshot()
            local = self.local_copilot.answer("다음 작업을 알려줘", state)
            return CopilotReply(local.text, "ASSEMBLY")
        if "assembly_show_reference" in capabilities:
            reference = self._current_reference_image()
            message = (f"{self.tracker.step.title} 참고 사진을 표시합니다."
                       if reference and self.tracker.step else
                       "현재 단계에 표시할 참고 사진이 없습니다.")
            return CopilotReply(message, "ASSEMBLY", action="DISPLAY_ON",
                                reference_image=reference)
        if "describe_scene" in capabilities:
            return self._describe(text, current_frame)
        if "manual_generate_from_pdf" in capabilities:
            return self._manual_request()
        if "web_search" in capabilities:
            return self._web_search(text)
        if decision.clarification_question:
            reply = decision.clarification_question
        elif decision.reply:
            reply = decision.reply
        else:
            reply = self.conversation.respond(
                text, route=decision.route,
                context={"available_steps": [step.capability for step in decision.steps]})
        return CopilotReply(reply, "CONVERSATION")

    @staticmethod
    def _semantic_clarification(decision, capabilities: set[str]) -> str:
        if decision.clarification_question:
            return decision.clarification_question
        assembly = capabilities & {
            "assembly_identify_step", "assembly_check_progress",
            "assembly_next_instruction", "assembly_start_or_resume",
        }
        if {"assembly_identify_step", "assembly_check_progress"} <= assembly:
            return "현재 단계를 찾을까요, 아니면 지금 조립이 올바른지 검사할까요?"
        if "assembly_start_or_resume" in assembly:
            return "저장된 단계부터 조립 작업을 이어서 시작할까요?"
        if "assembly_next_instruction" in assembly:
            return "현재 단계 다음에 할 작업을 안내할까요?"
        if assembly:
            return "현재 단계 확인과 조립 상태 검사 중 어느 것을 원하시나요?"
        return "말씀하신 내용으로 무엇을 확인하거나 실행하면 될까요?"

    def close(self) -> None:
        self.conversation.close()
        self.db.close()
