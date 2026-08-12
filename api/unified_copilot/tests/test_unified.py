from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unified_copilot.intents import UnifiedIntentRouter
from unified_copilot.app import _assembly_context, _request_error_reply, _request_label
from unified_copilot.realtime_voice import RealtimeToolRequest, TOOL
from unified_copilot.turn_manager import CentralTurnManager, TaskRegistry
from unified_copilot.engine import UnifiedCopilotEngine
from unified_copilot.manual_service import ManualGenerationService, sha256
from unified_copilot.memory import CopilotStateDB
from assembly_copilot.manual import load_manual
from assembly_copilot.state_tracker import AssemblyStateTracker
from assembly_copilot.question_router import AssemblyQuestionRouter
from assembly_copilot.manual_retriever import ManualRetriever
from assembly_copilot.assessment_models import ValidatedAssessment
from VoiceProcessing.conversation_router import ConversationDecision, ConversationRoute


class UnifiedIntentTests(unittest.TestCase):
    def test_realtime_forwards_every_turn_to_central_manager(self):
        self.assertEqual(TOOL["name"], "dispatch_user_turn")
        self.assertEqual(TOOL["parameters"]["required"], ["utterance"])
        request = RealtimeToolRequest("call-1", "가체결했어")
        self.assertEqual(request.tool_name, "dispatch_user_turn")
        self.assertEqual(request.utterance, "가체결했어")

    def test_realtime_pcm_rate_matches_playback_rate(self):
        source = (Path(__file__).parents[1] / "realtime_voice.py").read_text()
        self.assertIn('"format": {"type": "audio/pcm", "rate": 24_000}', source)
        self.assertIn('kind == "response.output_audio.done"', source)
        response_done = source.split('elif kind == "response.done":', 1)[1]
        self.assertIn("server_audio_done.set()", response_done)
        self.assertIn("playback_active.clear()", response_done)
        capture_loop = source.split("def capture_loop()", 1)[1].split(
            "def playback_loop()", 1)[0]
        self.assertNotIn("if playback_active.is_set()", capture_loop)
        self.assertIn('"tool_choice": "required"', source)
        self.assertIn('"tools": [TOOL]', source)

    def test_turn_manager_carries_step_focus_into_followup(self):
        manager = CentralTurnManager.__new__(CentralTurnManager)
        manager.focus_step = None
        manager.pending_proposal = None
        manager.focus_domain = "conversation"
        first = manager._resolve_context("1단계 말고 6단계 설명해 줘")
        second = manager._resolve_context("이미지도 보여 줘")
        self.assertEqual(manager.focus_step, 6)
        self.assertEqual(first, "1단계 말고 6단계 설명해 줘")
        self.assertEqual(second, "6단계 이미지도 보여 줘")

    def test_korean_step_and_loaded_manual_view_are_not_general_chat_or_generation(self):
        router = UnifiedIntentRouter()
        step = router.route("육단계 작업 설명해 줘")
        view = router.route("현재 매뉴얼과 6단계 페이지 보여 줘")
        page = router.route("5단계 페이지 보여 줘")
        page_particle = router.route("그리고 4단계 페이지도 띄워 줘")
        step_screen = router.route("3단계 화면 보여 줘")
        reference_screen = router.route("3단계 참고 화면 보여 줘")
        generate = router.route("새 PDF 매뉴얼 만들어 줘")
        self.assertEqual((step.domain, step.intent), ("ASSEMBLY", "TARGET_STEP_INFO"))
        self.assertEqual((view.domain, view.intent), ("ASSEMBLY", "SHOW_LOADED_MANUAL"))
        self.assertEqual((page.domain, page.intent), ("ASSEMBLY", "SHOW_LOADED_MANUAL"))
        self.assertEqual((page_particle.domain, page_particle.intent),
                         ("ASSEMBLY", "SHOW_LOADED_MANUAL"))
        self.assertEqual((step_screen.domain, step_screen.intent),
                         ("ASSEMBLY", "SHOW_LOADED_MANUAL"))
        self.assertEqual((reference_screen.domain, reference_screen.intent),
                         ("ASSEMBLY", "SHOW_LOADED_MANUAL"))
        self.assertEqual((generate.domain, generate.intent), ("MANUAL", "GENERATE_OR_SCAN"))

    def test_task_registry_cancelled_task_cannot_look_active(self):
        registry = TaskRegistry()
        item = registry.add(1, 1, "VISION", "지금 상태 확인해 줘")
        registry.set_status(item.task_id, "RUNNING")
        self.assertEqual(registry.cancel_active(), 1)
        self.assertEqual(registry.status(item.task_id), "CANCELLED")
        self.assertEqual(registry.active(), [])

    def test_compound_page_and_search_request_is_split_into_real_actions(self):
        manager = CentralTurnManager.__new__(CentralTurnManager)
        manager.focus_step = 6
        manager.focus_domain = "assembly"
        manager.pending_proposal = None
        text = manager._resolve_context(
            "3단계 페이지 띄워주면서 금천구 근처 맛집 좀 알려줘")
        self.assertEqual(
            manager._independent_parts(text),
            ["3단계 페이지 보여 줘", "금천구 근처 맛집 좀 알려줘"])
        corrected = manager._resolve_context(
            "지금 6단계 페이지 띄우고 있어 3단계로 수정해줘 그리고 프린터 오늘 날씨도 알려줘")
        self.assertEqual(
            manager._independent_parts(corrected),
            ["지금 6단계 페이지 띄우고 있어 3단계로 수정해줘",
             "오늘 날씨도 알려줘"])

    def test_task_status_question_is_deterministic(self):
        manager = CentralTurnManager.__new__(CentralTurnManager)
        self.assertIsNotNone(manager._STATUS.search("지금 처리 중인 요청 알려줘"))

    def test_loaded_manual_is_not_an_active_work_session(self):
        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(Path(__file__).parents[2] / "assembly_manuals/template/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="inactive-work")
        engine.work_session_active = False
        self.assertFalse(_assembly_context(engine)["work_active"])

    def test_response_label_preserves_original_question(self):
        label = _request_label(7, "지금 몇 단계처럼 보여?")
        self.assertIn("요청 7", label)
        self.assertIn("지금 몇 단계처럼 보여?", label)

    def test_named_completion_does_not_complete_the_wrong_current_step(self):
        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="named-completion")
        engine.question_router = AssemblyQuestionRouter()
        reply = engine._assembly("6단계 완료했습니다", [], 1)
        self.assertEqual(engine.tracker.step.id, "step_01")
        self.assertIn("현재 기록은 1단계", reply.text)
        self.assertIn("6단계까지", reply.text)

    def test_semantic_conversation_plan_can_start_work_without_keyword(self):
        class SemanticConversation:
            def route(self, utterance, *, capabilities, context):
                self.capabilities = capabilities
                self.context = context
                return ConversationDecision(
                    route=ConversationRoute.CAPABILITY_PLAN,
                    capability="assembly_start_or_resume")

        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(Path(__file__).parents[2] / "assembly_manuals/template/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="semantic-start")
        engine.conversation = SemanticConversation()
        reply = engine._chat("그거 이제 해볼까?", [], None, 1)
        self.assertEqual(reply.action, "DISPLAY_ON")
        self.assertIn("assembly_start_or_resume", engine.conversation.capabilities)
        self.assertEqual(engine.conversation.context["current_step"], "step_01")

    def test_ambiguous_semantic_plan_asks_one_targeted_question(self):
        decision = ConversationDecision(
            route=ConversationRoute.CLARIFY,
            steps=[], confidence=.4,
            clarification_question="현재 단계를 찾을까요, 조립 상태를 검사할까요?")
        reply = UnifiedCopilotEngine._semantic_clarification(decision, set())
        self.assertEqual(reply, decision.clarification_question)

    def test_visual_completion_requires_two_consistent_observations(self):
        class MemoryDB:
            def save_runtime_state(self, key, value):
                self.saved = (key, value)

        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="auto-next")
        engine.db = MemoryDB()
        checked = ValidatedAssessment(
            assessment="STEP_COMPLETE", observed_step_id="step_01",
            claimed_step_id=None, confidence=.75,
            instruction="1단계가 완료된 상태입니다.", observed_facts=[],
            issues=[], accepted=True, validation_notes=[])
        first = engine._apply_assessment(checked)
        self.assertEqual(engine.tracker.step.id, "step_01")
        self.assertIn("1/2회", first.text)
        reply = engine._apply_assessment(checked)
        self.assertEqual(engine.tracker.step.id, "step_02")
        self.assertEqual(reply.action, "DISPLAY_ON")
        self.assertIn("이제 2단계 작업을 시작", reply.text)
        self.assertTrue(reply.reference_image.endswith("page_006.jpg"))

    def test_wrong_step_never_mutates_tracker(self):
        class MemoryDB:
            def save_runtime_state(self, key, value):
                raise AssertionError("state must not be saved")

        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="no-wrong-transition")
        engine.db = MemoryDB()
        checked = ValidatedAssessment(
            assessment="WRONG_STEP", observed_step_id="step_04",
            claimed_step_id=None, confidence=.9, instruction="다른 단계로 보입니다.",
            observed_facts=[], issues=[], accepted=True, validation_notes=[])
        reply = engine._apply_assessment(checked)
        self.assertEqual(engine.tracker.step.id, "step_01")
        self.assertIn("변경하지 않습니다", reply.text)

    def test_truncated_vision_json_has_short_user_safe_error(self):
        reply = _request_error_reply(
            ValueError("Invalid JSON: EOF while parsing a string at column 1517"))
        self.assertEqual(reply.domain, "ERROR")
        self.assertIn("시각 판정 응답", reply.text)
        self.assertNotIn("Invalid JSON", reply.text)

    def test_api_timeout_has_specific_recovery_message(self):
        class APITimeoutError(Exception):
            pass

        reply = _request_error_reply(APITimeoutError("Request timed out."))
        self.assertEqual(reply.domain, "ERROR")
        self.assertIn("시간이 초과", reply.text)
        self.assertIn("ASSEMBLY_VISION_TIMEOUT_SECONDS", reply.text)

    def test_domains_are_separated_before_general_chat(self):
        router = UnifiedIntentRouter()
        self.assertEqual(router.route("현재 몇 단계야?").domain, "ASSEMBLY")
        self.assertEqual(router.route("다운로드한 PDF로 매뉴얼 만들어줘").domain, "MANUAL")
        self.assertEqual(router.route("이 부품이 뭐야?").domain, "VISION")
        self.assertEqual(router.route("오늘 기분 어때?").domain, "CONVERSATION")

    def test_named_step_information_is_not_a_start_or_visual_check(self):
        router = UnifiedIntentRouter()
        for text in ("6단계 작업은 뭐야?", "6단계 작업 어떻게 해?",
                     "6단계 과정을 설명해 줘"):
            with self.subTest(text=text):
                routed = router.route(text)
                self.assertEqual((routed.domain, routed.intent),
                                 ("ASSEMBLY", "TARGET_STEP_INFO"))
                question = AssemblyQuestionRouter().route(text)
                self.assertEqual(question.intent, "EXPLAIN_TARGET_STEP")
                self.assertEqual(question.claimed_step_id, "step_06")

    def test_target_step_explanation_does_not_move_current_step(self):
        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="target-info")
        engine.question_router = AssemblyQuestionRouter()
        engine.work_session_active = False
        reply = engine._assembly("6단계 작업은 뭐야?", [], 1)
        self.assertEqual(engine.tracker.step.id, "step_01")
        self.assertIn("6단계", reply.text)
        self.assertIn("1단계로 그대로", reply.text)

    def test_natural_assembly_checks_do_not_fall_into_chat_or_start(self):
        router = UnifiedIntentRouter()
        for text in ("지금 제가 잘 걸었나요?", "제가 잘했는지 확인해 주세요.",
                     "완료했는지 확인해 주세요.", "5단계 작업 완료했나요?",
                     "지금 어떤 단계까지 작업을 완료한 것 같아?",
                     "현재 화면을 봤을 때 어디까지 완료한 것 같아?"):
            with self.subTest(text=text):
                intent = router.route(text)
                self.assertEqual((intent.domain, intent.intent),
                                 ("ASSEMBLY", "QUESTION"))

    def test_completion_estimate_is_inspection_not_state_mutation(self):
        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="completion-estimate")
        engine.question_router = AssemblyQuestionRouter()
        before = engine.tracker.step.id
        reply = engine._assembly(
            "지금 어떤 단계까지 작업을 완료한 것 같아?", [], 1)
        self.assertEqual(engine.tracker.step.id, before)
        self.assertIn("RealSense 프레임", reply.text)

    def test_display_complaint_never_repeats_reference_action(self):
        router = UnifiedIntentRouter()
        for text in (
            "그 사진은 왜 보여주는 거야? 요청한 기억이 없는데.",
            "참고 사진을 요청한 적이 없는데 왜 화면에 띄워주냐고.",
        ):
            with self.subTest(text=text):
                intent = router.route(text)
                self.assertEqual((intent.domain, intent.intent),
                                 ("SYSTEM", "DISPLAY_FEEDBACK"))

    def test_camera_availability_question_does_not_show_reference(self):
        intent = UnifiedIntentRouter().route(
            "리얼센스 카메라로 실시간으로 보여주고 있는데 "
            "어떻게 실제 화면 이미지가 없어?")
        self.assertEqual((intent.domain, intent.intent),
                         ("SYSTEM", "CAMERA_FEEDBACK"))

    def test_live_camera_and_reference_images_are_distinct_actions(self):
        router = UnifiedIntentRouter()
        live_commands = (
            "실제 작업 이미지 띄워볼래",
            "참고 사진 말고 현재 실제 작업 이미지",
            "내가 작업 중인 이미지를 띄워줘",
            "내가 작업 중인 작업대 카메라를 화면에 띄우라고",
            "카메라 화면 화면에 띄워줘",
            "맞아 왜 안 띄워주는 거야",
        )
        for text in live_commands:
            with self.subTest(text=text):
                intent = router.route(text)
                self.assertEqual((intent.domain, intent.intent),
                                 ("SYSTEM", "DISPLAY_LIVE"))
        reference = router.route("3단계 참고 사진 띄워줘")
        self.assertNotEqual(reference.intent, "DISPLAY_LIVE")

    def test_live_display_correction_is_not_visual_no_answer(self):
        class PendingDB:
            def pending(self):
                return {"operation_type": "VISUAL_CLARIFICATION", "payload": {
                    "current": {"description": "벨트가 팽팽함"}}}

        class Engine:
            db = PendingDB()

        manager = CentralTurnManager.__new__(CentralTurnManager)
        manager.engine = Engine()
        self.assertIsNone(manager._pending_visual_answer(
            "유사한 단계가 아니라 실제 작업 중인 화면을 띄워달라고"))

    def test_natural_step_selection_confirmation_is_accepted(self):
        router = UnifiedIntentRouter()
        for text in ("바로 넘어가도 돼", "넘어가도 괜찮아"):
            intent = router.route(text, has_pending_confirmation=True)
            self.assertEqual((intent.domain, intent.intent),
                             ("CONFIRMATION", "ACCEPT"))

    def test_connected_work_question_routes_to_vlm_lane(self):
        intent = UnifiedIntentRouter().route("잘 연결한 것 같은데 어때 보여?")
        self.assertEqual((intent.domain, intent.intent), ("ASSEMBLY", "QUESTION"))

    def test_described_belt_progress_before_next_reidentifies_stage(self):
        question = AssemblyQuestionRouter().route(
            "벨트 연결까지 했는데 그 다음 작업 어떻게 돼?")
        self.assertEqual(question.intent, "IDENTIFY_STEP")

    def test_manual_mapping_challenge_is_grounded_in_loaded_yaml(self):
        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[3]
            / "dum_E_project/assembly_manuals/mini_conveyor_module/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="mapping-grounding")
        question = AssemblyQuestionRouter().route(
            "타이밍 벨트 장력 유지해서 거는 거는 7단계인 걸로 "
            "알고 있는데 왜 5단계로 설명해주니?")
        self.assertEqual(question.intent, "CHALLENGE_STEP_MAPPING")
        reply = engine._step_mapping_reply(question)
        self.assertIn("7단계는", reply.text)
        self.assertIn("5단계", reply.text)
        self.assertIn("assembly.yaml", reply.text)

    def test_return_to_incomplete_step_revokes_later_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
            manual = load_manual(
                Path(__file__).parents[3]
                / "dum_E_project/assembly_manuals/mini_conveyor_module/assembly.yaml")
            engine.tracker = AssemblyStateTracker(manual, session_id="reopen-seven")
            engine.tracker.complete_through(7)
            engine.question_router = AssemblyQuestionRouter()
            engine.db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            engine.work_session_active = True
            reply = engine._assembly(
                "7단계가 아직 덜 된 것 같은데 단계로 돌아갈 수 있을까?", [], 1)
            self.assertEqual(engine.tracker.step.order, 7)
            self.assertNotIn("step_07", engine.tracker.snapshot().completed_steps)
            self.assertIn("7단계 이후의 완료 기록은 취소", reply.text)

    def test_current_and_claimed_checks_do_not_search_unrelated_steps(self):
        manual = load_manual(
            Path(__file__).parents[3]
            / "dum_E_project/assembly_manuals/mini_conveyor_module/assembly.yaml")
        tracker = AssemblyStateTracker(manual, session_id="scoped-check")
        tracker.select_step_number(7)
        retriever = ManualRetriever(manual)
        current = AssemblyQuestionRouter().route("나 지금 잘하고 있는 거니?")
        claimed = AssemblyQuestionRouter().route("지금 7단계 작업을 잘하고 있는 것 같아?")
        self.assertEqual([step.order for step in retriever.retrieve(current, tracker.snapshot())],
                         [7])
        self.assertEqual([step.order for step in retriever.retrieve(claimed, tracker.snapshot())],
                         [7])

    def test_work_explanation_image_is_reference_not_live_camera(self):
        intent = UnifiedIntentRouter().route("작업 설명 이미지 보여줄래?")
        self.assertEqual((intent.domain, intent.intent), ("SYSTEM", "SHOW_REFERENCE"))

    def test_realsense_analysis_meta_question_reports_actual_frame_path(self):
        router = UnifiedIntentRouter()
        for text in (
            "내 작업대 카메라 화면이 안 보이니",
            "내 화면에 카메라가 보이는데 왜 인식하지 못하니",
            "RealSense 카메라 이미지를 프레임별로 가져가서 분석해야 하는 거 아니니?",
        ):
            with self.subTest(text=text):
                intent = router.route(text)
                self.assertEqual((intent.domain, intent.intent),
                                 ("SYSTEM", "CAMERA_FEEDBACK"))

    def test_realtime_cannot_rewrite_assembly_question_as_completion(self):
        source = (Path(__file__).parents[1] / "app.py").read_text()
        preserve = source.split("preserve_raw = (", 1)[1].split(")\n", 1)[0]
        self.assertIn('raw_intent.domain == "ASSEMBLY"', preserve)

    def test_short_yes_only_confirms_when_operation_is_pending(self):
        router = UnifiedIntentRouter()
        self.assertEqual(router.route("네").domain, "CONVERSATION")
        self.assertEqual(router.route("네", has_pending_confirmation=True).intent, "ACCEPT")
        natural = router.route("응 그렇게 해줘", has_pending_confirmation=True)
        self.assertEqual((natural.domain, natural.intent), ("CONFIRMATION", "ACCEPT"))
        filler = router.route("음, 그렇게 수정해줘", has_pending_confirmation=True)
        self.assertEqual((filler.domain, filler.intent), ("CONFIRMATION", "ACCEPT"))
        visual_yes = router.route("네 맞아요", has_pending_confirmation=True)
        self.assertEqual((visual_yes.domain, visual_yes.intent),
                         ("CONFIRMATION", "ACCEPT"))
        visual_no = router.route("아니요, 그렇지 않아", has_pending_confirmation=True)
        self.assertEqual((visual_no.domain, visual_no.intent),
                         ("CONFIRMATION", "REJECT"))
        screen_step = router.route("화면에 보이는 스텝도 4로 수정해줘")
        self.assertEqual((screen_step.domain, screen_step.intent),
                         ("ASSEMBLY", "ACTIVE_STEP_UPDATE"))

    def test_natural_confirmation_really_changes_selected_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
            manual = load_manual(
                Path(__file__).parents[2]
                / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
            engine.tracker = AssemblyStateTracker(manual, session_id="confirm-selection")
            engine.tracker.select_step_number(6)
            engine.db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            engine.work_session_active = True
            engine.db.set_pending("SELECT_ASSEMBLY_STEP", {"step_order": 1})
            intent = UnifiedIntentRouter().route(
                "응 그렇게 해줘", has_pending_confirmation=True)
            reply = engine._confirmation(intent.intent)
            self.assertEqual(engine.tracker.step.order, 1)
            self.assertEqual(reply.action, "DISPLAY_ON")
            self.assertIn("1단계", reply.text)

    def test_display_and_runtime_controls_are_distinct(self):
        router = UnifiedIntentRouter()
        self.assertEqual(router.route("AR 화면 보여줘").intent, "DISPLAY_ON")
        self.assertEqual(router.route("컨베이어 조립 AR 창 띄워줘").intent, "DISPLAY_ON")
        self.assertEqual(router.route("카메라 화면 닫아줘").intent, "DISPLAY_OFF")
        saved = router.route("3단계까지 한 것을 기억해 주고 창 닫아줘")
        self.assertEqual(saved.intent, "SAVE_PROGRESS_AND_DISPLAY_OFF")
        korean = router.route("삼 단계까지 완료한 것으로 기억하고 창 닫아줘")
        self.assertEqual(korean.intent, "SAVE_PROGRESS_AND_DISPLAY_OFF")
        stopping = router.route("3단계 작업까지만 하고 이제 종료할게")
        self.assertEqual(stopping.intent, "SAVE_PROGRESS_AND_CLARIFY_STOP")
        self.assertEqual(router.route("에이어 창 띄워줘").intent, "DISPLAY_ON")
        stopped = router.route("이제 작업 종료해줘")
        self.assertEqual((stopped.domain, stopped.intent), ("SYSTEM", "PAUSE_WORK"))
        self.assertEqual(router.route("종료해 줘.").intent, "CLARIFY_STOP_SCOPE")
        self.assertEqual(router.route("코파일럿 종료").intent, "STOP_COPILOT")
        remembered = router.route("지금 작업을 기억하고 종료해 줘")
        self.assertEqual((remembered.domain, remembered.intent),
                         ("SYSTEM", "CLARIFY_STOP_SCOPE"))

    def test_user_progress_and_verified_progress_are_separate(self):
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        tracker = AssemblyStateTracker(manual, session_id="progress-provenance")
        claimed = tracker.complete_through(3)
        self.assertEqual(claimed.current_step_index, 3)
        self.assertEqual(claimed.verification_status, "USER_CONFIRMED")
        self.assertEqual(claimed.user_confirmed_steps, claimed.completed_steps)
        self.assertEqual(claimed.verified_completed_steps, ())
        verified = tracker.complete_through(3, verified=True)
        self.assertEqual(verified.verification_status, "VERIFIED")
        self.assertEqual(verified.verified_completed_steps, verified.completed_steps)

        tracker.complete_through(5)
        partially_verified = tracker.complete_through(3, verified=True)
        self.assertEqual(partially_verified.current_step_index, 5)
        self.assertEqual(len(partially_verified.user_confirmed_steps), 5)
        self.assertEqual(len(partially_verified.verified_completed_steps), 3)
        self.assertEqual(partially_verified.verification_status, "USER_CONFIRMED")

    def test_explicit_progress_update_has_its_own_intent(self):
        router = UnifiedIntentRouter()
        for text in (
            "시각 조건 확인하지 말고 5단계 완료했으니까 작업 상태 업데이트해 줘",
            "내가 6단계까지 완료했으니까 작업 상태 업데이트 해줘",
            "7단계까지 확인했으니까 현재 작업 상태에 반영해 줘",
            "6단계 완료한 걸로 저장해줘",
            "6단계까지 완료한 걸로 저장해줘",
        ):
            with self.subTest(text=text):
                intent = router.route(text)
                self.assertEqual((intent.domain, intent.intent),
                                 ("ASSEMBLY", "USER_PROGRESS_UPDATE"))
        active = router.route("그러면 현재 단계를 7단계로 업데이트 해줘")
        self.assertEqual((active.domain, active.intent),
                         ("ASSEMBLY", "ACTIVE_STEP_UPDATE"))

    def test_progress_question_is_not_mistaken_for_start_work(self):
        intent = UnifiedIntentRouter().route(
            "지금 타이밍벨트 장력 조정까지 해서 다 연결했고, "
            "그 다음 작업 진행하려고 해. 몇 단계야?")
        self.assertEqual((intent.domain, intent.intent), ("ASSEMBLY", "QUESTION"))

    def test_named_future_completion_creates_one_clear_scope_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
            manual = load_manual(
                Path(__file__).parents[2]
                / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
            engine.tracker = AssemblyStateTracker(manual, session_id="completion-scope")
            engine.tracker.select_step_number(3)
            engine.question_router = AssemblyQuestionRouter()
            engine.db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            engine.work_session_active = True
            proposal = engine._assembly("6단계 완료했어", [], 1)
            self.assertIn("6단계까지 완료로 저장할까요", proposal.text)
            self.assertEqual(engine.db.pending()["operation_type"], "COMPLETE_THROUGH_STEP")
            result = engine._confirmation("ACCEPT")
            self.assertEqual(engine.tracker.step.order, 7)
            self.assertIn("다음은 7단계", result.text)

    def test_explicit_progress_save_supersedes_old_visual_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
            manual = load_manual(
                Path(__file__).parents[2]
                / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
            engine.tracker = AssemblyStateTracker(manual, session_id="explicit-save")
            engine.tracker.select_step_number(3)
            engine.db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            engine.work_session_active = True
            engine.db.set_pending("VISUAL_CLARIFICATION", {
                "step_order": 5,
                "current": {"id": "belt", "description": "벨트 장력",
                            "camera_result": "unknown"},
            })
            result = engine._user_progress_update("6단계까지 완료한 걸로 저장해줘")
            self.assertEqual(engine.tracker.step.order, 7)
            self.assertIsNone(engine.db.pending())
            self.assertIn("6단계까지 완료로 기록", result.text)

    def test_progress_updates_mutate_pointer_without_claiming_vision_verification(self):
        class MemoryDB:
            def save_runtime_state(self, key, value):
                self.saved = (key, value)

        engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
        manual = load_manual(
            Path(__file__).parents[2]
            / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        engine.tracker = AssemblyStateTracker(manual, session_id="spoken-progress")
        engine.db = MemoryDB()
        engine.work_session_active = False
        reply = engine._user_progress_update("6단계까지 완료했으니까 상태 업데이트")
        self.assertEqual(engine.tracker.step.order, 7)
        self.assertIn("사용자 확인 기준", reply.text)
        self.assertEqual(len(engine.tracker.snapshot().verified_completed_steps), 0)

        reply = engine._active_step_update("현재 단계를 4단계로 업데이트")
        self.assertEqual(engine.tracker.step.order, 4)
        self.assertIn("4단계", reply.text)

    def test_step_explanation_and_image_are_combined(self):
        text = "7단계 어떻게 해야 되는지 이미지랑 설명 띄워줘"
        routed = UnifiedIntentRouter().route(text)
        question = AssemblyQuestionRouter().route(text)
        self.assertEqual((routed.domain, routed.intent),
                         ("ASSEMBLY", "TARGET_STEP_INFO"))
        self.assertEqual(question.intent, "EXPLAIN_AND_SHOW_TARGET")

    def test_perception_requests_are_fifo_instead_of_busy_rejected(self):
        source = (Path(__file__).parents[1] / "app.py").read_text()
        self.assertNotIn("결과가 나온 뒤 다시 확인해 주세요", source)
        self.assertIn("cancel_prior_perception", source)

    def test_work_start_and_reference_image_commands(self):
        router = UnifiedIntentRouter()
        self.assertEqual(router.route("작업 시작하자").intent, "START_WORK")
        self.assertEqual(router.route("지금부터 시작해 볼까?").intent, "START_WORK")
        self.assertEqual(router.route("이전 작업을 이어서 하자").intent, "START_WORK")
        self.assertEqual(router.route("1단계 조립").intent, "START_WORK")
        show = router.route("사진도 같이 띄워줘")
        self.assertEqual((show.domain, show.intent), ("SYSTEM", "SHOW_REFERENCE"))

    def test_compound_resume_uses_target_step_not_first_completed_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
            manual = load_manual(
                Path(__file__).parents[2]
                / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
            engine.tracker = AssemblyStateTracker(manual, session_id="compound-resume")
            engine.db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            engine.work_session_active = False
            reply = engine._start_work(
                "작업 이어서 시작하자. 지금 1단계 완료했고 3단계부터 시작하면 될 것 같아.")
            self.assertEqual(engine.tracker.step.order, 2)
            self.assertEqual(engine.db.pending()["payload"]["step_order"], 3)
            self.assertIn("1단계 완료는 기록", reply.text)
            self.assertIn("2단계 완료 기록은 없습니다", reply.text)

    def test_natural_visual_answer_uses_pending_question_context(self):
        class PendingDB:
            def pending(self):
                return {"operation_type": "VISUAL_CLARIFICATION", "payload": {
                    "current": {"description": "타이밍벨트가 양쪽 풀리를 감싸며 팽팽함"}}}

        class Engine:
            db = PendingDB()

        manager = CentralTurnManager.__new__(CentralTurnManager)
        manager.engine = Engine()
        self.assertEqual(manager._pending_visual_answer(
            "네, 팽팽하게 놓은 상태가 맞습니다."), "ACCEPT")
        self.assertEqual(manager._pending_visual_answer("그 상태가 맞아"), "ACCEPT")
        self.assertIsNone(manager._pending_visual_answer(
            "모터 조립체도 단단히 고정했어요"))
        manager.engine.db.pending = lambda: {
            "operation_type": "VISUAL_CLARIFICATION", "payload": {"current": {
                "description": "왼쪽 풀리와 오른쪽 모터 유닛의 풀리 높이가 비슷함"}}}
        self.assertIsNone(manager._pending_visual_answer(
            "모터 조립체도 단단히 고정했어요"))

    def test_user_answers_complete_visual_checks_then_advances_to_next_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
            manual = load_manual(
                Path(__file__).parents[2]
                / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
            engine.tracker = AssemblyStateTracker(manual, session_id="assisted-completion")
            engine.db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            engine.work_session_active = True
            engine._visual_question_history = {}
            engine.db.set_pending("VISUAL_CLARIFICATION", {
                "step_order": 5,
                "current": {"id": "belt", "description": "벨트가 팽팽함",
                            "camera_result": "unknown"},
                "remaining": [{"id": "parallel", "description": "두 가닥이 평행함",
                               "camera_result": "unknown"}],
                "answers": [], "asked_ids": [], "resolution": "COMPLETE_STEP",
            })

            followup = engine._confirmation("ACCEPT")
            self.assertIn("다음 한 가지", followup.text)
            self.assertEqual(engine.db.pending()["payload"]["resolution"], "COMPLETE_STEP")
            proposal = engine._confirmation("ACCEPT")
            self.assertIn("5단계 완료 상태", proposal.text)
            self.assertEqual(engine.db.pending()["operation_type"],
                             "CONFIRM_USER_ASSISTED_COMPLETION")
            result = engine._confirmation("ACCEPT")
            self.assertEqual(engine.tracker.step.order, 6)
            self.assertEqual(engine.tracker.snapshot().verification_status, "USER_ASSISTED")
            self.assertIn("다음은 6단계", result.text)
            self.assertEqual(result.action, "DISPLAY_ON")

    def test_current_information_uses_search_domain(self):
        router = UnifiedIntentRouter()
        self.assertEqual(router.route("오늘 서울 날씨 알려줘").domain, "SEARCH")
        self.assertEqual(router.route("이 주소를 웹에서 검색해줘").domain, "SEARCH")
        self.assertEqual(router.route("내 위치 기준으로 한식 맛집을 찾아줘").domain,
                         "SEARCH")
        self.assertEqual(router.route("그 근처 카페 추천해줘").domain, "SEARCH")


class StateDBTests(unittest.TestCase):
    def test_pending_operation_is_explicitly_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = CopilotStateDB(Path(tmp) / "copilot.sqlite3")
            db.set_pending("REPLACE", {"path": "x"})
            self.assertEqual(db.pending()["operation_type"], "REPLACE")
            self.assertEqual(db.resolve_pending("REJECTED")["payload"]["path"], "x")
            self.assertIsNone(db.pending())
            db.close()


class ManualScanTests(unittest.TestCase):
    def test_pdf_hash_matches_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads, manuals = root / "downloads", root / "manuals"
            downloads.mkdir(); product = manuals / "product"; product.mkdir(parents=True)
            pdf = downloads / "manual.pdf"; pdf.write_bytes(b"%PDF-test")
            (product / "generation_manifest.json").write_text(json.dumps({
                "source_sha256": sha256(pdf), "product_slug": "product"}), encoding="utf-8")
            statuses = ManualGenerationService(downloads, manuals, generator=object()).scan()
            self.assertFalse(statuses[0].is_new)
            self.assertEqual(statuses[0].existing_product, "product")

    def test_preserved_source_pdf_without_manifest_is_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads, manuals = root / "downloads", root / "manuals"
            downloads.mkdir(); product = manuals / "legacy_product"; product.mkdir(parents=True)
            pdf = downloads / "manual.pdf"; pdf.write_bytes(b"%PDF-same")
            (product / "source_manual.pdf").write_bytes(pdf.read_bytes())
            status = ManualGenerationService(downloads, manuals, generator=object()).scan()[0]
            self.assertEqual(status.existing_product, "legacy_product")


if __name__ == "__main__":
    unittest.main()
