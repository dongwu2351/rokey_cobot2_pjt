from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from assembly_copilot.copilot import AssemblyCopilot
from assembly_copilot.assessment_models import AssemblyAssessment, VisualCheckResult
from assembly_copilot.assessment_validator import AssessmentValidator
from assembly_copilot.app import _apply_advance, _open_source
from assembly_copilot.realsense_source import _profile_candidates
from assembly_copilot.frame_buffer import RecentFrameBuffer
from assembly_copilot.manual import load_manual
from assembly_copilot.manual_retriever import ManualRetriever
from assembly_copilot.multimodal_assessor import _reference_images
from assembly_copilot.ar_overlay import _crop_reference_content
from assembly_copilot.models import AssemblyObservation
from assembly_copilot.question_router import AssemblyQuestionRouter
from assembly_copilot.state_tracker import AssemblyStateTracker
from assembly_copilot.voice_io import _TranscriptOnlyRouter
from VoiceProcessing.command_models import CommandResult
from speech_manager import SpeechManager


ROOT = PROJECT
MANUAL = ROOT / "assembly_manuals" / "template" / "assembly.yaml"


class ScaffoldTest(unittest.TestCase):
    def test_transcript_only_router_satisfies_audio_pipeline_contract(self):
        result = _TranscriptOnlyRouter().parse_command("현재 단계가 뭐죠?", None)
        self.assertIsInstance(result, CommandResult)
        self.assertEqual(result.raw_utterance, "현재 단계가 뭐죠?")
        self.assertFalse(result.actions)

    def test_realsense_profiles_fall_back_to_safe_defaults(self):
        candidates = _profile_candidates(1280, 720, 30)
        self.assertIn(((640, 480), (640, 480), 30), candidates)
        self.assertEqual(candidates[-1], (None, None, 0))

    def test_template_loads_and_manual_confirmation_advances(self):
        manual = load_manual(MANUAL)
        tracker = AssemblyStateTracker(manual, session_id="test")
        state = tracker.update(AssemblyObservation(1, visibility="VISIBLE"))
        self.assertEqual(state.current_step_id, "step_01")
        self.assertTrue(state.unsatisfied_conditions)
        self.assertNotIn("operator_confirmed", state.unsatisfied_conditions)
        finished = tracker.confirm_current_step()
        self.assertEqual(finished.status, "COMPLETED")

    def test_question_is_grounded_in_current_instruction(self):
        manual = load_manual(MANUAL)
        tracker = AssemblyStateTracker(manual, session_id="test")
        answer = AssemblyCopilot(manual).answer("다음은 뭘 해야 해?", tracker.snapshot())
        self.assertEqual(answer.intent, "NEXT_STEP")
        self.assertIn(manual.steps[0].title, answer.text)

    def test_claimed_step_is_extracted(self):
        question = AssemblyQuestionRouter().route("지금 step 3단계 하는데 이게 맞아?")
        self.assertEqual(question.intent, "CHECK_CLAIMED_STEP")
        self.assertEqual(question.claimed_step_id, "step_03")

    def test_explicit_completion_is_routed(self):
        question = AssemblyQuestionRouter().route("1단계 작업 다 했어")
        self.assertEqual(question.intent, "COMPLETE_STEP")

    def test_completion_question_uses_vision_without_mutating_state(self):
        question = AssemblyQuestionRouter().route("5단계 작업 완료했나요?")
        self.assertEqual(question.intent, "CHECK_CLAIMED_STEP")
        self.assertEqual(question.claimed_step_id, "step_05")

    def test_natural_progress_question_uses_vision_check(self):
        question = AssemblyQuestionRouter().route("이 정도면 잘 된 것 같아?")
        self.assertEqual(question.intent, "CHECK_PROGRESS")

    def test_described_work_before_next_question_reidentifies_step(self):
        question = AssemblyQuestionRouter().route(
            "타이밍벨트 걸고 프로파일 고정 너트 볼트를 최대한 조였는데 "
            "이 다음 작업이 뭘까요?")
        self.assertEqual(question.intent, "IDENTIFY_STEP")
        report = AssemblyQuestionRouter().route("모터 조립체도 단단히 고정했어요")
        self.assertEqual(report.intent, "IDENTIFY_STEP")

    def test_tts_echo_filter_keeps_distinct_user_interrupt(self):
        import time
        manager = SpeechManager(enabled=False)
        manager._last_text = "현재 단계 작업을 시작합니다"
        manager._last_ended_at = time.monotonic()
        self.assertTrue(manager.is_likely_echo("현재 단계 작업을 시작합니다"))
        self.assertFalse(manager.is_likely_echo("화면 닫아 줘"))

    def test_current_step_question_uses_all_yaml_steps(self):
        manual = load_manual(ROOT / "assembly_manuals" / "block_test" / "assembly.yaml")
        tracker = AssemblyStateTracker(manual, session_id="test")
        question = AssemblyQuestionRouter().route("현재 단계가 뭐죠?")
        candidates = ManualRetriever(manual).retrieve(question, tracker.snapshot())
        self.assertEqual(question.intent, "IDENTIFY_STEP")
        self.assertEqual(len(candidates), len(manual.steps))
        self.assertTrue(all(step.visual_state for step in candidates))
        self.assertFalse(any(step.references.images for step in candidates))

    def test_generated_manual_supplies_one_reference_image_per_step(self):
        manual = load_manual(
            ROOT / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        selected = _reference_images(list(manual.steps))
        self.assertEqual(len(selected), len(manual.steps))
        self.assertEqual([step.id for step, _ in selected],
                         [step.id for step in manual.steps])
        self.assertTrue(all(path.is_file() for _, path in selected))

    def test_pdf_reference_blank_margins_are_cropped(self):
        page = np.full((600, 400, 3), 255, np.uint8)
        page[40:360, 50:350] = 100
        cropped = _crop_reference_content(page)
        self.assertLess(cropped.shape[0], page.shape[0])
        self.assertLess(cropped.shape[1], page.shape[1])

    def test_current_work_variants_use_vision_identification(self):
        router = AssemblyQuestionRouter()
        questions = [
            "현재 작업 상태가 어때?", "어디까지 했어?",
            "무슨 작업을 하고 있어?", "지금 몇 스텝이야?",
            "지금 몇 단계처럼 보여?",
        ]
        self.assertTrue(all(router.route(q).intent == "IDENTIFY_STEP" for q in questions))

    def test_fresh_completion_advances_and_stale_one_does_not(self):
        manual = load_manual(MANUAL)
        tracker = AssemblyStateTracker(manual, session_id="test")
        advanced, _ = _apply_advance(tracker, {"source_step_id": "step_01"})
        self.assertEqual(advanced, "step_01")
        self.assertIsNone(tracker.step)
        stale, _ = _apply_advance(tracker, {"source_step_id": "step_01"})
        self.assertIsNone(stale)

    def test_recent_buffer_selects_time_ordered_frames(self):
        history = RecentFrameBuffer(seconds=8, sample_fps=4)
        image = np.zeros((20, 20, 3), np.uint8)
        for index in range(12):
            history.add(image, index * 300)
        selected = history.representative(5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, sorted(selected, key=lambda item: item.timestamp_ms))

    def test_validator_only_does_not_mutate_tracker(self):
        manual = load_manual(MANUAL)
        tracker = AssemblyStateTracker(manual, session_id="test")
        observation = AssemblyObservation(1, visibility="VISIBLE")
        state = tracker.update(observation)
        raw = AssemblyAssessment(
            assessment="STEP_COMPLETE", observed_step_id="step_01",
            claimed_step_id=None, confidence=.9, visible=True,
            instruction="완료로 보입니다.", needs_better_view=False)
        checked = AssessmentValidator(manual).validate(raw, state, observation)
        self.assertTrue(checked.accepted)
        self.assertEqual(tracker.step.id, "step_01")
        self.assertTrue(checked.validation_notes)

    def test_validator_uses_point_four_confidence_threshold(self):
        manual = load_manual(MANUAL)
        tracker = AssemblyStateTracker(manual, session_id="confidence-test")
        observation = AssemblyObservation(1, visibility="VISIBLE")
        state = tracker.update(observation)
        accepted = AssessmentValidator(manual).validate(AssemblyAssessment(
            assessment="CORRECT", observed_step_id="step_01",
            confidence=.40, visible=True, instruction="후보 단계입니다.",
            needs_better_view=False), state, observation)
        rejected = AssessmentValidator(manual).validate(AssemblyAssessment(
            assessment="CORRECT", observed_step_id="step_01",
            confidence=.39, visible=True, instruction="후보 단계입니다.",
            needs_better_view=False), state, observation)
        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)
        self.assertIn("40%", rejected.instruction)

    def test_validator_rejects_missing_required_structured_checks(self):
        manual = load_manual(
            ROOT / "assembly_manuals/conveyor-motor-timing-belt-assembly/assembly.yaml")
        tracker = AssemblyStateTracker(manual, session_id="structured-check")
        observation = AssemblyObservation(1, visibility="VISIBLE")
        state = tracker.update(observation)
        first = manual.steps[0]
        required = [item for item in first.visual_checks if item["kind"] == "required"]
        partial = [VisualCheckResult(
            check_id=required[0]["id"], result="true", evidence="화면에서 확인")]
        checked = AssessmentValidator(manual).validate(AssemblyAssessment(
            assessment="STEP_COMPLETE", observed_step_id=first.id,
            confidence=.99, visible=True, checks=partial,
            instruction="완료입니다.", needs_better_view=False), state, observation)
        self.assertFalse(checked.accepted)
        self.assertEqual(checked.assessment, "UNCERTAIN")
        self.assertLess(checked.confidence, 1.0)
        self.assertIn("필수 시각 조건", checked.instruction)

    def test_depth_quality_metadata_is_recorded(self):
        history = RecentFrameBuffer(seconds=2, sample_fps=4)
        image = np.zeros((30, 30, 3), np.uint8)
        depth = np.full((30, 30), 1000, np.uint16)
        history.add(image, 1, depth, 0.001)
        frame = history.representative(1)[0]
        self.assertEqual(frame.depth_valid_ratio, 1.0)
        self.assertGreaterEqual(frame.quality, 0.0)

    @patch("assembly_copilot.app.OpenCVSource")
    @patch("assembly_copilot.app.RealSenseSource")
    def test_auto_camera_falls_back_to_opencv(self, realsense, opencv):
        realsense.side_effect = RuntimeError("pyrealsense2 없음")
        opencv.return_value = object()
        args = Namespace(video=None, camera="auto", camera_index=2)

        source, camera_name = _open_source(args)

        self.assertIs(source, opencv.return_value)
        self.assertEqual(camera_name, "opencv")
        opencv.assert_called_once_with(2)

    @patch("assembly_copilot.app.OpenCVSource")
    @patch("assembly_copilot.app.RealSenseSource")
    def test_auto_camera_reports_when_no_camera_is_available(self, realsense, opencv):
        realsense.side_effect = RuntimeError("pyrealsense2 없음")
        opencv.side_effect = RuntimeError("카메라 없음")
        args = Namespace(video=None, camera="auto", camera_index=0)

        with self.assertRaisesRegex(RuntimeError, "사용 가능한 카메라"):
            _open_source(args)


if __name__ == "__main__":
    unittest.main()
