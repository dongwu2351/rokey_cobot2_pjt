from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from VoiceProcessing.TTS import SpeechSynthesizer, TTSConfig
from VoiceProcessing.assistive_models import (
    AssistiveResult,
    AssistiveState,
    BoundingBox,
    ComparisonConstraint,
    ComparisonOperator,
    Detection,
    PerceptionDecision,
    PerceptionPlan,
    SelectionPolicy,
    SpatialRelation,
)
from VoiceProcessing.assistive_cli import _object_at_point
from VoiceProcessing.assistive_processor import AssistiveCommandProcessor
from VoiceProcessing.command_models import Intent
from VoiceProcessing.grounding import SceneTracker
from VoiceProcessing.object_memory import ObjectMemory
from VoiceProcessing.realtime_runtime import FrameSnapshot, OpticalFlowBoxTracker
from VoiceProcessing.situated_parser import SituatedCommandParser


def detection(label: str, x1: float, x2: float, *, score: float = 0.9, **attrs):
    return Detection(
        label=label,
        score=score,
        bbox=BoundingBox(x1=x1, y1=10, x2=x2, y2=20),
        attributes=attrs,
    )


class FakeGrounder:
    def __init__(self, detections):
        self.detections = tuple(detections)
        self.calls = []

    def detect(self, image, query):
        self.calls.append((image, query))
        return self.detections


class FakeParser:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def parse(self, utterance, **kwargs):
        self.calls.append((utterance, kwargs))
        return self.plan


class ObjectMemoryTests(unittest.TestCase):
    def test_remembered_alias_recalls_prompt_without_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            with ObjectMemory(Path(directory) / "memory.sqlite3") as memory:
                memory.remember(
                    "flange adapter",
                    grounding_prompt="silver circular flange adapter with four holes",
                    aliases=("그 은색 어댑터", "플랜지 부품"),
                    attributes={"material": "metal"},
                    now_ms=100,
                )
                recalled = memory.recall("플랜지 부품")

                self.assertIsNotNone(recalled)
                self.assertEqual(recalled.canonical_name, "flange adapter")
                self.assertEqual(recalled.recall_count, 1)
                self.assertNotIn("coordinates", recalled.attributes)


class SituatedParserTests(unittest.TestCase):
    def test_known_concept_uses_local_fast_path_without_llm(self):
        parser = SituatedCommandParser(client=Mock())
        remembered = SimpleNamespace(
            canonical_name="flange adapter",
            grounding_prompt="silver flange adapter",
            aliases=("플랜지 부품",),
            attributes={},
        )

        plan = parser.parse("플랜지 부품 가져와", remembered=(remembered,))

        self.assertEqual(plan.decision, PerceptionDecision.GROUND)
        self.assertEqual(plan.target_description, "silver flange adapter")
        parser._client.responses.parse.assert_not_called()

    def test_spatial_clarification_reply_uses_local_fast_path(self):
        parser = SituatedCommandParser(client=Mock())
        pending = PerceptionPlan(
            decision=PerceptionDecision.CLARIFY,
            intent=Intent.FETCH,
            target_category="screw",
            clarification_question="왼쪽과 오른쪽 중 어느 것인가요?",
        )

        plan = parser.parse("왼쪽 것", pending_plan=pending)

        self.assertEqual(plan.decision, PerceptionDecision.GROUND)
        self.assertEqual(plan.spatial_relation, SpatialRelation.LEFTMOST)
        self.assertFalse(parser.last_used_llm)
        parser._client.responses.parse.assert_not_called()


class AssistiveProcessorTests(unittest.TestCase):
    def memory(self, directory):
        return ObjectMemory(Path(directory) / "memory.sqlite3")

    def test_unique_zero_shot_detection_is_grounded_but_not_executable(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="custom jig",
            target_description="blue L shaped assembly jig",
            source_object_expression="파란 L자 지그",
        )
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=FakeGrounder([detection("custom jig", 10, 40)]),
                memory=memory,
            )

            result = processor.process("파란 L자 지그 가져와", object())

            self.assertEqual(result.state, AssistiveState.GROUNDED)
            self.assertIsNotNone(result.selected_object_id)
            self.assertFalse(result.executable)
            self.assertFalse(result.geometry_verified)
            self.assertIsNotNone(memory.get("custom jig"))
            self.assertEqual(
                memory.recall("파란 L자 지그").canonical_name, "custom jig"
            )
            memory.close()

    def test_latest_image_callback_is_resolved_immediately_before_grounding(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="hammer",
        )
        grounder = FakeGrounder([detection("hammer", 10, 40)])
        latest_frame = object()
        provider = Mock(return_value=latest_frame)
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=grounder,
                memory=memory,
            )

            processor.process("해머 가져와", provider)

            provider.assert_called_once_with()
            self.assertIs(grounder.calls[0][0], latest_frame)
            memory.close()
    def test_multiple_candidates_ask_question_and_call_speaker(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="screw",
        )
        spoken = []
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=FakeGrounder(
                    [detection("screw", 10, 30), detection("screw", 50, 70)]
                ),
                memory=memory,
                speaker=spoken.append,
            )

            result = processor.process("나사 가져와", object())

            self.assertEqual(result.state, AssistiveState.CLARIFICATION_REQUIRED)
            self.assertIn("후보가 2개", result.clarification_question)
            self.assertEqual(spoken, [result.clarification_question])
            self.assertEqual(processor.pending_plan, plan)
            memory.close()

    def test_slightly_longer_selects_smallest_positive_difference(self):
        tracker = SceneTracker()
        initial = tracker.update(
            [
                detection("screw", 10, 30, length_mm=20),
                detection("screw", 50, 74, length_mm=24),
                detection("screw", 90, 130, length_mm=40),
            ],
            now_ms=100,
        )
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="screw",
            reference_expression="이 나사",
            exclude_reference=True,
            comparison=ComparisonConstraint(
                attribute="length",
                operator=ComparisonOperator.GREATER_THAN,
                reference_expression="이 나사",
                selection_policy=SelectionPolicy.NEAREST_GREATER,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=FakeGrounder(
                    [
                        detection("screw", 10, 30, length_mm=20),
                        detection("screw", 50, 74, length_mm=24),
                        detection("screw", 90, 130, length_mm=40),
                    ]
                ),
                memory=memory,
                tracker=tracker,
            )
            processor.set_focus(initial[0].id)

            result = processor.process(
                "이 나사 말고 조금 더 긴 나사 가져와", object()
            )

            self.assertEqual(result.state, AssistiveState.GROUNDED)
            self.assertEqual(result.selected_object_id, initial[1].id)
            self.assertTrue(result.geometry_verified)
            self.assertFalse(result.executable)
            memory.close()

    def test_comparison_resumes_after_user_selects_reference_box(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="screw",
            source_object_expression="나사",
            reference_expression="이 나사",
            exclude_reference=True,
            comparison=ComparisonConstraint(
                attribute="length",
                operator=ComparisonOperator.GREATER_THAN,
                reference_expression="이 나사",
                selection_policy=SelectionPolicy.NEAREST_GREATER,
            ),
        )
        parser = FakeParser(plan)
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=parser,
                grounder=FakeGrounder(
                    [
                        detection("screw", 10, 30, length_mm=20),
                        detection("screw", 50, 74, length_mm=24),
                        detection("screw", 90, 130, length_mm=40),
                    ]
                ),
                memory=memory,
            )

            first = processor.process(
                "이 나사 말고 조금 더 긴 나사 가져와", object()
            )
            reference = _object_at_point(first, 15, 15)
            self.assertEqual(first.state, AssistiveState.CLARIFICATION_REQUIRED)
            self.assertIsNotNone(reference)
            processor.set_focus(reference.id)

            second = processor.process("그것보다 조금 더 긴 것", object())

            self.assertEqual(second.state, AssistiveState.GROUNDED)
            self.assertEqual(second.selected_object_id, first.detections[1].id)
            self.assertEqual(parser.calls[1][1]["pending_plan"], plan)
            memory.close()

    def test_clarification_with_visual_query_returns_clickable_candidates(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.CLARIFY,
            intent=Intent.FETCH,
            target_category="rod",
            target_description="red rod",
            visual_query_alternatives=("red stick",),
            source_object_expression="빨간 막대",
            clarification_question="어떤 빨간 막대를 기준으로 할까요?",
        )
        grounder = FakeGrounder(
            [detection("red stick", 10, 30), detection("red stick", 50, 80)]
        )
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=grounder,
                memory=memory,
            )

            result = processor.process("좀 더 긴 빨간 막대 가져와", object())

            self.assertEqual(result.state, AssistiveState.CLARIFICATION_REQUIRED)
            self.assertEqual(len(result.detections), 2)
            self.assertEqual(grounder.calls[0][1], "red rod. red stick")
            self.assertEqual(result.clarification_question, plan.clarification_question)
            memory.close()

    def test_voice_spatial_hint_selects_left_candidate_without_click(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="rod",
            spatial_relation=SpatialRelation.LEFTMOST,
        )
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=FakeGrounder(
                    [detection("rod", 80, 110), detection("rod", 10, 40)]
                ),
                memory=memory,
            )

            result = processor.process("왼쪽 막대 가져와", object())

            self.assertEqual(result.state, AssistiveState.GROUNDED)
            self.assertIsNotNone(result.selected_object_id)
            selected = next(
                item for item in result.detections if item.id == result.selected_object_id
            )
            self.assertEqual(selected.bbox.x1, 10)
            memory.close()

    def test_comparison_without_focus_asks_voice_spatial_question(self):
        plan = PerceptionPlan(
            decision=PerceptionDecision.GROUND,
            intent=Intent.FETCH,
            target_category="screw",
            comparison=ComparisonConstraint(
                attribute="length",
                operator=ComparisonOperator.GREATER_THAN,
                reference_expression="this screw",
                selection_policy=SelectionPolicy.NEAREST_GREATER,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            memory = self.memory(directory)
            processor = AssistiveCommandProcessor(
                parser=FakeParser(plan),
                grounder=FakeGrounder(
                    [detection("screw", 10, 30), detection("screw", 50, 80)]
                ),
                memory=memory,
            )

            result = processor.process("그것보다 긴 나사", object())

            self.assertEqual(result.state, AssistiveState.CLARIFICATION_REQUIRED)
            self.assertIn("왼쪽", result.clarification_question)
            self.assertEqual(len(result.detections), 2)
            memory.close()


class FocusSelectionTests(unittest.TestCase):
    def test_click_selects_smallest_overlapping_box(self):
        tracker = SceneTracker()
        objects = tracker.update(
            [
                detection("screw", 10, 100),
                detection("screw", 20, 40),
            ],
            now_ms=100,
        )
        result = AssistiveResult(
            state=AssistiveState.CLARIFICATION_REQUIRED,
            detections=objects,
        )

        selected = _object_at_point(result, 25, 15)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.bbox.x1, 20)

    def test_click_outside_boxes_selects_nothing(self):
        result = AssistiveResult(state=AssistiveState.CLARIFICATION_REQUIRED)

        self.assertIsNone(_object_at_point(result, 1, 1))


class OpticalFlowTrackerTests(unittest.TestCase):
    def test_box_follows_translated_visual_features(self):
        first = np.zeros((100, 100, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        first[30:51, 20:41] = 255
        second[30:51, 26:47] = 255
        tracked = SceneTracker().update(
            [
                Detection(
                    label="part",
                    score=0.9,
                    bbox=BoundingBox(x1=18, y1=28, x2=43, y2=53),
                )
            ],
            now_ms=100,
        )
        tracker = OpticalFlowBoxTracker()
        tracker.reset(FrameSnapshot(1, 1.0, first), tracked)

        tracker.update(FrameSnapshot(2, 1.03, second))

        self.assertAlmostEqual(float(tracker.objects[0].bbox[0]), 24, delta=1.5)


class TTSTests(unittest.TestCase):
    def test_synthesize_requests_low_latency_wav(self):
        response = Mock()
        response.read.return_value = b"RIFF-test"
        client = Mock()
        client.audio.speech.create.return_value = response
        tts = SpeechSynthesizer(
            client=client,
            config=TTSConfig(model="gpt-4o-mini-tts", voice="marin"),
        )

        wav = tts.synthesize("어느 나사를 말씀하시나요?")

        self.assertEqual(wav, b"RIFF-test")
        request = client.audio.speech.create.call_args.kwargs
        self.assertEqual(request["response_format"], "wav")
        self.assertEqual(request["voice"], "marin")


if __name__ == "__main__":
    unittest.main()
