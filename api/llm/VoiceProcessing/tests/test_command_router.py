from __future__ import annotations

import json
import time
import unittest
import warnings
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

from pydantic import ValidationError

from VoiceProcessing.command_models import (
    Action,
    Ambiguity,
    CommandContext,
    ExecutableAction,
    Decision,
    Intent,
    ModelCandidate,
)
from VoiceProcessing.command_router import (
    CommandConfig,
    CommandRouter,
    LLMCommandParser,
)
from VoiceProcessing.keyword_extraction import ExtractKeyword


def grounded_context(
    *object_names: str,
    revision: str = "vision-rev-1",
    timestamp_ms: int | None = None,
):
    return {
        "robot_state": "ready",
        "visible_objects": [
            {
                "id": f"{object_name}-{index}",
                "canonical_name": object_name,
                "snapshot_revision": revision,
            }
            for index, object_name in enumerate(object_names, start=1)
        ],
        "snapshot_revision": revision,
        "snapshot_timestamp_ms": (
            timestamp_ms if timestamp_ms is not None else round(time.time() * 1_000)
        ),
    }


def candidate(
    *,
    decision: Decision = Decision.READY,
    actions: list[Action] | None = None,
    ambiguity: Ambiguity = Ambiguity.NONE,
    question: str | None = None,
) -> ModelCandidate:
    return ModelCandidate(
        decision=decision,
        actions=actions or [],
        ambiguity=ambiguity,
        clarification_question=question,
    )


def action(
    intent: Intent,
    object_name: str | None = None,
    destination: str | None = None,
    object_query: str | None = None,
) -> Action:
    return Action(
        intent=intent,
        object=object_name,
        destination=destination,
        object_query=object_query,
    )


class FakeLLMParser:
    """Network-free LLM parser double that records every invocation."""

    def __init__(
        self,
        result: ModelCandidate | object | None = None,
        *,
        error: Exception | None = None,
        responder: Callable[[str, CommandContext | None], object] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.responder = responder
        self.calls: list[tuple[str, CommandContext | None]] = []

    def parse(self, utterance: str, context: CommandContext | None = None):
        self.calls.append((utterance, context))
        if self.error is not None:
            raise self.error
        if self.responder is not None:
            return self.responder(utterance, context)
        return self.result


class FastCommandRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeLLMParser(error=AssertionError("LLM must not be called"))
        self.router = CommandRouter(llm_parser=self.llm)

    def assert_action(
        self,
        result,
        *,
        intent: Intent,
        object_name: str | None,
        destination: str | None,
    ) -> None:
        self.assertEqual(result.decision, Decision.READY)
        self.assertEqual(len(result.actions), 1)
        parsed = result.actions[0]
        self.assertEqual(parsed.intent, intent)
        self.assertEqual(parsed.object, object_name)
        self.assertEqual(parsed.destination, destination)

    def test_stop_uses_fast_stop_without_llm(self) -> None:
        for utterance in (
            "멈춰",
            "멈춰 주세요",
            "로봇 정지",
            "긴급 정지해 주세요",
            "중지해",
            "그만해줘",
            "긴급 스톱",
            "stop",
        ):
            with self.subTest(utterance=utterance):
                result = self.router.parse_command(utterance)
                self.assertEqual(result.route, "FAST_STOP")
                self.assert_action(
                    result,
                    intent=Intent.STOP,
                    object_name=None,
                    destination=None,
                )
        self.assertEqual(self.llm.calls, [])

    def test_fetch_aliases_use_fast_rule_without_llm(self) -> None:
        for utterance, object_name in (
            ("해머 가져와", "hammer"),
            ("망치 좀 갖다 줘", "hammer"),
            ("햄머 가져와", "hammer"),
            ("screwdriver 주세요", "screwdriver"),
            ("스패너 건네줘", "wrench"),
            ("못 박는 도구 가져와", "hammer"),
            ("나사 돌리는 도구 주세요", "screwdriver"),
            ("구멍 뚫는 도구 가져다줘", "drill"),
        ):
            with self.subTest(utterance=utterance):
                result = self.router.parse_command(
                    utterance,
                    grounded_context(object_name),
                )
                self.assertEqual(result.route, "FAST_RULE")
                self.assertEqual(result.decision, Decision.READY)
                self.assertEqual(result.actions[0].intent, Intent.FETCH)
                self.assertIsNone(result.actions[0].destination)
                self.assertEqual(
                    result.actions[0].resolved_object_id,
                    f"{object_name}-1",
                )
        self.assertEqual(self.llm.calls, [])

    def test_move_aliases_use_fast_rule_without_llm(self) -> None:
        cases = (
            ("hammer를 pos1으로 가져와", "hammer", "pos1"),
            ("망치를 1번 위치에 둬", "hammer", "pos1"),
            ("드라이버를 이번 위치로 옮겨줘", "screwdriver", "pos2"),
            ("렌치는 pos 3에 놓아줘", "wrench", "pos3"),
        )
        for utterance, object_name, destination in cases:
            with self.subTest(utterance=utterance):
                result = self.router.parse_command(
                    utterance,
                    grounded_context(object_name),
                )
                self.assertEqual(result.route, "FAST_RULE")
                self.assert_action(
                    result,
                    intent=Intent.MOVE,
                    object_name=object_name,
                    destination=destination,
                )
        self.assertEqual(self.llm.calls, [])

    def test_negated_motion_is_rejected_without_llm(self) -> None:
        for utterance in (
            "해머 가져오지 마",
            "망치 옮기지 마",
            "드라이버 안 가져와",
            "멈추지 마",
        ):
            with self.subTest(utterance=utterance):
                result = self.router.parse_command(utterance)
                self.assertEqual(result.route, "SAFETY")
                self.assertEqual(result.decision, Decision.REJECT)
                self.assertEqual(result.ambiguity, Ambiguity.NEGATED)
        self.assertEqual(result.actions, ())
        self.assertEqual(self.llm.calls, [])

    def test_default_router_does_not_construct_openai_client_on_fast_path(self) -> None:
        with patch("VoiceProcessing.command_router.OpenAI") as openai_client:
            router = CommandRouter()
            result = router.parse_command(
                "해머 가져와",
                grounded_context("hammer"),
            )

        self.assertEqual(result.route, "FAST_RULE")
        self.assertEqual(result.decision, Decision.READY)
        openai_client.assert_not_called()

    def test_fast_validator_exception_fails_closed(self) -> None:
        class BrokenValidator:
            def validate(self, candidate, context):
                raise RuntimeError("validator failed")

        result = CommandRouter(
            llm_parser=self.llm,
            validator=BrokenValidator(),
        ).parse_command("해머 가져와", grounded_context("hammer"))

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.SYSTEM_ERROR)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.error_code, "RuntimeError")


class LLMCommandRouteTests(unittest.TestCase):
    def test_llm_request_uses_structured_output_and_low_latency_controls(self) -> None:
        expected = candidate(
            actions=[action(Intent.FETCH, "hammer", object_query="망치")]
        )

        class FakeResponses:
            def __init__(self) -> None:
                self.request = None

            def parse(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(output_parsed=expected)

        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        parser = LLMCommandParser(
            config=CommandConfig(
                model="gpt-5.6-terra",
                reasoning_effort="none",
                timeout_seconds=3.5,
            ),
            client=client,
        )
        context = CommandContext.model_validate(grounded_context("hammer"))

        parsed = parser.parse("못 박는 도구 가져와", context)

        self.assertIs(parsed, expected)
        request = responses.request
        self.assertEqual(request["model"], "gpt-5.6-terra")
        self.assertIs(request["text_format"], ModelCandidate)
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertEqual(request["max_output_tokens"], 300)
        self.assertFalse(request["store"])
        self.assertEqual(request["timeout"], 3.5)
        payload = json.loads(request["input"])
        self.assertEqual(payload["utterance"], "못 박는 도구 가져와")
        self.assertEqual(payload["context"]["snapshot_revision"], "vision-rev-1")

    def test_llm_client_disables_automatic_retries(self) -> None:
        with patch("VoiceProcessing.command_router.OpenAI") as openai_client:
            parser = LLMCommandParser(api_key="test-key")
            parser.client

        openai_client.assert_called_once_with(
            api_key="test-key",
            max_retries=0,
        )

    def test_nonstandard_command_uses_fake_llm_once(self) -> None:
        expected = candidate(
            actions=[action(Intent.FETCH, "hammer", object_query="못 박는 도구")]
        )
        llm = FakeLLMParser(expected)
        router = CommandRouter(llm_parser=llm)

        result = router.parse_command(
            "못 박는 도구를 하나 가져다줘",
            grounded_context("hammer"),
        )

        self.assertEqual(result.route, "LLM")
        self.assertEqual(result.decision, Decision.READY)
        self.assertEqual(result.actions[0].intent, expected.actions[0].intent)
        self.assertEqual(result.actions[0].object, expected.actions[0].object)
        self.assertEqual(result.actions[0].resolved_object_id, "hammer-1")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0][0], "못 박는 도구를 하나 가져다줘")

    def test_mapping_context_is_validated_and_forwarded_to_llm(self) -> None:
        def resolve_recent(
            _utterance: str, context: CommandContext | None
        ) -> ModelCandidate:
            self.assertIsNotNone(context)
            assert context is not None
            self.assertEqual(context.recent_object, "hammer")
            return candidate(
                actions=[action(Intent.FETCH, context.recent_object, object_query="그거")]
            )

        llm = FakeLLMParser(responder=resolve_recent)
        router = CommandRouter(llm_parser=llm)

        result = router.parse_command(
            "아까 쓰던 그거 가져와",
            {
                **grounded_context("hammer"),
                "recent_object": "hammer",
            },
        )

        self.assertEqual(result.route, "LLM")
        self.assertEqual(result.decision, Decision.READY)
        self.assertEqual(result.actions[0].object, "hammer")
        self.assertEqual(len(llm.calls), 1)

    def test_llm_exception_fails_closed(self) -> None:
        llm = FakeLLMParser(error=TimeoutError("deadline exceeded"))
        result = CommandRouter(llm_parser=llm).parse_command("그거 가져와")

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.SYSTEM_ERROR)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.error_code, "TimeoutError")

    def test_malformed_llm_result_fails_closed(self) -> None:
        llm = FakeLLMParser({"decision": "READY", "actions": "not-a-list"})
        result = CommandRouter(llm_parser=llm).parse_command("복잡한 명령")

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.SYSTEM_ERROR)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.error_code, "AttributeError")


class ValidatorSafetyTests(unittest.TestCase):
    def route(self, proposed: ModelCandidate, *, context=None):
        return CommandRouter(llm_parser=FakeLLMParser(proposed)).parse_command(
            "규칙 경로에 없는 복잡한 발화",
            context,
        )

    def test_unsupported_object_is_rejected(self) -> None:
        result = self.route(
            candidate(actions=[action(Intent.FETCH, "knife", object_query="칼")])
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.UNSUPPORTED)
        self.assertEqual(result.actions, ())

    def test_move_without_destination_requires_clarification(self) -> None:
        result = self.route(
            candidate(actions=[action(Intent.MOVE, "hammer", object_query="해머")])
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.ambiguity, Ambiguity.MISSING_DESTINATION)
        self.assertEqual(result.actions, ())
        self.assertTrue(result.clarification_question)

    def test_stop_mixed_with_motion_is_rejected(self) -> None:
        result = self.route(
            candidate(
                actions=[
                    action(Intent.STOP),
                    action(Intent.FETCH, "hammer", object_query="해머"),
                ]
            )
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.INVALID_COMMAND)
        self.assertEqual(result.actions, ())

    def test_too_many_actions_are_rejected(self) -> None:
        proposed_actions = [
            action(Intent.FETCH, "hammer", object_query="해머") for _ in range(5)
        ]
        result = self.route(candidate(actions=proposed_actions))

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.INVALID_COMMAND)
        self.assertEqual(result.actions, ())

    def test_object_description_without_grounding_requires_clarification(self) -> None:
        result = self.route(
            candidate(
                actions=[
                    action(
                        Intent.FETCH,
                        object_name=None,
                        object_query="빨간 손잡이가 달린 도구",
                    )
                ]
            )
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.ambiguity, Ambiguity.VISION_GROUNDING_REQUIRED)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.grounding_query, "빨간 손잡이가 달린 도구")

    def test_nonready_candidate_cannot_retain_executable_actions(self) -> None:
        result = self.route(
            candidate(
                decision=Decision.CLARIFY,
                actions=[action(Intent.FETCH, "hammer", object_query="해머")],
                ambiguity=Ambiguity.CONTEXT_REQUIRED,
                question="어느 해머인가요?",
            )
        )

        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.actions, ())

    def test_ready_candidate_with_negated_ambiguity_is_rejected(self) -> None:
        result = self.route(
            candidate(
                decision=Decision.READY,
                actions=[action(Intent.FETCH, "hammer", object_query="해머")],
                ambiguity=Ambiguity.NEGATED,
                question="가져오지 않을까요?",
            ),
            context=grounded_context("hammer"),
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.INVALID_COMMAND)
        self.assertEqual(result.actions, ())

    def test_nonready_robot_state_blocks_motion(self) -> None:
        result = self.route(
            candidate(actions=[action(Intent.FETCH, "hammer", object_query="해머")]),
            context={
                "robot_state": "emergency_stop",
                "visible_objects": [
                    {
                        "id": "hammer-1",
                        "canonical_name": "hammer",
                        "snapshot_revision": "vision-rev-1",
                    }
                ],
            },
        )

        self.assertNotEqual(result.decision, Decision.READY)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.route, "SAFETY")

    def test_missing_vision_snapshot_blocks_motion(self) -> None:
        result = self.route(
            candidate(actions=[action(Intent.FETCH, "hammer", object_query="해머")]),
            context={"robot_state": "ready"},
        )

        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.ambiguity, Ambiguity.MISSING_OBJECT)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.route, "SAFETY")

    def test_contextless_motion_is_rejected_but_contextless_stop_is_ready(self) -> None:
        motion = CommandRouter(
            llm_parser=FakeLLMParser(error=AssertionError("LLM must not be called"))
        ).parse_command("해머 가져와")
        stopped = CommandRouter(
            llm_parser=FakeLLMParser(error=AssertionError("LLM must not be called"))
        ).parse_command("멈춰")

        self.assertEqual(motion.route, "SAFETY")
        self.assertEqual(motion.decision, Decision.REJECT)
        self.assertEqual(motion.ambiguity, Ambiguity.CONTEXT_REQUIRED)
        self.assertEqual(motion.actions, ())
        self.assertEqual(stopped.route, "FAST_STOP")
        self.assertEqual(stopped.decision, Decision.READY)
        self.assertEqual(stopped.actions[0].intent, Intent.STOP)
        self.assertIsNone(stopped.actions[0].resolved_object_id)

    def test_contextless_llm_motion_is_also_rejected(self) -> None:
        llm = FakeLLMParser(
            candidate(actions=[action(Intent.FETCH, "hammer", object_query="해머")])
        )

        result = CommandRouter(llm_parser=llm).parse_command("비정형 해머 요청")

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.CONTEXT_REQUIRED)
        self.assertEqual(result.actions, ())

    def test_snapshot_identity_is_required_for_motion(self) -> None:
        result = self.route(
            candidate(actions=[action(Intent.FETCH, "hammer", object_query="해머")]),
            context={
                "robot_state": "ready",
                "visible_objects": [
                    {
                        "id": "hammer-1",
                        "canonical_name": "hammer",
                        "snapshot_revision": "vision-rev-1",
                    }
                ],
            },
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.CONTEXT_REQUIRED)
        self.assertEqual(result.actions, ())

    def test_stale_or_implausibly_future_snapshot_is_rejected(self) -> None:
        now_seconds = 10.0
        router = CommandRouter(
            llm_parser=FakeLLMParser(error=AssertionError("LLM must not be called")),
            wall_clock=lambda: now_seconds,
        )

        for timestamp_ms in (6_999, 10_251):
            with self.subTest(timestamp_ms=timestamp_ms):
                result = router.parse_command(
                    "해머 가져와",
                    grounded_context("hammer", timestamp_ms=timestamp_ms),
                )
                self.assertEqual(result.route, "SAFETY")
                self.assertEqual(result.decision, Decision.REJECT)
                self.assertEqual(result.ambiguity, Ambiguity.STALE_CONTEXT)
        self.assertEqual(result.actions, ())

    def test_duplicate_visible_object_ids_fail_closed(self) -> None:
        result = CommandRouter(
            llm_parser=FakeLLMParser(error=AssertionError("LLM must not be called"))
        ).parse_command(
            "해머 가져와",
            {
                **grounded_context(),
                "visible_objects": [
                    {
                        "id": "duplicate",
                        "canonical_name": "hammer",
                        "snapshot_revision": "vision-rev-1",
                    },
                    {
                        "id": "duplicate",
                        "canonical_name": "screwdriver",
                        "snapshot_revision": "vision-rev-1",
                    },
                ],
            },
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.SYSTEM_ERROR)
        self.assertEqual(result.actions, ())

    def test_torn_snapshot_revision_fails_closed(self) -> None:
        result = CommandRouter(
            llm_parser=FakeLLMParser(error=AssertionError("LLM must not be called"))
        ).parse_command(
            "해머 가져와",
            {
                "robot_state": "ready",
                "visible_objects": [
                    {
                        "id": "hammer-1",
                        "canonical_name": "hammer",
                        "snapshot_revision": "camera-a",
                    }
                ],
                "snapshot_revision": "camera-b",
                "snapshot_timestamp_ms": round(time.time() * 1_000),
            },
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.SYSTEM_ERROR)
        self.assertEqual(result.actions, ())

    def test_same_resolved_object_cannot_appear_in_two_actions(self) -> None:
        result = self.route(
            candidate(
                actions=[
                    action(Intent.FETCH, "hammer", object_query="해머"),
                    action(
                        Intent.MOVE,
                        "hammer",
                        destination="pos1",
                        object_query="해머",
                    ),
                ]
            ),
            context=grounded_context("hammer"),
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.INVALID_COMMAND)
        self.assertEqual(result.actions, ())

    def test_invalid_context_fails_closed_instead_of_raising(self) -> None:
        router = CommandRouter(
            llm_parser=FakeLLMParser(
                candidate(actions=[action(Intent.FETCH, "hammer")])
            )
        )

        result = router.parse_command(
            "그거 가져와",
            {"visible_objects": "not-a-list"},
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertEqual(result.ambiguity, Ambiguity.SYSTEM_ERROR)
        self.assertEqual(result.actions, ())


class StructuredOutputBoundaryTests(unittest.TestCase):
    def test_llm_candidate_schema_cannot_contain_resolved_object_id(self) -> None:
        schema = json.dumps(ModelCandidate.model_json_schema(), sort_keys=True)
        self.assertNotIn("resolved_object_id", schema)
        self.assertIn(
            "resolved_object_id",
            json.dumps(ExecutableAction.model_json_schema(), sort_keys=True),
        )

        with self.assertRaises(ValidationError):
            Action(
                intent=Intent.FETCH,
                object="hammer",
                destination=None,
                object_query="해머",
                resolved_object_id="forged-id",
            )

    def test_validated_context_result_and_actions_are_immutable(self) -> None:
        context = CommandContext.model_validate(grounded_context("hammer"))
        self.assertIsInstance(context.visible_objects, tuple)

        with self.assertRaises(ValidationError):
            context.snapshot_revision = "forged-revision"
        with self.assertRaises(AttributeError):
            context.visible_objects.append(
                context.visible_objects[0]
            )
        with self.assertRaises(TypeError):
            context.visible_objects[0].attributes["color"] = "red"
        self.assertEqual(
            context.model_dump(mode="json")["visible_objects"][0]["attributes"],
            {},
        )

        result = CommandRouter(
            llm_parser=FakeLLMParser(error=AssertionError("LLM must not be called"))
        ).parse_command("해머 가져와", context)
        self.assertIsInstance(result.actions, tuple)

        with self.assertRaises(ValidationError):
            result.ambiguity = Ambiguity.NEGATED
        with self.assertRaises(ValidationError):
            result.actions[0].object = "knife"
        with self.assertRaises(AttributeError):
            result.actions.append(result.actions[0])


class ContextGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeLLMParser(error=AssertionError("LLM must not be called"))
        self.router = CommandRouter(llm_parser=self.llm)

    def test_unique_visible_object_allows_fast_command(self) -> None:
        timestamp_ms = round(time.time() * 1_000)
        result = self.router.parse_command(
            "해머 가져와",
            grounded_context(
                "hammer",
                revision="camera-42",
                timestamp_ms=timestamp_ms,
            ),
        )

        self.assertEqual(result.route, "FAST_RULE")
        self.assertEqual(result.decision, Decision.READY)
        self.assertEqual(result.actions[0].object, "hammer")
        self.assertIsInstance(result.actions[0], ExecutableAction)
        self.assertEqual(result.actions[0].resolved_object_id, "hammer-1")
        self.assertEqual(result.snapshot_revision, "camera-42")
        self.assertEqual(result.snapshot_timestamp_ms, timestamp_ms)
        self.assertEqual(self.llm.calls, [])

    def test_missing_visible_object_requires_clarification(self) -> None:
        result = self.router.parse_command(
            "해머 가져와",
            grounded_context(),
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.ambiguity, Ambiguity.MISSING_OBJECT)
        self.assertEqual(result.actions, ())
        self.assertEqual(result.grounding_query, "해머")
        self.assertEqual(self.llm.calls, [])

    def test_multiple_visible_objects_require_clarification(self) -> None:
        result = self.router.parse_command(
            "해머 가져와",
            {
                **grounded_context(),
                "visible_objects": [
                    {
                        "id": "hammer-1",
                        "canonical_name": "hammer",
                        "snapshot_revision": "vision-rev-1",
                    },
                    {
                        "id": "hammer-2",
                        "canonical_name": "hammer",
                        "snapshot_revision": "vision-rev-1",
                    },
                ]
            },
        )

        self.assertEqual(result.route, "SAFETY")
        self.assertEqual(result.decision, Decision.CLARIFY)
        self.assertEqual(result.ambiguity, Ambiguity.MULTIPLE_MATCHES)
        self.assertEqual(result.actions, ())
        self.assertEqual(self.llm.calls, [])


class LegacyAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeLLMParser(error=AssertionError("LLM must not be called"))
        self.extractor = ExtractKeyword(router=CommandRouter(llm_parser=self.llm))

    def assert_legacy_disabled(self, utterance: str, context=None) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.extractor.extract_keyword(utterance, context)

        self.assertIsNone(result)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertIn("disabled and always returns None", str(caught[0].message))
        self.assertEqual(self.llm.calls, [])

    def test_legacy_is_disabled_without_context(self) -> None:
        self.assert_legacy_disabled("해머 가져와")

    def test_legacy_is_disabled_even_with_grounded_context(self) -> None:
        self.assert_legacy_disabled(
            "hammer를 pos1으로 가져와",
            grounded_context("hammer"),
        )

    def test_legacy_stop_is_also_disabled(self) -> None:
        self.assert_legacy_disabled("멈춰")

    def test_structured_parse_command_preserves_execution_metadata(self) -> None:
        timestamp_ms = round(time.time() * 1_000)
        result = self.extractor.parse_command(
            "해머 가져와",
            grounded_context(
                "hammer",
                revision="legacy-migration-rev",
                timestamp_ms=timestamp_ms,
            ),
        )

        self.assertEqual(result.decision, Decision.READY)
        self.assertEqual(result.actions[0].resolved_object_id, "hammer-1")
        self.assertEqual(result.snapshot_revision, "legacy-migration-rev")
        self.assertEqual(result.snapshot_timestamp_ms, timestamp_ms)
        self.assertEqual(self.llm.calls, [])


if __name__ == "__main__":
    unittest.main()
