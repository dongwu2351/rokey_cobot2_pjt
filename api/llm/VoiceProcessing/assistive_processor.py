from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .assistive_models import (
    AssistiveResult,
    AssistiveState,
    ComparisonOperator,
    PerceptionDecision,
    PerceptionPlan,
    SelectionPolicy,
    SpatialRelation,
    TrackedObject,
)
from .grounding import OpenVocabularyGrounder, SceneTracker
from .object_memory import ObjectMemory
from .situated_parser import SituatedCommandParser
from .conversation_router import ConversationRoute, ConversationRouter
from .visual_memory import VisualMemory


DEFAULT_MEMORY_PATH = Path(__file__).with_name("data") / "object_memory.sqlite3"
DEFAULT_VISUAL_MEMORY_PATH = Path(__file__).with_name("data") / "visual_memory.sqlite3"


class AssistiveCommandProcessor:
    """Ground free-form commands without granting permission to move a robot."""

    def __init__(
        self,
        *,
        parser: SituatedCommandParser,
        grounder: OpenVocabularyGrounder,
        memory: ObjectMemory | None = None,
        visual_memory: VisualMemory | None = None,
        tracker: SceneTracker | None = None,
        speaker: Callable[[str], None] | None = None,
        conversation_router: ConversationRouter | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.parser = parser
        self.grounder = grounder
        self.memory = memory or ObjectMemory(DEFAULT_MEMORY_PATH)
        self.visual_memory = visual_memory or VisualMemory(DEFAULT_VISUAL_MEMORY_PATH)
        self.tracker = tracker or SceneTracker()
        self.speaker = speaker
        self.conversation_router = conversation_router
        self.clock = clock
        self.focused_object_id: str | None = None
        self.pending_plan: PerceptionPlan | None = None

    def set_focus(self, object_id: str | None) -> None:
        if object_id is not None and self.tracker.get(object_id) is None:
            raise ValueError(f"unknown focused object id: {object_id}")
        self.focused_object_id = object_id

    def process(self, utterance: str, image: Any) -> AssistiveResult:
        """Process a command against an image or a latest-image callback.

        A callback is resolved immediately before GPU inference, after any LLM
        parsing, so a slow semantic parse cannot make the grounded frame stale.
        """
        total_started = self.clock()
        parse_started = self.clock()
        try:
            if self.conversation_router is not None:
                conversation = self.conversation_router.route(
                    utterance,
                    capabilities=(
                        "ground_object",
                        "resolve_reference",
                        "illuminate_region",
                        "move_robot",
                        "grasp_object",
                        "place_object",
                        "speak",
                        "ask_clarification",
                    ),
                    context={
                        "focused_object": (
                            self.tracker.get(self.focused_object_id).model_dump(mode="json")
                            if self.focused_object_id and self.tracker.get(self.focused_object_id)
                            else None
                        ),
                        "visible_objects": [
                            item.model_dump(mode="json") for item in self.tracker.visible()
                        ],
                    },
                )
                if conversation.route not in (
                    ConversationRoute.ROBOT_COMMAND,
                    ConversationRoute.STOP,
                ):
                    conversation_image = None
                    if conversation.route == ConversationRoute.DESCRIBE:
                        conversation_image = image() if callable(image) else image
                    reply = conversation.reply
                    if reply is None and conversation.route != ConversationRoute.CLARIFY:
                        reply = self.conversation_router.respond(
                            utterance,
                            route=conversation.route,
                            context={"steps": [step.model_dump(mode="json") for step in conversation.steps]},
                            image=conversation_image,
                        )
                    if reply and self.speaker is not None:
                        self.speaker(reply)
                    return AssistiveResult(
                        state=AssistiveState.CONVERSATION,
                        conversation_route=conversation.route.value,
                        reply=reply,
                        clarification_question=conversation.clarification_question,
                        robot_action_allowed=False,
                        llm_used=True,
                        timings_ms={
                            "parse": round((self.clock() - parse_started) * 1_000, 3),
                            "vision": 0.0,
                            "total": round((self.clock() - total_started) * 1_000, 3),
                        },
                    )
            focus = self.tracker.get(self.focused_object_id)
            remembered = self.memory.recent(limit=20)
            plan = self.parser.parse(
                utterance,
                focus=focus.model_dump(mode="json") if focus else None,
                remembered=remembered,
                visible_summary=[
                    item.model_dump(mode="json") for item in self.tracker.visible()
                ],
                pending_plan=self.pending_plan,
                conversation_history=(
                    self.conversation_router.history
                    if self.conversation_router is not None
                    else ()
                ),
            )
            llm_used = getattr(self.parser, "last_used_llm", None)
            parse_ms = (self.clock() - parse_started) * 1_000
            if plan.decision == PerceptionDecision.STOP:
                self.pending_plan = None
                return self._result(
                    AssistiveState.STOPPED,
                    total_started,
                    plan=plan,
                    parse_ms=parse_ms,
                )
            clarification_only = plan.decision == PerceptionDecision.CLARIFY
            if clarification_only and plan.grounding_query is None:
                return self._clarify(plan, total_started, parse_ms=parse_ms)
            if plan.decision not in (
                PerceptionDecision.GROUND,
                PerceptionDecision.CLARIFY,
            ):
                return self._result(
                    AssistiveState.REJECTED,
                    total_started,
                    plan=plan,
                    parse_ms=parse_ms,
                )

            queries = plan.grounding_queries
            if not queries:
                raise RuntimeError("grounding plan has no query")
            primary_query = queries[0]
            memory_hit = False
            recalled = self.memory.recall(plan.target_category or primary_query)
            if recalled is not None:
                query = recalled.grounding_prompt
                memory_hit = True
            else:
                query = ". ".join(queries)

            vision_started = self.clock()
            grounding_image = image() if callable(image) else image
            detections = self.grounder.detect(grounding_image, query)
            tracked = self.tracker.update(detections)
            vision_ms = (self.clock() - vision_started) * 1_000
            if not tracked:
                question = f"'{primary_query}'에 해당하는 물체를 찾지 못했습니다. 위치나 특징을 더 알려주시겠어요?"
                return self._clarify(
                    plan,
                    total_started,
                    question=question,
                    state=AssistiveState.NOT_FOUND,
                    parse_ms=parse_ms,
                    vision_ms=vision_ms,
                    memory_hit=memory_hit,
                    query=query,
                )

            if clarification_only:
                return self._clarify(
                    plan,
                    total_started,
                    detections=tracked,
                    parse_ms=parse_ms,
                    vision_ms=vision_ms,
                    memory_hit=memory_hit,
                    query=query,
                )

            spatial = self._spatial_select(plan, tracked)
            visual_memory_hit = False
            visual_similarity = None
            visual_match = None
            if plan.target_category:
                try:
                    visual_match = self.visual_memory.best_match(
                        plan.target_category, grounding_image, tracked
                    )
                except (TypeError, ValueError):
                    visual_match = None
                if visual_match is not None:
                    visual_memory_hit = True
                    visual_similarity = visual_match[1].similarity
            if spatial is not None and plan.comparison is not None:
                self.focused_object_id = spatial.id
            if spatial is not None and plan.comparison is None:
                selected, geometry_verified, question = spatial, False, None
            elif visual_match is not None and plan.comparison is None:
                selected, geometry_verified, question = visual_match[0], False, None
            else:
                selected, geometry_verified, question = self._select(plan, tracked)
            if selected is None:
                return self._clarify(
                    plan,
                    total_started,
                    question=question,
                    detections=tracked,
                    parse_ms=parse_ms,
                    vision_ms=vision_ms,
                    memory_hit=memory_hit,
                    query=query,
                )

            canonical = plan.target_category or selected.label
            self.memory.remember(
                canonical,
                grounding_prompt=selected.label,
                aliases=tuple(
                    alias
                    for alias in (selected.label, plan.source_object_expression)
                    if alias
                ),
                attributes={
                    key: value
                    for key, value in selected.attributes.items()
                    if key not in {"x", "y", "z", "pose", "coordinates"}
                },
            )
            # Keep an appearance sample for future candidate ranking. The
            # current frame is still required; no stale coordinates are saved.
            try:
                self.visual_memory.remember(canonical, grounding_image, selected.bbox)
            except (TypeError, ValueError):
                # Visual memory is an optimization; grounding must still work
                # when a backend supplies a non-image object.
                pass
            self.pending_plan = None
            return self._result(
                AssistiveState.GROUNDED,
                total_started,
                plan=plan,
                detections=tracked,
                selected_object_id=selected.id,
                memory_hit=memory_hit,
                grounding_query=query,
                geometry_verified=geometry_verified,
                visual_memory_hit=visual_memory_hit,
                visual_similarity=visual_similarity,
                llm_used=llm_used,
                parse_ms=parse_ms,
                vision_ms=vision_ms,
            )
        except Exception as exc:
            return self._result(
                AssistiveState.ERROR,
                total_started,
                error=type(exc).__name__,
                error_detail=str(exc)[:500],
            )

    def _select(
        self,
        plan: PerceptionPlan,
        objects: Sequence[TrackedObject],
    ) -> tuple[TrackedObject | None, bool, str | None]:
        focus = self.tracker.get(self.focused_object_id)
        if plan.comparison is None:
            if len(objects) == 1:
                return objects[0], False, None
            if plan.reference_expression and focus and any(
                item.id == focus.id for item in objects
            ):
                return focus, False, None
            return None, False, f"후보가 {len(objects)}개입니다. 위치나 특징을 더 알려주시겠어요?"

        if focus is None:
            return (
                None,
                False,
                "비교 기준 물체를 왼쪽, 오른쪽, 위쪽, 아래쪽 중 어디로 할까요?",
            )
        candidates = [item for item in objects if item.id != focus.id]
        if not candidates:
            return None, False, "기준 물체와 비교할 다른 후보를 찾지 못했습니다."

        attribute = plan.comparison.attribute
        reference_value, reference_verified = self._metric(focus, attribute)
        measured = [
            (item, *self._metric(item, attribute))
            for item in candidates
        ]
        operator = plan.comparison.operator
        if operator == ComparisonOperator.GREATER_THAN:
            eligible = [item for item in measured if item[1] > reference_value]
        elif operator == ComparisonOperator.LESS_THAN:
            eligible = [item for item in measured if item[1] < reference_value]
        else:
            tolerance = max(2.0, abs(reference_value) * 0.05)
            eligible = [
                item for item in measured if abs(item[1] - reference_value) <= tolerance
            ]
        if not eligible:
            return None, False, "요청하신 비교 조건에 맞는 물체를 찾지 못했습니다."

        policy = plan.comparison.selection_policy
        if policy in (SelectionPolicy.NEAREST_GREATER, SelectionPolicy.NEAREST_LESS):
            selected, _, verified = min(
                eligible, key=lambda item: abs(item[1] - reference_value)
            )
        elif policy == SelectionPolicy.HIGHEST:
            selected, _, verified = max(eligible, key=lambda item: item[1])
        else:
            selected, _, verified = min(eligible, key=lambda item: item[1])
        return selected, bool(reference_verified and verified), None

    @staticmethod
    def _spatial_select(
        plan: PerceptionPlan, objects: Sequence[TrackedObject]
    ) -> TrackedObject | None:
        relation = plan.spatial_relation
        if relation is None or not objects:
            return None
        if relation == SpatialRelation.LEFTMOST:
            return min(objects, key=lambda item: item.bbox.center[0])
        if relation == SpatialRelation.RIGHTMOST:
            return max(objects, key=lambda item: item.bbox.center[0])
        if relation == SpatialRelation.TOPMOST:
            return min(objects, key=lambda item: item.bbox.center[1])
        if relation == SpatialRelation.BOTTOMMOST:
            return max(objects, key=lambda item: item.bbox.center[1])
        center_x = max(item.bbox.x2 for item in objects) / 2
        center_y = max(item.bbox.y2 for item in objects) / 2
        if relation == SpatialRelation.CENTER:
            return min(
                objects,
                key=lambda item: (item.bbox.center[0] - center_x) ** 2
                + (item.bbox.center[1] - center_y) ** 2,
            )
        # In the absence of depth, screen-space size is a safe visual proxy
        # only; the result remains geometry_verified=False.
        if relation == SpatialRelation.NEAREST:
            return max(objects, key=lambda item: item.bbox.long_side_px)
        if relation == SpatialRelation.FARTHEST:
            return min(objects, key=lambda item: item.bbox.long_side_px)
        return None

    @staticmethod
    def _metric(item: TrackedObject, attribute: str) -> tuple[float, bool]:
        key = f"{attribute}_mm"
        measured = item.attributes.get(key)
        if isinstance(measured, (int, float)):
            return float(measured), True
        # Pixel size is suitable for visual proposal only, never robot execution.
        return item.bbox.long_side_px, False

    def _clarify(
        self,
        plan: PerceptionPlan,
        total_started: float,
        *,
        question: str | None = None,
        state: AssistiveState = AssistiveState.CLARIFICATION_REQUIRED,
        detections: Sequence[TrackedObject] = (),
        parse_ms: float = 0.0,
        vision_ms: float = 0.0,
        memory_hit: bool = False,
        query: str | None = None,
    ) -> AssistiveResult:
        spoken = question or plan.clarification_question or "조금 더 구체적으로 말씀해 주세요."
        self.pending_plan = plan
        if self.speaker is not None:
            self.speaker(spoken)
        return self._result(
            state,
            total_started,
            plan=plan,
            detections=detections,
            clarification_question=spoken,
            parse_ms=parse_ms,
            vision_ms=vision_ms,
            memory_hit=memory_hit,
            grounding_query=query,
            llm_used=getattr(self.parser, "last_used_llm", None),
        )

    def _result(
        self,
        state: AssistiveState,
        total_started: float,
        *,
        plan: PerceptionPlan | None = None,
        detections: Sequence[TrackedObject] = (),
        selected_object_id: str | None = None,
        clarification_question: str | None = None,
        memory_hit: bool = False,
        grounding_query: str | None = None,
        geometry_verified: bool = False,
        visual_memory_hit: bool = False,
        visual_similarity: float | None = None,
        llm_used: bool | None = None,
        parse_ms: float = 0.0,
        vision_ms: float = 0.0,
        error: str | None = None,
        error_detail: str | None = None,
    ) -> AssistiveResult:
        return AssistiveResult(
            state=state,
            plan=plan,
            detections=tuple(detections),
            selected_object_id=selected_object_id,
            clarification_question=clarification_question,
            memory_hit=memory_hit,
            grounding_query=grounding_query,
            geometry_verified=geometry_verified,
            visual_memory_hit=visual_memory_hit,
            visual_similarity=visual_similarity,
            llm_used=llm_used,
            executable=False,
            timings_ms={
                "parse": round(parse_ms, 3),
                "vision": round(vision_ms, 3),
                "total": round((self.clock() - total_started) * 1_000, 3),
            },
            error=error,
            error_detail=error_detail,
        )
