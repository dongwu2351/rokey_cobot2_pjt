from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class Intent(str, Enum):
    FETCH = "FETCH"
    MOVE = "MOVE"
    PLACE = "PLACE"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"


class Decision(str, Enum):
    READY = "READY"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"


class Ambiguity(str, Enum):
    NONE = "NONE"
    MISSING_OBJECT = "MISSING_OBJECT"
    MISSING_DESTINATION = "MISSING_DESTINATION"
    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"
    STALE_CONTEXT = "STALE_CONTEXT"
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
    VISION_GROUNDING_REQUIRED = "VISION_GROUNDING_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"
    NEGATED = "NEGATED"
    INVALID_COMMAND = "INVALID_COMMAND"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class RobotState(str, Enum):
    READY = "ready"
    BUSY = "busy"
    EMERGENCY_STOP = "emergency_stop"
    FAULT = "fault"
    UNKNOWN = "unknown"


class Action(BaseModel):
    """An untrusted semantic action proposed by a rule or the LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: Intent
    object: str | None = Field(max_length=64)
    destination: str | None = Field(max_length=64)
    object_query: str | None = Field(max_length=256)


class ExecutableAction(Action):
    """An action promoted by the local validator after perception grounding."""

    resolved_object_id: str | None = Field(default=None, min_length=1, max_length=128)


class ModelCandidate(BaseModel):
    """Only fields the LLM is allowed to propose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Decision
    actions: tuple[Action, ...]
    ambiguity: Ambiguity
    clarification_question: str | None


class VisibleObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    id: str = Field(min_length=1, max_length=128)
    canonical_name: str = Field(min_length=1, max_length=64)
    snapshot_revision: str = Field(min_length=1, max_length=128)
    location: str | None = Field(default=None, max_length=128)
    attributes: Mapping[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        max_length=16,
    )

    @field_validator("attributes")
    @classmethod
    def limit_attributes(cls, value):
        for key, item in value.items():
            if not key or len(key) > 64:
                raise ValueError("attribute names must contain 1-64 characters")
            if isinstance(item, str) and len(item) > 128:
                raise ValueError("attribute strings cannot exceed 128 characters")
        return MappingProxyType(dict(value))

    @field_serializer("attributes")
    def serialize_attributes(self, value):
        return dict(value)


class CommandContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visible_objects: tuple[VisibleObject, ...] | None = Field(
        default=None,
        max_length=50,
    )
    recent_object: str | None = Field(default=None, max_length=128)
    robot_state: RobotState = RobotState.UNKNOWN
    snapshot_revision: str | None = Field(default=None, min_length=1, max_length=128)
    snapshot_timestamp_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_unique_visible_object_ids(self):
        if self.visible_objects is None:
            return self
        object_ids = [item.id for item in self.visible_objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("visible object ids must be unique within a snapshot")
        if self.snapshot_revision is not None and any(
            item.snapshot_revision != self.snapshot_revision
            for item in self.visible_objects
        ):
            raise ValueError(
                "every visible object must belong to the context snapshot revision"
            )
        return self


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    decision: Decision
    actions: tuple[ExecutableAction, ...]
    ambiguity: Ambiguity
    clarification_question: str | None
    route: Literal["FAST_STOP", "FAST_RULE", "LLM", "SAFETY"]
    raw_utterance: str
    grounding_query: str | None = None
    snapshot_revision: str | None = Field(default=None, min_length=1, max_length=128)
    snapshot_timestamp_ms: int | None = Field(default=None, ge=0)
    latency_ms: float = 0.0
    error_code: str | None = None

    @model_validator(mode="after")
    def require_safe_execution_contract(self):
        if self.decision != Decision.READY:
            if self.actions:
                raise ValueError("non-ready commands cannot retain actions")
            if self.ambiguity == Ambiguity.NONE:
                raise ValueError("non-ready commands require an ambiguity reason")
            return self

        if self.ambiguity != Ambiguity.NONE or self.clarification_question is not None:
            raise ValueError("ready commands cannot be ambiguous or ask a question")
        if not self.actions:
            raise ValueError("ready commands require at least one action")

        stop_actions = [action for action in self.actions if action.intent == Intent.STOP]
        if stop_actions:
            action = stop_actions[0]
            if (
                len(self.actions) != 1
                or action.object is not None
                or action.destination is not None
                or action.object_query is not None
                or action.resolved_object_id is not None
            ):
                raise ValueError("STOP must be the only action and carry no motion data")
            return self

        for action in self.actions:
            if action.intent not in (Intent.FETCH, Intent.MOVE, Intent.PLACE):
                raise ValueError("ready motion contains an unsupported intent")
            if action.object is None or action.resolved_object_id is None:
                raise ValueError("ready motion requires an object and resolved id")
            if action.intent == Intent.FETCH and action.destination is not None:
                raise ValueError("FETCH cannot carry a destination")
            if action.intent in (Intent.MOVE, Intent.PLACE) and action.destination is None:
                raise ValueError("MOVE and PLACE require a destination")
        resolved_ids = [action.resolved_object_id for action in self.actions]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("ready motion actions require unique resolved object ids")
        if self.snapshot_revision is None or self.snapshot_timestamp_ms is None:
            raise ValueError(
                "ready motion commands require snapshot revision and timestamp"
            )
        return self
