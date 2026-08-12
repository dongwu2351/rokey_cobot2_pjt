from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .command_models import Intent


class PerceptionDecision(str, Enum):
    GROUND = "GROUND"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"
    STOP = "STOP"


class ComparisonOperator(str, Enum):
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    EQUAL = "EQUAL"


class SelectionPolicy(str, Enum):
    NEAREST_GREATER = "NEAREST_GREATER"
    NEAREST_LESS = "NEAREST_LESS"
    HIGHEST = "HIGHEST"
    LOWEST = "LOWEST"


class SpatialRelation(str, Enum):
    LEFTMOST = "LEFTMOST"
    RIGHTMOST = "RIGHTMOST"
    TOPMOST = "TOPMOST"
    BOTTOMMOST = "BOTTOMMOST"
    CENTER = "CENTER"
    NEAREST = "NEAREST"
    FARTHEST = "FARTHEST"


class ComparisonConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attribute: Literal["length", "width", "height", "size"]
    operator: ComparisonOperator
    reference_expression: str = Field(min_length=1, max_length=128)
    selection_policy: SelectionPolicy


class PerceptionPlan(BaseModel):
    """Untrusted semantic plan. It can request perception but never robot motion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: PerceptionDecision
    intent: Intent
    target_category: str | None = Field(default=None, max_length=128)
    target_description: str | None = Field(default=None, max_length=512)
    visual_query_alternatives: tuple[str, ...] = Field(
        default=(), max_length=3
    )
    source_object_expression: str | None = Field(default=None, max_length=128)
    spatial_relation: SpatialRelation | None = None
    destination: str | None = Field(default=None, max_length=128)
    reference_expression: str | None = Field(default=None, max_length=128)
    exclude_reference: bool = False
    comparison: ComparisonConstraint | None = None
    clarification_question: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_decision_contract(self):
        if self.decision == PerceptionDecision.GROUND:
            if self.intent not in (Intent.FETCH, Intent.MOVE, Intent.PLACE):
                raise ValueError("GROUND requires a supported manipulation intent")
            if not self.target_category and not self.target_description:
                raise ValueError("GROUND requires a target query")
            if self.clarification_question is not None:
                raise ValueError("GROUND cannot contain a clarification question")
        elif self.decision == PerceptionDecision.CLARIFY:
            if not self.clarification_question:
                raise ValueError("CLARIFY requires a question")
        elif self.decision == PerceptionDecision.STOP:
            if self.intent != Intent.STOP:
                raise ValueError("STOP decision requires STOP intent")
        return self

    @property
    def grounding_query(self) -> str | None:
        return self.target_description or self.target_category

    @property
    def grounding_queries(self) -> tuple[str, ...]:
        primary = self.target_description or self.target_category
        return tuple(
            dict.fromkeys(
                query.strip()
                for query in (primary, *self.visual_query_alternatives)
                if query and query.strip()
            )
        )


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def require_positive_area(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive area")
        return self

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def long_side_px(self) -> float:
        return max(self.x2 - self.x1, self.y2 - self.y1)


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=256)
    score: float = Field(ge=0, le=1)
    bbox: BoundingBox
    attributes: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


class TrackedObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    score: float = Field(ge=0, le=1)
    bbox: BoundingBox
    attributes: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    last_seen_ms: int = Field(ge=0)


class Point3D(BaseModel):
    """A point in a named metric coordinate frame (metres)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float
    z: float


class SpatialEstimate(BaseModel):
    """Optional 3D estimate produced after depth/calibration is available."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: Point3D
    frame: str = Field(min_length=1, max_length=64)
    method: Literal["DEPTH", "STEREO", "TRIANGULATION"]
    confidence: float = Field(ge=0, le=1)


class AssistiveState(str, Enum):
    GROUNDED = "GROUNDED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    NOT_FOUND = "NOT_FOUND"
    STOPPED = "STOPPED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    CONVERSATION = "CONVERSATION"


class AssistiveResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: AssistiveState
    plan: PerceptionPlan | None = None
    detections: tuple[TrackedObject, ...] = ()
    selected_object_id: str | None = None
    clarification_question: str | None = None
    memory_hit: bool = False
    visual_memory_hit: bool = False
    visual_similarity: float | None = None
    llm_used: bool | None = None
    grounding_query: str | None = None
    geometry_verified: bool = False
    spatial_estimate: SpatialEstimate | None = None
    grasp_point: Point3D | None = None
    source_camera_index: int | None = None
    conversation_route: str | None = None
    reply: str | None = None
    robot_action_allowed: bool = False
    executable: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
