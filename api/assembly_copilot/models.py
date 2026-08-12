from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReferenceMedia:
    images: tuple[str, ...] = ()
    videos: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AssemblyStep:
    id: str
    order: int
    title: str
    instruction: str
    required_parts: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    preconditions: tuple[dict[str, Any], ...] = ()
    completion_conditions: tuple[dict[str, Any], ...] = ()
    error_conditions: tuple[dict[str, Any], ...] = ()
    visual_checks: tuple[dict[str, Any], ...] = ()
    visual_state: dict[str, Any] = field(default_factory=dict)
    references: ReferenceMedia = field(default_factory=ReferenceMedia)


@dataclass(frozen=True)
class AssemblyManual:
    manual_id: str
    product: str
    version: str
    description: str
    parts: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    steps: tuple[AssemblyStep, ...]


@dataclass(frozen=True)
class AssemblyObservation:
    timestamp_ms: int
    frame_id: str = "camera_color_optical_frame"
    objects: tuple[dict[str, Any], ...] = ()
    relations: tuple[dict[str, Any], ...] = ()
    measurements: dict[str, float] = field(default_factory=dict)
    visibility: str = "UNKNOWN"
    evidence_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssemblyState:
    manual_id: str
    session_id: str
    current_step_id: str | None
    current_step_index: int
    status: str
    confidence: float
    completed_steps: tuple[str, ...]
    satisfied_conditions: tuple[str, ...]
    unsatisfied_conditions: tuple[str, ...]
    warnings: tuple[dict[str, Any], ...]
    evidence_image: str | None = None
    user_confirmed_steps: tuple[str, ...] = ()
    verified_completed_steps: tuple[str, ...] = ()
    progress_update_source: str = "INITIAL"
    verification_status: str = "UNVERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
