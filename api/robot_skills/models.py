"""Data contracts for robot skills, per ROBOT_SKILL_INTEGRATION_HANDOFF.md.

The LLM/copilot side only ever produces the semantic request (query, class,
attributes). Joints, TCP coordinates and velocities never appear here - the
physical skill node owns those.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

SKILL_STATES = (
    "ACCEPTED",
    "SEARCHING_TARGET",
    "AMBIGUOUS_TARGET",
    "TARGET_LOCKED",
    "PLANNING_PREGRASP",
    "WAITING_FOR_CONFIRMATION",
    "MOVING_SERVOJ",
    "ALIGNING_SPEEDL",
    "GRASPING",
    "VERIFYING_GRASP",
    "TRACKING_HAND",
    "WAITING_FOR_HANDOVER",
    "RELEASING",
    "RETREATING",
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "SAFETY_STOPPED",
)

TERMINAL_STATES = {"SUCCEEDED", "CANCELLED", "FAILED", "SAFETY_STOPPED"}

ERROR_CODES = (
    "TARGET_NOT_FOUND",
    "TARGET_AMBIGUOUS",
    "TARGET_LOST",
    "INVALID_DEPTH",
    "NO_GRASP_CANDIDATE",
    "PATH_BLOCKED",
    "HUMAN_TOO_CLOSE",
    "CONTROL_HANDOFF_FAILED",
    "GRASP_FAILED",
    "HAND_NOT_STABLE",
    "HANDOVER_TIMEOUT",
    "DRIVER_DISCONNECTED",
    "SAFETY_STOP",
)


@dataclass
class FetchTarget:
    query: str
    class_name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    resolved_object_id: str | None = None


@dataclass
class FetchOptions:
    dry_run: bool = True
    require_confirmation: bool = True
    timeout_seconds: float = 90.0


@dataclass
class FetchRequest:
    target: FetchTarget
    options: FetchOptions = field(default_factory=FetchOptions)
    skill: str = "fetch_object"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    destination_type: str = "user_handover"

    def to_payload(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "request_id": self.request_id,
            "target": {
                "query": self.target.query,
                "class_name": self.target.class_name,
                "attributes": dict(self.target.attributes),
                "resolved_object_id": self.target.resolved_object_id,
            },
            "destination": {"type": self.destination_type},
            "options": {
                "dry_run": self.options.dry_run,
                "require_confirmation": self.options.require_confirmation,
                "timeout_seconds": self.options.timeout_seconds,
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FetchRequest":
        target = payload.get("target", {})
        options = payload.get("options", {})
        return cls(
            target=FetchTarget(
                query=target.get("query", ""),
                class_name=target.get("class_name"),
                attributes=dict(target.get("attributes", {})),
                resolved_object_id=target.get("resolved_object_id"),
            ),
            options=FetchOptions(
                dry_run=bool(options.get("dry_run", True)),
                require_confirmation=bool(options.get("require_confirmation", True)),
                timeout_seconds=float(options.get("timeout_seconds", 90.0)),
            ),
            skill=payload.get("skill", "fetch_object"),
            request_id=payload.get("request_id") or uuid.uuid4().hex,
            destination_type=payload.get("destination", {}).get(
                "type", "user_handover"
            ),
        )


@dataclass
class SkillFeedback:
    request_id: str
    state: str
    progress: float = 0.0
    message: str = ""
    requires_confirmation: bool = False
    recoverable: bool = True
    error_code: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in SKILL_STATES:
            raise ValueError(f"unknown skill state: {self.state}")
        if self.error_code is not None and self.error_code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.error_code}")

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass
class SkillResult:
    request_id: str
    outcome: str
    message: str
    resolved_object_id: str | None = None
    grasp_verified: bool = False
    handover_verified: bool = False
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in TERMINAL_STATES:
            raise ValueError(f"outcome must be terminal, got: {self.outcome}")
        if self.error_code is not None and self.error_code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.error_code}")


# Korean object words -> detector class names the physical skill understands.
OBJECT_CLASS_ALIASES = {
    "해머": "hammer",
    "망치": "hammer",
    # Live STT misrecognitions of spoken "해머" (2026-08-10: "해먹 가져와").
    # A literal hammock cannot appear in this cell, so the collision is safe.
    "해먹": "hammer",
    "함마": "hammer",
    "함머": "hammer",
    "햄머": "hammer",
    "hammer": "hammer",
    "드라이버": "screwdriver",
    "screwdriver": "screwdriver",
    "렌치": "wrench",
    "스패너": "wrench",
    "wrench": "wrench",
    "펜치": "pliers",
    "플라이어": "pliers",
    "pliers": "pliers",
    "드릴": "drill",
    "drill": "drill",
}


#: Longest alias first. "빨간 드라이버" contains "드라이버", so dictionary
#: order would decide which screwdriver the robot fetches - and a later edit
#: that reorders the aliases would silently change the robot's behaviour.
_ALIASES_BY_LENGTH = sorted(OBJECT_CLASS_ALIASES.items(),
                            key=lambda item: len(item[0]), reverse=True)


def resolve_object_class(text: str) -> str | None:
    lowered = text.lower()
    for alias, class_name in _ALIASES_BY_LENGTH:
        if alias in lowered:
            return class_name
    return None


def eul(word: str) -> str:
    """Object particle matching the final consonant ("해머를", "드라이버를").

    These strings are spoken aloud, and TTS reads a literal "해머을(를)" out
    as written - it sounds like a bug to the user."""
    word = (word or "").strip()
    if not word:
        return "을"
    last = word[-1]
    if "가" <= last <= "힣":
        return "을" if (ord(last) - 0xAC00) % 28 else "를"
    return "을"
