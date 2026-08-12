"""What the inspection model needs to know about the step being judged.

`InspectStepSkill` asks its `manual` for `current_step()`. Nothing in the
copilot has that method - the skill was handed an `AssemblyManual`, which
only holds the static list of steps - so every call raised, was swallowed,
and the model judged the photograph with no idea what the step required. It
said so, correctly and uselessly: "5단계 기준 정보가 없어 확정할 수 없습니다".

This adapter closes that gap. It reads the LIVE tracker rather than a
snapshot taken at start-up, because the operator advances steps mid-session
and a step description frozen at launch is worse than none: it would have the
model confidently judging step 5 work against step 1's instructions.
"""
from __future__ import annotations

from pathlib import Path

RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def reference_image_for_step(step) -> str | None:
    """The photograph of this step DONE, if the manual carries one.

    Last raster reference wins: manuals put the finished state last, and the
    diagrams (SVG) cannot be sent to a vision model."""
    if step is None:
        return None
    for value in reversed(getattr(step.references, "images", ()) or ()):
        path = Path(value)
        if path.suffix.lower() in RASTER_SUFFIXES and path.is_file():
            return str(path)
    return None


class TrackerStepContext:
    """`current_step()` over the tracker the copilot is actually using."""

    def __init__(self, tracker) -> None:
        self._tracker = tracker

    def current_step(self) -> dict | None:
        step = getattr(self._tracker, "step", None)
        if step is None:
            return None
        total = len(getattr(self._tracker.manual, "steps", ()) or ())
        return {
            "order": step.order,
            "total": total,
            "title": step.title,
            "instruction": step.instruction,
            "completion": "; ".join(
                condition.description
                for condition in (step.completion_conditions or [])
                if getattr(condition, "description", "")),
            "reference_image": reference_image_for_step(step),
        }
