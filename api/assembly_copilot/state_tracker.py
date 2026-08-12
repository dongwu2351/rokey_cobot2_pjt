from __future__ import annotations

import time
import threading
import uuid
from typing import Any

from .models import AssemblyManual, AssemblyObservation, AssemblyState


class AssemblyStateTracker:
    """Deterministic state machine; an LLM never marks a step complete."""

    def __init__(self, manual: AssemblyManual, *, session_id: str | None = None) -> None:
        self.manual = manual
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.index = 0
        self.completed: list[str] = []
        self.user_confirmed: list[str] = []
        self.verified_completed: list[str] = []
        self.progress_update_source = "INITIAL"
        self.observation: AssemblyObservation | None = None
        self._manual_confirmed: set[str] = set()
        self.lock = threading.RLock()

    @property
    def step(self):
        return self.manual.steps[self.index] if self.index < len(self.manual.steps) else None

    def update(self, observation: AssemblyObservation) -> AssemblyState:
        with self.lock:
            self.observation = observation
            return self.snapshot()

    def confirm_current_step(self) -> AssemblyState:
        with self.lock:
            if self.step is not None:
                self._manual_confirmed.add(self.step.id)
                if self.step.id not in self.user_confirmed:
                    self.user_confirmed.append(self.step.id)
                self.progress_update_source = "USER_CONFIRMED"
            state = self.snapshot()
            if self.step is not None and not state.unsatisfied_conditions:
                self.completed.append(self.step.id)
                self.index += 1
                state = self.snapshot()
            return state

    def previous_step(self) -> AssemblyState:
        with self.lock:
            if self.index > 0:
                self.index -= 1
                step_id = self.manual.steps[self.index].id
                self.completed = [value for value in self.completed if value != step_id]
                self.user_confirmed = [value for value in self.user_confirmed
                                       if value != step_id]
                self.verified_completed = [value for value in self.verified_completed
                                           if value != step_id]
                self._manual_confirmed.discard(step_id)
            return self.snapshot()

    def select_step_number(self, number: int, *, source: str | None = None) -> AssemblyState:
        """Select a declared manual step without claiming any step is complete."""
        with self.lock:
            for index, step in enumerate(self.manual.steps):
                if step.order == number:
                    self.index = index
                    if source is not None:
                        self.progress_update_source = source
                    return self.snapshot()
        raise ValueError(f"매뉴얼에 {number}단계가 없습니다")

    def reopen_step(self, number: int, *, source: str = "USER_REOPENED") -> AssemblyState:
        """Return to an incomplete step and revoke completion from it onward."""
        with self.lock:
            target_index = next(
                (index for index, step in enumerate(self.manual.steps)
                 if step.order == number), None)
            if target_index is None:
                raise ValueError(f"매뉴얼에 {number}단계가 없습니다")
            retained = {step.id for step in self.manual.steps[:target_index]}
            self.completed = [value for value in self.completed if value in retained]
            self.user_confirmed = [value for value in self.user_confirmed if value in retained]
            self.verified_completed = [value for value in self.verified_completed
                                       if value in retained]
            self._manual_confirmed.intersection_update(retained)
            self.index = target_index
            self.progress_update_source = source
            return self.snapshot()

    def restore(self, current_step_id: str | None,
                completed_steps: list[str] | tuple[str, ...],
                user_confirmed_steps: list[str] | tuple[str, ...] = (),
                verified_completed_steps: list[str] | tuple[str, ...] = (),
                progress_update_source: str = "RESTORED") -> AssemblyState:
        """Restore only IDs that still exist in the currently loaded manual."""
        with self.lock:
            valid = {step.id for step in self.manual.steps}
            self.completed = [value for value in completed_steps if value in valid]
            self.user_confirmed = [value for value in user_confirmed_steps if value in valid]
            self.verified_completed = [value for value in verified_completed_steps
                                       if value in valid]
            # Old saved sessions only had completed_steps. Preserve their
            # position without falsely upgrading them to vision-verified.
            if not self.user_confirmed and self.completed:
                self.user_confirmed = list(self.completed)
            self.progress_update_source = progress_update_source
            if current_step_id is None:
                self.index = len(self.manual.steps)
            else:
                self.index = next(
                    (index for index, step in enumerate(self.manual.steps)
                     if step.id == current_step_id), 0)
            return self.snapshot()

    def complete_through(self, number: int, *, verified: bool = False,
                         source: str | None = None) -> AssemblyState:
        """Advance the work pointer while retaining the evidence provenance."""
        with self.lock:
            matching = [index for index, step in enumerate(self.manual.steps)
                        if step.order <= number]
            if not matching:
                raise ValueError(f"매뉴얼에 {number}단계가 없습니다")
            last = max(matching)
            if verified:
                self.verified_completed = [step.id for step in self.manual.steps[:last + 1]]
                # Verification may trail an explicit operator checkpoint. Do
                # not move the active work pointer backwards merely because a
                # later camera check verified fewer steps.
                claimed_indexes = [index for index, step in enumerate(self.manual.steps)
                                   if step.id in self.user_confirmed]
                active_last = max([last, *claimed_indexes])
                self.completed = [step.id for step in self.manual.steps[:active_last + 1]]
                self.progress_update_source = "VISION_VERIFIED"
            else:
                self.completed = [step.id for step in self.manual.steps[:last + 1]]
                self.user_confirmed = list(self.completed)
                self._manual_confirmed.update(self.completed)
                self.progress_update_source = "USER_CONFIRMED"
                active_last = last
            if source is not None:
                self.progress_update_source = source
            self.index = active_last + 1
            return self.snapshot()

    def snapshot(self) -> AssemblyState:
        step = self.step
        if step is None:
            verification = ("VERIFIED" if self.completed
                            and set(self.completed).issubset(self.verified_completed)
                            else "USER_CONFIRMED")
            return AssemblyState(
                self.manual.manual_id, self.session_id, None, self.index,
                "COMPLETED", 1.0, tuple(self.completed), (), (), (),
                user_confirmed_steps=tuple(self.user_confirmed),
                verified_completed_steps=tuple(self.verified_completed),
                progress_update_source=self.progress_update_source,
                verification_status=verification)
        satisfied, unsatisfied = [], []
        for condition in step.completion_conditions:
            ok, label = self._evaluate(condition)
            (satisfied if ok else unsatisfied).append(label)
        warnings = tuple(w for condition in step.error_conditions
                         if (w := self._evaluate_error(condition)) is not None)
        visibility = self.observation.visibility if self.observation else "UNKNOWN"
        confidence = 0.0 if visibility in {"UNKNOWN", "OCCLUDED"} else 1.0
        if step.completion_conditions:
            confidence *= len(satisfied) / len(step.completion_conditions)
        return AssemblyState(
            self.manual.manual_id, self.session_id, step.id, self.index,
            "WARNING" if warnings else "IN_PROGRESS", confidence,
            tuple(self.completed), tuple(satisfied), tuple(unsatisfied), warnings,
            self.observation.evidence_image if self.observation else None,
            tuple(self.user_confirmed), tuple(self.verified_completed),
            self.progress_update_source,
            ("USER_ASSISTED" if self.progress_update_source == "USER_ASSISTED" else
             "VERIFIED" if self.completed
             and set(self.completed).issubset(self.verified_completed)
             else "USER_CONFIRMED" if self.user_confirmed else "UNVERIFIED"),
        )

    def _evaluate(self, condition: dict[str, Any]) -> tuple[bool, str]:
        kind = str(condition.get("type", "TODO"))
        # Human-facing diagnostics must describe the condition, not leak schema
        # identifiers such as `operator_confirmed`.
        label = str(condition.get("description") or condition.get("id") or kind)
        obs = self.observation
        if kind == "manual_confirmation":
            return bool(self.step and self.step.id in self._manual_confirmed), label
        if obs is None or obs.visibility == "OCCLUDED":
            return False, label
        if kind == "part_visible":
            part = condition.get("part")
            return any(obj.get("class") == part for obj in obs.objects), label
        if kind == "measurement":
            value = obs.measurements.get(str(condition.get("name")))
            if value is None:
                return False, label
            return _compare(float(value), condition), label
        if kind == "relation":
            return any(all(rel.get(key) == condition.get(key)
                           for key in ("subject", "relation", "object"))
                       for rel in obs.relations), label
        return False, f"{label} (평가기 미구현: {kind})"

    def _evaluate_error(self, condition: dict[str, Any]) -> dict[str, Any] | None:
        # Error conditions use the same predicates; a matching predicate means warning.
        matched, label = self._evaluate(condition)
        if not matched:
            return None
        return {"id": condition.get("id", label),
                "message": condition.get("message", "매뉴얼의 오류 조건이 감지됐습니다")}


def _compare(value: float, condition: dict[str, Any]) -> bool:
    if "max" in condition and value > float(condition["max"]):
        return False
    if "min" in condition and value < float(condition["min"]):
        return False
    if "expected" in condition:
        tolerance = float(condition.get("tolerance", 0.0))
        return abs(value - float(condition["expected"])) <= tolerance
    return True
