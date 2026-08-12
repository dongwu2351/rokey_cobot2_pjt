"""RobotSkillManager - the single authority over physical skill execution.

Exactly one physical skill runs at a time. Cancellation, timeout and the
feedback fan-out all live here; the copilot engine only submits semantic
requests and reads state.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

from .base import RobotSkill
from .models import FetchRequest, SkillFeedback, SkillResult
from .registry import SkillRegistry


class RobotSkillManager:
    def __init__(self, registry: SkillRegistry,
                 on_feedback: Callable[[SkillFeedback], None] | None = None,
                 on_result: Callable[[SkillResult], None] | None = None) -> None:
        self.registry = registry
        self.on_feedback = on_feedback
        self.on_result = on_result
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._active_request: FetchRequest | None = None
        self._active_skill: RobotSkill | None = None
        self._last_feedback: SkillFeedback | None = None
        self._last_result: SkillResult | None = None

    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def active_request_id(self) -> str | None:
        with self._lock:
            if self.busy and self._active_request is not None:
                return self._active_request.request_id
            return None

    def snapshot(self) -> dict:
        """Thread-safe view for UI/diagnostics."""
        with self._lock:
            feedback = self._last_feedback
            result = self._last_result
            request = self._active_request
            return {
                "busy": self.busy,
                "request_id": request.request_id if request else None,
                "query": request.target.query if request else None,
                "state": feedback.state if feedback else None,
                "message": feedback.message if feedback else "",
                "progress": feedback.progress if feedback else 0.0,
                "error_code": feedback.error_code if feedback else None,
                "last_outcome": result.outcome if result else None,
                "last_outcome_message": result.message if result else "",
            }

    # ------------------------------------------------------------------
    def submit(self, request: FetchRequest) -> tuple[bool, str]:
        """Start a skill. Refuses while another physical skill is running -
        two concurrent physical skills are never allowed."""
        with self._lock:
            if self.busy:
                active = self._active_request
                label = active.target.query if active else "다른 작업"
                return False, f"이미 '{label}' 작업을 수행 중입니다. 완료 후 다시 요청해 주세요."
            skill = self.registry.get(request.skill)
            if skill is None:
                return False, f"'{request.skill}' 스킬이 등록되어 있지 않습니다."
            self._cancel = threading.Event()
            self._active_request = request
            self._active_skill = skill
            self._last_result = None
            self._last_feedback = None
            self._thread = threading.Thread(
                target=self._run, args=(skill, request, self._cancel),
                name=f"robot-skill-{request.skill}", daemon=True)
            self._thread.start()
        return True, "요청을 접수했습니다."

    def cancel_active(self) -> bool:
        """Cancel the running skill. Also fires the skill's emergency stop so
        the physical stream halts NOW, not at the next poll."""
        with self._lock:
            skill = self._active_skill
            running = self.busy
            self._cancel.set()
        if running and skill is not None:
            try:
                skill.emergency_stop()
            except Exception:
                traceback.print_exc()
        return running

    # ------------------------------------------------------------------
    def _run(self, skill: RobotSkill, request: FetchRequest,
             cancel: threading.Event) -> None:
        deadline = time.monotonic() + request.options.timeout_seconds
        timer = threading.Timer(
            request.options.timeout_seconds, cancel.set)
        timer.daemon = True
        timer.start()

        def feedback(item: SkillFeedback) -> None:
            with self._lock:
                self._last_feedback = item
            if self.on_feedback is not None:
                try:
                    self.on_feedback(item)
                except Exception:
                    traceback.print_exc()

        try:
            result = skill.run(request, feedback, cancel)
        except Exception as exc:  # a skill bug must not kill the manager
            traceback.print_exc()
            result = SkillResult(
                request_id=request.request_id, outcome="FAILED",
                message=f"스킬 내부 오류: {exc}", error_code=None)
        finally:
            timer.cancel()
        if (result.outcome == "CANCELLED"
                and time.monotonic() >= deadline):
            result = SkillResult(
                request_id=request.request_id, outcome="FAILED",
                message="제한 시간 안에 작업을 마치지 못했습니다.",
                error_code="HANDOVER_TIMEOUT")
        with self._lock:
            self._last_result = result
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:
                traceback.print_exc()
