"""Abstract robot skill interface.

A skill's run() executes on the manager's worker thread. It must:
- emit feedback through the callback (never touch UI directly),
- poll cancel_event frequently and stop the PHYSICAL motion before returning
  a CANCELLED result,
- always return a terminal SkillResult (never raise for expected failures).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

from .models import FetchRequest, SkillFeedback, SkillResult

FeedbackCallback = Callable[[SkillFeedback], None]


class RobotSkill(ABC):
    name: str = "abstract"

    @abstractmethod
    def run(self, request: FetchRequest, feedback: FeedbackCallback,
            cancel_event: threading.Event) -> SkillResult:
        ...

    def emergency_stop(self) -> None:
        """Best-effort immediate physical stop, callable from ANY thread even
        while run() is mid-flight. Default: nothing (mock skills)."""
