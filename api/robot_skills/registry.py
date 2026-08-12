"""Skill name -> implementation registry."""
from __future__ import annotations

from .base import RobotSkill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, RobotSkill] = {}

    def register(self, name: str, skill: RobotSkill) -> None:
        self._skills[name] = skill

    def get(self, name: str) -> RobotSkill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills)
