"""Robot skills: the copilot's only doorway to physical robot behaviour.

Build the default stack with `create_manager(mode)`:
    mode="auto"  real bridge when rclpy is importable, mock otherwise
    mode="mock"  always the mock (UI/conversation testing)
    mode="ros"   require the real bridge
    mode="off"   no skills registered (requests fail politely)
"""
from __future__ import annotations

from .manager import RobotSkillManager
from .models import (FetchOptions, FetchRequest, FetchTarget, SkillFeedback,
                     SkillResult, resolve_object_class)
from .registry import SkillRegistry

__all__ = [
    "RobotSkillManager", "SkillRegistry", "FetchRequest", "FetchTarget",
    "FetchOptions", "SkillFeedback", "SkillResult", "resolve_object_class",
    "create_manager",
]


def create_manager(mode: str = "auto", *, on_feedback=None, on_result=None,
                   vision: str = "auto", manual=None, step_context=None
                   ) -> tuple[RobotSkillManager, str]:
    """(manager, resolved_mode)

    `vision` picks the backend that judges an inspection photograph:
    mock / openai / auto. Mock costs nothing, so the whole flow can be
    rehearsed before spending a token on it.

    `step_context` answers current_step() for the step being judged - pass
    TrackerStepContext(engine.tracker), never the static manual: the manual
    knows every step and therefore which one is being worked on, not."""
    registry = SkillRegistry()
    resolved = mode
    if mode == "off":
        return (RobotSkillManager(registry, on_feedback=on_feedback,
                                  on_result=on_result), "off")
    if mode in ("auto", "ros"):
        try:
            import rclpy  # noqa: F401
            from .fetch_object import FetchObjectSkill, QuickCommandSkill
            from .ros_bridge import WebcamPnPBridge
            bridge = WebcamPnPBridge()
            registry.register("fetch_object", FetchObjectSkill(bridge))
            registry.register("robot_home", QuickCommandSkill(
                bridge, "home", "홈 복귀", wait_for_home=True))
            registry.register("gripper_open", QuickCommandSkill(
                bridge, "gripper_open", "그리퍼 열기"))
            registry.register("gripper_close", QuickCommandSkill(
                bridge, "gripper_close", "그리퍼 닫기"))
            from .take_from_hand import TakeFromHandSkill
            registry.register("take_from_hand", TakeFromHandSkill(bridge))
            from .inspect_step import InspectStepSkill
            from .inspect_vision import build_analyser
            registry.register("inspect_step", InspectStepSkill(
                bridge, build_analyser(vision), manual=step_context))
            resolved = "ros"
        except Exception:
            if mode == "ros":
                raise
            resolved = "mock"
    if resolved == "mock" or mode == "mock":
        from .mock_fetch import MockFetchSkill, MockQuickSkill
        registry.register("fetch_object", MockFetchSkill())
        registry.register("robot_home", MockQuickSkill("홈 복귀"))
        registry.register("gripper_open", MockQuickSkill("그리퍼 열기"))
        registry.register("gripper_close", MockQuickSkill("그리퍼 닫기"))
        from .mock_fetch import MockInspectSkill
        from .inspect_vision import build_analyser
        registry.register("inspect_step",
                          MockInspectSkill(build_analyser(vision)))
        registry.register("take_from_hand", MockQuickSkill("손에서 받기"))
        resolved = "mock"
    return (RobotSkillManager(registry, on_feedback=on_feedback,
                              on_result=on_result), resolved)
