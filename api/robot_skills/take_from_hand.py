"""Skill: take what the operator is holding and put it down somewhere free.

The mirror image of the handover, and the only manoeuvre where the gripper
closes near a person's fingers - so the physical app owns every safety rule
(steady-hand gate, a floor above the palm, abort on movement) and this skill
only narrates what it is doing and reports how it ended.
"""
from __future__ import annotations

import time

from .base import RobotSkill
from .models import FetchRequest, SkillFeedback, SkillResult

POLL_SEC = 0.1
TIMEOUT_SEC = 90.0

#: App state -> what the person should be told. Taking from a hand needs
#: narration more than any other skill: the operator has to know when to hold
#: still and when they can let go.
STATE_MESSAGES = {
    "TAKE_FROM_HAND": ("MOVING_SERVOJ", 0.4, "손을 펴서 그대로 들고 계세요."),
    "PLACING": ("RELEASING", 0.8, "빈 곳에 내려놓고 있습니다."),
    "HOMING": ("ACCEPTED", 0.1, "준비하고 있습니다."),
}


class TakeFromHandSkill(RobotSkill):
    name = "take_from_hand"
    label = "손에서 받기"

    def __init__(self, bridge):
        self.bridge = bridge

    def run(self, request: FetchRequest, feedback, cancel) -> SkillResult:
        if not self.bridge.connect() or not self.bridge.app_alive():
            message = "로봇 앱이 실행 중이지 않습니다."
            feedback(SkillFeedback(request_id=request.request_id,
                                   state="FAILED", message=message,
                                   error_code="DRIVER_DISCONNECTED"))
            return SkillResult(request_id=request.request_id, outcome="FAILED",
                               message=message,
                               error_code="DRIVER_DISCONNECTED")
        feedback(SkillFeedback(
            request_id=request.request_id, state="ACCEPTED", progress=0.05,
            message="손 위의 물건을 받아서 빈 곳에 내려놓겠습니다. "
                    "손을 펴고 가만히 계세요."))
        self.bridge.send_take_from_hand(request.request_id)

        deadline = time.monotonic() + TIMEOUT_SEC
        last = None
        while time.monotonic() < deadline:
            if cancel.is_set():
                self.bridge.send_stop()
                return SkillResult(request_id=request.request_id,
                                   outcome="CANCELLED",
                                   message="받기를 취소했습니다.")
            state, message, _, _ = self.bridge.latest()
            mapped = STATE_MESSAGES.get(state)
            if mapped and mapped != last:
                last = mapped
                skill_state, progress, spoken = mapped
                feedback(SkillFeedback(
                    request_id=request.request_id, state=skill_state,
                    progress=progress, message=spoken))
            if state == "IDLE" and message:
                if "Done" in message:
                    done = "받아서 빈 곳에 내려놓았습니다."
                    feedback(SkillFeedback(
                        request_id=request.request_id, state="SUCCEEDED",
                        progress=1.0, message=done))
                    return SkillResult(request_id=request.request_id,
                                       outcome="SUCCEEDED", message=done)
                for fragment, spoken, code in (
                        ("No steady hand", "손을 인식하지 못했습니다. "
                         "카메라 두 대가 보이도록 손을 펴 주세요.",
                         "TARGET_NOT_FOUND"),
                        ("Hand moved", "손이 움직여서 받지 못했습니다.",
                         "HAND_NOT_STABLE"),
                        ("Nothing in the gripper", "물건을 잡지 못했습니다. "
                         "손바닥 위에 올려 주세요.", "GRASP_FAILED"),
                        ("No free space", "내려놓을 빈 자리가 없습니다.",
                         "PATH_BLOCKED"),
                        ("Stopped", "정지했습니다.", None)):
                    if fragment in message:
                        feedback(SkillFeedback(
                            request_id=request.request_id, state="FAILED",
                            message=spoken, error_code=code))
                        return SkillResult(request_id=request.request_id,
                                           outcome="FAILED", message=spoken,
                                           error_code=code)
            time.sleep(POLL_SEC)
        self.bridge.send_stop()
        message = "시간 안에 받지 못했습니다."
        feedback(SkillFeedback(request_id=request.request_id, state="FAILED",
                               message=message, error_code="HANDOVER_TIMEOUT"))
        return SkillResult(request_id=request.request_id, outcome="FAILED",
                           message=message, error_code="HANDOVER_TIMEOUT")

    def emergency_stop(self) -> None:
        try:
            self.bridge.send_stop()
        except Exception:
            pass
