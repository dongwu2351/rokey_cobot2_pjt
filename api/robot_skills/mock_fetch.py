"""Mock fetch skill: full state timeline, no robot, no ROS.

Lets the whole conversation -> confirmation -> execution -> UI pipeline be
exercised on any laptop. `실패` in the query simulates TARGET_NOT_FOUND.
"""
from __future__ import annotations

import threading
import time

from .base import RobotSkill
from .models import FetchRequest, SkillFeedback, SkillResult, eul


class MockQuickSkill(RobotSkill):
    """Instant mock for home/gripper commands."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = label

    def run(self, request: FetchRequest, feedback, cancel_event: threading.Event
            ) -> SkillResult:
        if cancel_event.wait(0.05):
            return SkillResult(request_id=request.request_id,
                               outcome="CANCELLED", message="취소했습니다.")
        feedback(SkillFeedback(
            request_id=request.request_id, state="SUCCEEDED", progress=1.0,
            message=f"{self.label} 완료."))
        return SkillResult(
            request_id=request.request_id, outcome="SUCCEEDED",
            message=f"{self.label}{eul(self.label)} 완료했습니다.")


class MockFetchSkill(RobotSkill):
    name = "fetch_object"

    #: (state, progress, message, dwell seconds)
    TIMELINE = (
        ("ACCEPTED", 0.05, "요청을 접수했습니다.", 0.2),
        ("SEARCHING_TARGET", 0.15, "웹캠 3대로 대상을 찾고 있습니다.", 0.6),
        ("TARGET_LOCKED", 0.25, "대상을 찾아 고정했습니다.", 0.3),
        ("PLANNING_PREGRASP", 0.35, "접근 경로를 계획하고 있습니다.", 0.3),
        ("MOVING_SERVOJ", 0.5, "사람 팔을 피해 접근하고 있습니다.", 0.8),
        ("ALIGNING_SPEEDL", 0.6, "손목 카메라로 정밀 정렬 중입니다.", 0.5),
        ("GRASPING", 0.7, "물체를 잡고 있습니다.", 0.4),
        ("VERIFYING_GRASP", 0.75, "파지 상태를 확인했습니다.", 0.2),
        ("TRACKING_HAND", 0.85, "손을 추적하고 있습니다. 손을 내밀어 주세요.", 0.8),
        ("RELEASING", 0.95, "손 위에 내려놓고 있습니다.", 0.4),
        ("RETREATING", 0.98, "물러나는 중입니다.", 0.2),
    )

    def __init__(self, step_scale: float = 1.0) -> None:
        #: shrink dwell times in unit tests
        self.step_scale = step_scale

    def run(self, request: FetchRequest, feedback, cancel_event: threading.Event
            ) -> SkillResult:
        if "실패" in request.target.query:
            feedback(SkillFeedback(
                request_id=request.request_id, state="SEARCHING_TARGET",
                progress=0.1, message="대상을 찾고 있습니다."))
            feedback(SkillFeedback(
                request_id=request.request_id, state="FAILED", progress=0.1,
                message="요청한 물체를 찾지 못했습니다.",
                error_code="TARGET_NOT_FOUND", recoverable=True))
            return SkillResult(
                request_id=request.request_id, outcome="FAILED",
                message="요청한 물체를 찾지 못했습니다.",
                error_code="TARGET_NOT_FOUND")
        for state, progress, message, dwell in self.TIMELINE:
            if cancel_event.wait(dwell * self.step_scale):
                feedback(SkillFeedback(
                    request_id=request.request_id, state="CANCELLED",
                    progress=progress, message="요청이 취소되었습니다."))
                return SkillResult(
                    request_id=request.request_id, outcome="CANCELLED",
                    message="사용자 요청으로 취소했습니다.")
            feedback(SkillFeedback(
                request_id=request.request_id, state=state,
                progress=progress, message=message))
        tidy = request.destination_type == "fixed_storage"
        query = request.target.query
        message = (f"{query}{eul(query)} 보관 위치에 정리했습니다."
                   if tidy else
                   f"{query}{eul(query)} 손 위에 전달했습니다.")
        feedback(SkillFeedback(
            request_id=request.request_id, state="SUCCEEDED", progress=1.0,
            message=message))
        return SkillResult(
            request_id=request.request_id, outcome="SUCCEEDED",
            message=message, resolved_object_id="mock-1",
            grasp_verified=True, handover_verified=not tidy)


class MockInspectSkill(RobotSkill):
    """Inspection without a robot: pretend to go and look, then judge.

    Lets the whole "am I doing this right?" conversation be exercised - and
    demoed - with no arm, no cameras and (with the mock analyser) no tokens.
    """

    name = "inspect_step"
    label = "작업 확인"

    def __init__(self, analyser, image=None):
        self.analyser = analyser
        self.image = image

    def run(self, request, feedback, cancel):
        feedback(SkillFeedback(
            request_id=request.request_id, state="ACCEPTED", progress=0.1,
            message="가리키신 곳을 보러 가겠습니다. 손을 잠깐 치워 주세요."))
        for _ in range(6):
            if cancel.is_set():
                return SkillResult(request_id=request.request_id,
                                   outcome="CANCELLED",
                                   message="작업 확인을 취소했습니다.")
            time.sleep(0.05)
        feedback(SkillFeedback(
            request_id=request.request_id, state="MOVING_SERVOJ",
            progress=0.35, message="촬영 위치로 이동하고 있습니다."))
        time.sleep(0.1)
        feedback(SkillFeedback(
            request_id=request.request_id, state="VERIFYING_GRASP",
            progress=0.75, message="사진을 살펴보고 있습니다."))
        verdict = self.analyser.analyse(
            self.image or "(mock image)",
            request.target.query or "지금 이거 잘하고 있나요?", {})
        if verdict.need_closer:
            feedback(SkillFeedback(
                request_id=request.request_id, state="MOVING_SERVOJ",
                progress=0.55, message="조금 더 가까이서 다시 보겠습니다."))
            time.sleep(0.05)
            verdict = self.analyser.analyse(
                self.image or "(mock image, closer)",
                request.target.query or "지금 이거 잘하고 있나요?", {})
        feedback(SkillFeedback(
            request_id=request.request_id, state="SUCCEEDED", progress=1.0,
            message=verdict.spoken, evidence=verdict.to_payload()))
        return SkillResult(request_id=request.request_id, outcome="SUCCEEDED",
                           message=verdict.spoken)

    def emergency_stop(self) -> None:
        pass
