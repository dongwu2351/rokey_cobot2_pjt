"""fetch_object skill backed by the real webcam pick&place ROS app.

Maps the physical app's state machine onto the handoff spec's feedback
states. The physical app owns ALL motion, avoidance and safety; this skill
presses its start/stop buttons and narrates progress.

App state -> spec state:
    HOMING            ACCEPTED          (moving to the observation pose)
    DETECT            SEARCHING_TARGET  (webcam triangulation)
    APPROACH          MOVING_SERVOJ     (streamed avoidance approach)
    REFINE            ALIGNING_SPEEDL   (wrist-camera fine alignment)
    GRASP             GRASPING / VERIFYING_GRASP
    TO_PLACE_VIEW     RETREATING
    DELIVER_TRACK     TRACKING_HAND / WAITING_FOR_HANDOVER
    DELIVER_RELEASE   RELEASING
    WAIT_PLACE_CLICK  WAITING_FOR_HANDOVER (click fallback)
    IDLE(Delivered)   SUCCEEDED
    ERROR             FAILED / SAFETY_STOPPED
"""
from __future__ import annotations

import threading
import time

from .base import RobotSkill
from .models import FetchRequest, SkillFeedback, SkillResult, eul
from .ros_bridge import WebcamPnPBridge

APP_STATE_MAP = {
    "HOMING": ("ACCEPTED", 0.08, "관측 자세로 이동하고 있습니다."),
    "DETECT": ("SEARCHING_TARGET", 0.18, "웹캠 3대로 대상을 찾고 있습니다."),
    # The app can park before the approach while an operator watches the plan
    # replan (its G key). Voice requests never hold, but a keyboard-started
    # run must not read as a stalled skill if it is being watched here.
    "ARMED": ("SEARCHING_TARGET", 0.3, "경로를 계획하고 출발 신호를 기다리고 있습니다."),
    "APPROACH": ("MOVING_SERVOJ", 0.4, "사람 팔을 피해 접근하고 있습니다."),
    "REFINE": ("ALIGNING_SPEEDL", 0.55, "손목 카메라로 정밀 정렬하고 있습니다."),
    "GRASP": ("GRASPING", 0.65, "물체를 잡고 있습니다."),
    "TO_PLACE_VIEW": ("RETREATING", 0.75, "전달 자세로 이동하고 있습니다."),
    "DELIVER_TRACK": ("TRACKING_HAND", 0.85, "손을 추적하고 있습니다. 손을 내밀어 주세요."),
    "DELIVER_RELEASE": ("RELEASING", 0.95, "손 위에 내려놓고 있습니다."),
    "WAIT_PLACE_CLICK": ("WAITING_FOR_HANDOVER", 0.85,
                          "전달 위치 지정을 기다리고 있습니다."),
    "PLACING": ("RELEASING", 0.95, "지정 위치에 내려놓고 있습니다."),
}

STALE_TELEMETRY_SEC = 5.0
POLL_SEC = 0.1


class QuickCommandSkill(RobotSkill):
    """Short operator-style commands routed through the same exclusive
    manager so they can never overlap a running fetch: home / gripper."""

    def __init__(self, bridge: WebcamPnPBridge, command: str, label: str,
                 settle_seconds: float = 2.5, wait_for_home: bool = False
                 ) -> None:
        self.bridge = bridge
        self.command = command
        self.label = label
        self.settle_seconds = settle_seconds
        self.wait_for_home = wait_for_home
        self.name = command

    def emergency_stop(self) -> None:
        try:
            self.bridge.send_stop()
        except Exception:
            pass

    def run(self, request, feedback, cancel_event) -> "SkillResult":
        if not self.bridge.connect() or not self.bridge.app_alive():
            return SkillResult(
                request_id=request.request_id, outcome="FAILED",
                message="로봇 앱이 실행되어 있지 않습니다.",
                error_code="DRIVER_DISCONNECTED")
        feedback(SkillFeedback(
            request_id=request.request_id, state="ACCEPTED", progress=0.2,
            message=f"{self.label} 명령을 보냈습니다."))
        self.bridge.send_command(self.command)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if cancel_event.wait(0.2):
                self.bridge.send_stop()
                return SkillResult(
                    request_id=request.request_id, outcome="CANCELLED",
                    message=f"{self.label} 명령을 취소했습니다.")
            app_state, message, _, _ = self.bridge.latest()
            if self.wait_for_home:
                if "At HOME" in message:
                    break
            elif time.monotonic() > deadline - 30.0 + self.settle_seconds:
                break
        feedback(SkillFeedback(
            request_id=request.request_id, state="SUCCEEDED", progress=1.0,
            message=f"{self.label} 완료."))
        return SkillResult(
            request_id=request.request_id, outcome="SUCCEEDED",
            message=f"{self.label}{eul(self.label)} 완료했습니다.")


class FetchObjectSkill(RobotSkill):
    name = "fetch_object"

    def __init__(self, bridge: WebcamPnPBridge) -> None:
        self.bridge = bridge

    def emergency_stop(self) -> None:
        # Reachable from any thread; the physical app treats this exactly
        # like the operator's SPACE key.
        try:
            self.bridge.send_stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def run(self, request: FetchRequest, feedback, cancel_event: threading.Event
            ) -> SkillResult:
        if not self.bridge.connect() or not self.bridge.app_alive():
            feedback(SkillFeedback(
                request_id=request.request_id, state="FAILED",
                message="로봇 앱이 실행되어 있지 않습니다.",
                error_code="DRIVER_DISCONNECTED", recoverable=True))
            return SkillResult(
                request_id=request.request_id, outcome="FAILED",
                message="로봇 앱(webcam_pick_place)이 실행되어 있지 않습니다. "
                        "run_webcam_pnp.sh를 먼저 실행해 주세요.",
                error_code="DRIVER_DISCONNECTED")

        feedback(SkillFeedback(
            request_id=request.request_id, state="ACCEPTED", progress=0.02,
            message="로봇에 작업을 지시했습니다."))
        destination = ("storage"
                       if request.destination_type == "fixed_storage"
                       else "hand")
        self.bridge.send_start(request.request_id, destination)

        last_emitted: tuple[str, str] | None = None
        saw_grasp = False
        saw_tracking = False
        started_at = time.monotonic()
        while True:
            if cancel_event.wait(POLL_SEC):
                self.bridge.send_stop()
                feedback(SkillFeedback(
                    request_id=request.request_id, state="CANCELLED",
                    message="정지 명령을 보냈습니다."))
                return SkillResult(
                    request_id=request.request_id, outcome="CANCELLED",
                    message="사용자 요청으로 로봇을 정지했습니다.")

            app_state, message, payload, age = self.bridge.latest()
            if age > STALE_TELEMETRY_SEC and time.monotonic() - started_at > 8.0:
                self.bridge.send_stop()
                feedback(SkillFeedback(
                    request_id=request.request_id, state="FAILED",
                    message="로봇 상태 수신이 끊겼습니다.",
                    error_code="DRIVER_DISCONNECTED", recoverable=True))
                return SkillResult(
                    request_id=request.request_id, outcome="FAILED",
                    message="로봇 상태 수신이 끊겨 정지시켰습니다.",
                    error_code="DRIVER_DISCONNECTED")
            if app_state is None:
                continue

            if app_state == "IDLE":
                if "Delivered" in message or payload.get("delivered"):
                    feedback(SkillFeedback(
                        request_id=request.request_id, state="SUCCEEDED",
                        progress=1.0, message="전달을 완료했습니다."))
                    return SkillResult(
                        request_id=request.request_id, outcome="SUCCEEDED",
                        message=f"{request.target.query}{eul(request.target.query)} "
                                f"손 위에 전달했습니다.",
                        grasp_verified=saw_grasp,
                        handover_verified=saw_tracking)
                if "Done" in message:
                    feedback(SkillFeedback(
                        request_id=request.request_id, state="SUCCEEDED",
                        progress=1.0, message="작업을 완료했습니다."))
                    return SkillResult(
                        request_id=request.request_id, outcome="SUCCEEDED",
                        message=f"{request.target.query} 작업을 완료했습니다.",
                        grasp_verified=saw_grasp,
                        handover_verified=False)
                if "Stopped" in message:
                    return SkillResult(
                        request_id=request.request_id, outcome="CANCELLED",
                        message="로봇 쪽에서 정지되었습니다.")
                # IDLE before the run took off - keep waiting briefly.
                if time.monotonic() - started_at > 10.0:
                    self.bridge.send_stop()
                    return SkillResult(
                        request_id=request.request_id, outcome="FAILED",
                        message="로봇이 시작 명령에 반응하지 않았습니다.",
                        error_code="CONTROL_HANDOFF_FAILED")
                continue

            if app_state == "ERROR":
                lowered = message.lower()
                safety = ("protective" in lowered or "safety" in lowered
                          or "not responding" in lowered)
                error_code = "SAFETY_STOP" if safety else (
                    "TARGET_NOT_FOUND" if "triangulated" in lowered
                    or "cannot find" in lowered else "GRASP_FAILED")
                state = "SAFETY_STOPPED" if safety else "FAILED"
                feedback(SkillFeedback(
                    request_id=request.request_id, state=state,
                    message=message or "로봇 오류가 발생했습니다.",
                    error_code=error_code, recoverable=not safety))
                return SkillResult(
                    request_id=request.request_id,
                    outcome=state,
                    message=message or "로봇 오류가 발생했습니다.",
                    error_code=error_code)

            mapped = APP_STATE_MAP.get(app_state)
            if mapped is not None:
                state, progress, default_message = mapped
                if app_state == "GRASP":
                    saw_grasp = True
                if app_state in ("DELIVER_TRACK", "DELIVER_RELEASE"):
                    saw_tracking = True
                # Waiting-for-hand nuance inside DELIVER_TRACK.
                if app_state == "DELIVER_TRACK" and (
                        "Hold out" in message or "hand" in message.lower()
                        and "Tracking" not in message):
                    state = "WAITING_FOR_HANDOVER"
                # Obstacle narration comes straight from the physical app.
                if "[avoiding obstacle]" in message:
                    default_message = "사람 팔을 피해 경로를 갱신하며 접근하고 있습니다."
                if "waiting" in message.lower() and app_state == "APPROACH":
                    default_message = "팔이 통로를 막고 있어 잠시 대기하고 있습니다."
                key = (state, default_message)
                if key != last_emitted:
                    last_emitted = key
                    feedback(SkillFeedback(
                        request_id=request.request_id, state=state,
                        progress=progress, message=default_message,
                        evidence={"app_state": app_state,
                                  "app_message": message}))
