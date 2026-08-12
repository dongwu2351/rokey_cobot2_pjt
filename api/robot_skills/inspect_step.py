"""Skill: photograph what the operator is pointing at, then judge it.

Runs through the same exclusive manager as fetching, because it moves the
same arm - the two can never overlap. The physical half is the app's
"inspect" command (go, shoot, come home); this skill waits for the result,
hands the image to a vision model, and returns a verdict the copilot can
speak.

Split deliberately in two: a failed photograph is a robot problem
(HAND_NOT_STABLE, DRIVER_DISCONNECTED) while a failed judgement is a model
problem, and the operator needs to hear which one happened.
"""
from __future__ import annotations

import sys
import time

from .base import RobotSkill
from .models import FetchRequest, SkillFeedback, SkillResult

POLL_SEC = 0.1
#: The arm waits for the hand to move away, drives out, settles, shoots and
#: comes home. Generous, because the wait for the hand is the operator's own
#: pace, not the robot's.
CAPTURE_TIMEOUT_SEC = 45.0
#: One automatic second look, closer, when the model says the picture was too
#: far to judge. Bounded at one: a model that keeps asking would walk the arm
#: into the bench, and by then the honest answer is "I cannot tell".
CLOSER_FACTOR = 0.6
CLOSER_MIN_MM = 240.0


class InspectStepSkill(RobotSkill):
    """One picture, one verdict."""

    name = "inspect_step"
    label = "작업 확인"

    def __init__(self, bridge, analyser, manual=None):
        self.bridge = bridge
        self.analyser = analyser
        self.manual = manual          # optional: current step context

    # ------------------------------------------------------------------
    def run(self, request: FetchRequest, feedback, cancel) -> SkillResult:
        question = request.target.query or "지금 이거 잘하고 있나요?"
        standoff = request.target.attributes.get("standoff_mm")
        if not self.bridge.connect():
            return self._failed(request, "로봇 앱에 연결하지 못했습니다.",
                                "DRIVER_DISCONNECTED", feedback)
        if not self.bridge.app_alive():
            return self._failed(request, "로봇 앱이 실행 중이지 않습니다.",
                                "DRIVER_DISCONNECTED", feedback)

        feedback(SkillFeedback(
            request_id=request.request_id, state="ACCEPTED", progress=0.1,
            message="가리키신 곳을 보러 가겠습니다. 손을 잠깐 치워 주세요."))
        record = self._capture(request, feedback, cancel, standoff)
        if isinstance(record, SkillResult):
            return record

        feedback(SkillFeedback(
            request_id=request.request_id, state="VERIFYING_GRASP",
            progress=0.75, message="사진을 살펴보고 있습니다."))
        verdict = self._judge(record, question)

        # The model may ask to come closer - one more trip, then decide.
        if verdict.need_closer and not cancel.is_set():
            closer = max(CLOSER_MIN_MM,
                         (standoff or record.get("standoff_mm") or 420.0)
                         * CLOSER_FACTOR)
            feedback(SkillFeedback(
                request_id=request.request_id, state="MOVING_SERVOJ",
                progress=0.55,
                message="조금 더 가까이서 다시 보겠습니다."))
            retry = self._capture(request, feedback, cancel, closer)
            if not isinstance(retry, SkillResult):
                record = retry
                verdict = self._judge(record, question)

        # Verdict settled - let the arm go home instead of holding over the
        # work until its timeout.
        try:
            self.bridge.send_inspect_done()
        except Exception:
            pass

        outcome_message = verdict.spoken
        evidence = verdict.to_payload()
        evidence["pointed_at_mm"] = record.get("point")
        evidence["standoff_mm"] = record.get("standoff_mm")
        feedback(SkillFeedback(
            request_id=request.request_id, state="SUCCEEDED", progress=1.0,
            message=outcome_message, evidence=evidence))
        return SkillResult(request_id=request.request_id, outcome="SUCCEEDED",
                           message=outcome_message)

    # ------------------------------------------------------------------
    def _judge(self, record, question):
        # Prefer the copy with the pointed spot circled: without it the model
        # has to guess which of the parts in frame the question is about.
        image = record.get("marked_path") or record.get("path")
        return self.analyser.analyse(image, question, self._context(record),
                                     reference_image=self._reference())

    def _capture(self, request, feedback, cancel, standoff_mm=None):
        """One photograph. Returns the record, or a SkillResult on failure."""
        self.bridge.take_inspection()          # drop anything stale
        self.bridge.send_inspect(request.request_id,
                                 request.target.attributes.get("point"),
                                 standoff_mm=standoff_mm)

        deadline = time.monotonic() + CAPTURE_TIMEOUT_SEC
        record = None
        announced = False
        while time.monotonic() < deadline:
            if cancel.is_set():
                self.bridge.send_stop()
                return SkillResult(request_id=request.request_id,
                                   outcome="CANCELLED",
                                   message="작업 확인을 취소했습니다.")
            record = self.bridge.take_inspection(request.request_id)
            if record is not None:
                break
            state, message, _, _ = self.bridge.latest()
            if state == "INSPECT" and not announced:
                announced = True
                feedback(SkillFeedback(
                    request_id=request.request_id, state="MOVING_SERVOJ",
                    progress=0.35, message="촬영 위치로 이동하고 있습니다."))
            time.sleep(POLL_SEC)
        if record is None:
            return self._failed(request, "사진을 찍지 못했습니다.",
                                "CONTROL_HANDOFF_FAILED", feedback)
        if not record.get("ok"):
            reason = record.get("error", "")
            spoken = ("손이 계속 그 위에 있어 촬영하지 못했습니다."
                      if "hand" in reason else
                      f"촬영에 실패했습니다. ({reason})" if reason else
                      "촬영에 실패했습니다.")
            return self._failed(request, spoken, "HAND_NOT_STABLE", feedback)
        return record

    # ------------------------------------------------------------------
    def _context(self, record):
        context = {}
        if record.get("marked_path"):
            context["표시"] = "빨간 원이 작업자가 가리킨 지점입니다."
        if record.get("tilt_deg") is not None:
            tilt = record["tilt_deg"]
            context["촬영 각도"] = (
                "바로 위에서 내려다봄" if tilt < 5 else
                f"수직에서 {tilt:.0f}도 기울여 비스듬히 촬영")
        if record.get("point"):
            x, y, _ = record["point"]
            context["작업 위치(로봇 좌표 mm)"] = f"x={x:.0f}, y={y:.0f}"
        step = None
        if self.manual is not None:
            try:
                step = self.manual.current_step()
            except Exception as failure:
                # Swallowing this is how the model ended up judging blind.
                print(f"[검사] 단계 정보를 읽지 못했습니다: {failure}",
                      file=sys.stderr, flush=True)
                step = None
        if step:
            order, total = step.get("order"), step.get("total")
            context["현재 조립 단계"] = (
                f"{order}/{total}단계 - {step.get('title', '')}"
                if order else step.get("title", ""))
            context["단계 설명"] = step.get("instruction", "")
            if step.get("completion"):
                context["이 단계의 완료 조건"] = step["completion"]
        else:
            # Say it out loud rather than letting the model discover the gap
            # and blame the picture for it.
            context["주의"] = ("현재 단계 정보를 불러오지 못했습니다. "
                              "사진에 보이는 것만으로 판단하세요.")
        return context

    def _reference(self):
        """The manual's photograph of this step completed, for comparison."""
        if self.manual is None:
            return None
        try:
            step = self.manual.current_step() or {}
        except Exception:
            return None
        return step.get("reference_image")

    def _failed(self, request, message, code, feedback):
        feedback(SkillFeedback(
            request_id=request.request_id, state="FAILED", message=message,
            error_code=code, recoverable=True))
        return SkillResult(request_id=request.request_id, outcome="FAILED",
                           message=message, error_code=code)

    def emergency_stop(self) -> None:
        try:
            self.bridge.send_stop()
        except Exception:
            pass
