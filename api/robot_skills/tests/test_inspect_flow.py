"""Pointing-and-asking: photograph the spot, judge it, say something useful."""
from __future__ import annotations

import threading
import time
import unittest

from robot_skills.inspect_step import InspectStepSkill
from robot_skills.inspect_vision import MockVision, build_analyser, _clean
from robot_skills.models import FetchOptions, FetchRequest, FetchTarget


class FakeBridge:
    """The physical app, reduced to what the skill actually depends on."""

    def __init__(self, result=None, alive=True, delay=0.0, results=None):
        # `results` feeds one record per capture, so a re-shoot can be seen.
        self.results = list(results) if results else None
        self.result = result
        self.alive = alive
        self.delay = delay
        self.sent = []
        self._armed_at = None

    def connect(self, timeout=5.0):
        return True

    def app_alive(self):
        return self.alive

    def send_inspect(self, request_id=None, point_mm=None, standoff_mm=None):
        self.sent.append(("inspect", request_id, standoff_mm))
        if self.results:
            self.result = self.results.pop(0)
        self._armed_at = time.monotonic()

    def send_stop(self):
        self.sent.append(("stop", None, None))

    def send_inspect_done(self):
        self.sent.append(("done", None, None))

    def take_inspection(self, request_id=None, max_age=120.0):
        if self._armed_at is None or self.result is None:
            return None
        if time.monotonic() - self._armed_at < self.delay:
            return None
        payload, self.result = self.result, None
        return payload

    def snapshot(self):
        return "INSPECT", "", {}, 0.0


def _request(query="지금 이거 잘하고 있는 거 맞아?"):
    return FetchRequest(
        target=FetchTarget(query=query),
        options=FetchOptions(dry_run=False, require_confirmation=False,
                             timeout_seconds=30.0),
        skill="inspect_step")


class InspectSkillTests(unittest.TestCase):
    def _run(self, bridge, analyser=None):
        feedback = []
        skill = InspectStepSkill(bridge, analyser or MockVision(["CORRECT"]))
        result = skill.run(_request(), feedback.append, threading.Event())
        return result, feedback

    def test_photograph_then_verdict(self):
        bridge = FakeBridge({"ok": True, "path": "/tmp/x.jpg",
                             "point": [400.0, 0.0, 0.0]})
        result, feedback = self._run(bridge)
        self.assertEqual(result.outcome, "SUCCEEDED")
        self.assertIn("맞습니다", result.message)
        self.assertEqual(feedback[-1].evidence["verdict"], "CORRECT")
        # The arm is released from the viewing pose once the verdict is in.
        self.assertIn(("done", None, None), bridge.sent)
        # The point reaches the record so a UI can show WHERE it looked.
        self.assertEqual(feedback[-1].evidence["pointed_at_mm"],
                         [400.0, 0.0, 0.0])

    def test_hand_in_the_way_is_a_robot_failure_not_a_verdict(self):
        bridge = FakeBridge({"ok": False, "error": "hand stayed over the spot"})
        result, _ = self._run(bridge)
        self.assertEqual(result.outcome, "FAILED")
        self.assertEqual(result.error_code, "HAND_NOT_STABLE")
        self.assertIn("손", result.message)

    def test_app_not_running_fails_closed(self):
        result, _ = self._run(FakeBridge(alive=False))
        self.assertEqual(result.outcome, "FAILED")
        self.assertEqual(result.error_code, "DRIVER_DISCONNECTED")

    def test_no_answer_from_the_app_times_out(self):
        skill = InspectStepSkill(FakeBridge(None), MockVision())
        import robot_skills.inspect_step as module
        original = module.CAPTURE_TIMEOUT_SEC
        module.CAPTURE_TIMEOUT_SEC = 0.3
        try:
            result = skill.run(_request(), lambda item: None, threading.Event())
        finally:
            module.CAPTURE_TIMEOUT_SEC = original
        self.assertEqual(result.outcome, "FAILED")
        self.assertEqual(result.error_code, "CONTROL_HANDOFF_FAILED")

    def test_cancel_stops_the_arm(self):
        bridge = FakeBridge({"ok": True, "path": "/tmp/x.jpg"}, delay=5.0)
        cancel = threading.Event()
        cancel.set()
        skill = InspectStepSkill(bridge, MockVision())
        result = skill.run(_request(), lambda item: None, cancel)
        self.assertEqual(result.outcome, "CANCELLED")
        self.assertIn(("stop", None, None), bridge.sent)


class CloserRetryTests(unittest.TestCase):
    """The model may say "too far to tell" - the arm gets one second look."""

    def test_need_closer_triggers_one_nearer_shot(self):
        far = {"ok": True, "path": "/tmp/far.jpg", "standoff_mm": 420.0}
        near = {"ok": True, "path": "/tmp/near.jpg", "standoff_mm": 252.0}
        bridge = FakeBridge(results=[far, near])
        # First judgement asks to come closer, the second one decides.
        analyser = MockVision(["NOT_VISIBLE", "CORRECT"])
        skill = InspectStepSkill(bridge, analyser)
        feedback = []
        result = skill.run(_request(), feedback.append, threading.Event())
        shots = [item for item in bridge.sent if item[0] == "inspect"]
        self.assertEqual(len(shots), 2, "expected exactly one re-shoot")
        self.assertIsNone(shots[0][2])                 # first: default
        self.assertLess(shots[1][2], 420.0)            # second: closer
        self.assertGreaterEqual(shots[1][2], 240.0)    # but not into the bench
        self.assertEqual(result.outcome, "SUCCEEDED")
        self.assertIn("맞습니다", result.message)
        self.assertEqual(feedback[-1].evidence["verdict"], "CORRECT")

    def test_it_never_loops(self):
        records = [{"ok": True, "path": f"/tmp/{i}.jpg", "standoff_mm": 420.0}
                   for i in range(4)]
        bridge = FakeBridge(results=records)
        # A model that always wants closer must still be answered.
        skill = InspectStepSkill(bridge, MockVision(["NOT_VISIBLE"]))
        result = skill.run(_request(), lambda item: None, threading.Event())
        shots = [item for item in bridge.sent if item[0] == "inspect"]
        self.assertEqual(len(shots), 2)
        self.assertEqual(result.outcome, "SUCCEEDED")


class VisionContractTests(unittest.TestCase):
    def test_unknown_verdict_becomes_uncertain(self):
        verdict = _clean({"verdict": "LOOKS_FINE_TO_ME", "spoken": "네"},
                         "x.jpg", "test", 0.1)
        self.assertEqual(verdict.verdict, "UNCERTAIN")

    def test_empty_answer_still_says_something(self):
        verdict = _clean({}, "x.jpg", "test", 0.1)
        self.assertTrue(verdict.spoken)
        self.assertEqual(verdict.verdict, "UNCERTAIN")

    def test_confidence_is_clamped(self):
        self.assertEqual(_clean({"confidence": 9.0}, "", "", 0).confidence, 1.0)
        self.assertEqual(_clean({"confidence": "x"}, "", "", 0).confidence, 0.0)

    def test_missing_image_never_asserts_correct(self):
        from robot_skills.inspect_vision import OpenAIVision
        vision = OpenAIVision(api_key="dummy")
        verdict = vision.analyse("/nonexistent.jpg", "이거 맞아?")
        self.assertEqual(verdict.verdict, "NOT_VISIBLE")
        self.assertFalse(verdict.ok)

    def test_need_closer_is_carried_through(self):
        verdict = _clean({"verdict": "NOT_VISIBLE", "need_closer": True},
                         "x.jpg", "t", 0.0)
        self.assertTrue(verdict.need_closer)
        self.assertTrue(verdict.to_payload()["need_closer"])

    def test_auto_falls_back_to_mock_without_a_key(self):
        self.assertIs(type(build_analyser("auto", api_key="")), MockVision)


if __name__ == "__main__":
    unittest.main()
