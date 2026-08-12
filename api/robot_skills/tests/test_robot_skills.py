from __future__ import annotations

import threading
import time
import unittest

from robot_skills import (FetchOptions, FetchRequest, FetchTarget,
                          RobotSkillManager, SkillRegistry,
                          resolve_object_class)
from robot_skills.base import RobotSkill
from robot_skills.mock_fetch import MockFetchSkill
from robot_skills.models import SkillFeedback, SkillResult


def _request(query="해머", **options) -> FetchRequest:
    return FetchRequest(
        target=FetchTarget(query=query,
                           class_name=resolve_object_class(query)),
        options=FetchOptions(**options))


def _manager(skill, **kwargs) -> RobotSkillManager:
    registry = SkillRegistry()
    registry.register("fetch_object", skill)
    return RobotSkillManager(registry, **kwargs)


class ModelTest(unittest.TestCase):
    def test_payload_roundtrip(self):
        request = _request("빨간 해머")
        payload = request.to_payload()
        self.assertEqual(payload["target"]["class_name"], "hammer")
        self.assertEqual(payload["destination"]["type"], "user_handover")
        self.assertTrue(payload["options"]["dry_run"])
        restored = FetchRequest.from_payload(payload)
        self.assertEqual(restored.request_id, request.request_id)
        self.assertEqual(restored.target.query, "빨간 해머")

    def test_object_aliases(self):
        self.assertEqual(resolve_object_class("망치 좀 줘"), "hammer")
        self.assertEqual(resolve_object_class("십자 드라이버"), "screwdriver")
        self.assertIsNone(resolve_object_class("커피 좀"))

    def test_invalid_state_rejected(self):
        with self.assertRaises(ValueError):
            SkillFeedback(request_id="x", state="FLYING")
        with self.assertRaises(ValueError):
            SkillResult(request_id="x", outcome="MOVING_SERVOJ", message="no")


class ManagerTest(unittest.TestCase):
    def test_mock_success_timeline(self):
        feedbacks, results = [], []
        manager = _manager(MockFetchSkill(step_scale=0.01),
                           on_feedback=feedbacks.append,
                           on_result=results.append)
        accepted, _ = manager.submit(_request())
        self.assertTrue(accepted)
        deadline = time.time() + 5
        while manager.busy and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(manager.busy)
        self.assertEqual(results[-1].outcome, "SUCCEEDED")
        self.assertTrue(results[-1].handover_verified)
        states = [item.state for item in feedbacks]
        self.assertIn("TRACKING_HAND", states)
        self.assertEqual(states[-1], "SUCCEEDED")

    def test_single_physical_skill_at_a_time(self):
        manager = _manager(MockFetchSkill(step_scale=0.2))
        accepted, _ = manager.submit(_request("해머"))
        self.assertTrue(accepted)
        accepted2, message = manager.submit(_request("드라이버"))
        self.assertFalse(accepted2)
        self.assertIn("이미", message)
        manager.cancel_active()
        deadline = time.time() + 5
        while manager.busy and time.time() < deadline:
            time.sleep(0.01)

    def test_cancel(self):
        results = []
        manager = _manager(MockFetchSkill(step_scale=0.5),
                           on_result=results.append)
        manager.submit(_request())
        time.sleep(0.05)
        self.assertTrue(manager.cancel_active())
        deadline = time.time() + 5
        while manager.busy and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(results[-1].outcome, "CANCELLED")

    def test_failure_path(self):
        results = []
        manager = _manager(MockFetchSkill(step_scale=0.01),
                           on_result=results.append)
        manager.submit(_request("실패 해머"))
        deadline = time.time() + 5
        while manager.busy and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(results[-1].outcome, "FAILED")
        self.assertEqual(results[-1].error_code, "TARGET_NOT_FOUND")

    def test_timeout_fails_closed(self):
        class Sleeper(RobotSkill):
            name = "fetch_object"

            def run(self, request, feedback, cancel_event):
                cancel_event.wait(10.0)
                return SkillResult(request_id=request.request_id,
                                   outcome="CANCELLED", message="늦게 취소")

        results = []
        manager = _manager(Sleeper(), on_result=results.append)
        manager.submit(_request(timeout_seconds=0.2))
        deadline = time.time() + 5
        while manager.busy and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(results[-1].outcome, "FAILED")
        self.assertEqual(results[-1].error_code, "HANDOVER_TIMEOUT")

    def test_skill_exception_becomes_failed_result(self):
        class Exploder(RobotSkill):
            name = "fetch_object"

            def run(self, request, feedback, cancel_event):
                raise RuntimeError("boom")

        results = []
        manager = _manager(Exploder(), on_result=results.append)
        manager.submit(_request())
        deadline = time.time() + 5
        while manager.busy and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(results[-1].outcome, "FAILED")


class StateMappingTest(unittest.TestCase):
    def test_app_states_all_map_to_spec_states(self):
        from robot_skills.fetch_object import APP_STATE_MAP
        from robot_skills.models import SKILL_STATES
        for app_state, (state, progress, _) in APP_STATE_MAP.items():
            self.assertIn(state, SKILL_STATES, app_state)
            self.assertTrue(0.0 <= progress <= 1.0)


if __name__ == "__main__":
    unittest.main()
