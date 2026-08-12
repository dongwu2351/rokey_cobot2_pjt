"""fetch_object conversation flow: route -> confirm -> execute -> stop.

Runs entirely on the mock skill - no ROS, no robot - per the handoff spec's
completion criteria ("실제 로봇이 없어도 전체 대화·승인·UI 상태를 검증").
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from unified_copilot.engine import UnifiedCopilotEngine
from unified_copilot.intents import UnifiedIntentRouter
from unified_copilot.memory import CopilotStateDB

from robot_skills import RobotSkillManager, SkillRegistry
from robot_skills.mock_fetch import MockFetchSkill


from robot_skills.mock_fetch import MockQuickSkill


def _mock_manager(**kwargs) -> RobotSkillManager:
    registry = SkillRegistry()
    registry.register("fetch_object", MockFetchSkill(step_scale=0.01))
    registry.register("robot_home", MockQuickSkill("홈 복귀"))
    registry.register("gripper_open", MockQuickSkill("그리퍼 열기"))
    registry.register("gripper_close", MockQuickSkill("그리퍼 닫기"))
    return RobotSkillManager(registry, **kwargs)


def _engine(tmp: Path, manager: RobotSkillManager | None):
    engine = UnifiedCopilotEngine.__new__(UnifiedCopilotEngine)
    engine.router = UnifiedIntentRouter()
    engine.db = CopilotStateDB(tmp / "copilot.sqlite3")
    engine.robot_skills = manager
    return engine


def _wait_idle(manager: RobotSkillManager, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while manager.busy and time.time() < deadline:
        time.sleep(0.01)


class RouterTest(unittest.TestCase):
    def setUp(self):
        self.router = UnifiedIntentRouter()

    def test_fetch_phrasings_route_to_robot_skill(self):
        for text in ("해머 가져와줘", "망치 좀 챙겨줄래", "드라이버 건네줘",
                     "렌치 갖다 줘", "저기 있는 해머 좀 집어줘"):
            intent = self.router.route(text)
            self.assertEqual(intent.domain, "ROBOT_SKILL", text)
            self.assertEqual(intent.intent, "FETCH_OBJECT", text)

    def test_non_fetch_object_talk_stays_out(self):
        self.assertNotEqual(self.router.route("해머가 어디 있지?").domain,
                            "ROBOT_SKILL")
        self.assertNotEqual(self.router.route("오늘 날씨 어때").domain,
                            "ROBOT_SKILL")

    def test_stop_word_is_immediate_only_while_robot_active(self):
        active = self.router.route("멈춰", robot_skill_active=True)
        self.assertEqual((active.domain, active.intent),
                         ("ROBOT_SKILL", "STOP"))
        idle = self.router.route("멈춰", robot_skill_active=False)
        self.assertNotEqual(idle.domain, "ROBOT_SKILL")

    def test_stop_beats_fetch_wording_while_active(self):
        intent = self.router.route("해머 가져오는 거 그만", robot_skill_active=True)
        self.assertEqual(intent.intent, "STOP")

    def test_tidy_routes_before_fetch(self):
        for text in ("해머 정리해줘", "망치 제자리에 갖다 놔", "해머 좀 치워줘"):
            intent = self.router.route(text)
            self.assertEqual((intent.domain, intent.intent),
                             ("ROBOT_SKILL", "TIDY_OBJECT"), text)

    def test_quick_commands_route(self):
        self.assertEqual(self.router.route("홈으로 돌아가").intent, "HOME")
        self.assertEqual(self.router.route("그리퍼 열어줘").intent,
                         "GRIPPER_OPEN")
        self.assertEqual(self.router.route("그리퍼 닫아").intent,
                         "GRIPPER_CLOSE")
        # Assembly-step phrasing must not be swallowed by robot home.
        self.assertNotEqual(self.router.route("3단계로 돌아가고 싶어").domain,
                            "ROBOT_SKILL")


class FetchFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fetch_requires_confirmation_then_runs(self):
        results = []
        manager = _mock_manager(on_result=results.append)
        engine = _engine(self.tmp, manager)
        intent = engine.router.route("해머 좀 가져와줘")
        reply = engine._robot_skill(intent)
        self.assertIn("가져다드릴까요", reply.text)
        self.assertFalse(manager.busy)  # not yet: confirmation first
        pending = engine.db.pending()
        self.assertEqual(pending["operation_type"], "ROBOT_FETCH_OBJECT")
        self.assertEqual(pending["payload"]["class_name"], "hammer")

        confirm = engine._confirmation("ACCEPT")
        self.assertIn("가져오겠습니다", confirm.text)
        _wait_idle(manager)
        self.assertEqual(results[-1].outcome, "SUCCEEDED")
        self.assertTrue(results[-1].handover_verified)

    def test_reject_cancels_pending_without_running(self):
        manager = _mock_manager()
        engine = _engine(self.tmp, manager)
        engine._robot_skill(engine.router.route("드라이버 가져와줘"))
        reply = engine._confirmation("REJECT")
        self.assertIn("취소", reply.text)
        self.assertFalse(manager.busy)
        self.assertIsNone(engine.db.pending())

    def test_stale_or_foreign_confirmation_does_not_start_robot(self):
        manager = _mock_manager()
        engine = _engine(self.tmp, manager)
        reply = engine._confirmation("ACCEPT")  # nothing pending
        self.assertIn("확인을 기다리는 작업이 없습니다", reply.text)
        self.assertFalse(manager.busy)

    def test_duplicate_request_rejected_while_busy(self):
        manager = _mock_manager()
        registry = SkillRegistry()
        registry.register("fetch_object", MockFetchSkill(step_scale=0.3))
        manager = RobotSkillManager(registry)
        engine = _engine(self.tmp, manager)
        engine._robot_skill(engine.router.route("해머 가져와줘"))
        engine._confirmation("ACCEPT")
        self.assertTrue(manager.busy)
        busy_reply = engine._robot_skill(engine.router.route("드라이버 가져와줘"))
        self.assertIn("수행 중", busy_reply.text)
        stop = engine._robot_skill(
            engine.router.route("멈춰", robot_skill_active=True))
        self.assertIn("정지", stop.text)
        _wait_idle(manager)

    def test_unknown_object_asks_for_clarification(self):
        engine = _engine(self.tmp, _mock_manager())
        reply = engine._robot_skill(engine.router.route("공구 좀 가져와줘"))
        self.assertIn("어떤 공구", reply.text)
        self.assertIsNone(engine.db.pending())

    def test_without_manager_fetch_fails_politely(self):
        engine = _engine(self.tmp, None)
        reply = engine._robot_skill(engine.router.route("해머 가져와줘"))
        self.assertIn("연결되어 있지 않아", reply.text)

    def test_tidy_confirmation_carries_storage_destination(self):
        results = []
        manager = _mock_manager(on_result=results.append)
        engine = _engine(self.tmp, manager)
        reply = engine._robot_skill(engine.router.route("해머 정리해줘"))
        self.assertIn("정리해 둘까요", reply.text)
        pending = engine.db.pending()
        self.assertEqual(pending["payload"]["destination"], "fixed_storage")
        confirm = engine._confirmation("ACCEPT")
        self.assertIn("정리하겠습니다", confirm.text)
        _wait_idle(manager)
        self.assertEqual(results[-1].outcome, "SUCCEEDED")

    def test_quick_commands_run_without_confirmation(self):
        results = []
        manager = _mock_manager(on_result=results.append)
        engine = _engine(self.tmp, manager)
        reply = engine._robot_skill(engine.router.route("그리퍼 열어줘"))
        self.assertIn("실행합니다", reply.text)
        self.assertIsNone(engine.db.pending())  # no confirmation round
        _wait_idle(manager)
        self.assertEqual(results[-1].outcome, "SUCCEEDED")
        reply = engine._robot_skill(engine.router.route("홈으로 돌아가"))
        self.assertIn("홈 복귀", reply.text)
        _wait_idle(manager)
        self.assertEqual(results[-1].outcome, "SUCCEEDED")


class SpokenConfirmationTest(unittest.TestCase):
    """Live voice answers are open-ended; only "응" used to start the robot."""

    def setUp(self):
        self.router = UnifiedIntentRouter()

    def _answer(self, text):
        intent = self.router.route(text, has_pending_confirmation=True)
        return intent.domain, intent.intent

    def test_varied_agreement_accepts(self):
        for text in ("응", "어", "네", "넵", "ㅇㅇ", "오케이", "어 가져와",
                     "응 부탁해", "응, 부탁할게", "그래 해줘", "당연하지",
                     "물론이지", "좋아 진행해", "부탁드려요", "해줘", "가자"):
            with self.subTest(text=text):
                self.assertEqual(self._answer(text), ("CONFIRMATION", "ACCEPT"))

    def test_refusals_reject(self):
        for text in ("아니", "취소", "취소해줘", "하지 마", "잠깐만", "싫어",
                     "나중에"):
            with self.subTest(text=text):
                self.assertEqual(self._answer(text), ("CONFIRMATION", "REJECT"))

    def test_substantive_replies_route_normally(self):
        self.assertEqual(self._answer("해머 가져와줘"),
                         ("ROBOT_SKILL", "FETCH_OBJECT"))
        self.assertEqual(self._answer("그거 말고 드라이버 가져와줘"),
                         ("ROBOT_SKILL", "FETCH_OBJECT"))
        domain, _ = self._answer("작업대 위에 있어")
        self.assertNotEqual(domain, "CONFIRMATION")

    def test_stop_still_wins_while_robot_moves(self):
        intent = self.router.route("멈춰", has_pending_confirmation=True,
                                   robot_skill_active=True)
        self.assertEqual((intent.domain, intent.intent), ("ROBOT_SKILL", "STOP"))

    def test_stt_misheard_hammer_still_fetches(self):
        for text in ("해먹 가져와", "함마 가져다 줘", "햄머 좀 챙겨줘"):
            with self.subTest(text=text):
                intent = self.router.route(text)
                self.assertEqual((intent.domain, intent.intent),
                                 ("ROBOT_SKILL", "FETCH_OBJECT"))


if __name__ == "__main__":
    unittest.main()
