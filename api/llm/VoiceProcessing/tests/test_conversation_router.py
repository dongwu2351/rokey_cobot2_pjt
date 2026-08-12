import unittest

from VoiceProcessing.conversation_router import (
    CapabilityStep,
    ConversationDecision,
    ConversationRoute,
    ConversationRouter,
)


class ConversationRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = ConversationRouter(client=object())

    def test_stop_is_local(self):
        self.assertEqual(self.router.route("로봇 멈춰").route, ConversationRoute.STOP)

    def test_explicit_action_uses_command_path(self):
        decision = self.router.route("망치 가져와")
        self.assertEqual(decision.route, ConversationRoute.ROBOT_COMMAND)
        self.assertEqual(decision.command_text, "망치 가져와")
        self.assertTrue(decision.robot_action_allowed)

    def test_give_request_uses_command_path(self):
        decision = self.router.route("렌치를 줘")
        self.assertEqual(decision.route, ConversationRoute.ROBOT_COMMAND)
        self.assertTrue(decision.robot_action_allowed)

    def test_question_is_conversation(self):
        decision = self.router.route("이 드라이버는 어디에 써?")
        self.assertEqual(decision.route, ConversationRoute.TOOL_QUESTION)
        self.assertFalse(decision.robot_action_allowed)

    def test_advice_is_conversation(self):
        self.assertEqual(
            self.router.route("이 작업을 쉽게 하는 방법이 있을까?").route,
            ConversationRoute.ADVICE,
        )

    def test_capability_schema_uses_strict_scalar_fields(self):
        decision = ConversationDecision(
            route=ConversationRoute.CAPABILITY_PLAN,
            steps=[
                CapabilityStep(
                    capability="illuminate_region",
                    duration_s=10.0,
                    target="current_focus",
                )
            ],
            robot_action_allowed=True,
        )
        self.assertEqual(decision.steps[0].target, "current_focus")


if __name__ == "__main__":
    unittest.main()
