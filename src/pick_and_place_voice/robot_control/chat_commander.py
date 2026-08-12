"""LLM chat commander for the webcam pick&place robot.

Terminal chat with GPT-4o. Say anything that means "bring me the hammer"
("해머 가져와줘", "해머 좀 챙겨줘", "망치 필요해", ...) and it publishes
"start" on /webcam_pnp/command - the same thing pressing S in the app does.
"멈춰/그만" publishes "stop". Everything else is just conversation.

The app's live status (/webcam_pnp/status) is echoed into the chat, so the
LLM's replies and the robot's progress interleave naturally.

Run alongside the main app:

    ros2 run pick_and_place_voice webcam_pick_place --live --free-wrist --hand-place
    ros2 run pick_and_place_voice chat_commander
"""
import json
import os
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

SYSTEM_PROMPT = """\
너는 작업장 로봇 어시스턴트야. 사용자는 두산 M0609 로봇 옆에서 일하고 있고,
로봇은 웹캠으로 해머를 찾아 집어서 사용자의 손 위에 가져다주는 기능이 있어.

반드시 아래 JSON 형식으로만 답해 (다른 텍스트 금지):
{"reply": "<사용자에게 할 자연스러운 한국어 답변>", "action": "<fetch|stop|none>"}

action 규칙:
- 사용자가 해머(망치)를 가져와 달라는 의도라면 (어떤 표현이든: 가져와줘,
  챙겨줘, 건네줘, 필요해, 좀 줘 등) -> "fetch". reply에는 가져다주겠다고
  답하고, 손을 내밀고 있으라고 안내해.
- 멈추라는 의도(멈춰, 그만, 스톱) -> "stop"
- 그 외 일반 대화 -> "none". 자유롭게 대화해.
- 이미 로봇이 동작 중일 때 또 가져와 달라고 하면 진행 중이라고 알려주고 "none".
"""


def fallback_intent(text):
    """Rule-based intent when the LLM is unreachable (no credits, no net).
    Cruder conversation, identical robot behaviour."""
    lowered = text.lower()
    if any(word in lowered for word in ("멈춰", "그만", "정지", "스톱", "stop")):
        return "네, 멈출게요.", "stop"
    wants_hammer = any(word in lowered for word in ("해머", "망치", "hammer"))
    wants_action = any(
        word in lowered
        for word in ("가져", "챙겨", "갖다", "건네", "줘", "다오", "필요", "부탁")
    )
    if wants_hammer and wants_action:
        return "네! 해머 가져다드릴게요. 손을 내밀고 계세요.", "fetch"
    if wants_hammer:
        return "해머가 필요하시면 '해머 가져와줘'라고 말씀해 주세요.", "none"
    return "무엇을 도와드릴까요? (예: 해머 좀 챙겨줘)", "none"


def load_api_key():
    candidates = [
        Path(__file__).resolve().parents[1] / "resource" / ".env",
        Path.home() / "cobot2_ws_1" / "pick_and_place_voice" / "resource" / ".env",
    ]
    for path in candidates:
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY")


class ChatCommander(Node):
    def __init__(self):
        super().__init__("chat_commander")
        self.command_pub = self.create_publisher(
            StringMsg, "/webcam_pnp/command", 10
        )
        self.create_subscription(
            StringMsg, "/webcam_pnp/status", self._status_cb, 10
        )
        self.robot_busy = False
        self._last_status = ""

    def _status_cb(self, msg):
        status = msg.data
        if status == self._last_status:
            return
        self._last_status = status
        self.robot_busy = not status.startswith("[IDLE]")
        print(f"\r\033[K\033[90m🤖 {status}\033[0m")
        if "Delivered" in status:
            print("\033[92m🤖 해머를 손에 올려드렸어요!\033[0m")
        print("나: ", end="", flush=True)


def main():
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY를 resource/.env에서 찾지 못했습니다")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    rclpy.init()
    node = ChatCommander()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=" * 60)
    print(" 로봇 채팅 - 예: \"해머 좀 챙겨줘\" / \"멈춰\" / 일반 대화")
    print(" 종료: Ctrl+C 또는 'exit'")
    print("=" * 60)
    try:
        while True:
            try:
                user_text = input("나: ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit", "종료"):
                break

            state_note = (
                "(현재 로봇 상태: 동작 중)" if node.robot_busy
                else "(현재 로봇 상태: 대기)"
            )
            messages.append(
                {"role": "user", "content": f"{state_note} {user_text}"}
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                messages.append({"role": "assistant", "content": content})
                parsed = json.loads(content)
            except Exception as error:
                # LLM unavailable (no credits / no network): fall back to
                # keyword rules so the robot still answers the ask.
                messages.pop()
                reply, action = fallback_intent(user_text)
                parsed = {"reply": reply + "  (오프라인 모드)", "action": action}

            print(f"\033[96m로봇: {parsed.get('reply', '')}\033[0m")
            action = parsed.get("action", "none")
            if action == "fetch":
                node.command_pub.publish(StringMsg(data="start"))
                print("\033[93m   -> 로봇 출발! (start 명령 전송)\033[0m")
            elif action == "stop":
                node.command_pub.publish(StringMsg(data="stop"))
                print("\033[93m   -> 정지 명령 전송\033[0m")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
