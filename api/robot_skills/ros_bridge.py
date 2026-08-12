"""Bridge to the physical webcam pick&place ROS app (separate process).

The physical node is `pick_and_place_voice webcam_pick_place` running in its
own process with its own safety stack. This bridge only:

- publishes std_msgs/String on /webcam_pnp/command
    "start" (JSON {"command":"start","request_id":...} when available)
    "stop"
- subscribes to /webcam_pnp/status ("[STATE] human message") and
  /webcam_pnp/state_json ({"state","status","request_id","done",...})

It never streams motion. rclpy is imported lazily: without ROS the bridge
reports unavailable and skills fail closed with DRIVER_DISCONNECTED.
"""
from __future__ import annotations

import json
import re
import threading
import time


class WebcamPnPBridge:
    STATUS_PATTERN = re.compile(r"^\[([A-Z_]+)\]\s*(.*)$")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node = None
        self._executor = None
        self._thread = None
        self._command_pub = None
        self.available = False
        self._last_state: str | None = None
        self._last_message: str = ""
        self._last_json: dict = {}
        self._last_heard_at: float = 0.0
        self._inspection: dict | None = None
        self._inspection_at: float = 0.0

    # ------------------------------------------------------------------
    def _inspection_cb(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._inspection = payload
            self._inspection_at = time.monotonic()

    def take_inspection(self, request_id=None, max_age: float = 120.0):
        """The most recent inspection result, consumed once.

        Consumed rather than read so a stale success cannot answer the next
        question - "is this right?" must never be answered with the previous
        photograph."""
        with self._lock:
            payload = self._inspection
            age = time.monotonic() - self._inspection_at
            if payload is None or age > max_age:
                return None
            if request_id and payload.get("request_id") not in (None, request_id):
                return None
            self._inspection = None
            return payload

    def connect(self, timeout: float = 5.0) -> bool:
        if self.available:
            return True
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from std_msgs.msg import String as StringMsg
        except Exception:
            return False
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("robot_skill_bridge")
            self._command_pub = self._node.create_publisher(
                StringMsg, "/webcam_pnp/command", 10)
            self._node.create_subscription(
                StringMsg, "/webcam_pnp/status", self._status_cb, 10)
            self._node.create_subscription(
                StringMsg, "/webcam_pnp/state_json", self._state_json_cb, 10)
            self._node.create_subscription(
                StringMsg, "/webcam_pnp/inspection", self._inspection_cb, 10)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(
                target=self._executor.spin, daemon=True,
                name="robot-skill-bridge")
            self._thread.start()
        except Exception:
            return False
        # The physical app publishes a status line on every change; probe
        # liveness by waiting briefly for either topic to be discovered.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._node.count_subscribers("/webcam_pnp/command") > 0:
                break
            time.sleep(0.1)
        self.available = True
        return True

    def app_alive(self) -> bool:
        """The physical app is subscribed to our command topic."""
        if not self.available or self._node is None:
            return False
        try:
            return self._node.count_subscribers("/webcam_pnp/command") > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _status_cb(self, msg) -> None:
        match = self.STATUS_PATTERN.match(msg.data.strip())
        if match is None:
            return
        with self._lock:
            self._last_state = match.group(1)
            self._last_message = match.group(2)
            self._last_heard_at = time.monotonic()

    def _state_json_cb(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._last_json = payload
            if "state" in payload:
                self._last_state = str(payload["state"])
                self._last_message = str(payload.get("status", ""))
            self._last_heard_at = time.monotonic()

    def latest(self) -> tuple[str | None, str, dict, float]:
        """(app_state, message, json_payload, seconds_since_heard)"""
        with self._lock:
            age = (time.monotonic() - self._last_heard_at
                   if self._last_heard_at else float("inf"))
            return self._last_state, self._last_message, dict(self._last_json), age

    # ------------------------------------------------------------------
    def send_start(self, request_id: str | None = None,
                   destination: str | None = None) -> None:
        if request_id or destination:
            self._publish(json.dumps({
                "command": "start", "request_id": request_id,
                "destination": destination}))
        else:
            self._publish("start")

    def send_stop(self) -> None:
        self._publish("stop")

    def send_command(self, command: str) -> None:
        """Quick operator-style command: home / gripper_open / gripper_close."""
        self._publish(command)

    def send_inspect(self, request_id: str | None = None,
                     point_mm=None, standoff_mm=None) -> None:
        """Photograph the spot the operator is pointing at.

        With no point, the app uses whatever the finger is aimed at right
        now - which is the whole point of asking "is this right?" while
        pointing at the thing in question."""
        payload = {"command": "inspect", "request_id": request_id}
        if point_mm is not None:
            payload["point"] = [float(v) for v in point_mm]
        if standoff_mm:
            payload["standoff_mm"] = float(standoff_mm)
        self._publish(json.dumps(payload))

    def send_take_from_hand(self, request_id: str | None = None) -> None:
        """Take whatever is on the operator's hand and put it down."""
        self._publish(json.dumps({"command": "take_from_hand",
                                  "request_id": request_id}))

    def send_inspect_done(self) -> None:
        """Release the arm from the viewing pose.

        The app holds there after a shot so a second look costs no travel;
        without this it would hold for the full timeout after every verdict,
        which reads as the robot freezing over the operator's work."""
        self._publish(json.dumps({"command": "inspect_done"}))

    def _publish(self, data: str) -> None:
        if self._command_pub is None:
            return
        from std_msgs.msg import String as StringMsg
        self._command_pub.publish(StringMsg(data=data))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        self.available = False
