from __future__ import annotations

import copy
import queue
import re
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel


STATIC = Path(__file__).with_name("ui")

#: First-token phonetic net for spoken "자비스" after STT mangling
#: (샤비, 잡이, 차비스...). Onset ㅈ/ㅅ/ㅊ/ㅎ-계열 + second syllable ㅂ-계열.
_WAKE_FUZZY = re.compile(r"^[자쟈저짜사샤차하잡][비삐브바뷔이]")

#: Spoken "go back to sleep" forms ("이제 쉬어", "대기 모드", "잘 자"). Voice
#: STT decorates these with particles/politeness unpredictably, so a regex
#: instead of a fixed set - and the VOICE path must run the same check: an
#: LLM otherwise just answers "네, 쉬세요" and stays awake.
_SLEEP_RE = re.compile(
    r"^(?:자비스\s*)?(?:이제\s*|그만\s*|좀\s*|푹\s*)*"
    r"(?:쉬(?:어(?:라|요)?|어도\s*(?:돼|됩니다)|세요)|쉬고\s*있어|"
    r"대기\s*모드(?:로)?(?:\s*(?:가|들어가|전환(?:해)?))?(?:\s*줘)?|대기|"
    r"잘\s*자(?:요)?|자러\s*가|sleep)$")


class TextTurn(BaseModel):
    text: str


@dataclass(frozen=True)
class UIQueuedTurn:
    text: str


class HologramUIServer:
    """Threaded user/debug UI sharing the copilot's existing input queue."""

    #: Korean spellings of "jarvis". Korean STT never returns the ASCII word,
    #: so a purely ASCII gate silently rejects every spoken wake-up - and the
    #: assistant looks broken while it is merely still asleep. The odd entries
    #: are REAL misrecognitions caught live ("잡이스"...) - none of them are
    #: ordinary Korean words, so they cannot false-wake (which is why "서비스"
    #: is deliberately absent).
    WAKE_ALIASES = ("자비스", "자비스야", "자비쓰", "쟈비스", "재비스", "저비스",
                    "잡이스", "잡비스", "자브스", "자비수", "차비스", "하비스",
                    "자비스어", "자비스으")

    def __init__(self, input_queue: queue.Queue, *, host: str = "127.0.0.1",
                 port: int = 8765, wake_word: str = "jarvis",
                 open_browser: bool = True) -> None:
        self.input_queue = input_queue
        self.host = host
        self.port = port
        self.wake_word = wake_word.strip().lower()
        # Longest first: "자비스야 해머" must not match bare "자비스" and
        # leave "야 해머" as the request.
        self._wake_forms = sorted(
            {self.wake_word, *self.WAKE_ALIASES}, key=len, reverse=True)
        self.open_browser = open_browser
        self._lock = threading.RLock()
        self._frame: bytes | None = None
        #: The step photograph the copilot says it is showing. Held as encoded
        #: bytes rather than a path so the browser never gets a filesystem
        #: read and the panel keeps working if the manual folder moves.
        self._reference: bytes | None = None
        self._reference_media = "image/jpeg"
        self._state: dict[str, Any] = {
            "assistant": {
                "mode": "DORMANT",
                "message": f"{self.wake_word.upper()} 또는 '자비스'라고 부르면 깨어납니다",
                "wake_word": self.wake_word,
                "api_available": False,
                "voice_available": False,
            },
            "assembly": {"active": False, "product": "", "step": None,
                         "total": 0, "title": "", "instruction": "",
                         "status": "대기"},
            "conversation": [],
            # The reference photograph panel: label plus a version the client
            # puts in the image URL, so a new picture is fetched exactly once
            # instead of being polled or cached forever.
            "reference": {"available": False, "label": "", "version": 0},
            # What the microphone last delivered, including utterances the
            # wake gate dropped. Without this the user cannot tell "the mic
            # is dead" from "you are asleep and I ignored you".
            "heard": {"text": "", "gated": False, "at": None},
            "robot": {"available": False, "mode": "off", "busy": False,
                       "state": None, "message": "", "progress": 0.0,
                       "query": None, "error_code": None,
                       "last_outcome": None, "last_outcome_message": ""},
            "system": {"camera": "starting", "started_at": time.time(),
                       "last_frame_at": None, "last_request_at": None,
                       "last_reply_at": None},
        }
        self.app = FastAPI(title="DUM-E Hologram UI", docs_url=None, redoc_url=None)
        self._configure_routes()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def _configure_routes(self) -> None:
        @self.app.get("/")
        def index():
            return FileResponse(STATIC / "index.html")

        @self.app.get("/debug")
        def debug():
            return FileResponse(STATIC / "debug.html")

        @self.app.get("/assets/{name}")
        def asset(name: str):
            path = STATIC / name
            if not path.is_file() or path.parent != STATIC:
                raise HTTPException(404)
            return FileResponse(path)

        @self.app.get("/api/state")
        def state():
            with self._lock:
                return JSONResponse(copy.deepcopy(self._state))

        @self.app.get("/api/frame.jpg")
        def frame():
            with self._lock:
                payload = self._frame
            if payload is None:
                return Response(status_code=204)
            return Response(payload, media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @self.app.get("/api/reference.jpg")
        def reference():
            with self._lock:
                payload = self._reference
                media = self._reference_media
            if payload is None:
                return Response(status_code=204)
            return Response(payload, media_type=media,
                            headers={"Cache-Control": "no-store"})

        @self.app.post("/api/input")
        def text_input(turn: TextTurn):
            text = " ".join(turn.text.strip().split())
            if not text:
                raise HTTPException(400, "empty text")
            with self._lock:
                mode = self._state["assistant"]["mode"]
                self._state["system"]["last_request_at"] = time.time()
            wake_prefix, remainder = self.wake_match(text)
            if mode == "DORMANT":
                if not wake_prefix:
                    return {"accepted": False, "reason": "wake_word_required"}
                self.awaken()
                if remainder:
                    self.add_turn("user", remainder)
                    self.input_queue.put(UIQueuedTurn(remainder))
                return {"accepted": True, "awakened": True, "queued": bool(remainder)}
            if self.sleep_match(text):
                self.sleep()
                return {"accepted": True, "sleeping": True}
            if wake_prefix:
                text = remainder
                if not text:
                    self.add_turn("assistant", "네, 듣고 있어요. 무엇을 도와드릴까요?")
                    return {"accepted": True, "queued": False}
            self.add_turn("user", text)
            self.set_mode("THINKING", "요청을 이해하고 있어요")
            self.input_queue.put(UIQueuedTurn(text))
            return {"accepted": True, "queued": True}

        @self.app.post("/api/sleep")
        def sleep():
            self.sleep()
            return {"ok": True}

    def wake_match(self, text: str) -> tuple[bool, str]:
        """(is wake-prefixed, request remaining after the wake word).

        The ASCII word needs a following space so "jarvistest" is not a
        wake-up; Korean is agglutinative and STT writes "자비스 해머" or
        "자비스해머" unpredictably, so no separator is required there."""
        compact = " ".join(text.strip().split())
        lowered = compact.lower()
        for form in self._wake_forms:
            if lowered == form:
                return True, ""
            if not lowered.startswith(form):
                continue
            rest = compact[len(form):]
            if form.isascii() and rest[:1] not in (" ", ",", ".", "!", "?"):
                continue
            return True, rest.strip(" ,.!?")
        # Phonetic net for what STT actually does to "자비스" spoken in a
        # noisy lab: 샤비, 잡이스, 차비스... - the onset and vowel drift but
        # the shape (ㅈ/ㅅ-계열 + 비/브 + 짧은 단어) survives. Only the FIRST
        # token, and only short ones, so ordinary words ("자비로운", "서비스")
        # stay asleep. A rare false wake merely greets - robot actions still
        # need their own confirmation.
        first, _, tail = compact.partition(" ")
        if 2 <= len(first) <= 3 and _WAKE_FUZZY.match(first):
            return True, tail.strip(" ,.!?")
        return False, compact

    def sleep_match(self, text: str) -> bool:
        """True if `text` is a spoken/typed request to go back to DORMANT."""
        compact = " ".join(text.strip().lower().split()).strip(" .,!?~")
        return bool(_SLEEP_RE.match(compact))

    @property
    def mode(self) -> str:
        with self._lock:
            return self._state["assistant"]["mode"]

    @property
    def url(self) -> str:
        shown_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{shown_host}:{self.port}"

    def start(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port,
                                log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True,
                                        name="dume-hologram-ui")
        self._thread.start()
        if self.open_browser:
            threading.Thread(target=self._open_when_ready, daemon=True).start()

    def _open_when_ready(self) -> None:
        time.sleep(0.8)
        webbrowser.open(self.url)

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def awaken(self) -> None:
        with self._lock:
            self._state["assistant"].update(
                mode="AWAKE", message="온라인 상태 · 말씀하세요")
        self.add_turn("assistant", "안녕하세요. JARVIS 온라인입니다. 무엇을 도와드릴까요?")

    def gate_voice(self, text: str) -> tuple[bool, str]:
        """Apply the same wake gate to Realtime/local transcripts as the UI."""
        compact = " ".join(text.strip().split())
        with self._lock:
            dormant = self._state["assistant"]["mode"] == "DORMANT"
        prefixed, remainder = self.wake_match(compact)
        if dormant and not prefixed:
            return False, ""
        if prefixed:
            if dormant:
                self.awaken()
            return True, remainder
        return True, compact

    def sleep(self) -> None:
        with self._lock:
            self._state["assistant"].update(
                mode="DORMANT",
                message=f"{self.wake_word.upper()} 또는 '자비스'라고 부르면 깨어납니다")

    def set_mode(self, mode: str, message: str | None = None) -> None:
        with self._lock:
            self._state["assistant"]["mode"] = mode
            if message is not None:
                self._state["assistant"]["message"] = message

    def set_capabilities(self, *, api_available: bool, voice_available: bool) -> None:
        with self._lock:
            self._state["assistant"]["api_available"] = api_available
            self._state["assistant"]["voice_available"] = voice_available

    def add_turn(self, role: str, text: str) -> None:
        with self._lock:
            turns = self._state["conversation"]
            turns.append({"role": role, "text": text, "at": time.time()})
            del turns[:-12]
            if role == "assistant":
                self._state["system"]["last_reply_at"] = time.time()

    def update_assembly(self, context: dict[str, Any], *, active: bool | None = None) -> None:
        step = context.get("current_step")
        with self._lock:
            current = self._state["assembly"]
            current.update({
                "active": context.get("work_active", False) if active is None else active,
                "product": context.get("product", ""),
                "step": None if step is None else step.get("order"),
                "total": len(context.get("available_steps", [])),
                "title": "" if step is None else step.get("title", ""),
                "instruction": "" if step is None else step.get("instruction", ""),
                "status": context.get("status", "대기"),
                "completed_steps": context.get("completed_steps", []),
            })

    def set_heard(self, text: str, *, gated: bool) -> None:
        """Record a microphone transcript. `gated` means the wake gate
        dropped it (assistant asleep), so the UI can say so instead of
        looking broken."""
        compact = " ".join(text.strip().split())
        if not compact:
            return
        with self._lock:
            self._state["heard"] = {"text": compact, "gated": gated,
                                    "at": time.time()}

    def set_robot(self, snapshot: dict[str, Any], *, available: bool | None = None,
                  mode: str | None = None) -> None:
        """Robot skill status for both UIs. `snapshot` is
        RobotSkillManager.snapshot(); availability/mode change rarely."""
        with self._lock:
            robot = self._state["robot"]
            robot.update({key: snapshot.get(key) for key in (
                "busy", "state", "message", "progress", "query",
                "error_code", "last_outcome", "last_outcome_message")})
            if available is not None:
                robot["available"] = available
            if mode is not None:
                robot["mode"] = mode

    def set_reference(self, path, label: str = "") -> None:
        """Show (or with path=None clear) the step's reference photograph.

        Read into memory here so the panel cannot break later on a moved
        manual folder, and so the browser is never handed a local path."""
        if path is None:
            with self._lock:
                self._reference = None
                self._state["reference"] = {
                    "available": False, "label": "",
                    "version": self._state["reference"]["version"] + 1}
            return
        source = Path(path)
        try:
            payload = source.read_bytes()
        except OSError:
            # Saying "showing the photo" while showing nothing is the bug
            # this panel exists to fix - so a missing file clears it loudly.
            with self._lock:
                self._reference = None
                self._state["reference"] = {
                    "available": False,
                    "label": f"{label} (파일을 읽지 못했습니다)".strip(),
                    "version": self._state["reference"]["version"] + 1}
            return
        with self._lock:
            self._reference = payload
            self._reference_media = ("image/png"
                                     if source.suffix.lower() == ".png"
                                     else "image/jpeg")
            self._state["reference"] = {
                "available": True, "label": label or source.name,
                "version": self._state["reference"]["version"] + 1}

    def update_frame(self, image) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return
        with self._lock:
            self._frame = encoded.tobytes()
            self._state["system"]["camera"] = "online"
            self._state["system"]["last_frame_at"] = time.time()

    def set_error(self, message: str) -> None:
        self.set_mode("ERROR", message)
        self.add_turn("assistant", message)
