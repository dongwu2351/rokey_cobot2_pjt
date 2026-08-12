from __future__ import annotations

import audioop
import base64
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from typing import Any


@dataclass(frozen=True)
class RealtimeToolRequest:
    call_id: str
    utterance: str
    decision: str = "EXECUTE"
    clarification_question: str | None = None
    intent_hint: str = "PASS_THROUGH"
    interpreted_utterance: str | None = None
    speech_act: str = "PASS_THROUGH"
    target_step_order: int | None = None
    tool_name: str = "dispatch_user_turn"


@dataclass(frozen=True)
class RealtimeDirectReply:
    utterance: str
    reply: str


REALTIME_INSTRUCTIONS = """# 역할
당신은 중앙 코파일럿의 실시간 음성 입출력 계층이다. 모든 사용자 발화는 예외 없이
dispatch_user_turn 도구로 전달한다. 직접 답변하거나, 의도를 분류하거나, 시스템 상태를
추측하지 않는다. 자연스러운 답변 내용은 중앙 Turn Manager의 도구 결과로만 생성된다.

# 도구 경계
- 안부, 잡담, 조립, 비전, PDF, 검색, 종료, 취소를 포함한 모든 발화에
  dispatch_user_turn을 호출한다.
- utterance에는 STT로 들은 원문을 의역하지 않고 그대로 넣는다.
- '~라는 말씀이시죠'를 붙이거나 누락된 문맥을 임의로 보충하지 않는다.
- 도구를 호출하지 않는 음성 답변은 금지한다.

# 말투와 성격
- 따뜻하고 차분하며 눈치 빠른 동료처럼 말한다.
- 보고서 낭독체나 로봇 명령 확인체를 피하고 일상적인 구어체를 쓴다.
- 사용자의 말을 매번 되풀이하거나 '~라는 말씀이시죠'라고 확인하지 않는다.
- 보통 1~3문장으로 답하되, 실제 작업 절차에는 필요한 세부 내용을 생략하지 않는다.
- 같은 시작 문구와 문장 구조를 반복하지 말고 자연스럽게 표현을 바꾼다.
- 애매하지 않은 요청에는 확인 질문을 덧붙이지 않는다.

# 도구 결과 말하기
도구 결과가 돌아오면 text의 사실, 단계 번호, 안전 조건과 상태를 바꾸지 않는다.
다만 딱딱한 문장은 의미를 보존하면서 자연스러운 한국어 음성 문장으로 다듬는다.
내부 intent, 도구 이름, JSON은 말하지 않는다. confidence는 사용자가 판단 근거를
물었거나 결과 이해에 꼭 필요한 경우에만 말한다."""


TOOL = {
    "type": "function",
    "name": "dispatch_user_turn",
    "description": "모든 사용자 발화 원문을 중앙 Turn Manager에 전달한다.",
    "parameters": {
        "type": "object",
        "properties": {
            "utterance": {"type": "string", "description": "사용자가 말한 핵심 한국어 문장"},
        },
        "required": ["utterance"],
        "additionalProperties": False,
    },
}


def _quiet_native_stderr() -> None:
    if os.getenv("DUME_VERBOSE_LOGS", "false").lower() in {"1", "true", "yes", "on"}:
        return
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 2)
    os.close(fd)


def _realtime_process_main(requests, results, stop, *, api_key: str, model: str,
                           voice: str, eagerness: str, device_index: int | None,
                           capture_rate: int, initial_context: dict[str, Any],
                           speaker_gate=None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _quiet_native_stderr()
    import pyaudio
    from websockets.sync.client import connect

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    audio = pyaudio.PyAudio()
    input_stream = output_stream = None
    playback: queue.Queue[bytes | None] = queue.Queue(maxsize=128)
    send_lock = threading.Lock()
    response_active = threading.Event()
    playback_active = threading.Event()
    server_audio_done = threading.Event()
    server_audio_done.set()
    resample_state = None

    def send(ws, event: dict[str, Any]) -> None:
        with send_lock:
            ws.send(json.dumps(event, ensure_ascii=False))

    try:
        input_args = dict(format=pyaudio.paInt16, channels=1, rate=capture_rate,
                          input=True, frames_per_buffer=max(1, capture_rate // 50))
        if device_index is not None:
            input_args["input_device_index"] = device_index
        input_stream = audio.open(**input_args)
        output_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=24_000,
                                   output=True, frames_per_buffer=1200)
        with connect(url, additional_headers=headers, open_timeout=15,
                     max_size=8 * 1024 * 1024) as ws:
            instructions = REALTIME_INSTRUCTIONS + "\n현재 작업 문맥:\n" + json.dumps(
                initial_context, ensure_ascii=False)
            send(ws, {
                "type": "session.update",
                "session": {
                    "type": "realtime", "model": model,
                    "instructions": instructions,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24_000},
                            "transcription": {
                                "model": "gpt-4o-mini-transcribe",
                                "language": "ko",
                            },
                            "turn_detection": {
                                "type": "semantic_vad", "eagerness": eagerness,
                                "create_response": True, "interrupt_response": True,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24_000},
                            "voice": voice,
                        },
                    },
                    "tools": [TOOL], "tool_choice": "required",
                },
            })

            def capture_loop() -> None:
                nonlocal resample_state
                while not stop.is_set():
                    try:
                        pcm = input_stream.read(max(1, capture_rate // 50),
                                                exception_on_overflow=False)
                        # Keep streaming microphone audio during playback. The
                        # Realtime session has server-side VAD and
                        # interrupt_response enabled, so locally dropping every
                        # frame behind playback_active both disables barge-in and
                        # can silence all later turns if an audio completion event
                        # is delayed or omitted.
                        if capture_rate != 24_000:
                            pcm, resample_state = audioop.ratecv(
                                pcm, 2, 1, capture_rate, 24_000, resample_state)
                        send(ws, {"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(pcm).decode("ascii")})
                    except Exception:
                        stop.set()
                        break

            def playback_loop() -> None:
                while not stop.is_set():
                    try:
                        chunk = playback.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        continue
                    # While the assistant is DORMANT the model still generates
                    # speech on its own ("잠깐만요…"). Playing it (a) is a ghost
                    # voice from a supposedly sleeping assistant and (b) feeds
                    # the laptop speakers straight back into the internal mic,
                    # which the model answers again - an audible feedback loop.
                    # The parent gates the speaker; dropped audio still counts
                    # as "played" for the mute bookkeeping below.
                    if speaker_gate is not None and not speaker_gate.value:
                        if server_audio_done.is_set() and playback.empty():
                            playback_active.clear()
                        continue
                    try:
                        playback_active.set()
                        output_stream.write(chunk)
                    except Exception:
                        stop.set()
                        break
                    finally:
                        # Queue emptiness between streaming deltas does not mean
                        # the server finished the utterance. Only release the
                        # microphone after output_audio.done and local drain.
                        if server_audio_done.is_set() and playback.empty():
                            playback_active.clear()

            def result_loop() -> None:
                while not stop.is_set():
                    try:
                        item = results.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    wait_started = time.monotonic()
                    while response_active.is_set() and not stop.wait(0.05):
                        if time.monotonic() - wait_started > 20.0:
                            # A missing response.done must not permanently lock
                            # all later microphone turns.
                            response_active.clear()
                            break
                    if stop.is_set():
                        break
                    send(ws, {"type": "conversation.item.create", "item": {
                        "type": "function_call_output", "call_id": item["call_id"],
                        "output": json.dumps(item["output"], ensure_ascii=False),
                    }})
                    # The session requires a tool for user turns. This follow-up
                    # response must speak the tool result instead of recursively
                    # issuing another tool call.
                    response_active.set()
                    send(ws, {"type": "response.create", "response": {
                        "tool_choice": "none",
                        "metadata": {"source": "tool_result",
                                     "call_id": item["call_id"]},
                        "instructions": (
                            "도구 결과의 사실과 상태는 그대로 유지하되, 따뜻하고 자연스러운 "
                            "한국어 구어체로 전달하세요. 사용자의 말을 되풀이하지 말고, "
                            "보고서처럼 읽지 마세요. 새 도구 호출이나 상태 추측은 금지합니다."),
                    }})

            threads = [threading.Thread(target=capture_loop, daemon=True),
                       threading.Thread(target=playback_loop, daemon=True),
                       threading.Thread(target=result_loop, daemon=True)]
            for thread in threads:
                thread.start()

            direct_transcripts: dict[str, str] = {}
            # Last thing the USER was heard to say. The model is supposed to
            # put it in the tool's `utterance` argument, but when it omits it
            # the turn used to arrive empty and was discarded before the wake
            # gate - so "자비스" never woke anything and the only sign of life
            # was the model answering by voice. Consumed once so a stale line
            # can never be attached to a later turn.
            heard_speech: dict[str, Any] = {"text": "", "at": 0.0}

            def take_heard(max_age: float = 20.0) -> str:
                if not heard_speech["text"]:
                    return ""
                if time.monotonic() - heard_speech["at"] > max_age:
                    return ""
                text = heard_speech["text"]
                heard_speech["text"] = ""
                return text

            def emit_request(args: dict[str, Any], call_id: str, transcript: str,
                             tool_name: str) -> None:
                # The function argument belongs to this exact response/call_id.
                # A separate transcription event can arrive before or after it,
                # so FIFO-pairing the two shifts utterances between turns.
                raw = str(args.get("utterance", "")).strip() or transcript.strip()
                requests.put(RealtimeToolRequest(
                    call_id=call_id,
                    utterance=raw,
                    decision=str(args.get("decision", "EXECUTE")),
                    clarification_question=args.get("clarification_question"),
                    intent_hint=str(args.get("intent_hint", "PASS_THROUGH")),
                    interpreted_utterance=str(args.get("utterance", "")).strip() or None,
                    speech_act=str(args.get("speech_act", "PASS_THROUGH")),
                    target_step_order=(int(args["target_step_order"])
                                       if args.get("target_step_order") is not None else None),
                    tool_name=tool_name,
                ))

            def dispatch_turn(turn: dict[str, Any], transcript: str) -> None:
                if turn["kind"] == "tool":
                    emit_request(turn["args"], turn["call_id"], transcript,
                                 turn["tool_name"])
                else:
                    requests.put(RealtimeDirectReply(transcript.strip(), turn["reply"]))

            while not stop.is_set():
                try:
                    event = json.loads(ws.recv(timeout=0.25))
                except TimeoutError:
                    continue
                kind = event.get("type")
                if kind == "input_audio_buffer.speech_started":
                    # semantic_vad already has interrupt_response enabled.
                    # Sending response.cancel here races the server and produces
                    # "no active response found" errors after normal completion.
                    response_active.clear()
                    while True:
                        try:
                            playback.get_nowait()
                        except queue.Empty:
                            break
                    playback_active.clear()
                elif kind == "conversation.item.input_audio_transcription.completed":
                    # Kept only as a FALLBACK for a tool call that carries no
                    # utterance. It never overrides an utterance the model did
                    # provide, so turns cannot be cross-paired - and an empty
                    # turn is no longer silently dropped.
                    spoken = str(event.get("transcript", "")).strip()
                    if spoken:
                        heard_speech["text"] = spoken
                        heard_speech["at"] = time.monotonic()
                elif kind == "conversation.item.input_audio_transcription.failed":
                    pass
                elif kind == "response.created":
                    response_active.set()
                elif kind == "response.output_audio.delta":
                    server_audio_done.clear()
                    playback_active.set()
                    try:
                        playback.put_nowait(base64.b64decode(event.get("delta", "")))
                    except queue.Full:
                        pass
                elif kind == "response.output_audio.done":
                    server_audio_done.set()
                    if playback.empty():
                        playback_active.clear()
                elif kind == "response.output_audio_transcript.done":
                    direct_transcripts[str(event.get("response_id", ""))] = str(
                        event.get("transcript", "")).strip()
                elif kind == "response.done":
                    response_active.clear()
                    # This is the authoritative end of a response. Some
                    # sessions/transports may omit a separate output_audio.done
                    # event; without this fallback capture_loop can remain
                    # muted forever after the first spoken response.
                    server_audio_done.set()
                    if playback.empty():
                        playback_active.clear()
                    response = event.get("response") or {}
                    function_calls = []
                    for item in response.get("output", []):
                        if item.get("type") != "function_call":
                            continue
                        try:
                            args = json.loads(item.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        function_calls.append({
                            "kind": "tool", "args": args,
                            "call_id": str(item.get("call_id", "")),
                            "tool_name": str(item.get("name", "dispatch_user_turn")),
                        })
                    for call in function_calls:
                        dispatch_turn(call, take_heard())
                    metadata = response.get("metadata") or {}
                    if not function_calls and metadata.get("source") != "tool_result":
                        # Barge-in commonly leaves an interrupted response with
                        # no function call. It is not a user-visible protocol
                        # error and must not create a phantom turn.
                        direct_transcripts.pop(str(response.get("id", "")), None)
                    elif metadata.get("source") == "tool_result":
                        direct_transcripts.pop(str(response.get("id", "")), None)
                elif kind == "error":
                    response_active.clear()
                    detail = (event.get("error") or {}).get("message", "Realtime 오류")
                    requests.put(RealtimeToolRequest("", "", "ERROR", str(detail)))
    except Exception as exc:
        requests.put(RealtimeToolRequest("", "", "ERROR", f"Realtime 연결 실패: {exc}"))
    finally:
        stop.set()
        for stream in (input_stream, output_stream):
            if stream is not None:
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
        audio.terminate()


class RealtimeVoiceProcess:
    """Process-isolated OpenAI Realtime audio frontend and tool bridge."""

    def __init__(self, *, device_index: int | None, sample_rate: int,
                 initial_context: dict[str, Any], model: str | None = None,
                 voice: str | None = None, eagerness: str = "low") -> None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY가 없어 Realtime 음성을 시작할 수 없습니다")
        context = mp.get_context("spawn")
        self.questions = context.Queue()
        self._results = context.Queue()
        self._stop = context.Event()
        # Speaker gate, owned by the parent: closed while the assistant is
        # DORMANT so the model's self-initiated speech never reaches the
        # speakers (or the microphone again). Starts closed.
        self._speaker_gate = context.Value("b", False)
        self.process = context.Process(
            target=_realtime_process_main,
            kwargs={"requests": self.questions, "results": self._results,
                    "stop": self._stop, "api_key": key,
                    "model": model or os.getenv("REALTIME_MODEL", "gpt-realtime-2.1"),
                    "voice": voice or os.getenv("REALTIME_VOICE", "marin"),
                    "eagerness": eagerness, "device_index": device_index,
                    "capture_rate": sample_rate, "initial_context": initial_context,
                    "speaker_gate": self._speaker_gate},
            daemon=True, name="dume-realtime-voice")

    def set_speaker_open(self, is_open: bool) -> None:
        self._speaker_gate.value = bool(is_open)

    def start(self) -> None:
        self.process.start()

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def submit_result(self, call_id: str, output: dict[str, Any]) -> None:
        if call_id:
            self._results.put({"call_id": call_id, "output": output})

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def close(self) -> None:
        self._stop.set()
        self._results.put(None)
        self.process.join(timeout=4)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(timeout=2)
        self.process.close()
        self.questions.close(); self._results.close()
