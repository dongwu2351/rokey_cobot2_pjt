#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import re
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

from assembly_copilot.ar_overlay import draw_overlay
from assembly_copilot.frame_buffer import RecentFrameBuffer
from assembly_copilot.manual import load_manual
from assembly_copilot.models import AssemblyObservation
from assembly_copilot.realsense_source import OpenCVSource, RealSenseSource, SyntheticSource
from assembly_copilot.ros_camera_source import RosCameraSource
from assembly_copilot.voice_io import CopilotSpeech, VoiceQuestionProcess

from .engine import CopilotReply, UnifiedCopilotEngine
from .realtime_voice import RealtimeDirectReply, RealtimeToolRequest, RealtimeVoiceProcess
from .turn_manager import CentralTurnManager
from .ui_server import HologramUIServer, UIQueuedTurn

ROOT = Path(__file__).resolve().parents[1]


def _verbose_logs() -> bool:
    return os.getenv("DUME_VERBOSE_LOGS", "false").lower() in {"1", "true", "yes", "on"}


def _debug(message: str, *, error: bool = False) -> None:
    if _verbose_logs():
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


def _request_error_reply(exc: Exception) -> CopilotReply:
    """Return a short user-safe error while retaining details in stderr."""
    detail = str(exc)
    _debug(f"[요청 처리 오류] {type(exc).__name__}: {detail}", error=True)
    if ("Invalid JSON" in detail or "EOF while parsing" in detail
            or "output_parsed" in detail):
        return CopilotReply(
            "시각 판정 응답이 완성되기 전에 끊겼습니다. 작업물 전체가 보이게 한 뒤 "
            "현재 단계를 다시 물어봐 주세요.", "ERROR")
    if ("timeout" in detail.lower() or "timed out" in detail.lower()
            or type(exc).__name__ == "APITimeoutError"):
        return CopilotReply(
            "시각 판정 시간이 초과됐습니다. 네트워크 상태를 확인한 뒤 다시 물어봐 "
            "주세요. 필요하면 ASSEMBLY_VISION_TIMEOUT_SECONDS 값을 늘릴 수 있습니다.",
            "ERROR")
    return CopilotReply(
        "요청을 처리하지 못했습니다. 같은 질문을 다시 말씀해 주세요.", "ERROR")


def arguments():
    parser = argparse.ArgumentParser(
        description="자유 대화·조립 Vision·PDF 생성을 합친 단일 DUM-E 코파일럿")
    parser.add_argument(
        "--manual", type=Path,
        default=(ROOT / "assembly_manuals/conveyor-motor-timing-belt-assembly"
                 / "assembly.yaml"))
    parser.add_argument("--camera",
                        choices=("auto", "realsense", "ros", "opencv", "off"),
                        default="auto",
                        help="ros: realsense2_camera가 이미 물고 있는 카메라를 "
                             "토픽으로 공유해서 사용")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--voice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voice-backend", choices=("realtime", "local"),
                        default=os.getenv("DUME_VOICE_BACKEND", "realtime"),
                        help="realtime: 자연스러운 대화/VAD, local: 기존 녹음+STT")
    parser.add_argument("--no-wake", action="store_true")
    parser.add_argument("--device-index", type=int, default=12,
                        help="마이크 입력 장치(현재 시스템 Pulse: 12)")
    parser.add_argument("--sample-rate", type=int, default=16_000,
                        help="16kHz에서 Silero VAD 사용(권장)")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--end-silence-ms", type=int, default=450)
    parser.add_argument("--realtime-model",
                        default=os.getenv("REALTIME_MODEL", "gpt-realtime-2.1"))
    parser.add_argument("--realtime-voice",
                        default=os.getenv("REALTIME_VOICE", "marin"))
    parser.add_argument("--realtime-eagerness", choices=("low", "medium", "high", "auto"),
                        default=os.getenv("REALTIME_EAGERNESS", "low"),
                        help="semantic VAD 발화 종료 민감도(충분히 듣기는 low 권장)")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--tts-voice", default="marin",
                        help="TTS 음성 이름(예: marin, cedar, onyx, coral)")
    parser.add_argument(
        "--tts-style",
        default=("한국어로 짧고 또렷하게 말하세요. 친절하지만 과장하지 말고, "
                 "숙련된 작업 코치처럼 차분하고 자연스럽게 안내하세요."),
        help="TTS 말투·감정·억양 지시문")
    parser.add_argument("--tts-speed", type=float, default=1.0,
                        help="TTS 속도 0.25~4.0")
    parser.add_argument("--display", choices=("on-demand", "always", "off"),
                        default="on-demand",
                        help="AR 창 표시 정책(기본: 음성 요청 시 표시)")
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True,
                        help="홀로그램 사용자 UI와 /debug 진단 UI 실행")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)
    parser.add_argument("--open-ui", action=argparse.BooleanOptionalAction, default=True,
                        help="시작할 때 기본 브라우저로 사용자 UI 열기")
    parser.add_argument("--wake-word", default="jarvis",
                        help="UI와 음성 전면에서 사용할 호출어")
    parser.add_argument("--wake-word-model", type=Path,
                        help="local 음성 백엔드용 openWakeWord TFLite/ONNX 모델")
    parser.add_argument("--inspect-vision", choices=("auto", "mock", "openai"),
                        default="auto",
                        help="사진 판정 백엔드: mock은 토큰을 쓰지 않고 "
                             "전 구간을 그대로 시연/검증한다")
    parser.add_argument("--robot", choices=("auto", "mock", "ros", "off"),
                        default="auto",
                        help="fetch_object 로봇 스킬 백엔드: auto는 rclpy가 "
                             "있으면 실제 브리지, 없으면 mock")
    parser.add_argument("--vision-model")
    parser.add_argument("--history-seconds", type=float, default=8.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "copilot_data")
    parser.add_argument("--downloaded-dir", type=Path, default=ROOT / "downloaded_manuals")
    parser.add_argument("--manuals-dir", type=Path, default=ROOT / "assembly_manuals")
    return parser.parse_args()


def _open_source(args):
    if args.camera == "off":
        return SyntheticSource()
    if args.video:
        return OpenCVSource(str(args.video))
    if args.camera == "opencv":
        return OpenCVSource(args.camera_index)
    if args.camera == "ros":
        return RosCameraSource()
    if args.camera == "realsense":
        return RealSenseSource()
    try:
        return RealSenseSource()
    except RuntimeError as exc:
        # The usual reason the device will not open is that realsense2_camera
        # already has it - and then its topics are exactly the picture we
        # want, at no cost to the robot that depends on them.
        _debug(f"[카메라] RealSense 직접 열기 실패, ROS 토픽으로 전환합니다: {exc}")
        try:
            return RosCameraSource()
        except RuntimeError as ros_exc:
            _debug(f"[카메라] ROS 토픽도 실패, OpenCV {args.camera_index}번 사용: "
                   f"{ros_exc}", error=True)
            return OpenCVSource(args.camera_index)


class ReferenceWindow:
    """The step's reference photograph in its own resizable OS window.

    Separate from the web page on purpose: the operator wants the picture
    beside their hands (or on a second monitor) while the live feed keeps the
    page. It opens and closes on the display commands that already exist, so
    "화면 닫아" really closes something again.
    """

    TITLE = "JARVIS REFERENCE"
    #: Long side, in pixels. Big enough to read a bolt from arm's length,
    #: small enough not to cover the browser it sits next to.
    MAX_SIDE = 900

    def __init__(self) -> None:
        self._open = False
        self._path: str | None = None
        self._image = None
        self._label = ""

    def show(self, path, label: str = "") -> None:
        """Draw `path` (loading it only when it changes)."""
        if path is None:
            self.close()
            return
        if str(path) != self._path:
            image = cv2.imread(str(path))
            if image is None:
                _debug(f"[참고 사진] 이미지를 열지 못했습니다: {path}", error=True)
                self.close()
                return
            height, width = image.shape[:2]
            scale = min(1.0, self.MAX_SIDE / max(height, width))
            if scale < 1.0:
                image = cv2.resize(image, (round(width * scale),
                                           round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            self._image = image
            self._path = str(path)
            self._label = label
            # A fresh picture in the same window would otherwise keep the old
            # window size, letterboxing a portrait shot inside a landscape box.
            if self._open:
                cv2.resizeWindow(self.TITLE, image.shape[1], image.shape[0])
        if not self._open:
            cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.TITLE, self._image.shape[1],
                             self._image.shape[0])
            self._open = True
        cv2.imshow(self.TITLE, self._image)

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.TITLE)
            except cv2.error:
                pass
            self._open = False
        self._path = None
        self._image = None

    @property
    def open(self) -> bool:
        return self._open


def _assembly_context(engine: UnifiedCopilotEngine) -> dict:
    """Expose only authoritative state needed by the Realtime dialogue layer."""
    state = engine.tracker.snapshot()
    step = engine.tracker.step
    db = getattr(engine, "db", None)
    pending = db.pending() if db is not None else None
    return {
        "product": engine.tracker.manual.product,
        "manual_id": engine.tracker.manual.manual_id,
        "current_step": None if step is None else {
            "id": step.id, "order": step.order, "title": step.title,
            "instruction": step.instruction,
        },
        "completed_steps": list(state.completed_steps),
        "available_steps": [
            {"id": item.id, "order": item.order, "title": item.title}
            for item in engine.tracker.manual.steps
        ],
        "status": state.status,
        "work_active": getattr(engine, "work_session_active", False),
        "pending_confirmation": None if pending is None else {
            "type": pending["operation_type"],
            "payload": pending["payload"],
        },
        "rule": "단계 변경과 완료 승인은 작업 코파일럿 도구만 수행한다.",
    }


def _submit_realtime_result(listener, call_id: str | None,
                            reply: CopilotReply, engine: UnifiedCopilotEngine) -> None:
    if call_id and hasattr(listener, "submit_result"):
        listener.submit_result(call_id, {
            "text": reply.text,
            "domain": reply.domain,
            "stop": reply.stop,
            "assembly_context": _assembly_context(engine),
        })


def _request_label(number: int | str, utterance: str) -> str:
    compact = " ".join(utterance.split())
    if len(compact) > 42:
        compact = compact[:39] + "..."
    return f'요청 {number} · "{compact}"'


def main() -> int:
    args = arguments()
    env_path = ROOT / "llm" / "VoiceProcessing" / ".env"
    load_dotenv(env_path)
    if not 0.25 <= args.tts_speed <= 4.0:
        raise ValueError("--tts-speed는 0.25에서 4.0 사이여야 합니다")
    os.environ["TTS_VOICE"] = args.tts_voice
    os.environ["TTS_INSTRUCTIONS"] = args.tts_style
    os.environ["TTS_SPEED"] = str(args.tts_speed)
    if args.wake_word_model is not None:
        wake_model = args.wake_word_model.resolve()
        if not wake_model.is_file():
            raise FileNotFoundError(f"wake-word model not found: {wake_model}")
        os.environ["WAKE_WORD_MODEL_PATH"] = str(wake_model)
    api_available = bool(os.getenv("OPENAI_API_KEY"))
    if args.voice and not api_available:
        # The hologram text console is a first-class fallback, not an error
        # screen. Voice can be enabled later without changing the UI flow.
        args.voice = False
        print("JARVIS> OPENAI_API_KEY가 없어 음성 Realtime을 비활성화했습니다. "
              "홀로그램 UI의 텍스트 입력으로 로컬 기능을 테스트할 수 있습니다.",
              flush=True)
    manual_path = args.manual.resolve()
    manual = load_manual(manual_path)
    _debug(f"[활성 매뉴얼] {manual.product}")
    _debug(f"[매뉴얼 ID] {manual.manual_id} / version={manual.version}")
    _debug(f"[매뉴얼 경로] {manual_path}")
    source = _open_source(args)
    if isinstance(source, RealSenseSource):
        _debug(f"[카메라] RealSense 프로파일: {source.profile_label}")
    history = RecentFrameBuffer(seconds=args.history_seconds)
    engine = UnifiedCopilotEngine(
        manual, data_dir=args.data_dir, downloaded_dir=args.downloaded_dir,
        manuals_dir=args.manuals_dir, vision_model=args.vision_model)
    # Physical robot skills: the manager is the only execution authority;
    # the engine only files semantic requests and confirmations into it.
    from robot_skills import create_manager as create_robot_manager
    from robot_skills.step_context import TrackerStepContext
    robot_events: queue.Queue = queue.Queue()
    robot_manager, robot_mode = create_robot_manager(
        args.robot, on_feedback=robot_events.put, on_result=robot_events.put,
        vision=args.inspect_vision, manual=manual,
        step_context=TrackerStepContext(engine.tracker))
    engine.robot_skills = robot_manager
    print(f"JARVIS> 로봇 스킬 백엔드: {robot_mode}"
          f"  |  사진 판정: {args.inspect_vision}", flush=True)
    turn_manager = CentralTurnManager(engine)
    questions = queue.Queue()
    listener = None
    if args.voice:
        # Audio and RealSense stay process-isolated. Realtime performs only
        # dialogue/VAD; all authoritative work is delegated back to engine.
        if args.voice_backend == "realtime":
            listener = RealtimeVoiceProcess(
                device_index=args.device_index, sample_rate=args.sample_rate,
                initial_context=_assembly_context(engine), model=args.realtime_model,
                voice=args.realtime_voice, eagerness=args.realtime_eagerness)
        else:
            listener = VoiceQuestionProcess(
                no_wake=args.no_wake, device_index=args.device_index,
                sample_rate=args.sample_rate, vad_threshold=args.vad_threshold,
                end_silence_ms=args.end_silence_ms)
        questions = listener.questions
    ui = None
    if args.ui:
        ui = HologramUIServer(
            questions, host=args.ui_host, port=args.ui_port,
            wake_word=args.wake_word, open_browser=args.open_ui)
        ui.set_capabilities(api_available=api_available,
                            voice_available=bool(args.voice))
        ui.update_assembly(_assembly_context(engine))
        ui.set_robot(robot_manager.snapshot(),
                     available=(robot_mode != "off"), mode=robot_mode)
        ui.start()
        print(f"JARVIS UI> {ui.url}  |  diagnostics: {ui.url}/debug", flush=True)
    # Realtime already streams its own speech after receiving the tool result.
    speech = (CopilotSpeech(True, listener=listener)
              if args.speak and args.voice_backend == "local" else None)
    if listener:
        listener.start()
    dialogue_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="copilot-dialogue")
    perception_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="copilot-perception")
    background_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="copilot-background")
    pending_futures: dict[
        concurrent.futures.Future, tuple[int, str, str, str | None]
    ] = {}
    ordered_pending: set[int] = set()
    ready_replies: dict[int, tuple[CopilotReply, str, str, str | None]] = {}
    background_future: concurrent.futures.Future | None = None
    request_number = 0
    message = "통합 코파일럿이 준비됐습니다. 자유롭게 말씀해 주세요."
    should_stop = False
    exit_reason = "입력 스트림 종료"
    display_visible = args.display == "always"
    window_created = False
    reference_image = None
    reference_label = None
    camera_misses = 0
    # The reference photograph also gets its own OS window, so it can be
    # dragged onto a second screen and stay open while the operator works -
    # a panel inside the page competes with the live feed for the same space.
    # It answers the display commands that already exist: "화면 보여줘" opens
    # it, "화면 닫아" closes it.
    reference_window = ReferenceWindow()
    _debug("통합 코파일럿: 자유 대화 / 조립 판단 / 물체 설명 / PDF 매뉴얼 생성")
    _debug("화면은 기본적으로 숨겨지며 '화면 보여줘'로 엽니다.")
    _debug("종료: '작업 종료해줘' 음성 명령 또는 Ctrl+C")
    try:
        voice_failure_reported = False
        while not should_stop:
            if (listener is not None and not listener.is_alive()
                    and not voice_failure_reported):
                print("코파일럿> 음성 입력 프로세스가 종료되었습니다. 프로그램을 다시 "
                      "실행해 주세요. 상세 원인은 DUME_VERBOSE_LOGS=true로 확인할 수 "
                      "있습니다.", flush=True)
                voice_failure_reported = True
            packet = source.read()
            if packet is None:
                camera_misses += 1
                live_camera = isinstance(source, (RealSenseSource, RosCameraSource))
                if live_camera and camera_misses < 10:
                    if camera_misses == 1:
                        _debug("[카메라] 프레임 준비 중입니다. 카메라 입력을 재시도합니다.")
                    time.sleep(0.1)
                    continue
                if live_camera:
                    _debug("[카메라 오류] 카메라 프레임을 10회 연속 받지 못해 "
                           "안전하게 종료합니다. USB 연결과 카메라 점유 프로세스를 "
                           "확인해 주세요.", error=True)
                else:
                    _debug("[카메라] 영상 또는 OpenCV 입력이 종료됐습니다.")
                break
            camera_misses = 0
            history.add(packet.color, packet.timestamp_ms, packet.depth,
                        packet.camera_info.get("depth_scale"))
            if ui is not None:
                ui.update_frame(packet.color)
                ui.update_assembly(_assembly_context(engine))
            # Speaker follows the wake state: a dormant assistant is silent,
            # full stop. (Also breaks the speaker->internal-mic feedback loop
            # that had the model chattering "잠깐만요" to itself.)
            if listener is not None and hasattr(listener, "set_speaker_open"):
                listener.set_speaker_open(
                    ui is None or ui.mode != "DORMANT")
            # Robot skill feedback: narrate progress and mirror it into the
            # UI. Feedback items are change-only, so every one is worth a
            # line; results close the story.
            while True:
                try:
                    robot_event = robot_events.get_nowait()
                except queue.Empty:
                    break
                if ui is not None:
                    ui.set_robot(robot_manager.snapshot())
                narration = getattr(robot_event, "message", "")
                outcome = getattr(robot_event, "outcome", None)
                if outcome is not None:
                    # A queued second job ("...놓고 저거 가져와") starts here,
                    # once the first has actually finished - the manager runs
                    # one physical skill at a time by design.
                    queued = getattr(engine, "queued_robot_request", None)
                    if queued is not None:
                        engine.queued_robot_request = None
                        if outcome == "SUCCEEDED":
                            accepted, why = robot_manager.submit(queued)
                            follow = (engine.queued_robot_reply if accepted
                                      else why)
                        else:
                            follow = ("앞 작업이 끝나지 않아 다음 작업은 "
                                      "건너뜁니다.")
                        engine.queued_robot_reply = ""
                        print(f"코파일럿[로봇]> {follow}", flush=True)
                        if ui is not None:
                            ui.add_turn("assistant", follow)
                        if speech:
                            speech.say(follow)
                    # An inspection that came back CORRECT has already done
                    # the work of "is this step done?" - so offer the next
                    # step here instead of making the operator announce a
                    # completion the robot just photographed.
                    evidence = getattr(robot_event, "evidence", None) or {}
                    if outcome == "SUCCEEDED" and evidence.get("verdict"):
                        try:
                            offer = engine.after_inspection(
                                evidence["verdict"], narration)
                        except Exception as failure:
                            offer = ""
                            _debug(f"[검사 후속] 제안 생성 실패: {failure}",
                                   error=True)
                        if offer:
                            narration = f"{narration} {offer}"
                    line = f"코파일럿[로봇]> {narration}"
                    print(line, flush=True)
                    if ui is not None:
                        ui.add_turn("assistant", narration)
                        ui.set_mode("AWAKE",
                                    "로봇 작업 완료" if outcome == "SUCCEEDED"
                                    else f"로봇 작업 {outcome}")
                    if speech:
                        speech.say(narration)
                elif narration:
                    print(f"코파일럿[로봇]> {narration}", flush=True)
                    if ui is not None and getattr(robot_event, "state", "") in {
                            "SEARCHING_TARGET", "MOVING_SERVOJ",
                            "TRACKING_HAND", "WAITING_FOR_HANDOVER",
                            "RELEASING", "SAFETY_STOPPED", "FAILED"}:
                        ui.add_turn("assistant", narration)
            engine.tracker.update(AssemblyObservation(packet.timestamp_ms, visibility="VISIBLE"))
            for completed in [future for future in pending_futures
                              if future.done() and future is not background_future]:
                number, lane, utterance, call_id = pending_futures.pop(completed)
                try:
                    reply: CopilotReply = completed.result()
                except Exception as exc:
                    reply = _request_error_reply(exc)
                ready_replies[number] = (reply, lane, utterance, call_id)
            # Work can execute concurrently, but spoken/displayed answers keep
            # request order so a slow visual result cannot appear to answer the
            # following question.
            while ordered_pending:
                number = min(ordered_pending)
                if number not in ready_replies:
                    break
                reply, lane, utterance, call_id = ready_replies.pop(number)
                ordered_pending.remove(number)
                message = reply.text
                if reply.reference_image is not None:
                    reference_image = reply.reference_image
                    reference_label = reply.reference_label
                    # The web UI is the only screen most runs have; without
                    # this the copilot announces a reference photo that only
                    # the legacy OpenCV window could ever draw.
                    if ui is not None:
                        ui.set_reference(reference_image, reference_label or "")
                if reply.action == "DISPLAY_ON" and args.display != "off":
                    display_visible = True
                    if reply.reference_image is None:
                        reference_image = None
                        reference_label = None
                        if ui is not None:
                            ui.set_reference(None)
                elif reply.action == "DISPLAY_OFF":
                    display_visible = False
                    reference_window.close()
                    if window_created:
                        cv2.destroyWindow("DUM-E Unified Copilot")
                        window_created = False
                should_stop = reply.stop
                if should_stop:
                    exit_reason = "사용자 음성 종료 요청"
                if should_stop and background_future is not None:
                    message += " 진행 중인 PDF 저장 작업을 안전하게 마친 뒤 종료합니다."
                print(f"코파일럿[{_request_label(number, utterance)}]> {message}",
                      flush=True)
                if ui is not None:
                    ui.add_turn("assistant", message)
                    ui.update_assembly(_assembly_context(engine))
                    ui.set_mode("AWAKE", "응답 준비 완료")
                _submit_realtime_result(listener, call_id, reply, engine)
                if speech:
                    speech.say(message)
            if background_future is not None and background_future.done():
                metadata = pending_futures.pop(background_future, None)
                try:
                    reply = background_future.result()
                except Exception as exc:
                    reply = CopilotReply(f"백그라운드 작업에 실패했습니다: {exc}", "ERROR")
                number = metadata[0] if metadata else "?"
                utterance = metadata[2] if metadata else "백그라운드 작업"
                print(f"코파일럿[{_request_label(number, utterance)}]> {reply.text}",
                      flush=True)
                if ui is not None:
                    ui.add_turn("assistant", reply.text)
                    ui.update_assembly(_assembly_context(engine))
                    ui.set_mode("AWAKE", "백그라운드 작업 완료")
                call_id = metadata[3] if metadata else None
                _submit_realtime_result(listener, call_id, reply, engine)
                message = reply.text
                if speech:
                    speech.say(reply.text)
                background_future = None
            # Drain several utterances every frame. A slow Vision request no
            # longer prevents dialogue or immediate controls from being accepted.
            for _ in range(4):
                try:
                    incoming = questions.get_nowait()
                except queue.Empty:
                    break
                from_ui = isinstance(incoming, UIQueuedTurn)
                if from_ui:
                    incoming = incoming.text
                if isinstance(incoming, RealtimeDirectReply):
                    # The Realtime model answers small talk by voice on its
                    # own. While DORMANT the wake check still has to run HERE:
                    # "자비스" often arrives as a direct reply, not a tool
                    # call, and dropping it unexamined means the wake word can
                    # never wake anything.
                    if ui is not None and ui.mode == "DORMANT":
                        matched, remainder = ui.wake_match(incoming.utterance)
                        ui.set_heard(incoming.utterance or "(음성 감지)",
                                     gated=not matched)
                        if not matched:
                            _debug(f"[wake gate] 대기 중 직접응답 무시: {incoming.utterance}")
                            continue
                        ui.gate_voice(incoming.utterance)  # awakens
                        if remainder:
                            request_number += 1
                            number = request_number
                            print(f"사용자[요청 {number}]> {remainder}", flush=True)
                            print(f"코파일럿[{_request_label(number, remainder)}]> "
                                  f"{incoming.reply}", flush=True)
                            ui.add_turn("user", remainder)
                            ui.add_turn("assistant", incoming.reply)
                        continue
                    # Spoken "이제 쉬어"/"대기 모드": the Realtime model answers
                    # these itself ("네, 쉬세요") and would stay awake forever.
                    # The UI text box already sleeps in /api/input; voice must
                    # get the same treatment here.
                    if ui is not None and ui.sleep_match(incoming.utterance):
                        ui.set_heard(incoming.utterance, gated=False)
                        ui.add_turn("user", incoming.utterance)
                        ui.sleep()
                        if listener is not None and hasattr(listener, "set_speaker_open"):
                            listener.set_speaker_open(False)
                        ui.add_turn("assistant",
                                    "대기 모드로 들어갑니다. '자비스'라고 부르면 깨어납니다.")
                        print("코파일럿> 대기 모드 전환", flush=True)
                        continue
                    request_number += 1
                    number = request_number
                    print(f"사용자[요청 {number}]> {incoming.utterance}", flush=True)
                    print(f"코파일럿[{_request_label(number, incoming.utterance)}]> "
                          f"{incoming.reply}", flush=True)
                    message = incoming.reply
                    if ui is not None:
                        ui.set_heard(incoming.utterance, gated=False)
                        ui.add_turn("user", incoming.utterance)
                        ui.add_turn("assistant", incoming.reply)
                    continue
                call_id = None
                if isinstance(incoming, RealtimeToolRequest):
                    call_id = incoming.call_id
                    if incoming.decision == "ERROR":
                        message = incoming.clarification_question or "Realtime 음성 오류입니다."
                        print(f"코파일럿> {message}", flush=True)
                        continue
                    text = incoming.utterance.strip()
                    # Asleep means asleep: while DORMANT the ONLY thing a
                    # voice turn can do is wake us. No clarifications, no
                    # "couldn't hear you" replies (each spoken result is a
                    # ghost voice from an apparently sleeping assistant).
                    if ui is not None and ui.mode == "DORMANT":
                        matched, remainder = ui.wake_match(text)
                        ui.set_heard(
                            text or "(소리는 들렸지만 말을 알아듣지 못함)",
                            gated=not matched)
                        if not matched:
                            _debug(f"[wake gate] 대기 중 음성 무시: {text!r}")
                            continue
                        if not remainder:
                            ui.gate_voice(text)  # awakens + greeting turn
                            greeting = CopilotReply(
                                "네, JARVIS 온라인입니다. 무엇을 도와드릴까요?",
                                "SYSTEM")
                            _submit_realtime_result(listener, call_id,
                                                    greeting, engine)
                            ui.set_mode("AWAKE", "온라인 상태 · 말씀하세요")
                            continue
                        # Wake word plus a request: awaken here and let the
                        # normal pipeline handle the remainder below.
                    if not text:
                        # Audio arrived but carried no words. Say so in the UI
                        # too - a console-only message looks like a dead mic.
                        reply = CopilotReply("말씀을 정확히 듣지 못했습니다. 다시 말씀해 주세요.",
                                             "CONVERSATION")
                        print(f"코파일럿> {reply.text}", flush=True)
                        if ui is not None:
                            ui.set_heard("(소리는 들렸지만 말을 알아듣지 못함)", gated=True)
                        _submit_realtime_result(listener, call_id, reply, engine)
                        continue
                    if incoming.decision == "CLARIFY":
                        request_number += 1
                        number = request_number
                        print(f"사용자[요청 {number}]> {text}", flush=True)
                        reply = CopilotReply(
                            incoming.clarification_question
                            or "현재 작업에 대해 한 번만 더 구체적으로 말씀해 주세요.",
                            "CONVERSATION")
                        print(f"코파일럿[{_request_label(number, text)}]> {reply.text}",
                              flush=True)
                        _submit_realtime_result(listener, call_id, reply, engine)
                        continue
                else:
                    text = str(incoming)
                display_text = text
                if ui is not None and not from_ui:
                    accepted, gated_text = ui.gate_voice(display_text)
                    ui.set_heard(display_text, gated=not accepted)
                    if not accepted:
                        _debug(f"[wake gate] 호출어 없는 음성 무시: {display_text}")
                        continue
                    if not gated_text:
                        greeting = CopilotReply(
                            "안녕하세요. JARVIS 온라인입니다. 무엇을 도와드릴까요?",
                            "SYSTEM")
                        _submit_realtime_result(listener, call_id, greeting, engine)
                        ui.set_mode("AWAKE", "온라인 상태 · 말씀하세요")
                        continue
                    text = display_text = gated_text
                # Voice "이제 쉬어"/"대기 모드" arriving as a tool call: sleep
                # instead of letting the engine chat past it. The tool call
                # still gets a result so the Realtime session stays healthy
                # (its spoken ack is muted by the DORMANT speaker gate).
                if ui is not None and not from_ui and ui.sleep_match(display_text):
                    ui.add_turn("user", display_text)
                    ui.sleep()
                    if listener is not None and hasattr(listener, "set_speaker_open"):
                        listener.set_speaker_open(False)
                    ui.add_turn("assistant",
                                "대기 모드로 들어갑니다. '자비스'라고 부르면 깨어납니다.")
                    print("코파일럿> 대기 모드 전환", flush=True)
                    if call_id is not None:
                        _submit_realtime_result(
                            listener, call_id,
                            CopilotReply("대기 모드로 들어갑니다.", "SYSTEM"), engine)
                    continue
                if ui is not None and not from_ui:
                    ui.add_turn("user", display_text)
                    ui.set_mode("THINKING", "요청을 이해하고 있어요")
                cancel_prior_perception = bool(re.search(
                    r"(?:요청|검사|분석|처리).*?(?:취소|그만|중단|멈춰)",
                    display_text, re.I))
                if cancel_prior_perception:
                    for old_future, old_meta in list(pending_futures.items()):
                        if old_meta[1] != "PERCEPTION" or old_future.done():
                            continue
                        old_number, _, old_text, old_call_id = old_meta
                        old_future.cancel()
                        pending_futures.pop(old_future, None)
                        ordered_pending.discard(old_number)
                        ready_replies.pop(old_number, None)
                        cancelled = CopilotReply(
                            f"요청 {old_number}의 시각 분석 응답을 취소했습니다.",
                            "ASSEMBLY")
                        print(f"코파일럿[{_request_label(old_number, old_text)}]> "
                              f"{cancelled.text}", flush=True)
                        _submit_realtime_result(
                            listener, old_call_id, cancelled, engine)
                raw_intent = turn_manager.classify(display_text)
                offline_visual = (
                    raw_intent.domain == "VISION"
                    or (raw_intent.domain == "ASSEMBLY"
                        and raw_intent.intent == "QUESTION")
                )
                if not api_available and (raw_intent.domain == "CONVERSATION"
                                          or offline_visual):
                    request_number += 1
                    number = request_number
                    if offline_visual:
                        offline_text = ("현재 카메라 화면은 UI에서 확인할 수 있지만 AI 시각 "
                                        "판정은 API 키를 설정한 뒤 사용할 수 있습니다.")
                    else:
                        offline_text = ("현재는 오프라인 텍스트 테스트 모드입니다. 작업 시작, "
                                        "단계 설명, 참고 이미지 같은 로컬 조립 기능을 시험해 "
                                        "주세요. 자유 대화는 API 키를 설정하면 활성화됩니다.")
                    print(f"코파일럿[{_request_label(number, display_text)}]> "
                          f"{offline_text}", flush=True)
                    if ui is not None:
                        ui.add_turn("assistant", offline_text)
                        ui.set_mode("AWAKE", "오프라인 테스트 모드")
                    continue
                # Deterministic system controls and explicit progress updates
                # must keep the user's original wording. Realtime speech-act
                # normalization can otherwise erase "N단계까지 완료" or
                # "확인 없이 업데이트" and route it as a generic step query.
                preserve_raw = (
                    raw_intent.domain == "SYSTEM"
                    # Realtime provides audio turn-taking and a semantic hint,
                    # but the central deterministic router owns assembly
                    # semantics. Never let a Realtime speech-act rewrite an
                    # inspection question into a completion command.
                    or raw_intent.domain == "ASSEMBLY"
                    # Physical requests and stop words keep the user's exact
                    # wording all the way to the engine.
                    or raw_intent.domain == "ROBOT_SKILL"
                )
                speech_act = (incoming.speech_act
                              if isinstance(incoming, RealtimeToolRequest)
                              else "PASS_THROUGH")
                target_order = (incoming.target_step_order
                                if isinstance(incoming, RealtimeToolRequest) else None)
                if preserve_raw:
                    text = display_text
                elif speech_act == "START_WORK":
                    text = (f"{target_order}단계 작업을 시작해 줘."
                            if target_order is not None else "조립 작업을 시작해 줘.")
                elif speech_act == "EXPLAIN_TARGET_STEP" and target_order is not None:
                    text = f"{target_order}단계 작업을 설명해 줘."
                elif speech_act == "SHOW_TARGET_REFERENCE" and target_order is not None:
                    text = f"{target_order}단계 참고 사진을 보여 줘."
                elif speech_act == "SELECT_TARGET_STEP" and target_order is not None:
                    text = f"{target_order}단계로 작업 위치를 변경해 줘."
                elif speech_act == "IDENTIFY_CURRENT_STEP":
                    text = "카메라를 보고 현재 몇 단계인지 확인해 줘."
                elif speech_act == "CHECK_PROGRESS":
                    text = f"{text} 현재 작업이 잘 되었는지 확인해 줘."
                elif (speech_act == "COMPLETE_CURRENT_STEP"
                      or (isinstance(incoming, RealtimeToolRequest)
                          and incoming.intent_hint == "COMPLETE_CURRENT_STEP")):
                    # Normalize diverse natural completion statements at the
                    # boundary; the engine remains the only state mutator.
                    text = "현재 단계 작업을 완료했습니다."
                elif (isinstance(incoming, RealtimeToolRequest)
                      and incoming.intent_hint == "CHECK_ASSEMBLY"
                      and not any(word in text for word in ("확인", "검사", "잘", "맞"))):
                    text = f"{text} 잘 되었는지 확인해 주세요."
                if speech is not None and not speech.accept_transcript(text):
                    continue
                request_number += 1
                number = request_number
                print(f"사용자[요청 {number}]> {display_text}", flush=True)
                current = packet.color.copy()
                frames = history.representative(3)
                if (isinstance(incoming, RealtimeToolRequest)
                        and incoming.tool_name == "search_current_information"):
                    future = dialogue_executor.submit(
                        turn_manager.handle, text, frames=frames,
                        current_frame=current, timestamp_ms=packet.timestamp_ms)
                    pending_futures[future] = (number, "GENERAL_SEARCH", display_text, call_id)
                    ordered_pending.add(number)
                    continue
                intent = raw_intent if preserve_raw else turn_manager.classify(text)
                pending_operation = engine.db.pending()
                long_running = (
                    intent.domain == "MANUAL"
                    or (intent.domain == "CONFIRMATION" and intent.intent == "ACCEPT"
                        and pending_operation is not None
                        and pending_operation["operation_type"] == "REGENERATE_PDFS")
                )
                if intent.domain in ("SYSTEM", "ROBOT_SKILL"):
                    # ROBOT_SKILL joins the synchronous control lane: a stop
                    # word must never queue behind a slow LLM/Vision call.
                    reply = turn_manager.handle(
                        text, frames=frames, current_frame=current,
                        timestamp_ms=packet.timestamp_ms)
                    future = concurrent.futures.Future()
                    future.set_result(reply)
                    pending_futures[future] = (number, "CONTROL", display_text, call_id)
                    ordered_pending.add(number)
                elif long_running:
                    if background_future is not None:
                        message = ("이미 PDF 매뉴얼 작업이 진행 중입니다. 완료될 때까지 "
                                   "대화와 조립 질문은 계속할 수 있습니다.")
                        print(f"코파일럿[{_request_label(number, display_text)}]> "
                              f"{message}", flush=True)
                        busy = CopilotReply(message, "MANUAL")
                        _submit_realtime_result(listener, call_id, busy, engine)
                        if speech:
                            speech.say(message)
                    else:
                        background_future = background_executor.submit(
                            turn_manager.handle, text, frames=frames, current_frame=current,
                            timestamp_ms=packet.timestamp_ms)
                        pending_futures[background_future] = (
                            number, "BACKGROUND", display_text, call_id)
                        message = ("PDF 매뉴얼 작업을 백그라운드에서 시작했습니다. "
                                   "작업 중에도 계속 말씀하셔도 됩니다.")
                        print(f"코파일럿[{_request_label(number, display_text)}]> "
                              f"{message}", flush=True)
                else:
                    perception = (intent.domain == "VISION" or
                                  (intent.domain == "ASSEMBLY"
                                   and intent.intent == "QUESTION"))
                    lane = "PERCEPTION" if perception else "DIALOGUE"
                    executor = perception_executor if perception else dialogue_executor
                    future = executor.submit(
                        turn_manager.handle, text, frames=frames, current_frame=current,
                        timestamp_ms=packet.timestamp_ms)
                    pending_futures[future] = (number, lane, display_text, call_id)
                    ordered_pending.add(number)
                    message = f"요청 {number}을 {lane.lower()} 작업으로 접수했습니다."
                    _debug(f"[요청 {number}/{lane}] {message}")
            key = 255
            # With the web UI on, the only extra window worth opening is the
            # reference photograph - the live feed already lives in the page.
            if args.ui and args.display != "off":
                if display_visible and reference_image is not None:
                    reference_window.show(reference_image, reference_label or "")
                elif reference_window.open:
                    reference_window.close()
                if reference_window.open:
                    # Highgui only repaints while its event loop is pumped.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        reference_window.close()
            # The web work scene replaces the legacy OpenCV window when UI is
            # enabled. Keep the old window available for headless/diagnostic runs.
            if display_visible and not args.ui:
                state = engine.tracker.snapshot()
                shown = draw_overlay(
                    packet.color, engine.tracker.step, state, message,
                    reference_image=reference_image,
                    reference_label=reference_label,
                    controls_text=("[A] terminal ask  [Q/Esc] exit | "
                                   "voice: hide screen / end task"))
                cv2.imshow("DUM-E Unified Copilot", shown)
                window_created = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    exit_reason = "AR 창 종료 키"
                    break
                if key == ord("a"):
                    text = input("질문> ").strip()
                    if text:
                        questions.put(text)
    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
        _debug("통합 코파일럿 종료를 요청받았습니다.")
    finally:
        _debug(f"[코파일럿 종료 처리] 이유: {exit_reason}")
        # Copilot shutdown must never leave the robot moving: cancel fires
        # the skill's emergency stop (the physical app's SPACE) first.
        try:
            if robot_manager.cancel_active():
                print("코파일럿[로봇]> 종료 전에 로봇을 정지시켰습니다.", flush=True)
        except Exception:
            pass
        for future in pending_futures:
            future.cancel()
        if background_future is not None:
            background_future.cancel()
        dialogue_executor.shutdown(wait=True, cancel_futures=True)
        perception_executor.shutdown(wait=True, cancel_futures=True)
        # A running PDF build must finish its atomic rename before shared state closes.
        background_executor.shutdown(wait=True, cancel_futures=True)
        if listener:
            listener.close()
        if speech:
            speech.close()
        engine.close()
        source.close()
        if ui is not None:
            ui.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
