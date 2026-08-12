from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from .assistive_models import AssistiveResult, AssistiveState, TrackedObject
from .assistive_processor import AssistiveCommandProcessor, DEFAULT_MEMORY_PATH
from .command_models import Ambiguity, CommandResult, Decision
from .conversation_router import ConversationRouter
from .grounding import FallbackGrounder, GroundingDINOBackend, YOLOEBackend
from .object_memory import ObjectMemory
from .situated_parser import SituatedCommandParser
from .TTS import AI_VOICE_DISCLOSURE, SpeechSynthesizer, TTSConfig
from .voice_pipeline import build_pipeline
from .streaming_preview import StreamingPreview
from .speculative_intent import SpeculativeIntentEngine


class _TranscriptOnlyRouter:
    """Keep the audio pipeline but defer all semantics to the situated parser."""

    def parse_command(self, utterance, context):
        return CommandResult(
            decision=Decision.REJECT,
            actions=(),
            ambiguity=Ambiguity.VISION_GROUNDING_REQUIRED,
            clarification_question=None,
            route="SAFETY",
            raw_utterance=utterance,
            grounding_query=utterance,
        )


def _draw_result(frame, result: AssistiveResult):
    import cv2

    for item in result.detections:
        box = item.bbox
        selected = item.id == result.selected_object_id
        color = (0, 255, 0) if selected else (0, 180, 255)
        cv2.rectangle(
            frame,
            (round(box.x1), round(box.y1)),
            (round(box.x2), round(box.y2)),
            color,
            3 if selected else 2,
        )
        cv2.putText(
            frame,
            f"{item.id} {item.label} {item.score:.2f}",
            (round(box.x1), max(20, round(box.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return frame


def _object_at_point(
    result: AssistiveResult, x: int, y: int
) -> TrackedObject | None:
    """Return the smallest visible box containing a user click."""
    matches = [
        item
        for item in result.detections
        if item.bbox.x1 <= x <= item.bbox.x2
        and item.bbox.y1 <= y <= item.bbox.y2
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (item.bbox.x2 - item.bbox.x1)
        * (item.bbox.y2 - item.bbox.y1),
    )


class _FocusSelector:
    """Bridge OpenCV mouse clicks to the processor's deictic focus."""

    def __init__(self, processor: AssistiveCommandProcessor) -> None:
        self.processor = processor
        self.result: AssistiveResult | None = None
        self.selected_id: str | None = None

    def show(self, result: AssistiveResult) -> None:
        self.result = result
        self.selected_id = None

    def on_mouse(self, event, x, y, flags, userdata) -> None:
        import cv2

        if event != cv2.EVENT_LBUTTONDOWN or self.result is None:
            return
        selected = _object_at_point(self.result, x, y)
        if selected is None:
            print("BB 내부를 클릭해 주세요.", flush=True)
            return
        self.processor.set_focus(selected.id)
        self.selected_id = selected.id
        print(
            json.dumps(
                {
                    "event": "FOCUS_SELECTED",
                    "object_id": selected.id,
                    "label": selected.label,
                    "bbox": selected.bbox.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free-form voice to zero-shot webcam grounding prototype"
    )
    parser.add_argument("--text", help="skip microphone and process this command")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="keep the camera and models warm while listening for voice commands",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="run camera/UI continuously while audio and grounding use a worker",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--camera-indices",
        help="comma-separated camera indices for realtime multi-camera mode (e.g. 4,6)",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--no-wake", action="store_true")
    parser.add_argument("--start-timeout-ms", type=int, default=10_000)
    parser.add_argument("--end-silence-ms", type=int, default=450)
    parser.add_argument("--max-record-ms", type=int, default=12_000)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument(
        "--grounding-backend",
        choices=("dino", "yoloe", "cascade"),
        default="dino",
        help="fast YOLOE, precise Grounding DINO, or YOLOE with DINO fallback",
    )
    parser.add_argument("--yoloe-model", default="yoloe-26s-seg.pt")
    parser.add_argument("--yoloe-confidence", type=float, default=0.15)
    parser.add_argument("--yoloe-image-size", type=int, default=512)
    parser.add_argument("--yoloe-device")
    parser.add_argument("--cascade-min-score", type=float, default=0.30)
    parser.add_argument("--memory-db", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument(
        "--conversation-db",
        type=Path,
        default=Path(__file__).with_name("data") / "conversation_memory.sqlite3",
    )
    parser.add_argument("--speak", action="store_true")
    parser.add_argument(
        "--stream-tts",
        action="store_true",
        help="start audio playback as TTS PCM chunks arrive",
    )
    parser.add_argument(
        "--post-tts-cooldown-ms",
        type=int,
        default=1200,
        help="delay microphone reopening after TTS to avoid acoustic feedback",
    )
    parser.add_argument("--disclose-ai-voice", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--partial-preview",
        action="store_true",
        help="print best-effort partial STT while the user is speaking",
    )
    parser.add_argument("--save-frame", type=Path)
    parser.add_argument(
        "--debug-click-focus",
        action="store_true",
        help="optional developer fallback; normal operation uses voice clarification",
    )
    args = parser.parse_args()
    camera_indices: tuple[int, ...] | None = None
    if args.camera_indices:
        try:
            camera_indices = tuple(
                int(value.strip())
                for value in args.camera_indices.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error("--camera-indices must be comma-separated integers")
        if not camera_indices:
            parser.error("--camera-indices must contain at least one index")
    if args.realtime:
        args.continuous = True
        args.show = True
    if args.post_tts_cooldown_ms < 0:
        parser.error("--post-tts-cooldown-ms must not be negative")
    if args.text is not None and args.continuous:
        parser.error("--continuous is available for microphone input, not --text")

    try:
        import cv2
    except ImportError as exc:
        parser.error(f"OpenCV is required; install requirements-vision.txt ({exc})")

    primary_camera_index = camera_indices[0] if camera_indices else args.camera_index
    capture = cv2.VideoCapture(primary_camera_index)
    if not capture.isOpened():
        parser.error(f"cannot open webcam index {primary_camera_index}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    dino = GroundingDINOBackend(model_id=args.model_id)
    yoloe = YOLOEBackend(
        model_path=args.yoloe_model,
        confidence=args.yoloe_confidence,
        image_size=args.yoloe_image_size,
        device=args.yoloe_device,
    )
    grounder = {
        "dino": dino,
        "yoloe": yoloe,
        "cascade": FallbackGrounder(
            yoloe, dino, minimum_primary_score=args.cascade_min_score
        ),
    }[args.grounding_backend]
    print(
        f"{args.yoloe_model} 모델을 준비하고 있습니다..."
        if args.grounding_backend in {"yoloe", "cascade"}
        else "Grounding DINO 모델을 준비하고 있습니다...",
        flush=True,
    )
    if not args.realtime:
        grounder.load()

    tts = (
        SpeechSynthesizer(config=TTSConfig(streaming=True))
        if args.speak and args.stream_tts
        else SpeechSynthesizer()
        if args.speak
        else None
    )
    if tts is not None and args.disclose_ai_voice and not args.realtime:
        tts.speak(AI_VOICE_DISCLOSURE)

    processor = AssistiveCommandProcessor(
        parser=SituatedCommandParser(),
        grounder=grounder,
        memory=ObjectMemory(args.memory_db),
        conversation_router=ConversationRouter(memory_path=args.conversation_db),
        speaker=tts.speak if tts is not None and not args.realtime else None,
    )
    if args.realtime:
        from .realtime_runtime import RealtimeAssistiveRuntime

        capture.release()
        pipeline_args = SimpleNamespace(
            no_wake=args.no_wake,
            continuous=True,
            device_index=args.device_index,
            sample_rate=args.sample_rate,
            vad_threshold=args.vad_threshold,
            start_timeout_ms=args.start_timeout_ms,
            max_record_ms=args.max_record_ms,
            end_silence_ms=args.end_silence_ms,
        )
        pipeline = build_pipeline(pipeline_args)
        pipeline.router = _TranscriptOnlyRouter()
        partial_preview = None
        if args.partial_preview:
            speculative = SpeculativeIntentEngine(processor.conversation_router)

            def on_partial(text: str, duration_ms: int) -> None:
                print(
                    json.dumps(
                        {
                            "event": "PARTIAL_TRANSCRIPT",
                            "text": text,
                            "duration_ms": duration_ms,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                speculative(text, duration_ms)

            partial_preview = StreamingPreview(
                pipeline.transcriber,
                sample_rate=pipeline.microphone.config.rate,
                on_text=on_partial,
            )
            pipeline.partial_callback = partial_preview
            pipeline.partial_interval_ms = 700
        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        runtime = RealtimeAssistiveRuntime(
            camera_index=args.camera_index,
            camera_indices=camera_indices,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            pipeline=pipeline,
            processor=processor,
            wait_for_wake=not args.no_wake,
            model_loader=grounder.load,
            speaker=tts.speak if tts is not None else None,
            disclosure_text=(
                AI_VOICE_DISCLOSURE
                if tts is not None and args.disclose_ai_voice
                else None
            ),
            post_tts_cooldown_ms=args.post_tts_cooldown_ms,
            save_frame=args.save_frame,
            debug_click_focus=args.debug_click_focus,
        )
        try:
            runtime.run()
        except KeyboardInterrupt:
            pass
        finally:
            capture.release()
            if partial_preview is not None:
                partial_preview.close()
            processor.memory.close()
            if processor.conversation_router is not None:
                processor.conversation_router.close()
        return

    focus_selector = _FocusSelector(processor) if args.show else None
    if focus_selector is not None:
        cv2.namedWindow("Assistive grounding")
        cv2.setMouseCallback("Assistive grounding", focus_selector.on_mouse)

    try:
        pipeline = None
        if args.text is None:
            pipeline_args = SimpleNamespace(
                no_wake=args.no_wake,
                continuous=False,
                device_index=args.device_index,
                sample_rate=args.sample_rate,
                vad_threshold=args.vad_threshold,
                start_timeout_ms=args.start_timeout_ms,
                max_record_ms=args.max_record_ms,
                end_silence_ms=args.end_silence_ms,
            )
            pipeline = build_pipeline(pipeline_args)
            pipeline.router = _TranscriptOnlyRouter()

        while True:
            command = args.text
            audio_total_ms = 0.0
            if pipeline is not None:
                audio_result = pipeline.run_once(wait_for_wake=not args.no_wake)
                command = audio_result.transcript
                audio_total_ms = audio_result.timings.total_ms
                if not command:
                    print(
                        json.dumps(
                            audio_result.to_dict(), ensure_ascii=False, indent=2
                        ),
                        flush=True,
                    )
                    if args.continuous:
                        continue
                    return

            frame_started = time.perf_counter()
            ok, frame = capture.read()
            frame_ms = (time.perf_counter() - frame_started) * 1_000
            if not ok or frame is None:
                raise RuntimeError("webcam frame capture failed")
            result = processor.process(command, frame)
            output = result.to_dict()
            output["utterance"] = command
            output["timings_ms"]["audio_to_transcript"] = round(audio_total_ms, 3)
            output["timings_ms"]["frame_capture"] = round(frame_ms, 3)
            output["timings_ms"]["voice_to_box"] = round(
                audio_total_ms + frame_ms + result.timings_ms.get("total", 0.0), 3
            )
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)

            annotated = _draw_result(frame, result)
            if args.save_frame:
                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_frame), annotated)
            if args.show:
                focus_selector.show(result)
                cv2.imshow("Assistive grounding", annotated)
                needs_reference_click = bool(
                    args.continuous
                    and result.state == AssistiveState.CLARIFICATION_REQUIRED
                    and result.detections
                )
                if needs_reference_click:
                    print(
                        "비교 기준으로 삼을 BB를 클릭하세요. 종료하려면 q를 누르세요.",
                        flush=True,
                    )
                    while focus_selector.selected_id is None:
                        if cv2.waitKey(20) & 0xFF == ord("q"):
                            return
                    if tts is not None:
                        tts.speak(
                            "기준 물체로 선택했습니다. 이어서 원하는 물체를 말씀해 주세요."
                        )
                else:
                    delay_ms = 1 if args.continuous else 0
                    if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                        break
            if (
                args.continuous
                and tts is not None
                and result.clarification_question
                and args.post_tts_cooldown_ms
            ):
                time.sleep(args.post_tts_cooldown_ms / 1_000)
            if not args.continuous:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
        processor.memory.close()
        if processor.conversation_router is not None:
            processor.conversation_router.close()


if __name__ == "__main__":
    main()
