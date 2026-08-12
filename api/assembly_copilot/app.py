#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assembly_copilot.ar_overlay import draw_overlay
from assembly_copilot.assessment_validator import AssessmentValidator
from assembly_copilot.copilot import AssemblyCopilot
from assembly_copilot.frame_buffer import RecentFrameBuffer
from assembly_copilot.manual import load_manual
from assembly_copilot.manual_retriever import ManualRetriever
from assembly_copilot.models import AssemblyObservation
from assembly_copilot.multimodal_assessor import MultimodalAssessor
from assembly_copilot.question_router import AssemblyQuestionRouter
from assembly_copilot.reference_player import ReferenceVideoPlayer
from assembly_copilot.realsense_source import OpenCVSource, RealSenseSource
from assembly_copilot.state_tracker import AssemblyStateTracker
from assembly_copilot.storage import SessionStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL = ROOT / "assembly_manuals" / "template" / "assembly.yaml"
DEFAULT_SESSIONS = ROOT / "assembly_data" / "sessions"


def arguments():
    parser = argparse.ArgumentParser(description="로봇 동작 없는 RealSense 작업 코파일럿")
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument(
        "--camera", choices=("auto", "realsense", "opencv"), default="auto",
        help="카메라 종류(기본: RealSense 시도 후 OpenCV로 자동 전환)",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", type=Path, help="카메라 대신 개발용 영상 파일")
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--question", help="카메라 없이 매뉴얼/질문 흐름 한 번 확인")
    parser.add_argument("--no-record", action="store_true", help="세션 기록을 저장하지 않음")
    parser.add_argument("--voice", action="store_true", help="마이크로 작업 질문을 계속 수신")
    parser.add_argument("--no-wake", action="store_true", help="--voice에서 웨이크워드 없이 녹음")
    parser.add_argument("--device-index", type=int, help="마이크 장치 번호")
    parser.add_argument("--sample-rate", type=int, help="마이크 샘플레이트")
    parser.add_argument("--speak", action="store_true", help="답변을 AI 음성으로 재생")
    parser.add_argument("--no-ai", action="store_true", help="Vision 비교 없이 로컬 안내만")
    parser.add_argument("--vision-model", help="ASSEMBLY_VISION_MODEL 대신 사용할 모델")
    parser.add_argument("--history-seconds", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    manual = load_manual(args.manual)
    tracker = AssemblyStateTracker(manual)
    copilot = AssemblyCopilot(manual)
    router = AssemblyQuestionRouter()
    if args.question:
        state = tracker.snapshot()
        print(json.dumps(copilot.answer(args.question, state).__dict__,
                         ensure_ascii=False, indent=2))
        return 0

    source, camera_name = _open_source(args)
    history = RecentFrameBuffer(seconds=args.history_seconds)
    retriever = ManualRetriever(manual)
    assessor = MultimodalAssessor(model=args.vision_model)
    validator = AssessmentValidator(manual)
    video_player = ReferenceVideoPlayer()
    question_queue = queue.Queue()
    result_queue = queue.Queue()
    worker_busy = False
    voice_listener = speech = None
    if args.voice:
        from assembly_copilot.voice_io import VoiceQuestionListener
        voice_listener = VoiceQuestionListener(
            question_queue.put, no_wake=args.no_wake,
            device_index=args.device_index, sample_rate=args.sample_rate)
        voice_listener.start()
    if args.speak:
        from assembly_copilot.voice_io import CopilotSpeech
        speech = CopilotSpeech(True)
    store = None if args.no_record else SessionStore(
        args.sessions_dir, tracker.session_id,
        {"session_id": tracker.session_id, "manual_id": manual.manual_id,
         "manual_version": manual.version, "started_at_ms": round(time.time() * 1000),
         "camera": camera_name})
    message = ""
    reference = None
    recording = False
    print("키: A 질문, C 현재 단계 수동 완료, P 사진, V 영상 녹화, B 이전 단계, Q 종료")
    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            history.add(packet.color, packet.timestamp_ms)
            if store and recording:
                store.write_video(packet.color)
            observation = AssemblyObservation(
                timestamp_ms=packet.timestamp_ms, visibility="VISIBLE")
            state = tracker.update(observation)
            step = tracker.step
            try:
                question, answer_data, answer_reference = result_queue.get_nowait()
                worker_busy = False
                message = str(answer_data["instruction"])
                advanced_from = None
                if answer_data.get("advance_requested"):
                    advanced_from, progress_message = _apply_advance(tracker, answer_data)
                    message = progress_message or message
                reference = answer_reference
                if isinstance(reference, dict):
                    try:
                        video_player.play(reference)
                        reference = None
                    except RuntimeError as exc:
                        message = str(exc)
                print(f"코파일럿> {message}")
                if speech:
                    speech.say(message)
                if store:
                    evidence = store.save_keyframe(packet.color, f"question_{packet.timestamp_ms}")
                    store.event("QUESTION_ANSWER", {"question": question,
                                "answer": answer_data, "evidence": evidence})
                    if advanced_from:
                        store.event("STEP_AUTO_ADVANCE", {
                            "step_id": advanced_from,
                            "reason": answer_data.get("advance_reason"),
                            "state": tracker.snapshot().to_dict(),
                            "evidence": evidence})
            except queue.Empty:
                pass
            if not worker_busy:
                try:
                    queued_question = question_queue.get_nowait()
                except queue.Empty:
                    queued_question = None
                if queued_question:
                    routed = router.route(queued_question)
                    print(f"[질문 분류] {routed.intent}: {queued_question}", flush=True)
                    worker_busy = True
                    message = "현재 장면과 매뉴얼을 비교하고 있습니다."
                    threading.Thread(
                        target=_answer_worker,
                        args=(queued_question, router, copilot, retriever, assessor,
                              validator, tracker.snapshot(), observation,
                              history.representative(6), step, not args.no_ai,
                              result_queue),
                        daemon=True, name="assembly-assessment").start()
            video_reference = video_player.frame()
            shown = draw_overlay(packet.color, step, state, message,
                                 video_reference if video_reference is not None else reference)
            cv2.imshow("DUM-E Assembly Copilot", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                question = input("질문> ").strip()
                if question:
                    question_queue.put(question)
            elif key == ord("c"):
                before = tracker.step.id if tracker.step else None
                state = tracker.confirm_current_step()
                message = "단계 완료를 기록했습니다."
                if store:
                    evidence = store.save_keyframe(packet.color, f"complete_{before}")
                    store.event("STEP_CONFIRM", {"step_id": before, "evidence": evidence,
                                                 "state": state.to_dict()})
            elif key == ord("b"):
                tracker.previous_step(); message = "이전 단계로 돌아갔습니다."
            elif key == ord("p") and store:
                path = store.save_keyframe(packet.color, f"manual_{packet.timestamp_ms}")
                message = f"이미지 저장: {Path(path).name}"
            elif key == ord("v") and store:
                recording = not recording
                if recording:
                    store.start_video(packet.color); message = "영상 녹화를 시작했습니다."
                else:
                    store.close(); message = "영상 녹화를 종료했습니다."
    finally:
        source.close()
        video_player.close()
        if voice_listener:
            voice_listener.close()
        if speech:
            speech.close()
        if store:
            store.close()
        cv2.destroyAllWindows()
    return 0


def _reference_for(step):
    if step and step.references.images:
        return step.references.images[0]
    if step and step.references.videos:
        return step.references.videos[0]
    return None


def _apply_advance(tracker, answer):
    """Apply only a fresh completion decision for the still-current step."""
    current_id = tracker.step.id if tracker.step else None
    source_id = answer.get("source_step_id")
    if current_id != source_id:
        return None, "이전 단계에 대한 판정이므로 현재 단계에는 적용하지 않았습니다."
    before_index = tracker.index
    tracker.confirm_current_step()
    if tracker.index == before_index:
        return None, "다른 완료 조건이 남아 있어 단계는 유지합니다."
    next_step = tracker.step
    return source_id, (f"{source_id} 완료를 확인했습니다. "
                       + (f"다음은 {next_step.title}입니다."
                          if next_step else "모든 단계를 완료했습니다."))


def _answer_worker(question_text, router, copilot, retriever, assessor, validator,
                   state, observation, frames, step, use_ai, output):
    question = router.route(question_text)
    reference = _reference_for(step) if question.intent == "SHOW_REFERENCE" else None
    try:
        if question.intent == "COMPLETE_STEP":
            answer = {"assessment": "USER_CONFIRMED_COMPLETE",
                      "observed_step_id": state.current_step_id,
                      "claimed_step_id": question.claimed_step_id,
                      "confidence": 1.0,
                      "instruction": "사용자의 완료 확인을 받았습니다.",
                      "observed_facts": [], "issues": [], "accepted": True,
                      "validation_notes": ["사용자 명시적 완료 확인"],
                      "advance_requested": True,
                      "advance_reason": "USER_CONFIRMATION",
                      "source_step_id": state.current_step_id}
        elif question.intent in {"IDENTIFY_STEP", "CHECK_PROGRESS",
                                 "CHECK_CLAIMED_STEP"} and use_ai:
            if not assessor.available:
                raise RuntimeError("OPENAI_API_KEY가 없어 Vision 비교를 실행할 수 없습니다")
            raw = assessor.assess(question, state, observation,
                                  retriever.retrieve(question, state), frames)
            answer = validator.validate(raw, state, observation).model_dump(mode="json")
            answer["advance_requested"] = bool(
                answer["accepted"] and answer["assessment"] == "STEP_COMPLETE"
                and answer["observed_step_id"] == state.current_step_id)
            answer["advance_reason"] = "VISION_STEP_COMPLETE"
            answer["source_step_id"] = state.current_step_id
        else:
            local = copilot.answer(question_text, state)
            answer = {"assessment": local.assessment,
                      "observed_step_id": state.current_step_id,
                      "claimed_step_id": question.claimed_step_id,
                      "confidence": local.confidence, "instruction": local.text,
                      "observed_facts": [], "issues": [], "accepted": True,
                      "validation_notes": ["로컬 매뉴얼 안내"],
                      "advance_requested": False,
                      "advance_reason": None,
                      "source_step_id": state.current_step_id}
    except Exception as exc:
        answer = {"assessment": "UNCERTAIN", "observed_step_id": state.current_step_id,
                  "claimed_step_id": question.claimed_step_id, "confidence": 0.0,
                  "instruction": f"AI 작업 상태 확인을 완료하지 못했습니다: {exc}",
                  "observed_facts": [], "issues": [], "accepted": False,
                  "validation_notes": [f"{type(exc).__name__}: {exc}"],
                  "advance_requested": False, "advance_reason": None,
                  "source_step_id": state.current_step_id}
    output.put((question_text, answer, reference))


def _open_source(args):
    if args.video:
        return OpenCVSource(str(args.video)), "video"
    if args.camera == "opencv":
        return OpenCVSource(args.camera_index), "opencv"
    if args.camera == "realsense":
        return RealSenseSource(), "realsense"

    try:
        return RealSenseSource(), "realsense"
    except RuntimeError as exc:
        print(
            f"[카메라] RealSense를 사용할 수 없어 OpenCV 카메라 "
            f"{args.camera_index}번으로 전환합니다: {exc}",
            file=sys.stderr,
        )
        try:
            return OpenCVSource(args.camera_index), "opencv"
        except RuntimeError as opencv_exc:
            raise RuntimeError(
                "사용 가능한 카메라를 찾지 못했습니다. RealSense를 사용하려면 "
                "pyrealsense2를 설치하고, 일반 카메라라면 --camera-index 값을 "
                "확인하세요. 영상 파일은 --video /경로/영상.mp4로 실행할 수 "
                f"있습니다. (RealSense: {exc}; OpenCV: {opencv_exc})"
            ) from None


if __name__ == "__main__":
    raise SystemExit(main())
