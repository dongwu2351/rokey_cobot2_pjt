#!/usr/bin/env python3
"""Single safe entry point for the merged DUM-E project.

Only one microphone-owning runtime is started at a time.  Arguments following
the mode are forwarded unchanged to the selected module.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _verbose_logs() -> bool:
    return os.getenv("DUME_VERBOSE_LOGS", "false").lower() in {"1", "true", "yes", "on"}


def _command(mode: str, forwarded: list[str]) -> list[str]:
    python = sys.executable
    if mode == "conversation":
        return [python, "-m", "VoiceProcessing.assistive_cli", *forwarded]
    if mode == "assembly":
        return [python, "-m", "assembly_copilot.app", *forwarded]
    if mode == "manual-generator":
        return [python, "-m", "manual_generator.cli", *forwarded]
    if mode == "copilot":
        return [python, "-m", "unified_copilot.app", *forwarded]
    raise ValueError(mode)


def _run_tests(env: dict[str, str]) -> int:
    suites = (
        ("unified_copilot/tests",),
        ("assembly_copilot/tests",),
        ("manual_generator/tests",),
        ("llm/VoiceProcessing/tests",),
        ("robot_skills/tests",),
    )
    for (start_dir,) in suites:
        command = [sys.executable, "-m", "unittest", "discover", "-s", start_dir, "-v"]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DUM-E 통합 프로젝트 실행기")
    parser.add_argument(
        "mode", choices=("copilot", "conversation", "assembly", "manual-generator", "tests"),
        help="한 번에 하나의 런타임만 선택",
    )
    args, forwarded = parser.parse_known_args()
    env = os.environ.copy()
    voice_parent = str(ROOT / "llm")
    env["PYTHONPATH"] = voice_parent + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Native extensions (RealSense, OpenCV, PortAudio) can terminate the child
    # without a Python traceback. Make those failures diagnosable at the launcher.
    env.setdefault("PYTHONFAULTHANDLER", "1")
    if args.mode == "tests":
        return _run_tests(env)
    child = subprocess.Popen(_command(args.mode, forwarded), cwd=ROOT, env=env)
    try:
        returncode = child.wait()
    except KeyboardInterrupt:
        # The terminal sends SIGINT to both launcher and copilot. Let the child
        # finish its camera/audio cleanup without printing a launcher traceback.
        try:
            returncode = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate()
            returncode = child.wait(timeout=5)
    if returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = "UNKNOWN"
        if _verbose_logs():
            print(f"[통합 실행기] 하위 프로세스가 신호 {name}({number})로 종료됐습니다.",
                  file=sys.stderr, flush=True)
        else:
            print("코파일럿> 프로그램이 비정상 종료되었습니다.", flush=True)
    elif returncode > 0:
        if _verbose_logs():
            print(f"[통합 실행기] 하위 프로세스가 오류 코드 "
                  f"{returncode}로 종료됐습니다.", file=sys.stderr, flush=True)
        else:
            print("코파일럿> 프로그램 실행 중 오류가 발생했습니다.", flush=True)
    elif _verbose_logs():
        print("[통합 실행기] 코파일럿이 정상 종료됐습니다.", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
