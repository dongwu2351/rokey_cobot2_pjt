#!/usr/bin/env python3
"""음성 출력 전담. **부르는 쪽을 절대 막지 않는다.**

    speech.say("이동합니다.")     -> 즉시 반환. 재생은 전용 스레드가 한다.
    speech.prepare(문장)          -> 미리 합성만 해 둔다 (재생 안 함)
    speech.cancel("사용자 발화")   -> 재생 중이면 즉시 끊는다

★ 왜 분리했나 — 이게 없을 때 무슨 일이 있었나
  TTS 왕복+재생이 4.9초인데 그걸 **ROS 콜백 스레드에서** 직접 기다렸다.
  그동안 /dum_e/target 과 /curobo/goal 이 하나도 갱신되지 않았고, 승인 직전
  검증이 **낡은 값과 비교**할 수 있었다. 음성이 제어 신뢰성을 갉아먹은 것이다.
  ROS 콜백은 큐에 넣고 즉시 돌아가야 한다.

★ 캐시 — 여기가 체감의 대부분이다
  로봇이 하는 말은 대부분 반복된다("취소했습니다", "이동합니다", ...).
  같은 문장을 매번 4.9초 주고 사는 것은 낭비다. 문장 해시로 WAV 를 디스크에
  남겨 두면 두 번째부터는 **재생 시간만** 든다.

★ 선합성(prepare) — 반복이 아닌 문장도 빠르게
  "초록색 테이프를 찾았습니다" 는 물체마다 다르다. 하지만 **명령을 알아들은
  순간 이미 이름을 안다.** 인지가 물체를 찾는 동안(보통 1초 이내) 미리 합성해
  두면, 정작 물어볼 때는 캐시에서 즉시 나온다.

★ 세대(generation) 번호
  취소된 작업의 음성이 뒤늦게 재생되면 안 된다. 작업마다 번호를 붙이고,
  번호가 지난 요청은 재생 직전에 버린다.
"""
from __future__ import annotations

import hashlib
import difflib
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

CACHE_DIR = Path(__file__).with_name("cache_tts")

# 우선순위 — 작을수록 먼저. 안전 관련 문장이 잡담보다 늦게 나오면 안 된다.
PRIO_SAFETY = 0     # 승인 거부, 정지
PRIO_CONFIRM = 1    # 되묻기
PRIO_STATUS = 2     # 진행 상황
PRIO_CHAT = 3
# ★ 선합성과 워밍업의 우선순위를 나눈다.
#   같은 값이면 시작 시 워밍업 6문장(각 2초)이 큐를 먼저 차지해서, 정작 지금
#   필요한 질문의 선합성이 12초 뒤로 밀린다. 실제로 그래서 첫 명령에서
#   선합성 효과가 통째로 사라졌다 (캐시 적중 0).
PRIO_PREPARE = 5    # 곧 쓸 문장을 미리
PRIO_WARMUP = 9     # 언젠가 쓸 문장을 미리 (제일 뒤)

# 자주 쓰는 고정 문구. 시작할 때 백그라운드로 미리 만들어 둔다.
WARMUP_PHRASES = (
    "이동합니다.",
    "취소했습니다.",
    "네 또는 아니오로 답해 주세요.",
    "목표가 아직 물체로 맞춰지지 않았습니다. 다시 말씀해 주세요.",
    "위치를 확인하지 못했습니다. 다시 말씀해 주세요.",
    "정지합니다.",
)


class SpeechManager:
    def __init__(self, *, enabled=True, cache_dir=CACHE_DIR, warmup=True,
                 on_playback_start=None, on_playback_end=None):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.tts = None
        self.generation = 0
        self.stats = {"cache_hit": 0, "cache_miss": 0, "cancelled": 0}
        self._mem: dict[str, Path] = {}
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._last_started_at = None
        self._current_text = ""
        self._last_text = ""
        self._last_ended_at = 0.0
        self._on_playback_start = on_playback_start
        self._on_playback_end = on_playback_end
        if not enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from VoiceProcessing.TTS import SpeechSynthesizer
        self.tts = SpeechSynthesizer()
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="speech-worker")
        self._worker.start()
        if warmup:
            threading.Thread(target=self._warmup, daemon=True,
                             name="speech-warmup").start()

    # ── 바깥에서 부르는 것 ────────────────────────────────────────────
    def say(self, text, *, priority=PRIO_STATUS, generation=None):
        """즉시 반환한다. 재생은 워커가 한다."""
        text = (text or "").strip()
        if not text:
            return
        print(f"  [말] {text}", flush=True)
        if not self.enabled:
            return
        with self._lock:
            self._seq += 1
            gen = self.generation if generation is None else generation
            self._q.put((priority, self._seq, ("play", text, gen, time.monotonic())))

    def prepare(self, text, *, priority=PRIO_PREPARE):
        """재생하지 않고 합성만 미리 해 둔다. 실패해도 조용히 넘어간다."""
        text = (text or "").strip()
        if not text or not self.enabled or self._path(text).exists():
            return
        with self._lock:
            self._seq += 1
            # 선합성은 실제 발화보다 뒤, 워밍업보다 앞.
            self._q.put((priority, self._seq, ("prep", text, None, None)))

    def cancel(self, reason=""):
        """재생 중이면 끊는다. 큐에 쌓인 재생 요청도 버린다."""
        with self._lock:
            self.generation += 1
            proc = self._proc
        if proc is not None and proc.poll() is None:
            self.stats["cancelled"] += 1
            try:
                proc.terminate()
            except Exception:                       # noqa: BLE001
                pass
            if reason:
                print(f"  [말 끊음] {reason}", flush=True)

    def is_playing(self):
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def is_likely_echo(self, transcript):
        """Heuristically reject STT text originating from our own loudspeaker."""
        now = time.monotonic()
        with self._lock:
            reference = self._current_text
            if not reference and now - self._last_ended_at <= 3.0:
                reference = self._last_text
        heard = re.sub(r"[^0-9A-Za-z가-힣]", "", transcript or "").lower()
        spoken = re.sub(r"[^0-9A-Za-z가-힣]", "", reference or "").lower()
        if len(heard) < 2 or len(spoken) < 2:
            return False
        if heard in spoken or spoken in heard:
            return True
        return difflib.SequenceMatcher(None, heard, spoken).ratio() >= .58

    def close(self):
        self._stop.set()
        self.cancel()

    # ── 내부 ────────────────────────────────────────────────────────
    def _path(self, text):
        config = getattr(self.tts, "config", None)
        profile = (f"{getattr(config, 'model', '')}|{getattr(config, 'voice', '')}|"
                   f"{getattr(config, 'instructions', '')}|{getattr(config, 'speed', '')}")
        key = hashlib.sha1(f"{profile}|{text}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}.wav"

    def _warmup(self):
        for phrase in WARMUP_PHRASES:
            if self._stop.is_set():
                return
            self.prepare(phrase, priority=PRIO_WARMUP)

    def _synth(self, text):
        """문장 -> WAV 경로. 캐시에 있으면 네트워크를 타지 않는다."""
        path = self._path(text)
        if path.exists():
            self.stats["cache_hit"] += 1
            return path
        self.stats["cache_miss"] += 1
        data = self.tts.synthesize(text)
        # 반쯤 쓰인 파일이 캐시로 남으면 다음 재생이 깨진다. 임시로 쓰고 옮긴다.
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path

    def _play(self, path, text, gen, requested_at):
        with self._lock:
            if gen is not None and gen != self.generation:
                return                              # 취소된 작업의 음성이다
            proc = subprocess.Popen(
                ["aplay", "-q", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc = proc
            self._current_text = text
        if callable(self._on_playback_start):
            self._on_playback_start()
        if requested_at is not None:
            self._last_started_at = (time.monotonic() - requested_at) * 1000
        try:
            proc.wait()
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
                self._last_text = text
                self._last_ended_at = time.monotonic()
                self._current_text = ""
            if callable(self._on_playback_end):
                self._on_playback_end()

    def _run(self):
        while not self._stop.is_set():
            try:
                _, _, job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            kind, text, gen, requested_at = job
            try:
                if gen is not None and gen != self.generation:
                    continue                        # 재생 전에 이미 취소됐다
                path = self._synth(text)
                if kind == "play":
                    self._play(path, text, gen, requested_at)
            except Exception as exc:                # noqa: BLE001
                # 음성은 보조 수단이다. 실패해도 로봇 작업은 계속돼야 한다.
                print(f"  [TTS 실패 - 화면으로만] {type(exc).__name__}: {exc}",
                      flush=True)

    def last_latency_ms(self):
        return self._last_started_at
