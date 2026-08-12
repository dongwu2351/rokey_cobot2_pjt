"""Judge a photographed assembly step: "am I doing this right?"

The picture comes from the wrist camera, aimed at whatever the operator's
finger was pointing at. This module turns it into a verdict a person can act
on - and, deliberately, into an "I cannot tell" when the picture does not
support one. A confident wrong answer about someone's assembly is worse than
no answer: they will follow it.

Three backends behind one interface, so the whole flow can be built and
exercised before a single token is spent:

    mock    canned verdicts, no network - for wiring, tests and demos
    openai  the real vision model
    auto    openai if a key is configured, otherwise mock

The verdict vocabulary matches the UI design brief: CORRECT,
NEEDS_CORRECTION, UNCERTAIN, WRONG_STEP, NOT_VISIBLE.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERDICTS = ("CORRECT", "NEEDS_CORRECTION", "UNCERTAIN", "WRONG_STEP",
            "NOT_VISIBLE")

#: What the model is asked to produce. Kept small on purpose: every field has
#: to earn its tokens, and a long schema invites the model to pad.
RESPONSE_SCHEMA = {
    "verdict": "one of CORRECT / NEEDS_CORRECTION / UNCERTAIN / WRONG_STEP / NOT_VISIBLE",
    "spoken": "one or two short sentences for the operator, in Korean",
    "observations": ["what is actually visible, short phrases"],
    "next_action": "what to do next, or empty if nothing is needed",
    "confidence": "0.0-1.0",
    "need_closer": "true only if a closer photograph would settle it",
}

SYSTEM_PROMPT = (
    "You inspect an assembly in progress from a single close-up photograph "
    "taken by a robot's wrist camera, aimed at the spot the operator pointed "
    "at with their finger.\n"
    "A RED CIRCLE marks that spot. Judge what is inside and immediately "
    "around it - the frame usually contains other parts and tools that are "
    "not what was asked about.\n"
    "The camera may be tilted rather than overhead, which is deliberate: it "
    "shows the work from the operator's own viewing angle. Do not mistake "
    "perspective for a crooked part.\n"
    "Judge ONLY what the photograph shows. If the relevant part is out of "
    "frame, blurred, or hidden, answer NOT_VISIBLE - do not infer it from the "
    "step description. If the picture is clear but you cannot decide, answer "
    "UNCERTAIN. A wrong confident answer makes the operator assemble it "
    "wrongly, so uncertainty is the safer and more useful reply.\n"
    "If the marked part is simply too small or too far to resolve, set "
    "need_closer true - the robot can go back and take a closer photograph, "
    "and one more look is cheaper than a guess. Do not set it when the view "
    "is close enough and the answer is genuinely ambiguous.\n"
    "Answer in Korean in the 'spoken' field, briefly, as a colleague looking "
    "over their shoulder would. Return JSON only."
)


@dataclass
class InspectionVerdict:
    verdict: str
    spoken: str
    observations: list[str] = field(default_factory=list)
    next_action: str = ""
    confidence: float = 0.0
    need_closer: bool = False
    model: str = ""
    image: str = ""
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "spoken": self.spoken,
            "observations": list(self.observations),
            "next_action": self.next_action, "confidence": self.confidence,
            "need_closer": self.need_closer,
            "model": self.model, "image": self.image,
            "elapsed_s": round(self.elapsed_s, 2), "error": self.error,
        }


def _clean(payload: dict, image: str, model: str, elapsed: float) -> InspectionVerdict:
    verdict = str(payload.get("verdict", "")).upper().strip()
    if verdict not in VERDICTS:
        # An unrecognised verdict is not a verdict. Downgrading to UNCERTAIN
        # keeps a model that ignored the vocabulary from asserting anything.
        verdict = "UNCERTAIN"
    observations = payload.get("observations") or []
    if isinstance(observations, str):
        observations = [observations]
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return InspectionVerdict(
        verdict=verdict,
        spoken=str(payload.get("spoken", "")).strip()
        or "사진으로는 확실히 판단하기 어렵습니다.",
        observations=[str(o) for o in observations][:6],
        next_action=str(payload.get("next_action", "")).strip(),
        confidence=max(0.0, min(1.0, confidence)),
        need_closer=bool(payload.get("need_closer", False)),
        model=model, image=image, elapsed_s=elapsed,
    )


class MockVision:
    """Deterministic verdicts, no network, no tokens.

    Cycles the vocabulary so a demo or a test sees every branch - including
    the awkward ones - instead of only the happy path."""

    name = "mock"

    def __init__(self, sequence=None):
        self._sequence = list(sequence or
                              ["CORRECT", "NEEDS_CORRECTION", "UNCERTAIN",
                               "NOT_VISIBLE"])
        self._index = 0

    def analyse(self, image_path, question, context=None,
                reference_image=None):
        verdict = self._sequence[self._index % len(self._sequence)]
        self._index += 1
        spoken = {
            "CORRECT": "지금 하시는 방향 맞습니다. 그대로 진행하세요.",
            "NEEDS_CORRECTION": "볼트가 비스듬히 들어간 것 같습니다. 풀고 수직으로 다시 넣어보세요.",
            "UNCERTAIN": "사진만으로는 확실하지 않습니다. 조금 더 가까이서 다시 볼까요?",
            "WRONG_STEP": "이건 다음 단계 부품 같습니다. 순서를 한 번 확인해 주세요.",
            "NOT_VISIBLE": "가리키신 부분이 사진에 잘 안 보입니다. 손을 치우고 다시 시도해 주세요.",
        }[verdict]
        return _clean({"verdict": verdict, "spoken": spoken,
                       "observations": ["(mock) 실제 모델 호출 없음"],
                       "next_action": "", "confidence": 0.5,
                       # Exercise the re-shoot path in demos and tests.
                       "need_closer": verdict in ("NOT_VISIBLE", "UNCERTAIN")},
                      str(image_path), "mock", 0.0)


class VisionResponseTruncated(RuntimeError):
    """The model ran out of budget before it closed its JSON.

    Measured on real inspection photographs, a verdict costs ~275-300
    completion tokens of which ~135-170 is reasoning. At a 400 ceiling a
    slightly busier picture simply runs out mid-object, and the parse error
    that follows used to surface as the generic "지금은 판단을 도와드릴 수
    없습니다" - indistinguishable from a dead network.
    """


class OpenAIVision:
    """The real call. One image, one short schema, one response."""

    name = "openai"

    #: Roughly four times the measured cost of a verdict. Reasoning length is
    #: not something the caller can predict from the photograph, so the
    #: headroom has to absorb the bad cases rather than the average one.
    DEFAULT_MAX_TOKENS = int(os.environ.get("ASSEMBLY_VISION_MAX_TOKENS", "1200"))
    #: A second attempt doubles the ceiling, but never past this - a model
    #: that cannot answer in 2400 tokens is not going to answer in 10000,
    #: and the operator is standing there waiting.
    MAX_TOKEN_CEILING = 2400

    def __init__(self, model=None, api_key=None, max_tokens=None,
                 detail="auto"):
        self.model = model or os.environ.get("ASSEMBLY_VISION_MODEL",
                                             "gpt-5.6-sol")
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip().strip('"')
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.detail = detail

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    #: One retry, then give up. The failures that actually happen here are
    #: transient (rate limit while the Realtime session is also billing, a
    #: gateway blip, a slow first token) and cost the operator a whole trip of
    #: the arm - the picture is already taken, so retrying is nearly free.
    RETRY_DELAY_SEC = 1.5

    @staticmethod
    def _explain(error: Exception) -> tuple[str, bool]:
        """(what to say, is it worth retrying)

        A missing key never fixes itself, and telling someone to "try again
        in a moment" when it will fail identically forever wastes their take.
        """
        if isinstance(error, VisionResponseTruncated):
            return ("사진 판정이 길어져 답변이 잘렸습니다. 다시 시도합니다.", True)
        text = str(error).lower()
        if "401" in text or "api key" in text or "invalid_api_key" in text:
            return ("비전 모델 인증에 실패했습니다. API 키 설정을 확인해 주세요.",
                    False)
        if "429" in text or "quota" in text or "rate limit" in text:
            return ("비전 모델 사용량 한도에 걸렸습니다. 잠시 후 다시 시도해 주세요.",
                    True)
        if ("timeout" in text or "timed out" in text or "connection" in text
                or "502" in text or "503" in text or "504" in text):
            return ("비전 모델 연결이 불안정합니다. 잠시 후 다시 시도해 주세요.",
                    True)
        return ("지금은 판단을 도와드릴 수 없습니다. 잠시 후 다시 시도해 주세요.",
                True)

    def analyse(self, image_path, question, context=None,
                reference_image=None):
        started = time.monotonic()
        path = Path(image_path)
        if not path.is_file():
            return InspectionVerdict(
                "NOT_VISIBLE", "사진 파일을 찾지 못했습니다.", model=self.model,
                image=str(path), error=f"missing image: {path}")
        try:
            from openai import OpenAI
        except Exception as error:
            return InspectionVerdict(
                "UNCERTAIN", "비전 모델을 사용할 수 없습니다.", model=self.model,
                image=str(path), error=f"openai sdk unavailable: {error}")

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        lines = [f"작업자 질문: {question.strip() or '지금 이거 잘하고 있나요?'}"]
        for key, value in (context or {}).items():
            if value:
                lines.append(f"{key}: {value}")
        reference_encoded = None
        if reference_image:
            reference_path = Path(reference_image)
            if reference_path.is_file():
                reference_encoded = base64.b64encode(
                    reference_path.read_bytes()).decode("ascii")
                lines.append(
                    "두 번째 이미지는 조립서에 실린 이 단계의 참고 사진입니다. "
                    "첫 번째(로봇이 방금 찍은 실제 작업 사진)와 비교해 판단하세요. "
                    "촬영 각도와 배경은 다를 수 있으니 부품의 결합 관계만 비교하고, "
                    "구도 차이를 오조립으로 보지 마세요.")
        lines.append("Return JSON with exactly these keys: "
                     + ", ".join(RESPONSE_SCHEMA))
        client = OpenAI(api_key=self.api_key)
        payload = None
        last_error: Exception | None = None
        cap = self.max_tokens
        for attempt in (1, 2):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": "\n".join(lines)},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}",
                                "detail": self.detail}},
                        ] + ([{"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{reference_encoded}",
                            "detail": self.detail}}]
                            if reference_encoded else [])},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=cap,
                )
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                if not content or choice.finish_reason == "length":
                    raise VisionResponseTruncated(
                        f"{cap} 토큰 상한에서 응답이 잘렸습니다 "
                        f"(finish_reason={choice.finish_reason}, "
                        f"본문 {len(content)}자)")
                payload = json.loads(content)
                break
            except Exception as error:
                last_error = error
                spoken, retryable = self._explain(error)
                # Say it where a person can read it. A verdict that only
                # carries the polite sentence leaves "why did it fail?"
                # unanswerable after the fact - which is exactly what
                # happened the first time this fired during a recording.
                print(f"[검사 비전] 호출 실패 ({attempt}/2): {error}",
                      file=sys.stderr, flush=True)
                if not retryable or attempt == 2:
                    return InspectionVerdict(
                        "UNCERTAIN", spoken, model=self.model, image=str(path),
                        elapsed_s=time.monotonic() - started, error=str(error))
                if isinstance(error, VisionResponseTruncated):
                    # Retrying at the same ceiling reproduces the same cut.
                    cap = min(cap * 2, self.MAX_TOKEN_CEILING)
                time.sleep(self.RETRY_DELAY_SEC)
        if payload is None:                      # unreachable, kept honest
            return InspectionVerdict(
                "UNCERTAIN", self._explain(last_error or Exception(""))[0],
                model=self.model, image=str(path),
                elapsed_s=time.monotonic() - started, error=str(last_error))
        return _clean(payload, str(path), self.model,
                      time.monotonic() - started)


def build_analyser(mode="auto", **kwargs):
    """`mode` is mock / openai / auto (openai when a key exists)."""
    mode = (mode or "auto").lower()
    if mode == "mock":
        return MockVision()
    real = OpenAIVision(**kwargs)
    if mode == "openai":
        return real
    return real if real.configured else MockVision()
