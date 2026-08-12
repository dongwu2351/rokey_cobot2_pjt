"""Open-ended conversation and capability planning.

This module deliberately does not execute robot actions.  It turns a free-form
utterance into either a perception command, a spoken reply, or a capability
plan that a separate safety-gated executor can handle.
"""

from __future__ import annotations

import json
import os
import re
import base64
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .conversation_memory import ConversationMemory


ENV_PATH = Path(__file__).with_name(".env")


class ConversationRoute(str, Enum):
    ROBOT_COMMAND = "ROBOT_COMMAND"
    CHAT = "CHAT"
    TOOL_QUESTION = "TOOL_QUESTION"
    ADVICE = "ADVICE"
    DESCRIBE = "DESCRIBE"
    CAPABILITY_PLAN = "CAPABILITY_PLAN"
    CLARIFY = "CLARIFY"
    STOP = "STOP"


class CapabilityStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str = Field(min_length=1, max_length=80)
    # Explicit scalar slots keep the schema compatible with strict structured
    # outputs. Capability-specific adapters can validate/transform them later.
    target: str | None = Field(default=None, max_length=256)
    reference: str | None = Field(default=None, max_length=256)
    duration_s: float | None = Field(default=None, ge=0, le=86_400)
    value: str | None = Field(default=None, max_length=256)
    risk: str = Field(default="low", pattern="^(low|medium|high)$")


class ConversationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: ConversationRoute
    reply: str | None = Field(default=None, max_length=2000)
    command_text: str | None = Field(default=None, max_length=1000)
    clarification_question: str | None = Field(default=None, max_length=500)
    capability: str | None = Field(default=None, max_length=80)
    steps: list[CapabilityStep] = Field(default_factory=list, max_length=12)
    robot_action_allowed: bool = False
    confidence: float = Field(default=1.0, ge=0, le=1)


CONVERSATION_INSTRUCTIONS = """당신은 사람을 돕는 협동로봇의 대화·작업 플래너다.
사용자의 자유로운 한국어 발화를 의도에 맞는 route로 분류하고, 필요한 경우
사용 가능한 capability를 조합한다. 자연스러운 대화나 설명은 reply에 한국어로
작성한다. 실제 로봇 동작은 안전 검증기가 별도로 승인하므로, route가 ROBOT_COMMAND
또는 CAPABILITY_PLAN일 때만 robot_action_allowed를 true로 한다.

사용자가 정해진 명령 문구를 말해야 한다고 가정하지 않는다. 최근 대화, 현재 제품,
현재 조립 단계와 직전 질문을 이용해 생략·대명사·정정·후속 요청의 의미를 파악한다.
예를 들어 '그럼 다음은?', '이게 맞아?', '사진도 보여줘', '아니 그 단계 말고'처럼
불완전한 문장도 문맥으로 해석한다. 확정할 근거가 부족할 때만 짧게 되묻는다.
먼저 최근 대화와 현재 상태로 생략된 목적어를 복원한다. 가능한 의도가 하나로
수렴하면 사용자의 표현을 고쳐 말하게 하지 말고 해당 capability를 선택한다. 서로
다른 행동 두 가지가 비슷하게 가능할 때만 CLARIFY와 한 문장의 선택 질문을 반환한다.
confidence는 의도 해석의 확신도로 보정하며, 근거가 약하면 0.6 미만으로 둔다.

사용 가능한 capability:
- ground_object: 카메라에서 물체 찾기
- resolve_reference: '여기', '이것', '내가 들고 있는 것' 같은 지시어 해소
- illuminate_region: 특정 영역에 조명 비추기
- move_robot: 로봇을 위치로 이동
- grasp_object: 물체 파지
- place_object: 물체 놓기
- speak: 음성 응답
- ask_clarification: 사용자에게 되묻기
- assembly_start_or_resume: 현재 매뉴얼의 조립 작업을 시작하거나 저장 단계부터 재개
- assembly_identify_step: 실시간 카메라와 YAML로 현재 조립 단계 판단
- assembly_check_progress: 현재 작업의 정상 여부와 오류 확인
- assembly_next_instruction: 현재 상태에 맞는 다음 작업 안내
- assembly_show_reference: 현재 단계 참고 자료 표시
- manual_generate_from_pdf: PDF에서 제품 조립 매뉴얼 생성
- web_search: 최신 날씨·주소·일반 웹 정보 검색

'이제 해볼까', '그거 계속하자', '전에 하던 것부터 해보자', '준비됐어'처럼
시작이라는 단어가 없어도 최근 대화와 active_manual/current_step을 근거로 조립을
시작·재개하려는 뜻이면 assembly_start_or_resume를 선택한다. 단순한 일상 대화인지
조립 재개인지 근거가 부족한 경우에만 한 번 짧게 확인한다.

등록되지 않은 하드웨어 기능이 필요하면 임의로 성공을 주장하지 말고
CAPABILITY_PLAN과 해당 capability를 반환한다. 위치·좌표·물체 ID를 추측하지 않는다.
애매하면 CLARIFY로 짧게 되묻는다."""

CONVERSATION_RESPONSE_INSTRUCTIONS = """당신은 사람을 돕는 협동로봇의 대화 상대다.
최근 대화와 현재 발화를 참고해 자연스럽고 짧은 한국어로 답한다.
반드시 일반 문장만 출력하고 JSON, route 이름, 내부 상태를 출력하지 않는다.
모르는 사실은 추측하지 말고 필요한 정보를 질문한다. 실제 로봇 동작을 수행했다고
주장하지 않는다."""


class ConversationRouter:
    """Route free-form speech without making any physical-world changes."""

    _STOP = re.compile(r"^(?:로봇\s*)?(?:멈춰|정지|중지|스톱|stop)(?:\s*해|\s*줘)?[.!?]?$", re.I)
    _ACTION = re.compile(
        r"(?:가져|갖다|건네|집어|잡아|놓아|옮겨|이동|비춰|비추|켜|꺼|열어|닫아|정리|조립)"
    )
    _KNOWN_COMMAND = re.compile(
        r"(?:가져|갖다|건네|집어|놓아|옮겨|이동|잡아|줘|주세요|줄래|다오|부탁)"
    )
    _QUESTION = re.compile(r"(?:뭐|무엇|어디|왜|어떻게|용도|사용법|설명|알려|맞아|가능)")
    _DESCRIBE = re.compile(r"(?:손에\s*들고|들고\s*있는|이\s*물체|이것|뭐야|무엇이야)")
    _ADVICE = re.compile(r"(?:조언|도와|어떻게 하면|방법|추천|좋을까|어떨까)")

    def __init__(
        self,
        *,
        model: str | None = None,
        instructions: str = CONVERSATION_INSTRUCTIONS,
        client: Any | None = None,
        timeout_seconds: float | None = None,
        vision_model: str | None = None,
        memory_path: str | Path | None = None,
    ) -> None:
        load_dotenv(ENV_PATH)
        self.model = model or os.getenv("CONVERSATION_MODEL", os.getenv("COMMAND_MODEL", "gpt-5.6-terra"))
        self.instructions = instructions
        self.timeout_seconds = timeout_seconds or float(os.getenv("CONVERSATION_TIMEOUT_SECONDS", "8"))
        self.vision_model = vision_model or os.getenv("VISION_MODEL", self.model)
        self._client = client
        self._memory = ConversationMemory(memory_path) if memory_path else None
        self._history: deque[dict[str, str]] = deque(
            self._memory.recent(12) if self._memory else (), maxlen=12
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(api_key=key, max_retries=0)
        return self._client

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        return tuple(self._history)

    def remember(self, role: str, content: str, route: str | None = None) -> None:
        """Record turns handled by external capability lanes (search/vision/etc.)."""
        self._remember(role, content, route)

    def route(
        self,
        utterance: str,
        *,
        capabilities: Sequence[str] = (),
        context: Mapping[str, Any] | None = None,
    ) -> ConversationDecision:
        text = utterance.strip()
        if not text:
            return ConversationDecision(
                route=ConversationRoute.CLARIFY,
                clarification_question="어떤 작업이나 이야기를 도와드릴까요?",
                robot_action_allowed=False,
            )
        self._remember("user", text)
        if self._STOP.fullmatch(text):
            return ConversationDecision(route=ConversationRoute.STOP)
        # Fast path: explicit physical-action language remains on the grounding path.
        if self._KNOWN_COMMAND.search(text) and not self._QUESTION.search(text):
            return ConversationDecision(
                route=ConversationRoute.ROBOT_COMMAND,
                command_text=text,
                robot_action_allowed=True,
            )
        if self._ADVICE.search(text):
            return ConversationDecision(route=ConversationRoute.ADVICE)
        if self._DESCRIBE.search(text) and self._QUESTION.search(text):
            return ConversationDecision(route=ConversationRoute.DESCRIBE)
        # A standalone conversation CLI can use the cheap question route. The
        # unified copilot supplies capabilities, so ambiguous questions need
        # semantic planning against current context instead of keyword lock-in.
        if self._QUESTION.search(text) and not capabilities:
            return ConversationDecision(route=ConversationRoute.TOOL_QUESTION)
        return self._llm_route(text, capabilities=capabilities, context=context)

    def respond(
        self,
        utterance: str,
        *,
        route: ConversationRoute,
        context: Mapping[str, Any] | None = None,
        image: Any | None = None,
    ) -> str:
        """Generate a spoken response for non-motion conversation routes."""
        prompt = {
            "route": route.value,
            "utterance": utterance,
            "context": dict(context or {}),
            "conversation_history": list(self._history),
            "instruction": "짧고 자연스러운 한국어 음성 응답을 작성하라. 로봇이 실제로 행동했다고 주장하지 말라.",
        }
        request_input: Any = json.dumps(prompt, ensure_ascii=False)
        if image is not None and route == ConversationRoute.DESCRIBE:
            encoded = self._encode_image(image)
            if encoded is not None:
                request_input = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "사진을 보고 질문에 답하세요. 손에 들고 있는 물체를 "
                                    "식별하고, 확실하지 않으면 추측하지 말고 그렇게 말하세요.\n"
                                    + json.dumps(prompt, ensure_ascii=False)
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{encoded}",
                            },
                        ],
                    }
                ]
        response = self.client.responses.create(
            model=self.vision_model if image is not None and route == ConversationRoute.DESCRIBE else self.model,
            instructions=CONVERSATION_RESPONSE_INSTRUCTIONS,
            input=request_input,
            max_output_tokens=300,
            store=False,
            timeout=self.timeout_seconds,
        )
        text = getattr(response, "output_text", None)
        if not text:
            # SDK versions may omit the convenience property while still
            # returning text content in output[].
            chunks: list[str] = []
            for item in getattr(response, "output", ()) or ():
                for content in getattr(item, "content", ()) or ():
                    value = getattr(content, "text", None)
                    if value:
                        chunks.append(str(value))
            text = "\n".join(chunks)
        if not text:
            if route == ConversationRoute.DESCRIBE:
                text = "물체를 확인하려면 카메라에 잘 보이도록 들어 보여주세요."
            else:
                text = "말씀하신 내용을 아직 확인하지 못했습니다. 조금 더 설명해 주시겠어요?"
        text = str(text).strip()
        if len(text) > 500:
            # Keep spoken replies responsive; offer continuation in the next turn.
            shortened = text[:500]
            boundary = max(shortened.rfind("."), shortened.rfind("요."), shortened.rfind("다."))
            text = shortened[: boundary + 1] if boundary > 180 else shortened + "…"
        self._remember("assistant", text)
        return text

    def _remember(self, role: str, content: str, route: str | None = None) -> None:
        self._history.append({"role": role, "content": content})
        if self._memory is not None:
            self._memory.add(role, content, route)

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()

    @staticmethod
    def _encode_image(image: Any) -> str | None:
        """Encode an OpenCV BGR frame as a compact JPEG for vision input."""
        try:
            import cv2
            import numpy as np

            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                return None
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
            )
            if not ok:
                return None
            return base64.b64encode(buffer.tobytes()).decode("ascii")
        except (ImportError, TypeError, ValueError):
            return None

    def _llm_route(
        self,
        text: str,
        *,
        capabilities: Sequence[str],
        context: Mapping[str, Any] | None,
    ) -> ConversationDecision:
        payload = {
            "utterance": text,
            "available_capabilities": list(capabilities),
            "context": dict(context or {}),
            "conversation_history": list(self._history),
        }
        response = self.client.responses.parse(
            model=self.model,
            instructions=self.instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=ConversationDecision,
            reasoning={"effort": "none"},
            max_output_tokens=700,
            store=False,
            timeout=self.timeout_seconds,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("conversation model did not return a decision")
        return parsed
