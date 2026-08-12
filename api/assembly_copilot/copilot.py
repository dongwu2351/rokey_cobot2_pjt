from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AssemblyManual, AssemblyState


@dataclass(frozen=True)
class CopilotAnswer:
    intent: str
    assessment: str
    text: str
    current_step_id: str | None
    confidence: float


class AssemblyCopilot:
    """Grounded local response baseline; a multimodal adapter can replace compose()."""

    CHECK = re.compile(r"(잘|맞|정상|제대로|확인|검사|문제)")
    NEXT = re.compile(r"(다음|뭘\s*해야|무엇을\s*해야|어떻게\s*해야)")
    REFERENCE = re.compile(r"(사진|영상|예시|정상\s*모양|보여)")

    def __init__(self, manual: AssemblyManual) -> None:
        self.manual = manual

    def answer(self, question: str, state: AssemblyState) -> CopilotAnswer:
        step = None if state.current_step_id is None else self.manual.steps[state.current_step_index]
        if step is None:
            return CopilotAnswer("STATUS", "COMPLETE", "매뉴얼의 모든 단계를 완료했습니다.",
                                 None, 1.0)
        if self.REFERENCE.search(question):
            if step.references.images:
                text = f"{step.title} 참고 이미지를 AR 화면에 표시했습니다."
            elif step.references.videos:
                text = f"{step.title} 참고 영상이 등록되어 있습니다."
            else:
                text = "현재 단계에 참고 이미지나 영상이 아직 등록되지 않았습니다."
            return CopilotAnswer("SHOW_REFERENCE", "REFERENCE", text, step.id, state.confidence)
        if self.CHECK.search(question):
            if state.warnings:
                detail = " ".join(str(w.get("message", "오류 조건 감지")) for w in state.warnings)
                text, assessment = f"수정이 필요합니다. {detail}", "NEEDS_CORRECTION"
            elif state.unsatisfied_conditions:
                missing = ", ".join(state.unsatisfied_conditions)
                text = f"아직 완료로 확인할 수 없습니다. 확인되지 않은 조건: {missing}."
                assessment = "NOT_VERIFIED"
            else:
                text, assessment = "현재 단계의 등록된 완료 조건을 만족합니다.", "OK"
            return CopilotAnswer("CHECK_PROGRESS", assessment, text, step.id, state.confidence)
        if self.NEXT.search(question):
            return CopilotAnswer("NEXT_STEP", "INSTRUCTION",
                                 f"현재는 {step.order}단계, {step.title}입니다. {step.instruction}",
                                 step.id, state.confidence)
        return CopilotAnswer("EXPLAIN", "INSTRUCTION",
                             f"{step.title}: {step.instruction}", step.id, state.confidence)
