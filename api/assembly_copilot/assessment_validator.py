from __future__ import annotations

from .assessment_models import AssemblyAssessment, ValidatedAssessment
from .models import AssemblyManual, AssemblyObservation, AssemblyState


class AssessmentValidator:
    def __init__(self, manual: AssemblyManual, minimum_confidence: float = 0.40) -> None:
        self.manual = manual
        self.step_ids = {step.id for step in manual.steps}
        self.minimum_confidence = minimum_confidence

    def validate(self, result: AssemblyAssessment, state: AssemblyState,
                 observation: AssemblyObservation | None) -> ValidatedAssessment:
        notes: list[str] = []
        accepted = True
        assessment = result.assessment
        instruction = result.instruction
        step = next((item for item in self.manual.steps
                     if item.id == result.observed_step_id), None)
        application_confidence = result.confidence
        if result.observed_step_id and result.observed_step_id not in self.step_ids:
            notes.append("AI가 매뉴얼에 없는 단계를 반환함")
            accepted = False
        if result.claimed_step_id and result.claimed_step_id not in self.step_ids:
            notes.append("사용자가 주장한 단계가 현재 매뉴얼에 없음")
        obscured = observation is None or observation.visibility in {"UNKNOWN", "OCCLUDED"}
        if obscured or not result.visible or result.needs_better_view:
            assessment = "NOT_VISIBLE"
            instruction = "작업 상태를 확인할 수 있도록 손과 공구를 잠시 치우고 작업물 전체를 보여 주세요."
            notes.append("시야 불충분으로 확정 판정을 차단함")
            accepted = False
        elif (not step or not step.visual_checks) and result.confidence < self.minimum_confidence:
            assessment = "UNCERTAIN"
            instruction = ("가장 유사한 후보는 찾았지만 신뢰도 40% 기준에 미달해 "
                           "확정하기 어렵습니다. 작업물이 정면에 보이게 한 뒤 다시 "
                           "질문해 주세요.")
            notes.append("신뢰도 기준 미달")
            accepted = False
        if step is not None and result.assessment in {"CORRECT", "STEP_COMPLETE"}:
            checks = {item.check_id: item for item in result.checks}
            defined = {str(item.get("id")): item for item in step.visual_checks}
            unknown_ids = sorted(set(checks) - set(defined))
            if unknown_ids:
                notes.append("YAML에 없는 시각 체크를 무시함: " + ", ".join(unknown_ids))
            required = [item for item in step.visual_checks
                        if item.get("kind", "required") == "required"
                        and bool(item.get("required", True))]
            if required:
                verified_count = sum(
                    1 for item in required
                    if checks.get(str(item.get("id"))) is not None
                    and checks[str(item.get("id"))].result == "true"
                )
                # Do not trust the model's self-reported probability. For
                # structured manuals this value is deterministic evidence
                # coverage; temporal agreement is enforced by the engine.
                application_confidence = verified_count / len(required)
            missing_or_unverified = [
                str(item.get("id")) for item in required
                if checks.get(str(item.get("id"))) is None
                or checks[str(item.get("id"))].result != "true"
            ]
            error_hits = [
                str(item.get("id")) for item in step.visual_checks
                if item.get("kind") == "error"
                and checks.get(str(item.get("id"))) is not None
                and checks[str(item.get("id"))].result == "true"
            ]
            if error_hits:
                assessment = "NEEDS_CORRECTION"
                accepted = False
                instruction = ("오류 조건이 관측되어 단계를 변경하지 않습니다: "
                               + ", ".join(error_hits))
                notes.append("결정론적 오류 체크 감지")
            elif required and missing_or_unverified:
                assessment = "UNCERTAIN"
                accepted = False
                instruction = ("필수 시각 조건을 모두 확인하지 못해 단계를 변경하지 "
                               "않습니다: " + ", ".join(missing_or_unverified))
                notes.append("필수 시각 체크 미충족")
        if result.assessment == "STEP_COMPLETE":
            notes.append("검증된 AI 완료 판정: 앱이 현재 단계 일치 여부를 다시 확인함")
        return ValidatedAssessment(
            assessment=assessment, observed_step_id=result.observed_step_id,
            claimed_step_id=result.claimed_step_id, confidence=application_confidence,
            instruction=instruction, observed_facts=result.observed_facts,
            checks=result.checks, issues=result.issues, accepted=accepted,
            validation_notes=notes)
