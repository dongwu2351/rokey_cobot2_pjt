from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AssessmentIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=80)
    detail: str = Field(min_length=1, max_length=500)
    evidence: str | None = Field(default=None, max_length=300)


class VisualCheckResult(BaseModel):
    """A model observation keyed to a deterministic manual check."""

    model_config = ConfigDict(extra="forbid")
    check_id: str = Field(min_length=1, max_length=80)
    result: Literal["true", "false", "unknown"]
    evidence: str = Field(min_length=1, max_length=500)


class AssemblyAssessment(BaseModel):
    """Untrusted multimodal output; AssessmentValidator checks it before display."""

    model_config = ConfigDict(extra="forbid")
    assessment: Literal[
        "CORRECT", "NEEDS_CORRECTION", "WRONG_STEP", "STEP_COMPLETE",
        "NOT_VISIBLE", "UNCERTAIN",
    ]
    observed_step_id: str | None = Field(default=None, max_length=80)
    claimed_step_id: str | None = Field(default=None, max_length=80)
    confidence: float = Field(ge=0, le=1)
    visible: bool
    observed_facts: list[str] = Field(default_factory=list, max_length=12)
    checks: list[VisualCheckResult] = Field(default_factory=list, max_length=40)
    manual_requirements: list[str] = Field(default_factory=list, max_length=12)
    issues: list[AssessmentIssue] = Field(default_factory=list, max_length=8)
    instruction: str = Field(min_length=1, max_length=900)
    needs_better_view: bool


class ValidatedAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment: str
    observed_step_id: str | None
    claimed_step_id: str | None
    confidence: float
    instruction: str
    observed_facts: list[str]
    checks: list[VisualCheckResult] = Field(default_factory=list)
    issues: list[AssessmentIssue]
    accepted: bool
    validation_notes: list[str]
