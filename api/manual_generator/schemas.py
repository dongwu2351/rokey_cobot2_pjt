from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name_ko: str
    visual_queries: list[str] = Field(default_factory=list)
    description: str


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name_ko: str
    description: str


class VisualState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    new_elements: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    distinguish_prev: str | None = None
    distinguish_next: str | None = None
    camera_note: str
    uncertain: list[str] = Field(default_factory=list)


class StepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^step_[0-9]{2,3}$")
    order: int = Field(ge=1)
    title: str
    instruction: str
    required_parts: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    precondition_description: str | None = None
    completion_description: str
    error_conditions: list[str] = Field(default_factory=list)
    visual_state: VisualState
    source_pages: list[int] = Field(min_length=1)
    source_evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class GeneratedManual(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manual_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    product_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    product: str
    version: str
    description: str
    manufacturer: str | None = None
    source_document_title: str
    parts: list[PartSpec]
    tools: list[ToolSpec]
    steps: list[StepSpec] = Field(min_length=1)
    global_warnings: list[str] = Field(default_factory=list)
    review_required: list[str] = Field(default_factory=list)
