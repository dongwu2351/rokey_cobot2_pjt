from __future__ import annotations

from dataclasses import asdict

from .models import AssemblyManual, AssemblyState, AssemblyStep
from .question_router import AssemblyQuestion


class ManualRetriever:
    def __init__(self, manual: AssemblyManual) -> None:
        self.manual = manual
        self.by_id = {step.id: step for step in manual.steps}

    def retrieve(self, question: AssemblyQuestion, state: AssemblyState) -> list[AssemblyStep]:
        # Visual step identification compares the live scene against every YAML
        # step. The assessor also attaches one reference image per returned step.
        if question.intent == "IDENTIFY_STEP":
            return list(self.manual.steps)
        if question.intent == "CHECK_CLAIMED_STEP":
            claimed = self.by_id.get(question.claimed_step_id or "")
            return [claimed] if claimed is not None else []
        indexes: set[int] = set()
        if 0 <= state.current_step_index < len(self.manual.steps):
            indexes.add(state.current_step_index)
        claimed = self.by_id.get(question.claimed_step_id or "")
        if claimed is not None:
            index = self.manual.steps.index(claimed)
            indexes.update((max(0, index - 1), index))
        return [self.manual.steps[index] for index in sorted(indexes)]

    @staticmethod
    def prompt_data(steps: list[AssemblyStep]) -> list[dict]:
        out = []
        for step in steps:
            data = asdict(step)
            # Binary reference images are attached separately by the assessor;
            # avoid duplicating absolute file paths in the model's JSON input.
            data.pop("references", None)
            out.append(data)
        return out
