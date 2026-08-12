from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from .analyzer import OpenAIManualAnalyzer
from .pdf_tools import crop_page_content, render_pages, validate_pdf
from .schemas import GeneratedManual


class ManualPackageGenerator:
    def __init__(self, output_root: Path, *, analyzer: Any | None = None,
                 dpi: int = 160) -> None:
        self.output_root = output_root.resolve()
        self.analyzer = analyzer or OpenAIManualAnalyzer()
        self.dpi = dpi

    def generate(self, pdf: Path, *, detail: str = "high",
                 overwrite: bool = False) -> Path:
        pdf = pdf.resolve()
        page_count = validate_pdf(pdf)
        result: GeneratedManual = self.analyzer.analyze(pdf, detail=detail)
        self._validate_result(result, page_count)
        target = self.output_root / result.product_slug
        self.output_root.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"출력 제품 폴더가 이미 있습니다: {target} (--overwrite로 백업 후 교체)")

        with tempfile.TemporaryDirectory(prefix="manual-build-", dir=self.output_root) as tmp:
            build = Path(tmp) / result.product_slug
            rendered_dir = Path(tmp) / "rendered"
            build.mkdir()
            pages = render_pages(pdf, rendered_dir, dpi=self.dpi)
            references = self._copy_step_pages(result, pages, build)
            shutil.copy2(pdf, build / "source_manual.pdf")
            manual_data = self._manual_dict(result, references, pdf.name)
            (build / "assembly.yaml").write_text(
                yaml.safe_dump(manual_data, allow_unicode=True, sort_keys=False,
                               width=1000), encoding="utf-8")
            manifest = self._manifest(result, pdf, page_count, references)
            (build / "generation_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            (build / "REVIEW_REQUIRED.md").write_text(
                self._review_report(result, pdf.name), encoding="utf-8")
            self._validate_generated_yaml(build / "assembly.yaml")

            if target.exists():
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup = target.with_name(f"{target.name}.backup_{stamp}")
                target.rename(backup)
            build.rename(target)
        return target

    @staticmethod
    def _validate_result(result: GeneratedManual, page_count: int) -> None:
        part_ids = {item.id for item in result.parts}
        tool_ids = {item.id for item in result.tools}
        step_ids = [step.id for step in result.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("AI 결과에 중복 단계 ID가 있습니다")
        orders = [step.order for step in result.steps]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("단계 order가 1부터 연속적이지 않습니다")
        for step in result.steps:
            unknown_parts = set(step.required_parts) - part_ids
            unknown_tools = set(step.required_tools) - tool_ids
            bad_pages = [p for p in step.source_pages if p < 1 or p > page_count]
            if unknown_parts or unknown_tools or bad_pages:
                raise ValueError(
                    f"{step.id} 참조 오류: parts={sorted(unknown_parts)}, "
                    f"tools={sorted(unknown_tools)}, pages={bad_pages}")

    @staticmethod
    def _copy_step_pages(result: GeneratedManual, pages: dict[int, Path],
                         build: Path) -> dict[str, list[str]]:
        references: dict[str, list[str]] = {}
        for step in sorted(result.steps, key=lambda item: item.order):
            step_dir = build / "media" / step.id
            step_dir.mkdir(parents=True, exist_ok=True)
            refs: list[str] = []
            for page_number in sorted(set(step.source_pages)):
                name = f"page_{page_number:03d}.jpg"
                crop_page_content(pages[page_number], step_dir / name)
                refs.append(f"media/{step.id}/{name}")
            references[step.id] = refs
        return references

    @staticmethod
    def _manual_dict(result: GeneratedManual, references: dict[str, list[str]],
                     source_name: str) -> dict[str, Any]:
        steps = []
        ordered = sorted(result.steps, key=lambda item: item.order)
        for index, step in enumerate(ordered):
            preconditions = []
            if index > 0:
                preconditions.append({
                    "id": f"{ordered[index - 1].id}_done",
                    "type": "manual_confirmation",
                    "description": step.precondition_description
                    or f"{ordered[index - 1].title} 완료",
                })
            errors = [
                {"id": f"visual_error_{n:02d}", "type": "vision_review",
                 "description": value}
                for n, value in enumerate(step.error_conditions, start=1)
            ]
            visual = step.visual_state.model_dump(mode="json")
            visual["source_evidence"] = step.source_evidence
            visual["uncertain"] = list(dict.fromkeys(
                visual.get("uncertain", []) + step.uncertainties))
            visual_checks = []
            for n, value in enumerate(step.visual_state.new_elements, start=1):
                visual_checks.append({
                    "id": f"visual_element_{n:02d}", "kind": "required",
                    "description": value, "required": True,
                })
            for n, value in enumerate(step.visual_state.spatial_relations, start=1):
                visual_checks.append({
                    "id": f"visual_relation_{n:02d}", "kind": "required",
                    "description": value, "required": True,
                })
            for n, value in enumerate(step.error_conditions, start=1):
                visual_checks.append({
                    "id": f"visual_error_{n:02d}", "kind": "error",
                    "description": value, "required": False,
                })
            steps.append({
                "id": step.id, "order": step.order, "title": step.title,
                "instruction": step.instruction,
                "required_parts": step.required_parts,
                "required_tools": step.required_tools,
                "safety_notes": step.safety_notes,
                "preconditions": preconditions,
                "completion_conditions": [{
                    "id": "operator_confirmed", "type": "manual_confirmation",
                    "description": step.completion_description,
                }],
                "error_conditions": errors,
                "visual_checks": visual_checks,
                "references": {"images": references[step.id], "videos": []},
                "visual_state": visual,
                "source_pages": sorted(set(step.source_pages)),
            })
        return {
            "manual_id": result.manual_id, "product": result.product,
            "version": result.version,
            "description": result.description,
            "source": {"pdf": "source_manual.pdf", "original_filename": source_name,
                       "title": result.source_document_title,
                       "manufacturer": result.manufacturer,
                       "generated_by_ai": True,
                       "requires_human_review": True},
            "parts": [item.model_dump(mode="json") for item in result.parts],
            "tools": [item.model_dump(mode="json") for item in result.tools],
            "global_warnings": result.global_warnings,
            "steps": steps,
        }

    @staticmethod
    def _manifest(result, pdf, page_count, references):
        return {
            "source_pdf": pdf.name, "source_size_bytes": pdf.stat().st_size,
            "source_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "source_pages": page_count, "manual_id": result.manual_id,
            "product_slug": result.product_slug, "step_count": len(result.steps),
            "generated_at_ms": round(time.time() * 1000),
            "step_images": references, "human_review_required": True,
        }

    @staticmethod
    def _review_report(result: GeneratedManual, source_name: str) -> str:
        lines = [
            f"# {result.product} PDF 변환 검수 보고서", "",
            f"- 원본 PDF: `{source_name}`",
            f"- 문서 제목: {result.source_document_title}",
            f"- 생성 단계 수: {len(result.steps)}", "- 상태: **사람 검수 전 초안**", "",
            "## 반드시 확인할 항목", "",
        ]
        items = result.review_required or ["원본 PDF와 단계 순서 및 체결 기준 대조"]
        lines.extend(f"- {item}" for item in items)
        lines += ["", "## 단계별 불확실성", ""]
        for step in sorted(result.steps, key=lambda item: item.order):
            uncertain = list(dict.fromkeys(step.visual_state.uncertain + step.uncertainties))
            lines.append(f"### {step.title}")
            lines.append("")
            lines.append(f"- 원본 페이지: {', '.join(map(str, step.source_pages))}")
            lines.extend(f"- {item}" for item in uncertain)
            if not uncertain:
                lines.append("- AI가 별도 불확실성을 기록하지 않았음(사람 검수는 필요)")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _validate_generated_yaml(path: Path) -> None:
        from assembly_copilot.manual import load_manual
        manual = load_manual(path)
        if not all(step.visual_state for step in manual.steps):
            raise ValueError("생성된 매뉴얼의 일부 단계에 visual_state가 없습니다")
