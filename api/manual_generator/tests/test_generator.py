from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from assembly_copilot.manual import load_manual
from manual_generator.generator import ManualPackageGenerator
from manual_generator.schemas import GeneratedManual


def sample_result() -> GeneratedManual:
    return GeneratedManual.model_validate({
        "manual_id": "TEST_PRODUCT_V1", "product_slug": "test_product",
        "product": "테스트 제품", "version": "1.0-draft",
        "description": "PDF 변환 테스트", "manufacturer": None,
        "source_document_title": "Test manual",
        "parts": [{"id": "part_a", "name_ko": "부품 A",
                   "visual_queries": ["red square part"], "description": "붉은 부품"}],
        "tools": [], "global_warnings": [], "review_required": ["치수 확인"],
        "steps": [{
            "id": "step_01", "order": 1, "title": "부품 배치",
            "instruction": "부품 A를 작업대에 놓으세요.",
            "required_parts": ["part_a"], "required_tools": [],
            "safety_notes": [], "precondition_description": None,
            "completion_description": "부품이 놓여 있음", "error_conditions": [],
            "visual_state": {"summary": "붉은 부품이 보임",
                             "new_elements": ["부품 A"],
                             "spatial_relations": ["작업대 위"],
                             "distinguish_prev": None,
                             "distinguish_next": None,
                             "camera_note": "정면", "uncertain": []},
            "source_pages": [1], "source_evidence": ["1페이지 도면"],
            "uncertainties": [],
        }],
    })


class FakeAnalyzer:
    def analyze(self, pdf, *, detail="high"):
        return sample_result()


class GeneratorTest(unittest.TestCase):
    def test_builds_loadable_package_and_step_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "input.pdf"
            Image.new("RGB", (300, 300), "white").save(pdf, "PDF")
            target = ManualPackageGenerator(
                root / "output", analyzer=FakeAnalyzer(), dpi=72).generate(pdf)

            manual = load_manual(target / "assembly.yaml")
            self.assertEqual(manual.manual_id, "TEST_PRODUCT_V1")
            self.assertTrue(manual.steps[0].visual_state)
            self.assertTrue(manual.steps[0].visual_checks)
            self.assertTrue((target / "media/step_01/page_001.jpg").is_file())
            self.assertTrue((target / "source_manual.pdf").is_file())
            self.assertTrue((target / "generation_manifest.json").is_file())
            self.assertTrue((target / "REVIEW_REQUIRED.md").is_file())

    def test_refuses_existing_product_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "input.pdf"
            Image.new("RGB", (100, 100), "white").save(pdf, "PDF")
            generator = ManualPackageGenerator(root / "output", analyzer=FakeAnalyzer(), dpi=72)
            generator.generate(pdf)
            with self.assertRaises(FileExistsError):
                generator.generate(pdf)


if __name__ == "__main__":
    unittest.main()
