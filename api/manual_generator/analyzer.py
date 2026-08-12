from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .schemas import GeneratedManual

ROOT = Path(__file__).resolve().parents[1]

INSTRUCTIONS = """당신은 제조사 PDF 조립서를 DUM-E 작업 코파일럿용 데이터로 변환한다.
PDF의 본문과 모든 페이지 이미지를 함께 분석한다. 문서에 명시되지 않은 부품 규격,
토크, 방향, 안전 수치를 만들어내지 않는다. 추론한 내용은 uncertainties와
review_required에 명시한다. 광고, 보증, 목차, 반복 설명은 조립 단계로 만들지 않는다.
각 단계는 작업자가 수행하는 하나의 검증 가능한 상태 변화여야 하며 source_pages에는
근거가 있는 1부터 시작하는 PDF 페이지 번호만 넣는다. source_evidence에는 해당 페이지의
근거 문구나 도면 번호를 짧게 요약한다. visual_state는 기준 JPG 없이 RealSense 화면만으로
현재 단계를 구분할 수 있도록 물체, 위치, 방향, 결합 관계와 전후 단계 차이를 구체적으로
작성한다. 카메라로 확인할 수 없는 토크, 내부 나사산, 전기적 정상 여부는 uncertain에 둔다.
모든 설명은 한국어로 쓰되 ID와 product_slug는 규칙에 맞는 영문 식별자로 작성한다."""


class OpenAIManualAnalyzer:
    def __init__(self, *, model: str | None = None, client: Any | None = None,
                 timeout_seconds: float = 180.0) -> None:
        load_dotenv(ROOT / "llm" / "VoiceProcessing" / ".env")
        self.model = model or os.getenv("MANUAL_GENERATOR_MODEL", "gpt-5.6")
        self._client = client
        self.timeout_seconds = timeout_seconds

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다")
            self._client = OpenAI(api_key=key, max_retries=1)
        return self._client

    def analyze(self, pdf: Path, *, detail: str = "high") -> GeneratedManual:
        encoded = base64.b64encode(pdf.read_bytes()).decode("ascii")
        response = self.client.responses.parse(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=[{"role": "user", "content": [
                {"type": "input_file", "filename": pdf.name,
                 "file_data": f"data:application/pdf;base64,{encoded}",
                 "detail": detail},
                {"type": "input_text", "text": (
                    "이 PDF에서 공식적으로 확인되는 조립 절차를 추출하고, "
                    "단계별 RealSense 시각 판정 기준을 포함한 매뉴얼을 생성하세요.")},
            ]}],
            text_format=GeneratedManual,
            max_output_tokens=12000,
            store=False,
            timeout=self.timeout_seconds,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("PDF 분석 응답에 구조화된 매뉴얼이 없습니다")
        return parsed
