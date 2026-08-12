from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import OpenAIManualAnalyzer
from .generator import ManualPackageGenerator

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(
        description="downloaded_manuals의 PDF를 제품별 assembly_manual 패키지로 변환")
    parser.add_argument("pdf", nargs="*", type=Path,
                        help="처리할 PDF. 생략하면 --input-dir의 모든 PDF 처리")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "downloaded_manuals")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assembly_manuals")
    parser.add_argument("--model", help="기본값: MANUAL_GENERATOR_MODEL 또는 gpt-5.6")
    parser.add_argument("--detail", choices=("low", "auto", "high"), default="high")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true",
                        help="동일 제품 폴더를 타임스탬프 백업 후 새 결과로 교체")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    pdfs = [item.resolve() for item in args.pdf]
    if not pdfs:
        pdfs = sorted(args.input_dir.resolve().glob("*.pdf"))
    if not pdfs:
        print(f"처리할 PDF가 없습니다: {args.input_dir}", file=sys.stderr)
        return 2
    analyzer = OpenAIManualAnalyzer(model=args.model)
    generator = ManualPackageGenerator(args.output_dir, analyzer=analyzer, dpi=args.dpi)
    failed = 0
    for pdf in pdfs:
        print(f"[분석 시작] {pdf}", flush=True)
        try:
            target = generator.generate(pdf, detail=args.detail, overwrite=args.overwrite)
            print(f"[생성 완료] {target}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[생성 실패] {pdf.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
