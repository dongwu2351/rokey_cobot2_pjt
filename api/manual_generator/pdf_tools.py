from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader


MAX_PDF_BYTES = 50 * 1024 * 1024


def validate_pdf(path: Path) -> int:
    path = path.resolve()
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError(f"PDF 파일이 아닙니다: {path}")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(f"PDF가 API 단일 파일 제한 50MB를 초과합니다: {path.name}")
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"암호화된 PDF를 읽을 수 없습니다: {path.name}") from exc
    count = len(reader.pages)
    if count < 1:
        raise ValueError(f"페이지가 없는 PDF입니다: {path.name}")
    return count


def render_pages(pdf: Path, destination: Path, *, dpi: int = 160) -> dict[int, Path]:
    """Render every PDF page to JPEG using Poppler's pdftoppm."""
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("pdftoppm이 없습니다. Ubuntu에서는 poppler-utils를 설치하세요.")
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination / "page"
    subprocess.run(
        [executable, "-jpeg", "-r", str(dpi), str(pdf), str(prefix)],
        check=True, capture_output=True, text=True,
    )
    rendered = sorted(destination.glob("page-*.jpg"))
    if not rendered:
        raise RuntimeError(f"PDF 페이지 렌더링 결과가 없습니다: {pdf.name}")
    return {index: path for index, path in enumerate(rendered, start=1)}


def crop_page_content(source: Path, destination: Path, *, padding: int = 20) -> None:
    """Remove blank PDF margins without inventing or relabelling visual evidence."""
    import cv2
    import numpy as np

    image = cv2.imread(str(source))
    if image is None:
        shutil.copy2(source, destination)
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Include light diagrams/text but ignore scanner-white borders.
    mask = gray < 245
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        shutil.copy2(source, destination)
        return
    x, y, width, height = cv2.boundingRect(points)
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        shutil.copy2(source, destination)
        return
    cv2.imwrite(str(destination), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
