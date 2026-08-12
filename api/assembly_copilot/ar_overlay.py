from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np

from .models import AssemblyState, AssemblyStep


def draw_overlay(frame, step: AssemblyStep | None, state: AssemblyState,
                 message: str = "", reference_image=None, controls_text: str | None = None,
                 reference_label: str | None = None):
    """Screen-space AR HUD. Spatial arrows can be added after task geometry exists."""
    out = frame.copy()
    h, w = out.shape[:2]
    panel_h = min(190, h)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
    status_color = (80, 210, 255) if state.status == "WARNING" else (100, 230, 130)
    title = "ASSEMBLY COMPLETE" if step is None else f"STEP {step.order}: {step.title}"
    _text(out, title, (20, 34), 0.78, status_color, 2)
    if step is not None:
        _multiline(out, step.instruction, 20, 66, w - 40, (235, 235, 235))
    if message:
        _multiline(out, message, 20, 128, w - 40, (100, 220, 255))
    controls = (controls_text or
                "[A] ask  [C] confirm  [P] photo  [V] video  [B] back  [Q] quit")
    _text(out, controls,
          (20, panel_h - 14), 0.43, (190, 190, 190), 1)
    if reference_image is not None:
        ref = (reference_image.copy() if isinstance(reference_image, np.ndarray)
               else cv2.imread(str(reference_image)) if Path(str(reference_image)).is_file()
               else None)
        if ref is not None:
            # A reference request is a viewing mode, not a tiny AR thumbnail.
            # Cover the camera area below the status header with a dedicated,
            # high-contrast panel so the selected step is unmistakable.
            ref = _crop_reference_content(ref)
            area_y = panel_h
            cv2.rectangle(out, (0, area_y), (w, h), (12, 14, 18), -1)
            label = reference_label or f"REFERENCE · {Path(str(reference_image)).name}"
            _text(out, label, (20, area_y + 34), .7, (255, 210, 80), 2)
            max_h = max(80, h - area_y - 62)
            max_w = max(120, w - 40)
            scale = min(max_w / ref.shape[1], max_h / ref.shape[0])
            rw = max(1, round(ref.shape[1] * scale))
            rh = max(1, round(ref.shape[0] * scale))
            ref = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_AREA)
            x, y = (w - rw) // 2, area_y + 50 + max(0, (max_h - rh) // 2)
            out[y:y + rh, x:x + rw] = ref
            cv2.rectangle(out, (x, y), (x + rw, y + rh), (255, 210, 80), 2)
    return out


def _crop_reference_content(ref: np.ndarray) -> np.ndarray:
    """Remove mostly blank PDF-page margins without cropping real photographs."""
    if ref.ndim != 3 or min(ref.shape[:2]) < 20:
        return ref
    gray = cv2.cvtColor(ref[:, :, :3], cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 245)
    if len(xs) < 100:
        return ref
    pad = max(8, round(min(ref.shape[:2]) * .025))
    x0, x1 = max(0, int(xs.min()) - pad), min(ref.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(ref.shape[0], int(ys.max()) + pad + 1)
    # Do not turn a few page specks into an extreme crop.
    if (x1 - x0) * (y1 - y0) < ref.shape[0] * ref.shape[1] * .08:
        return ref
    return ref[y0:y1, x0:x1]


def _text(img, text, origin, scale, color, thickness):
    if text.isascii():
        cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)
        return
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        rgb = tuple(reversed(color))
        draw.text(origin, text, font=_font(max(13, round(30 * scale))), fill=rgb,
                  stroke_width=max(0, thickness - 1))
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, "See terminal for instruction", origin,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


@lru_cache(maxsize=8)
def _font(size):
    from PIL import ImageFont
    candidates = (
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _multiline(img, text, x, y, max_width, color):
    words = text.split()
    line, lines = "", []
    for word in words:
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, .52, 1)[0][0]
        if line and width > max_width:
            lines.append(line); line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, value in enumerate(lines[:2]):
        _text(img, value, (x, y + index * 24), .52, color, 1)
