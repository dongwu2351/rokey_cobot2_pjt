"""AR-style overlay compositing for the pick-and-place window.

The rig is already a real AR tracker: every webcam has calibrated intrinsics
and a base->camera extrinsic, so a base-frame point lands on the right pixel.
What was missing was the *drawing* - flat 2px polylines carry no depth, so a
correctly projected plan still read as a diagram taped over the video.

This module keeps the geometry in the caller (it owns the cameras) and takes
already-projected pixels, adding three things the old direct-to-frame drawing
could not do:

* perspective weight - a ribbon whose width and colour follow depth,
* bloom - overlays drawn to a separate layer, blurred, added back, so lines
  emit light instead of sitting on top of the image,
* contact shadow - the same path flattened onto the table, darkening the
  frame, which is what makes a floating curve read as floating.

Everything routes through HudCanvas so `fancy=False` restores the previous
look through the same call sites: one implementation, and the F key can flip
between them live during a demo.
"""

from __future__ import annotations

import numpy as np
import cv2

try:                                            # crisp type beats HERSHEY
    from PIL import Image, ImageDraw, ImageFont
except Exception:                               # pillow missing -> putText
    Image = ImageDraw = ImageFont = None


FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

TABLE_Z_MM = 0.0            # shadow plane; a centimetre off is invisible here


# ----------------------------------------------------------------------
# geometry helpers (base frame, metres unless the name says _mm)
# ----------------------------------------------------------------------
def resample_polyline(points_mm, step_mm=12.0, min_points=2):
    """Even samples along a polyline so ribbon width varies smoothly."""
    points = np.asarray(points_mm, dtype=float)
    if len(points) < 2:
        return points
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(arc[-1])
    if total <= 1e-6:
        return points
    count = max(min_points, int(total / max(step_mm, 1.0)) + 1)
    wanted = np.linspace(0.0, total, count)
    out = np.empty((count, points.shape[1]), dtype=float)
    for axis in range(points.shape[1]):
        out[:, axis] = np.interp(wanted, arc, points[:, axis])
    return out


def arc_lengths(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros(len(points))
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])


def depths_in_camera(T_cam_base, points_m):
    """Optical-axis depth of each point - vectorised depth_of()."""
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    cam = (T_cam_base[:3, :3] @ points.T).T + T_cam_base[:3, 3]
    return cam[:, 2]


def capsule_rings(start_m, end_m, radius_m, rings=6, segments=18):
    """Rings along a capsule axis plus two cap rings.

    A pair of circles and a fat line reads as 2D no matter how correct the
    projection is; a stack of rings reads as a volume immediately."""
    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
        length = 1e-6
    else:
        axis = axis / length
    # Any vector not parallel to the axis gives an orthonormal ring basis.
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ seed)) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    out = []
    for t in np.linspace(0.0, 1.0, rings):
        centre = start + (end - start) * t
        out.append(centre + radius_m * circle)
    # Cap bulges: one shrinking ring past each end, so the ends read as
    # rounded rather than chopped off.
    for base, direction in ((start, -axis), (end, axis)):
        frac = 0.72
        offset = radius_m * np.sqrt(max(0.0, 1.0 - frac * frac))
        out.append(base + direction * offset + radius_m * frac * circle)
    return out


def capsule_spines(start_m, end_m, radius_m, count=4, segments=2):
    """Longitudinal lines along the capsule surface.

    Rings alone read as a coil; a few spines give the volume a silhouette."""
    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ seed)) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    out = []
    for theta in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
        offset = radius_m * (np.cos(theta) * u + np.sin(theta) * v)
        t = np.linspace(0.0, 1.0, max(2, segments))[:, None]
        out.append(start + (end - start) * t + offset)
    return out


def occluded_by_capsule(cam_pos_m, points_m, start_m, end_m, radius_m,
                        near_margin=0.97):
    """Which points sit *behind* the hand capsule from this camera.

    The webcams carry no depth, but the obstacle is a tracked 3D volume, so
    the occlusion it causes is known exactly: walk each viewing ray from the
    camera to the point and ask whether it passes through the capsule before
    reaching it. Drawing those stretches as a faint ghost is what stops the
    plan from looking painted on top of the operator's arm.

    Vectorised segment-to-segment closest approach; returns a bool array."""
    camera = np.asarray(cam_pos_m, dtype=float)
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    start = np.asarray(start_m, dtype=float)
    axis = np.asarray(end_m, dtype=float) - start

    to_point = points - camera                      # (N, 3) ray direction
    offset = camera - start                         # (3,)
    a = np.einsum("ij,ij->i", to_point, to_point)
    e = float(axis @ axis)
    f = float(axis @ offset)
    c = to_point @ offset
    b = to_point @ axis
    if e < 1e-12:                                   # degenerate capsule
        e = 1e-12
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 1e-12, (b * f - c * e) / denom, 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = np.clip((b * s + f) / e, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(a > 1e-12, np.clip((t * b - c) / a, 0.0, 1.0), 0.0)
    closest_ray = camera + s[:, None] * to_point
    closest_axis = start + t[:, None] * axis
    distance = np.linalg.norm(closest_ray - closest_axis, axis=1)
    # s < 1 means the blocking geometry is nearer than the point itself; a
    # point inside the capsule is not "behind" it.
    return (distance < radius_m) & (s < near_margin)


def axis_segments(pose, length_m=0.08):
    """(start, end, colour) for the three axes of a 4x4 pose."""
    origin = pose[:3, 3]
    colours = ((60, 60, 255), (60, 255, 60), (255, 180, 60))   # X, Y, Z (BGR)
    return [(origin, origin + pose[:3, axis] * length_m, colours[axis])
            for axis in range(3)]


# ----------------------------------------------------------------------
# text
# ----------------------------------------------------------------------
class _FontCache:
    """Rasterised strings, keyed by (text, size).

    HERSHEY strokes are half of why the window looked like a debug tool. Real
    glyphs cost a PIL round-trip, so each string is rendered once as an alpha
    mask and tinted at blit time - a moving label re-blits, never re-renders."""

    def __init__(self):
        self._fonts = {}
        self._masks = {}
        self._path = None
        if ImageFont is not None:
            for candidate in FONT_CANDIDATES:
                try:
                    ImageFont.truetype(candidate, 14)
                    self._path = candidate
                    break
                except Exception:
                    continue

    @property
    def available(self):
        return self._path is not None

    def _font(self, size):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self._path, size)
        return self._fonts[size]

    def mask(self, text, size):
        """Alpha mask (h, w) uint8 for `text`, or None if PIL is unavailable."""
        if not self.available or not text:
            return None
        key = (text, size)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        font = self._font(size)
        left, top, right, bottom = font.getbbox(text)
        width = max(1, right - left + 2)
        height = max(1, bottom - top + 2)
        image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(image).text((1 - left, 1 - top), text, font=font, fill=255)
        mask = np.asarray(image, dtype=np.uint8)
        if len(self._masks) > 512:              # labels carry live numbers
            self._masks.clear()
        self._masks[key] = mask
        return mask


_FONTS = _FontCache()


# ----------------------------------------------------------------------
# canvas
# ----------------------------------------------------------------------
class HudCanvas:
    """One quadrant's overlay. Buffers are allocated once and reused.

    Draws go to an additive `layer` (bloomed on composite) and a subtractive
    `dark` mask (contact shadows). With fancy=False every call falls through
    to the plain cv2 drawing this replaced, so the two looks share call sites.
    """

    def __init__(self, width, height, fancy=True, bloom_scale=4, bloom=True):
        self.width = width
        self.height = height
        self.fancy = fancy
        # Window chrome is text on solid bars: it wants the real font but not
        # a full-canvas blur, which at 1280x720 would cost more than all four
        # quadrants together.
        self.bloom_enabled = bloom
        self.bloom_scale = bloom_scale
        if not bloom:
            self.layer = self.dark = None
            self._layer_used = self._dark_used = False
            self._plain = None
            return
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.dark = np.zeros((height, width), dtype=np.uint8)
        self._layer_used = False
        self._dark_used = False
        self._plain = None                      # frame drawn on when not fancy
        # Scratch buffers. Every cv2 call below passes dst= into these: at
        # frame rate the allocator, not the filtering, was the cost (3.2ms ->
        # 1.0ms per quadrant), and this render shares a thread with the
        # control loop.
        small_w = max(1, width // bloom_scale)
        small_h = max(1, height // bloom_scale)
        self._small = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        self._small_blur = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        self._bloom = np.zeros((height, width, 3), dtype=np.uint8)
        self._dark_small = np.zeros((small_h, small_w), dtype=np.uint8)
        self._dark_soft = np.zeros((height, width), dtype=np.uint8)
        self._dark_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        self._out = np.zeros((height, width, 3), dtype=np.uint8)

    # -- lifecycle ----------------------------------------------------
    def begin(self, view):
        """Start a frame. `view` is the resized quadrant we will composite on."""
        if self.fancy and self.bloom_enabled:
            if self._layer_used:
                self.layer[:] = 0
            if self._dark_used:
                self.dark[:] = 0
            self._layer_used = False
            self._dark_used = False
        self._plain = view
        return self

    def _target(self):
        if self.fancy and self.bloom_enabled:
            self._layer_used = True
            return self.layer
        return self._plain

    # -- primitives ---------------------------------------------------
    def line(self, p0, p1, color, thickness=2):
        if p0 is None or p1 is None:
            return
        cv2.line(self._target(), _pt(p0), _pt(p1), color, thickness, cv2.LINE_AA)

    def polyline(self, pixels, color, thickness=2, closed=False):
        points = _valid_array(pixels)
        if points is None or len(points) < 2:
            return
        cv2.polylines(self._target(), [points], closed, color, thickness,
                      cv2.LINE_AA)

    def circle(self, centre, radius_px, color, thickness=2):
        if centre is None:
            return
        cv2.circle(self._target(), _pt(centre), max(3, int(radius_px)), color,
                   thickness, cv2.LINE_AA)

    def marker(self, centre, color, size=14):
        if centre is None:
            return
        cv2.drawMarker(self._target(), _pt(centre), color, cv2.MARKER_CROSS,
                       size, 2)

    # -- composite shapes --------------------------------------------
    def ribbon(self, pixels, depths, color_near, color_far, phase_mm=0.0,
               arc_mm=None, width_gain=520.0, width_range=(2.0, 13.0),
               occluded=None):
        """Depth-weighted path. Near segments are wide and bright, far ones
        thin and cold; `phase_mm` marches the gaps so the plan reads as live
        rather than as a static drawing.

        `occluded` marks samples hidden behind a tracked volume: those draw as
        a thin ghost rather than vanishing, so the operator can still see
        where the plan goes while it clearly passes *behind* their arm."""
        if not self.fancy:
            self.polyline(pixels, color_near, 2)
            return
        points = list(pixels)
        depths = np.asarray(depths, dtype=float)
        if arc_mm is None:
            arc_mm = np.zeros(len(points))
        span = float(np.nanmax(depths) - np.nanmin(depths)) or 1.0
        near = float(np.nanmin(depths))
        target = self._target()
        for index in range(len(points) - 1):
            first, second = points[index], points[index + 1]
            if first is None or second is None:
                continue
            hidden = occluded is not None and bool(occluded[index])
            if phase_mm and not hidden and _dash_gap(arc_mm[index], phase_mm):
                continue
            depth = max(0.05, float(depths[index]))
            half = float(np.clip(width_gain / (depth * 1000.0),
                                 width_range[0], width_range[1])) * 0.5
            t = float(np.clip((depth - near) / span, 0.0, 1.0))
            color = _lerp_color(color_near, color_far, t)
            if hidden:
                half = min(half, 1.2)
                color = tuple(int(channel * 0.34) for channel in color)
            quad = _segment_quad(first, second, half)
            if quad is None:
                continue
            # LINE_8: the bloom pass softens the edges anyway, and antialiased
            # fills cost about a third of the whole overlay budget.
            cv2.fillConvexPoly(target, quad, color, cv2.LINE_8)

    def shadow(self, pixels, strength=110, thickness=4, track_color=None):
        """Contact shadow on the table plane - the cue that sells height.

        Also draws a dim track in the glow layer: on a dark table a pure
        shadow has nothing to darken, and the ground reference disappears."""
        if not self.fancy:
            return
        points = _valid_array(pixels)
        if points is None or len(points) < 2:
            return
        cv2.polylines(self.dark, [points], False, int(strength), thickness,
                      cv2.LINE_8)
        self._dark_used = True
        if track_color is not None:
            cv2.polylines(self.layer, [points], False, track_color, 1,
                          cv2.LINE_AA)
            self._layer_used = True

    def drop_lines(self, pixels, shadow_pixels, color, every=10):
        """Sparse, faint verticals tying the path to its shadow. Too many of
        these read as a fence and hide the path they are supposed to place."""
        if not self.fancy:
            return
        count = min(len(pixels), len(shadow_pixels))
        for index in range(every // 2, count, every):
            top, bottom = pixels[index], shadow_pixels[index]
            if top is None or bottom is None:
                continue
            cv2.line(self.layer, _pt(top), _pt(bottom), color, 1, cv2.LINE_AA)
            self._layer_used = True

    def rings(self, ring_pixels, color, view_depths=None):
        """Capsule/volume rings, dimmed with depth so the far side recedes.

        The fade is normalised across THIS capsule, not over absolute depth:
        a forearm spans a few centimetres, so an absolute ramp would paint
        every ring the same brightness and the shell would stay flat."""
        if not self.fancy:
            return
        target = self._target()
        fades = None
        if view_depths is not None and len(view_depths) == len(ring_pixels):
            depths = np.asarray(view_depths, dtype=float)
            near, far = float(np.min(depths)), float(np.max(depths))
            span = far - near
            if span > 1e-4:
                fades = 1.0 - 0.6 * (depths - near) / span
        for index, ring in enumerate(ring_pixels):
            points = _valid_array(ring)
            if points is None or len(points) < 3:
                continue
            fade = 1.0 if fades is None else float(fades[index])
            cv2.polylines(target, [points], True,
                          tuple(int(c * fade) for c in color), 1, cv2.LINE_AA)

    def brackets(self, box, color, size_frac=0.28, thickness=2):
        """Corner brackets instead of a full rectangle: a reticle, not a
        detector debug box."""
        x1, y1, x2, y2 = (int(v) for v in box)
        if not self.fancy:
            cv2.rectangle(self._target(), (x1, y1), (x2, y2), color, thickness)
            return
        target = self._target()
        span = max(6, int(min(x2 - x1, y2 - y1) * size_frac))
        for corner_x, step_x in ((x1, 1), (x2, -1)):
            for corner_y, step_y in ((y1, 1), (y2, -1)):
                cv2.line(target, (corner_x, corner_y),
                         (corner_x + step_x * span, corner_y), color,
                         thickness, cv2.LINE_AA)
                cv2.line(target, (corner_x, corner_y),
                         (corner_x, corner_y + step_y * span), color,
                         thickness, cv2.LINE_AA)

    def text(self, text, org, color, size=15, shadow=True):
        if not text:
            return
        mask = _FONTS.mask(text, size) if self.fancy else None
        if mask is None:
            cv2.putText(self._target(), text, (int(org[0]), int(org[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color, 1,
                        cv2.LINE_AA)
            return
        # PIL bbox origin is top-left; callers think in baselines like putText.
        self._blit(mask, int(org[0]), int(org[1]) - mask.shape[0], color,
                   shadow=shadow)

    def _blit(self, mask, x, y, color, shadow=False):
        height, width = mask.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + width), min(self.height, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        clip = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        target = self._target()
        patch = target[y0:y1, x0:x1]
        tint = (clip[..., None].astype(np.uint16)
                * np.asarray(color, dtype=np.uint16)) // 255
        np.copyto(patch, np.minimum(patch.astype(np.uint16) + tint, 255)
                  .astype(np.uint8))
        if shadow and self.fancy and self.bloom_enabled:
            self.dark[y0:y1, x0:x1] = np.maximum(
                self.dark[y0:y1, x0:x1], (clip.astype(np.uint16) * 3 // 4)
                .astype(np.uint8))
            self._dark_used = True

    # -- output -------------------------------------------------------
    def composite(self, view):
        """Apply shadows, then bloom, then the crisp core."""
        if not self.fancy or not self.bloom_enabled:
            return view                          # already drawn straight on
        if not (self._layer_used or self._dark_used):
            return view
        if view.shape[:2] != (self.height, self.width):
            return view                          # size drifted; skip, not crash
        out = self._out
        np.copyto(out, view)
        if self._dark_used:
            # Blur small, then subtract: darkening the plate under the plan is
            # what makes a floating ribbon read as floating. Subtraction is
            # cheaper than a multiply and, on video, indistinguishable.
            cv2.resize(self.dark, (self._dark_small.shape[1],
                                   self._dark_small.shape[0]),
                       dst=self._dark_small, interpolation=cv2.INTER_LINEAR)
            cv2.GaussianBlur(self._dark_small, (0, 0), 1.6,
                             dst=self._dark_small)
            cv2.resize(self._dark_small, (self.width, self.height),
                       dst=self._dark_soft, interpolation=cv2.INTER_LINEAR)
            cv2.cvtColor(self._dark_soft, cv2.COLOR_GRAY2BGR,
                         dst=self._dark_bgr)
            cv2.subtract(out, self._dark_bgr, dst=out)
        if self._layer_used:
            cv2.resize(self.layer, (self._small.shape[1], self._small.shape[0]),
                       dst=self._small, interpolation=cv2.INTER_LINEAR)
            cv2.GaussianBlur(self._small, (0, 0), 3.0, dst=self._small_blur)
            cv2.resize(self._small_blur, (self.width, self.height),
                       dst=self._bloom, interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(out, 1.0, self._bloom, 0.85, 0.0, dst=out)
            cv2.add(out, self.layer, dst=out)
        return out


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _pt(pixel):
    return (int(round(pixel[0])), int(round(pixel[1])))


def _valid_array(pixels):
    points = [p for p in pixels if p is not None]
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.int32)


def _lerp_color(near, far, t):
    return tuple(int(near[i] + (far[i] - near[i]) * t) for i in range(3))


def _dash_gap(arc, phase, period=52.0, duty=0.74):
    return ((arc + phase) % period) > period * duty


def _segment_quad(first, second, half_width):
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return None
    nx, ny = -dy / length * half_width, dx / length * half_width
    return np.array([
        [first[0] + nx, first[1] + ny],
        [second[0] + nx, second[1] + ny],
        [second[0] - nx, second[1] - ny],
        [first[0] - nx, first[1] - ny],
    ], dtype=np.int32)
