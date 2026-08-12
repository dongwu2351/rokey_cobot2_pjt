"""Which object is a person pointing at?

The rig already turns two camera views of a point into a 3D position. Two
points on the index finger - the knuckle and the tip - therefore give a 3D
RAY, and the object a person means is the one that ray passes closest to.

Closest is measured as an ANGLE, not a distance. A finger aimed 3 cm wide of
a tool at arm's length is aimed at it; the same 3 cm at the far end of the
bench is aimed at its neighbour. Angle is what the pointer controls and what
the observer reads, so it is what the match should use.

Nothing here talks to a camera or a robot: it takes triangulated points and
returns a choice, which keeps it testable and reusable from the app, a demo
tool, or an LLM front end.
"""

from __future__ import annotations

import math

import numpy as np

#: Beyond this angle from the finger's line, a person would not say they were
#: pointing at it. Roughly the cone your own finger feels like it covers.
MAX_POINT_ANGLE_DEG = 22.0
#: An object behind the fingertip is not being pointed at, however well the
#: infinite line happens to pass through it.
MIN_FORWARD_MM = 40.0
#: The knuckle-to-tip segment is short (~7 cm), so small pixel errors swing
#: the direction. Reject a ray built from a segment shorter than this.
MIN_SEGMENT_MM = 35.0


def pointer_ray(rig, pointers_by_cam, require_gesture=True):
    """(origin_mm, unit_direction) of the index finger, or None.

    `pointers_by_cam` is {cam: (mcp_u, mcp_v, tip_u, tip_v, is_pointing)} -
    the first hand from HandIntruderDetector.detect(). Two cameras minimum,
    because one view cannot place a finger in space."""
    usable = {cam: p for cam, p in pointers_by_cam.items() if p is not None}
    if require_gesture:
        usable = {cam: p for cam, p in usable.items() if p[4]}
    if len(usable) < 2:
        return None
    knuckle, gap_a = rig.triangulate({cam: (p[0], p[1]) for cam, p in usable.items()})
    tip, gap_b = rig.triangulate({cam: (p[2], p[3]) for cam, p in usable.items()})
    if knuckle is None or tip is None:
        return None
    # A finger is small; if the views disagree by more than a finger's width
    # they are not looking at the same finger.
    if max(gap_a or 1.0, gap_b or 1.0) > 0.035:
        return None
    origin = np.asarray(tip, dtype=float) * 1000.0
    axis = (np.asarray(tip, dtype=float) - np.asarray(knuckle, dtype=float)) * 1000.0
    length = float(np.linalg.norm(axis))
    if length < MIN_SEGMENT_MM:
        return None
    return origin, axis / length


def angle_to(origin_mm, direction, point_mm):
    """(angle deg off the ray, distance along it mm, perpendicular miss mm)."""
    offset = np.asarray(point_mm, dtype=float) - np.asarray(origin_mm, dtype=float)
    forward = float(np.dot(offset, direction))
    perpendicular = float(np.linalg.norm(offset - forward * np.asarray(direction)))
    if forward <= 1e-6:
        return 180.0, forward, perpendicular
    return math.degrees(math.atan2(perpendicular, forward)), forward, perpendicular


def select_pointed(candidates, origin_mm, direction,
                   max_angle_deg=MAX_POINT_ANGLE_DEG,
                   min_forward_mm=MIN_FORWARD_MM):
    """Pick from {key: position_mm}. Returns (key, info) or (None, info).

    info carries every candidate's angle so a caller can show why - a UI that
    only says "this one" is impossible to trust or debug."""
    scored = {}
    best_key, best_angle = None, None
    for key, position in candidates.items():
        angle, forward, miss = angle_to(origin_mm, direction, position)
        scored[key] = {"angle_deg": angle, "forward_mm": forward,
                       "miss_mm": miss}
        if forward < min_forward_mm or angle > max_angle_deg:
            continue
        if best_angle is None or angle < best_angle:
            best_key, best_angle = key, angle
    # A near-tie is not a choice. Two tools side by side within a couple of
    # degrees means the honest answer is "which one?", not a coin flip.
    runner_up = None
    for key, info in scored.items():
        if key == best_key or info["angle_deg"] > max_angle_deg:
            continue
        if info["forward_mm"] < min_forward_mm:
            continue
        if runner_up is None or info["angle_deg"] < scored[runner_up]["angle_deg"]:
            runner_up = key
    ambiguous = (
        best_key is not None and runner_up is not None
        and scored[runner_up]["angle_deg"] - best_angle < 4.0
    )
    return best_key, {"scores": scored, "best_angle_deg": best_angle,
                      "runner_up": runner_up, "ambiguous": ambiguous}


def ray_plane_point(origin_mm, direction, plane_z_mm=0.0, max_distance_mm=1500.0):
    """Where the finger's line meets the work surface, or None.

    Pointing at a PLACE rather than at an object: for "am I doing this bit
    right?" there may be no detected object at all, just a spot on the part
    the person is working on. The table plane turns the ray back into a
    single point the robot can go and look at."""
    origin = np.asarray(origin_mm, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if abs(direction[2]) < 1e-6:
        return None                       # pointing along the table, not at it
    distance = (plane_z_mm - origin[2]) / direction[2]
    if distance <= 0.0 or distance > max_distance_mm:
        return None                       # behind the finger, or absurdly far
    return origin + direction * distance


class PointingSmoother:
    """Hold a selection until another one clearly wins.

    A raw per-frame choice flickers between neighbours while the hand
    breathes. Requiring a challenger to lead for several consecutive frames
    keeps the highlight still without adding lag to a deliberate move."""

    def __init__(self, switch_frames=4, hold_seconds=1.0):
        self.switch_frames = switch_frames
        self.hold_seconds = hold_seconds
        self.selection = None
        self._candidate = None
        self._count = 0
        self._last_seen = None

    def update(self, key, now):
        if key is not None and key == self.selection:
            self._candidate, self._count = None, 0
            self._last_seen = now
            return self.selection
        if key is None:
            # Keep the last choice briefly: a finger drops out of a view for
            # a frame or two constantly, and blanking the highlight each time
            # makes the whole feature look broken.
            if (self._last_seen is not None
                    and now - self._last_seen < self.hold_seconds):
                return self.selection
            self.selection = None
            self._candidate, self._count = None, 0
            return None
        if key == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = key, 1
        if self._count >= self.switch_frames:
            self.selection = key
            self._candidate, self._count = None, 0
        self._last_seen = now
        return self.selection
