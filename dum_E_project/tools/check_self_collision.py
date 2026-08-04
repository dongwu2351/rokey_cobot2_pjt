#!/usr/bin/env python3
"""
collision sphere 가 과도해서 self-collision 오탐을 내는지 측정한다.

무작위 관절 자세를 대량으로 뽑아, SRDF 에서 '무시하지 않는' 링크 쌍끼리
구-구 최소거리를 계산한다. 거리가 음수면 그 자세는 자기충돌로 판정되며,
실제로는 안 부딪히는 자세라면 그게 곧 오탐이다.

실행:  ~/curobo_env/bin/python check_self_collision.py
"""
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
URDF = HERE / "m0609_flat.urdf"
SPHERES = HERE / "collision_spheres.yaml"

LINKS = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]

# URDF 의 joint origin (parent -> child). 모든 축은 z (0 0 1), revolute.
JOINTS = [  # (xyz, rpy)
    ([0.0, 0.0, 0.1345], [0.0, 0.0, 0.0]),          # joint_1: base_link -> link_1
    ([0.0, 0.0062, 0.0], [0.0, -1.571, -1.571]),    # joint_2: link_1 -> link_2
    ([0.411, 0.0, 0.0], [0.0, 0.0, 1.571]),         # joint_3: link_2 -> link_3
    ([0.0, -0.368, 0.0], [1.571, 0.0, 0.0]),        # joint_4: link_3 -> link_4
    ([0.0, 0.0, 0.0], [-1.571, 0.0, 0.0]),          # joint_5: link_4 -> link_5
    ([0.0, -0.121, 0.0], [1.571, 0.0, 0.0]),        # joint_6: link_5 -> link_6
]

# MoveIt 이 쓰는 실제 운용 한계 (URDF 의 ±2pi 보다 현실적)
JOINT_LIMITS = np.array([
    [-3.14, 3.14], [-1.30, 1.30], [-2.00, 2.00],
    [-3.14, 3.14], [-2.00, 2.00], [-3.14, 3.14],
])

# SRDF 의 disable_collisions -> 검사하지 않는 쌍
IGNORE = {
    ("base_link", "link_1"), ("base_link", "link_2"),
    ("link_1", "link_2"), ("link_1", "link_3"),
    ("link_2", "link_3"),
    ("link_3", "link_4"), ("link_3", "link_5"), ("link_3", "link_6"),
    ("link_4", "link_5"),
    ("link_5", "link_6"),
}


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = (
        np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    )
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fk(q):
    """각 링크의 base_link 기준 (R, t) 반환."""
    out = {"base_link": (np.eye(3), np.zeros(3))}
    R, t = np.eye(3), np.zeros(3)
    for i, (xyz, rpy) in enumerate(JOINTS):
        Rj = rpy_to_mat(*rpy) @ rot_z(q[i])
        t = t + R @ np.array(xyz)
        R = R @ Rj
        out[LINKS[i + 1]] = (R.copy(), t.copy())
    return out


def main():
    data = yaml.safe_load(SPHERES.read_text())["collision_spheres"]
    local = {k: (np.array([s["center"] for s in v]),
                 np.array([s["radius"] for s in v])) for k, v in data.items()}

    pairs = [(a, b) for i, a in enumerate(LINKS) for b in LINKS[i + 1:]
             if (a, b) not in IGNORE]
    print(f"검사 대상 링크 쌍 {len(pairs)}개: " +
          ", ".join(f"{a}~{b}" for a, b in pairs) + "\n")

    rng = np.random.default_rng(0)
    N = 20000
    Q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(N, 6))

    worst = {p: np.inf for p in pairs}
    hits = {p: 0 for p in pairs}
    n_collide = 0

    for q in Q:
        T = fk(q)
        world = {}
        for lk in LINKS:
            C, R = local[lk]
            Rm, tv = T[lk]
            world[lk] = (C @ Rm.T + tv, R)

        bad = False
        for a, b in pairs:
            Ca, Ra = world[a]
            Cb, Rb = world[b]
            d = np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=-1)
            d -= Ra[:, None] + Rb[None, :]
            m = d.min()
            if m < worst[(a, b)]:
                worst[(a, b)] = m
            if m < 0:
                hits[(a, b)] += 1
                bad = True
        if bad:
            n_collide += 1

    print(f"무작위 자세 {N}개 중 자기충돌 판정: {n_collide}개 "
          f"({n_collide / N * 100:.2f}%)\n")
    print(f"{'링크 쌍':24s} {'최소여유[m]':>12s} {'충돌자세':>10s}")
    for p in sorted(pairs, key=lambda p: worst[p]):
        flag = "  <-- 접촉" if worst[p] < 0 else ""
        print(f"{p[0]+'~'+p[1]:24s} {worst[p]:12.4f} "
              f"{hits[p]:9d}{flag}")


if __name__ == "__main__":
    main()
