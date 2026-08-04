#!/usr/bin/env python3
"""
M0609 collision sphere 생성기.

각 링크의 STL 정점을 k-means로 군집화하고, 군집을 완전히 감싸는 구를 만든다.
구 개수는 링크별 목표 반지름을 만족할 때까지 자동으로 늘어난다.

출력: collision_spheres.yaml  (cuRobo robot_cfg 및 RViz 시각화에 그대로 사용)

의존성: numpy 만 필요. (시스템 python3 로 실행 가능 — cuRobo 환경 불필요)
"""
import struct
from pathlib import Path

import numpy as np

MESH_DIR = Path(
    "/home/rokey/cobot_ws/src/doosan-robot2/dsr_description2/mujoco_models/m0609/assets"
)
OUT = Path(__file__).parent / "collision_spheres.yaml"

# URDF 상 모든 <collision> origin 이 rpy=0 xyz=0 이므로
# 메시 좌표 = 링크 좌표 (mm -> m 스케일만 적용하면 된다)
LINK_MESHES = {
    "base_link": ["MF0609_0_0.stl"],
    "link_1": ["MF0609_1_0.stl"],
    "link_2": ["MF0609_2_0.stl", "MF0609_2_1.stl", "MF0609_2_2.stl"],
    "link_3": ["MF0609_3_0.stl"],
    "link_4": ["MF0609_4_0.stl", "MF0609_4_1.stl"],
    "link_5": ["MF0609_5_0.stl"],
    "link_6": ["MF0609_6_0.stl"],
}

# 링크별 목표 구 반지름 상한 [m].
# 중심이 내부에 앉으므로, 대략 "그 부위 단면의 절반" 정도가 적정선이다.
# 이보다 헐겁게 잡으면 구 몇 개로 끝나버려 표면 밖 돌출이 커진다.
TARGET_R = {
    "base_link": 0.060,  # 본체 높이가 0.097 뿐인 납작한 형상
    "link_1": 0.065,
    "link_2": 0.065,  # 상완 튜브 단면 0.10x0.10 -> 반대각 0.071 이 하한
    "link_3": 0.055,
    "link_4": 0.060,
    "link_5": 0.055,
    "link_6": 0.055,
}
MAX_SPHERES = 30  # 메시당 상한 (계산 비용 방어)
FIT_SAMPLE = 20000  # k-means 는 이만큼만 샘플링해서 적합 (반지름은 전체 정점으로 계산)


def load_stl(path: Path) -> np.ndarray:
    """binary STL -> (N,3) 표면 정점 배열 [m]"""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84 : 84 + n * 50], dtype=np.uint8).reshape(n, 50)
    floats = rec[:, :48].copy().view(np.float32).reshape(n, 12)
    verts = floats[:, 3:12].reshape(-1, 3)  # normal 3개는 버리고 v0,v1,v2
    return verts.astype(np.float64) * 0.001


def interior_points(path: Path, surface: np.ndarray) -> np.ndarray:
    """메시 내부(부피)를 채우는 점들. 실패하면 표면 정점으로 폴백.

    k-means 를 표면 정점에 돌리면 중심이 표면 위에 앉아서 구가 통째로 바깥으로
    튀어나간다(반경의 85% 가 밖으로). 내부 점에 돌리면 중심이 중심축 근처에
    앉아서 훨씬 얇게 감싼다.
    """
    try:
        import trimesh

        mesh = trimesh.load(str(path), force="mesh")
        mesh.apply_scale(0.001)  # mm -> m
        pitch = max(float(mesh.extents.max()) / 45.0, 0.004)
        vox = mesh.voxelized(pitch=pitch).fill()
        pts = np.asarray(vox.points, dtype=np.float64)
        if len(pts) >= 20:
            return pts
    except Exception as e:  # noqa: BLE001
        print(f"   (내부 샘플링 실패 -> 표면 사용: {type(e).__name__}: {e})")
    return surface


def assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """각 점을 가장 가까운 중심에 배정. 메모리 절약을 위해 청크 단위로."""
    out = np.empty(len(X), dtype=np.int32)
    for i in range(0, len(X), 4096):
        chunk = X[i : i + 4096]
        d = ((chunk[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        out[i : i + 4096] = d.argmin(1)
    return out


def kmeans(X: np.ndarray, k: int, iters: int = 40, seed: int = 0):
    """k-means++ 초기화 + Lloyd 반복."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1).min(1)
        s = d2.sum()
        C.append(X[rng.choice(len(X), p=d2 / s)] if s > 0 else X[rng.integers(len(X))])
    C = np.array(C)
    for _ in range(iters):
        lab = assign(X, C)
        newC = np.array(
            [X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)]
        )
        if np.allclose(newC, C, atol=1e-6):
            break
        C = newC
    return C


def fit_link(surface: np.ndarray, inner: np.ndarray, target_r: float, seed: int = 0):
    """목표 반지름을 만족하는 최소 개수의 구를 찾는다.

    중심은 내부 점(inner)에서 뽑고, 반지름은 표면 정점(surface)까지의 거리로 정한다.
    -> 중심은 안쪽에 앉으면서 표면은 확실히 덮인다.
    """
    rng = np.random.default_rng(seed)
    Ifit = inner if len(inner) <= FIT_SAMPLE else inner[
        rng.choice(len(inner), FIT_SAMPLE, replace=False)
    ]

    best = None
    for k in range(1, MAX_SPHERES + 1):
        C = kmeans(Ifit, k, seed=seed)
        lab = assign(surface, C)  # 표면 전체를 중심들에 배정
        radii = np.zeros(k)
        for j in range(k):
            m = lab == j
            radii[j] = np.linalg.norm(surface[m] - C[j], axis=1).max() if m.any() else 0.0
        keep = radii > 1e-6
        best = (C[keep], radii[keep])
        if radii.max() <= target_r:
            break
    return best


def coverage_check(V: np.ndarray, C: np.ndarray, R: np.ndarray) -> float:
    """정점 중 몇 %가 어떤 구 안에 들어가는지. 1.0 이어야 정상."""
    inside = np.zeros(len(V), dtype=bool)
    for i in range(0, len(V), 4096):
        chunk = V[i : i + 4096]
        d = np.linalg.norm(chunk[:, None, :] - C[None, :, :], axis=-1)
        inside[i : i + 4096] = (d <= R[None, :] + 1e-9).any(1)
    return inside.mean()


def main():
    lines = ["# M0609 collision spheres (자동 생성 — gen_collision_spheres.py)",
             "collision_spheres:"]
    total = 0

    for link, meshes in LINK_MESHES.items():
        # 메시별로 따로 적합한 뒤 합친다.
        # 한 링크의 메시들을 뭉쳐서 k-means 를 돌리면 정점이 많은 부품(예: link_2 의
        # 긴 튜브)으로 구가 쏠려서, 정점이 적은 부품(어깨/팔꿈치 하우징)이 지나치게
        # 큰 구를 받게 된다.
        parts = [load_stl(MESH_DIR / m) for m in meshes]
        Cs, Rs = [], []
        for i, (mname, Vp) in enumerate(zip(meshes, parts)):
            inner = interior_points(MESH_DIR / mname, Vp)
            c, r = fit_link(Vp, inner, TARGET_R[link], seed=i)
            Cs.append(c)
            Rs.append(r)
        C, R = np.vstack(Cs), np.concatenate(Rs)

        V = np.vstack(parts)
        cov = coverage_check(V, C, R)
        total += len(C)

        # 각 구가 표면 밖으로 얼마나 튀어나가는지 (작을수록 얇게 잘 감싼 것)
        prot = np.array([r_ - np.linalg.norm(V - c_, axis=1).min()
                         for c_, r_ in zip(C, R)])

        print(
            f"{link:10s}  구 {len(C):2d}개  r {R.min():.3f}~{R.max():.3f}m  "
            f"돌출 평균 {prot.mean():.4f}m  커버리지 {cov*100:.2f}%"
        )
        if cov < 1.0:
            print(f"   !! {link} 커버리지 미달 — TARGET_R 를 낮추세요")

        lines.append(f"  {link}:")
        for c, r in zip(C, R):
            lines.append(
                f"    - center: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]\n"
                f"      radius: {r:.4f}"
            )

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n총 {total}개 구 -> {OUT}")


if __name__ == "__main__":
    main()
