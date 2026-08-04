#!/usr/bin/env python3
"""
장애물 추적 — 무작위로 움직이는 장애물 + 칼만필터.

데모에서 장애물 위치를 해석적으로 미분해 속도를 얻던 건 반칙이다.
실제로는 YOLO 가 노이즈 섞인 위치만 30Hz 로 주고, 속도는 추정해야 한다.
여기서 그 구조를 그대로 흉내낸다.

    RandomMover   무작위로 부드럽게 움직이는 장애물 (진자운동이 아님)
    KalmanCV      등속모델 칼만필터. 노이즈 낀 위치 관측 -> 위치 + 속도
"""
import numpy as np


class RandomMover:
    """느리게 바뀌는 목표점을 향해 이동. 진자처럼 예측 가능하지 않다."""

    def __init__(self, center, box, speed=0.10, change_every=3.0, seed=0):
        self.c = np.asarray(center, float)
        self.box = np.asarray(box, float)      # 중심에서의 반경 [x,y,z]
        self.speed = speed
        self.change_every = change_every
        self.rng = np.random.default_rng(seed)
        self.p = self.c.copy()
        self.target = self._pick()
        self.t_next = change_every

    def _pick(self):
        return self.c + self.rng.uniform(-self.box, self.box)

    def step(self, t, dt):
        if t > self.t_next:
            self.target = self._pick()
            self.t_next = t + self.change_every
        d = self.target - self.p
        n = np.linalg.norm(d)
        if n > 1e-6:
            self.p = self.p + d / n * min(self.speed * dt, n)
        return self.p.copy()

    def observe(self, noise=0.012):
        """YOLO 관측을 흉내낸다 — 노이즈 낀 위치만 준다. 속도는 안 준다."""
        return self.p + self.rng.normal(0.0, noise, 3)


class KalmanCV:
    """등속(constant velocity) 모델 칼만필터.  상태 = [x,y,z,vx,vy,vz]

    관측은 위치뿐인데 속도가 추정돼 나온다. 그 속도가 있어야 미래 위치를
    예측할 수 있고, 그래야 로봇이 '반응' 이 아니라 '예측' 으로 비켜간다.
    """

    def __init__(self, p0, q_acc=0.15, r_pos=0.015):
        self.x = np.r_[np.asarray(p0, float), np.zeros(3)]
        # 초기 속도 공분산이 크면 predict_ahead 의 불확실성이 폭발한다
        # (0.8초 예측에 61cm). 처음 몇 프레임만 지나면 수렴하므로 작게 시작.
        self.P = np.diag([0.02] * 3 + [0.05] * 3)
        # q_acc 가 크면 필터가 관측 노이즈를 그대로 따라가서 마커가 떨린다.
        # 장애물이 부드럽게 움직이는 상황이면 작게 잡는 게 맞다 (0.6 -> 0.15).
        self.q_acc = q_acc
        self.R = np.eye(3) * r_pos ** 2

    def predict(self, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # 등가속 모델의 이산 프로세스 노이즈
        q = self.q_acc ** 2
        G = np.r_[np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt]
        self.Q = (G @ G.T) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        y = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    @property
    def pos(self):
        return self.x[:3].copy()

    @property
    def vel(self):
        return self.x[3:].copy()

    def predict_ahead(self, t):
        """t 초 뒤 예상 위치 + 그때의 위치 불확실성(표준편차)."""
        p = self.x[:3] + self.x[3:] * t
        # 속도 공분산이 시간에 비례해 위치 불확실성으로 번진다
        var = np.diag(self.P)[:3] + np.diag(self.P)[3:] * t ** 2
        return p, np.sqrt(var)
