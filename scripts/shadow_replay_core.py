#!/usr/bin/env python3
"""그림자 재생기의 순수 코어 — 무엇을 언제 보낼지 (numpy only, ROS 무의존).

sim 이 기록한 관절 목표를 실기로 다시 흘려보내는 계획을 세운다. ROS 노드는 이 계획을
발행하기만 한다. 그래야 "언제 무엇을 보내는가"를 로봇 없이 검증할 수 있다.

세 가지가 여기 있는 이유는 전부 안전이다:

  · **진입 램프** — 기록의 첫 프레임은 sim 의 리셋 자세다. 실기가 어디 있든 거기로 한
    스텝에 보내면 그건 이동이 아니라 도약이다. 실측 자세에서 `PARK_SPEED_RAD_PER_SEC`
    (robotctl 이 쓰는 0.1 rad/s, "이동 중에 비상정지까지 걸어갈 수 있는 속도") 로 들어간다.
  · **결손 프레임 거부** — 비유한 값이 있으면 보간하지 않고 거부한다. 없는 명령을 지어내면
    로봇은 그걸 그대로 실행한다.
  · **rate_scale ≤ 1** — 느리게만 재생한다. sim 보다 빠르게 낼 이유가 없고, 그러면
    요구 속도가 실기 한계 밖으로 나간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: robotctl `PARK_SPEED_RAD_PER_SEC` 과 같은 값·같은 이유(사람이 반응할 수 있는 속도).
PARK_SPEED_RAD_PER_SEC = 0.1


def approach_ramp(start: np.ndarray, first: np.ndarray, *, speed: float,
                  dt: float) -> np.ndarray:
    """실측 자세 → 기록 첫 프레임까지의 프레임열(양 끝 포함)."""
    start = np.asarray(start, dtype=float).reshape(-1)
    first = np.asarray(first, dtype=float).reshape(-1)
    if start.shape != first.shape:
        raise ValueError(f"start{start.shape} / first{first.shape} 길이 불일치")
    if speed <= 0.0 or dt <= 0.0:
        raise ValueError("speed 와 dt 는 양수여야 한다")

    span = float(np.max(np.abs(first - start)))
    steps = int(np.ceil(span / (speed * dt)))
    if steps <= 0:
        return start.reshape(1, -1).copy()
    weights = np.linspace(0.0, 1.0, steps + 1).reshape(-1, 1)
    return start + weights * (first - start)


def frame_schedule(*, n_frames: int, step_dt: float, rate_scale: float) -> np.ndarray:
    """각 프레임의 발행 시각[s]. 경로는 그대로 두고 시간만 늘린다."""
    if not 0.0 < rate_scale <= 1.0:
        raise ValueError(f"rate_scale 은 (0, 1] 이어야 한다 — 받은 값 {rate_scale}")
    if step_dt <= 0.0:
        raise ValueError("step_dt 는 양수여야 한다")
    return np.arange(n_frames, dtype=float) * (step_dt / rate_scale)


@dataclass
class ReplayPlan:
    """검증된 재생 계획. 만들어지면 그대로 발행해도 되는 상태다."""

    arm_target: np.ndarray
    grip_target: np.ndarray
    step_dt: float
    rate_scale: float
    joint_names: list[str]
    gripper_name: str
    schedule: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.arm_target = np.asarray(self.arm_target, dtype=float)
        self.grip_target = np.asarray(self.grip_target, dtype=float).reshape(-1)
        if self.arm_target.ndim != 2:
            raise ValueError(f"arm_target 은 (프레임, 관절) 이어야 한다 — {self.arm_target.shape}")
        if self.arm_target.shape[1] != len(self.joint_names):
            raise ValueError(
                f"관절 수 불일치: 목표 {self.arm_target.shape[1]} vs 이름 {len(self.joint_names)}"
            )
        if self.arm_target.shape[0] != self.grip_target.shape[0]:
            raise ValueError(
                f"프레임 수 불일치: 팔 {self.arm_target.shape[0]} vs 그리퍼 "
                f"{self.grip_target.shape[0]}"
            )
        for name, values in (("arm_target", self.arm_target), ("grip_target", self.grip_target)):
            if not np.all(np.isfinite(values)):
                bad = int(np.argwhere(~np.isfinite(values))[0][0])
                raise ValueError(
                    f"{name} 프레임 {bad} 이 유한하지 않다 — 보간하지 않고 거부한다"
                )
        self.schedule = frame_schedule(
            n_frames=self.arm_target.shape[0], step_dt=self.step_dt,
            rate_scale=self.rate_scale,
        )

    @property
    def n_frames(self) -> int:
        return int(self.arm_target.shape[0])

    @property
    def publish_dt(self) -> float:
        return self.step_dt / self.rate_scale

    @property
    def peak_joint_speed(self) -> float:
        """이 재생이 실기에 요구하는 최대 관절 속도[rad/s].

        실기 능력(프로필 velocity, 펌웨어 게인)과 대볼 유일한 수치다. 재생 전에 안다.
        """
        if self.n_frames < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(self.arm_target, axis=0))) / self.publish_dt)

    def ramp_from(self, measured: np.ndarray) -> np.ndarray:
        return approach_ramp(measured, self.arm_target[0],
                             speed=PARK_SPEED_RAD_PER_SEC, dt=self.publish_dt)
