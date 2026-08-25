#!/usr/bin/env python3
"""both/pour_sensor(양팔 15D) 배포 어댑터 — numpy 순수, Isaac/ROS 불필요.

배경: 기존 pour 배포 스택(`pour_obs_builder` / `pour_action_decoder` /
`pour_obs_geometry`)은 `tesollo/right/pour_v1`(한팔·12D)용으로 작성됐다. RA-L 논문의
env는 `tesollo/both/pour_sensor`(양팔·15D)다.

**두 env를 대조한 결과 오른팔 경로는 완전히 동일하다** (2026-08-16 확인):
  - actor obs 55D 레이아웃 동일 — pour_v1 빌더가 이미 `left_arm_joint_pos/vel`(9+9)
    슬롯과 `target_cup_pos/quat` 인자를 갖고 있다(한팔에선 0/정적값으로 채웠을 뿐).
  - 컵 지오메트리 상수 동일: pour_point/opening z=0.100, outer_radius=0.045,
    dyn_lo/hi=0.15/0.30.
  - 디코더 상수·모드 플래그 동일: palm_delta 0.03/15°, ema 0.7, tilt gate 0.06/0.25,
    beta 0.854/3.0/0.06, z_margin 0.03, inner_radius 0.041, corridor(0.015,-0.02,0.12,20),
    ready latch 0.60, b_trajectory·palm pivot·z_lock·orient_release.

따라서 이 모듈은 **차이분만** 담는다:
  1. action[12:15] = 왼팔(receiver) TCP 증분 → `LeftTcpController`
  2. receiver 컵 pose = 왼손 FK ∘ follow offset → `receiver_cup_pose`
  3. 15D → 12D 분해 후 기존 디코더 재사용 → `split_bimanual_action`

M0(frozen receiver) 축소 배포에서는 `LeftTcpController(mode="frozen")`을 쓴다.
논문이 M4 95.1% ≈ M0 94.0%(C0)를 보고했으므로 정당한 축소이며, 왼팔은 rest 자세로
리시버를 든 채 고정된다.

상수 drift는 `test_pour_sensor_bimanual.py`가 hdgp cfg와 대조해 감시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pour_obs_builder import compose_pose
from pour_obs_geometry import quat_apply  # noqa: F401  (재수출: 노드에서 사용)

# --- both/pour_sensor env_cfg 값 (drift-guard 감시) ---------------------------
ACTION_DIM = 15
NUM_RIGHT_ACTION = 12          # [0:6] palm, [6] α, [7:12] hand — pour_v1과 동일
LEFT_TCP_ACTION_DELTA_M = 0.01          # policy step당 TCP 최대 이동량 [m]
LEFT_TCP_WORKSPACE_RANGE = (0.08, 0.08, 0.08)   # rest 기준 half-extent [m]
LEFT_TCP_Z_DOWN_M = 0.0        # ★ z 하강 금지 — receiver 컵 테이블 관통 방지(s2r 안전)
LEFT_CUP_FOLLOW_LOCAL_Z = 0.05  # 왼손 body frame Z 방향 컵 offset [m]

RECEIVER_MODES = ("learned", "frozen")


def _as3(v, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape[0] != 3:
        raise ValueError(f"{name} expected 3 values, got {a.shape[0]}")
    return a


def split_bimanual_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """15D action → (오른팔 12D, 왼팔 TCP 3D).

    오른팔 12D는 기존 `pour_action_decoder.decode()`에 그대로 넘길 수 있다.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM}D action, got {a.shape[0]}")
    return a[:NUM_RIGHT_ACTION].copy(), np.clip(a[NUM_RIGHT_ACTION:], -1.0, 1.0)


@dataclass
class LeftTcpController:
    """왼팔 receiver TCP 목표를 rest 기준 누적으로 관리 (env `_pre_physics_step` 포팅).

    env 식: `target = clamp(target + action*delta, rest-min_range, rest+max_range)`
    이며 z 하한만 `left_tcp_z_down_m`(기본 0 = rest 아래 금지)로 따로 캡한다.

    mode="frozen"이면 action을 무시하고 rest를 유지한다 (M0 축소 배포).
    """

    rest_pos_b: np.ndarray
    mode: str = "learned"
    action_scale: float = 1.0          # EXP-2 receiver_action_scale
    delay_steps: int = 0               # EXP-2 receiver_action_delay_steps
    hold_steps: int = 0                # episode_hold_steps — 초기 N스텝 rest 유지
    _target: np.ndarray = field(init=False)
    _delay_buf: list = field(init=False, default_factory=list)
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode not in RECEIVER_MODES:
            raise ValueError(f"mode must be one of {RECEIVER_MODES}, got {self.mode!r}")
        self.rest_pos_b = _as3(self.rest_pos_b, "rest_pos_b")
        wr = np.asarray(LEFT_TCP_WORKSPACE_RANGE, dtype=np.float64)
        self._max = self.rest_pos_b + wr
        wr_min = wr.copy()
        wr_min[2] = LEFT_TCP_Z_DOWN_M
        self._min = self.rest_pos_b - wr_min
        self._target = self.rest_pos_b.copy()
        self._delay_buf = [np.zeros(3) for _ in range(max(self.delay_steps, 0))]

    @property
    def target_pos_b(self) -> np.ndarray:
        return self._target.copy()

    def reset(self) -> None:
        self._target = self.rest_pos_b.copy()
        self._delay_buf = [np.zeros(3) for _ in range(max(self.delay_steps, 0))]
        self._step = 0

    def step(self, left_action: np.ndarray) -> np.ndarray:
        """action[12:15] → 새 TCP 목표(base 프레임). 매 정책 스텝 1회 호출."""
        a = _as3(left_action, "left_action")
        if self.mode == "frozen":
            self._step += 1
            self._target = self.rest_pos_b.copy()
            return self.target_pos_b

        if self.action_scale != 1.0:
            a = a * float(self.action_scale)
        if self.delay_steps > 0:
            self._delay_buf.append(a)
            a = self._delay_buf.pop(0)
        if self._step < self.hold_steps:
            self._step += 1
            self._target = self.rest_pos_b.copy()
            return self.target_pos_b

        self._step += 1
        self._target = np.clip(
            self._target + a * LEFT_TCP_ACTION_DELTA_M, self._min, self._max
        )
        return self.target_pos_b


def left_cup_follow_offset() -> tuple[np.ndarray, np.ndarray]:
    """왼손 body → receiver 컵의 고정 변환 (env `_left_cup_follow_{offset,quat}`).

    위치 [0, 0, local_z], 회전 R_y(+90°). 컵은 왼손에 kinematic-follow 하므로
    실물에서도 파지가 유지되는 한 이 offset이 성립한다(캘리브 1회로 대체 가능).
    """
    pos = np.array([0.0, 0.0, LEFT_CUP_FOLLOW_LOCAL_Z], dtype=np.float64)
    half = math.pi / 4.0  # R_y(+90°) → quat(w=cos45°, y=sin45°)
    quat = np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)
    return pos, quat


def receiver_cup_pose(
    left_hand_pos_b: np.ndarray,
    left_hand_quat_b: np.ndarray,
    follow_pos: np.ndarray | None = None,
    follow_quat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """왼손 FK pose → receiver 컵 pose (base 프레임).

    pour_v1 배포에서는 타깃 컵이 정적이었으나 pour_sensor에서는 왼팔이 들고 있다.
    비전 없이 왼팔 엔코더 FK만으로 얻는다 — actor obs 55D가 전부 proprio/FK인 이유.
    """
    if follow_pos is None or follow_quat is None:
        fp, fq = left_cup_follow_offset()
        follow_pos = fp if follow_pos is None else follow_pos
        follow_quat = fq if follow_quat is None else follow_quat
    return compose_pose(left_hand_pos_b, left_hand_quat_b, follow_pos, follow_quat)
