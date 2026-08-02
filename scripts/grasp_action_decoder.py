#!/usr/bin/env python3
"""grasp-v1 action(11D) → 관절 명령 디코더 (순수 numpy, Isaac/Fabrics 무의존).

env `grasp_right_env.py` `_pre_physics_step` + `_apply_action` 의 **손가락·lift 경로**를
충실 복제한다. 팔의 Fabrics IK(palm delta→arm)는 warp 의존이라 호출자(라이브 노드)가
담당하고, 여기서는 hdgp 무의존으로 검증 가능한 부분만 다룬다:

  1. palm delta 스케일  : action[:6] ∈[-1,1] → Δpalm (scale_palm_delta)
  2. 손가락 접촉-게이트 적응 폐쇄 : GraspFingerController (stateful)
  3. lift 진입 접촉 래치 : LiftLatch (compute_lift_readiness 단일-env 포팅)
  4. joint7-only lift-wait 목표 : joint7_lift_wait_target
  5. lift 팔 선형보간   : lift_arm_interp

상수(finger_close_speed, joint7 delta, hold_steps 등)와 손 포즈(APPROACH/FULL_GRIP),
관절 한계는 호출자가 grasp_v1 preset/cfg 에서 주입한다 — 이 모듈에 중복 정의하지 않아
소스와의 drift 를 막는다(테스트는 합성값 사용).
"""

from __future__ import annotations

import numpy as np

NUM_ARM_DOF = 7
NUM_HAND_DOF = 20
NUM_FINGERTIPS = 5
JOINTS_PER_FINGER = 4

# grasp_v1 cfg 기본값 (호출자가 override 가능; 여기 정의는 참조 편의용)
DEFAULT_FINGER_CLOSE_SPEED = 0.05
DEFAULT_LIFT_WAIT_JOINT7_DELTA = 0.31
DEFAULT_WARM_J7_MIN = 0.20
DEFAULT_WARM_J7_MAX = 1.50
DEFAULT_LIFT_MIN_GRIP_FINGERS = 3
DEFAULT_GRASP_READY_HOLD_STEPS = 8


# ---------------------------------------------------------------------------
# palm delta 스케일 (env: scale(palm_action, delta_mins, delta_maxs))
# ---------------------------------------------------------------------------
def scale_palm_delta(
    palm_action: np.ndarray,   # (6,) ∈ [-1, 1]
    delta_mins: np.ndarray,    # (6,)
    delta_maxs: np.ndarray,    # (6,)
) -> np.ndarray:
    """[-1,1] palm action → [delta_mins, delta_maxs] Δpalm. env `scale` 동일."""
    a = np.asarray(palm_action, dtype=np.float64).reshape(-1)
    lo = np.asarray(delta_mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(delta_maxs, dtype=np.float64).reshape(-1)
    if not (a.shape == lo.shape == hi.shape == (6,)):
        raise ValueError("palm_action/delta_mins/delta_maxs must all be shape (6,)")
    return 0.5 * (a + 1.0) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# joint7-only lift-wait target (env: compute_joint7_lift_wait_target)
# ---------------------------------------------------------------------------
def joint7_lift_wait_target(
    actual_arm: np.ndarray,   # (7,)
    joint7_delta: float = DEFAULT_LIFT_WAIT_JOINT7_DELTA,
    joint7_min: float = DEFAULT_WARM_J7_MIN,
    joint7_max: float = DEFAULT_WARM_J7_MAX,
) -> np.ndarray:
    """grasp 팔 자세 유지 + joint7(index 6)만 lift-wait 로 이동, clamp."""
    arm = np.asarray(actual_arm, dtype=np.float64).reshape(-1)
    if arm.shape != (NUM_ARM_DOF,):
        raise ValueError(f"actual_arm expected (7,), got {arm.shape}")
    target = arm.copy()
    target[6] = float(np.clip(target[6] + float(joint7_delta), joint7_min, joint7_max))
    return target


def lift_arm_interp(
    lift_arm_start: np.ndarray,   # (7,)  latch 시점 실제 팔
    prelift_target: np.ndarray,   # (7,)  joint7 lift-wait 목표
    progress: float,              # [0,1]
) -> np.ndarray:
    """lift-wait 팔 선형보간 (env `_apply_action` arm_target_lift)."""
    start = np.asarray(lift_arm_start, dtype=np.float64).reshape(-1)
    tgt = np.asarray(prelift_target, dtype=np.float64).reshape(-1)
    p = float(np.clip(progress, 0.0, 1.0))
    return start * (1.0 - p) + tgt * p


# ---------------------------------------------------------------------------
# lift 진입 접촉 래치 (env: compute_lift_readiness, 단일-env·envelope 게이트 미사용)
# ---------------------------------------------------------------------------
class LiftLatch:
    """≥min_contacts 손가락이 hold_steps 연속 접촉 → lift 래치(유지). 접촉=tip|mid|distal."""

    def __init__(
        self,
        min_contacts: int = DEFAULT_LIFT_MIN_GRIP_FINGERS,
        hold_steps: int = DEFAULT_GRASP_READY_HOLD_STEPS,
    ) -> None:
        self.min_contacts = int(min_contacts)
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.hold_count = 0
        self.latched = False

    def update(self, num_grip_fingers: int) -> bool:
        """num_grip_fingers = (tip|mid|distal) 접촉 손가락 수. 래치 상태 반환."""
        if not self.latched:
            if int(num_grip_fingers) >= self.min_contacts:
                self.hold_count += 1
            else:
                self.hold_count = 0
            if self.hold_count >= self.hold_steps:
                self.latched = True
        return self.latched


# ---------------------------------------------------------------------------
# 손가락 접촉-게이트 적응 폐쇄 (env `_pre_physics_step` 손 경로, stateful)
# ---------------------------------------------------------------------------
class GraspFingerController:
    """관절별 finger_close_buf 적분기.

    cmd = 0.5(finger_action+1) 폐쇄 속도[0,1], 손가락→4관절 확장.
    게이트: _1/_2 무게이트(풀클로즈), _3(PIP)/_4(DIP)는 (distal|tip) 접촉 시 동결.
    hand_target = lerp(open, full_grip, close_buf), 관절 한계 clamp.
    """

    def __init__(
        self,
        hand_open: np.ndarray,        # (20,)  HAND_APPROACH_POSE
        hand_full_grip: np.ndarray,   # (20,)  HAND_FULL_GRIP_POSE
        close_speed: float = DEFAULT_FINGER_CLOSE_SPEED,
        lower_limits: np.ndarray | None = None,   # (20,)
        upper_limits: np.ndarray | None = None,   # (20,)
    ) -> None:
        self.hand_open = self._v(hand_open, "hand_open")
        self.hand_full_grip = self._v(hand_full_grip, "hand_full_grip")
        self.close_speed = float(close_speed)
        self.lower = None if lower_limits is None else self._v(lower_limits, "lower_limits")
        self.upper = None if upper_limits is None else self._v(upper_limits, "upper_limits")
        self.reset()

    @staticmethod
    def _v(arr: np.ndarray, name: str) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64).reshape(-1)
        if out.shape != (NUM_HAND_DOF,):
            raise ValueError(f"{name} expected (20,), got {out.shape}")
        return out

    def reset(self) -> None:
        self.close_buf = np.zeros(NUM_HAND_DOF, dtype=np.float64)

    def step(
        self,
        finger_action: np.ndarray,          # (5,) ∈ [-1, 1]
        tip_contact: np.ndarray,            # (5,) binary
        distal_contact: np.ndarray | None = None,  # (5,) binary; None → tip-only 게이트
    ) -> np.ndarray:
        """1스텝 진행 → hand target (20,).

        _3(PIP)/_4(DIP) 동결 게이트 = (distal | tip). distal_contact=None 이면 **tip-only**
        (라이브 배포: distal/middle 은 critic 전용·감지 불가라 제어에 쓰지 않음). distal 을
        주면 env `_pre_physics_step` 와 동일(sim parity 검증용).
        """
        cmd = 0.5 * (np.clip(self._c(finger_action, "finger_action"), -1.0, 1.0) + 1.0)  # (5,)
        tip = self._c(tip_contact, "tip_contact")
        distal = np.zeros(NUM_FINGERTIPS) if distal_contact is None else self._c(distal_contact, "distal_contact")

        g_zero = np.zeros(NUM_FINGERTIPS, dtype=np.float64)          # _1, _2 무게이트
        g34 = np.clip(distal + tip, 0.0, 1.0)                        # _3, _4 동결 게이트
        gate = np.stack([g_zero, g_zero, g34, g34], axis=1).reshape(-1)  # (20,) finger-major
        cmd20 = np.repeat(cmd, JOINTS_PER_FINGER)                    # (20,) finger-major

        advance = self.close_speed * cmd20 * (1.0 - gate)
        self.close_buf = np.clip(self.close_buf + advance, 0.0, 1.0)

        hand = self.hand_open + self.close_buf * (self.hand_full_grip - self.hand_open)
        if self.lower is not None and self.upper is not None:
            hand = np.clip(hand, self.lower, self.upper)
        return hand

    @staticmethod
    def _c(arr: np.ndarray, name: str) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64).reshape(-1)
        if out.shape != (NUM_FINGERTIPS,):
            raise ValueError(f"{name} expected (5,), got {out.shape}")
        return out


def num_grip_fingers(
    tip_contact: np.ndarray,
    middle_contact: np.ndarray,
    distal_contact: np.ndarray,
) -> int:
    """(tip|mid|distal) 접촉 손가락 수 (lift latch 입력). env num_grip_fingers 포팅."""
    tip = np.asarray(tip_contact, dtype=bool).reshape(-1)
    mid = np.asarray(middle_contact, dtype=bool).reshape(-1)
    dist = np.asarray(distal_contact, dtype=bool).reshape(-1)
    if not (tip.shape == mid.shape == dist.shape == (NUM_FINGERTIPS,)):
        raise ValueError("tip/middle/distal contact must all be shape (5,)")
    return int((tip | mid | dist).sum())
