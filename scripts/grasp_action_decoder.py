#!/usr/bin/env python3
"""grasp_v1 action(21D) → 관절 명령 디코더 (순수 numpy, Isaac/Fabrics 무의존).

env `grasp_right_env.py` `_pre_physics_step` + `_apply_action` 의 **손가락·lift 경로**를
충실 복제한다. 팔의 Fabrics IK(palm delta→arm)는 warp 의존이라 호출자(라이브 노드)가
담당하고, 여기서는 hdgp 무의존으로 검증 가능한 부분만 다룬다:

  1. palm delta 스케일  : action[:6] ∈[-1,1] → Δpalm (scale_palm_delta)
  2. 손가락 접촉-게이트 적응 폐쇄 : GraspFingerController (stateful)
     action[6:21] = (5 손가락 × 3 채널). 채널 0=_1 외전 / 1=_2 MCP / 2=_3·_4 공통.
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

# ★08.16 계약: action 21D = palm 6 + 손가락 15(5×3 채널)
NUM_PALM_ACTION = 6
NUM_FINGER_CHANNELS = 3
NUM_FINGER_ACTION = NUM_FINGERTIPS * NUM_FINGER_CHANNELS      # 15
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION             # 21
PALM_SLICE = slice(0, NUM_PALM_ACTION)
FINGER_SLICE = slice(NUM_PALM_ACTION, NUM_ACTIONS)

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
# 손가락 채널 순수 변환 (env `_pre_physics_step` 1:1)
# ---------------------------------------------------------------------------
def couple_four_fingers(finger_action: np.ndarray) -> np.ndarray:
    """(5,3) → (5,3). 검지~소지(1:5)를 **채널별 평균**으로 치환, 엄지(0)는 독립.

    3지 국소최적을 action 표현 단계에서 차단한다("특정 손가락만 안 닫힘"을 표현 불가하게).
    채널별로 평균내므로 4지가 자세를 공유하되 외전/MCP/PIP 비율은 정책이 정한다.

    ★clamp **이전**에 평균한다 — sim 은 `_pre_physics_step` 에서 평균한 뒤 한참 뒤
      `cmd_ch = 0.5*(finger_action.clamp(-1,1)+1)` 로 clamp 한다. 순서를 바꾸면
      비대칭 입력에서 값이 달라진다(예 4지 [3,0,0,0]: 평균 0.75 vs clamp 선행 0.25).
    """
    fa = np.asarray(finger_action, dtype=np.float64)
    if fa.shape != (NUM_FINGERTIPS, NUM_FINGER_CHANNELS):
        raise ValueError(f"finger_action expected (5,3), got {fa.shape}")
    common4 = fa[1:5, :].mean(axis=0)
    return np.concatenate([fa[0:1, :], np.tile(common4, (4, 1))], axis=0)


def expand_channels_to_joints(cmd_ch: np.ndarray) -> np.ndarray:
    """(5,3) → (20,) finger-major. [_1,_2,_3,_4] ← [ch0, ch1, ch2, ch2].

    _3(PIP)·_4(DIP)는 한 채널을 공유한다 — sim 의 `torch.stack([c0,c1,c2,c2])` 와 동일.
    """
    c = np.asarray(cmd_ch, dtype=np.float64)
    if c.shape != (NUM_FINGERTIPS, NUM_FINGER_CHANNELS):
        raise ValueError(f"cmd_ch expected (5,3), got {c.shape}")
    return np.stack([c[:, 0], c[:, 1], c[:, 2], c[:, 2]], axis=1).reshape(-1)


def freeze_gate20(
    tip_contact: np.ndarray,
    distal_contact: np.ndarray | None = None,
    latched: bool = False,
    retighten_after_latch: bool = False,
) -> np.ndarray:
    """(20,) 관절별 동결 게이트. _1/_2 무게이트, _3/_4 = clip(distal|tip).

    distal_contact=None → **tip-only**(라이브 배포: distal/middle 은 critic 전용이라
    실기에서 감지 불가). 이 차이는 의도된 것이다.

    retighten_after_latch 는 sim cfg 양측 False — 래치 후 동결 유지가 현재 계약이다.
    True 로 바뀌면 배포도 같이 바꿔야 한다.
    """
    tip = np.asarray(tip_contact, dtype=np.float64).reshape(-1)
    if tip.shape != (NUM_FINGERTIPS,):
        raise ValueError(f"tip_contact expected (5,), got {tip.shape}")
    if distal_contact is None:
        distal = np.zeros(NUM_FINGERTIPS, dtype=np.float64)
    else:
        distal = np.asarray(distal_contact, dtype=np.float64).reshape(-1)
        if distal.shape != (NUM_FINGERTIPS,):
            raise ValueError(f"distal_contact expected (5,), got {distal.shape}")
    g_zero = np.zeros(NUM_FINGERTIPS, dtype=np.float64)
    g34 = np.clip(distal + tip, 0.0, 1.0)
    gate = np.stack([g_zero, g_zero, g34, g34], axis=1).reshape(-1)
    if retighten_after_latch and latched:
        gate = np.zeros_like(gate)          # 래치 후 동결 해제
    return gate


# ---------------------------------------------------------------------------
# 손가락 접촉-게이트 적응 폐쇄 (env `_pre_physics_step` 손 경로, stateful)
# ---------------------------------------------------------------------------
class GraspFingerController:
    """관절별 finger_close_buf 적분기 (env `_pre_physics_step` 손 경로 1:1).

    15D action → (5,3) 채널 → 4지 공통닫힘 → 절대 폐쇄도 → 20관절 전개 →
    **변화율 상한**으로 목표 추종 → lerp(open, full_grip) → 관절 한계 clamp.

    ★08.16 래칫 제거: 구현은 `delta = clip(cmd20 − close_buf, ±rate)` 다. 구 방식
      (`advance = rate × cmd20 ≥ 0`)은 단조 증가만 가능해 탐색 노이즈 평균(cmd≈0.5)
      만으로도 80스텝이면 완전 폐쇄에 도달하고 되돌릴 수 없었다 — 정책이 "얼마나
      닫을지"를 표현할 수 없었고 채널을 분리해도 전부 1.0 으로 포화했다.
      즉 PIP/DIP 채널 분리와 래칫 제거는 **한 묶음**이다.

    ★연산 순서 고정: clip(delta) → ×(1−gate) → += → clip(0,1).
      sim 이 rate 를 먼저 자르고 그 다음 게이트를 곱한다. 순서를 바꾸면 게이트 해제
      tick 에서 값이 튄다.

    ★동결은 유지: 접촉한 관절은 그 자리에 멈춰 컵 형상에 드리워진다 — 감쌈 생성
      메커니즘이자 다형상 적응의 근거다. 여기를 건드리면 3지 국소최적으로 회귀한다.
    """

    def __init__(
        self,
        hand_open: np.ndarray,        # (20,)  HAND_APPROACH_POSE
        hand_full_grip: np.ndarray,   # (20,)  HAND_FULL_GRIP_POSE
        close_speed: float = DEFAULT_FINGER_CLOSE_SPEED,
        lower_limits: np.ndarray | None = None,   # (20,)
        upper_limits: np.ndarray | None = None,   # (20,)
        couple_four: bool = True,                 # cfg.couple_four_fingers (양측 True)
        retighten_after_latch: bool = False,      # cfg (양측 False)
    ) -> None:
        self.hand_open = self._v(hand_open, "hand_open")
        self.hand_full_grip = self._v(hand_full_grip, "hand_full_grip")
        self.close_speed = float(close_speed)
        self.lower = None if lower_limits is None else self._v(lower_limits, "lower_limits")
        self.upper = None if upper_limits is None else self._v(upper_limits, "upper_limits")
        self.couple_four = bool(couple_four)
        self.retighten_after_latch = bool(retighten_after_latch)
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
        finger_action: np.ndarray,                 # (15,) 또는 (5,3) ∈ [-1, 1]
        tip_contact: np.ndarray,                   # (5,) binary
        distal_contact: np.ndarray | None = None,  # (5,) binary; None → tip-only 게이트
        latched: bool = False,                     # lift 래치 상태(retighten 용)
    ) -> np.ndarray:
        """1스텝 진행 → hand target (20,).

        distal_contact=None 이면 **tip-only** 게이트(라이브 배포: distal/middle 은
        critic 전용이라 감지 불가). distal 을 주면 sim `_pre_physics_step` 와 동일.
        """
        fa = np.asarray(finger_action, dtype=np.float64)
        if fa.shape == (NUM_FINGER_ACTION,):
            fa = fa.reshape(NUM_FINGERTIPS, NUM_FINGER_CHANNELS)
        elif fa.shape != (NUM_FINGERTIPS, NUM_FINGER_CHANNELS):
            raise ValueError(
                f"finger_action expected (15,) or (5,3), got {fa.shape}"
                " — 구 계약 5D 라면 21D action 으로 전환이 필요하다"
            )

        if self.couple_four:
            fa = couple_four_fingers(fa)                             # ★clamp 이전
        cmd_ch = 0.5 * (np.clip(fa, -1.0, 1.0) + 1.0)                # (5,3) 절대 폐쇄도
        cmd20 = expand_channels_to_joints(cmd_ch)                    # (20,)
        gate20 = freeze_gate20(
            tip_contact, distal_contact, latched, self.retighten_after_latch
        )

        # ★래칫 제거 — 절대 목표를 향해 변화율 상한으로 이동(감소 가능)
        delta = np.clip(cmd20 - self.close_buf, -self.close_speed, self.close_speed)
        self.close_buf = np.clip(self.close_buf + delta * (1.0 - gate20), 0.0, 1.0)

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
