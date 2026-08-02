"""grasp_action_decoder 순수 포팅 ↔ hdgp env 실제 함수 cross-parity.

hdgp `grasp_right_utils.py` 의 함수(torch, Isaac 무의존)를 직접 임포트해 내 numpy
포팅과 수치 일치를 검증한다. (env `_pre_physics_step` 의 손가락 적분기는 인라인이라
임포트 불가 — 그 부분은 test_grasp_action_decoder.py 의 단위 테스트 + 소스 대조로 담보.)

hdgp 소스가 없으면 스킵.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_HDGP_OPENARM = Path(__file__).resolve().parent.parent.parent / "hdgp" / "source" / "openarm"
_UTILS_PATH = _HDGP_OPENARM / "openarm" / "tesollo" / "right" / "grasp_v1" / "grasp_right_utils.py"

pytestmark = pytest.mark.skipif(
    not _UTILS_PATH.exists(), reason="hdgp grasp_v1 소스 없음"
)

if _UTILS_PATH.exists():
    sys.path.insert(0, str(_HDGP_OPENARM))
    import torch
    from openarm.tesollo.right.grasp_v1 import grasp_right_utils as U

from grasp_action_decoder import (
    DEFAULT_GRASP_READY_HOLD_STEPS,
    DEFAULT_LIFT_MIN_GRIP_FINGERS,
    LiftLatch,
    joint7_lift_wait_target,
    scale_palm_delta,
)


def test_scale_matches_env():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-1, 1, size=6)
        lo = rng.uniform(-0.3, 0, size=6)
        hi = rng.uniform(0, 0.3, size=6)
        mine = scale_palm_delta(a, lo, hi)
        env = U.scale(
            torch.tensor(a), torch.tensor(lo), torch.tensor(hi)
        ).numpy()
        assert np.allclose(mine, env, atol=1e-9)


def test_joint7_lift_wait_matches_env():
    rng = np.random.default_rng(1)
    for _ in range(20):
        arm = rng.uniform(-1.5, 1.5, size=7)
        mine = joint7_lift_wait_target(arm, joint7_delta=0.31, joint7_min=0.20, joint7_max=1.50)
        env = U.compute_joint7_lift_wait_target(
            torch.tensor(arm).unsqueeze(0),
            joint7_delta=0.31, joint7_min=0.20, joint7_max=1.50,
        )[0].numpy()
        assert np.allclose(mine, env, atol=1e-9)


def test_lift_latch_matches_env_sequence():
    # env compute_lift_readiness 를 단일-env 로 스텝 구동하며 LiftLatch 와 대조
    latch = LiftLatch(min_contacts=DEFAULT_LIFT_MIN_GRIP_FINGERS, hold_steps=DEFAULT_GRASP_READY_HOLD_STEPS)
    hold = torch.zeros(1)
    latched = torch.zeros(1, dtype=torch.bool)
    rng = np.random.default_rng(2)
    grip_seq = list(rng.integers(0, 6, size=40))
    for grip in grip_seq:
        hold, _ready, latched = U.compute_lift_readiness(
            num_contacts=torch.tensor([float(grip)]),
            is_grasp_phase=~latched,
            previous_hold_count=hold,
            previous_latched=latched,
            min_contacts=DEFAULT_LIFT_MIN_GRIP_FINGERS,
            hold_steps=DEFAULT_GRASP_READY_HOLD_STEPS,
        )
        mine = latch.update(int(grip))
        assert bool(latched.item()) == mine, f"grip={grip} env={latched.item()} mine={mine}"
        assert int(hold.item()) == latch.hold_count
