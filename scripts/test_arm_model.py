#!/usr/bin/env python3
"""팔 유효관성·PD 모델 테스트 (numpy only).

이 두 모듈은 "정책이 요구하는 속도 vs 실기가 낼 수 있는 속도" 비교의 근거다. 값이
틀리면 실험 결론이 통째로 틀리므로, **출처가 있는 수치인지**와 물리적 타당성을 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arm_inertia import effective_inertia, parse_urdf  # noqa: E402
from arm_pd_model import (  # noqa: E402
    CALIB_JSON,
    SIM_DAMPING,
    SIM_STIFFNESS,
    MockArm,
    bandwidth_gap,
    load_arm_pd,
    second_order_characteristics,
)
from robot_profile import WS_ROOT  # noqa: E402

URDF = WS_ROOT / "urdf/generated/rl/openarm_tesollo_bi_s_rl.urdf"
ARM = [f"r_aj_{i}" for i in range(1, 8)]

needs_urdf = pytest.mark.skipif(not URDF.exists(), reason="자산 URDF 없음")
needs_calib = pytest.mark.skipif(not CALIB_JSON.exists(), reason="팔 캘리브 없음")


# --------------------------------------------------------------------------
# 유효 관성 — 자산에서 계산 (지어내지 않는다)
# --------------------------------------------------------------------------

@needs_urdf
def test_inertia_positive_and_reasonable():
    I = effective_inertia(URDF, ARM)
    assert I.shape == (7,)
    assert np.all(I > 0), "관성은 양수여야 한다"
    # 어깨(근위)가 손목(원위)보다 훨씬 커야 물리적으로 말이 된다
    assert I[0] > I[6] * 5


@needs_urdf
def test_inertia_depends_on_pose():
    """유효 관성은 자세 함수다 — 상수로 취급하면 안 된다."""
    a = effective_inertia(URDF, ARM, {n: 0.0 for n in ARM})
    b = effective_inertia(URDF, ARM, {"r_aj_4": 1.2})
    assert not np.allclose(a, b)


@needs_urdf
def test_unknown_joint_raises():
    with pytest.raises(KeyError):
        effective_inertia(URDF, ["없는관절"])


@needs_urdf
def test_urdf_has_inertial_data():
    model = parse_urdf(URDF)
    n = sum(1 for v in model["links"].values() if v is not None)
    assert n > 30, f"inertial 보유 링크가 너무 적다: {n}"


# --------------------------------------------------------------------------
# PD 게인 — 실측 캘리브에서만
# --------------------------------------------------------------------------

@needs_calib
def test_load_arm_pd_groups():
    kp, kd, fc, src = load_arm_pd()
    assert kp.shape == kd.shape == fc.shape == (7,)
    assert np.all(kp > 0) and np.all(kd > 0)
    # 근위 4관절이 손목 3관절보다 강해야 한다(실측 경향)
    assert kp[:4].min() > kp[4:].max()
    assert src


def test_missing_calibration_raises(tmp_path):
    """캘리브가 없으면 **예외** — 추정값으로 조용히 돌리면 실험이 무의미해진다."""
    with pytest.raises(FileNotFoundError, match="캘리브"):
        load_arm_pd(tmp_path / "없음.json")


def test_missing_group_raises(tmp_path):
    import json
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"groups": {"arm_proximal": {"stiffness": 1, "damping": 1}}}))
    with pytest.raises(KeyError, match="arm_elbow"):
        load_arm_pd(p)


# --------------------------------------------------------------------------
# 2차 특성 — 대역폭 격차
# --------------------------------------------------------------------------

def test_second_order_math():
    wn, z = second_order_characteristics([100.0], [20.0], [1.0])
    assert wn[0] == pytest.approx(10.0)
    assert z[0] == pytest.approx(1.0)


def test_zero_inertia_raises():
    with pytest.raises(ValueError):
        second_order_characteristics([1.0], [1.0], [0.0])


@needs_urdf
@needs_calib
def test_sim_arm_tracks_faster_than_real():
    """★sim 팔이 실기보다 빠르다 = 정책이 droop·지연을 겪어본 적 없다.

    이 관계가 뒤집히면(실기가 더 빠름) 게인 격차 가설 자체가 무효이므로 고정한다.
    """
    I = effective_inertia(URDF, ARM)
    g = bandwidth_gap(I)
    assert np.all(g["ratio"] > 1.0), f"대역폭비 = {g['ratio']}"
    assert g["ratio"].mean() > 2.0
    # sim 은 과감쇠(진동 없음), 실기 근위는 부족감쇠(오버슛)
    assert np.all(g["zeta_sim"] > 1.0)
    assert g["zeta_real"].min() < 1.0


def test_sim_gains_match_env_cfg():
    """sim 게인 상수가 env_cfg 의 ImplicitActuatorCfg 값과 같아야 한다."""
    assert (SIM_STIFFNESS, SIM_DAMPING) == (400.0, 80.0)


# --------------------------------------------------------------------------
# MockArm
# --------------------------------------------------------------------------

def test_rate_model_respects_max_vel():
    arm = MockArm(np.zeros(7), "rate", max_vel=0.5, dt=1 / 60)
    arm.step(np.full(7, 10.0))
    assert np.all(np.abs(arm.q) <= 0.5 / 60 + 1e-12)


def test_rate_model_reaches_target_eventually():
    arm = MockArm(np.zeros(7), "rate", max_vel=99.0, dt=1 / 60)
    arm.step(np.full(7, 0.3))
    assert np.allclose(arm.q, 0.3)


@needs_calib
@needs_urdf
def test_pd_model_converges_to_setpoint():
    """정상상태 오차는 **쿨롱 마찰 데드밴드**(Fc/kp) 수준이어야 한다 — 물리적으로 옳다."""
    kp, kd, fc, _ = load_arm_pd()
    I = effective_inertia(URDF, ARM)
    arm = MockArm(np.zeros(7), "pd", max_vel=99.0, dt=1 / 60, kp=kp, kd=kd, fc=fc, inertia=I)
    for _ in range(600):
        arm.step(np.full(7, 0.2))
    assert np.all(np.isfinite(arm.q)), "발산"
    err = np.abs(arm.q - 0.2)
    deadband = fc / kp
    assert np.all(err < np.maximum(deadband * 4.0, 0.02)), f"오차 {err} vs 데드밴드 {deadband}"


@needs_calib
@needs_urdf
def test_pd_model_stable_at_control_rate():
    """★저관성 관절(kd/I·dt ≫ 2)에서 발산하지 않아야 한다 — 감쇠 암시적 처리의 근거."""
    kp, kd, fc, _ = load_arm_pd()
    I = effective_inertia(URDF, ARM)
    assert np.any(kd / I * (1 / 60) > 2.0), "이 자산에선 명시적 적분이 불안정한 관절이 없다"
    arm = MockArm(np.zeros(7), "pd", max_vel=99.0, dt=1 / 60, kp=kp, kd=kd, fc=fc, inertia=I)
    for _ in range(1200):
        arm.step(np.full(7, 0.5))
    assert np.all(np.isfinite(arm.q))
    assert np.all(np.abs(arm.q) < 5.0), f"발산 조짐: {arm.q}"


@needs_calib
@needs_urdf
def test_pd_substeps_reduce_residual():
    """서브스텝을 늘리면 정상상태 오차가 마찰 데드밴드로 수렴한다."""
    kp, kd, fc, _ = load_arm_pd()
    I = effective_inertia(URDF, ARM)
    errs = []
    for ss in (1, 32):
        arm = MockArm(np.zeros(7), "pd", max_vel=99.0, dt=1 / 60,
                      kp=kp, kd=kd, fc=fc, inertia=I, substeps=ss)
        for _ in range(600):
            arm.step(np.full(7, 0.2))
        errs.append(np.abs(arm.q - 0.2).max())
    assert errs[1] < errs[0]


def test_substeps_validated():
    with pytest.raises(ValueError, match="substeps"):
        MockArm(np.zeros(7), "rate", max_vel=1.0, dt=1 / 60, substeps=0)


def test_pd_model_requires_all_params():
    with pytest.raises(ValueError, match="pd 모델"):
        MockArm(np.zeros(7), "pd", max_vel=1.0, dt=1 / 60)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="rate|pd"):
        MockArm(np.zeros(7), "spring", max_vel=1.0, dt=1 / 60)
