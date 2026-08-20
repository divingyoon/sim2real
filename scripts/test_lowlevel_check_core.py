#!/usr/bin/env python3
"""lowlevel_check_core 테스트 — 가이드 TEST1/TEST2 프로토콜의 순수 로직."""

from __future__ import annotations

import numpy as np
import pytest

from lowlevel_check_core import (
    JointVerdict,
    StepSpec,
    build_step_plan,
    clamp_command,
    evaluate_hold,
    evaluate_step,
    summarize_sign_table,
)

JOINTS = ["j1", "j2", "j3"]
LIMITS = {
    "j1": {"lower": -1.0, "upper": 1.0},
    "j2": {"lower": -1.0, "upper": 1.0},
    "j3": {"lower": -0.05, "upper": 0.05},   # 좁은 한계 — clamp 검사용
}


# --------------------------------------------------------------- build_step_plan
def test_plan_covers_every_joint_and_both_directions():
    plan = build_step_plan(JOINTS, amplitudes=(0.02, 0.05))
    steps = [s for s in plan if s.phase == "step"]
    for j in JOINTS:
        amps = sorted(s.amplitude for s in steps if s.joint == j)
        assert amps == [-0.05, -0.02, 0.02, 0.05]


def test_plan_starts_with_hold():
    plan = build_step_plan(JOINTS)
    assert plan[0].phase == "hold"
    assert plan[0].joint is None


def test_plan_returns_immutable_tuple_of_frozen_specs():
    plan = build_step_plan(JOINTS)
    assert isinstance(plan, tuple)
    with pytest.raises(Exception):
        plan[0].amplitude = 9.9          # frozen dataclass


def test_plan_rejects_amplitude_above_safety_cap():
    with pytest.raises(ValueError, match="진폭"):
        build_step_plan(JOINTS, amplitudes=(0.5,))


def test_plan_rejects_empty_joints():
    with pytest.raises(ValueError, match="관절"):
        build_step_plan([])


# ----------------------------------------------------------------- clamp_command
def test_clamp_limits_step_size_from_base():
    base = np.zeros(3)
    target = np.array([1.0, 0.0, 0.0])
    out = clamp_command(base, target, JOINTS, LIMITS, max_step=0.1)
    assert out[0] == pytest.approx(0.1)


def test_clamp_respects_joint_limits():
    base = np.array([0.0, 0.0, 0.04])
    target = np.array([0.0, 0.0, 0.09])
    out = clamp_command(base, target, JOINTS, LIMITS, max_step=1.0)
    assert out[2] == pytest.approx(0.05)      # j3 상한


def test_clamp_does_not_mutate_inputs():
    base = np.zeros(3)
    target = np.array([1.0, 1.0, 1.0])
    base_copy, target_copy = base.copy(), target.copy()
    clamp_command(base, target, JOINTS, LIMITS, max_step=0.1)
    assert np.array_equal(base, base_copy)
    assert np.array_equal(target, target_copy)


def test_clamp_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="차원"):
        clamp_command(np.zeros(2), np.zeros(3), JOINTS, LIMITS, max_step=0.1)


def test_clamp_raises_on_unknown_joint():
    with pytest.raises(KeyError):
        clamp_command(np.zeros(3), np.zeros(3), ["j1", "j2", "zz"], LIMITS, max_step=0.1)


# ----------------------------------------------------------------- evaluate_step
def _spec(joint, amp):
    return StepSpec(phase="step", joint=joint, amplitude=amp, duration_s=2.0)


def test_step_verdict_ok_when_only_target_joint_moves_correctly():
    base = np.zeros(3)
    end = np.array([0.049, 0.0, 0.0])
    v = evaluate_step(JOINTS, base, end, _spec("j1", 0.05))
    assert isinstance(v, JointVerdict)
    assert v.ok
    assert v.measured == pytest.approx(0.049)


def test_step_verdict_fails_on_wrong_sign():
    base = np.zeros(3)
    end = np.array([-0.05, 0.0, 0.0])
    v = evaluate_step(JOINTS, base, end, _spec("j1", 0.05))
    assert not v.ok
    assert "부호" in v.reason


def test_step_verdict_fails_when_magnitude_far_off():
    base = np.zeros(3)
    end = np.array([0.005, 0.0, 0.0])          # 10% 만 움직임
    v = evaluate_step(JOINTS, base, end, _spec("j1", 0.05))
    assert not v.ok
    assert "크기" in v.reason


def test_step_verdict_fails_on_crosstalk():
    base = np.zeros(3)
    end = np.array([0.05, 0.03, 0.0])          # j2 가 같이 움직임
    v = evaluate_step(JOINTS, base, end, _spec("j1", 0.05))
    assert not v.ok
    assert "간섭" in v.reason
    assert v.crosstalk_joint == "j2"


def test_step_verdict_reports_ratio():
    base = np.zeros(3)
    end = np.array([0.025, 0.0, 0.0])
    v = evaluate_step(JOINTS, base, end, _spec("j1", 0.05))
    assert v.ratio == pytest.approx(0.5)


def test_step_verdict_requires_step_phase():
    with pytest.raises(ValueError, match="step"):
        evaluate_step(JOINTS, np.zeros(3), np.zeros(3),
                      StepSpec(phase="hold", joint=None, amplitude=0.0, duration_s=1.0))


# ----------------------------------------------------------------- evaluate_hold
def test_hold_reports_per_joint_max_drift_signed():
    samples = [np.zeros(3), np.array([-0.01, 0.002, 0.0]), np.array([-0.03, 0.001, 0.0])]
    drift = evaluate_hold(JOINTS, samples)
    assert drift["j1"] == pytest.approx(-0.03)
    assert drift["j2"] == pytest.approx(0.002)
    assert drift["j3"] == pytest.approx(0.0)


def test_hold_uses_first_sample_as_reference():
    samples = [np.array([0.5, 0.0, 0.0]), np.array([0.48, 0.0, 0.0])]
    drift = evaluate_hold(JOINTS, samples)
    assert drift["j1"] == pytest.approx(-0.02)


def test_hold_raises_on_empty():
    with pytest.raises(ValueError, match="샘플"):
        evaluate_hold(JOINTS, [])


# ------------------------------------------------------------ summarize_sign_table
def test_sign_table_rows_carry_measured_sign_per_joint():
    verdicts = [
        evaluate_step(JOINTS, np.zeros(3), np.array([0.05, 0, 0]), _spec("j1", 0.05)),
        evaluate_step(JOINTS, np.zeros(3), np.array([-0.05, 0, 0]), _spec("j1", -0.05)),
        evaluate_step(JOINTS, np.zeros(3), np.array([0, -0.05, 0]), _spec("j2", 0.05)),
    ]
    table = summarize_sign_table(verdicts)
    assert table["j1"]["sign"] == 1.0
    assert table["j2"]["sign"] == -1.0        # 명령 +, 실측 − → 부호 반전 관절
    assert table["j1"]["ok"] is True
    assert table["j2"]["ok"] is False


def test_sign_table_marks_unmeasured_joint_as_none():
    verdicts = [evaluate_step(JOINTS, np.zeros(3), np.array([0.05, 0, 0]), _spec("j1", 0.05))]
    table = summarize_sign_table(verdicts, all_joints=JOINTS)
    assert table["j3"]["sign"] is None
