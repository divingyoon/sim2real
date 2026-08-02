from pathlib import Path

import numpy as np
import pytest

from jtc_bridge_core import (
    JointRemap,
    load_profile_joints,
    safe_time_from_start,
    time_from_start_sec,
    velocity_limited_target,
)

# 로컬 robot_control profile (있으면 실제 매핑으로 통합 검증)
_PROFILE = Path("/home/user/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml")

ARM_CANON = [f"r_aj_{i}" for i in range(1, 8)]
ARM_SOURCE = [f"openarm_right_joint{i}" for i in range(1, 8)]
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HAND_CANON = [f"r_hj_{f}_{j}" for f in _FINGERS for j in range(1, 5)]
HAND_SOURCE = [f"rj_dg_{fi}_{j}" for fi in range(1, 6) for j in range(1, 5)]


# 합성 profile (부호·순서 뒤섞기 검증용)
def _synthetic_profile():
    return {
        "r_aj_1": {"source": "s1", "sign": 1.0, "lower": -1.0, "upper": 1.0, "unit": "rad"},
        "r_aj_2": {"source": "s2", "sign": -1.0, "lower": -2.0, "upper": 2.0, "unit": "rad"},
        "r_aj_3": {"source": "s3", "sign": 1.0, "lower": 0.0, "upper": 0.5, "unit": "rad"},
    }


def test_synthetic_reorder_and_sign():
    prof = _synthetic_profile()
    # 입력 canonical 순서 [r_aj_1, r_aj_2, r_aj_3], 출력 source 순서 [s3, s1, s2] (뒤섞음)
    remap = JointRemap(["r_aj_1", "r_aj_2", "r_aj_3"], ["s3", "s1", "s2"], prof)
    out = remap.apply([0.4, 0.9, 0.3])
    # s3 <- r_aj_3(0.3)*1 clamp[0,0.5]=0.3 ; s1 <- r_aj_1(0.4)*1=0.4 ; s2 <- r_aj_2(0.9)*-1=-0.9
    assert np.allclose(out, [0.3, 0.4, -0.9])


def test_synthetic_clamp():
    prof = _synthetic_profile()
    remap = JointRemap(["r_aj_1", "r_aj_2", "r_aj_3"], ["s1", "s2", "s3"], prof)
    out = remap.apply([5.0, 0.0, 9.0])
    assert out[0] == pytest.approx(1.0)   # r_aj_1 clamp upper 1.0
    assert out[2] == pytest.approx(0.5)   # r_aj_3 clamp upper 0.5


def test_missing_source_raises():
    prof = _synthetic_profile()
    with pytest.raises(KeyError):
        JointRemap(["r_aj_1", "r_aj_2", "r_aj_3"], ["s1", "no_such"], prof)


def test_wrong_input_length_raises():
    prof = _synthetic_profile()
    remap = JointRemap(["r_aj_1", "r_aj_2", "r_aj_3"], ["s1", "s2", "s3"], prof)
    with pytest.raises(ValueError):
        remap.apply([0.1, 0.2])


def test_time_from_start_positive():
    assert time_from_start_sec(1.0 / 60.0, 2.0) == pytest.approx(2.0 / 60.0)
    for bad in [(0.0, 2.0), (1 / 60, 0.0), (-1.0, 2.0)]:
        with pytest.raises(ValueError):
            time_from_start_sec(*bad)


def test_safe_tfs_small_delta_uses_min():
    # 작은 델타(0.01rad) → max_vel 0.5 면 0.02s 필요 < min 0.033 → min 유지
    cur = np.zeros(7)
    tgt = np.full(7, 0.01)
    assert safe_time_from_start(cur, tgt, max_vel=0.5, min_tfs=0.0333) == pytest.approx(0.0333)


def test_safe_tfs_big_jump_slows_down():
    # 큰 점프(0.9rad) → 0.5rad/s 면 1.8s 필요 (min 0.033 무시)
    cur = np.zeros(7)
    tgt = np.zeros(7); tgt[3] = 0.9
    assert safe_time_from_start(cur, tgt, max_vel=0.5, min_tfs=0.0333) == pytest.approx(1.8)


def test_safe_tfs_uses_max_joint_displacement():
    cur = np.zeros(3)
    tgt = np.array([0.1, 0.6, 0.2])   # 최대 0.6
    assert safe_time_from_start(cur, tgt, max_vel=0.3, min_tfs=0.01) == pytest.approx(2.0)


def test_safe_tfs_shape_mismatch_and_bad_params():
    with pytest.raises(ValueError):
        safe_time_from_start(np.zeros(3), np.zeros(4), 0.5, 0.03)
    with pytest.raises(ValueError):
        safe_time_from_start(np.zeros(3), np.zeros(3), 0.0, 0.03)
    with pytest.raises(ValueError):
        safe_time_from_start(np.zeros(3), np.zeros(3), 0.5, 0.0)


# ---------------------------------------------------------------------------
# velocity_limited_target — interpolation="none" 컨트롤러용 세트포인트 rate-limit
# 시그니처: (target, last_setpoint, actual, max_vel, dt, max_follow_err)
# ---------------------------------------------------------------------------
_STEP = 0.1 * (1.0 / 60.0)   # max_vel 0.1 · dt 1/60


def _vlt(target, last, actual, max_vel=0.1, dt=1.0 / 60.0, follow=1.0):
    return velocity_limited_target(target, last, actual, max_vel, dt, follow)


def test_vlt_advances_from_last_by_step():
    # 큰 목표 → 직전 세트포인트에서 step 만큼만 전진(캡 넉넉).
    out = _vlt(np.full(7, 0.5), np.zeros(7), np.zeros(7))
    assert np.allclose(out, _STEP)


def test_vlt_small_delta_reaches_target_no_overshoot():
    # 델타가 step 이하면 목표 그대로.
    tgt = np.array([0.0005, -0.0003, 0.0])
    out = _vlt(tgt, np.zeros(3), np.zeros(3))
    assert np.allclose(out, tgt)


def test_vlt_deadlock_fix_advances_while_actual_frozen():
    # ★핵심: 실제(actual) 가 얼어 있어도 세트포인트는 직전 기준으로 계속 전진.
    #   (실제 기준 클램프면 두 스텝 다 actual+step 로 고정 = 교착)
    actual = np.zeros(3)          # 팔이 정지마찰로 안 움직임
    last = np.zeros(3)
    out1 = _vlt(np.full(3, 1.0), last, actual, follow=1.0)
    out2 = _vlt(np.full(3, 1.0), out1, actual, follow=1.0)
    assert np.allclose(out1, _STEP)
    assert np.allclose(out2, 2 * _STEP)   # 전진함 (교착 아님)
    assert np.all(out2 > out1)


def test_vlt_follow_err_caps_lead_over_actual():
    # 팔이 막혀 세트포인트가 앞서가도 실제 ±follow 이내로 캡(급발진 방지).
    out = _vlt(np.full(3, 1.0), np.full(3, 0.5), np.zeros(3), follow=0.1)
    assert np.allclose(out, 0.1)   # 0.5+step 이지만 actual(0)+0.1 로 캡


def test_vlt_follow_cap_both_directions():
    out = _vlt(np.full(2, -1.0), np.full(2, -0.5), np.zeros(2), follow=0.1)
    assert np.allclose(out, -0.1)


def test_vlt_returns_new_array_no_mutation():
    last = np.zeros(3)
    tgt = np.full(3, 0.5)
    out = _vlt(tgt, last, np.zeros(3))
    assert out is not last and out is not tgt
    assert np.allclose(last, 0.0)   # 입력 불변


def test_vlt_bad_params_and_shape():
    with pytest.raises(ValueError):
        _vlt(np.zeros(4), np.zeros(3), np.zeros(3))          # shape 불일치
    with pytest.raises(ValueError):
        velocity_limited_target(np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 1 / 60, 0.1)
    with pytest.raises(ValueError):
        velocity_limited_target(np.zeros(3), np.zeros(3), np.zeros(3), 0.1, 0.0, 0.1)
    with pytest.raises(ValueError):
        velocity_limited_target(np.zeros(3), np.zeros(3), np.zeros(3), 0.1, 1 / 60, 0.0)


# ---------------------------------------------------------------------------
# 실제 profile 통합 (있을 때만)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _PROFILE.exists(), reason="robot_control profile 없음")
def test_real_profile_arm_identity():
    prof = load_profile_joints(_PROFILE)
    remap = JointRemap(ARM_CANON, ARM_SOURCE, prof)
    vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    out = remap.apply(vals)
    # sign=1·한계 내 값 → 그대로 (index 정렬 r_aj_i↔openarm_right_jointi)
    assert np.allclose(out, vals)


@pytest.mark.skipif(not _PROFILE.exists(), reason="robot_control profile 없음")
def test_real_profile_hand_identity_finger_major():
    prof = load_profile_joints(_PROFILE)
    remap = JointRemap(HAND_CANON, HAND_SOURCE, prof)
    vals = np.linspace(-1.0, 1.0, 20)
    out = remap.apply(vals)
    # r_hj_{finger}_{j} ↔ rj_dg_{fingerIdx}_{j} finger-major 동일 정렬, sign=1, 한계 ±1.5 내
    assert np.allclose(out, vals)


@pytest.mark.skipif(not _PROFILE.exists(), reason="robot_control profile 없음")
def test_real_profile_hand_clamps_to_limits():
    prof = load_profile_joints(_PROFILE)
    remap = JointRemap(HAND_CANON, HAND_SOURCE, prof)
    out = remap.apply(np.full(20, 5.0))   # 한계 초과
    assert np.all(out <= 1.5 + 1e-9)      # 손 관절 upper 1.5
    assert np.all(out >= -1.5 - 1e-9)


@pytest.mark.skipif(not _PROFILE.exists(), reason="robot_control profile 없음")
def test_real_profile_all_signs_positive_units_rad():
    prof = load_profile_joints(_PROFILE)
    for name in ARM_CANON + HAND_CANON:
        assert prof[name]["sign"] == 1.0
        assert prof[name]["unit"] == "rad"
