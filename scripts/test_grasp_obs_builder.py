#!/usr/bin/env python3
"""grasp_obs_builder(154D) 유닛 테스트 — 세그먼트 순서·부호·정규화·검증.

08.16 계약 변경: 접촉 binary 5D → tip_force_local 15D, joint_pos_err 20D 신설,
last_actions 11→21. 총 114 → **154**.
"""

from __future__ import annotations

import numpy as np
import pytest

from grasp_obs_builder import (
    ACTOR_OBS_DIM,
    CONTACT_FORCE_MAX,
    JOINT_POS_ERR_MAX,
    NUM_ACTIONS,
    NUM_OBJECT_CLASSES,
    OBS_SEGMENTS,
    OBS_SLICES,
    REAL_CUP_INDEX,
    assemble_actor_obs,
    compute_joint_pos_err,
    make_object_onehot,
    normalize_tip_force,
)


def _inputs(**over):
    inp = dict(
        arm_joint_pos=np.arange(7, dtype=float),
        arm_joint_vel=np.arange(7, dtype=float) * 0.1,
        finger_joint_pos=np.arange(20, dtype=float) * 0.01,
        finger_joint_vel=np.arange(20, dtype=float) * 0.001,
        palm_center=np.array([0.3, -0.2, 0.4]),
        fingertip_pos=np.arange(15, dtype=float).reshape(5, 3) * 0.01,
        cup_pos=np.array([0.35, -0.1, 0.3]),
        tip_force_local=np.zeros((5, 3)),
        hand_cmd_prev=np.arange(20, dtype=float) * 0.01,
        last_actions=np.linspace(-1.0, 1.0, NUM_ACTIONS),
        object_onehot=make_object_onehot(REAL_CUP_INDEX),
    )
    inp.update(over)
    return inp


# --------------------------------------------------------------------------
# 차원·레이아웃
# --------------------------------------------------------------------------

def test_dim_is_154():
    assert ACTOR_OBS_DIM == 154
    assert sum(dim for _, dim in OBS_SEGMENTS) == 154
    assert assemble_actor_obs(**_inputs()).shape == (154,)


def test_segment_offsets_match_sim_layout():
    """sim `_get_observations` 의 torch.cat 순서 그대로여야 한다."""
    expected = {
        "arm_joint_pos": (0, 7),
        "arm_joint_vel": (7, 14),
        "finger_joint_pos": (14, 34),
        "finger_joint_vel": (34, 54),
        "palm_center_pos": (54, 57),
        "fingertip_pos_rel_palm": (57, 72),
        "palm_to_cup": (72, 75),
        "cup_to_fingertip": (75, 90),
        "tip_force_local": (90, 105),
        "joint_pos_err": (105, 125),
        "last_actions": (125, 146),
        "object_onehot": (146, 154),
    }
    for name, (start, stop) in expected.items():
        assert OBS_SLICES[name] == slice(start, stop), name


def test_slice_order_carries_values():
    obs = assemble_actor_obs(**_inputs())
    inp = _inputs()
    assert np.allclose(obs[OBS_SLICES["arm_joint_pos"]], inp["arm_joint_pos"])
    assert np.allclose(obs[OBS_SLICES["arm_joint_vel"]], inp["arm_joint_vel"])
    assert np.allclose(obs[OBS_SLICES["finger_joint_pos"]], inp["finger_joint_pos"])
    assert np.allclose(obs[OBS_SLICES["finger_joint_vel"]], inp["finger_joint_vel"])
    assert np.allclose(obs[OBS_SLICES["palm_center_pos"]], inp["palm_center"])
    assert np.allclose(obs[OBS_SLICES["last_actions"]], inp["last_actions"])
    assert np.allclose(obs[OBS_SLICES["object_onehot"]], inp["object_onehot"])


# --------------------------------------------------------------------------
# 기하 부호
# --------------------------------------------------------------------------

def test_palm_to_cup_sign_is_cup_minus_palm():
    inp = _inputs(palm_center=np.array([0.1, 0.2, 0.3]), cup_pos=np.array([0.4, 0.6, 0.9]))
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[OBS_SLICES["palm_to_cup"]], [0.3, 0.4, 0.6])


def test_cup_to_fingertip_sign_is_tip_minus_cup():
    tips = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    inp = _inputs(fingertip_pos=tips, cup_pos=np.array([0.5, 0.5, 0.5]))
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[OBS_SLICES["cup_to_fingertip"]].reshape(5, 3), tips - 0.5)


def test_fingertip_rel_palm_sign_is_tip_minus_palm():
    tips = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    inp = _inputs(fingertip_pos=tips, palm_center=np.array([0.5, 0.5, 0.5]))
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[OBS_SLICES["fingertip_pos_rel_palm"]].reshape(5, 3), tips - 0.5)


# --------------------------------------------------------------------------
# tip_force_local (15D) — tip-local 그대로, 10N 정규화
# --------------------------------------------------------------------------

def test_tip_force_normalized_by_contact_force_max():
    f = np.zeros((5, 3))
    f[0] = [5.0, 0.0, 0.0]
    assert normalize_tip_force(f)[0] == pytest.approx(5.0 / CONTACT_FORCE_MAX)


def test_tip_force_clamped_both_directions():
    f = np.full((5, 3), 20.0)
    assert np.allclose(normalize_tip_force(f), 1.0)
    assert np.allclose(normalize_tip_force(-f), -1.0)


def test_tip_force_sign_preserved():
    """방향 정보가 살아야 한다 — norm 으로 뭉개면 안 된다."""
    f = np.zeros((5, 3))
    f[2] = [0.0, 0.0, -3.0]
    out = normalize_tip_force(f).reshape(5, 3)
    assert out[2, 2] == pytest.approx(-0.3)


def test_tip_force_is_tip_major_in_obs():
    """레이아웃 [tip0_xyz, tip1_xyz, ...] — sim 의 .view(N,-1) 과 같은 순서."""
    f = np.zeros((5, 3))
    f[2] = [1.0, 2.0, 3.0]
    obs = assemble_actor_obs(**_inputs(tip_force_local=f))
    seg = obs[OBS_SLICES["tip_force_local"]]
    assert np.allclose(seg[6:9], [0.1, 0.2, 0.3])
    assert np.allclose(np.delete(seg, [6, 7, 8]), 0.0)


# --------------------------------------------------------------------------
# joint_pos_err (20D) — 부호 보존이 핵심
# --------------------------------------------------------------------------

def test_joint_pos_err_is_cmd_minus_actual():
    cmd = np.full(20, 0.6)
    act = np.zeros(20)
    assert np.allclose(compute_joint_pos_err(cmd, act), 0.5)   # 0.6/1.2


def test_joint_pos_err_sign_preserved_when_cmd_below_actual():
    """지령이 실측보다 작으면 **음수** — abs() 로 지우면 방향 정보가 사라진다."""
    err = compute_joint_pos_err(np.zeros(20), np.full(20, 0.6))
    assert np.all(err < 0)
    assert err[0] == pytest.approx(-0.5)


def test_joint_pos_err_clamped():
    assert np.allclose(compute_joint_pos_err(np.full(20, 5.0), np.zeros(20)), 1.0)
    assert np.allclose(compute_joint_pos_err(np.full(20, -5.0), np.zeros(20)), -1.0)


def test_joint_pos_err_uses_max_constant():
    assert JOINT_POS_ERR_MAX == 1.2
    err = compute_joint_pos_err(np.full(20, JOINT_POS_ERR_MAX), np.zeros(20))
    assert np.allclose(err, 1.0)


def test_joint_pos_err_in_obs_uses_measured_finger_pos():
    """obs 안의 오차는 인자로 준 실측 finger_joint_pos 를 쓴다(내부 재계산 금지)."""
    inp = _inputs(finger_joint_pos=np.zeros(20), hand_cmd_prev=np.full(20, 0.6))
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[OBS_SLICES["joint_pos_err"]], 0.5)


# --------------------------------------------------------------------------
# onehot
# --------------------------------------------------------------------------

def test_onehot_from_index():
    oh = make_object_onehot(3)
    assert oh.shape == (NUM_OBJECT_CLASSES,) and oh[3] == 1.0 and oh.sum() == 1.0


def test_onehot_from_name():
    assert np.array_equal(make_object_onehot("shaker_body"), make_object_onehot(4))


def test_real_cup_default_is_cup_big_s100():
    assert REAL_CUP_INDEX == 1
    assert make_object_onehot("cup_big_s100")[REAL_CUP_INDEX] == 1.0


def test_onehot_invalid_raises():
    with pytest.raises(ValueError):
        make_object_onehot(8)
    with pytest.raises(ValueError):
        make_object_onehot("없는물체")


# --------------------------------------------------------------------------
# 검증 — 조용히 통과하면 안 되는 것들
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,bad", [
    ("arm_joint_pos", np.zeros(6)),
    ("arm_joint_vel", np.zeros(8)),
    ("finger_joint_pos", np.zeros(19)),
    ("finger_joint_vel", np.zeros(21)),
    ("palm_center", np.zeros(2)),
    ("fingertip_pos", np.zeros((4, 3))),
    ("cup_pos", np.zeros(4)),
    ("tip_force_local", np.zeros((5, 2))),
    ("hand_cmd_prev", np.zeros(19)),
    ("last_actions", np.zeros(11)),          # ★구 계약 11D 는 거부되어야 한다
    ("object_onehot", np.zeros(7)),
])
def test_wrong_input_dims_raise(key, bad):
    with pytest.raises(ValueError):
        assemble_actor_obs(**_inputs(**{key: bad}))


def test_positional_call_rejected():
    """keyword-only — 위치인자 오배선이 조용히 통과하던 경로를 막는다."""
    inp = _inputs()
    with pytest.raises(TypeError):
        assemble_actor_obs(inp["arm_joint_pos"], inp["arm_joint_vel"])   # type: ignore[misc]


@pytest.mark.parametrize("key", ["arm_joint_pos", "cup_pos", "tip_force_local"])
def test_nan_input_raises(key):
    inp = _inputs()
    arr = np.array(inp[key], dtype=float)
    arr.reshape(-1)[0] = np.nan
    with pytest.raises((RuntimeError, ValueError)):
        assemble_actor_obs(**_inputs(**{key: arr}))
