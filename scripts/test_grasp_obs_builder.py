import numpy as np
import pytest

from grasp_obs_builder import (
    ACTOR_OBS_DIM,
    BASE_OBS_DIM,
    NUM_OBJECT_CLASSES,
    OBJECT_NAMES,
    REAL_CUP_INDEX,
    assemble_actor_obs,
    make_object_onehot,
)


# ---------------------------------------------------------------------------
# 고정 입력 헬퍼: 슬라이스별로 구분되는 값을 채워 순서를 검증한다.
# ---------------------------------------------------------------------------
def _inputs(onehot=None):
    return dict(
        arm_joint_pos=np.arange(7, dtype=np.float64) + 100.0,       # 100..106
        arm_joint_vel=np.arange(7, dtype=np.float64) + 200.0,       # 200..206
        finger_joint_pos=np.arange(20, dtype=np.float64) + 300.0,   # 300..319
        finger_joint_vel=np.arange(20, dtype=np.float64) + 400.0,   # 400..419
        palm_center=np.array([1.0, 2.0, 3.0]),
        fingertip_pos=np.zeros((5, 3)),                             # tips at origin
        cup_pos=np.array([10.0, 20.0, 30.0]),
        binary_contact=np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        last_actions=np.arange(11, dtype=np.float64) + 500.0,       # 500..510
        object_onehot=onehot if onehot is not None else make_object_onehot(REAL_CUP_INDEX),
    )


def test_dim_is_114_and_base_106():
    assert ACTOR_OBS_DIM == 114
    assert BASE_OBS_DIM == 106
    assert NUM_OBJECT_CLASSES == 8
    obs = assemble_actor_obs(**_inputs())
    assert obs.shape == (114,)


def test_slice_order():
    obs = assemble_actor_obs(**_inputs())
    assert np.allclose(obs[0:7], np.arange(7) + 100.0)      # arm_pos
    assert np.allclose(obs[7:14], np.arange(7) + 200.0)     # arm_vel
    assert np.allclose(obs[14:34], np.arange(20) + 300.0)   # finger_pos
    assert np.allclose(obs[34:54], np.arange(20) + 400.0)   # finger_vel
    assert np.allclose(obs[54:57], [1.0, 2.0, 3.0])         # palm_center
    # fingertip_rel_palm 15D (tips - palm) = -palm repeated 5x (tips at origin)
    assert np.allclose(obs[57:72], np.tile([-1.0, -2.0, -3.0], 5))
    # palm_to_cup 3D = cup - palm
    assert np.allclose(obs[72:75], [9.0, 18.0, 27.0])
    # cup_to_fingertip 15D (tips - cup) = -cup repeated 5x
    assert np.allclose(obs[75:90], np.tile([-10.0, -20.0, -30.0], 5))
    assert np.allclose(obs[90:95], [1.0, 0.0, 1.0, 0.0, 1.0])   # binary_contact
    assert np.allclose(obs[95:106], np.arange(11) + 500.0)      # last_actions
    assert np.allclose(obs[106:114], make_object_onehot(REAL_CUP_INDEX))  # onehot


def test_palm_to_cup_sign_is_cup_minus_palm():
    inp = _inputs()
    inp["palm_center"] = np.array([0.0, 0.0, 0.0])
    inp["cup_pos"] = np.array([1.0, 2.0, 3.0])
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[72:75], [1.0, 2.0, 3.0])


def test_cup_to_fingertip_sign_is_tip_minus_cup():
    inp = _inputs()
    inp["cup_pos"] = np.array([0.0, 0.0, 0.0])
    inp["fingertip_pos"] = np.arange(15, dtype=np.float64).reshape(5, 3)
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[75:90], np.arange(15))


def test_fingertip_rel_palm_sign_is_tip_minus_palm():
    inp = _inputs()
    inp["palm_center"] = np.array([0.0, 0.0, 0.0])
    inp["fingertip_pos"] = np.arange(15, dtype=np.float64).reshape(5, 3)
    obs = assemble_actor_obs(**inp)
    assert np.allclose(obs[57:72], np.arange(15))


def test_onehot_from_index():
    oh = make_object_onehot(1)
    assert oh.shape == (8,)
    assert np.allclose(oh, [0, 1, 0, 0, 0, 0, 0, 0])
    assert oh.sum() == 1.0


def test_onehot_from_name():
    assert np.allclose(make_object_onehot("cup_big_s100"), make_object_onehot(1))
    assert np.allclose(make_object_onehot("shaker_body"), make_object_onehot(4))


def test_real_cup_default_is_cup_big_s100():
    assert REAL_CUP_INDEX == 1
    assert OBJECT_NAMES[REAL_CUP_INDEX] == "cup_big_s100"
    assert len(OBJECT_NAMES) == 8


def test_onehot_invalid_raises():
    for bad in (8, -1, "no_such_object"):
        with pytest.raises((ValueError, IndexError)):
            make_object_onehot(bad)


def test_wrong_input_dims_raise():
    for key, bad in [
        ("arm_joint_pos", np.zeros(6)),
        ("finger_joint_pos", np.zeros(19)),
        ("binary_contact", np.zeros(4)),
        ("last_actions", np.zeros(10)),
        ("fingertip_pos", np.zeros((4, 3))),
        ("object_onehot", np.zeros(7)),
    ]:
        inp = _inputs()
        inp[key] = bad
        with pytest.raises(ValueError):
            assemble_actor_obs(**inp)
