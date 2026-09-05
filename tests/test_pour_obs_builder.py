"""pour obs 조립 검증 (numpy만, Isaac 불필요)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pour_obs_builder import (  # noqa: E402
    ACTOR_OBS_DIM,
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    assemble_actor_obs,
    compose_pose,
    finger_grasp_progress,
    quat_mul,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_grasp_progress_zero_at_approach():
    prog = finger_grasp_progress(np.array(HAND_APPROACH_POSE))
    assert np.allclose(prog, 0.0)


def test_grasp_progress_one_at_grasp():
    prog = finger_grasp_progress(np.array(HAND_GRASP_POSE))
    assert np.allclose(prog, 1.0)


def test_grasp_progress_halfway():
    mid = (np.array(HAND_APPROACH_POSE) + np.array(HAND_GRASP_POSE)) / 2
    prog = finger_grasp_progress(mid)
    assert np.allclose(prog, 0.5, atol=1e-9)


def test_grasp_progress_shape_and_clamp():
    prog = finger_grasp_progress(np.full(20, 10.0))  # 과굽힘 → clamp 1
    assert prog.shape == (5,)
    assert np.allclose(prog, 1.0)


def test_grasp_progress_rejects_wrong_length():
    with pytest.raises(ValueError):
        finger_grasp_progress(np.zeros(19))


def test_quat_mul_identity():
    q = np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)])
    assert np.allclose(quat_mul(IDENT, q), q)
    assert np.allclose(quat_mul(q, IDENT), q)


def test_compose_pose_identity_offset_returns_palm():
    palm_pos = np.array([0.3, -0.2, 0.4])
    palm_quat = np.array([np.cos(0.5), 0.0, 0.0, np.sin(0.5)])
    pos, quat = compose_pose(palm_pos, palm_quat, np.zeros(3), IDENT)
    assert np.allclose(pos, palm_pos)
    assert np.allclose(quat, palm_quat)


def test_compose_pose_translation_offset_in_parent_frame():
    # palm이 z축 90° 회전 상태에서 palm-frame +x 오프셋 → world +y로.
    palm_pos = np.array([0.0, 0.0, 0.0])
    palm_quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # z 90
    pos, _ = compose_pose(palm_pos, palm_quat, np.array([0.1, 0.0, 0.0]), IDENT)
    assert np.allclose(pos, [0.0, 0.1, 0.0], atol=1e-9)


def test_assemble_obs_dim_and_layout():
    obs = assemble_actor_obs(
        arm_joint_pos=np.arange(7.0),
        arm_joint_vel=np.zeros(7),
        finger_joint_pos=np.array(HAND_GRASP_POSE),
        left_arm_joint_pos=np.zeros(9),
        left_arm_joint_vel=np.zeros(9),
        source_cup_pos=np.array([0.4, 0.0, 0.3]),
        source_cup_quat=IDENT,
        target_cup_pos=np.array([0.4, 0.2, 0.1]),
        target_cup_quat=IDENT,
        last_palm_actions=np.full(6, 0.5),
    )
    assert obs.shape == (ACTOR_OBS_DIM,)
    assert np.allclose(obs[0:7], np.arange(7.0))       # arm_joint_pos
    assert np.allclose(obs[14:19], 1.0)                # grasp progress (grasp pose)
    assert np.allclose(obs[49:55], 0.5)                # last_palm_actions


def test_assemble_rejects_wrong_arm_length():
    with pytest.raises(ValueError):
        assemble_actor_obs(
            np.zeros(6), np.zeros(7), np.array(HAND_GRASP_POSE),
            np.zeros(9), np.zeros(9),
            np.array([0.4, 0.0, 0.3]), IDENT, np.array([0.4, 0.2, 0.1]), IDENT,
            np.zeros(6),
        )


def test_hand_poses_match_pour_v1_preset():
    preset = (
        Path(__file__).resolve().parents[1]
        / "../hdgp/source/openarm/openarm/tesollo/right/pour_v1/pour_right_preset.py"
    ).resolve()
    if not preset.is_file():
        pytest.skip(f"pour_v1 preset not found: {preset}")
    import re

    text = preset.read_text()
    for name, ours in [("HAND_APPROACH_POSE", HAND_APPROACH_POSE), ("HAND_GRASP_POSE", HAND_GRASP_POSE)]:
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
        assert m, f"{name} not found"
        # 인라인 주석(#...)에 숫자가 있으므로 줄별로 주석을 제거한 뒤 추출한다.
        body = "\n".join(line.split("#", 1)[0] for line in m.group(1).splitlines())
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", body)]
        assert nums == list(ours), f"{name} drift: preset={nums} ours={list(ours)}"
