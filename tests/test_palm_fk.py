"""palm FK·grasp offset·joint 어댑터 검증. hdgp openarm_fk와 직접 대조(drift-guard)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from palm_fk import (  # noqa: E402
    extract_joints,
    grasp_offset_from_snapshot,
    palm_pose,
    quat_inverse_wxyz,
    rot_to_quat_wxyz,
)
from pour_obs_builder import compose_pose  # noqa: E402
from pour_obs_geometry import quat_apply  # noqa: E402

HDGP_FK = (
    Path(__file__).resolve().parents[1] / "../hdgp/scripts/tools/openarm_fk.py"
).resolve()

TEST_QS = [
    [0.0] * 7,
    [0.5, 0.1, 0.4, 1.3, -0.2, 0.0, 0.0],   # ARM_START_POSE
    [0.5, 0.5, -0.6, 0.7, 0.0, 0.0, 1.0],   # 캘리브 기준점
    [1.0, -0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
    [-0.3, 0.4, 0.9, 1.8, 0.6, -0.5, 0.7],
]


def _load_hdgp_fk():
    if not HDGP_FK.is_file():
        pytest.skip(f"hdgp openarm_fk not found: {HDGP_FK}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("hdgp_openarm_fk", HDGP_FK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quat_roundtrip_matches_rotation():
    rng = np.random.default_rng(7)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        q = rot_to_quat_wxyz(R)
        v = rng.normal(size=3)
        assert np.allclose(quat_apply(q, v), R @ v, atol=1e-9)


def test_palm_pose_position_matches_hdgp_openarm_fk():
    """drift-guard: 위치가 hdgp 도구와 완전히 같아야 한다 (체인/오프셋 동일)."""
    hdgp = _load_hdgp_fk()
    for q in TEST_QS:
        pos, _ = palm_pose(q)
        ref = hdgp.arm_fk(q)["palm_center"]
        assert np.allclose(pos, ref, atol=1e-9), f"q={q}: {pos} != {ref}"


def test_palm_quat_axes_match_hdgp_palm_dirs():
    """quat이 회전시킨 +X/+Z가 hdgp palm_x_dir/palm_z_dir와 일치해야 한다."""
    hdgp = _load_hdgp_fk()
    for q in TEST_QS:
        _, quat = palm_pose(q)
        ref = hdgp.arm_fk(q)
        assert np.allclose(quat_apply(quat, [1, 0, 0]), ref["palm_x_dir"], atol=1e-9)
        assert np.allclose(quat_apply(quat, [0, 0, 1]), ref["palm_z_dir"], atol=1e-9)


def test_grasp_offset_roundtrip_recovers_cup_pose():
    """스냅샷으로 얻은 offset을 palm에 합성하면 원래 컵 pose가 나와야 한다."""
    palm_pos, palm_quat = palm_pose(TEST_QS[1])
    cup_pos = np.array([0.42, -0.18, 0.31])
    ang = 0.6
    cup_quat = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])

    off_pos, off_quat = grasp_offset_from_snapshot(palm_pos, palm_quat, cup_pos, cup_quat)
    rec_pos, rec_quat = compose_pose(palm_pos, palm_quat, off_pos, off_quat)

    assert np.allclose(rec_pos, cup_pos, atol=1e-9)
    # 쿼터니언 부호 자유도 고려해 |dot|≈1로 비교.
    assert abs(float(np.dot(rec_quat, cup_quat))) == pytest.approx(1.0, abs=1e-9)


def test_grasp_offset_tracks_palm_motion():
    """offset 고정 후 palm이 움직이면 복원 컵 pose도 강체로 따라가야 한다."""
    p0, q0 = palm_pose(TEST_QS[1])
    cup_pos = p0 + quat_apply(q0, [0.0, 0.05, 0.10])  # palm 프레임 고정점
    off_pos, off_quat = grasp_offset_from_snapshot(p0, q0, cup_pos, q0)

    p1, q1 = palm_pose(TEST_QS[4])
    rec_pos, _ = compose_pose(p1, q1, off_pos, off_quat)
    expected = p1 + quat_apply(q1, [0.0, 0.05, 0.10])
    assert np.allclose(rec_pos, expected, atol=1e-9)


def test_quat_inverse():
    q = np.array([np.cos(0.4), 0.2, 0.3, 0.1])
    q /= np.linalg.norm(q)
    v = np.array([0.3, -0.7, 0.2])
    assert np.allclose(quat_apply(quat_inverse_wxyz(q), quat_apply(q, v)), v, atol=1e-12)


def test_extract_joints_reorders_by_name():
    names = ["b", "c", "a"]
    values = [2.0, 3.0, 1.0]
    out = extract_joints(names, values, ("a", "b", "c"))
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_extract_joints_missing_raises():
    with pytest.raises(KeyError, match="missing"):
        extract_joints(["a"], [1.0], ("a", "b"))
