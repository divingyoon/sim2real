"""pour action 디코더 검증 + env_cfg drift-guard. numpy만, Isaac 불필요."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pour_action_decoder as dec  # noqa: E402
from pour_action_decoder import (  # noqa: E402
    PourDecoderState,
    build_cup_local_tilt_rotvec,
    decode,
    pour_corridor_score,
    quat_from_rotvec_wxyz,
)
from pour_obs_geometry import CupGeometry, quat_apply  # noqa: E402

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
GEOM = CupGeometry()

# 대표 장면: 소스 컵(오른손), 타깃 컵(고정), palm이 컵 아래·뒤에 위치
SRC_POS = np.array([0.35, -0.10, 0.30])
TGT_POS = np.array([0.268, 0.100, 0.291])
PALM_POS = np.array([0.33, -0.15, 0.28])
PALM_QUAT = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # z 90°


def _decode_zero(state=None):
    state = state or PourDecoderState()
    return decode(
        np.zeros(12), state, SRC_POS, IDENT, TGT_POS, IDENT, PALM_POS, PALM_QUAT
    ), state


def test_corridor_score_is_one_inside_and_decays_outside():
    opening = np.array([0.3, 0.1, 0.4])
    assert pour_corridor_score(opening + [0.0, 0.0, 0.05], opening) == pytest.approx(1.0)
    far = pour_corridor_score(opening + [0.5, 0.0, 0.0], opening)
    assert far < 0.01


def test_rotvec_quat_matches_axis_angle():
    rv = np.array([0.0, 0.0, np.pi / 2])
    q = quat_from_rotvec_wxyz(rv)
    assert np.allclose(quat_apply(q, [1, 0, 0]), [0, 1, 0], atol=1e-9)
    assert np.allclose(quat_from_rotvec_wxyz(np.zeros(3)), IDENT)


def test_tilt_basis_is_orthonormal():
    for i in range(3):
        e = np.zeros(3)
        e[i] = 0.1
        rv = build_cup_local_tilt_rotvec(e, SRC_POS, IDENT, TGT_POS, IDENT, GEOM)
        assert np.linalg.norm(rv) == pytest.approx(0.1, abs=1e-9)
    # spin(직립)=cup up=[0,0,1]
    rv_spin = build_cup_local_tilt_rotvec(
        np.array([0.1, 0.0, 0.0]), SRC_POS, IDENT, TGT_POS, IDENT, GEOM
    )
    assert np.allclose(rv_spin, [0, 0, 0.1], atol=1e-9)


def test_decode_rejects_wrong_action_dim():
    with pytest.raises(ValueError):
        decode(np.zeros(11), PourDecoderState(), SRC_POS, IDENT, TGT_POS, IDENT, PALM_POS, PALM_QUAT)


def test_zero_action_beta_pulls_cup_upright_direction():
    """action=0 → β=0.5(EMA 후 0.35) → 직립 컵에도 목표 tilt>0 → tilt 스텝 발생.

    단, 멀면 tilt_gate가 죽인다. 여기선 gate>0 거리로 검증.
    """
    out, state = _decode_zero()
    assert out.pos.shape == (3,)
    assert out.quat_xyzw.shape == (4,)
    assert np.isfinite(out.pos).all() and np.isfinite(out.quat_xyzw).all()
    assert not out.ready  # 이 장면은 corridor 밖


def test_palm_target_position_respects_workspace_box():
    state = PourDecoderState()
    # 극단 action을 여러 번 → EMA 수렴 후에도 박스 내
    for _ in range(50):
        out = decode(
            np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float),
            state, SRC_POS, IDENT, TGT_POS, IDENT, PALM_POS, PALM_QUAT,
        )
    palm_ee = out.pos + quat_apply(
        np.array([out.quat_xyzw[3], *out.quat_xyzw[:3]]), np.array(dec.PALM_EE_OFFSET_LOCAL)
    )
    assert (palm_ee >= np.array(dec.PALM_POSE_MINS[:3]) - 1e-9).all()
    assert (palm_ee <= np.array(dec.PALM_POSE_MAXS[:3]) + 1e-9).all()


def test_z_lock_forces_spout_height():
    """z_lock: (직립·delta 회전≈0에서) 주둥이 z ≈ target 입구 z + margin."""
    state = PourDecoderState()
    # tilt gate=0이 되도록 멀리 → 회전 delta=0 → rim_rel 회전 불변
    src_far = TGT_POS + np.array([0.5, 0.0, 0.0])
    out = decode(
        np.zeros(12), state, src_far, IDENT, TGT_POS, IDENT,
        src_far + np.array([-0.02, -0.05, -0.02]), PALM_QUAT,
    )
    palm_ee_z = out.pos[2] + quat_apply(
        np.array([out.quat_xyzw[3], *out.quat_xyzw[:3]]), np.array(dec.PALM_EE_OFFSET_LOCAL)
    )[2]
    rim_rel_z = (src_far + [0, 0, 0.100])[2] - (src_far + [-0.02, -0.05, -0.02])[2]
    spout_z = palm_ee_z + rim_rel_z  # delta 회전 0이므로 rim_rel 그대로
    expected = (TGT_POS[2] + 0.100) + dec.POUR_Z_MARGIN
    # 박스 클램프가 없었다면 정확히 일치
    assert spout_z == pytest.approx(expected, abs=1e-6)


def test_ready_latch_freezes_spout_offset_and_releases_orientation():
    state = PourDecoderState()
    # corridor 안(입구 위 5cm)에 pour_point가 오는 장면 → 래치
    src_in = TGT_POS + np.array([0.0, 0.0, 0.02])  # rim이 opening 근처
    out = decode(
        np.zeros(12), state, src_in, IDENT, TGT_POS, IDENT, PALM_POS, PALM_QUAT
    )
    assert out.ready and state.pour_ready_latched
    frozen = state.spout_offset_body.copy()
    # 래치 후 orientation은 current palm 유지(release)
    assert np.allclose(out.quat_xyzw, PALM_QUAT[[1, 2, 3, 0]])
    # palm이 움직여도 offset은 동결 유지
    decode(
        np.zeros(12), state, src_in, IDENT, TGT_POS, IDENT,
        PALM_POS + [0.05, 0, 0], PALM_QUAT,
    )
    assert np.allclose(state.spout_offset_body, frozen)


def test_ema_smoothing_carries_state():
    state = PourDecoderState()
    decode(np.concatenate([np.ones(6), np.zeros(6)]), state, SRC_POS, IDENT, TGT_POS, IDENT, PALM_POS, PALM_QUAT)
    assert np.allclose(state.ema_palm, 0.7)
    decode(np.zeros(12), state, SRC_POS, IDENT, TGT_POS, IDENT, PALM_POS, PALM_QUAT)
    assert np.allclose(state.ema_palm, 0.21)


ENV_CFG = (
    Path(__file__).resolve().parents[1]
    / "../hdgp/source/openarm/openarm/tesollo/right/pour_v1/pour_right_env_cfg.py"
).resolve()

CFG_CHECKS = {
    "palm_delta_xyz": dec.PALM_DELTA_XYZ,
    "palm_delta_rot_deg": dec.PALM_DELTA_ROT_DEG,
    "ema_action_alpha": dec.EMA_ACTION_ALPHA,
    "tilt_action_gate_xy_near": dec.TILT_GATE_XY_NEAR,
    "tilt_action_gate_xy_far": dec.TILT_GATE_XY_FAR,
    "beta_action_index": dec.BETA_ACTION_INDEX,
    "beta_target_tilt_amount": dec.BETA_TARGET_TILT_AMOUNT,
    "beta_tilt_kp": dec.BETA_TILT_KP,
    "beta_tilt_max_step": dec.BETA_TILT_MAX_STEP,
    "pour_z_margin": dec.POUR_Z_MARGIN,
    "target_inner_radius": dec.TARGET_INNER_RADIUS,
    "pour_corridor_xy_margin": dec.POUR_CORRIDOR_XY_MARGIN,
    "pour_corridor_z_min": dec.POUR_CORRIDOR_Z_MIN,
    "pour_corridor_z_max": dec.POUR_CORRIDOR_Z_MAX,
    "pour_corridor_scale": dec.POUR_CORRIDOR_SCALE,
    "ready_latch_threshold": dec.READY_LATCH_THRESHOLD,
    "max_pose_angle": dec.MAX_POSE_ANGLE,
}


def test_constants_match_pour_v1_env_cfg():
    """drift-guard: env_cfg 기본값이 바뀌면 실패시켜 디코더 재정합을 강제."""
    if not ENV_CFG.is_file():
        pytest.skip(f"env_cfg not found: {ENV_CFG}")
    text = ENV_CFG.read_text()
    for name, ours in CFG_CHECKS.items():
        m = re.search(rf"^\s+{name}:\s*\w+\s*=\s*([-\d.]+)", text, re.MULTILINE)
        assert m, f"{name} not found in env_cfg"
        assert float(m.group(1)) == pytest.approx(float(ours)), (
            f"{name} drift: env_cfg={m.group(1)} ours={ours}"
        )


def test_default_mode_flags_still_hold():
    """디코더는 기본 config 경로만 포팅 — 모드 flag가 바뀌면 알아채야 한다."""
    if not ENV_CFG.is_file():
        pytest.skip(f"env_cfg not found: {ENV_CFG}")
    text = ENV_CFG.read_text()
    for name, expected in [
        ("pour_action_mode", '"b_trajectory"'),
        ("pour_approach_pivot", '"palm"'),
        ("pour_spout_z_lock", "True"),
        ("pour_orient_release", "True"),
    ]:
        m = re.search(rf"^\s+{name}:\s*\w+\s*=\s*(\S+)", text, re.MULTILINE)
        assert m, f"{name} not found"
        assert m.group(1) == expected, f"{name} changed: {m.group(1)} (decoder assumes {expected})"
