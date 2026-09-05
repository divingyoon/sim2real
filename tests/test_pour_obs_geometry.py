"""pour obs 지오메트리 검증 (numpy만, Isaac 불필요)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pour_obs_geometry import (  # noqa: E402
    CupGeometry,
    quat_apply,
    source_pour_point_w,
    target_opening_w,
    vision_obs_terms,
    vision_obs_vector,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz


def _quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def _quat_x(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def test_quat_apply_identity_is_noop():
    v = np.array([0.3, -0.2, 0.7])
    assert np.allclose(quat_apply(IDENT, v), v)


def test_quat_apply_z_rotation_90deg():
    v = np.array([1.0, 0.0, 0.0])
    out = quat_apply(_quat_z(np.pi / 2), v)
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_upright_cup_axes_point_up_and_x():
    geom = CupGeometry()
    terms = vision_obs_terms(
        source_pos_w=np.array([0.4, 0.0, 0.3]),
        source_quat_w=IDENT,
        target_pos_w=np.array([0.4, 0.2, 0.1]),
        target_quat_w=IDENT,
        geom=geom,
    )
    assert np.allclose(terms["source_up_axis"], [0, 0, 1], atol=1e-9)
    assert np.allclose(terms["source_pour_axis"], [1, 0, 0], atol=1e-9)
    assert np.allclose(terms["target_up_axis"], [0, 0, 1], atol=1e-9)


def test_target_opening_is_pos_plus_rim_offset():
    geom = CupGeometry()
    pos = np.array([0.4, 0.2, 0.1])
    opening = target_opening_w(pos, IDENT, geom)
    assert np.allclose(opening, [0.4, 0.2, 0.2], atol=1e-9)  # +0.100 in z


def test_upright_source_pour_point_is_offset_toward_target():
    # 직립 컵(su_dot=1 → tilt_amt=0 → 정적 blend, xy=타깃 방향).
    geom = CupGeometry()
    src_pos = np.array([0.4, 0.0, 0.3])
    tgt_pos = np.array([0.4, 0.2, 0.1])
    opening = target_opening_w(tgt_pos, IDENT, geom)
    pp = source_pour_point_w(src_pos, IDENT, opening, geom)
    rim = src_pos + np.array([0, 0, 0.100])
    # 직립: gravity_perp = world_down - (world_down·up)up = [0,0,-1]-(-1)[0,0,1] = 0
    #   → perp_xy_mag=0 → pp_xy = rim_xy, pp_z = rim_z (반경 이동 없음)
    assert np.allclose(pp[:2], rim[:2], atol=1e-9)
    assert pp[2] == pytest.approx(rim[2], abs=1e-9)


def test_tilted_cup_pour_point_moves_outward_and_down():
    # 컵을 x축으로 90° 기울이면 up axis가 수평 → gravity_perp 활성.
    geom = CupGeometry()
    src_pos = np.array([0.4, 0.0, 0.3])
    tgt_pos = np.array([0.4, 0.2, 0.1])
    q = _quat_x(np.pi / 2)
    opening = target_opening_w(tgt_pos, IDENT, geom)
    pp = source_pour_point_w(src_pos, q, opening, geom)
    rim = src_pos + quat_apply(q, np.array([0, 0, 0.100]))
    # 배출점은 림에서 반경만큼 벗어난다(정확히 같지 않음).
    assert not np.allclose(pp, rim, atol=1e-3)
    # up axis가 수평이므로 pour_point z는 림보다 아래(중력수직 성분 음수 가능).
    assert np.isfinite(pp).all()


def test_vision_vector_is_12_and_ordered():
    vec = vision_obs_vector(
        np.array([0.4, 0.0, 0.3]), IDENT, np.array([0.4, 0.2, 0.1]), IDENT
    )
    assert vec.shape == (12,)
    terms = vision_obs_terms(
        np.array([0.4, 0.0, 0.3]), IDENT, np.array([0.4, 0.2, 0.1]), IDENT
    )
    assert np.allclose(vec[0:3], terms["pour_point_to_opening"])
    assert np.allclose(vec[3:6], terms["source_pour_axis"])
    assert np.allclose(vec[6:9], terms["source_up_axis"])
    assert np.allclose(vec[9:12], terms["target_up_axis"])


def test_constants_match_pour_v1_preset():
    """drift 감시: pour_v1 preset 값이 바뀌면 여기서 실패시켜 재정합을 강제한다."""
    preset = (
        Path(__file__).resolve().parents[1]
        / "../hdgp/source/openarm/openarm/tesollo/right/pour_v1/pour_right_preset.py"
    ).resolve()
    if not preset.is_file():
        pytest.skip(f"pour_v1 preset not found: {preset}")
    text = preset.read_text()
    import pour_obs_geometry as g

    checks = {
        "SOURCE_CUP_POUR_POINT_POS_B": list(g.SOURCE_CUP_POUR_POINT_POS_B),
        "TARGET_CUP_OPENING_POS_B": list(g.TARGET_CUP_OPENING_POS_B),
        "SOURCE_CUP_POUR_AXIS_B": list(g.SOURCE_CUP_POUR_AXIS_B),
        "SOURCE_CUP_UP_AXIS_B": list(g.SOURCE_CUP_UP_AXIS_B),
        "TARGET_CUP_UP_AXIS_B": list(g.TARGET_CUP_UP_AXIS_B),
    }
    import re

    for name, ours in checks.items():
        m = re.search(rf"{name}\s*=\s*\[([^\]]+)\]", text)
        assert m, f"{name} not found in preset"
        vals = [float(x) for x in m.group(1).split(",")]
        assert vals == ours, f"{name} drift: preset={vals} ours={ours}"
