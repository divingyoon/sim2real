"""hand_cmd 도구 — 순수 보간/미러/거부 규칙 (ROS 없음)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from policy_control import contract as C

SIM2REAL = Path(__file__).resolve().parents[2]
CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
needs_contract = pytest.mark.skipif(not CONTRACT.exists(), reason="자산 계약 없음")


def _load(name: str):
    path = SIM2REAL / f"policy_control/tools/{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def H():
    import sys
    sys.path.insert(0, str(SIM2REAL / "policy_control/tools"))     # `from palm_cmd import check_domain`
    return _load("hand_cmd")


@needs_contract
def test_close_interpolates_open_to_grip_and_left_mirrors(H):
    from openarm.tesollo.left.grasp_v1 import grasp_left_preset as P

    c = C.load_contract(CONTRACT)
    for side in ("right", "left"):
        s = c.side(side)
        grip = H.grip_pose(side, list(s.hand_joints))
        open_ = dict(s.home_hand)
        assert H.hand_targets(open_, grip, 0.0) == open_
        assert H.hand_targets(open_, grip, 1.0) == grip
        mid = H.hand_targets(open_, grip, 0.5)
        assert all(mid[j] == pytest.approx(0.5 * (open_[j] + grip[j])) for j in open_)
    r, l = H.grip_pose("right", list(c.side("right").hand_joints)), H.grip_pose("left", list(c.side("left").hand_joints))
    for (rj, v), sgn in zip(r.items(), P._HAND_SIGN):
        assert l["l" + rj[1:]] == pytest.approx(sgn * v)
    assert r["r_hj_index_2"] == pytest.approx(1.9) and l["l_hj_thumb_2"] == pytest.approx(1.57)


def test_overrides_and_validation(H):
    open_, grip = {"r_hj_a": 0.0, "r_hj_b": 1.0}, {"r_hj_a": 1.0, "r_hj_b": 2.0}
    out = H.hand_targets(open_, grip, 0.25, {"r_hj_b": 0.3})
    assert out == {"r_hj_a": 0.25, "r_hj_b": 0.3} and open_ == {"r_hj_a": 0.0, "r_hj_b": 1.0}
    with pytest.raises(ValueError):
        H.hand_targets(open_, grip, 1.5)
    with pytest.raises(ValueError):
        H.hand_targets(open_, grip, 0.0, {"nope": 1.0})
    assert H.parse_overrides(["r_hj_a=0.5", " r_hj_b = 1"]) == {"r_hj_a": 0.5, "r_hj_b": 1.0}
    with pytest.raises(ValueError):
        H.parse_overrides(["r_hj_a"])


@needs_contract
def test_cli_dry_run_and_domain_refusal(H, monkeypatch, capsys):
    monkeypatch.setenv("ROS_DOMAIN_ID", "0")
    assert H.main(["--contract", str(CONTRACT), "--side", "left", "--close", "0.5", "--dry-run"]) == 3
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    assert H.main(["--contract", str(CONTRACT), "--side", "left", "--close", "0.5", "--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert H.main(["--contract", str(CONTRACT), "--side", "right", "--close", "2", "--dry-run"]) == 2
