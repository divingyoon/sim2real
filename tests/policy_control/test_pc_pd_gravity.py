"""M5 — pd_gravity: off | integral_droop | model_tau_ff.

model_tau_ff 는 gravity_comp_node 와 **같은 수학**(robot_control.kinematics 의
chain_from_urdf + with_payload + gravity_torque, 실측 q, per-joint scale, cap)이어야 한다 —
테스트가 그 수학을 직접 다시 계산해 대조하고, 동결 스냅샷(derived/)으로 URDF 드리프트를 잡는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import pd_gravity as G
from policy_control import pd_law as L

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
CONFIG = SIM2REAL / "policy_control" / "config"
DERIVED = SIM2REAL / "tests/fixtures/policy_control/derived"
LEFT_CONTRACT = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_CONTRACT = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
URDF = RL_WS / "urdf/generated/rl/openarm_tesollo_sensor_rl.urdf"
ARM_R = [f"r_aj_{i}" for i in range(1, 8)]
SCALE = [1.1, 1.1, 1.1, 1.0, 1.1, 0.9, 1.1]
PAYLOAD = [0.8350, -0.00450, -0.01723, 0.22147]

pytestmark = pytest.mark.unit
needs_right = pytest.mark.skipif(not RIGHT_CONTRACT.exists(), reason="right_g1 contract 없음")
needs_left = pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")


def _node_math(q: np.ndarray, scale=SCALE, payload=PAYLOAD) -> np.ndarray:
    """gravity_comp_node.py 의 계산을 그대로(재구현이 아니라 같은 라이브러리 호출)."""
    from robot_control.kinematics import chain_from_urdf, with_payload

    chain = chain_from_urdf(URDF.read_text(), ARM_R, "r_hl_palm_ee")
    if payload is not None:
        chain = with_payload(chain, payload[0], payload[1:])
    return chain.gravity_torque(q) * np.asarray(scale)


def _model_cfg(**over) -> G.ModelGravityCfg:
    base = dict(urdf=URDF, tip_link="r_hl_palm_ee", joints=tuple(ARM_R), scale=tuple(SCALE),
                payload=tuple(PAYLOAD), cap_nm=20.0)
    base.update(over)
    return G.ModelGravityCfg(**base)


# ---------------------------------------------------------------- model_tau_ff
@needs_right
def test_home_pose_torque_matches_gravity_comp_node_math():
    contract = C.load_contract(RIGHT_CONTRACT)
    home = np.asarray(contract.pd.home_arm)
    model = G.build_model_gravity(_model_cfg())
    tau = model(home)
    assert tau.shape == (7,)
    assert np.allclose(tau, _node_math(home), atol=1e-9)
    assert np.all(np.abs(tau) > 0.0)                              # 홈에서 0 인 관절은 없다


@needs_right
def test_home_pose_torque_matches_frozen_snapshot():
    contract = C.load_contract(RIGHT_CONTRACT)
    snap = json.loads((DERIVED / "right_g1_home_gravity_tau.json").read_text())
    assert snap["joints"] == ARM_R and snap["scale"] == SCALE and snap["payload"] == PAYLOAD
    assert np.allclose(snap["home_q"], contract.pd.home_arm)
    tau = G.build_model_gravity(_model_cfg())(np.asarray(snap["home_q"]))
    assert np.allclose(tau, snap["tau_nm"], atol=1e-6)


def test_model_uses_measured_q_not_a_cached_pose():
    model = G.build_model_gravity(_model_cfg())
    a = model(np.zeros(7))
    b = model(np.array([0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert not np.allclose(a, b)
    assert np.allclose(a, _node_math(np.zeros(7)))


def test_model_scale_and_payload_change_torque():
    q = np.array([0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0])
    base = G.build_model_gravity(_model_cfg(scale=(1.0,) * 7, payload=None))(q)
    scaled = G.build_model_gravity(_model_cfg(scale=(2.0,) * 7, payload=None))(q)
    loaded = G.build_model_gravity(_model_cfg(scale=(1.0,) * 7))(q)
    assert np.allclose(scaled, 2.0 * base)
    assert np.any(np.abs(loaded) > np.abs(base) + 1e-6)
    assert np.allclose(base, _node_math(q, scale=[1.0] * 7, payload=None))


def test_model_cap_detection():
    model = G.build_model_gravity(_model_cfg(cap_nm=20.0))
    q = np.array([0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert not model.over_cap(model(q))
    hot = G.build_model_gravity(_model_cfg(scale=(50.0,) * 7, cap_nm=20.0))
    assert hot.over_cap(hot(q))
    assert not hot.over_cap(np.zeros(7))


def test_model_rejects_bad_inputs():
    with pytest.raises(ValueError):
        G.build_model_gravity(_model_cfg(scale=(1.0,) * 6))
    with pytest.raises(ValueError):
        G.build_model_gravity(_model_cfg(payload=(1.0, 0.0)))
    with pytest.raises(ValueError):
        G.build_model_gravity(_model_cfg(cap_nm=0.0))
    with pytest.raises(G.GravityConfigError):
        G.build_model_gravity(_model_cfg(urdf=URDF.with_name("nope.urdf")))
    with pytest.raises(G.GravityConfigError):
        G.build_model_gravity(_model_cfg(joints=tuple(ARM_R[:6]) + ("r_aj_99",)))
    model = G.build_model_gravity(_model_cfg())
    with pytest.raises(ValueError):
        model(np.zeros(6))
    with pytest.raises(ValueError):
        model(np.full(7, np.nan))


# ---------------------------------------------------------------- mode resolution / exclusion
@pytest.mark.parametrize("cfg_mode,contract_mode,ok", [
    ("off", "off", True), ("integral_droop", "integral_droop", True),
    ("model_tau_ff", "model_tau_ff", True),
    ("integral_droop", "model_tau_ff", False), ("model_tau_ff", "integral_droop", False),
    ("off", "model_tau_ff", False), ("model_tau_ff", "off", False),
])
def test_gravity_conflict(cfg_mode, contract_mode, ok):
    msg = G.gravity_conflict(cfg_mode, contract_mode)
    assert (msg is None) == ok
    if not ok:
        assert cfg_mode in msg and contract_mode in msg


def test_gravity_conflict_rejects_unknown_mode():
    with pytest.raises(ValueError):
        G.gravity_conflict("warp", "off")


@needs_left
def test_make_gravity_left_droop_returns_zero_tau_and_contract_params():
    contract = C.load_contract(LEFT_CONTRACT)
    cfg = L.load_pd_config(CONFIG / "pd_left.yaml")
    fn = G.make_gravity(cfg.gravity, contract)
    assert np.all(fn(np.asarray(contract.pd.home_arm)) == 0.0)   # droop 은 τ 가 아니라 q 에 얹힌다
    gain, limit = G.droop_params(contract)
    assert gain == 0.05 and np.allclose(limit, contract.pd.gravity.limit)


@needs_right
def test_make_gravity_right_model_from_yaml():
    contract = C.load_contract(RIGHT_CONTRACT)
    cfg = L.load_pd_config(CONFIG / "pd_right.yaml")
    fn = G.make_gravity(cfg.gravity, contract)
    home = np.asarray(contract.pd.home_arm)
    assert np.allclose(fn(home), _node_math(home), atol=1e-9)
    with pytest.raises(G.GravityConfigError):
        G.droop_params(contract)                                 # 모델 계약에 droop 파라미터 없음


@needs_left
def test_make_gravity_refuses_conflicting_modes():
    left = C.load_contract(LEFT_CONTRACT)
    right_cfg = L.load_pd_config(CONFIG / "pd_right.yaml")
    with pytest.raises(G.GravityConfigError):
        G.make_gravity(right_cfg.gravity, left)


def test_make_gravity_off():
    fn = G.make_gravity(G.GravityBlock(mode="off"), contract=None, n_joints=7)
    assert np.all(fn(np.ones(7)) == 0.0)
    with pytest.raises(ValueError):
        fn(np.ones(6))


def test_gravity_block_requires_model_fields():
    with pytest.raises(G.GravityConfigError):
        G.GravityBlock(mode="model_tau_ff")                      # urdf/tip_link/scale/cap 없음
    with pytest.raises(G.GravityConfigError):
        G.GravityBlock(mode="integral_droop", urdf=URDF)         # droop 인데 모델 필드
    with pytest.raises(G.GravityConfigError):
        G.GravityBlock(mode="sideways")


def test_make_gravity_needs_contract_or_joint_count():
    with pytest.raises(G.GravityConfigError):
        G.make_gravity(G.GravityBlock(mode="off"), contract=None)
    with pytest.raises(G.GravityConfigError):
        G.make_gravity(G.GravityBlock(mode="model_tau_ff", urdf=URDF, tip_link="r_hl_palm_ee",
                                      scale=tuple(SCALE), cap_nm=20.0), contract=None, n_joints=7)


def test_droop_params_rejects_bad_contract_values():
    from dataclasses import replace
    contract = C.load_contract(LEFT_CONTRACT)
    bad = replace(contract, pd=replace(contract.pd, gravity=replace(contract.pd.gravity, gain=0.0)))
    with pytest.raises(G.GravityConfigError):
        G.droop_params(bad)


# ---------------------------------------------------------------- 09.06 양팔(asset 계약) — 팔별 tip_link/payload 매핑
ASSET_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
ASSET_URDF = RL_WS / "hdgp/assets/robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl.urdf"
needs_asset = pytest.mark.skipif(not ASSET_CONTRACT.exists(), reason="asset contract 없음")


def _sided_block(**over) -> G.GravityBlock:
    base = dict(mode="model_tau_ff", urdf=ASSET_URDF, tip_link={"left": "l_hl_palm_ee", "right": "r_hl_palm_ee"},
                scale=(1.0,) * 7, payload={"left": (1.0, 0.0, 0.0, 0.0), "right": None}, cap_nm=20.0)
    base.update(over)
    return G.GravityBlock(**base)


def test_block_for_side_resolves_mappings_and_passes_plain_through():
    blk = _sided_block()
    assert blk.sided
    left = G.block_for_side(blk, "left")
    assert left.tip_link == "l_hl_palm_ee" and left.payload == (1.0, 0.0, 0.0, 0.0) and left.scale == (1.0,) * 7
    right = G.block_for_side(blk, "right")
    assert right.tip_link == "r_hl_palm_ee" and right.payload is None and not right.sided
    plain = G.GravityBlock(mode="off")
    assert G.block_for_side(plain, "left") == plain and G.block_for_side(plain, None) == plain


def test_block_for_side_refuses_missing_side_or_no_side():
    blk = _sided_block(tip_link={"left": "l_hl_palm_ee"})
    with pytest.raises(G.GravityConfigError):
        G.block_for_side(blk, "right")
    with pytest.raises(G.GravityConfigError):
        G.block_for_side(blk, None)
    with pytest.raises(G.GravityConfigError):
        G.GravityBlock(mode="model_tau_ff", urdf=ASSET_URDF, tip_link={"up": "x"}, scale=(1.0,) * 7, cap_nm=1.0)


@needs_asset
def test_make_gravity_per_side_from_dg5f_m_yaml_matches_chain_math():
    from robot_control.kinematics import chain_from_urdf, with_payload

    contract = C.load_contract(ASSET_CONTRACT)
    cfg = L.load_pd_config(CONFIG / "pd_dg5f_m.yaml")
    q = np.array([0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0])
    for side in ("left", "right"):
        blk = G.block_for_side(cfg.gravity, side)
        fn = G.make_gravity(cfg.gravity, contract, side=side)
        chain = chain_from_urdf(ASSET_URDF.read_text(), list(contract.side(side).arm_joints), blk.tip_link)
        chain = with_payload(chain, blk.payload[0], list(blk.payload[1:]))
        assert np.allclose(fn(q), chain.gravity_torque(q) * np.asarray(blk.scale), atol=1e-9)
        assert np.all(np.abs(fn(q)[[1, 3]]) > 1.0)                    # 어깨·팔꿈치는 확실히 들린다
    with pytest.raises(G.GravityConfigError):
        G.make_gravity(cfg.gravity, contract)                          # 팔별 매핑인데 팔을 안 골랐다


@needs_asset
def test_dg5f_m_payload_equals_the_urdf_finger_lump():
    """pd_dg5f_m.yaml 의 payload = 체인이 못 싣는 손가락 링크(revolute 너머) 질량·무게중심(열린 손, palm_ee 프레임)."""
    from arm_inertia import _link_transforms, _subtree_links, parse_urdf

    contract = C.load_contract(ASSET_CONTRACT)
    cfg = L.load_pd_config(CONFIG / "pd_dg5f_m.yaml")
    model = parse_urdf(str(ASSET_URDF))
    for side in ("left", "right"):
        s = contract.side(side)
        p = side[0]
        q = {**{j: 0.0 for j in s.arm_joints}, **s.home_hand}
        tf = _link_transforms(model, q)
        links = set()
        for jn in model["joints"]:
            if jn.startswith(f"{p}_hj_"):
                links |= set(_subtree_links(model, jn))
        mass, com = 0.0, np.zeros(3)
        for ln in links:
            info = model["links"].get(ln)
            if info is None:
                continue
            R, t = tf[ln]
            mass += info["mass"]
            com += info["mass"] * (t + R @ info["com"])
        com /= mass
        R_tip, t_tip = tf[f"{p}_hl_palm_ee"]
        local = R_tip.T @ (com - t_tip)
        payload = G.block_for_side(cfg.gravity, side).payload
        assert abs(payload[0] - mass) < 5e-4 and np.allclose(payload[1:], local, atol=2e-5), (side, mass, local)


@needs_asset
def test_droop_params_and_contract_gravity_take_the_side():
    contract = C.load_contract(ASSET_CONTRACT)
    for side in ("left", "right"):
        g, joints = G.contract_gravity(contract, side)
        assert g.mode == "model_tau_ff" and joints == tuple(contract.side(side).arm_joints)
        with pytest.raises(G.GravityConfigError):
            G.droop_params(contract, side)
