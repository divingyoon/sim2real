"""M2 — obs_segments / obs_core / fk_numpy 단위 테스트.

순열은 이름으로 만든다(섞으면 obs 가 바뀐다 — 148 mm 사고). rot6d 는 좌 행우선·우 열 스택.
스테일이면 obs 를 내지 않는다. NaN·차원 불일치는 에러.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import fk_numpy, obs_core, obs_segments, sources

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
ROBOTS = SIM2REAL / "policy_control/config/robots"
LEFT_CONTRACT = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_CONTRACT = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"

needs_left = pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")
needs_right = pytest.mark.skipif(not RIGHT_CONTRACT.exists(), reason="right contract 없음")

HAND_PROFILE = [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)]
TIPS = tuple(f"r_hl_{f}_tip" for f in ("thumb", "index", "middle", "ring", "pinky"))
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _state(**kw) -> sources.RobotState:
    base = dict(arm_q=np.zeros(7), arm_qd=np.zeros(7), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"),
                ee_q=np.full(2, 0.044), ee_qd=np.zeros(2), object_pos=np.array([0.4, 0.2, 0.3]),
                object_quat=IDENT, tip_force=None, tip_names=(), head=None, decoder_target=None,
                stamps={}, stale=(), missing=())
    base.update(kw)
    return sources.RobotState(**base)


# ------------------------------------------------------------------ registry
@needs_left
@needs_right
def test_registry_covers_every_builder_in_both_contracts():
    names = set()
    for p in (LEFT_CONTRACT, RIGHT_CONTRACT):
        names |= {s.builder for s in C.load_contract(p).obs.segments}
    missing = names - set(obs_segments.BUILDERS)
    assert not missing, missing


def test_unknown_builder_is_an_error():
    with pytest.raises(obs_segments.ObsBuildError, match="nope"):
        obs_segments.builder("nope")


# ------------------------------------------------------------------ FK
@pytest.fixture(scope="module")
def left_fk():
    c = C.load_contract(LEFT_CONTRACT)
    return fk_numpy.make_fk(c, rl_ws=RL_WS)


@needs_left
def test_left_fk_matches_recorded_sim_bodies(left_fk, fixtures_dir):
    import json
    L = json.load(open(fixtures_dir / "left_v2B25_obs_layout.json"))["state"]
    jn, jp = L["joint_names"], np.array(L["joint_pos"])
    bn, bp = L["body_names"], np.array(L["body_pos_env_local"]).reshape(-1, 3)
    q = np.array([jp[jn.index(f"l_aj_{i}")] for i in range(1, 8)])
    g = np.array([jp[jn.index("l_hj_gripper_1")], jp[jn.index("l_hj_gripper_2")]])
    pose = left_fk.palm_pose(q, g)
    assert pose.palm_body == "l_hl_gripper_base"
    np.testing.assert_allclose(pose.palm_pos, bp[bn.index("l_hl_gripper_base")], atol=1e-5)
    np.testing.assert_allclose(pose.extra["tcp"], bp[bn.index("l_hl_gripper_tcp")], atol=1e-5)
    tips = dict(zip(pose.tip_names, pose.tips))
    np.testing.assert_allclose(tips["l_hl_gripper_left_finger"], bp[bn.index("l_hl_gripper_left_finger")], atol=1e-5)
    np.testing.assert_allclose(tips["l_hl_gripper_right_finger"], bp[bn.index("l_hl_gripper_right_finger")], atol=1e-5)


def test_fabric_fk_adapter_converts_euler_zyx_and_orders_tips():
    calls = []

    def palm6(q):
        calls.append(np.array(q))
        return np.array([0.3, -0.2, 0.4, 1.5707963267948966, 0.0, 1.5707963267948966])

    def tips(q):
        return np.arange(15.0).reshape(5, 3)

    fk = fk_numpy.FabricFK(palm6, tips, tip_names=TIPS, palm_body="palm")
    pose = fk.palm_pose(np.zeros(7), np.ones(20))
    assert calls[0].shape == (27,) and calls[0][7] == 1.0
    np.testing.assert_allclose(pose.palm_pos, [0.3, -0.2, 0.4])
    assert np.linalg.norm(pose.palm_quat) == pytest.approx(1.0)
    from grasp_s2r_obs_builder import rot6d_columns
    from grasp_s2r_core import _rot_euler_zyx
    R = _rot_euler_zyx([1.5707963267948966, 0.0, 1.5707963267948966])
    np.testing.assert_allclose(rot6d_columns(pose.palm_quat), np.concatenate([R[:, 0], R[:, 1]]), atol=1e-12)
    assert pose.tip_names == TIPS and pose.tips.shape == (5, 3)


def test_make_fk_fabric_kind_requires_callables():
    c = C.load_contract(RIGHT_CONTRACT)
    with pytest.raises(fk_numpy.FKError):
        fk_numpy.make_fk(c, rl_ws=RL_WS)


# ------------------------------------------------------------------ left core
@pytest.fixture
def left_core(left_fk):
    c = C.load_contract(LEFT_CONTRACT)
    cfg = sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml")
    return obs_core.ObsCore(c, cfg, left_fk)


@needs_left
def test_left_tick_shape_segments_and_home_relative(left_core):
    c = left_core.contract
    st = _state(arm_q=np.array(c.pd.home_arm))
    left_core.reset(st)
    out = left_core.tick(st, last_action=None)
    assert out.obs.shape == (49,) and out.valid and out.seq == 0
    seg = obs_core.split_segments(out.obs, c)
    np.testing.assert_allclose(seg["joint_pos"], 0.0, atol=1e-12)        # 홈 = 0 (mdp.joint_pos_rel)
    np.testing.assert_allclose(seg["actions"], 0.0)
    np.testing.assert_allclose(seg["target_object_position"], c.obs.segment("target_object_position").params["goal"])
    np.testing.assert_allclose(seg["object_position"], st.object_pos)
    assert seg["gripper_gate"][0] == 0.0
    assert seg["cup_upright"][0] == pytest.approx(1.0)
    assert out.aux["gate_open"] is False
    out2 = left_core.tick(st, last_action=np.full(7, 0.5))
    assert out2.seq == 1
    np.testing.assert_allclose(obs_core.split_segments(out2.obs, c)["actions"], 0.5)


@needs_left
def test_left_rot6d_is_row_major_interleaved(left_core, left_fk):
    c = left_core.contract
    st = _state(arm_q=np.array(c.pd.home_arm) + 0.1)
    left_core.reset(st)
    seg = obs_core.split_segments(left_core.tick(st, None).obs, c)
    from left_obs_builder import quat_to_matrix
    R = quat_to_matrix(left_fk.palm_pose(st.arm_q, st.ee_q).palm_quat)
    np.testing.assert_allclose(seg["palm_rot"], R[:, :2].reshape(-1), atol=1e-12)
    assert not np.allclose(seg["palm_rot"], np.concatenate([R[:, 0], R[:, 1]]))


@needs_left
def test_left_stale_source_makes_obs_invalid(left_core):
    st = _state(arm_q=np.array(left_core.contract.pd.home_arm))
    left_core.reset(st)
    out = left_core.tick(dataclasses.replace(st, stale=("arm",)), None)
    assert out.valid is False and any("arm" in r for r in out.reasons)
    assert out.seq == 0
    out2 = left_core.tick(dataclasses.replace(st, missing=("object",)), None)
    assert out2.valid is False and out2.seq == 1               # seq 는 미발행 tick 에도 증가


@needs_left
def test_left_nan_and_wrong_action_dim_raise(left_core):
    st = _state(arm_q=np.array(left_core.contract.pd.home_arm))
    left_core.reset(st)
    bad_q = st.arm_q.copy()
    bad_q[2] = np.nan
    with pytest.raises(obs_core.ObsError):
        left_core.tick(dataclasses.replace(st, arm_q=bad_q), None)
    with pytest.raises(obs_core.ObsError):
        left_core.tick(st, last_action=np.zeros(6))


@needs_left
def test_left_object_modes(left_fk):
    c = C.load_contract(LEFT_CONTRACT)
    cfg = sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml")
    home = np.array(c.pd.home_arm)
    st0 = _state(arm_q=home, object_pos=np.array([0.40, 0.20, 0.30]))
    st1 = _state(arm_q=home, object_pos=np.array([0.45, 0.20, 0.30]))
    for mode, want in (("latch_at_reset", st0.object_pos), ("live", st1.object_pos)):
        core = obs_core.ObsCore(c, sources.with_object_mode(cfg, mode), left_fk)
        core.reset(st0)
        seg = obs_core.split_segments(core.tick(st1, None).obs, c)
        np.testing.assert_allclose(seg["object_position"], want)
    # attach_after_gate: 게이트가 닫혀 있는 동안은 latch 와 같다
    core = obs_core.ObsCore(c, cfg, left_fk)
    core.reset(st0)
    seg = obs_core.split_segments(core.tick(st1, None).obs, c)
    np.testing.assert_allclose(seg["object_position"], st0.object_pos)


@needs_left
def test_left_attach_moves_object_with_gripper_after_gate(left_fk):
    """게이트가 열린 순간 컵을 턱 프레임에 굳히고, 이후 손목이 움직이면 같이 간다."""
    c = C.load_contract(LEFT_CONTRACT)
    cfg = sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml")
    core = obs_core.ObsCore(c, cfg, left_fk)
    home = np.array(c.pd.home_arm)
    st = _state(arm_q=home)
    # 컵을 턱 사이 파지 위치에 놓고 그 자리에서 에피소드를 시작해 게이트를 연다
    pose = left_fk.palm_pose(home, st.ee_q)
    jaw_mid = pose.tips.mean(axis=0)
    from left_obs_builder import quat_to_matrix
    approach = quat_to_matrix(pose.palm_quat)[:, 2]
    band = c.obs.segment("gripper_gate").params["band_axis"]
    pad = c.obs.segment("gripper_gate").params["pad_offset"]
    cup = jaw_mid + approach * pad - np.array([0.0, 0.0, 0.5 * (band[0] + band[1])])
    st_grasp = dataclasses.replace(st, object_pos=cup)
    core.reset(st_grasp)
    out = core.tick(st_grasp, None)
    assert out.aux["gate_open"] is True
    seg0 = obs_core.split_segments(out.obs, c)
    # 팔이 움직이면(그리퍼 위치 변화) 컵도 같은 강체 변환을 따른다
    st_moved = dataclasses.replace(st_grasp, arm_q=home + np.array([0.0, -0.1, 0, 0.1, 0, 0, 0]))
    out2 = core.tick(st_moved, None)
    seg1 = obs_core.split_segments(out2.obs, c)
    pose2 = left_fk.palm_pose(st_moved.arm_q, st_moved.ee_q)
    delta_jaw = pose2.tips.mean(axis=0) - jaw_mid
    assert np.linalg.norm(delta_jaw) > 0.01
    assert np.linalg.norm(seg1["object_position"] - seg0["object_position"]) > 0.005
    assert out2.aux["attached"] is True


# ------------------------------------------------------------------ right core
class _StubFK:
    """palm 6D + tips 를 고정값으로 주는 fabric FK 대역."""

    def __init__(self):
        self.palm6 = np.array([0.3, -0.3, 0.4, 1.5707963267948966, 0.0, 1.5707963267948966])
        self.tips = np.array([[0.4, -0.3, 0.4], [0.42, -0.28, 0.41], [0.43, -0.3, 0.42],
                              [0.42, -0.32, 0.41], [0.4, -0.34, 0.4]])
        self.last_q = None

    def palm(self, q):
        self.last_q = np.array(q)
        return self.palm6

    def tip(self, q):
        return self.tips


def _right_state(c, hand_prof=None, hand_names=None):
    hand_prof = np.array(c.action.hand.params["open_pose"]) if hand_prof is None else hand_prof
    names = tuple(HAND_PROFILE if hand_names is None else hand_names)
    return _state(arm_q=np.array(c.pd.home_arm), ee_names=names, ee_q=hand_prof, ee_qd=np.zeros(20),
                  object_pos=np.array([0.35, -0.17, 0.28]), tip_force=np.zeros((5, 3)), tip_names=TIPS)


@pytest.fixture
def right_parts():
    c = C.load_contract(RIGHT_CONTRACT)
    cfg = sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml")
    stub = _StubFK()
    fk = fk_numpy.make_fk(c, rl_ws=RL_WS, palm_pose_fn=stub.palm, tips_fn=stub.tip)
    return c, cfg, stub, obs_core.ObsCore(c, cfg, fk)


@needs_right
def test_right_tick_layout_and_joint_err_open_pose_before_first_target(right_parts):
    c, cfg, stub, core = right_parts
    st = _right_state(c)
    core.reset(st)
    out = core.tick(st, None)
    assert out.obs.shape == (155,) and out.valid
    seg = obs_core.split_segments(out.obs, c)
    np.testing.assert_allclose(seg["arm_q"], c.pd.home_arm)
    np.testing.assert_allclose(seg["joint_err"], 0.0)                 # 목표 없음 → open pose == 실측
    np.testing.assert_allclose(seg["palm_pos"], stub.palm6[:3])
    np.testing.assert_allclose(seg["palm_to_obj"], st.object_pos - stub.palm6[:3])
    np.testing.assert_allclose(seg["tips_rel_palm"], (stub.tips - stub.palm6[:3]).reshape(-1))
    np.testing.assert_allclose(seg["obj_to_tips"], (stub.tips - st.object_pos).reshape(-1))
    goal = st.object_pos + np.array(c.obs.segment("goal_rel").params["goal_offset"])
    np.testing.assert_allclose(seg["goal_rel"], goal - st.object_pos)
    # fabric 은 자기 관절 순서(계약 fabric.joint_order)로 받는다 — 프로필 순과 같은지 이름으로 확인
    hand_fab = [stub.last_q[7 + i] for i in range(20)]
    want = [st.ee_q[HAND_PROFILE.index(n)] for n in c.fabric.joint_order[7:]]
    np.testing.assert_allclose(hand_fab, want)


@needs_right
def test_right_hand_q_is_isaac_dof_order_built_by_name(right_parts):
    c, cfg, stub, core = right_parts
    hand = np.linspace(-1.0, 1.0, 20)
    st = _right_state(c, hand_prof=hand)
    core.reset(st)
    seg = obs_core.split_segments(core.tick(st, None).obs, c)
    want = [hand[HAND_PROFILE.index(n)] for n in c.obs.joint_orders["hand_obs"]]
    np.testing.assert_allclose(seg["hand_q"], want)
    assert not np.allclose(seg["hand_q"], hand)                        # 프로필 순 ≠ DOF 순


@needs_right
def test_right_scrambled_hand_names_change_obs_negative(right_parts):
    """148 mm 사고 회귀: 이름을 섞어 넣으면(=값이 다른 관절에 붙으면) obs 가 달라져야 한다."""
    c, cfg, stub, core = right_parts
    hand = np.linspace(-1.0, 1.0, 20)
    st = _right_state(c, hand_prof=hand)
    core.reset(st)
    a = core.tick(st, None).obs
    scrambled = list(HAND_PROFILE)
    scrambled[0], scrambled[5] = scrambled[5], scrambled[0]           # thumb_1 ↔ index_2
    st2 = _right_state(c, hand_prof=hand, hand_names=scrambled)
    core.reset(st2)
    b = core.tick(st2, None).obs
    seg_a, seg_b = obs_core.split_segments(a, c), obs_core.split_segments(b, c)
    assert not np.allclose(seg_a["hand_q"], seg_b["hand_q"])
    assert not np.allclose(seg_a["joint_err"], seg_b["joint_err"])


@needs_right
def test_right_joint_err_uses_decoder_target_in_profile_order(right_parts):
    c, cfg, stub, core = right_parts
    st = _right_state(c)
    core.reset(st)
    tgt = st.ee_q + 0.6
    seg = obs_core.split_segments(core.tick(st, None, decoder_target=tgt).obs, c)
    np.testing.assert_allclose(seg["joint_err"], 0.5)                  # 0.6 / 1.2
    seg2 = obs_core.split_segments(core.tick(st, None, decoder_target=st.ee_q + 5.0).obs, c)
    np.testing.assert_allclose(seg2["joint_err"], 1.0)                 # ±1 클램프
    with pytest.raises(obs_core.ObsError):
        core.tick(st, None, decoder_target=np.zeros(19))


@needs_right
def test_right_rot6d_is_column_stacked(right_parts):
    c, cfg, stub, core = right_parts
    st = _right_state(c)
    core.reset(st)
    seg = obs_core.split_segments(core.tick(st, None).obs, c)
    from grasp_s2r_core import _rot_euler_zyx
    R = _rot_euler_zyx(stub.palm6[3:])
    np.testing.assert_allclose(seg["palm_ax"], np.concatenate([R[:, 0], R[:, 1]]), atol=1e-9)


@needs_right
def test_right_tip_force_is_local_passthrough_scaled(right_parts):
    c, cfg, stub, core = right_parts
    st = _right_state(c)
    force = np.arange(15.0).reshape(5, 3)
    st = dataclasses.replace(st, tip_force=force)
    core.reset(st)
    seg = obs_core.split_segments(core.tick(st, None).obs, c)
    np.testing.assert_allclose(seg["tip_force"], (force / 10.0).reshape(-1))
    # tip 순서는 이름으로 — 순서를 뒤집어 주면 다시 계약 순으로 돌아온다
    st_rev = dataclasses.replace(st, tip_force=force[::-1].copy(), tip_names=TIPS[::-1])
    seg2 = obs_core.split_segments(core.tick(st_rev, None).obs, c)
    np.testing.assert_allclose(seg2["tip_force"], seg["tip_force"])


@needs_right
def test_right_missing_tip_force_is_invalid_not_zero(right_parts):
    c, cfg, stub, core = right_parts
    st = dataclasses.replace(_right_state(c), tip_force=None, missing=("tip_force",))
    core.reset(st)
    out = core.tick(st, None)
    assert out.valid is False and any("tip_force" in r for r in out.reasons)


# ------------------------------------------------------------------ 09.06 side + urdf_chain FK (dg5f-m 자산)
DG5FM_CONTRACT = SIM2REAL / "logs/policy/right_g1/deploy_contract.dg5f-m.json"
ASSET_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
needs_dg5fm = pytest.mark.skipif(not (DG5FM_CONTRACT.exists() and ASSET_CONTRACT.exists()), reason="dg5f-m 계약 없음")


@pytest.fixture(scope="module")
def dg5fm():
    c = C.load_contract(DG5FM_CONTRACT)
    return c, fk_numpy.make_fk(c, rl_ws=RL_WS, kind="urdf_chain")


@needs_dg5fm
@pytest.mark.parametrize("robot_yaml", ["dg5f_m_bi_fake.yaml", "dg5f_m_right_fake.yaml"])
def test_urdf_chain_core_builds_right_obs_from_bimanual_and_single_yaml(dg5fm, robot_yaml):
    c, fk = dg5fm
    cfg = sources.load_robot_cfg(ROBOTS / robot_yaml)
    core = obs_core.ObsCore(c, cfg, fk, side="right")
    assert core.side.side == "right" and set(core.cfg.sources) >= {"arm", "ee", "tip_force", "object"}
    assert fk.palm_body == "r_hl_palm" == core.side.palm_body
    st = _right_state(c)
    core.reset(st)
    out = core.tick(st, None)
    assert out.valid and out.obs.shape == (155,)
    seg = obs_core.split_segments(out.obs, c)
    pose = fk.palm_pose(st.arm_q, st.ee_q)                     # 손은 프로필 순 = fk.hand_joints 순
    np.testing.assert_allclose(seg["palm_pos"], pose.palm_pos)
    np.testing.assert_allclose(seg["tips_rel_palm"], (pose.tips - pose.palm_pos).reshape(-1))
    from grasp_s2r_obs_builder import rot6d_columns
    np.testing.assert_allclose(seg["palm_ax"], rot6d_columns(pose.palm_quat), atol=1e-12)
    R = pose.extra["palm_rot"]
    np.testing.assert_allclose(seg["palm_ax"], np.concatenate([R[:, 0], R[:, 1]]), atol=1e-12)
    np.testing.assert_allclose(seg["joint_err"], 0.0)
    # 기본 side = primary(right)
    assert obs_core.ObsCore(c, cfg, fk).side.side == "right"


@needs_dg5fm
def test_obs_core_refuses_wrong_side_control_only_and_foreign_palm(dg5fm):
    c, fk = dg5fm
    bi = sources.load_robot_cfg(ROBOTS / "dg5f_m_bi_fake.yaml")
    with pytest.raises(obs_core.ObsError, match="no side"):
        obs_core.ObsCore(c, bi, fk, side="middle")
    with pytest.raises(obs_core.ObsError, match="side 'left'"):        # g1 계약에는 우팔만 있다
        obs_core.ObsCore(c, bi, fk, side="left")
    with pytest.raises(obs_core.ObsError, match="left 팔인데 right 팔을 요청"):     # 한 팔 yaml 이 다른 팔
        obs_core.ObsCore(c, sources.load_robot_cfg(ROBOTS / "dg5f_m_left_fake.yaml"), fk, side="right")
    ctl = C.load_contract(ASSET_CONTRACT)
    with pytest.raises(obs_core.ObsError, match="control-only"):
        obs_core.ObsCore(ctl, bi, fk_numpy.make_fk(ctl, rl_ws=RL_WS), side="right")
    left_fk = fk_numpy.make_fk(ctl, rl_ws=RL_WS, side="left")            # 왼손 FK 를 우팔 계약에 꽂으면 거부
    with pytest.raises(obs_core.ObsError, match="palm_body"):
        obs_core.ObsCore(c, bi, left_fk, side="right")
    old_gripper = sources.load_robot_cfg(ROBOTS / "left_gripper_fake.yaml")
    with pytest.raises(obs_core.ObsError):
        obs_core.ObsCore(C.load_contract(LEFT_CONTRACT), old_gripper, fk, side="left")   # 그리퍼 계약 + DG-5F FK


@needs_dg5fm
def test_segment_body_param_is_the_run_name_when_an_asset_is_bound(dg5fm):
    """dg5f-m 계약의 세그먼트 `body: palm` 은 런 dump 의 옛 이름 — 자산 바인딩이 있으면 side.palm_body 만 본다."""
    c, fk = dg5fm
    assert c.asset is not None and c.obs.segment("palm_pos").params["body"] == "palm"
    assert c.side("right").palm_body == "r_hl_palm"
    seg = c.obs.segment("palm_pos")
    pose = fk.palm_pose(np.zeros(7), np.zeros(20))
    inp = obs_segments.ObsInputs(contract=c, state=None, fk=pose, object_pos=np.zeros(3), object_quat=IDENT,
                                 goal=np.zeros(3), last_action=np.zeros(21), gate=None, decoder_target=None,
                                 palm_body="r_hl_palm", hand_joints=tuple(HAND_PROFILE))
    np.testing.assert_allclose(obs_segments.body_pos(inp, seg), pose.palm_pos)
    bad = dataclasses.replace(inp, palm_body="l_hl_palm")
    with pytest.raises(obs_segments.ObsBuildError, match="palm_body"):
        obs_segments.body_pos(bad, seg)
    # 자산 바인딩이 없는 v1 계약은 세그먼트 body 와 FK 가 같아야 한다
    v1 = dataclasses.replace(c, asset=None)
    with pytest.raises(obs_segments.ObsBuildError, match="계약 body"):
        obs_segments.body_pos(dataclasses.replace(inp, contract=v1), seg)
