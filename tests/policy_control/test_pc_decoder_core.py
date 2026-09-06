"""M4 — ActionDecoder: 계약(palm convention · hand decoder)으로 액션을 해석한다.

오라클은 `scripts/` 의 순수 모듈(`gripper_left_palm_command` · `grasp_s2r_palm_command` ·
`grasp_s2r_synergy` · `left_policy_core.gripper_command`)이다 — 같은 액션을 넣으면 같은
출력이 나와야 한다. 케이지 캘리브·close_gate 는 `grasp_s2r_core.GraspS2RCore` 의 식이다.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import decoder_core as D

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"

needs_left = pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")
needs_right = pytest.mark.skipif(not RIGHT_JSON.exists(), reason="right_g1 contract 없음")

N_HAND = 20
SOFT_LIMITS = np.array([[-1.5708, 1.5708]] * N_HAND)


@pytest.fixture(scope="module")
def left() -> C.DeployContract:
    return C.load_contract(LEFT_JSON)


@pytest.fixture(scope="module")
def right() -> C.DeployContract:
    return C.load_contract(RIGHT_JSON)


def _actions(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.5, 1.5, size=(n, dim))


# ---------------------------------------------------------------- left: absolute_palm
@needs_left
def test_absolute_palm_matches_reference(left):
    from gripper_left_palm_command import PalmCommand, PalmCommandCfg

    p = left.action.palm
    ref = PalmCommand(PalmCommandCfg(
        box_lo=tuple(p.box_lo), box_hi=tuple(p.box_hi), euler_center=tuple(p.euler_center),
        max_pose_angle=p.max_pose_angle, pos_rate_limit=p.pos_rate_limit,
        rot_rate_limit=p.rot_rate_limit))
    dec = D.ActionDecoder(left)
    dec.reset()
    for a in _actions(60, left.policy.action_dim):
        out = dec.step(a, D.DecoderAux(gate_open=True))
        np.testing.assert_allclose(out.palm6, ref.step(a[:6]), rtol=0, atol=1e-12)
        assert out.hand_target is None and out.syn_vel is None
        assert out.gripper_cmd in (left.action.hand.params["open"], left.action.hand.params["close"])


@needs_left
def test_absolute_palm_first_command_is_unprimed(left):
    p = left.action.palm
    dec = D.ActionDecoder(left)
    dec.reset()
    a = np.ones(7)
    first = dec.step(a, D.DecoderAux(gate_open=True)).palm6
    # 첫 지령은 리미터 없이 박스 꼭짓점 / 회전 반폭까지 간다
    np.testing.assert_allclose(first[:3], p.box_hi, atol=1e-12)
    np.testing.assert_allclose(first[3:], np.asarray(p.euler_center) + p.max_pose_angle, atol=1e-12)
    # 두 번째부터는 스텝 이동량이 pos_rate_limit 로 묶인다
    second = dec.step(-a, D.DecoderAux(gate_open=True)).palm6
    assert np.linalg.norm(second[:3] - first[:3]) == pytest.approx(p.pos_rate_limit, abs=1e-12)
    assert np.linalg.norm(second[3:] - first[3:]) == pytest.approx(p.rot_rate_limit, abs=1e-12)
    # reset 하면 다시 unprimed
    dec.reset()
    again = dec.step(a, D.DecoderAux(gate_open=True)).palm6
    np.testing.assert_allclose(again, first, atol=1e-12)


@needs_left
def test_binary_gripper_gate_forces_open(left):
    from left_policy_core import gripper_command

    h = left.action.hand.params
    dec = D.ActionDecoder(left)
    dec.reset()
    close = np.array([0.0] * 6 + [-0.7])
    open_ = np.array([0.0] * 6 + [0.7])
    assert dec.step(close, D.DecoderAux(gate_open=False)).gripper_cmd == h["open"]
    assert dec.step(close, D.DecoderAux(gate_open=True)).gripper_cmd == h["close"]
    assert dec.step(open_, D.DecoderAux(gate_open=True)).gripper_cmd == h["open"]
    for a, g in ((close, True), (close, False), (open_, True), (open_, False)):
        assert dec.step(a, D.DecoderAux(gate_open=g)).gripper_cmd == gripper_command(a[6], g)


@needs_left
def test_binary_gripper_requires_gate(left):
    dec = D.ActionDecoder(left)
    dec.reset()
    with pytest.raises(D.DecoderError):
        dec.step(np.zeros(7), D.DecoderAux())


@needs_left
def test_wrong_action_dim_raises(left):
    dec = D.ActionDecoder(left)
    dec.reset()
    with pytest.raises(D.DecoderError):
        dec.step(np.zeros(6), D.DecoderAux(gate_open=True))


@needs_left
def test_unknown_convention_raises(left):
    bad = dataclasses.replace(
        left, action=dataclasses.replace(
            left.action, palm=dataclasses.replace(left.action.palm, convention="polar")))
    with pytest.raises(C.ContractError):
        D.ActionDecoder(bad)


# ---------------------------------------------------------------- right: delta_anchor
def _cage_inputs():
    """palm 원점·단위 회전, 엄지 (0.1,0,0) vs 나머지 (-0.1,0,0) → 중심 0 · 반경 0.1."""
    palm6 = np.zeros(6)
    tips = np.array([[0.1, 0, 0], [-0.1, 0, 0], [-0.1, 0, 0], [-0.1, 0, 0], [-0.1, 0, 0]], float)
    return palm6, tips


def _reset_right(dec: D.ActionDecoder, right: C.DeployContract, obj):
    palm6, tips = _cage_inputs()
    dec.reset(object_pos=obj, hand_q=np.asarray(right.action.hand.params["open_pose"]),
              palm6=palm6, tips=tips)


@needs_right
def test_delta_anchor_matches_reference(right):
    from grasp_s2r_palm_command import PalmCmdCfg, PalmCommand

    p = right.action.palm
    ref = PalmCommand(PalmCmdCfg(
        delta_xyz=tuple(p.delta_xyz), delta_rot_deg=p.delta_rot_deg,
        anchor_mode=p.anchor["mode"], anchor_offset_xyz=tuple(p.anchor["offset_xyz"]),
        rate_limit_m=p.pos_rate_limit, rate_limit_rot_deg=p.rot_rate_limit_deg,
        palm_box_min=tuple(p.box_lo), palm_box_max=tuple(p.box_hi),
        rot_center_deg=tuple(p.rot_center_deg), rot_half_deg=p.rot_half_deg,
        home_palm=tuple(p.home_palm)))
    obj = np.array([0.362, -0.16, 0.2823])
    ref.reset(object_spawn_pos=obj - np.asarray(p.anchor["fab_to_env"]))
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, obj)
    aux = D.DecoderAux(palm6=np.zeros(6), object_pos=obj)
    for a in _actions(60, right.policy.action_dim, seed=1):
        out = dec.step(a, aux)
        np.testing.assert_allclose(out.palm6, ref.step(a[:6]), rtol=0, atol=1e-12)
        assert out.gripper_cmd is None
        assert out.hand_target.shape == (N_HAND,)


@needs_right
def test_delta_anchor_first_command_unprimed_then_rate_limited(right):
    p = right.action.palm
    obj = np.array([0.362, -0.16, 0.2823])
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, obj)
    aux = D.DecoderAux(palm6=np.zeros(6), object_pos=obj)
    anchor = obj - np.asarray(p.anchor["fab_to_env"]) + np.asarray(p.anchor["offset_xyz"])
    a = np.zeros(21)
    a[:3] = 1.0                                   # +delta_xyz 전부
    first = dec.step(a, aux).palm6
    want = np.minimum(anchor + np.asarray(p.delta_xyz), p.box_hi)
    np.testing.assert_allclose(first[:3], want, atol=1e-12)   # 첫 지령: 리미터 없음
    a[:3] = -1.0
    second = dec.step(a, aux).palm6
    assert np.linalg.norm(second[:3] - first[:3]) == pytest.approx(p.pos_rate_limit, abs=1e-12)


@needs_right
def test_delta_anchor_subtracts_fab_to_env(right):
    f2e = [0.01, 0.02, 0.03]
    palm = dataclasses.replace(right.action.palm, anchor={**right.action.palm.anchor, "fab_to_env": f2e})
    shifted = dataclasses.replace(right, action=dataclasses.replace(right.action, palm=palm))
    obj = np.array([0.362, -0.16, 0.30])
    aux = D.DecoderAux(palm6=np.zeros(6), object_pos=obj)
    outs = []
    for c in (right, shifted):
        dec = D.ActionDecoder(c, hand_soft_limits=SOFT_LIMITS)
        _reset_right(dec, c, obj)
        outs.append(dec.step(np.zeros(21), aux).palm6)
    np.testing.assert_allclose(outs[0][:3] - outs[1][:3], f2e, atol=1e-12)
    np.testing.assert_allclose(outs[0][3:], outs[1][3:], atol=1e-12)


@needs_right
def test_spawn_anchor_requires_object_pos(right):
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    palm6, tips = _cage_inputs()
    with pytest.raises(D.DecoderError):
        dec.reset(hand_q=np.zeros(N_HAND), palm6=palm6, tips=tips)


# ---------------------------------------------------------------- close gate / cage
@needs_right
def test_cage_calibration_matches_core_formula():
    palm6, tips = _cage_inputs()
    cage = D.calibrate_cage(palm6, tips)
    np.testing.assert_allclose(cage.offset_palm, [0.0, 0.0, 0.0], atol=1e-12)
    assert cage.radius == pytest.approx(0.1)
    # palm 을 옮기고 z 축 90° 돌리면 오프셋은 palm 프레임 기준으로 나온다
    palm6b = np.array([0.5, 0.2, 0.1, math.pi / 2, 0.0, 0.0])
    tips_b = tips + np.array([0.5, 0.2, 0.1]) + np.array([0.0, 0.05, 0.0])
    cage_b = D.calibrate_cage(palm6b, tips_b)
    np.testing.assert_allclose(cage_b.offset_palm, [0.05, 0.0, 0.0], atol=1e-12)


@needs_right
def test_close_gate_ramp_values(right):
    g = right.action.hand.params["close_gate"]
    assert g["enabled"] and g["ramp"] == pytest.approx(0.5) and g["z_deadband"] == pytest.approx(0.03)
    obj0 = np.array([0.362, -0.16, 0.2823])
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, obj0)
    r, ramp = 0.1, g["ramp"]
    palm6 = np.zeros(6)

    def gate(obj):
        return dec.close_gate(palm6, np.asarray(obj))

    assert gate([0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert gate([r, 0.0, 0.0]) == pytest.approx(0.0)
    assert gate([r * (1.0 - ramp / 2.0), 0.0, 0.0]) == pytest.approx(0.5)
    assert gate([2.0 * r, 0.0, 0.0]) == pytest.approx(0.0)
    # z 데드밴드: 밴드 안의 z 는 거리에 안 들어간다
    assert gate([0.0, 0.0, g["z_deadband"] * 0.9]) == pytest.approx(1.0)
    assert gate([0.0, 0.0, g["z_deadband"] + r]) == pytest.approx(0.0)
    # 게이트 = clip((r − d) / (ramp·r), 0, 1) 일반식
    d = 0.07
    assert gate([d, 0.0, 0.0]) == pytest.approx(min(1.0, max(0.0, (r - d) / (ramp * r))))


@needs_right
def test_close_gate_disabled_is_one(right):
    params = {**right.action.hand.params,
              "close_gate": {**right.action.hand.params["close_gate"], "enabled": False}}
    hand = dataclasses.replace(right.action.hand, params=params)
    c = dataclasses.replace(right, action=dataclasses.replace(right.action, hand=hand))
    dec = D.ActionDecoder(c, hand_soft_limits=SOFT_LIMITS)
    dec.reset(object_pos=np.zeros(3), hand_q=np.zeros(N_HAND))   # 케이지 불필요
    assert dec.close_gate(np.zeros(6), np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)


@needs_right
def test_close_gate_needs_cage(right):
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    with pytest.raises(D.DecoderError):
        dec.reset(object_pos=np.zeros(3), hand_q=np.zeros(N_HAND))


# ---------------------------------------------------------------- synergy hand
@needs_right
def test_synergy_target_order_and_pose_tables(right):
    from grasp_s2r_synergy import HAND_JOINT_NAMES

    h = right.action.hand
    assert list(HAND_JOINT_NAMES) == h.joints
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    assert dec.hand_joints == tuple(h.joints)
    np.testing.assert_allclose(dec.hand.open_pose, h.params["open_pose"])
    np.testing.assert_allclose(dec.hand.grip_pose, h.params["grip_pose"])
    obj = np.array([0.362, -0.16, 0.2823])
    _reset_right(dec, right, obj)
    # 게이트 1(케이지 중심 = 물체 = 원점) 에서 전부 닫기 → 가동 관절이 close_speed 만큼 진행
    aux = D.DecoderAux(palm6=np.zeros(6), object_pos=np.zeros(3))
    a = np.zeros(21)
    a[6:] = 1.0
    open_pose = np.asarray(h.params["open_pose"])
    grip_pose = np.asarray(h.params["grip_pose"])
    out = dec.step(a, aux)
    want = np.clip(open_pose + (grip_pose - open_pose) * h.params["close_speed"],
                   SOFT_LIMITS[:, 0], SOFT_LIMITS[:, 1])
    np.testing.assert_allclose(out.hand_target, want, atol=1e-12)
    assert out.close_gate == pytest.approx(1.0)
    np.testing.assert_allclose(out.syn_vel, (want - open_pose) / right.rate.step_dt, atol=1e-9)


@needs_right
def test_synergy_matches_reference_with_gate(right):
    from grasp_s2r_synergy import SynergyCfg, SynergyHand

    h = right.action.hand.params
    ref = SynergyHand(SynergyCfg(**{k: h[k] for k in SynergyCfg.__dataclass_fields__}),
                      soft_limits=SOFT_LIMITS)
    ref.reset(hand_q=np.asarray(h["open_pose"]))
    obj = np.array([0.362, -0.16, 0.2823])
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, obj)
    palm6 = np.zeros(6)
    rng = np.random.default_rng(3)
    for a in _actions(80, 21, seed=2):
        obj_now = obj + rng.uniform(-0.12, 0.12, size=3)
        hand_q = ref.target + rng.uniform(-0.4, 0.4, size=N_HAND)
        aux = D.DecoderAux(palm6=palm6, object_pos=obj_now, hand_q=hand_q)
        out = dec.step(a, aux)
        gate = dec.close_gate(palm6, obj_now)
        want = ref.step(a[6:], close_gate=gate, hand_q=hand_q)
        np.testing.assert_allclose(out.hand_target, want, atol=1e-12)
        assert out.close_gate == pytest.approx(gate)


@needs_right
def test_synergy_stall_freeze_blocks_closing(right):
    obj = np.array([0.362, -0.16, 0.2823])
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, obj)
    a = np.zeros(21)
    a[6:] = 1.0
    open_pose = np.asarray(right.action.hand.params["open_pose"])
    # 실측이 목표에서 0.5 rad 뒤처지고 한계에서도 떨어져 있으면 _3/_4 는 동결
    stalled = open_pose - 0.5
    out = dec.step(a, D.DecoderAux(palm6=np.zeros(6), object_pos=np.zeros(3), hand_q=stalled))
    names = right.action.hand.joints
    frozen = [i for i, n in enumerate(names) if n.endswith(("_3", "_4"))]
    np.testing.assert_allclose(out.hand_target[frozen], open_pose[frozen], atol=1e-12)
    moving = [i for i, n in enumerate(names) if n.endswith("_2") and "pinky" not in n and "thumb" not in n]
    assert np.all(out.hand_target[moving] > open_pose[moving])


@needs_right
def test_synergy_requires_palm_and_object(right):
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, np.array([0.362, -0.16, 0.2823]))
    with pytest.raises(D.DecoderError):
        dec.step(np.zeros(21), D.DecoderAux())


@needs_right
def test_pose_table_mismatch_is_contract_error(right):
    params = {**right.action.hand.params, "grip_pose": [0.0] * N_HAND}
    hand = dataclasses.replace(right.action.hand, params=params)
    c = dataclasses.replace(right, action=dataclasses.replace(right.action, hand=hand))
    with pytest.raises(C.ContractError):
        D.ActionDecoder(c, hand_soft_limits=SOFT_LIMITS)


@needs_right
def test_step_does_not_mutate_inputs(right):
    obj = np.array([0.362, -0.16, 0.2823])
    dec = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    _reset_right(dec, right, obj)
    a = np.full(21, 0.3)
    a_copy = a.copy()
    hand_q = np.zeros(N_HAND)
    dec.step(a, D.DecoderAux(palm6=np.zeros(6), object_pos=obj, hand_q=hand_q))
    np.testing.assert_array_equal(a, a_copy)
    np.testing.assert_array_equal(hand_q, np.zeros(N_HAND))


# ---------------------------------------------------------------- v2 sides · control-only DirectDecoder
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
DG5FM_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.dg5f-m.json"
needs_asset = pytest.mark.skipif(not ASSET_JSON.exists(), reason="asset contract 없음")
needs_dg5fm = pytest.mark.skipif(not DG5FM_JSON.exists(), reason="right_g1 dg5f-m contract 없음")


@pytest.fixture(scope="module")
def asset() -> C.DeployContract:
    return C.load_contract(ASSET_JSON)


def test_euler_zyx_inverse_and_quaternion_round_trip():
    from grasp_s2r_core import _quat_from_matrix

    rng = np.random.default_rng(0)
    for _ in range(50):
        e = rng.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi])
        R = D.rot_euler_zyx(e)
        np.testing.assert_allclose(D.euler_zyx_from_rot(R), e, atol=1e-12)
        q = _quat_from_matrix(R)
        np.testing.assert_allclose(D.rot_from_quat(q), R, atol=1e-12)
        np.testing.assert_allclose(D.rot_euler_zyx(D.euler_zyx_from_quat(q)), R, atol=1e-12)
    R = D.rot_euler_zyx([0.3, np.pi / 2, 0.7])                        # 짐벌락: ex 0 으로 접되 회전은 같다
    e = D.euler_zyx_from_rot(R)
    assert e[2] == 0.0
    np.testing.assert_allclose(D.rot_euler_zyx(e), R, atol=1e-9)
    with pytest.raises(D.DecoderError):
        D.rot_from_quat([0.0, 0.0, 0.0, 0.0])


@needs_asset
@pytest.mark.parametrize("side", ["left", "right"])
def test_direct_decoder_holds_measured_hand_and_follows_hand_cmd(asset, side):
    dec = D.make_decoder(asset, side=side)
    assert isinstance(dec, D.DirectDecoder) and dec.kind == D.KIND_DIRECT and dec.hand is None and dec.cage is None
    assert dec.side == side and dec.hand_joints == tuple(asset.side(side).hand_joints) and len(dec.hand_joints) == 20
    with pytest.raises(D.DecoderError):
        dec.step(np.zeros(6), D.DecoderAux())                            # reset 전
    meas = np.linspace(-0.5, 0.5, 20)
    dec.reset(hand_q=meas)
    palm = np.array([0.3, -0.2, 0.4, 1.5, 0.0, 1.5])
    out = dec.step(palm, D.DecoderAux())
    np.testing.assert_array_equal(out.palm6, palm)
    np.testing.assert_array_equal(out.hand_target, meas)                 # 손 목표 = 리셋 실측 유지
    assert out.gripper_cmd is None and out.close_gate == 1.0 and np.all(out.syn_vel == 0.0)
    cmd = meas + 0.2
    out2 = dec.step(palm, D.DecoderAux(hand_cmd=cmd))
    np.testing.assert_array_equal(out2.hand_target, cmd)
    np.testing.assert_allclose(out2.syn_vel, np.full(20, 0.2 / asset.rate.step_dt))
    out3 = dec.step(palm, D.DecoderAux())
    np.testing.assert_array_equal(out3.hand_target, cmd)                 # hand_cmd 없으면 직전 목표 유지
    assert np.all(out3.syn_vel == 0.0) and out3.hand_target is not out2.hand_target
    cmd[:] = 9.0                                                         # 입력 배열을 바꿔도 상태는 안 바뀐다
    np.testing.assert_array_equal(dec.step(palm, D.DecoderAux()).hand_target, meas + 0.2)


@needs_asset
def test_direct_decoder_validates_inputs(asset):
    dec = D.DirectDecoder(asset, side="left")
    with pytest.raises(D.DecoderError):
        dec.reset()                                                      # hand_q 필요(손 20관절)
    dec.reset(hand_q=np.zeros(20))
    with pytest.raises(D.DecoderError):
        dec.step(np.zeros(5), D.DecoderAux())
    with pytest.raises(D.DecoderError):
        dec.step(np.full(6, np.nan), D.DecoderAux())
    with pytest.raises(D.DecoderError):
        dec.step(np.zeros(6), D.DecoderAux(hand_cmd=np.zeros(19)))
    with pytest.raises(C.ContractError):
        D.ActionDecoder(asset, side="left")                              # control-only 에는 액션 디코더가 없다
    with pytest.raises(C.ContractError):
        D.make_decoder(asset, side="middle")


@needs_dg5fm
@needs_right
def test_action_decoder_side_param_matches_legacy_on_rebased_contract(right):
    dg = C.load_contract(DG5FM_JSON)
    assert dg.primary_side == "right" and dg.side("right").action_groups == ["palm", "hand"]
    obj = np.array([0.362, -0.16, 0.2823])
    a = D.ActionDecoder(right, hand_soft_limits=SOFT_LIMITS)
    b = D.make_decoder(dg, side="right", hand_soft_limits=SOFT_LIMITS)
    assert isinstance(b, D.ActionDecoder) and b.side == "right" and b.kind == D.KIND_SYNERGY
    _reset_right(a, right, obj)
    _reset_right(b, dg, obj)
    aux = D.DecoderAux(palm6=np.zeros(6), object_pos=obj)
    for act in _actions(40, 21, seed=5):
        oa, ob = a.step(act, aux), b.step(act, aux)
        np.testing.assert_array_equal(oa.palm6, ob.palm6)
        np.testing.assert_array_equal(oa.hand_target, ob.hand_target)
    with pytest.raises(C.ContractError):
        D.ActionDecoder(dg, side="left")                                 # 이 계약에는 left 가 없다


@needs_right
def test_side_soft_limits_from_side_hand(right):
    np.testing.assert_allclose(D.side_soft_limits(right), right.action.hand.params["soft_limits"])
    np.testing.assert_allclose(D.side_soft_limits(right, "right"), right.action.hand.params["soft_limits"])


@needs_asset
def test_side_soft_limits_none_for_control_only(asset):
    assert D.side_soft_limits(asset, "left") is None and D.side_soft_limits(asset) is None
