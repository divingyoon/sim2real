"""M4 — FabricCore: 계약으로 Fabrics 를 세우고 set_features 1회 → decimation 번 적분한다.

GPU 없이 돈다 — fabric/integrator 자리에 **가짜**를 꽂아 호출 순서·dt·형상·순열만 잠근다.
실제 fabrics_sim 과의 수치 등가는 `test_pc_golden_fabric_parity.py`(gpu) 가 본다.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from policy_control import contract as C
from policy_control import fabric_core as F

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
needs_left = pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")
needs_right = pytest.mark.skipif(not RIGHT_JSON.exists(), reason="right_g1 contract 없음")


class FakeFabric:
    """set_features 호출을 기록하고, FK 는 q 의 결정적 함수로 낸다."""

    def __init__(self, num_joints: int) -> None:
        self.num_joints = num_joints
        self.default_config = torch.zeros(1, num_joints)
        self.calls: list[dict] = []

    def set_features(self, hand, palm, convention, q, qd, obj_ids, obj_ind, damping):
        self.calls.append({"hand": hand.clone(), "palm": palm.clone(), "conv": convention,
                           "q": q.clone(), "qd": qd.clone(), "damping": damping.clone()})

    def get_palm_pose(self, q, convention):
        assert convention == "euler_zyx"
        return torch.cat([q[:, :3], q[:, :3] * 0.5], dim=1)

    def get_fingertip_positions(self, q):
        return q[:, :5].unsqueeze(-1).repeat(1, 1, 3)


class FakeIntegrator:
    """step 마다 q += 0.1, qd = 1 — 호출 횟수·dt 를 기록한다."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def step(self, q, qd, qdd, dt):
        self.calls.append(float(dt))
        return q + 0.1, torch.ones_like(qd), qdd


def _backend(n: int) -> F.FabricBackend:
    return F.FabricBackend(fabric=FakeFabric(n), integrator=FakeIntegrator(),
                           object_ids=None, object_indicator=None, device="cpu")


@pytest.fixture(scope="module")
def left() -> C.DeployContract:
    return C.load_contract(LEFT_JSON)


@pytest.fixture(scope="module")
def right() -> C.DeployContract:
    return C.load_contract(RIGHT_JSON)


# ---------------------------------------------------------------- call sequence
@needs_left
def test_left_step_calls_set_features_once_then_integrates_decimation_times(left):
    be = _backend(7)
    core = F.FabricCore(left, "cpu", backend=be)
    palm6 = np.arange(6, dtype=float)
    out = core.step(palm6)
    assert len(be.fabric.calls) == 1
    assert be.integrator.calls == [pytest.approx(left.fabric.dt)] * left.fabric.decimation
    call = be.fabric.calls[0]
    assert call["conv"] == "euler_zyx"
    np.testing.assert_allclose(call["palm"].numpy()[0], palm6)
    assert call["hand"].shape == (1, 5) and float(call["hand"].abs().sum()) == 0.0
    assert call["damping"].shape == (1, 1) and float(call["damping"][0, 0]) == left.fabric.damping
    # set_features 는 적분 **전** 상태(홈)를 받는다
    np.testing.assert_allclose(call["q"].numpy()[0], left.fabric.home_q, atol=1e-6)
    # 두 번째 스텝: 다시 1회 + n 회, 상태가 이어진다
    core.step(palm6)
    assert len(be.fabric.calls) == 2 and len(be.integrator.calls) == 2 * left.fabric.decimation
    np.testing.assert_allclose(be.fabric.calls[1]["q"].numpy()[0], out.q_full, atol=1e-6)


@needs_left
def test_left_output_shapes_and_values(left):
    core = F.FabricCore(left, "cpu", backend=_backend(7))
    out = core.step(np.zeros(6))
    n, dec = 7, left.fabric.decimation
    assert isinstance(out, F.JointTarget)
    assert out.q_arm.shape == (n,) and out.qd_arm.shape == (n,)
    assert out.q_full.shape == (n,) and out.substeps.shape == (dec, n)
    np.testing.assert_allclose(out.q_full, np.asarray(left.fabric.home_q) + 0.1 * dec, atol=1e-6)
    np.testing.assert_allclose(out.substeps[-1], out.q_full)
    np.testing.assert_allclose(out.substeps[0], np.asarray(left.fabric.home_q) + 0.1, atol=1e-6)
    np.testing.assert_allclose(out.qd_arm, np.ones(n) * left.fabric.vel_ff_scale)
    assert core.joint_names == tuple(left.fabric.joint_order)


@needs_left
def test_vel_ff_scale_scales_qd(left):
    c = dataclasses.replace(left, fabric=dataclasses.replace(left.fabric, vel_ff_scale=0.25))
    core = F.FabricCore(c, "cpu", backend=_backend(7))
    np.testing.assert_allclose(core.step(np.zeros(6)).qd_arm, np.full(7, 0.25))


@needs_left
def test_reset_seeds_state_and_rejects_bad_length(left):
    be = _backend(7)
    core = F.FabricCore(left, "cpu", backend=be)
    core.step(np.zeros(6))
    q_home = np.linspace(-1, 1, 7)
    core.reset(q_home)
    np.testing.assert_allclose(core.q, q_home)
    core.step(np.zeros(6))
    np.testing.assert_allclose(be.fabric.calls[-1]["q"].numpy()[0], q_home, atol=1e-6)
    assert float(be.fabric.calls[-1]["qd"].abs().sum()) == 0.0
    with pytest.raises(F.FabricError):
        core.reset(np.zeros(6))


@needs_left
def test_left_rejects_hand_target_and_sync(left):
    core = F.FabricCore(left, "cpu", backend=_backend(7))
    with pytest.raises(F.FabricError):
        core.step(np.zeros(6), hand_target=np.zeros(20))
    with pytest.raises(F.FabricError):
        core.sync_hand(np.zeros(20))


@needs_left
def test_num_joints_mismatch_is_error(left):
    with pytest.raises(F.FabricError):
        F.FabricCore(left, "cpu", backend=_backend(27))


@needs_left
def test_bad_palm_shape_is_error(left):
    core = F.FabricCore(left, "cpu", backend=_backend(7))
    with pytest.raises(F.FabricError):
        core.step(np.zeros(7))


@needs_left
def test_fk_callables(left):
    core = F.FabricCore(left, "cpu", backend=_backend(7))
    q = np.arange(7, dtype=float)
    pose = core.palm_pose(q)
    assert pose.shape == (6,)
    np.testing.assert_allclose(pose, np.concatenate([q[:3], q[:3] * 0.5]))
    assert core.tips(q).shape == (5, 3)


# ---------------------------------------------------------------- right: hand slot
@needs_right
def test_right_hand_target_synced_before_set_features(right):
    be = _backend(27)
    core = F.FabricCore(right, "cpu", backend=be)
    hand = np.linspace(0.0, 1.9, 20)
    out = core.step(np.zeros(6), hand_target=hand)
    q_seen = be.fabric.calls[0]["q"].numpy()[0]
    np.testing.assert_allclose(q_seen[:7], right.fabric.home_q[:7], atol=1e-6)
    np.testing.assert_allclose(q_seen[7:], hand, atol=1e-6)     # 프로필 순 == fabric 순
    assert out.q_arm.shape == (7,) and out.q_full.shape == (27,)
    assert out.substeps.shape == (right.fabric.decimation, 27)


@needs_right
def test_right_hand_sync_required(right):
    core = F.FabricCore(right, "cpu", backend=_backend(27))
    assert right.fabric.hand_sync == "syn_target"
    with pytest.raises(F.FabricError):
        core.step(np.zeros(6))
    with pytest.raises(F.FabricError):
        core.step(np.zeros(6), hand_target=np.zeros(19))


@needs_right
def test_right_sync_hand_permutes_by_name(right):
    # 손 슬롯 순서를 fabric 쪽에서 뒤집은 계약 → 이름 순열로 되돌아와야 한다
    order = list(right.fabric.joint_order[:7]) + list(reversed(right.fabric.joint_order[7:]))
    home = list(right.fabric.home_q[:7]) + list(reversed(right.fabric.home_q[7:]))
    c = dataclasses.replace(right, fabric=dataclasses.replace(right.fabric, joint_order=order, home_q=home))
    be = _backend(27)
    core = F.FabricCore(c, "cpu", backend=be)
    hand = np.linspace(0.0, 1.9, 20)            # contract.action.hand.joints 순
    core.sync_hand(hand)
    np.testing.assert_allclose(core.q[7:], hand[::-1], atol=1e-6)
    core.step(np.zeros(6), hand_target=hand)
    np.testing.assert_allclose(be.fabric.calls[0]["q"].numpy()[0, 7:], hand[::-1], atol=1e-6)


@needs_right
def test_hand_sync_unknown_mode_is_contract_error(right):
    c = dataclasses.replace(right, fabric=dataclasses.replace(right.fabric, hand_sync="telepathy"))
    with pytest.raises(C.ContractError):
        F.FabricCore(c, "cpu", backend=_backend(27))


# ---------------------------------------------------------------- world / factory
@needs_right
def test_table_world_dict_matches_training_formula(right):
    w = right.fabric.world
    d = F.table_world_dict(right, table_z=0.2)
    assert set(d) == {"table"} and d["table"]["type"] == "box" and d["table"]["env_index"] == "all"
    sx, sy, th = (float(v) for v in d["table"]["scaling"].split())
    cx, cy, cz, *quat = (float(v) for v in d["table"]["transform"].split())
    lo, hi = np.asarray(right.action.palm.box_lo), np.asarray(right.action.palm.box_hi)
    assert sx == pytest.approx(hi[0] - lo[0] + 2 * w["margin_xy"])
    assert sy == pytest.approx(hi[1] - lo[1] + 2 * w["margin_xy"])
    assert th == pytest.approx(w["thickness"])
    assert (cx, cy) == (pytest.approx(0.5 * (lo[0] + hi[0])), pytest.approx(0.5 * (lo[1] + hi[1])))
    assert cz == pytest.approx(0.2 - 0.5 * w["thickness"])
    assert quat == [0.0, 0.0, 0.0, 1.0]


@needs_right
def test_table_world_needs_table_z(right):
    with pytest.raises(F.FabricError):
        F.table_world_dict(right, table_z=None)
    off = dataclasses.replace(right, fabric=dataclasses.replace(
        right.fabric, world={**right.fabric.world, "table_obstacle": False}))
    assert F.table_world_dict(off, table_z=None) is None


@needs_left
def test_make_fabric_requires_cuda(left):
    with pytest.raises(F.FabricError):
        F.make_fabric(left, "cpu")


@needs_left
def test_unknown_world_spec_is_contract_error(left):
    c = dataclasses.replace(left, fabric=dataclasses.replace(left.fabric, world={"mesh": "x"}))
    with pytest.raises(C.ContractError):
        F.FabricCore(c, "cpu", backend=_backend(7))


# ---------------------------------------------------------------- v2 sides: joint keys · control-only sides
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
needs_asset = pytest.mark.skipif(not ASSET_JSON.exists(), reason="asset_openarm_dg5f-m_bi_rl contract 없음")
#: 모든 fabric URDF(좌·우·레거시·dg5f-m)의 내부 관절 이름 — 우측 이름뿐이다
FABRIC_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)] + [f"rj_dg_{f}_{k}" for f in range(1, 6) for k in range(1, 5)]


class NamedFakeFabric(FakeFabric):
    """URDF 관절 이름을 노출하는 가짜 — FabricCore 의 joint_order ↔ URDF 대조를 켠다."""

    def __init__(self, names) -> None:
        super().__init__(len(names))
        self._names = list(names)

    def get_joint_names(self):
        return list(self._names)


def _named_backend(names) -> F.FabricBackend:
    return F.FabricBackend(fabric=NamedFakeFabric(names), integrator=FakeIntegrator(),
                           object_ids=None, object_indicator=None, device="cpu")


@pytest.fixture(scope="module")
def asset() -> C.DeployContract:
    return C.load_contract(ASSET_JSON)


def test_joint_key_ignores_side_prefix():
    assert (F.joint_key("l_aj_3") == F.joint_key("r_aj_3") == F.joint_key("openarm_right_joint3")
            == F.joint_key("openarm_left_joint3") == ("arm", 3))
    assert F.joint_key("l_hj_index_2") == F.joint_key("rj_dg_2_2") == F.joint_key("lj_dg_2_2") == ("hand", 2, 2)
    assert F.joint_key("r_hj_thumb_1") == ("hand", 1, 1) and F.joint_key("l_hj_pinky_4") == ("hand", 5, 4)
    for bad in ("l_hj_gripper_1", "openarm_right_joint8", "rj_dg_6_1", "palm"):
        with pytest.raises(C.ContractError):
            F.joint_key(bad)


@needs_asset
@pytest.mark.parametrize("side", ["left", "right"])
def test_check_joint_names_maps_both_sides_onto_right_named_urdf(asset, side):
    order = asset.side(side).fabric.joint_order
    assert all(j.startswith(side[0] + "_") for j in order)
    F.check_joint_names(order, FABRIC_NAMES)                       # 접두사 무시, 팔 번호·손가락/마디 번호로 대응
    scrambled = FABRIC_NAMES[:7] + FABRIC_NAMES[8:] + FABRIC_NAMES[7:8]
    with pytest.raises(C.ContractError, match="slots"):
        F.check_joint_names(order, scrambled)
    with pytest.raises(C.ContractError, match="27"):
        F.check_joint_names(order, FABRIC_NAMES[:7])


def test_name_permutation_semantics():
    np.testing.assert_array_equal(F.name_permutation(["a", "b", "c"], ["c", "a"]), [2, 0])
    with pytest.raises(C.ContractError):
        F.name_permutation(["a"], ["z"])


@needs_asset
@pytest.mark.parametrize("side", ["left", "right"])
def test_control_only_side_builds_from_side_cfg(asset, side):
    s = asset.side(side)
    be = _named_backend(FABRIC_NAMES)
    core = F.FabricCore(asset, "cpu", side=side, backend=be)
    assert core.side == side and core.n == 27 and core.n_arm == 7 and core.n_hand == 20
    assert core.joint_names == tuple(s.fabric.joint_order) and core.arm_joints == tuple(s.arm_joints)
    assert core.hand_joints == tuple(s.hand_joints)               # control-only: 자산 canonical 순
    assert core.cfg == s.fabric and core.cfg.hand_sync == "syn_target"
    np.testing.assert_allclose(core.q, s.fabric.home_q)
    hand = np.linspace(0.1, 2.0, 20)
    out = core.step(np.zeros(6), hand_target=hand)
    np.testing.assert_allclose(be.fabric.calls[0]["q"].numpy()[0, 7:], hand, atol=1e-6)   # 항등 순열
    assert out.q_full.shape == (27,) and out.substeps.shape == (s.fabric.decimation, 27)
    with pytest.raises(F.FabricError):
        core.step(np.zeros(6))                                     # syn_target: hand_target 필수


@needs_asset
def test_control_only_scrambled_urdf_is_rejected(asset):
    scrambled = FABRIC_NAMES[:7] + list(reversed(FABRIC_NAMES[7:]))
    with pytest.raises(C.ContractError, match="joint_order"):
        F.FabricCore(asset, "cpu", side="left", backend=_named_backend(scrambled))


@needs_asset
def test_default_side_is_primary_and_unknown_side_rejected(asset):
    core = F.FabricCore(asset, "cpu", backend=_backend(27))
    assert core.side == asset.primary_side == "right"
    with pytest.raises(C.ContractError):
        F.FabricCore(asset, "cpu", side="middle", backend=_backend(27))


@needs_asset
def test_side_without_fabric_is_contract_error(asset):
    left = dataclasses.replace(asset.side("left"), fabric=None)
    c = dataclasses.replace(asset, sides={**asset.sides, "left": left})
    with pytest.raises(C.ContractError, match="fabric"):
        F.FabricCore(c, "cpu", side="left", backend=_backend(27))


@needs_asset
def test_world_by_filename_for_both_tesollo_sides(asset):
    assert F.world_kind(asset, "left") == F.world_kind(asset, "right") == "filename"
    assert asset.side("left").fabric.world["filename"] == "open_tesollo_left_boxes_no_table"
    assert asset.side("right").fabric.world["filename"] == "open_tesollo_boxes_no_table"
    assert asset.side("left").fabric.use_body_repulsion_pairs and asset.side("right").fabric.use_body_repulsion_pairs


@needs_asset
def test_table_world_on_control_only_side_needs_palm_box(asset):
    left = asset.side("left")
    fab = dataclasses.replace(left.fabric, world={"table_obstacle": True, "margin_xy": 0.1, "thickness": 0.05})
    c = dataclasses.replace(asset, sides={**asset.sides, "left": dataclasses.replace(left, fabric=fab)})
    assert F.world_kind(c, "left") == "table"
    with pytest.raises(C.ContractError, match="palm box"):
        F.table_world_dict(c, table_z=0.2, side="left")


@needs_asset
def test_make_fabric_side_requires_cuda(asset):
    with pytest.raises(F.FabricError):
        F.make_fabric(asset, "cpu", side="left")


@needs_right
def test_primary_side_mirrors_legacy_flat_sections(right):
    """dataclasses.replace(contract, fabric=…) 는 primary 팔에 그대로 적용된다(단일팔 코드·골든 변형 규약)."""
    c = dataclasses.replace(right, fabric=dataclasses.replace(right.fabric, damping=3.5))
    core = F.FabricCore(c, "cpu", backend=_backend(27))
    assert core.cfg.damping == 3.5 and float(core._damping[0, 0]) == 3.5
    assert core.hand_joints == tuple(right.action.hand.joints)
