"""계약 v2 — 양팔 `sides` + 자산 바인딩 + 제어 전용(정책 없음) 계약.

원칙은 v1 과 같다: 숫자는 전부 데이터 파일(자산 manifest/URDF, control_gains.yaml, 런 dump,
학습 소스 상수)에서 온다. 여기서는 (1) v1 파일이 그대로 로드되며 한 팔 side 가 유도되는지,
(2) 자산 manifest → 양팔 side 가 만들어지는지, (3) 런 계약을 자산에 재기반하면 fabric URDF 만
바뀌고 정책 의미는 그대로인지, (4) 검증이 잘못된 side 를 거부하는지를 잠근다.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from policy_control import contract as C
from policy_control import contract_assets as A
from policy_control import contract_build as B

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
LEFT_RUN = SIM2REAL / "logs/policy/left_v2B25"
RIGHT_RUN = SIM2REAL / "logs/policy/right_g1"
ASSET = A.ASSETS["openarm_dg5f-m_bi_rl"]
FABRIC_ROOT = RL_WS / "hdgp/source/FABRICS/src/fabrics_sim"

needs_left = pytest.mark.skipif(not (LEFT_RUN / "nn").exists(), reason="left_v2B25 run dir 없음")
needs_right = pytest.mark.skipif(not (RIGHT_RUN / "nn").exists(), reason="right_g1 run dir 없음")
needs_asset = pytest.mark.skipif(not ASSET.manifest.exists(), reason="dg5f-m 자산 없음")


def _load_tool():
    import importlib.util

    path = SIM2REAL / "policy_control/tools/build_deploy_contract.py"
    spec = importlib.util.spec_from_file_location("build_deploy_contract_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ctl():
    return A.build_asset_contract()


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load(ASSET.manifest.read_text())


# ---------------------------------------------------------------- control-only from the asset
@needs_asset
def test_default_asset_is_bimanual_dg5f_m(ctl):
    assert A.DEFAULT_ASSET == "openarm_dg5f-m_bi_rl"
    assert ctl.schema == C.SCHEMA and ctl.control_only
    assert ctl.side_names == ["left", "right"] and ctl.primary_side == "right"
    assert ctl.asset.name == "openarm_dg5f-m_bi_rl" and ctl.asset.ee_kind == "dg5f"
    assert (RL_WS / ctl.asset.urdf).exists() and (RL_WS / ctl.asset.manifest).exists()
    assert ctl.policy.obs_dim == 0 and ctl.policy.action_dim == 0 and ctl.action.groups == ()
    assert ctl.action.palm is None and ctl.action.hand is None and ctl.run.checkpoint == ""


@needs_asset
def test_sides_come_from_the_manifest(ctl, manifest):
    order = manifest["control_joint_order"]
    for side in ("right", "left"):
        s = ctl.side(side)
        p = side[0]
        assert s.arm_joints == [j for j in order if j.startswith(f"{p}_aj_")]
        assert s.hand_joints == [j for j in order if j.startswith(f"{p}_hj_")] and len(s.hand_joints) == 20
        assert s.palm_body == f"{p}_hl_palm" and s.palm_body in manifest["link_order"]
        assert s.tip_bodies == [f"{p}_hl_{f}_tip" for f in A.FINGERS]
        assert all(b in manifest["link_order"] for b in s.tip_bodies)
        assert s.pd_groups == [f"{side}_arm", f"{side}_hand"]
        assert s.ee_kind == "dg5f" and s.action_groups == [] and s.palm is None and s.hand is None


@needs_asset
def test_side_fabrics_exist_on_disk(ctl):
    for side, cls in (("right", "OpenArmTeoslloPoseFabric"), ("left", "OpenArmTeoslloLeftPoseFabric")):
        f = ctl.side(side).fabric
        assert f.class_name == cls
        assert f.robot_dir == f"openarm_dg5f-m_bi_{side}"
        assert (FABRIC_ROOT / "models/robots/urdf" / f.robot_dir / f"{f.robot_dir}.urdf").exists()
        assert (FABRIC_ROOT / "fabric_params" / f.params).exists()
        assert (FABRIC_ROOT / "worlds" / f"{f.world['filename']}.yaml").exists()
        assert f.joint_order == ctl.side(side).arm_joints + ctl.side(side).hand_joints
        assert len(f.home_q) == 27 and f.use_body_repulsion_pairs
    # the legacy top-level fabric is the primary side's
    assert ctl.fabric == ctl.side("right").fabric
    assert ctl.pd.groups == ["right_arm", "right_hand"]


@needs_asset
def test_homes_zero_and_hand_open_mirrored(ctl):
    from grasp_s2r_synergy import HAND_JOINT_NAMES, HAND_OPEN_POSE
    from openarm.tesollo.left.grasp_v1 import grasp_left_preset as P

    assert ctl.side("right").home_arm == [0.0] * 7 and ctl.side("left").home_arm == [0.0] * 7
    right_open = dict(zip(HAND_JOINT_NAMES, HAND_OPEN_POSE))
    assert ctl.side("right").home_hand == {j: float(right_open[j]) for j in ctl.side("right").hand_joints}
    left = ctl.side("left").home_hand
    for (rj, v), sgn in zip(right_open.items(), P._HAND_SIGN):
        assert left["l" + rj[1:]] == pytest.approx(sgn * v)
    assert left["l_hj_thumb_2"] == pytest.approx(1.57)      # opposition flips with the mirror


@needs_asset
@needs_right
def test_home_from_run_mirrors_the_other_arm():
    from openarm.tesollo.left.grasp_v1 import grasp_left_preset as P

    c = A.build_asset_contract(home="run:logs/policy/right_g1")
    g1 = B.build_contract(RIGHT_RUN)
    assert c.side("right").home_arm == pytest.approx(g1.pd.home_arm)
    assert c.side("left").home_arm == pytest.approx([s * v for s, v in zip(P._ARM_SIGN, g1.pd.home_arm)])


@needs_asset
def test_gains_are_the_driver_gains(ctl):
    real = C.load_driver_gains(A.GAINS_YAML)
    for side in ctl.side_names:
        g = ctl.side(side).sim_gains
        assert g.kp == [real[i][0] for i in range(1, 8)] and g.kd == [real[i][1] for i in range(1, 8)]
        assert C.compare_gains(ctl, A.GAINS_YAML, side=side).ok
    assert C.require_gains(ctl, A.GAINS_YAML).ok


@needs_asset
def test_roundtrip_and_tool(ctl, tmp_path):
    out = tmp_path / "c.json"
    C.save_contract(ctl, out)
    back = C.load_contract(out)
    assert back == ctl
    raw = json.loads(out.read_text())
    assert raw["schema"] == C.SCHEMA and set(raw["sides"]) == {"left", "right"} and raw["control_only"]
    tool_main = _load_tool().main
    assert tool_main(["--asset", "openarm_dg5f-m_bi_rl", "--sides", "left", "--primary", "left",
                      "--out", str(tmp_path / "left.json")]) == 0
    one = C.load_contract(tmp_path / "left.json")
    assert one.side_names == ["left"] and one.fabric.class_name == "OpenArmTeoslloLeftPoseFabric"


@needs_asset
def test_single_side_contract_and_bad_primary():
    c = A.build_asset_contract(sides=("left",), primary="left")
    assert c.side_names == ["left"] and c.pd.groups == ["left_arm", "left_hand"]
    with pytest.raises(C.ContractError):
        A.build_asset_contract(sides=("right",), primary="left")
    with pytest.raises(C.ContractError):
        A.build_asset_contract(asset="no_such_asset")
    with pytest.raises(C.ContractError):
        A.build_asset_contract(home="somewhere")


# ---------------------------------------------------------------- binding run contracts to assets
@needs_right
@needs_asset
def test_bind_right_g1_to_dg5f_m_changes_only_the_robot_model():
    base = B.build_contract(RIGHT_RUN)
    bound = B.build_contract(RIGHT_RUN, asset="openarm_dg5f-m_bi_rl")
    assert bound.fabric.robot_dir == "openarm_dg5f-m_bi_right"
    assert bound.fabric.params == "openarm_dg5f-m_right_pose_params.yaml"
    assert replace(bound.fabric, robot_dir=base.fabric.robot_dir, params=base.fabric.params) == base.fabric
    assert bound.obs.fk["kind"] == "fabric" and bound.obs.fk["urdf"].endswith("openarm_dg5f-m_bi_rl.urdf")
    assert bound.obs.segments == base.obs.segments and bound.policy == base.policy and bound.pd == base.pd
    assert bound.action.palm == base.action.palm
    lim = B._urdf_joint_limits(ASSET.urdf, list(bound.action.hand.joints))
    assert bound.action.hand.params["soft_limits"] == lim
    assert {k: v for k, v in bound.action.hand.params.items() if k != "soft_limits"} == \
        {k: v for k, v in base.action.hand.params.items() if k != "soft_limits"}
    s = bound.side("right")
    assert s.palm_body == "r_hl_palm" and len(s.tip_bodies) == 5 and s.hand_joints == list(bound.action.hand.joints)
    assert s.action_groups == ["palm", "hand"] and bound.asset.name == "openarm_dg5f-m_bi_rl"
    assert not bound.control_only


@needs_left
def test_bind_left_v2b25_to_gripper_asset():
    bound = B.build_contract(LEFT_RUN, grasp_band="v1", asset="openarm_gripper_bi_rl")
    s = bound.side("left")
    assert bound.fabric.robot_dir == "openarm_gripper_bi_left" and bound.fabric.class_name == "OpenArmGripperLeftPoseFabric"
    assert s.ee_kind == "gripper" and s.hand_joints == ["l_hj_gripper_1", "l_hj_gripper_2"]
    assert s.palm_body == "l_hl_gripper_base" and s.tip_bodies == [] and s.action_groups == ["palm", "gripper"]
    with pytest.raises(C.ContractError):        # the gripper asset has no right-arm fabric
        A.bind_asset(B.build_contract(RIGHT_RUN), "openarm_gripper_bi_rl")


# ---------------------------------------------------------------- v1 compatibility + validation
@needs_left
def test_v1_file_loads_with_a_derived_side(tmp_path):
    c = B.build_contract(LEFT_RUN, grasp_band="v1")
    raw = C.to_dict(c)
    raw.pop("sides"), raw.pop("primary_side"), raw.pop("asset"), raw.pop("control_only")
    raw["schema"] = "policy_control/deploy_contract/v1"
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(raw))
    v1 = C.load_contract(p)
    s = v1.side("left")
    assert v1.side_names == ["left"] and v1.primary_side == "left" and v1.asset is None
    assert s.arm_joints == c.obs.joint_orders["arm"] and s.ee_kind == "gripper"
    assert s.fabric == c.fabric and s.palm == c.action.palm and s.hand == c.action.hand
    assert s.home_arm == c.pd.home_arm and s.sim_gains == c.pd.sim_gains and s.action_groups == ["palm", "gripper"]


@needs_asset
def test_validation_rejects_inconsistent_sides(ctl):
    bad_key = replace(ctl, sides={"left": ctl.side("right"), "right": ctl.side("right")})
    with pytest.raises(C.ContractError):
        C.validate(bad_key)
    with pytest.raises(C.ContractError):
        C.validate(replace(ctl, primary_side="up"))
    wrong_arm = replace(ctl.side("left"), arm_joints=ctl.side("right").arm_joints)
    with pytest.raises(C.ContractError):
        C.validate(replace(ctl, sides={**ctl.sides, "left": wrong_arm}))
    with pytest.raises(C.ContractError):
        C.validate(replace(ctl, control_only=False))          # a policy contract needs palm/hand decoders
    with pytest.raises(C.ContractError):
        ctl.side("up")


@needs_right
def test_validation_requires_sides_to_claim_every_action_group():
    c = B.build_contract(RIGHT_RUN)
    half = replace(c.side("right"), action_groups=["palm"])
    with pytest.raises(C.ContractError):
        C.validate(replace(c, sides={"right": half}))
    with pytest.raises(C.ContractError):
        C.validate(replace(c, sides={"right": replace(c.side("right"), action_groups=["palm", "hand", "foot"])}))
