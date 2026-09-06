"""M2 — sources: robot yaml → 리더 3종 → frozen RobotState.

canonical↔source 이름·부호·한계는 robot_control 프로필에서 온다. 스테일/결손은 스냅샷이
보고한다. 관절은 이름으로 옮기고, 결손은 에러다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_control import codec, sources

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
ROBOTS = SIM2REAL / "policy_control/config/robots"
PROFILE = SIM2REAL.parent / "robot_control/src/robot_control/profiles/openarm_tesollo.yaml"

LEFT_ARM = [f"l_aj_{i}" for i in range(1, 8)]
RIGHT_ARM = [f"r_aj_{i}" for i in range(1, 8)]
HAND_PROFILE = [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)]
TIPS = [f"r_hl_{f}_tip" for f in ("thumb", "index", "middle", "ring", "pinky")]


@pytest.fixture(params=["left_gripper_real", "left_gripper_fake", "right_dg5f_real", "right_dg5f_fake"])
def robot_yaml(request) -> Path:
    return ROBOTS / f"{request.param}.yaml"


# ------------------------------------------------------------------ yaml loading
def test_all_four_robot_yamls_load_and_validate(robot_yaml):
    cfg = sources.load_robot_cfg(robot_yaml)
    assert cfg.joint_profile.exists()
    assert {"arm", "ee", "object"} <= set(cfg.sources)
    assert cfg.table.top in (pytest.approx(0.200), pytest.approx(0.205))   # fake = 학습 sim 테이블 0.200
    # 좌 > 0; 우는 sim 파지 행동(손가락이 판에 닿음)을 허용해 −0.01 까지 둔다
    assert cfg.table.clearance_min >= -0.01
    for s in cfg.sources.values():
        assert s.type in ("joint_state", "float_array", "pose")
        assert s.stale_sec > 0


def test_left_and_right_yaml_details():
    left = sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml")
    assert left.sources["arm"].topic == "/joint_states"
    assert left.sources["ee"].mirror == {"l_hj_gripper_2": "l_hj_gripper_1"}
    assert left.sources["ee"].velocity == "zero"
    assert left.sources["object"].topic == "/objects/cup_big_s100/pose"
    assert left.sources["object"].mode == "attach_after_gate"
    right = sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml")
    assert right.sources["ee"].topic == "/dg5f_right/joint_states"
    assert list(right.sources["ee"].joints) == HAND_PROFILE
    assert right.sources["tip_force"].type == "float_array"
    assert right.sources["tip_force"].topic == "/dg5f_right/tip_forces_xyz"
    assert list(right.sources["tip_force"].tips) == TIPS
    assert right.sources["head"].required is False


def test_load_robot_cfg_rejects_bad_schema(tmp_path):
    base = yaml.safe_load((ROBOTS / "left_gripper_real.yaml").read_text())
    bad = dict(base)
    bad["sources"] = dict(base["sources"])
    bad["sources"]["arm"] = dict(base["sources"]["arm"], type="udp")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(sources.RobotCfgError, match="udp"):
        sources.load_robot_cfg(p)
    bad2 = {k: v for k, v in base.items() if k != "table"}
    p.write_text(yaml.safe_dump(bad2))
    with pytest.raises(sources.RobotCfgError, match="table"):
        sources.load_robot_cfg(p)


# ------------------------------------------------------------------ left source set
def _left_set() -> sources.SourceSet:
    return sources.SourceSet(sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml"))


def _left_joint_msg(q7, grip, stamp=0.0):
    prof = sources.load_profile(PROFILE)
    names = [prof[j]["source"] for j in LEFT_ARM] + [prof["l_hj_gripper_1"]["source"]]
    vals = list(q7) + [grip]
    # 실기 /joint_states 는 우팔도 같이 실린다 — 여분 관절은 무시돼야 한다
    names += ["openarm_right_joint1"]
    vals += [9.9]
    return codec.decode_joint_state(codec.encode_joint_state(names, vals, velocity=[0.1] * len(vals), stamp=stamp))


def test_left_snapshot_reorders_by_source_name_and_mirrors_gripper():
    ss = _left_set()
    q7 = np.arange(7, dtype=float) * 0.1
    ss.update_from_joint_state("arm", _left_joint_msg(q7, 0.03, stamp=1.0), now=1.0)
    ss.update_from_joint_state("ee", _left_joint_msg(q7, 0.03), now=1.0)
    ss.update_from_pose("object", codec.PoseSample(np.array([0.4, 0.2, 0.3]), np.array([1.0, 0, 0, 0]), "base_link", 1.0), now=1.0)
    st = ss.snapshot(now=1.1)
    np.testing.assert_allclose(st.arm_q, q7)
    np.testing.assert_allclose(st.arm_qd, 0.1)
    assert st.ee_names == ("l_hj_gripper_1", "l_hj_gripper_2")
    np.testing.assert_allclose(st.ee_q, [0.03, 0.03])
    np.testing.assert_allclose(st.ee_qd, [0.0, 0.0])          # velocity: zero
    np.testing.assert_allclose(st.object_pos, [0.4, 0.2, 0.3])
    assert st.tip_force is None and st.head is None
    assert st.stale == () and st.missing == ()
    assert st.stamps["arm"] == pytest.approx(1.0)


def test_left_missing_joint_in_message_is_an_error():
    ss = _left_set()
    msg = codec.decode_joint_state(codec.encode_joint_state(["openarm_left_joint1"], [0.0]))
    with pytest.raises(codec.CodecError):
        ss.update_from_joint_state("arm", msg, now=0.0)


def test_snapshot_reports_stale_and_missing_required_sources():
    ss = _left_set()
    ss.update_from_joint_state("arm", _left_joint_msg(np.zeros(7), 0.0), now=0.0)
    st = ss.snapshot(now=0.1)
    assert "arm" not in st.stale
    assert set(st.missing) >= {"ee", "object"}
    assert "head" not in st.missing                           # optional
    st2 = ss.snapshot(now=5.0)
    assert "arm" in st2.stale


def test_snapshot_returns_new_arrays_each_time():
    ss = _left_set()
    ss.update_from_joint_state("arm", _left_joint_msg(np.zeros(7), 0.0), now=0.0)
    a = ss.snapshot(now=0.0)
    b = ss.snapshot(now=0.0)
    assert a.arm_q is not b.arm_q
    with pytest.raises((ValueError, AttributeError)):
        a.arm_q[0] = 1.0                                       # frozen + read-only


def test_unknown_source_name_is_an_error():
    ss = _left_set()
    with pytest.raises(sources.RobotCfgError):
        ss.update_from_joint_state("nope", _left_joint_msg(np.zeros(7), 0.0), now=0.0)


# ------------------------------------------------------------------ right source set
def test_right_hand_tip_force_and_head():
    ss = sources.SourceSet(sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml"))
    prof = sources.load_profile(PROFILE)
    hand_src = [prof[j]["source"] for j in HAND_PROFILE]
    hand_vals = np.linspace(0, 1, 20)
    ss.update_from_joint_state("ee", codec.decode_joint_state(
        codec.encode_joint_state(hand_src[::-1], hand_vals[::-1], velocity=list(hand_vals[::-1] * 2))), now=0.0)
    ss.update_from_float_array("tip_force", codec.decode_float_array(
        codec.encode_float_array(np.arange(15.0), ("tip", "axis"), (5, 3), seq=0)), now=0.0)
    ss.update_from_joint_state("head", codec.decode_joint_state(
        codec.encode_joint_state(["head_j_tilt", "head_j_pan"], [-0.3, 0.1])), now=0.0)
    st = ss.snapshot(now=0.0)
    assert st.ee_names == tuple(HAND_PROFILE)
    np.testing.assert_allclose(st.ee_q, hand_vals)              # 이름으로 되돌린다
    np.testing.assert_allclose(st.ee_qd, hand_vals * 2)
    assert st.tip_force.shape == (5, 3)
    np.testing.assert_allclose(st.tip_force[1], [3, 4, 5])
    assert st.tip_names == tuple(TIPS)
    np.testing.assert_allclose(st.head, [0.1, -0.3])            # yaml joints 순서 (pan, tilt)


def test_right_tip_force_wrong_size_is_an_error():
    ss = sources.SourceSet(sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml"))
    bad = codec.decode_float_array(codec.encode_float_array(np.arange(5.0), ("tip",), (5,), seq=0))
    with pytest.raises(codec.CodecError):
        ss.update_from_float_array("tip_force", bad, now=0.0)


def test_decoder_target_source_is_stored_by_canonical_name():
    ss = sources.SourceSet(sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml"))
    msg = codec.encode_joint_target(list(reversed(HAND_PROFILE)), q=np.arange(20.0)[::-1], qd=np.zeros(20), episode="e", seq=1)
    ss.update_from_joint_state("decoder_target", codec.decode_joint_state(msg), now=0.0)
    st = ss.snapshot(now=0.0)
    np.testing.assert_allclose(st.decoder_target, np.arange(20.0))


def test_profile_limits_exposed():
    ss = _left_set()
    lo, hi = ss.limits(["l_aj_4", "l_hj_gripper_1"])
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [2.44346, 0.044])


# ---------------------------------------------------------------- 09.06 양팔 DG-5F-M yaml (joint_profiles + 팔 접미사)
ROBOTS = SIM2REAL / "policy_control/config/robots"
LEFT_HAND_PROFILE = SIM2REAL / "config/openarm_tesollo_left_hand.yaml"


@pytest.mark.parametrize("name", ["dg5f_m_right_real", "dg5f_m_right_fake", "dg5f_m_left_real", "dg5f_m_left_fake"])
def test_dg5f_m_single_arm_yamls_load(name):
    cfg = sources.load_robot_cfg(ROBOTS / f"{name}.yaml")
    side = "left" if "left" in name else "right"
    assert cfg.joint_profiles == (PROFILE, LEFT_HAND_PROFILE) and cfg.joint_profile == PROFILE
    assert cfg.sides == (side,)
    assert cfg.sources["arm"].joints == tuple(f"{side[0]}_aj_{i}" for i in range(1, 8))
    assert len(cfg.sources["ee"].joints) == 20 and cfg.sources["ee"].topic == f"/dg5f_{side}/joint_states"
    assert cfg.sources["ee"].role == "ee" and cfg.sources["ee"].side == ""
    assert set(cfg.groups) == {f"{side}_arm", f"{side}_hand"}
    assert cfg.groups[f"{side}_hand"]["namespace"] == f"dg5f_{side}"
    assert cfg.table.top == (0.200 if name.endswith("fake") else 0.205)


@pytest.mark.parametrize("name", ["dg5f_m_bi_real", "dg5f_m_bi_fake"])
def test_dg5f_m_bimanual_yaml_has_sided_roles(name):
    cfg = sources.load_robot_cfg(ROBOTS / f"{name}.yaml")
    assert cfg.sides == ("left", "right")
    for side in ("left", "right"):
        arm, ee = cfg.sources[f"arm_{side}"], cfg.sources[f"ee_{side}"]
        assert (arm.role, arm.side) == ("arm", side) and (ee.role, ee.side) == ("ee", side)
        assert all(j.startswith(side[0] + "_hj_") for j in ee.joints) and len(ee.joints) == 20
        assert cfg.sources[f"tip_force_{side}"].role == "tip_force"
    assert cfg.sources["object"].role == "object" and cfg.sources["object"].side == ""
    assert set(cfg.groups) == {"left_arm", "left_hand", "right_arm", "right_hand"}


def test_merged_profile_has_both_hands_and_refuses_duplicates(tmp_path):
    prof = sources.load_profile([PROFILE, LEFT_HAND_PROFILE])
    assert prof["l_hj_thumb_2"]["source"] == "lj_dg_1_2" and prof["r_hj_thumb_2"]["source"] == "rj_dg_1_2"
    assert prof["l_hj_thumb_2"]["lower"] == pytest.approx(0.0) and prof["r_hj_thumb_2"]["upper"] == pytest.approx(0.0)
    with pytest.raises(sources.RobotCfgError):
        sources.load_profile([PROFILE, PROFILE])
    assert sources.load_profile(PROFILE) == sources.load_profile([PROFILE])


def test_sided_yaml_rejects_bare_roles_and_object_suffix(tmp_path):
    base = (ROBOTS / "dg5f_m_bi_fake.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(base.replace("  arm_left:", "  arm:"))
    with pytest.raises(sources.RobotCfgError):
        sources.load_robot_cfg(bad)
    bad.write_text(base.replace("  object:", "  object_left:"))
    with pytest.raises(sources.RobotCfgError):
        sources.load_robot_cfg(bad)
    bad.write_text(base.replace("joint_profiles:", "joint_profile: x\njoint_profiles:"))
    with pytest.raises(sources.RobotCfgError):
        sources.load_robot_cfg(bad)


# ---------------------------------------------------------------- 09.06 select_side — 양팔 yaml 에서 한 팔 고르기
RIGHT_HAND_SRC = [f"rj_dg_{f}_{i}" for f in range(1, 6) for i in range(1, 5)]


@pytest.mark.parametrize("side", ["right", "left"])
def test_select_side_keeps_one_arm_with_bare_roles(side):
    cfg = sources.load_robot_cfg(ROBOTS / "dg5f_m_bi_fake.yaml")
    one = sources.select_side(cfg, side)
    assert set(one.sources) == {"arm", "ee", "tip_force", "decoder_target", "object", "head"}
    assert one.sides == (side,)
    for role in ("arm", "ee", "tip_force", "decoder_target"):
        s = one.sources[role]
        assert s.name == role and s.role == role and s.side == side
    assert one.sources["arm"].joints == tuple(f"{side[0]}_aj_{i}" for i in range(1, 8))
    assert all(j.startswith(side[0] + "_hj_") for j in one.sources["ee"].joints)
    assert one.sources["ee"].topic == f"/dg5f_{side}/joint_states"
    assert one.sources["object"] is cfg.sources["object"] and one.sources["head"] is cfg.sources["head"]
    assert one.groups == cfg.groups and one.table == cfg.table
    assert cfg.sides == ("left", "right")                       # 원본은 그대로(새 객체)
    assert sources.select_side(one, side) == one                 # 멱등
    with pytest.raises(sources.RobotCfgError):
        sources.select_side(one, "left" if side == "right" else "right")


def test_select_side_on_single_arm_yaml_checks_the_arm():
    right = sources.load_robot_cfg(ROBOTS / "right_dg5f_fake.yaml")
    assert sources.select_side(right, "right") is right
    with pytest.raises(sources.RobotCfgError, match="right 팔인데 left"):
        sources.select_side(right, "left")
    left = sources.load_robot_cfg(ROBOTS / "left_gripper_fake.yaml")
    assert sources.select_side(left, "left") is left
    with pytest.raises(sources.RobotCfgError):
        sources.select_side(left, "right")
    with pytest.raises(sources.RobotCfgError, match="허용"):
        sources.select_side(left, "both")


def test_source_set_on_selected_bimanual_yaml_snapshots_that_arm_only():
    cfg = sources.select_side(sources.load_robot_cfg(ROBOTS / "dg5f_m_bi_fake.yaml"), "right")
    ss = sources.SourceSet(cfg)
    prof = sources.load_profile(cfg.joint_profiles)
    arm_src = [prof[j]["source"] for j in RIGHT_ARM] + [prof[j]["source"] for j in LEFT_ARM]
    vals = list(np.arange(7.0)) + [9.0] * 7                                # 좌팔 값은 무시돼야 한다
    ss.update_from_joint_state("arm", codec.decode_joint_state(
        codec.encode_joint_state(arm_src, vals, velocity=[0.5] * 14)), now=0.0)
    hand = np.linspace(0.0, 1.0, 20)
    ss.update_from_joint_state("ee", codec.decode_joint_state(
        codec.encode_joint_state(RIGHT_HAND_SRC[::-1], hand[::-1], velocity=list(hand[::-1]))), now=0.0)
    ss.update_from_float_array("tip_force", codec.decode_float_array(
        codec.encode_float_array(np.arange(15.0), ("tip", "axis"), (5, 3), seq=0)), now=0.0)
    cup = codec.PoseSample(np.array([0.4, -0.2, 0.3]), np.array([1.0, 0, 0, 0]), "base_link", 0.0)
    ss.update_from_pose("object", cup, now=0.0)
    st = ss.snapshot(now=0.1)
    np.testing.assert_allclose(st.arm_q, np.arange(7.0))
    assert st.ee_names == tuple(HAND_PROFILE)
    np.testing.assert_allclose(st.ee_q, hand)
    assert st.tip_force.shape == (5, 3) and st.tip_names == tuple(TIPS)
    assert st.missing == () and st.stale == ()
    with pytest.raises(sources.RobotCfgError):
        ss.update_from_joint_state("arm_left", codec.decode_joint_state(codec.encode_joint_state(arm_src, vals)), now=0.0)


def test_source_set_on_unselected_bimanual_yaml_reports_missing_not_crash():
    ss = sources.SourceSet(sources.load_robot_cfg(ROBOTS / "dg5f_m_bi_fake.yaml"))
    st = ss.snapshot(now=0.0)
    assert st.arm_q is None and st.ee_q is None and st.ee_names == ()
    assert {"arm_left", "arm_right", "ee_left", "ee_right", "object"} <= set(st.missing)
