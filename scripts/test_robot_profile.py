#!/usr/bin/env python3
"""로봇 구성 프로필 로더 + 자산 신원 검증 테스트.

★가장 중요한 테스트는 `test_fabrics_asset_matches_sim` 이다. 배포가 Fabrics 자산을
하드코딩(기본값 `openarm_tesollo`)한 탓에 sim 이 `openarm_tesollo_bi_s` 로 옮긴 뒤에도
배포는 구 자산에 머물러 **palm 이 6.5cm 어긋난 채** 동작했다(실측 palm_link z
0.12863 vs 0.1935). palm 은 obs 154D 중 36차원의 기준이자 Fabrics IK 목표다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from robot_profile import (  # noqa: E402
    HDGP_OPENARM_SRC,
    ROBOT_CONFIG_DIR,
    ee_limit_arrays,
    load_joint_profiles,
    load_robot_profile,
)

SIDES = ["right", "left"]
PROFILES = {"right": "tesollo_bi_s__right", "left": "tesollo_bi_s__left"}

# grasp_left_preset.py `_HAND_SIGN` — 손가락별 4관절 좌우 미러 부호(07-28 FK 확정)
HAND_SIGN = {
    "thumb": [-1, -1, -1, -1],
    "index": [-1, 1, 1, 1],
    "middle": [-1, 1, 1, 1],
    "ring": [-1, 1, 1, 1],
    "pinky": [-1, -1, 1, 1],
}


@pytest.mark.parametrize("side", SIDES)
def test_profile_loads(side):
    p = load_robot_profile(PROFILES[side])
    assert p.acting_side == side
    assert p.ee_type == "tesollo_dg5f"
    assert p.ee_dof == 20
    assert p.contract.obs_dim == 154
    assert p.contract.action_dim == 21


@pytest.mark.parametrize("side", SIDES)
def test_joint_names_follow_manifest(side):
    """관절 이름·순서의 진실원천은 자산 매니페스트다."""
    p = load_robot_profile(PROFILES[side])
    prefix = p.side_prefix
    assert p.arm_canonical == tuple(f"{prefix}_aj_{i}" for i in range(1, 8))
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    assert p.ee_canonical == tuple(
        f"{prefix}_hj_{f}_{j}" for f in fingers for j in range(1, 5)
    )
    assert len(p.arm_source) == 7 and len(p.ee_source) == 20


@pytest.mark.parametrize("side,arm_src,ee_src", [
    ("right", "openarm_right_joint1", "rj_dg_1_1"),
    ("left", "openarm_left_joint1", "lj_dg_1_1"),
])
def test_source_mapping(side, arm_src, ee_src):
    p = load_robot_profile(PROFILES[side])
    assert p.arm_source[0] == arm_src
    assert p.ee_source[0] == ee_src


@pytest.mark.parametrize("side,topic_key,expected", [
    ("right", "arm_traj", "/right_joint_trajectory_controller/joint_trajectory"),
    ("right", "ee_traj", "/dg5f_right/dg5f_right_controller/joint_trajectory"),
    ("left", "arm_traj", "/left_joint_trajectory_controller/joint_trajectory"),
    ("left", "ee_traj", "/dg5f_left/dg5f_left_controller/joint_trajectory"),
])
def test_topics(side, topic_key, expected):
    assert load_robot_profile(PROFILES[side]).topics[topic_key] == expected


def test_ee_limits_mirror_between_sides():
    """좌손 한계는 우손의 `_HAND_SIGN` 미러여야 한다(벤더 좌손 URDF 와 20/20 확인된 규칙)."""
    r_lo, r_hi = ee_limit_arrays(load_robot_profile(PROFILES["right"]))
    l_lo, l_hi = ee_limit_arrays(load_robot_profile(PROFILES["left"]))
    idx = 0
    for finger, signs in HAND_SIGN.items():
        for s in signs:
            if s < 0:
                assert l_lo[idx] == pytest.approx(-r_hi[idx], abs=1e-6), f"{finger}[{idx}] lower"
                assert l_hi[idx] == pytest.approx(-r_lo[idx], abs=1e-6), f"{finger}[{idx}] upper"
            else:
                assert l_lo[idx] == pytest.approx(r_lo[idx], abs=1e-6), f"{finger}[{idx}] lower"
                assert l_hi[idx] == pytest.approx(r_hi[idx], abs=1e-6), f"{finger}[{idx}] upper"
            idx += 1
    assert idx == 20


# --------------------------------------------------------------------------
# ★자산 신원 — 이 테스트가 6.5cm palm 어긋남의 재발 방어선이다
# --------------------------------------------------------------------------

def _sim_fabrics_assets(side: str) -> set[str]:
    """hdgp env 소스에서 실제로 쓰는 robot_dir_name 을 뽑는다(Isaac 없이 정규식)."""
    env = (HDGP_OPENARM_SRC / "openarm" / "tesollo" / side / "grasp_v1"
           / f"grasp_{side}_env.py")
    if not env.exists():
        pytest.skip(f"hdgp grasp_v1 소스 없음: {env}")
    return set(re.findall(r'robot_dir_name\s*=\s*"([^"]+)"', env.read_text()))


@pytest.mark.parametrize("side", SIDES)
def test_fabrics_asset_matches_sim(side):
    """프로필의 Fabrics 자산이 **학습 sim 과 동일**해야 한다."""
    p = load_robot_profile(PROFILES[side])
    sim_assets = _sim_fabrics_assets(side)
    assert sim_assets, f"{side} env 에서 robot_dir_name 을 찾지 못함"
    assert p.fabrics.robot_dir in sim_assets, (
        f"Fabrics 자산 불일치: 프로필 {p.fabrics.robot_dir!r} vs sim {sorted(sim_assets)}\n"
        "  구 자산(openarm_tesollo)은 palm 이 6.5cm 짧다 — obs 36차원과 IK 목표가 동시에 틀린다."
    )


@pytest.mark.parametrize("side,world", [
    ("right", "open_tesollo_boxes_no_table"),
    ("left", "open_tesollo_left_boxes_no_table"),
])
def test_fabrics_world(side, world):
    assert load_robot_profile(PROFILES[side]).fabrics.world == world


# --------------------------------------------------------------------------
# 실패 경로 — 조용히 넘어가면 안 되는 것들
# --------------------------------------------------------------------------

def test_unknown_profile_name_raises():
    with pytest.raises(FileNotFoundError):
        load_robot_profile("존재하지_않는_구성")


def test_bad_acting_side_raises(tmp_path):
    src = yaml.safe_load((ROBOT_CONFIG_DIR / "tesollo_bi_s__right.yaml").read_text())
    src["acting_side"] = "both"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(src))
    with pytest.raises(ValueError, match="acting_side"):
        load_robot_profile(bad)


def test_contract_mismatch_raises(tmp_path):
    """프로필 obs_dim 이 hdgp 와 다르면 로드 자체가 실패해야 한다."""
    if not HDGP_OPENARM_SRC.exists():
        pytest.skip("hdgp 소스 없음")
    src = yaml.safe_load((ROBOT_CONFIG_DIR / "tesollo_bi_s__right.yaml").read_text())
    src["contract"]["obs_dim"] = 114          # 구 계약
    bad = tmp_path / "stale_contract.yaml"
    bad.write_text(yaml.safe_dump(src))
    with pytest.raises(ValueError, match="obs 계약 불일치"):
        load_robot_profile(bad)


def test_ee_dof_mismatch_raises(tmp_path):
    src = yaml.safe_load((ROBOT_CONFIG_DIR / "tesollo_bi_s__right.yaml").read_text())
    src["end_effector"]["dof"] = 6            # rh56f1 착각
    bad = tmp_path / "bad_dof.yaml"
    bad.write_text(yaml.safe_dump(src))
    with pytest.raises(ValueError, match="end_effector.dof"):
        load_robot_profile(bad)


def test_joint_profile_conflict_raises(tmp_path):
    """같은 canonical 이 값이 다르게 두 번 나오면 조용히 덮어쓰지 말고 실패."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    base = {"canonical": "r_aj_1", "source": "openarm_right_joint1", "sign": 1,
            "unit": "rad", "lower": -1.0, "upper": 1.0}
    a.write_text(yaml.safe_dump({"joints": [base]}))
    b.write_text(yaml.safe_dump({"joints": [{**base, "upper": 2.0}]}))
    with pytest.raises(ValueError, match="충돌"):
        load_joint_profiles([a, b])


def test_joint_profile_duplicate_identical_ok(tmp_path):
    a = tmp_path / "a.yaml"
    base = {"canonical": "r_aj_1", "source": "openarm_right_joint1", "sign": 1,
            "unit": "rad", "lower": -1.0, "upper": 1.0}
    a.write_text(yaml.safe_dump({"joints": [base]}))
    merged = load_joint_profiles([a, a])
    assert merged["r_aj_1"]["upper"] == 1.0
