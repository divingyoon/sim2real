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

import numpy as np
import pytest
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from robot_profile import (  # noqa: E402
    HDGP_OPENARM_SRC,
    ROBOT_CONFIG_DIR,
    available_profiles,
    ee_limit_arrays,
    hdgp_task_dir,
    load_joint_profiles,
    load_robot_profile,
)

SIDES = ["right", "left"]
# bi_s 좌우 쌍 — 미러 관계 검증 전용
PROFILES = {"right": "tesollo_bi_s__right", "left": "tesollo_bi_s__left"}
# ★모든 구성을 자동으로 덮는다. 새 프로필을 추가하면 즉시 검증 대상이 된다.
ALL_PROFILES = available_profiles()

# grasp_left_preset.py `_HAND_SIGN` — 손가락별 4관절 좌우 미러 부호(07-28 FK 확정)
HAND_SIGN = {
    "thumb": [-1, -1, -1, -1],
    "index": [-1, 1, 1, 1],
    "middle": [-1, 1, 1, 1],
    "ring": [-1, 1, 1, 1],
    "pinky": [-1, -1, 1, 1],
}


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_profile_loads(name):
    """모든 구성이 매니페스트·계약 검증을 통과해야 한다."""
    p = load_robot_profile(name)
    assert p.acting_side in ("right", "left")
    assert p.ee_dof == len(p.ee_canonical)
    assert p.contract.obs_dim > 0 and p.contract.action_dim > 0
    assert len(p.arm_canonical) == len(p.arm_source) == 7


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_joint_names_follow_manifest(name):
    """관절 이름·순서의 진실원천은 자산 매니페스트다."""
    p = load_robot_profile(name)
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

def _sim_fabrics_assets(profile) -> set[str]:
    """해당 태스크의 env 소스에서 실제로 쓰는 robot_dir_name 을 뽑는다(Isaac 없이 정규식)."""
    env = hdgp_task_dir(profile) / f"grasp_{profile.acting_side}_env.py"
    if not env.exists():
        pytest.skip(f"hdgp 태스크 소스 없음: {env}")
    return set(re.findall(r'robot_dir_name\s*=\s*"([^"]+)"', env.read_text()))


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabrics_asset_matches_sim(name):
    """프로필의 Fabrics 자산이 **학습 sim 과 동일**해야 한다.

    구성마다 자산이 다르다: grasp_v1 은 bi_s(DG-5FS), grasp_sensor 는 openarm_tesollo
    (DG-5F 원본). palm 이 6.5cm 다르므로 혼용하면 obs 36차원과 IK 가 동시에 틀린다.
    """
    p = load_robot_profile(name)
    sim_assets = _sim_fabrics_assets(p)
    assert sim_assets, f"{name} env 에서 robot_dir_name 을 찾지 못함"
    assert p.fabrics.robot_dir in sim_assets, (
        f"Fabrics 자산 불일치 [{name}]: 프로필 {p.fabrics.robot_dir!r} vs sim {sorted(sim_assets)}\n"
        "  bi_s(DG-5FS) 와 openarm_tesollo(DG-5F) 는 palm 이 6.5cm 다르다 — 구성별로 맞춰야 한다."
    )


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabrics_world_matches_sim(name):
    """world 도 sim 과 같아야 한다(좌측은 전용 world)."""
    p = load_robot_profile(name)
    env = hdgp_task_dir(p) / f"grasp_{p.acting_side}_env.py"
    if not env.exists():
        pytest.skip("hdgp 태스크 소스 없음")
    worlds = set(re.findall(r'world_filename\s*=\s*"([^"]+)"', env.read_text()))
    assert p.fabrics.world in worlds, f"{name}: {p.fabrics.world!r} vs sim {sorted(worlds)}"


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


# --------------------------------------------------------------------------
# 유휴(반대편) 팔 — sim 은 rest 고정으로 학습하고 그 팔은 물리 충돌체다
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_idle_arm_resolved(name):
    from robot_profile import idle_arm_rest_pose

    p = load_robot_profile(name)
    other = "l" if p.acting_side == "right" else "r"
    assert p.idle_arm_canonical == tuple(f"{other}_aj_{i}" for i in range(1, 8))
    assert len(p.idle_arm_source) == 7
    assert p.idle_arm_source[0] == f"openarm_{'left' if other == 'l' else 'right'}_joint1"
    rest = idle_arm_rest_pose(p)
    assert len(rest) == 7


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_idle_arm_rest_is_mirror_of_home(name):
    """유휴 팔 rest = 파지 팔 홈의 부호 미러 — sim `_build_home_pose` 가 강제하는 관계."""
    from robot_profile import (
        expected_q_home_arm,
        idle_arm_rest_pose,
        load_arm_mirror_sign,
    )

    p = load_robot_profile(name)
    sign = np.array(load_arm_mirror_sign(p)[:7])
    q_home = np.array(expected_q_home_arm(p))
    rest = np.array(idle_arm_rest_pose(p))
    assert np.allclose(rest, sign * q_home, atol=0.05), (
        f"{name}: 유휴 팔 rest 가 홈의 미러가 아니다\n  rest={rest.round(4)}\n"
        f"  기대={(sign * q_home).round(4)}"
    )


def test_grasp_sensor_home_differs_from_bi_s():
    """자산이 다르면 q_home 도 다르다 — 구성 간 프로필 혼용 금지의 근거."""
    from robot_profile import expected_q_home_arm

    if "tesollo_sensor__right" not in ALL_PROFILES:
        pytest.skip("grasp_sensor 프로필 없음")
    a = np.array(expected_q_home_arm(load_robot_profile("tesollo_sensor__right")))
    b = np.array(expected_q_home_arm(load_robot_profile("tesollo_bi_s__right")))
    assert np.abs(a - b).max() > 0.1, "두 구성의 홈이 같다면 자산 가정이 틀렸다"
