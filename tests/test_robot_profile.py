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

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from robot_profile import (  # noqa: E402
    load_env_cfg_literals,
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
    """관절 이름·순서의 진실원천은 자산 매니페스트다.

    EE 를 매니페스트에서 유도한다. 손가락 20개를 여기 적어 두면 그건 진실원천이 아니라
    **두 번째 사본**이고, DG-5F 가 아닌 구성(좌 2지 그리퍼)에서 바로 어긋난다.
    """
    from robot_profile import load_manifest_joint_order

    p = load_robot_profile(name)
    prefix = p.side_prefix
    order = load_manifest_joint_order(p.manifest_path)

    assert p.arm_canonical == tuple(f"{prefix}_aj_{i}" for i in range(1, 8))
    assert p.ee_canonical == tuple(j for j in order if j.startswith(f"{prefix}_hj_"))
    assert p.ee_canonical, "매니페스트에 이 side 의 EE 관절이 없다"
    assert len(p.arm_source) == 7
    assert len(p.ee_source) == len(p.ee_canonical) == p.ee_dof


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


def _mirrors_the_home_pose(profile) -> bool:
    """sim 이 유휴 팔 rest 를 파지 팔 홈에서 **만들어 내는가**.

    DirectRL 트랙의 `grasp_{side}_env.py` 는 `_ARM_MIRROR_SIGN` 으로 rest 를 홈에서
    파생한다. manager-based 인 gripper/left 에는 그 파일도 그 구성도 없고, 유휴 우팔은
    독립적으로 정한 주차 자세다(`RIGHT_ARM_REST_JOINT_POS = [0, 0.3, 0, 2.0, 0, 0, 0]`).
    거기에 미러 관계를 요구하면 **사실이 아닌 것을 단언**하게 된다.
    """
    return (hdgp_task_dir(profile) / f"grasp_{profile.acting_side}_env.py").exists()


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_idle_arm_rest_is_mirror_of_home(name):
    """유휴 팔 rest = 파지 팔 홈의 부호 미러 — sim `_build_home_pose` 가 강제하는 관계."""
    from robot_profile import (
        expected_q_home_arm,
        idle_arm_rest_pose,
        load_arm_mirror_sign,
    )

    p = load_robot_profile(name)
    if not _mirrors_the_home_pose(p):
        pytest.skip("sim 이 rest 를 홈에서 파생하지 않는 구성 — 아래 테스트가 따로 덮는다")
    sign = np.array(load_arm_mirror_sign(p)[:7])
    q_home = np.array(expected_q_home_arm(p))
    rest = np.array(idle_arm_rest_pose(p))
    assert np.allclose(rest, sign * q_home, atol=0.05), (
        f"{name}: 유휴 팔 rest 가 홈의 미러가 아니다\n  rest={rest.round(4)}\n"
        f"  기대={(sign * q_home).round(4)}"
    )


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_the_idle_arm_rest_is_readable_whether_or_not_it_is_a_mirror(name):
    """위 테스트가 skip 된 구성도 유휴 팔 자세는 반드시 읽혀야 한다.

    실기 유휴 팔이 sim 과 다른 곳에 있으면 장면이 다르고, 학습된 궤적은 그 장면에서
    안전하다는 근거를 잃는다. 미러인지 아닌지와 무관하게 **값이 있어야** 확인할 수 있다.
    """
    from robot_profile import idle_arm_rest_pose

    p = load_robot_profile(name)

    rest = idle_arm_rest_pose(p)

    assert len(rest) == 7
    assert all(np.isfinite(rest))


def test_grasp_sensor_home_differs_from_bi_s():
    """자산이 다르면 q_home 도 다르다 — 구성 간 프로필 혼용 금지의 근거."""
    from robot_profile import expected_q_home_arm

    if "tesollo_sensor__right" not in ALL_PROFILES:
        pytest.skip("grasp_sensor 프로필 없음")
    a = np.array(expected_q_home_arm(load_robot_profile("tesollo_sensor__right")))
    b = np.array(expected_q_home_arm(load_robot_profile("tesollo_bi_s__right")))
    assert np.abs(a - b).max() > 0.1, "두 구성의 홈이 같다면 자산 가정이 틀렸다"


# --------------------------------------------------------------------------
# manager-based 구성 (gripper/left/grasp_sensor) — 검증이 조용히 사라지면 안 된다
# --------------------------------------------------------------------------

MANAGER_BASED = "gripper_left"


def _needs_gripper_left():
    if MANAGER_BASED not in ALL_PROFILES:
        pytest.skip(f"{MANAGER_BASED} 프로필 없음")
    return load_robot_profile(MANAGER_BASED)


def test_the_fabrics_check_actually_runs_for_a_manager_based_task():
    """★skip 은 통과가 아니다.

    `test_fabrics_asset_matches_sim` 은 `grasp_{side}_env.py` 를 찾는다. manager-based
    태스크에는 그 파일이 없으니 조용히 skip 되고, 자산이 어긋나도 아무도 모른다 —
    이 저장소가 obs 계약 추출기에서 이미 한 번 당한 실패 방식이다. 그러니 그 태스크의
    자산 이름이 실제로 **어딘가에서 읽혀 대조되는지**를 따로 못박는다.
    """
    from robot_profile import sim_fabrics_assets, sim_fabrics_worlds

    profile = _needs_gripper_left()

    assets = sim_fabrics_assets(profile)
    worlds = sim_fabrics_worlds(profile)

    assert assets, "sim 소스에서 fabric 자산 이름을 하나도 못 찾았다 — 검사가 무의미하다"
    assert worlds, "sim 소스에서 fabric world 이름을 하나도 못 찾았다"
    assert profile.fabrics.robot_dir in assets, (profile.fabrics.robot_dir, sorted(assets))
    assert profile.fabrics.world in worlds, (profile.fabrics.world, sorted(worlds))


def test_the_gripper_profile_names_the_single_jaw_the_manifest_declares():
    """자산 매니페스트에 좌 그리퍼 관절은 `l_hj_gripper_1` 하나뿐이다.

    sim USD 는 mimic 을 잃어 두 조를 다 지령하지만(preset 주석), 실기 URDF 는 mimic 이
    살아 있고 매니페스트도 한 개만 싣는다. 배포가 두 개를 보내려 하면 여기서 걸린다.
    """
    profile = _needs_gripper_left()

    assert profile.ee_canonical == ("l_hj_gripper_1",)
    assert profile.ee_dof == 1


def test_the_gripper_stroke_matches_the_description(): 
    """프로필 한계가 실기 스트로크(0.044 m)여야 한다 — robot_control 프로필에서 온다."""
    profile = _needs_gripper_left()

    limit = profile.joint_limits["l_hj_gripper_1"]

    assert limit["lower"] == pytest.approx(0.0)
    assert limit["upper"] == pytest.approx(0.044)


def test_a_manager_based_contract_is_verified_against_the_checkpoint():
    """차원의 진실원천이 소스에 없으면 체크포인트다 — 건너뛰지 않는다.

    이 태스크는 `*_constants.py` 가 없다. obs 차원은 `ObservationManager` 가 런타임에
    조립하므로 정적으로는 셀 수 없다. 그래서 계약을 **학습 산출물**에 대고 검증한다:
    actor 첫 층이 obs, mu 헤드가 action 이다. 추측이 들어갈 자리가 없다.
    """
    from robot_profile import checkpoint_contract

    profile = _needs_gripper_left()
    checkpoint = profile.contract.checkpoint
    assert checkpoint is not None, "manager-based 구성은 체크포인트를 지목해야 한다"
    if not checkpoint.is_file():
        pytest.skip(f"체크포인트 부재: {checkpoint}")

    obs_dim, action_dim = checkpoint_contract(checkpoint)

    assert (obs_dim, action_dim) == (profile.contract.obs_dim, profile.contract.action_dim)


# --------------------------------------------------------------------------
# 액션 규약 — 짐작하면 §8 의 사고가 그대로 재현된다
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_every_profile_declares_what_its_action_means(name):
    """규약을 안 적으면 배포가 a=0 의 뜻을 짐작한다.

    §8 이 그 사고다: sim 은 a=0 을 "컵 정준 pregrasp 로 접근"으로, 배포는 "홈에 머물라"로
    읽었고 팔은 홈 근처에서만 움직였다. 값이 아니라 **의미**를 계약에 넣는다.
    """
    from robot_profile import ACTION_CONVENTIONS

    p = load_robot_profile(name)

    assert p.action_convention in ACTION_CONVENTIONS


def test_a_profile_without_a_declared_convention_is_refused(tmp_path):
    """기본값을 두지 않는다 — 기본값은 조용한 짐작의 다른 이름이다."""
    source = ROBOT_CONFIG_DIR / f"{ALL_PROFILES[0]}.yaml"
    raw = yaml.safe_load(source.read_text())
    raw.pop("action", None)
    broken = tmp_path / "no_convention.yaml"
    broken.write_text(yaml.safe_dump(raw, allow_unicode=True))

    with pytest.raises(ValueError, match="action.convention"):
        load_robot_profile(broken)


def test_the_two_conventions_partition_the_profiles():
    """모든 구성이 정확히 한 쪽에 속한다 — 어느 쪽에도 없으면 계약 테스트에서 사라진다."""
    from robot_profile import ABSOLUTE_PALM, DELTA_ANCHOR, profiles_with_convention

    delta = set(profiles_with_convention(DELTA_ANCHOR))
    absolute = set(profiles_with_convention(ABSOLUTE_PALM))

    assert delta.isdisjoint(absolute)
    assert delta | absolute == set(ALL_PROFILES)


def test_the_absolute_convention_is_what_the_sim_action_term_implements():
    """선언만 하고 sim 이 다르면 선언이 거짓말이 된다.

    절대 규약의 표식은 액션 항이 박스 중심에 정규화 액션을 얹는다는 것이다. 델타 규약이면
    거기에 기준점(anchor) 버퍼가 들어간다.
    """
    from robot_profile import ABSOLUTE_PALM

    profile = _needs_gripper_left()
    assert profile.action_convention == ABSOLUTE_PALM

    term = hdgp_task_dir(profile) / "grasp_left_fabric_action.py"
    if not term.is_file():
        pytest.skip(f"액션 항 소스 없음: {term}")
    source = term.read_text()

    assert "self._box_center + actions[:, :3].clamp(-1.0, 1.0) * self._box_half" in source
    assert "pregrasp" not in source, "절대 규약이라 선언했는데 기준점 개념이 소스에 있다"


# ── RNN 체크포인트의 관측 차원 ─────────────────────────────────────────
# fab_test42(vision-3090) 실측에서 드러났다: actor_mlp 입력 1096 = LSTM 1024 + obs 72.
# 첫 층을 obs 로 읽던 구현은 RNN 정책에 대해 조용히 틀린 값을 냈다.

def test_a_recurrent_checkpoints_obs_comes_from_the_normalizer_not_the_mlp():
    import torch

    from robot_profile import _obs_dim_from_state

    state = {
        "a2c_network.actor_mlp.0.weight": torch.zeros(512, 1096),
        "a2c_network.rnn.rnn.weight_ih_l0": torch.zeros(4096, 72),
        "a2c_network.rnn.rnn.weight_hh_l0": torch.zeros(4096, 1024),
        "running_mean_std.running_mean": torch.zeros(72),
    }
    assert _obs_dim_from_state(state, "t") == 72


def test_a_recurrent_checkpoint_without_a_normalizer_falls_back_to_the_rnn_input():
    import torch

    from robot_profile import _obs_dim_from_state

    state = {
        "a2c_network.actor_mlp.0.weight": torch.zeros(512, 1096),
        "a2c_network.rnn.rnn.weight_ih_l0": torch.zeros(4096, 72),
    }
    assert _obs_dim_from_state(state, "t") == 72


def test_a_plain_mlp_checkpoint_still_reads_the_first_layer():
    import torch

    from robot_profile import _obs_dim_from_state

    state = {"a2c_network.actor_mlp.0.weight": torch.zeros(256, 36)}
    assert _obs_dim_from_state(state, "t") == 36


def test_a_recurrent_checkpoint_with_no_readable_obs_refuses_the_mlp_guess():
    import pytest
    import torch

    from robot_profile import _obs_dim_from_state

    state = {
        "a2c_network.actor_mlp.0.weight": torch.zeros(512, 1096),
        "a2c_network.rnn.rnn.weight_hh_l0": torch.zeros(4096, 1024),
    }
    with pytest.raises(KeyError, match="concat_input"):
        _obs_dim_from_state(state, "t")


def test_recurrence_is_detectable_because_deployment_must_carry_hidden_state():
    from robot_profile import checkpoint_is_recurrent

    assert checkpoint_is_recurrent({"a2c_network.rnn.rnn.weight_hh_l0": 1})
    assert not checkpoint_is_recurrent({"a2c_network.actor_mlp.0.weight": 1})


# ── 클래스별 리터럴 읽기 ───────────────────────────────────────────────────
TWO_CLASSES = '''
class A:
    shared: float = 1.0
    only_a: int = 7

class B:
    shared: float = 2.0
    only_b: int = 9
'''


def _cfg(tmp_path, text):
    p = tmp_path / "cfg.py"
    p.write_text(text)
    return p


def test_literals_without_a_class_still_fail_loudly_on_conflict(tmp_path):
    """같은 이름이 다른 값으로 두 번 나오면 어느 쪽이 진짜인지 추측하지 않는다."""
    import pytest
    with pytest.raises(ValueError, match="shared"):
        load_env_cfg_literals(_cfg(tmp_path, TWO_CLASSES))


def test_literals_can_be_scoped_to_one_class(tmp_path):
    """한 파일에 좌·우 cfg 가 같이 사는 경우가 있다 — 어느 클래스인지 말할 수 있어야 한다."""
    out = load_env_cfg_literals(_cfg(tmp_path, TWO_CLASSES), class_name="B")

    assert out["shared"] == 2.0
    assert out["only_b"] == 9
    assert "only_a" not in out


def test_scoping_to_a_missing_class_says_so(tmp_path):
    import pytest
    with pytest.raises(KeyError, match="없다"):
        load_env_cfg_literals(_cfg(tmp_path, TWO_CLASSES), class_name="Z")
