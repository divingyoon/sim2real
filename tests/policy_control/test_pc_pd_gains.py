"""M5 — pd_gains: control_gains.yaml 게이트(contract.compare_gains/require_gains 재사용).

좌 v2B25 OK · 우 g1 OK(kd 는 r2s fit → 정보, >5 는 MIT 패킷 한계 밖 'impossible') ·
우 d3(KUKA 게인 학습판) MISMATCH → accept_sim_mismatch 없이는 거부. DG-5F 손 PID 기대값은
설정에서 노출만 한다(검증은 M7 노드).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from policy_control import contract as C
from policy_control import contract_build as B
from policy_control import pd_gains as GN
from policy_control import pd_law as L

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
CONFIG = SIM2REAL / "policy_control" / "config"
LEFT_CONTRACT = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_CONTRACT = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
D3_RUN = SIM2REAL / "logs/policy/right_d3"
GAINS = RL_WS / "urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml"

pytestmark = pytest.mark.unit
needs_left = pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")
needs_right = pytest.mark.skipif(not RIGHT_CONTRACT.exists(), reason="right_g1 contract 없음")
needs_d3 = pytest.mark.skipif(not (D3_RUN / "nn").exists(), reason="right_d3 run dir 없음")


def _cfg(accept: bool = False) -> GN.GainsBlock:
    return GN.GainsBlock(yaml=GAINS, accept_sim_mismatch=accept)


@needs_left
def test_left_ok_and_kd_matches_driver():
    rep = GN.load_and_check(_cfg(), C.load_contract(LEFT_CONTRACT))
    assert rep.ok and rep.reasons == [] and rep.kd_note == ""
    assert rep.real_kp == [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]
    assert rep.accepted_mismatch is False


@needs_right
def test_right_g1_is_refused_by_the_gate_unless_explicitly_accepted():
    """★2026-09-06 벤더 전용 규칙 이후 g1(r2s kd)은 engage 를 통과하지 못한다.

    이전 계약은 "kd 는 정보"라 게이트를 통과했다. 지금은 kd 도 게이트다 — 실기 모터가
    받지 않는 kd 로 학습한 정책이 조용히 실기에 올라가는 경로를 막는 것이 목적이다.
    """
    contract = C.load_contract(RIGHT_CONTRACT)
    with pytest.raises(C.GainMismatch):
        GN.load_and_check(_cfg(accept=False), contract)
    rep = GN.load_and_check(_cfg(accept=True), contract)
    assert not rep.ok and rep.accepted_mismatch is True
    assert all("kd" in r for r in rep.reasons), "kp 는 이미 벤더값이었다"
    assert "impossible" in rep.kd_note                           # kd 7.053 > MIT_KD_MAX 5
    assert rep.impossible_kd == ("r_aj_1", "r_aj_3", "r_aj_4")


@needs_d3
def test_right_d3_mismatch_refused_unless_accepted():
    d3 = B.build_contract(D3_RUN)
    with pytest.raises(C.GainMismatch):
        GN.load_and_check(_cfg(accept=False), d3)
    rep = GN.load_and_check(_cfg(accept=True), d3)
    assert not rep.ok and rep.reasons and rep.accepted_mismatch is True


@needs_left
def test_missing_gains_yaml_is_an_error(tmp_path):
    with pytest.raises(GN.GainsError):
        GN.load_and_check(GN.GainsBlock(yaml=tmp_path / "nope.yaml", accept_sim_mismatch=False),
                          C.load_contract(LEFT_CONTRACT))


def test_gains_block_validation(tmp_path):
    with pytest.raises(GN.GainsError):
        GN.GainsBlock(yaml=GAINS, accept_sim_mismatch="no")


def test_expected_hand_gains_from_config():
    right = L.load_pd_config(CONFIG / "pd_right.yaml")
    left = L.load_pd_config(CONFIG / "pd_left.yaml")
    assert GN.expected_hand_gains(right) == GN.HandGains(pid_p=1.5, pid_d=0.0)   # 벤더 PID(09.06)
    assert GN.expected_hand_gains(left) is None


def test_configs_point_at_the_driver_gains_file():
    for name in ("pd_left.yaml", "pd_right.yaml"):
        cfg = L.load_pd_config(CONFIG / name)
        assert cfg.gains.yaml.resolve() == GAINS.resolve()
        assert cfg.gains.accept_sim_mismatch is False


@needs_left
def test_malformed_gains_yaml_is_a_gains_error(tmp_path):
    p = tmp_path / "gains.yaml"
    p.write_text("joint1: {kp: 70.0, kd: 2.75}\n")            # joint2..7 없음
    with pytest.raises(GN.GainsError):
        GN.load_and_check(GN.GainsBlock(yaml=p, accept_sim_mismatch=False), C.load_contract(LEFT_CONTRACT))


# ---------------------------------------------------------------- 09.06 양팔(asset 계약) — 팔별 게이트
ASSET_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
needs_asset = pytest.mark.skipif(not ASSET_CONTRACT.exists(), reason="asset contract 없음")


@needs_asset
def test_asset_contract_gate_passes_per_side():
    contract = C.load_contract(ASSET_CONTRACT)
    for side in ("left", "right"):
        rep = GN.load_and_check(_cfg(), contract, side=side)
        assert rep.ok and rep.reasons == [] and rep.kd_note == "" and rep.impossible_kd == ()
        assert rep.real_kp == [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]
    with pytest.raises(GN.GainsError):
        GN.load_and_check(_cfg(), contract, side="up")


@needs_asset
def test_side_gate_matches_legacy_primary_gate():
    """side 를 안 주면 primary side 를 본다 — 같은 계약에서 두 경로가 같은 답을 내야 한다.

    ★자산(제어 전용) 계약을 쓴다. g1 은 09.06 벤더 전용 규칙에서 게이트를 통과하지
    못하므로(r2s kd) 이 동치 검사의 재료로 쓸 수 없다.
    """
    contract = C.load_contract(ASSET_CONTRACT)
    primary = contract.primary_side
    legacy = GN.load_and_check(_cfg(), contract)
    sided = GN.load_and_check(_cfg(), contract, side=primary)
    assert legacy == sided and legacy.ok
    other = next(s for s in contract.side_names if s != primary)
    assert GN.load_and_check(_cfg(), contract, side=other).ok      # 양팔 모두 벤더값
    with pytest.raises(GN.GainsError):
        GN.load_and_check(_cfg(), C.load_contract(LEFT_CONTRACT), side="right")   # 계약에 없는 팔


def test_dg5f_m_configs_point_at_the_driver_gains_file_and_expect_hand_pid():
    for name in ("pd_dg5f_m.yaml", "pd_dg5f_m_fake.yaml"):
        cfg = L.load_pd_config(CONFIG / name)
        assert cfg.gains.yaml.resolve() == GAINS.resolve() and cfg.gains.accept_sim_mismatch is False
        # ★2026-09-06 벤더 전용 규칙: DG-5F 손도 벤더 기본 p 1.5 / d 0 (구 튜닝 4.5 폐기).
        assert GN.expected_hand_gains(cfg) == GN.HandGains(pid_p=1.5, pid_d=0.0)
    assert L.load_pd_config(CONFIG / "pd_dg5f_m.yaml").execute is False
    assert L.load_pd_config(CONFIG / "pd_dg5f_m_fake.yaml").execute is True


# ---------------------------------------------------------------- 벤더 게인 사본 일치 (2026-09-06 규칙)
def test_every_control_gains_copy_in_the_workspace_agrees():
    """팔 게인 사본이 갈라지면 **게이트가 대조하는 파일과 로봇이 읽는 파일이 달라진다**.

    실기 kp/kd 는 `robot_control/.../openarm_description/config/arm/v10/control_gains.yaml`
    이 xacro 를 통해 하드웨어 인터페이스의 kp_/kd_ 로 들어가 MIT 패킷에 실린다. 배포 계약
    게이트는 `urdf/vendor/...` 사본을 본다. 학습(hdgp)은 패키지 안 제 사본을 본다. 셋(과
    install 사본들)이 같은 값이어야 "벤더 게인만 쓴다"가 성립한다.
    """
    copies = sorted(RL_WS.rglob("control_gains.yaml"))
    hdgp_copy = RL_WS / "hdgp/source/openarm/openarm/agnostic/modules/vendor_arm_control_gains.yaml"
    if hdgp_copy.exists():
        copies.append(hdgp_copy)
    assert len(copies) >= 2, f"사본이 하나뿐이다 — 경로 규칙이 바뀌었나: {copies}"
    loaded = {p: C.load_driver_gains(p) for p in copies}
    reference = loaded[GAINS]
    differing = {str(p.relative_to(RL_WS)): g for p, g in loaded.items() if g != reference}
    assert not differing, f"벤더 게인 사본이 갈라졌다(기준 {GAINS}): {differing}"
    assert reference == {1: (70.0, 2.75), 2: (70.0, 2.5), 3: (70.0, 2.0), 4: (60.0, 2.0),
                         5: (10.0, 0.7), 6: (10.0, 0.6), 7: (10.0, 0.5)}


def test_hand_pid_expectation_equals_the_dg5f_vendor_driver_file():
    """손 기대 PID 가 벤더 드라이버 yaml 과 같은가 — 팔과 같은 규칙을 손에도 적용한다."""
    vendor = RL_WS / "urdf/vendor/delto_m_ros2/dg5f_driver/config/dg5f_both_pid_all_controller.yaml"
    if not vendor.is_file():
        pytest.skip(f"DG-5F 벤더 PID 원본 없음: {vendor}")
    import re
    gains = {(float(m.group(1)), float(m.group(3)))
             for m in re.finditer(r"p:\s*([0-9.]+),\s*i:\s*([0-9.]+),\s*d:\s*([0-9.]+)", vendor.read_text())}
    assert len(gains) == 1, f"벤더 파일 안에서 관절마다 게인이 다르다: {sorted(gains)}"
    vendor_p, vendor_d = gains.pop()
    for name in ("pd_dg5f_m.yaml", "pd_dg5f_m_fake.yaml", "pd_right.yaml"):
        cfg = L.load_pd_config(CONFIG / name)
        assert GN.expected_hand_gains(cfg) == GN.HandGains(pid_p=vendor_p, pid_d=vendor_d), name
    for robot in sorted((CONFIG / "robots").glob("*dg5f*.yaml")):
        text = robot.read_text()
        assert "pid_p: 4.5" not in text, f"{robot.name}: 폐기된 튜닝 게인 4.5 가 남아 있다"
