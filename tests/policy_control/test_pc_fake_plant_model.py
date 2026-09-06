"""M8 — fake plant: MockArm 이 MIT 3중 지령(q*, q̇*, τ_ff)과 중력을 먹는다.

τ = kp(q*−q) + kd(q̇*−q̇) + τ_ff − g(q) − Fc·sgn(q̇). 기존 `step(cmd)` 호출은 그대로
(q̇*=0, τ_ff=0, 중력 없음) 동작해야 한다 — 기존 fake_arm_bridge/테스트 호환.
"""
from __future__ import annotations

import numpy as np
import pytest

from arm_pd_model import MockArm

KP = np.array([70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0])
KD = np.array([2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5])
FC = np.full(7, 0.05)
I = np.array([0.5, 0.5, 0.2, 0.2, 0.05, 0.05, 0.02])


def _arm(**kw):
    return MockArm(np.zeros(7), "pd", max_vel=99.0, dt=1 / 100, kp=KP, kd=KD, fc=FC, inertia=I, **kw)


def test_legacy_step_signature_unchanged():
    arm = _arm()
    for _ in range(300):
        arm.step(np.full(7, 0.3))
    assert np.allclose(arm.q, 0.3, atol=2e-2)


def test_tau_ff_holds_against_gravity():
    """중력 g(q)=const 를 τ_ff 로 정확히 상쇄하면 세트포인트에 정확히 머문다."""
    g = np.array([0.0, 8.0, 0.0, 4.0, 0.0, 0.5, 0.0])
    arm = _arm(gravity=lambda q: g)
    for _ in range(400):
        arm.step(np.zeros(7))                       # 보상 없음 → 처짐
    droop = arm.q.copy()
    assert np.all(droop[[1, 3, 5]] < -1e-3)          # 중력 방향으로 처진다
    arm2 = _arm(gravity=lambda q: g)
    for _ in range(400):
        arm2.step(np.zeros(7), qd_cmd=np.zeros(7), tau_ff=g)   # 완전 보상
    assert np.allclose(arm2.q, 0.0, atol=1e-4)


def test_velocity_feedforward_reduces_lag():
    """q̇* 를 실으면 등속 추종의 정상상태 지연 (kd/kp)·v 가 사라진다."""
    v = 0.5
    lag_no_ff, lag_ff = [], []
    for use_ff in (False, True):
        arm = _arm()
        q_cmd = np.zeros(7)
        for _ in range(300):
            q_cmd = q_cmd + v * arm.dt
            arm.step(q_cmd, qd_cmd=(np.full(7, v) if use_ff else np.zeros(7)), tau_ff=np.zeros(7))
        (lag_ff if use_ff else lag_no_ff).append(float(np.abs(q_cmd - arm.q)[0]))
    assert lag_no_ff[0] > 0.5 * (KD[0] / KP[0]) * v      # JTC 시대 지연이 보인다
    assert lag_ff[0] < 0.25 * lag_no_ff[0]                # 속도 전향으로 대부분 사라진다


def test_effort_output_is_reported():
    arm = _arm(gravity=lambda q: np.zeros(7))
    arm.step(np.full(7, 0.1), qd_cmd=np.zeros(7), tau_ff=np.zeros(7))
    tau = arm.tau
    assert tau.shape == (7,)
    assert np.all(tau[:4] > 0.0)                         # 스프링이 목표 쪽으로 민다


def test_rate_model_ignores_torque_inputs():
    arm = MockArm(np.zeros(7), "rate", max_vel=1.0, dt=1 / 100)
    arm.step(np.full(7, 0.1), qd_cmd=np.ones(7), tau_ff=np.ones(7))
    assert np.allclose(arm.q, 0.01)


def test_bad_shapes_are_rejected():
    arm = _arm()
    with pytest.raises(ValueError):
        arm.step(np.zeros(7), qd_cmd=np.zeros(6))
    with pytest.raises(ValueError):
        arm.step(np.zeros(7), tau_ff=np.zeros(3))


# ------------------------------------------------------------------ 09.06 계약 모드 fake 플랜트(한 팔 단위)
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]
ASSET_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
BI_ROBOT = SIM2REAL / "policy_control/config/robots/dg5f_m_bi_fake.yaml"
PD_FAKE = SIM2REAL / "policy_control/config/pd_dg5f_m_fake.yaml"
needs_asset = pytest.mark.skipif(not ASSET_CONTRACT.exists(), reason="asset contract 없음")


@pytest.fixture(scope="module")
def plant_side():
    import sys

    sys.path.insert(0, str(SIM2REAL / "scripts" / "fakes"))
    import fake_arm_side as S

    from policy_control import contract as C
    from policy_control.sources import load_profile, load_robot_cfg

    contract = C.load_contract(ASSET_CONTRACT)
    profile = load_profile(load_robot_cfg(BI_ROBOT).joint_profiles)
    return S, contract, profile


@needs_asset
def test_side_spec_from_contract_names_and_home(plant_side):
    S, contract, profile = plant_side
    left = S.side_spec_from_contract(contract, profile, "left")
    assert left.canonical == tuple(f"l_aj_{i}" for i in range(1, 8))
    assert left.source == tuple(f"openarm_left_joint{i}" for i in range(1, 8)) and np.all(left.sign == 1.0)
    assert np.allclose(left.home, contract.side("left").home_arm) and left.jtc_topic.startswith("/left_joint_trajectory")
    assert S.side_spec_from_contract(contract, profile, "right").source[0] == "openarm_right_joint1"


@needs_asset
def test_side_arm_commands_by_name_and_reports_rows(plant_side):
    S, contract, profile = plant_side
    spec = S.side_spec_from_contract(contract, profile, "right")
    arm = S.SideArm(spec, model="rate", max_vel=1.0, dt=0.01, kp=None, kd=None, fc=None, inertia=None)
    arm.set_jtc(list(spec.source[::-1]), [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])      # 이름으로 재배열
    np.testing.assert_allclose(arm.cmd, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert arm.forward_vector([1.0] * 6) is None and arm.received == 1
    arm.set_forward("position", np.full(7, 0.05))
    arm.step(mit=True)
    names, pos, vel, eff = arm.rows(with_effort=False)
    assert names == list(spec.source) and np.allclose(pos, 0.01) and eff == [0.0] * 7 and len(vel) == 7
    with pytest.raises(ValueError):
        arm.set_forward("torque", np.zeros(7))
    s_names, s_pos, s_vel, s_eff = S.static_rows(spec, np.full(7, 0.2))
    assert s_names == list(spec.source) and np.allclose(s_pos, 0.2) and s_vel == [0.0] * 7


@needs_asset
def test_plant_gravity_is_the_pd_yaml_model(plant_side):
    from policy_control import pd_gravity as G
    from policy_control import pd_law as L

    S, contract, profile = plant_side
    q = np.array([0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0])
    for side in ("left", "right"):
        g_plant = S.gravity_from_pd_config(PD_FAKE, contract, side)(q)
        g_pd = G.make_gravity(L.load_pd_config(PD_FAKE).gravity, contract, side=side)(q)
        np.testing.assert_allclose(g_plant, g_pd)
        spec = S.side_spec_from_contract(contract, profile, side)
        urdf = SIM2REAL.parent / contract.asset.urdf
        assert np.all(S.inertia_at(urdf, spec) > 0.0) and S.inertia_at(urdf, spec).shape == (7,)
        g_bare = S.gravity_from_urdf(urdf, spec, contract.side(side).palm_body)(q)
        assert np.abs(g_pd[1]) > np.abs(g_bare[1])                  # 페이로드(손가락)가 어깨 토크를 키운다
