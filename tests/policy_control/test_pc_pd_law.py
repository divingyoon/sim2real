"""M5 — pd_law: 한 관절 그룹의 tick 법칙(순수 함수)과 engage/release 블렌드.

잠그는 규칙(플랜 §4.4):
  · rate-limit 은 **직전 세트포인트** 기준(실측 기준이면 처짐 뒤에 명령이 떨어져 팔이 안 움직인다)
  · lead 캡은 실측 대비 lead_vel·lead_sec
  · droop 적분은 정책 스텝당 1회, TRACKING 에서만, clamp
  · q̇* 전향은 TRACKING + fresh 에서만, ±vel_ff_cap
  · τ = clip(τ_req) + G(q_meas), 합이 cap 을 넘으면 fault + τ 0
  · ★(kd/kp)·q̇* 위치 보정 **부재** 회귀 — 속도 인터페이스와 이중 보상이 된다
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_control import contract as C
from policy_control import pd_law as L

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
CONFIG = SIM2REAL / "policy_control" / "config"
PROFILE = RL_WS / "robot_control/src/robot_control/profiles/openarm_tesollo.yaml"
LEFT_CONTRACT = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
N = 7

pytestmark = pytest.mark.unit


def _cfg(**over) -> L.PdLawCfg:
    base = dict(max_vel=1.0, lead_sec=0.1, lead_vel=2.0, vel_ff_scale=1.0, vel_ff_cap=0.5,
                effort_cap=20.0, lower=np.full(N, -3.0), upper=np.full(N, 3.0),
                gravity_mode="off")
    base.update(over)
    return L.PdLawCfg(**base)


def _inputs(**over) -> L.PdInputs:
    base = dict(q_target=np.zeros(N), qd_target=np.zeros(N), tau_request=np.zeros(N),
                q_meas=np.zeros(N), qd_meas=np.zeros(N), target_fresh=True,
                new_policy_step=True, advance=True, tracking=True, dt=0.01)
    base.update(over)
    return L.PdInputs(**base)


def _state(q_sp: float = 0.0) -> L.PdState:
    return L.initial_state(np.full(N, q_sp))


# ---------------------------------------------------------------- dataclasses / validation
def test_cfg_rejects_bad_values():
    with pytest.raises(ValueError):
        _cfg(max_vel=0.0)
    with pytest.raises(ValueError):
        _cfg(lower=np.full(N, 1.0), upper=np.full(N, -1.0))
    with pytest.raises(ValueError):
        _cfg(gravity_mode="magic")
    with pytest.raises(ValueError):                       # droop 모드인데 파라미터 없음
        _cfg(gravity_mode="integral_droop")
    with pytest.raises(ValueError):                       # off 모드인데 droop 파라미터
        _cfg(droop_gain=0.05, droop_limit=np.full(N, 0.1))


def test_step_rejects_shape_and_nan():
    with pytest.raises(ValueError):
        L.step(_state(), _inputs(q_target=np.zeros(N - 1)), _cfg())
    with pytest.raises(ValueError):
        L.step(_state(), _inputs(q_meas=np.full(N, np.nan)), _cfg())
    with pytest.raises(ValueError):
        L.step(_state(), _inputs(dt=0.0), _cfg())


def test_step_returns_new_objects_and_does_not_mutate_inputs():
    st = _state()
    q_target = np.full(N, 0.3)
    inp = _inputs(q_target=q_target)
    st2, cmd = L.step(st, inp, _cfg())
    assert st2 is not st and cmd.q is not st2.q_setpoint
    assert np.all(q_target == 0.3) and np.all(st.q_setpoint == 0.0)
    cmd.q[0] = 99.0
    assert st2.q_setpoint[0] != 99.0


# ---------------------------------------------------------------- rate limit / lead / limits
def test_rate_limit_is_from_previous_setpoint_not_measurement():
    # 세트포인트 0.5, 실측 0.0(처짐), 목표 0.6: 실측 기준이면 0.01, 세트포인트 기준이면 0.51
    st = _state(0.5)
    st2, cmd = L.step(st, _inputs(q_target=np.full(N, 0.6), q_meas=np.zeros(N)),
                      _cfg(max_vel=1.0, lead_vel=100.0))
    assert np.allclose(st2.q_setpoint, 0.51)
    assert np.allclose(cmd.q, 0.51)
    assert "velocity" in cmd.limited                             # 0.1 rad 요구 > 0.01 허용


def test_rate_limit_flags_velocity_when_target_far():
    st2, cmd = L.step(_state(), _inputs(q_target=np.full(N, 1.0)), _cfg(max_vel=1.0, lead_vel=100.0))
    assert np.allclose(st2.q_setpoint, 0.01)
    assert "velocity" in cmd.limited


def test_lead_cap_bounds_setpoint_against_measurement():
    # lead_vel 2.0 × lead_sec 0.1 = 0.2 rad. 세트포인트 0.5 vs 실측 0 → 0.2 로 깎고 flag
    st = _state(0.5)
    st2, cmd = L.step(st, _inputs(q_target=np.full(N, 0.5)), _cfg(lead_vel=2.0, lead_sec=0.1))
    assert np.allclose(st2.q_setpoint, 0.2)
    assert "lead" in cmd.limited


def test_target_outside_limits_is_clipped_and_flagged():
    cfg = _cfg(lower=np.full(N, -0.1), upper=np.full(N, 0.1), max_vel=100.0, lead_vel=100.0)
    st2, cmd = L.step(_state(), _inputs(q_target=np.full(N, 0.5)), cfg)
    assert np.allclose(st2.q_setpoint, 0.1)
    assert "position" in cmd.limited


def test_hold_freezes_setpoint_and_zeroes_velocity():
    st = _state(0.3)
    st2, cmd = L.step(st, _inputs(q_target=np.full(N, 1.0), qd_target=np.full(N, 0.4),
                                  advance=False, tracking=False), _cfg())
    assert np.allclose(st2.q_setpoint, 0.3)
    assert np.all(cmd.qd == 0.0)


# ---------------------------------------------------------------- droop
def _droop_cfg(**over) -> L.PdLawCfg:
    return _cfg(gravity_mode="integral_droop", droop_gain=0.05, droop_limit=np.full(N, 0.1),
                max_vel=100.0, lead_vel=100.0, **over)


def test_droop_integrates_once_per_policy_step_and_adds_to_command():
    cfg = _droop_cfg()
    tgt = np.full(N, 0.2)
    st1, cmd1 = L.step(_state(0.2), _inputs(q_target=tgt, q_meas=np.zeros(N), new_policy_step=True), cfg)
    assert np.allclose(st1.droop, 0.05 * 0.2)
    assert np.allclose(cmd1.q, 0.2 + 0.01)
    st2, _ = L.step(st1, _inputs(q_target=tgt, q_meas=np.zeros(N), new_policy_step=False), cfg)
    assert np.allclose(st2.droop, st1.droop)                    # 같은 정책 스텝 안에서는 불변
    st3, _ = L.step(st2, _inputs(q_target=tgt, q_meas=np.zeros(N), new_policy_step=True), cfg)
    assert np.allclose(st3.droop, 2 * 0.05 * 0.2)


def test_droop_is_clamped_and_frozen_outside_tracking():
    cfg = _droop_cfg()
    st = L.PdState(q_setpoint=np.zeros(N), droop=np.full(N, 0.099))
    st2, _ = L.step(st, _inputs(q_target=np.full(N, 1.0), q_meas=np.zeros(N)), cfg)
    assert np.allclose(st2.droop, 0.1)
    st3, _ = L.step(st2, _inputs(q_target=np.full(N, 1.0), tracking=False, advance=True), cfg)
    assert np.allclose(st3.droop, st2.droop)


def test_reset_droop_returns_zeroed_copy():
    st = L.PdState(q_setpoint=np.full(N, 0.3), droop=np.full(N, 0.05))
    st2 = L.reset_droop(st)
    assert np.all(st2.droop == 0.0) and np.allclose(st2.q_setpoint, 0.3)
    assert np.all(st.droop == 0.05)


def test_no_droop_when_mode_off():
    st2, cmd = L.step(_state(), _inputs(q_target=np.full(N, 0.5), q_meas=np.zeros(N)),
                      _cfg(max_vel=100.0, lead_vel=100.0))
    assert np.all(st2.droop == 0.0)
    assert np.allclose(cmd.q, 0.5)


# ---------------------------------------------------------------- velocity feed-forward
def test_vel_ff_scaled_and_capped_only_while_tracking_and_fresh():
    qd = np.full(N, 3.0)
    _, cmd = L.step(_state(), _inputs(qd_target=qd), _cfg(vel_ff_scale=0.5, vel_ff_cap=1.0))
    assert np.allclose(cmd.qd, 1.0)                              # 1.5 → cap 1.0
    _, cmd = L.step(_state(), _inputs(qd_target=qd), _cfg(vel_ff_scale=0.1, vel_ff_cap=1.0))
    assert np.allclose(cmd.qd, 0.3)
    _, cmd = L.step(_state(), _inputs(qd_target=qd, target_fresh=False), _cfg())
    assert np.all(cmd.qd == 0.0)
    _, cmd = L.step(_state(), _inputs(qd_target=qd, tracking=False), _cfg())
    assert np.all(cmd.qd == 0.0)


def test_vel_ff_cap_zero_means_no_feedforward():
    _, cmd = L.step(_state(), _inputs(qd_target=np.full(N, 1.0)), _cfg(vel_ff_cap=0.0))
    assert np.all(cmd.qd == 0.0)


def test_regression_no_kd_over_kp_position_lead():
    """right_inference_node 의 q_cmd += (kd/kp)·q̇* 는 폐기: q 명령이 q̇* 에 무관해야 한다."""
    cfg = _cfg(max_vel=100.0, lead_vel=100.0)
    tgt = np.full(N, 0.2)
    _, still = L.step(_state(), _inputs(q_target=tgt, qd_target=np.zeros(N)), cfg)
    _, moving = L.step(_state(), _inputs(q_target=tgt, qd_target=np.full(N, 2.0)), cfg)
    assert np.array_equal(still.q, moving.q)
    assert np.allclose(moving.q, tgt)                            # 위치 지령 = 목표 그대로
    assert not np.allclose(moving.qd, 0.0)                       # 속도는 속도 인터페이스로


# ---------------------------------------------------------------- effort
def test_tau_request_clipped_to_cap_without_fault():
    _, cmd = L.step(_state(), _inputs(tau_request=np.full(N, 25.0)), _cfg(effort_cap=20.0))
    assert np.allclose(cmd.tau, 20.0)
    assert not cmd.effort_fault


def test_gravity_term_added_on_measured_q_and_over_cap_faults():
    seen = []

    def gravity(q):
        seen.append(np.array(q))
        return np.full(N, 5.0)

    cfg = _cfg(gravity_mode="model_tau_ff", effort_cap=20.0)
    q_meas = np.full(N, 0.7)
    _, cmd = L.step(_state(), _inputs(q_meas=q_meas, tau_request=np.zeros(N)), cfg, gravity_fn=gravity)
    assert np.allclose(cmd.tau, 5.0) and not cmd.effort_fault
    assert np.allclose(seen[-1], q_meas)                         # 지령이 아니라 실측 자세
    _, cmd = L.step(_state(), _inputs(q_meas=q_meas, tau_request=np.full(N, 25.0)), cfg, gravity_fn=gravity)
    assert cmd.effort_fault and np.all(cmd.tau == 0.0)           # 20 + 5 > cap → 0 송출


def test_gravity_term_kept_in_hold():
    cfg = _cfg(gravity_mode="model_tau_ff")
    _, cmd = L.step(_state(), _inputs(advance=False, tracking=False), cfg, gravity_fn=lambda q: np.full(N, 2.0))
    assert np.allclose(cmd.tau, 2.0)


def test_model_mode_requires_gravity_fn():
    with pytest.raises(ValueError):
        L.step(_state(), _inputs(), _cfg(gravity_mode="model_tau_ff"))


# ---------------------------------------------------------------- blend (engage / release)
@pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 1.0])
def test_blend_engage_identity(s):
    rng = np.random.default_rng(0)
    ref, q, g = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N) * 5
    kp = np.array([70, 70, 70, 60, 10, 10, 10.0])
    q_star, tau_ff = L.blend_engage(ref, g, kp, s)
    assert np.allclose(kp * (q_star - q) + tau_ff, kp * (ref - q))
    assert np.allclose(tau_ff, s * g)


def test_blend_release_is_reverse_of_engage():
    rng = np.random.default_rng(1)
    ref, g = rng.normal(size=N), rng.normal(size=N)
    kp = np.full(N, 10.0)
    assert np.allclose(L.blend_release(ref, g, kp, 0.3)[1], L.blend_engage(ref, g, kp, 0.7)[1])
    q_end, tau_end = L.blend_release(ref, g, kp, 1.0)
    assert np.allclose(tau_end, 0.0) and np.allclose(q_end, ref)


def test_blend_fraction_and_bad_args():
    assert L.blend_fraction(0.0, 1.0) == 0.0
    assert L.blend_fraction(0.5, 1.0) == 0.5
    assert L.blend_fraction(3.0, 1.0) == 1.0
    with pytest.raises(ValueError):
        L.blend_fraction(0.1, 0.0)
    with pytest.raises(ValueError):
        L.blend_engage(np.zeros(N), np.zeros(N), np.zeros(N), 0.5)      # kp 0
    with pytest.raises(ValueError):
        L.blend_engage(np.zeros(N), np.zeros(N), np.ones(N), 1.5)       # s 범위 밖


# ---------------------------------------------------------------- config loader
def test_load_left_and_right_configs():
    left = L.load_pd_config(CONFIG / "pd_left.yaml")
    right = L.load_pd_config(CONFIG / "pd_right.yaml")
    assert left.execute is False and right.execute is False
    assert left.pd_hz == 100.0 and left.watchdog_sec == 0.25 and left.lead_sec == 0.1
    assert left.max_vel.reduced == 0.25 and left.max_vel.full == 2.0
    assert left.vel_ff_cap == 0.0 and left.effort_cap == 20.0
    assert left.gravity.mode == "integral_droop" and left.gripper is not None and left.hand is None
    assert right.gravity.mode == "model_tau_ff" and right.hand is not None and right.gripper is None
    assert right.gravity.urdf.exists() and right.gravity.tip_link == "r_hl_palm_ee"
    assert len(right.gravity.scale) == 7 and len(right.gravity.payload) == 4
    assert left.thermal[0].joint == "l_aj_7" and right.thermal[0].joint == "r_aj_7"
    assert right.hand.pid_p == 1.5 and left.gripper.close_overtravel_m == 0.008   # 벤더 PID(09.06)
    assert left.gains.yaml.exists()


def test_load_pd_config_rejects_missing_and_unknown_keys(tmp_path):
    p = tmp_path / "pd.yaml"
    p.write_text("execute: false\n")
    with pytest.raises(L.PdConfigError):
        L.load_pd_config(p)
    src = (CONFIG / "pd_left.yaml").read_text() + "\nbogus_knob: 1\n"
    p.write_text(src)
    with pytest.raises(L.PdConfigError):
        L.load_pd_config(p)
    with pytest.raises(L.PdConfigError):
        L.load_pd_config(tmp_path / "nope.yaml")


def test_load_pd_config_rejects_execute_true_by_type_error(tmp_path):
    src = (CONFIG / "pd_left.yaml").read_text().replace("execute: false", "execute: yes_please")
    p = tmp_path / "pd.yaml"
    p.write_text(src)
    with pytest.raises(L.PdConfigError):
        L.load_pd_config(p)


def test_limits_from_profile_reuses_jtc_bridge_core():
    lower, upper, vel = L.limits_from_profile(PROFILE, ["l_aj_1", "l_aj_4"])
    assert np.allclose(lower, [-3.49066, 0.0]) and np.allclose(upper, [1.39626, 2.44346])
    assert np.allclose(vel, [2.0, 2.0])
    with pytest.raises(KeyError):
        L.limits_from_profile(PROFILE, ["l_aj_99"])


@pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")
def test_law_cfg_from_config_takes_contract_values():
    cfg = L.load_pd_config(CONFIG / "pd_left.yaml")
    contract = C.load_contract(LEFT_CONTRACT)
    lower, upper, _ = L.limits_from_profile(PROFILE, contract.pd.sim_gains.joints)
    law = L.law_cfg_from_config(cfg, contract, stage="reduced", lower=lower, upper=upper)
    assert law.max_vel == 0.25 and law.vel_ff_scale == contract.fabric.vel_ff_scale
    assert law.gravity_mode == "integral_droop" and law.droop_gain == 0.05
    assert np.allclose(law.droop_limit, contract.pd.gravity.limit)
    ramp = L.law_cfg_from_config(cfg, contract, stage="ramp", lower=lower, upper=upper)
    assert ramp.max_vel == cfg.ramp_speed
    with pytest.raises(ValueError):
        L.law_cfg_from_config(cfg, contract, stage="warp", lower=lower, upper=upper)


# ---------------------------------------------------------------- error branches
def test_cfg_rejects_negative_caps_and_bad_droop_limit():
    with pytest.raises(ValueError):
        _cfg(vel_ff_cap=-1.0)
    with pytest.raises(ValueError):
        _cfg(gravity_mode="integral_droop", droop_gain=0.05, droop_limit=np.full(N - 1, 0.1))


def test_blend_release_rejects_bad_fraction():
    with pytest.raises(ValueError):
        L.blend_release(np.zeros(N), np.zeros(N), np.ones(N), -0.1)


def test_load_pd_config_rejects_bad_shapes(tmp_path):
    base = (CONFIG / "pd_left.yaml").read_text()
    cases = {
        "not_mapping": "- 1\n- 2\n",
        "bad_yaml": "side: [unclosed\n",
        "pd_hz_zero": base.replace("pd_hz: 100.0", "pd_hz: 0"),
        "pd_hz_text": base.replace("pd_hz: 100.0", "pd_hz: fast"),
        "max_vel_missing_key": base.replace("  full: 2.0\n", ""),
        "gravity_unknown_key": base.replace("  mode: integral_droop", "  mode: integral_droop\n  warp: 1"),
        "gravity_droop_with_urdf": base.replace("  mode: integral_droop", "  mode: integral_droop\n  urdf: x.urdf"),
        "accept_not_bool": base.replace("accept_sim_mismatch: false", "accept_sim_mismatch: nope"),
        "thermal_bad": base.replace("act_sec: 300}", "act_sec: 0}"),
    }
    for name, text in cases.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(text)
        with pytest.raises(L.PdConfigError):
            L.load_pd_config(p)


def test_limits_from_profile_requires_velocity(tmp_path):
    p = tmp_path / "prof.yaml"
    p.write_text("joints:\n  - {canonical: a, source: s, lower: -1, upper: 1}\n")
    with pytest.raises(KeyError):
        L.limits_from_profile(p, ["a"])


@pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")
def test_law_cfg_from_config_refuses_gravity_conflict():
    cfg = L.load_pd_config(CONFIG / "pd_right.yaml")
    contract = C.load_contract(LEFT_CONTRACT)
    lower, upper, _ = L.limits_from_profile(PROFILE, contract.pd.sim_gains.joints)
    with pytest.raises(L.GravityConfigError):
        L.law_cfg_from_config(cfg, contract, stage="full", lower=lower, upper=upper)


# ---------------------------------------------------------------- 09.06 양팔(asset 계약) — 팔별 law cfg / gravity 매핑 파싱
ASSET_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
needs_asset = pytest.mark.skipif(not ASSET_CONTRACT.exists(), reason="asset contract 없음")
needs_left = pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")


def test_dg5f_m_yaml_gravity_block_is_per_side():
    cfg = L.load_pd_config(CONFIG / "pd_dg5f_m.yaml")
    assert cfg.side == "both" and cfg.gravity.mode == "model_tau_ff" and cfg.gravity.sided
    assert cfg.gravity.tip_link == {"left": "l_hl_palm_ee", "right": "r_hl_palm_ee"}
    assert set(cfg.gravity.payload) == {"left", "right"} and len(cfg.gravity.payload["left"]) == 4
    assert cfg.gravity.scale == (1.0,) * 7 and cfg.gravity.urdf.name == "openarm_dg5f-m_bi_rl.urdf"
    assert {r.joint for r in cfg.thermal} == {"l_aj_7", "r_aj_7"} and cfg.gripper is None and cfg.hand is not None


@needs_asset
def test_law_cfg_from_config_per_side_on_asset_contract():
    contract = C.load_contract(ASSET_CONTRACT)
    cfg = L.load_pd_config(CONFIG / "pd_dg5f_m.yaml")
    for side in ("left", "right"):
        lower, upper, _ = L.limits_from_profile(PROFILE, contract.side(side).arm_joints)
        law = L.law_cfg_from_config(cfg, contract, "full", lower, upper, side=side)
        assert law.gravity_mode == "model_tau_ff" and law.droop_gain is None and law.vel_ff_scale == 1.0
        assert law.max_vel == cfg.max_vel.full and np.allclose(law.lower, lower)
    with pytest.raises(C.ContractError):
        L.law_cfg_from_config(cfg, contract, "full", lower, upper, side="up")


@needs_left
def test_law_cfg_side_equals_legacy_for_single_arm_contract():
    contract = C.load_contract(LEFT_CONTRACT)
    cfg = L.load_pd_config(CONFIG / "pd_left.yaml")
    lower, upper, _ = L.limits_from_profile(PROFILE, contract.pd.sim_gains.joints)
    a = L.law_cfg_from_config(cfg, contract, "reduced", lower, upper)
    b = L.law_cfg_from_config(cfg, contract, "reduced", lower, upper, side="left")
    assert a.droop_gain == b.droop_gain and np.allclose(a.droop_limit, b.droop_limit) and a.vel_ff_scale == b.vel_ff_scale
    assert L.side_vel_ff_scale(contract, None) == L.side_vel_ff_scale(contract, "left") == 1.0


def test_gravity_block_rejects_unknown_side_keys(tmp_path):
    raw = yaml.safe_load((CONFIG / "pd_dg5f_m.yaml").read_text())
    raw["gravity"]["tip_link"] = {"left": "l_hl_palm_ee", "up": "x"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(L.PdConfigError):
        L.load_pd_config(p)
