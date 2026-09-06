"""M5 — pd_state: 5상태 FSM(IDLE→RAMPING→TRACKING→HOLD→RELEASING→IDLE), engage 거부 사유,
fault 감지, thermal 적분. 전이 분기 100 % 를 목표로 모든 (상태, 이벤트) 쌍을 돈다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from policy_control import pd_state as S

pytestmark = pytest.mark.unit

P = S.Phase
ALL_EVENTS = ("engage", "ramp_done", "fault", "release", "zero_tick")


def _at(phase: P, **over) -> S.FsmState:
    base = dict(phase=phase, hold_reason=None, zero_ticks=0, release_zero_ticks=3)
    base.update(over)
    return S.FsmState(**base)


# ---------------------------------------------------------------- happy path
def test_full_cycle():
    st = S.initial_fsm(release_zero_ticks=3)
    assert st.phase is P.IDLE
    st = S.transition(st, "engage")
    assert st.phase is P.RAMPING
    st = S.transition(st, "ramp_done")
    assert st.phase is P.TRACKING
    st = S.transition(st, "fault", reason="watchdog")
    assert st.phase is P.HOLD and st.hold_reason == "watchdog"
    st = S.transition(st, "release")
    assert st.phase is P.RELEASING and st.zero_ticks == 0
    st = S.transition(st, "zero_tick")
    st = S.transition(st, "zero_tick")
    assert st.phase is P.RELEASING and st.zero_ticks == 2
    st = S.transition(st, "zero_tick")
    assert st.phase is P.IDLE and st.zero_ticks == 0 and st.hold_reason is None


def test_transition_returns_new_frozen_state():
    st = S.initial_fsm(5)
    st2 = S.transition(st, "engage")
    assert st.phase is P.IDLE and st2 is not st
    with pytest.raises(Exception):
        st2.phase = P.IDLE                                     # frozen


def test_initial_fsm_validates_zero_ticks():
    with pytest.raises(ValueError):
        S.initial_fsm(0)


# ---------------------------------------------------------------- exhaustive transition table
LEGAL = {
    (P.IDLE, "engage"): P.RAMPING,
    (P.RAMPING, "ramp_done"): P.TRACKING,
    (P.RAMPING, "fault"): P.HOLD,
    (P.RAMPING, "release"): P.RELEASING,
    (P.TRACKING, "fault"): P.HOLD,
    (P.TRACKING, "release"): P.RELEASING,
    (P.HOLD, "fault"): P.HOLD,
    (P.HOLD, "release"): P.RELEASING,
    (P.RELEASING, "zero_tick"): P.RELEASING,
}


@pytest.mark.parametrize("phase,event", list(itertools.product(list(P), ALL_EVENTS)))
def test_every_phase_event_pair(phase, event):
    st = _at(phase, hold_reason="x" if phase is P.HOLD else None)
    if (phase, event) in LEGAL:
        out = S.transition(st, event, reason="r" if event == "fault" else None)
        assert out.phase is LEGAL[(phase, event)]
    else:
        with pytest.raises(S.TransitionError):
            S.transition(st, event, reason="r" if event == "fault" else None)


def test_unknown_event_and_missing_reason():
    with pytest.raises(S.TransitionError):
        S.transition(_at(P.TRACKING), "dance")
    with pytest.raises(S.TransitionError):
        S.transition(_at(P.TRACKING), "fault")                  # fault 는 사유 필수


def test_hold_keeps_first_reason_and_appends_new():
    st = S.transition(_at(P.TRACKING), "fault", reason="watchdog")
    st = S.transition(st, "fault", reason="estop")
    assert st.hold_reason == "watchdog; estop"
    st = S.transition(st, "fault", reason="estop")               # 중복은 한 번만
    assert st.hold_reason == "watchdog; estop"


def test_release_clears_hold_reason_only_at_idle():
    st = S.transition(_at(P.HOLD, hold_reason="estop"), "release")
    assert st.hold_reason == "estop"                             # status 에 남긴다
    st = S.transition(_at(P.RELEASING, zero_ticks=2, release_zero_ticks=3, hold_reason="estop"), "zero_tick")
    assert st.phase is P.IDLE and st.hold_reason is None


# ---------------------------------------------------------------- law flags
@pytest.mark.parametrize("phase,advance,tracking", [
    (P.IDLE, False, False), (P.RAMPING, True, False), (P.TRACKING, True, True),
    (P.HOLD, False, False), (P.RELEASING, False, False),
])
def test_law_flags(phase, advance, tracking):
    assert S.law_flags(phase) == (advance, tracking)


# ---------------------------------------------------------------- engage refusals
def _check(**over) -> S.EngageCheck:
    base = dict(execute=True, state_age_sec=0.01, stale_sec=0.5, gains_ok=True,
                accept_sim_mismatch=False, gravity_conflict=None,
                effort_controller_active=False, estop_latched=False, phase=P.IDLE)
    base.update(over)
    return S.EngageCheck(**base)


def test_engage_ok_when_everything_fine():
    assert S.engage_refusals(_check()) == []


def test_engage_collects_every_refusal_reason():
    reasons = S.engage_refusals(_check(execute=False, state_age_sec=1.0, gains_ok=False,
                                       gravity_conflict="config droop vs contract model",
                                       effort_controller_active=True, estop_latched=True,
                                       phase=P.TRACKING))
    text = "\n".join(reasons)
    assert len(reasons) == 7
    for key in ("execute", "stale", "gain", "gravity", "effort controller", "estop", "IDLE"):
        assert key in text, key


def test_engage_gain_mismatch_passes_only_with_accept_flag():
    assert S.engage_refusals(_check(gains_ok=False, accept_sim_mismatch=True)) == []
    assert any("gain" in r for r in S.engage_refusals(_check(gains_ok=False)))


def test_engage_state_never_seen_is_stale():
    assert any("stale" in r for r in S.engage_refusals(_check(state_age_sec=None)))


# ---------------------------------------------------------------- fault detection
def _faults(**over) -> S.FaultInputs:
    base = dict(target_age_sec=0.0, watchdog_sec=0.25, tracking_err=0.0, abort_tracking=0.3,
                target_clipped=False, effort_fault=False, estop_latched=False,
                thermal_act=(), switch_failed=False)
    base.update(over)
    return S.FaultInputs(**base)


def test_no_faults_nominal():
    assert S.detect_faults(_faults()) == []
    assert S.detect_faults(_faults(target_age_sec=0.25)) == []   # 경계는 통과
    assert S.detect_faults(_faults(tracking_err=0.3)) == []


def test_every_fault_reason_is_named():
    reasons = S.detect_faults(_faults(target_age_sec=0.3, tracking_err=0.31, target_clipped=True,
                                      effort_fault=True, estop_latched=True,
                                      thermal_act=("r_aj_7",), switch_failed=True))
    assert len(reasons) == 7
    text = "\n".join(reasons)
    for key in ("watchdog", "tracking", "joint limit", "effort", "estop", "thermal r_aj_7", "switch"):
        assert key in text, key


def test_watchdog_needs_a_target_first():
    # 아직 목표를 한 번도 못 받았으면(None) watchdog 이 아니라 '목표 없음' 사유
    reasons = S.detect_faults(_faults(target_age_sec=None))
    assert reasons == [] or any("target" in r for r in reasons)


# ---------------------------------------------------------------- thermal
RULES = (S.ThermalRule(joint="r_aj_7", effort_nm=1.5, act_sec=3.0, warn_sec=1.0),
         S.ThermalRule(joint="r_aj_1", effort_nm=5.0, act_sec=10.0))


def test_thermal_rule_validation():
    with pytest.raises(ValueError):
        S.ThermalRule(joint="j", effort_nm=0.0, act_sec=1.0)
    with pytest.raises(ValueError):
        S.ThermalRule(joint="j", effort_nm=1.0, act_sec=0.0)
    with pytest.raises(ValueError):
        S.ThermalRule(joint="j", effort_nm=1.0, act_sec=1.0, warn_sec=2.0)


def test_thermal_integrates_above_threshold_and_decays_below():
    st = S.thermal_init(RULES)
    assert st.hot_sec == (0.0, 0.0)
    for _ in range(5):
        st = S.thermal_step(st, RULES, {"r_aj_7": -2.0, "r_aj_1": 0.0}, dt=0.1)   # |τ| 기준
    assert np.isclose(st.hot_sec[0], 0.5) and st.hot_sec[1] == 0.0
    st = S.thermal_step(st, RULES, {"r_aj_7": 1.0, "r_aj_1": 0.0}, dt=0.2)
    assert np.isclose(st.hot_sec[0], 0.3)
    for _ in range(10):
        st = S.thermal_step(st, RULES, {"r_aj_7": 0.0, "r_aj_1": 0.0}, dt=0.2)
    assert st.hot_sec[0] == 0.0                                   # 0 아래로는 안 내려간다


def test_thermal_levels_warn_then_act():
    st = S.thermal_init(RULES)
    assert S.thermal_levels(st, RULES) == {"r_aj_7": "ok", "r_aj_1": "ok"}
    st = S.ThermalState(hot_sec=(1.5, 0.0))
    assert S.thermal_levels(st, RULES)["r_aj_7"] == "warn"
    st = S.ThermalState(hot_sec=(3.0, 10.0))
    assert S.thermal_levels(st, RULES) == {"r_aj_7": "act", "r_aj_1": "act"}
    assert S.thermal_act_joints(st, RULES) == ("r_aj_7", "r_aj_1")


def test_thermal_step_rejects_missing_joint_and_bad_dt():
    st = S.thermal_init(RULES)
    with pytest.raises(KeyError):
        S.thermal_step(st, RULES, {"r_aj_7": 0.0}, dt=0.1)
    with pytest.raises(ValueError):
        S.thermal_step(st, RULES, {"r_aj_7": 0.0, "r_aj_1": 0.0}, dt=0.0)
    with pytest.raises(ValueError):
        S.thermal_step(S.ThermalState(hot_sec=(0.0,)), RULES, {"r_aj_7": 0.0, "r_aj_1": 0.0}, dt=0.1)


def test_thermal_rules_from_config_dicts():
    rules = S.thermal_rules_from_config([{"joint": "l_aj_7", "effort_nm": 5.0, "act_sec": 300}])
    assert rules == (S.ThermalRule(joint="l_aj_7", effort_nm=5.0, act_sec=300.0, warn_sec=None),)
    with pytest.raises(ValueError):
        S.thermal_rules_from_config([{"joint": "l_aj_7", "effort_nm": 5.0}])
    with pytest.raises(ValueError):
        S.thermal_rules_from_config([{"joint": "l_aj_7", "effort_nm": 5.0, "act_sec": 1, "x": 1}])
