"""pd 노드 상태기계(순수, ROS 무의존) — 플랜 §4.4.

    IDLE ──engage──▶ RAMPING ──ramp_done──▶ TRACKING
                        │ fault               │ fault
                        ▼                     ▼
                      HOLD(reason) ◀──fault── HOLD
      RAMPING/TRACKING/HOLD ──release──▶ RELEASING ──zero_tick×N──▶ IDLE

HOLD 는 세트포인트 동결·q̇*=0·τ_ff 유지(급감 금지)를 뜻한다 — 그 해석은 `law_flags` 가
pd_law 의 (advance, tracking) 두 플래그로 넘긴다. engage 거부 사유와 fault 사유는 각각
한 리스트로 모아 status 에 그대로 실린다(어느 캡이 걸렸는지 사후에 알 수 있게).

모든 상태는 frozen dataclass 이고 전이는 새 객체를 돌려준다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence


class TransitionError(RuntimeError):
    """The requested event is not legal in the current phase."""


class Phase(Enum):
    IDLE = "IDLE"
    RAMPING = "RAMPING"
    TRACKING = "TRACKING"
    HOLD = "HOLD"
    RELEASING = "RELEASING"


EVENTS = ("engage", "ramp_done", "fault", "release", "zero_tick")
_MOVING = (Phase.RAMPING, Phase.TRACKING)
_ENGAGED = (Phase.RAMPING, Phase.TRACKING, Phase.HOLD)


@dataclass(frozen=True)
class FsmState:
    phase: Phase
    hold_reason: str | None
    zero_ticks: int                 # RELEASING 에서 q̇*=0·τ=0 을 몇 번 송출했나
    release_zero_ticks: int         # IDLE 로 가기 위해 필요한 횟수(설정)


def initial_fsm(release_zero_ticks: int) -> FsmState:
    if int(release_zero_ticks) < 1:
        raise ValueError(f"release_zero_ticks must be >= 1, got {release_zero_ticks}")
    return FsmState(phase=Phase.IDLE, hold_reason=None, zero_ticks=0,
                    release_zero_ticks=int(release_zero_ticks))


def transition(state: FsmState, event: str, reason: str | None = None) -> FsmState:
    """Apply *event* and return the new state; illegal pairs raise TransitionError."""
    if event not in EVENTS:
        raise TransitionError(f"unknown event {event!r}; expected one of {EVENTS}")
    phase = state.phase
    if event == "engage" and phase is Phase.IDLE:
        return replace(state, phase=Phase.RAMPING, hold_reason=None, zero_ticks=0)
    if event == "ramp_done" and phase is Phase.RAMPING:
        return replace(state, phase=Phase.TRACKING)
    if event == "fault" and phase in _ENGAGED:
        return _hold(state, reason)
    if event == "release" and phase in _ENGAGED:
        return replace(state, phase=Phase.RELEASING, zero_ticks=0)
    if event == "zero_tick" and phase is Phase.RELEASING:
        return _zero_tick(state)
    raise TransitionError(f"event {event!r} is not legal in phase {phase.value}")


def _hold(state: FsmState, reason: str | None) -> FsmState:
    if not reason:
        raise TransitionError("a fault needs a reason")
    # 사유는 **종류**(콜론 앞)별로 하나만 남긴다 — 'watchdog: target stale 0.252 s' 처럼 숫자만
    # 다른 사유가 100 Hz 로 쌓여 문자열이 무한히 자라는 것을 막는다(첫 메시지를 보존).
    reasons = [] if state.hold_reason is None else state.hold_reason.split("; ")
    kinds = {r.split(":", 1)[0] for r in reasons}
    if reason.split(":", 1)[0] not in kinds:
        reasons = [*reasons, reason]
    return replace(state, phase=Phase.HOLD, hold_reason="; ".join(reasons))


def _zero_tick(state: FsmState) -> FsmState:
    ticks = state.zero_ticks + 1
    if ticks >= state.release_zero_ticks:
        return replace(state, phase=Phase.IDLE, hold_reason=None, zero_ticks=0)
    return replace(state, zero_ticks=ticks)


def law_flags(phase: Phase) -> tuple[bool, bool]:
    """(advance, tracking) for pd_law: setpoint may move / policy feed-forward allowed."""
    return (phase in _MOVING, phase is Phase.TRACKING)


# ------------------------------------------------------------------ engage refusals
@dataclass(frozen=True)
class EngageCheck:
    execute: bool
    state_age_sec: float | None     # None = 실측을 한 번도 못 받음
    stale_sec: float
    gains_ok: bool
    accept_sim_mismatch: bool
    gravity_conflict: str | None
    effort_controller_active: bool
    estop_latched: bool
    phase: Phase


def engage_refusals(check: EngageCheck) -> list[str]:
    """Every reason engage must be refused, in one list (empty = allowed)."""
    reasons: list[str] = []
    if not check.execute:
        reasons.append("execute is false (dry run) — set execute:=true to touch controllers")
    if check.state_age_sec is None or check.state_age_sec > check.stale_sec:
        reasons.append(f"joint state stale (age {check.state_age_sec} s > {check.stale_sec} s)")
    if not check.gains_ok and not check.accept_sim_mismatch:
        reasons.append("driver kp != trained kp (gain mismatch); accept_sim_mismatch not set")
    if check.gravity_conflict:
        reasons.append(f"gravity mode conflict: {check.gravity_conflict}")
    if check.effort_controller_active:
        reasons.append("forward effort controller already active (gravity_comp_node?)")
    if check.estop_latched:
        reasons.append("estop latched")
    if check.phase is not Phase.IDLE:
        reasons.append(f"phase {check.phase.value} is not IDLE")
    return reasons


# ------------------------------------------------------------------ fault detection
@dataclass(frozen=True)
class FaultInputs:
    target_age_sec: float | None    # None = 아직 첫 목표 없음(watchdog 대상 아님)
    watchdog_sec: float
    tracking_err: float             # max |q_setpoint − q_meas| [rad]
    abort_tracking: float
    target_clipped: bool            # 목표가 관절 한계 밖이었다(pd_law 'position')
    effort_fault: bool              # τ 합이 cap 초과(pd_law)
    estop_latched: bool
    thermal_act: Sequence[str]      # thermal_act_joints(...) 결과
    switch_failed: bool


def detect_faults(inp: FaultInputs) -> list[str]:
    """Reasons that must send the FSM to HOLD this tick (empty = none)."""
    reasons: list[str] = []
    if inp.target_age_sec is not None and inp.target_age_sec > inp.watchdog_sec:
        reasons.append(f"watchdog: target stale {inp.target_age_sec:.3f} s > {inp.watchdog_sec} s")
    if inp.tracking_err > inp.abort_tracking:
        reasons.append(f"tracking error {inp.tracking_err:.3f} rad > {inp.abort_tracking}")
    if inp.target_clipped:
        reasons.append("joint limit: target outside profile bounds")
    if inp.effort_fault:
        reasons.append("effort cap exceeded (tau zeroed)")
    if inp.estop_latched:
        reasons.append("estop latched")
    for joint in inp.thermal_act:
        reasons.append(f"thermal {joint}: effort above threshold too long")
    if inp.switch_failed:
        reasons.append("controller switch failed")
    return reasons


# ------------------------------------------------------------------ thermal
@dataclass(frozen=True)
class ThermalRule:
    joint: str
    effort_nm: float                # |τ| 가 이 값을 넘는 시간을 적분한다
    act_sec: float                  # 누적이 이 값에 닿으면 HOLD
    warn_sec: float | None = None   # 경고만(없으면 경고 단계 없음)

    def __post_init__(self) -> None:
        if self.effort_nm <= 0.0 or self.act_sec <= 0.0:
            raise ValueError(f"thermal rule {self.joint}: effort_nm and act_sec must be > 0")
        if self.warn_sec is not None and not 0.0 < self.warn_sec < self.act_sec:
            raise ValueError(f"thermal rule {self.joint}: warn_sec must be in (0, act_sec)")


@dataclass(frozen=True)
class ThermalState:
    hot_sec: tuple                  # rules 와 같은 순서


_RULE_KEYS = {"joint", "effort_nm", "act_sec", "warn_sec"}


def thermal_rules_from_config(items: Sequence[Mapping]) -> tuple[ThermalRule, ...]:
    rules = []
    for item in items:
        unknown = set(item) - _RULE_KEYS
        missing = {"joint", "effort_nm", "act_sec"} - set(item)
        if unknown or missing:
            raise ValueError(f"thermal rule {dict(item)}: unknown {sorted(unknown)}, missing {sorted(missing)}")
        warn = item.get("warn_sec")
        rules.append(ThermalRule(joint=str(item["joint"]), effort_nm=float(item["effort_nm"]),
                                 act_sec=float(item["act_sec"]),
                                 warn_sec=None if warn is None else float(warn)))
    return tuple(rules)


def thermal_init(rules: Sequence[ThermalRule]) -> ThermalState:
    return ThermalState(hot_sec=tuple(0.0 for _ in rules))


def thermal_step(state: ThermalState, rules: Sequence[ThermalRule],
                 efforts: Mapping[str, float], dt: float) -> ThermalState:
    """Integrate time spent above each rule's |effort|; decay at the same rate below it."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if len(state.hot_sec) != len(rules):
        raise ValueError(f"thermal state has {len(state.hot_sec)} entries for {len(rules)} rules")
    hot = []
    for rule, sec in zip(rules, state.hot_sec):
        if rule.joint not in efforts:
            raise KeyError(f"thermal rule joint {rule.joint!r} missing from efforts")
        above = abs(float(efforts[rule.joint])) > rule.effort_nm
        hot.append(sec + dt if above else max(0.0, sec - dt))
    return ThermalState(hot_sec=tuple(hot))


def thermal_levels(state: ThermalState, rules: Sequence[ThermalRule]) -> dict[str, str]:
    """joint → 'ok' | 'warn' | 'act'."""
    levels = {}
    for rule, sec in zip(rules, state.hot_sec):
        if sec >= rule.act_sec:
            levels[rule.joint] = "act"
        elif rule.warn_sec is not None and sec >= rule.warn_sec:
            levels[rule.joint] = "warn"
        else:
            levels[rule.joint] = "ok"
    return levels


def thermal_act_joints(state: ThermalState, rules: Sequence[ThermalRule]) -> tuple[str, ...]:
    return tuple(j for j, lvl in thermal_levels(state, rules).items() if lvl == "act")
