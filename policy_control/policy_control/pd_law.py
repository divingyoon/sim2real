"""pd 노드 tick 법칙(한 관절 그룹, 순수 함수) + engage/release 블렌드 + pd_*.yaml 로더.

tick(100 Hz 재발행) — 플랜 §4.4:
    q_t   = clip(q_target, lower, upper)
    q_sp  = velocity_limited_target(q_t, q_sp_prev, max_vel, dt)     ★직전 세트포인트 기준
    q_sp  = q_meas + clip(q_sp − q_meas, ±lead_vel·lead_sec)         (CommandGate.max_lead)
    droop = clip(droop + gain·(q_t − q_meas), ±limit)   정책 스텝당 1회, TRACKING 에서만
    q_cmd = clip(q_sp + droop, lower, upper)
    qd_cmd= clip(vel_ff_scale·q̇*, ±vel_ff_cap)         TRACKING + fresh 에서만, 아니면 0
    τ     = clip(τ_req, ±effort_cap) + G(q_meas)        합이 cap 을 넘으면 fault + 0

`(kd/kp)·q̇*` 위치 보정(right_inference_node)은 **없다** — 속도 인터페이스에 q̇* 를 직접
넣으므로 더하면 이중 보상이 된다(회귀 테스트가 잠근다). HOLD(advance=False)는 세트포인트
동결·q̇*=0·τ 유지.

블렌드: τ_ff(s)=s·G, q*(s)=ref−τ_ff/kp 이면 kp(q*−q)+τ_ff ≡ kp(ref−q) — engage/release 에서
토크 점프가 0 이다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from . import _paths
from .contract import DeployContract
from .pd_gains import GainsBlock, GainsError
from .pd_gravity import MODES, GravityBlock, GravityConfigError, contract_gravity, droop_params, gravity_conflict
from .pd_state import thermal_rules_from_config

from jtc_bridge_core import load_profile_joints, velocity_limited_target  # noqa: E402  (scripts/)

STAGES = ("ramp", "reduced", "full")


class PdConfigError(ValueError):
    """pd_*.yaml is missing, malformed, or carries a knob nobody reads."""


# ------------------------------------------------------------------ law dataclasses
@dataclass(frozen=True)
class PdLawCfg:
    max_vel: float                  # setpoint advance [rad/s], per stage
    lead_sec: float                 # max_lead = lead_vel·lead_sec (vs measured)
    lead_vel: float                 # profile joint velocity limit [rad/s]
    vel_ff_scale: float             # contract.fabric.vel_ff_scale
    vel_ff_cap: float               # 0 disables the velocity feed-forward
    effort_cap: float               # [N·m] on the summed τ
    lower: np.ndarray
    upper: np.ndarray
    gravity_mode: str
    droop_gain: float | None = None
    droop_limit: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("max_vel", "lead_sec", "lead_vel", "effort_cap"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.vel_ff_cap < 0.0 or self.vel_ff_scale < 0.0:
            raise ValueError("vel_ff_cap and vel_ff_scale must be >= 0")
        lo, hi = np.asarray(self.lower, dtype=np.float64), np.asarray(self.upper, dtype=np.float64)
        if lo.shape != hi.shape or lo.ndim != 1 or np.any(lo >= hi):
            raise ValueError("lower/upper must be 1-D, same length, lower < upper")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        if self.gravity_mode not in MODES:
            raise ValueError(f"gravity_mode {self.gravity_mode!r} not in {MODES}")
        has_droop = self.droop_gain is not None or self.droop_limit is not None
        if (self.gravity_mode == "integral_droop") != has_droop:
            raise ValueError("droop_gain/droop_limit are required iff gravity_mode == integral_droop")
        if has_droop:
            lim = np.asarray(self.droop_limit, dtype=np.float64)
            if self.droop_gain <= 0.0 or lim.shape != lo.shape or np.any(lim < 0.0):
                raise ValueError("droop_gain must be > 0 and droop_limit >= 0 with the joint count")
            object.__setattr__(self, "droop_limit", lim)


@dataclass(frozen=True)
class PdInputs:
    q_target: np.ndarray
    qd_target: np.ndarray
    tau_request: np.ndarray
    q_meas: np.ndarray
    qd_meas: np.ndarray
    target_fresh: bool              # joint_target younger than the watchdog
    new_policy_step: bool           # first tick after a new joint_target (droop integrates once)
    advance: bool                   # setpoint may move (RAMPING/TRACKING); False = HOLD
    tracking: bool                  # TRACKING: feed-forward + droop allowed
    dt: float


@dataclass(frozen=True)
class PdState:
    q_setpoint: np.ndarray
    droop: np.ndarray


@dataclass(frozen=True)
class PdCommand:
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    limited: tuple                  # subset of ('position', 'velocity', 'lead')
    effort_fault: bool


def initial_state(q_seed: np.ndarray) -> PdState:
    q = np.asarray(q_seed, dtype=np.float64).reshape(-1).copy()
    return PdState(q_setpoint=q, droop=np.zeros_like(q))


def reset_droop(state: PdState) -> PdState:
    return replace(state, droop=np.zeros_like(state.droop))


# ------------------------------------------------------------------ tick law
def step(state: PdState, inp: PdInputs, cfg: PdLawCfg,
         gravity_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[PdState, PdCommand]:
    """One pd tick for one joint group. Returns (new state, command); nothing is mutated."""
    n = cfg.lower.shape[0]
    q_target, qd_target, tau_req, q_meas = _checked(inp, n)
    if cfg.gravity_mode == "model_tau_ff" and gravity_fn is None:
        raise ValueError("gravity_mode model_tau_ff needs gravity_fn")

    limited: list[str] = []
    q_t = np.clip(q_target, cfg.lower, cfg.upper)
    if np.any(q_t != q_target):
        limited.append("position")

    q_sp, lead_flags = _advance_setpoint(state.q_setpoint, q_t, q_meas, inp, cfg)
    limited.extend(lead_flags)
    droop = _integrate_droop(state.droop, q_t, q_meas, inp, cfg)

    q_cmd = np.clip(q_sp + droop, cfg.lower, cfg.upper)
    feed_forward = inp.tracking and inp.target_fresh
    qd_cmd = np.clip(cfg.vel_ff_scale * qd_target, -cfg.vel_ff_cap, cfg.vel_ff_cap) if feed_forward \
        else np.zeros(n)
    tau, fault = _effort(tau_req, q_meas, cfg, gravity_fn)
    return (PdState(q_setpoint=q_sp, droop=droop),
            PdCommand(q=q_cmd, qd=qd_cmd, tau=tau, limited=tuple(limited), effort_fault=fault))


def _checked(inp: PdInputs, n: int) -> tuple[np.ndarray, ...]:
    if inp.dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {inp.dt}")
    out = []
    for name in ("q_target", "qd_target", "tau_request", "q_meas"):
        v = np.asarray(getattr(inp, name), dtype=np.float64).reshape(-1)
        if v.shape[0] != n or not np.isfinite(v).all():
            raise ValueError(f"{name} must be {n} finite values, got shape {v.shape}")
        out.append(v)
    return tuple(out)


def _advance_setpoint(q_sp_prev: np.ndarray, q_t: np.ndarray, q_meas: np.ndarray,
                      inp: PdInputs, cfg: PdLawCfg) -> tuple[np.ndarray, list[str]]:
    if not inp.advance:
        return q_sp_prev.copy(), []
    flags: list[str] = []
    q_sp = velocity_limited_target(q_t, q_sp_prev, cfg.max_vel, inp.dt)
    if np.any(np.abs(q_t - q_sp_prev) > cfg.max_vel * inp.dt + 1e-12):
        flags.append("velocity")
    max_lead = cfg.lead_vel * cfg.lead_sec
    lead = q_sp - q_meas
    bounded = np.clip(lead, -max_lead, max_lead)
    if np.any(bounded != lead):
        flags.append("lead")
    return q_meas + bounded, flags


def _integrate_droop(droop: np.ndarray, q_t: np.ndarray, q_meas: np.ndarray,
                     inp: PdInputs, cfg: PdLawCfg) -> np.ndarray:
    if cfg.gravity_mode != "integral_droop" or not (inp.tracking and inp.new_policy_step):
        return droop.copy()
    return np.clip(droop + cfg.droop_gain * (q_t - q_meas), -cfg.droop_limit, cfg.droop_limit)


def _effort(tau_req: np.ndarray, q_meas: np.ndarray, cfg: PdLawCfg,
            gravity_fn: Callable[[np.ndarray], np.ndarray] | None) -> tuple[np.ndarray, bool]:
    tau = np.clip(tau_req, -cfg.effort_cap, cfg.effort_cap)
    if cfg.gravity_mode == "model_tau_ff":
        tau = tau + np.asarray(gravity_fn(q_meas), dtype=np.float64).reshape(-1)
    if np.any(np.abs(tau) > cfg.effort_cap):
        return np.zeros_like(tau), True
    return tau, False


# ------------------------------------------------------------------ engage / release blend
def blend_fraction(elapsed_sec: float, blend_sec: float) -> float:
    if blend_sec <= 0.0:
        raise ValueError(f"blend_sec must be > 0, got {blend_sec}")
    return float(np.clip(elapsed_sec / blend_sec, 0.0, 1.0))


def blend_engage(ref: np.ndarray, gravity: np.ndarray, kp: np.ndarray,
                 s: float) -> tuple[np.ndarray, np.ndarray]:
    """(q*, τ_ff) with kp·(q*−q)+τ_ff ≡ kp·(ref−q): τ_ff = s·G, q* = ref − τ_ff/kp."""
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"blend fraction s must be in [0, 1], got {s}")
    kp_arr = np.asarray(kp, dtype=np.float64)
    if np.any(kp_arr <= 0.0):
        raise ValueError("kp must be > 0 on every joint")
    tau_ff = s * np.asarray(gravity, dtype=np.float64)
    q_star = np.asarray(ref, dtype=np.float64) - tau_ff / kp_arr
    return q_star, tau_ff


def blend_release(ref: np.ndarray, gravity: np.ndarray, kp: np.ndarray,
                  s: float) -> tuple[np.ndarray, np.ndarray]:
    """Reverse blend: s=0 keeps full τ_ff, s=1 reaches τ_ff=0 and q*=ref."""
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"blend fraction s must be in [0, 1], got {s}")
    return blend_engage(ref, gravity, kp, 1.0 - s)


# ------------------------------------------------------------------ pd_*.yaml
@dataclass(frozen=True)
class MaxVel:
    reduced: float
    full: float


@dataclass(frozen=True)
class Settle:
    clamp: float
    tol: float


@dataclass(frozen=True)
class GripperBlock:
    close_overtravel_m: float
    max_vel: float


@dataclass(frozen=True)
class HandBlock:
    pid_p: float
    pid_d: float
    max_vel: float


@dataclass(frozen=True)
class PdConfig:
    side: str
    execute: bool
    pd_hz: float
    ramp_speed: float
    watchdog_sec: float
    lead_sec: float
    lead_vel: float
    max_vel: MaxVel
    vel_ff_cap: float
    effort_cap: float
    abort_tracking: float
    release_zero_ticks: int
    blend_sec: float
    settle: Settle
    gravity: GravityBlock
    gains: GainsBlock
    thermal: tuple
    gripper: GripperBlock | None
    hand: HandBlock | None


_SCALARS = {"pd_hz": float, "ramp_speed": float, "watchdog_sec": float, "lead_sec": float,
            "lead_vel": float, "vel_ff_cap": float, "effort_cap": float, "abort_tracking": float,
            "release_zero_ticks": int, "blend_sec": float}
_KEYS = {"side", "execute", "max_vel", "settle", "gravity", "gains", "thermal", "gripper", "hand",
         *_SCALARS}


def load_pd_config(path: Path) -> PdConfig:
    """Strict loader: every key known, every knob typed, paths resolved against rl_ws."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise PdConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PdConfigError(f"{path}: top level must be a mapping")
    unknown, missing = set(raw) - _KEYS, _KEYS - set(raw)
    if unknown or missing:
        raise PdConfigError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    try:
        return _build_config(raw)
    except (GravityConfigError, GainsError, ValueError, TypeError, KeyError) as exc:
        raise PdConfigError(f"{path}: {exc}") from exc


def _build_config(raw: Mapping) -> PdConfig:
    if not isinstance(raw["execute"], bool):
        raise PdConfigError(f"execute must be a bool, got {raw['execute']!r}")
    scalars = {k: _num(raw[k], k, t) for k, t in _SCALARS.items()}
    for k in _SCALARS:
        if scalars[k] <= 0 and k != "vel_ff_cap":
            raise PdConfigError(f"{k} must be > 0, got {scalars[k]}")
    return PdConfig(
        side=str(raw["side"]), execute=raw["execute"], **scalars,
        max_vel=_block(MaxVel, raw["max_vel"], "max_vel"),
        settle=_block(Settle, raw["settle"], "settle"),
        gravity=_gravity_block(raw["gravity"]),
        gains=GainsBlock(yaml=_path(raw["gains"]["yaml"]),
                         accept_sim_mismatch=raw["gains"]["accept_sim_mismatch"]),
        thermal=thermal_rules_from_config(raw["thermal"] or []),
        gripper=None if raw["gripper"] is None else _block(GripperBlock, raw["gripper"], "gripper"),
        hand=None if raw["hand"] is None else _block(HandBlock, raw["hand"], "hand"),
    )


def _num(value, name: str, kind):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PdConfigError(f"{name} must be a number, got {value!r}")
    return kind(value)


def _block(cls, raw: Mapping, name: str):
    fields = cls.__dataclass_fields__
    if not isinstance(raw, Mapping) or set(raw) != set(fields):
        raise PdConfigError(f"{name}: expected keys {sorted(fields)}, got {raw!r}")
    return cls(**{k: _num(raw[k], f"{name}.{k}", float) for k in fields})


def _per_side(value, conv):
    """값 하나 또는 팔별 매핑({left: …, right: …}) → 같은 꼴로 변환(None 은 그대로)."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): conv(v) for k, v in value.items()}
    return conv(value)


def _floats(values) -> tuple:
    return tuple(float(v) for v in values)


def _gravity_block(raw: Mapping) -> GravityBlock:
    known = {"mode", "urdf", "tip_link", "scale", "payload", "cap_nm"}
    if not isinstance(raw, Mapping) or "mode" not in raw or set(raw) - known:
        raise PdConfigError(f"gravity: expected keys within {sorted(known)} with 'mode', got {raw!r}")
    return GravityBlock(
        mode=str(raw["mode"]),
        urdf=None if raw.get("urdf") is None else _path(raw["urdf"]),
        tip_link=_per_side(raw.get("tip_link"), str),
        scale=_per_side(raw.get("scale"), _floats),
        payload=_per_side(raw.get("payload"), _floats),
        cap_nm=None if raw.get("cap_nm") is None else float(raw["cap_nm"]),
    )


def _path(text: str) -> Path:
    p = Path(str(text)).expanduser()
    return p if p.is_absolute() else _paths.RL_WS / p


# ------------------------------------------------------------------ derived
def limits_from_profile(profile_path: Path, joints: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lower, upper, velocity) for canonical *joints* from a robot_control profile yaml."""
    prof = load_profile_joints(profile_path)
    rows = []
    for j in joints:
        if j not in prof:
            raise KeyError(f"joint {j!r} not in profile {profile_path}")
        if prof[j]["velocity"] is None:
            raise KeyError(f"joint {j!r} has no velocity limit in profile {profile_path}")
        rows.append((prof[j]["lower"], prof[j]["upper"], prof[j]["velocity"]))
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()


def side_vel_ff_scale(contract: DeployContract, side: str | None) -> float:
    """fabric.vel_ff_scale of *side* (legacy top-level fabric when side is None or the side has no fabric)."""
    fabric = contract.side(side).fabric if side is not None else None
    return float((fabric or contract.fabric).vel_ff_scale)


def law_cfg_from_config(config: PdConfig, contract: DeployContract, stage: str,
                        lower: np.ndarray, upper: np.ndarray, side: str | None = None) -> PdLawCfg:
    """yaml knobs + contract values → PdLawCfg for one stage ('ramp' | 'reduced' | 'full').

    ``side`` picks one arm of a bimanual contract (gravity mode / droop / vel_ff_scale of that side).
    """
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} not in {STAGES}")
    gravity, _ = contract_gravity(contract, side)
    conflict = gravity_conflict(config.gravity.mode, gravity.mode)
    if conflict:
        raise GravityConfigError(conflict)
    max_vel = {"ramp": config.ramp_speed, "reduced": config.max_vel.reduced,
               "full": config.max_vel.full}[stage]
    gain, limit = droop_params(contract, side) if config.gravity.mode == "integral_droop" else (None, None)
    return PdLawCfg(max_vel=max_vel, lead_sec=config.lead_sec, lead_vel=config.lead_vel,
                    vel_ff_scale=side_vel_ff_scale(contract, side), vel_ff_cap=config.vel_ff_cap,
                    effort_cap=config.effort_cap, lower=lower, upper=upper,
                    gravity_mode=config.gravity.mode, droop_gain=gain, droop_limit=limit)
