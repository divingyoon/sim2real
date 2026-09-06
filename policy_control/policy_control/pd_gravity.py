"""중력 모드 3종(계약이 정하고 설정이 확인한다) — 플랜 §4.4.

  off             τ_ff = 0
  integral_droop  τ_ff = 0. 보상은 pd_law 의 droop 적분(q 에 얹힘)이고 gain/limit 은 계약
                  (`contract.pd.gravity` 또는 `contract.sides[side].gravity`)에서 온다 — 좌 v2B25 (sim 로봇 중력 ON).
  model_tau_ff    τ_ff = scale ⊙ G(q_meas), gravity_comp_node.py 와 같은 수학:
                  robot_control.kinematics.chain_from_urdf + with_payload + Chain.gravity_torque.
                  cap 을 넘으면 노드가 0 을 보내고 HOLD 한다(`over_cap`) — 우 g1 (sim 중력 OFF).

두 모드를 동시에 켜는 것은 오류다(`gravity_conflict`) — 처짐을 두 번 보상하게 된다.

양팔 yaml: `tip_link` / `scale` / `payload` 는 값 하나(모든 팔 공통) 또는 팔별 매핑
``{left: …, right: …}`` 이다. ``block_for_side`` 가 한 팔 몫의 평범한 블록으로 푼다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from . import _paths  # noqa: F401  (robot_control on sys.path)
from .contract import SIDES, DeployContract

MODES = ("off", "integral_droop", "model_tau_ff")
_MODEL_FIELDS = ("urdf", "tip_link", "scale", "cap_nm")
_SIDED_FIELDS = ("tip_link", "scale", "payload")


class GravityConfigError(ValueError):
    """The gravity block is inconsistent with itself, the contract, or the URDF."""


# ------------------------------------------------------------------ yaml block
@dataclass(frozen=True)
class GravityBlock:
    """`gravity:` section of pd_*.yaml. Model fields are required iff mode == model_tau_ff.

    ``tip_link``/``scale``/``payload`` may be per-side mappings (``{left: …, right: …}``);
    resolve them with ``block_for_side`` before building a model.
    """
    mode: str
    urdf: Path | None = None
    tip_link: str | dict | None = None
    scale: tuple | dict | None = None
    payload: tuple | dict | None = None    # (MASS, X, Y, Z) or None
    cap_nm: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise GravityConfigError(f"gravity.mode {self.mode!r} not in {MODES}")
        present = [f for f in _MODEL_FIELDS if getattr(self, f) is not None]
        if self.mode == "model_tau_ff":
            missing = [f for f in _MODEL_FIELDS if getattr(self, f) is None]
            if missing:
                raise GravityConfigError(f"gravity.mode model_tau_ff needs {missing}")
        elif present or self.payload is not None:
            raise GravityConfigError(f"gravity.mode {self.mode} must not carry model fields {present}")
        for name in _SIDED_FIELDS:
            v = getattr(self, name)
            if isinstance(v, dict) and (not v or set(v) - set(SIDES)):
                raise GravityConfigError(f"gravity.{name}: per-side keys must be within {SIDES}, got {sorted(v)}")

    @property
    def sided(self) -> bool:
        return any(isinstance(getattr(self, f), dict) for f in _SIDED_FIELDS)


def _side_value(value, side: str | None, name: str):
    if not isinstance(value, dict):
        return value
    if side is None:
        raise GravityConfigError(f"gravity.{name} is per-side ({sorted(value)}) — a side must be selected")
    if side not in value:
        raise GravityConfigError(f"gravity.{name} has no entry for side {side!r} (has {sorted(value)})")
    return value[side]


def block_for_side(block: GravityBlock, side: str | None) -> GravityBlock:
    """Resolve per-side mappings to *side*'s plain values (a plain block passes through unchanged)."""
    return replace(block, **{f: _side_value(getattr(block, f), side, f) for f in _SIDED_FIELDS})


def gravity_conflict(config_mode: str, contract_mode: str) -> str | None:
    """None when the yaml agrees with the contract, else the refusal text."""
    for m in (config_mode, contract_mode):
        if m not in MODES:
            raise ValueError(f"gravity mode {m!r} not in {MODES}")
    if config_mode == contract_mode:
        return None
    return f"config gravity {config_mode} vs contract gravity {contract_mode}"


def contract_gravity(contract: DeployContract, side: str | None):
    """(GravityCfg, arm joint names) of *side* — or of the legacy primary section when side is None."""
    if side is None:
        return contract.pd.gravity, tuple(contract.pd.sim_gains.joints)
    s = contract.side(side)
    return s.gravity, tuple(s.sim_gains.joints)


def droop_params(contract: DeployContract, side: str | None = None) -> tuple[float, np.ndarray]:
    g, _ = contract_gravity(contract, side)
    if g.mode != "integral_droop" or g.gain is None or g.limit is None:
        raise GravityConfigError(f"contract gravity mode {g.mode!r} carries no droop gain/limit")
    limit = np.asarray(g.limit, dtype=np.float64)
    if g.gain <= 0.0 or np.any(limit < 0.0):
        raise GravityConfigError("droop gain must be > 0 and limits >= 0")
    return float(g.gain), limit


# ------------------------------------------------------------------ model
@dataclass(frozen=True)
class ModelGravityCfg:
    urdf: Path
    tip_link: str
    joints: tuple                   # canonical arm joint names, chain order
    scale: tuple                    # per joint
    payload: tuple | None           # (MASS, X, Y, Z) in the tip link frame
    cap_nm: float


@dataclass(frozen=True)
class ModelGravity:
    """Callable τ(q_meas) = scale ⊙ G(q); `over_cap` mirrors gravity_comp_node's stop rule."""
    chain: object
    scale: np.ndarray
    cap_nm: float

    def __call__(self, q_meas: np.ndarray) -> np.ndarray:
        q = np.asarray(q_meas, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self.scale) or not np.isfinite(q).all():
            raise ValueError(f"q_meas must be {len(self.scale)} finite values, got {q}")
        return self.chain.gravity_torque(q) * self.scale

    def over_cap(self, tau: np.ndarray) -> bool:
        return bool(np.max(np.abs(np.asarray(tau, dtype=np.float64))) > self.cap_nm)


def build_model_gravity(cfg: ModelGravityCfg) -> ModelGravity:
    from robot_control.kinematics import KinematicsError, chain_from_urdf, with_payload

    n = len(cfg.joints)
    scale = np.asarray(cfg.scale, dtype=np.float64).reshape(-1)
    if scale.shape[0] != n or not np.isfinite(scale).all():
        raise ValueError(f"scale needs {n} finite values, got {cfg.scale}")
    if cfg.payload is not None and len(cfg.payload) != 4:
        raise ValueError(f"payload must be (MASS, X, Y, Z), got {cfg.payload}")
    if cfg.cap_nm <= 0.0:
        raise ValueError(f"cap_nm must be > 0, got {cfg.cap_nm}")
    urdf = Path(cfg.urdf)
    if not urdf.is_file():
        raise GravityConfigError(f"gravity urdf missing: {urdf}")
    try:
        chain = chain_from_urdf(urdf.read_text(), list(cfg.joints), cfg.tip_link)
        if cfg.payload is not None:
            chain = with_payload(chain, float(cfg.payload[0]), [float(v) for v in cfg.payload[1:]])
    except KinematicsError as exc:
        raise GravityConfigError(f"{urdf.name}: {exc}") from exc
    return ModelGravity(chain=chain, scale=scale, cap_nm=float(cfg.cap_nm))


# ------------------------------------------------------------------ factory
def zero_gravity(n_joints: int) -> Callable[[np.ndarray], np.ndarray]:
    def zero(q_meas: np.ndarray) -> np.ndarray:
        q = np.asarray(q_meas, dtype=np.float64).reshape(-1)
        if q.shape[0] != n_joints:
            raise ValueError(f"q_meas must have {n_joints} values, got {q.shape[0]}")
        return np.zeros(n_joints)
    return zero


def make_gravity(block: GravityBlock, contract: DeployContract | None,
                 n_joints: int | None = None, side: str | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """yaml block + contract → τ_ff(q_meas). Refuses a mode disagreement with the contract.

    ``side`` selects the arm of a bimanual contract and resolves per-side block values.
    """
    blk = block_for_side(block, side)
    if contract is not None:
        g, joints = contract_gravity(contract, side)
        conflict = gravity_conflict(blk.mode, g.mode)
        if conflict:
            raise GravityConfigError(conflict)
    elif n_joints is None:
        raise GravityConfigError("make_gravity needs a contract or n_joints")
    else:
        joints = ()
    if blk.mode == "model_tau_ff":
        if not joints:
            raise GravityConfigError("model_tau_ff needs the contract's arm joint names")
        return build_model_gravity(ModelGravityCfg(
            urdf=blk.urdf, tip_link=blk.tip_link, joints=joints, scale=tuple(blk.scale),
            payload=None if blk.payload is None else tuple(blk.payload), cap_nm=blk.cap_nm))
    return zero_gravity(len(joints) if joints else int(n_joints))
