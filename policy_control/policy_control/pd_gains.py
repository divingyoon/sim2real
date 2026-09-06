"""게인 게이트 — control_gains.yaml(드라이버 kp/kd, bringup 시 1회 주입) vs 계약의 학습 게인.

kp 가 다르면 정책을 재현할 수 없으니 engage 를 거부한다(`accept_sim_mismatch` 로만 통과,
status 에 기록). kd 는 정보다: 우 g1 의 kd 는 r2s fit 이고 MIT 패킷 한계(5.0)를 넘는
값은 'impossible' 로 표기한다. 비교 자체는 contract.compare_gains/require_gains 를 재사용.

DG-5F 손 PID(벤더 p 1.5 / d 0.0 — 2026-09-06 벤더 전용 규칙)는 여기서 **기대값만** 노출한다 — 드라이버 파라미터 대조는
M7 노드가 GetParameters 로 한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .contract import (MIT_KD_MAX, ContractError, DeployContract, GainMismatch,
                       GainReport as _ContractReport, compare_gains)


class GainsError(ValueError):
    """The gains block or the driver gains file is unusable."""


@dataclass(frozen=True)
class GainsBlock:
    """`gains:` section of pd_*.yaml."""
    yaml: Path
    accept_sim_mismatch: bool

    def __post_init__(self) -> None:
        if not isinstance(self.accept_sim_mismatch, bool):
            raise GainsError(f"gains.accept_sim_mismatch must be a bool, got {self.accept_sim_mismatch!r}")


@dataclass(frozen=True)
class HandGains:
    pid_p: float
    pid_d: float


@dataclass(frozen=True)
class GainReport:
    ok: bool
    reasons: list
    kd_note: str
    real_kp: list
    real_kd: list
    impossible_kd: tuple            # joints whose trained kd cannot be encoded on the MIT packet
    accepted_mismatch: bool         # ok was False but accept_sim_mismatch let it through


def load_and_check(cfg: GainsBlock, contract: DeployContract, side: str | None = None) -> GainReport:
    """Compare driver gains with the contract; raise GainMismatch unless accepted.

    ``side`` picks one arm of a bimanual contract (``contract.sides[side].sim_gains``); the default is
    the legacy top-level ``pd.sim_gains`` (= primary side).
    """
    path = Path(cfg.yaml)
    if not path.is_file():
        raise GainsError(f"driver gains file missing: {path}")
    try:
        rep: _ContractReport = compare_gains(contract, path, side=side)
    except ContractError as exc:
        raise GainsError(str(exc)) from exc
    if not rep.ok and not cfg.accept_sim_mismatch:
        raise GainMismatch("; ".join(rep.reasons))
    return GainReport(ok=rep.ok, reasons=list(rep.reasons), kd_note=rep.kd_note,
                      real_kp=list(rep.real_kp), real_kd=list(rep.real_kd),
                      impossible_kd=_impossible_kd(contract, side),
                      accepted_mismatch=not rep.ok)


def _impossible_kd(contract: DeployContract, side: str | None = None) -> tuple:
    g = contract.pd.sim_gains if side is None else contract.side(side).sim_gains
    return tuple(j for j, kd in zip(g.joints, g.kd)
                 if re.fullmatch(r"[lr]_aj_[1-7]", j) and kd > MIT_KD_MAX)


def expected_hand_gains(config) -> HandGains | None:
    """DG-5F PID the node must find on the driver (None for a robot without the hand)."""
    hand = getattr(config, "hand", None)
    if hand is None:
        return None
    return HandGains(pid_p=float(hand.pid_p), pid_d=float(hand.pid_d))
