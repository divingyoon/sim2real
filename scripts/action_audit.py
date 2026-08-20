#!/usr/bin/env python3
"""정책 출력 감사 (numpy only) — 가이드 Stage 5 "오프라인 추론" 판정.

실기 관측으로 정책만 돌리고 **명령은 보내지 않은 채** 액션 분포를 본다. 여기서 걸러야
하는 것은 가이드가 나열한 그대로다:

    NaN / Inf / 포화 / [-1,1] 초과 / 일정한 출력(죽은 정책) / 급변

왜 별도 모듈인가: 판정 기준을 노드 안에 흩어두면 오프라인 재현기와 라이브 노드가 서로
다른 기준을 쓰게 된다. 한 곳에 두고 양쪽이 같은 걸 본다.
"""

from __future__ import annotations

import numpy as np

#: |a| 가 이 값 이상이면 포화로 센다(정책 tanh 출력의 실질 상한).
SATURATION_LEVEL = 0.99
#: 액션 계약 범위. 이걸 넘으면 디코더가 clip 해 **조용히** 의미가 달라진다.
ACTION_LIMIT = 1.0
#: 연속 스텝 변화가 이보다 크면 급변으로 본다(가이드 §14 Case E).
JUMP_WARN = 0.5
#: 전 스텝 표준편차가 이보다 작으면 "일정한 출력"으로 본다.
CONSTANT_STD = 1e-6


class ActionAudit:
    """액션을 스트리밍으로 받아 통계를 누적한다(전체 이력을 들고 있지 않는다)."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.steps = 0
        self.nan_steps = 0
        self.inf_steps = 0
        self.out_of_range_steps = 0
        self.max_jump = 0.0
        self.lo = np.full(self.dim, np.inf)
        self.hi = np.full(self.dim, -np.inf)
        self._sum = np.zeros(self.dim)
        self._sumsq = np.zeros(self.dim)
        self._sat = np.zeros(self.dim)
        self._prev: np.ndarray | None = None

    def add(self, action) -> None:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.dim:
            raise ValueError(f"액션 차원 불일치: {a.shape[0]} != {self.dim}")
        self.steps += 1
        if np.isnan(a).any():
            self.nan_steps += 1
        if np.isinf(a).any():
            self.inf_steps += 1

        finite = np.isfinite(a)
        if finite.any():
            f = np.where(finite, a, np.nan)
            self.lo = np.fmin(self.lo, f)
            self.hi = np.fmax(self.hi, f)
            clean = np.where(finite, a, 0.0)
            self._sum += clean
            self._sumsq += clean * clean
            self._sat += (np.abs(clean) >= SATURATION_LEVEL) & finite
            if (np.abs(clean[finite]) > ACTION_LIMIT).any():
                self.out_of_range_steps += 1
            if self._prev is not None:
                both = finite & np.isfinite(self._prev)
                if both.any():
                    self.max_jump = max(
                        self.max_jump, float(np.max(np.abs(clean - self._prev)[both]))
                    )
            self._prev = clean

    # -- 조회 -----------------------------------------------------------
    def mean(self) -> np.ndarray:
        return self._sum / self.steps if self.steps else np.zeros(self.dim)

    def std(self) -> np.ndarray:
        if self.steps < 2:
            return np.zeros(self.dim)
        var = self._sumsq / self.steps - self.mean() ** 2
        return np.sqrt(np.maximum(var, 0.0))

    def saturated_frac(self) -> np.ndarray:
        return self._sat / self.steps if self.steps else np.zeros(self.dim)

    def is_constant(self) -> bool:
        """2스텝 이상 받았고 전 차원 표준편차가 사실상 0이면 죽은 정책."""
        return self.steps >= 2 and bool(np.all(self.std() < CONSTANT_STD))


def audit_report(audit: ActionAudit) -> str:
    """사람이 읽는 한 줄 요약 + 문제 목록. 문제가 없으면 그렇게 말한다."""
    if audit.steps == 0:
        return "액션 표본 없음 — 정책이 한 번도 호출되지 않았다"

    problems: list[str] = []
    if audit.nan_steps:
        problems.append(f"NaN {audit.nan_steps}/{audit.steps} 스텝")
    if audit.inf_steps:
        problems.append(f"Inf {audit.inf_steps}/{audit.steps} 스텝")
    if audit.out_of_range_steps:
        problems.append(
            f"범위 초과(|a|>{ACTION_LIMIT}) {audit.out_of_range_steps}/{audit.steps} 스텝"
        )
    if audit.is_constant():
        problems.append("출력이 일정 — 정책이 관측에 반응하지 않는다")
    if audit.max_jump > JUMP_WARN:
        problems.append(f"급변 최대 {audit.max_jump:.3f} (>{JUMP_WARN})")
    sat = audit.saturated_frac()
    hot = np.where(sat > 0.5)[0]
    if hot.size:
        problems.append(
            "상시 포화 차원 " + ", ".join(f"a{i}({sat[i]*100:.0f}%)" for i in hot)
        )

    head = (
        f"{audit.steps} 스텝 · 범위 [{np.nanmin(audit.lo):+.3f}, {np.nanmax(audit.hi):+.3f}] "
        f"· 최대급변 {audit.max_jump:.3f} · 평균포화 {sat.mean()*100:.1f}%"
    )
    if not problems:
        return head + "\n  이상 없음"
    return head + "\n  " + "\n  ".join("⚠ " + p for p in problems)
