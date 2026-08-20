#!/usr/bin/env python3
"""저수준 제어 검증 순수 코어 (numpy only, ROS 무의존).

가이드(`sim2real_rl_debugging_guide.md`) Stage 0~2 를 **정책 없이** 수행하기 위한 로직.
ROS 노드(`lowlevel_check.py`)가 이 코어로 계획을 만들고 결과를 판정한다.

  TEST1 hold : q_cmd = q_measured 를 유지 → 관절별 드리프트(= 중력 처짐)를 잰다
  TEST2 step : 관절 하나만 ±Δ 움직여 **그 관절만 / 올바른 방향 / 올바른 크기**를 확인
               → 가이드 §6 의 sign·offset 표를 채운다

설계 원칙
  · 모든 함수는 새 배열을 반환한다(입력 불변).
  · 판정 실패를 조용히 넘기지 않고 사유 문자열을 남긴다.
  · 안전 상한(`MAX_SAFE_AMPLITUDE_RAD`)을 코드에 박아, 오타 하나로 팔이 크게 움직이는 사고를 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 단일 관절 스텝 진폭의 절대 상한. 저수준 검증에 이보다 큰 값은 필요하지 않다.
MAX_SAFE_AMPLITUDE_RAD = 0.15

#: 실측/명령 비가 이 범위를 벗어나면 크기 불일치. 하한이 넉넉한 이유 —
#: 중력 처짐이 큰 관절은 명령의 일부만 도달하는 게 **정상**이다(계획 B1).
RATIO_TOLERANCE = (0.4, 1.6)

#: 목표 외 관절이 이보다 많이 움직이면 간섭(매핑 오류)으로 본다.
CROSSTALK_LIMIT_RAD = 0.010

DEFAULT_AMPLITUDES = (0.02, 0.05, 0.10)


@dataclass(frozen=True)
class StepSpec:
    """계획의 한 구간. `phase="hold"` 면 `joint`/`amplitude` 는 의미 없음."""

    phase: str
    joint: str | None
    amplitude: float
    duration_s: float


@dataclass(frozen=True)
class JointVerdict:
    joint: str
    commanded: float
    measured: float
    ratio: float
    crosstalk_joint: str | None
    crosstalk: float
    ok: bool
    reason: str


def build_step_plan(
    joints,
    amplitudes=DEFAULT_AMPLITUDES,
    dwell_s: float = 2.0,
    hold_s: float = 10.0,
) -> tuple[StepSpec, ...]:
    """hold 1구간 + 관절마다 ±진폭 스텝 구간.

    각 스텝은 노드가 **기준 자세에서** 시작하므로(clamp_command 의 base) 계획에는
    복귀 구간을 넣지 않는다.
    """
    joints = list(joints)
    if not joints:
        raise ValueError("관절 목록이 비어 있다")
    amps = tuple(float(a) for a in amplitudes)
    if not amps:
        raise ValueError("진폭 목록이 비어 있다")
    for a in amps:
        if not (0.0 < abs(a) <= MAX_SAFE_AMPLITUDE_RAD):
            raise ValueError(f"진폭 {a} 가 안전 상한 {MAX_SAFE_AMPLITUDE_RAD} rad 를 벗어난다")

    plan: list[StepSpec] = [StepSpec("hold", None, 0.0, hold_s)]
    for j in joints:
        for a in amps:
            plan.append(StepSpec("step", j, +abs(a), dwell_s))
            plan.append(StepSpec("step", j, -abs(a), dwell_s))
    return tuple(plan)


def clamp_command(base, target, joints, limits: dict[str, dict], max_step: float) -> np.ndarray:
    """`target` 을 기준 자세 대비 `max_step`, 그리고 관절 한계 안으로 자른 **새 배열**."""
    base = np.asarray(base, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    joints = list(joints)
    if base.shape != target.shape or base.shape[0] != len(joints):
        raise ValueError(f"차원 불일치: base {base.shape}, target {target.shape}, 관절 {len(joints)}")
    delta = np.clip(target - base, -abs(max_step), abs(max_step))
    lo = np.array([limits[j]["lower"] for j in joints], dtype=np.float64)
    hi = np.array([limits[j]["upper"] for j in joints], dtype=np.float64)
    return np.clip(base + delta, lo, hi)


def evaluate_step(
    joints,
    base_q,
    end_q,
    spec: StepSpec,
    ratio_tolerance: tuple[float, float] = RATIO_TOLERANCE,
    crosstalk_limit: float = CROSSTALK_LIMIT_RAD,
) -> JointVerdict:
    """한 스텝의 결과를 판정. 실패 사유는 부호 → 크기 → 간섭 순으로 확정한다."""
    if spec.phase != "step" or spec.joint is None:
        raise ValueError(f"step 구간이 아니다: phase={spec.phase}")
    joints = list(joints)
    base_q = np.asarray(base_q, dtype=np.float64)
    end_q = np.asarray(end_q, dtype=np.float64)
    if base_q.shape != end_q.shape or base_q.shape[0] != len(joints):
        raise ValueError("차원 불일치: base_q/end_q/관절")

    idx = joints.index(spec.joint)
    delta = end_q - base_q
    measured = float(delta[idx])
    commanded = float(spec.amplitude)
    ratio = measured / commanded if commanded else float("nan")

    others = np.abs(np.delete(delta, idx))
    if others.size:
        k = int(np.argmax(others))
        ct_joint = [j for j in joints if j != spec.joint][k]
        ct = float(others[k])
    else:
        ct_joint, ct = None, 0.0

    reason = ""
    if measured == 0.0 or np.sign(measured) != np.sign(commanded):
        reason = f"부호 불일치: 명령 {commanded:+.3f} → 실측 {measured:+.4f}"
    elif not (ratio_tolerance[0] <= ratio <= ratio_tolerance[1]):
        reason = f"크기 불일치: 비율 {ratio:.2f} (허용 {ratio_tolerance})"
    elif ct > crosstalk_limit:
        reason = f"간섭: {ct_joint} 가 {ct:+.4f} rad 움직임"

    return JointVerdict(
        joint=spec.joint,
        commanded=commanded,
        measured=measured,
        ratio=ratio,
        crosstalk_joint=ct_joint,
        crosstalk=ct,
        ok=not reason,
        reason=reason,
    )


def evaluate_hold(joints, samples) -> dict[str, float]:
    """hold 구간의 관절별 **부호 있는 최대 드리프트**(첫 샘플 기준).

    부호를 지우지 않는다 — 중력 처짐은 항상 한쪽으로 나므로 부호가 곧 진단 정보다.
    """
    joints = list(joints)
    samples = [np.asarray(s, dtype=np.float64) for s in samples]
    if not samples:
        raise ValueError("샘플이 비어 있다")
    ref = samples[0]
    if any(s.shape != ref.shape for s in samples):
        raise ValueError("샘플 차원이 서로 다르다")
    if ref.shape[0] != len(joints):
        raise ValueError("차원 불일치: 샘플/관절")
    deltas = np.array([s - ref for s in samples])           # (N, J)
    worst = deltas[np.argmax(np.abs(deltas), axis=0), np.arange(len(joints))]
    return {j: float(worst[k]) for k, j in enumerate(joints)}


def summarize_sign_table(verdicts, all_joints=None) -> dict[str, dict]:
    """판정 목록 → 관절별 sign·ok 표 (가이드 §6).

    `sign` = 명령 부호 대비 실측 부호. 측정된 스텝이 없으면 **None** 으로 남긴다 —
    0.0 이나 1.0 으로 채우면 "안 재봤다"와 "재봤더니 정상"이 구분되지 않는다.
    """
    table: dict[str, dict] = {}
    for j in list(all_joints or []):
        table[j] = {"sign": None, "ok": None, "steps": 0, "reasons": []}
    for v in verdicts:
        row = table.setdefault(v.joint, {"sign": None, "ok": None, "steps": 0, "reasons": []})
        sign = float(np.sign(v.measured) * np.sign(v.commanded)) if v.measured else 0.0
        if row["sign"] is None:
            row["sign"] = sign
        row["steps"] += 1
        row["ok"] = v.ok if row["ok"] is None else (row["ok"] and v.ok)
        if v.reason:
            row["reasons"].append(v.reason)
    return table
