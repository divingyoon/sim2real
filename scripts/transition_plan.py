#!/usr/bin/env python3
"""파지 종료 → 붓기 시작 **전환 궤적**의 순수 코어 (sim·ROS 무의존).

**무엇을 푸는가.** 파지 정책이 끝난 자세와 붓기가 시작되는 자세가 다르다. 실기에는
텔레포트가 없으므로 팔이 그 사이를 **실제로 지나가야** 한다. 09.01 실측:

  · 우팔 E1 파지종료 → pour 시작 : L2 1.926 rad · 최대 76.0° (`r_aj_6`)
  · 좌팔 v2B25 파지종료 → pour rest : 아래 probe 로 측정

**왜 관절 직선인가.** 우팔에는 이 태스크에 fabric 이 없다. 계획기가 없으니 직선을
긋고 **sim 으로 확인**하는 것이 정직하다 — 되면 쓰고, 몸통을 관통하면 구간을 끊는다
([[both-arm-reset-trajectory]] 가 리셋 궤적에서 같은 길을 갔다).

**판정은 힘의 크기가 아니라 "새로 닿았는가" 다.** 관통력은 물리량이 아니고 solver·
자세에 따라 마구 변한다([[isaac-contact-measurement-traps]]). 시작 자세에서 이미
닿아 있던 것(파지한 컵, 손가락)은 기준선으로 빼고 **새로 닿은 몸통만** 센다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

#: 이보다 큰 접촉력을 "닿았다"로 본다 [N]. 수치 잡음과 실제 접촉을 가르는 선.
DEFAULT_CONTACT_THRESHOLD_N = 1.0

#: 링크가 작업면보다 이만큼 아래로 내려가면 경고한다 [m].
TABLE_CLEARANCE_M = 0.005


def steps_for(start: Sequence[float], goal: Sequence[float], *, max_vel: float, dt: float) -> int:
    """가장 많이 움직이는 관절이 속도 한계를 지키는 데 필요한 스텝 수."""
    span = float(np.abs(np.asarray(goal, float) - np.asarray(start, float)).max())
    return max(1, int(np.ceil(span / (max_vel * dt))))


def ramp(
    start: Sequence[float], goal: Sequence[float], *, max_vel: float, dt: float
) -> np.ndarray:
    """관절공간 직선 보간. (T, n) — 첫 행이 start, 마지막 행이 goal."""
    a = np.asarray(start, dtype=float)
    b = np.asarray(goal, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"start/goal 길이가 다르다: {a.shape} vs {b.shape}")
    if max_vel <= 0:
        raise ValueError(f"속도 한계는 양수여야 한다: {max_vel}")
    n = steps_for(a, b, max_vel=max_vel, dt=dt)
    if np.allclose(a, b):
        return a[None, :].copy()
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return a[None, :] + t * (b - a)[None, :]


def ramp_via(
    start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    goal: Sequence[float],
    *,
    max_vel: float,
    dt: float,
) -> np.ndarray:
    """경유점을 지나는 램프. 직선이 몸에 걸릴 때 자유 영역만 지나가게 끊는다.

    이어붙일 때 **경유점 프레임을 두 번 넣지 않는다** — 그러면 그 프레임에서 속도가
    0 이 되어 재생기가 멈춘 것으로 읽는다.
    """
    nodes = [list(start)] + [list(w) for w in waypoints] + [list(goal)]
    n = len(nodes[0])
    for i, node in enumerate(nodes):
        if len(node) != n:
            raise ValueError(f"{i}번째 자세의 길이가 다르다: {len(node)} vs {n}")
    segments = [ramp(a, b, max_vel=max_vel, dt=dt) for a, b in zip(nodes, nodes[1:])]
    return np.vstack([segments[0]] + [seg[1:] for seg in segments[1:]])


def contact_set(
    body_names: Sequence[str], forces: Sequence[float], *, threshold: float
) -> frozenset[str]:
    """접촉력이 임계를 넘은 몸통 이름 집합."""
    f = np.asarray(forces, dtype=float)
    if len(body_names) != f.shape[0]:
        raise ValueError(f"몸통 이름 {len(body_names)}개 vs 힘 {f.shape[0]}개 — 개수가 다르다")
    return frozenset(name for name, val in zip(body_names, f) if val > threshold)


def new_contacts(baseline: frozenset[str], current: frozenset[str]) -> frozenset[str]:
    """기준선에 없던 접촉만. 사라진 접촉은 세지 않는다."""
    return current - baseline


def describe_transition(
    label: str,
    path: np.ndarray,
    *,
    dt: float,
    worst: Mapping[str, tuple[int, float]],
    min_z: Mapping[str, float],
    table_z: float,
    baseline_z: Mapping[str, float] | None = None,
) -> str:
    """전환 하나의 판정문. `worst` = 몸통 → (최악 프레임, 그때 힘[N]).

    `baseline_z` 를 주면 **시작부터 작업면 아래에 있던 링크는 세지 않는다** — 베이스와
    몸통은 구조상 z=0 이라 그것을 침범으로 세면 무엇을 해도 실패한다.
    """
    n = path.shape[0]
    # 이동 시간은 **구간 수**×dt 다. 프레임 수로 재면 한 스텝만큼 부풀려진다.
    lines = [f"[{label}] 프레임 {n} · {(n - 1) * dt:.1f} s · 최대 |Δq| "
             f"{float(np.abs(path[-1] - path[0]).max()):.3f} rad"]
    base = baseline_z or {}
    low = {k: v for k, v in min_z.items()
           if v < table_z - TABLE_CLEARANCE_M and base.get(k, table_z) >= table_z}
    if worst:
        lines.append(f"  ❌ 새로 닿은 몸통 {len(worst)}개:")
        for name, (frame, force) in sorted(worst.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"     {name:24} 프레임 {frame:4d}  {force:.1f} N")
    if low:
        lines.append(f"  ❌ 작업면(z={table_z:.3f}) 아래로 내려간 링크 {len(low)}개:")
        for name, z in sorted(low.items(), key=lambda kv: kv[1]):
            lines.append(f"     {name:24} 최저 z {z:.4f}  ({(table_z - z)*1000:.0f} mm 아래)")
    if not worst and not low:
        lines.append("  ✅ 통과 — 새로 닿은 몸통 없음, 작업면 침범 없음")
    return "\n".join(lines)
