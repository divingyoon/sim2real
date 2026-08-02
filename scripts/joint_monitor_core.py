#!/usr/bin/env python3
"""joint_monitor 순수 로직 (ROS 무의존): 관절 분류·대시보드 포맷·CSV 행 조립.

sim2real 라이브 실행 중 팔(left/right)+tesollo 손 각 관절의 pos/vel/effort 를
터미널 표시 + CSV 기록하기 위한 순수 함수. effort(토크)를 함께 봐서 J7 같은
과부하 지점을 즉시 포착한다.
"""

from __future__ import annotations

from dataclasses import dataclass

# 그룹 표시 순서
GROUP_ORDER = ["right_arm", "left_arm", "tesollo_right", "gripper", "other"]
GROUP_LABEL = {
    "right_arm": "RIGHT ARM",
    "left_arm": "LEFT ARM",
    "tesollo_right": "TESOLLO (right hand)",
    "gripper": "GRIPPER",
    "other": "OTHER",
}


@dataclass(frozen=True)
class JointSample:
    pos: float
    vel: float
    eff: float


def classify_joint(name: str) -> str:
    """관절명(source 또는 canonical) → 그룹 키."""
    if name.startswith(("openarm_left_joint", "l_aj_")):
        return "left_arm"
    if name.startswith(("openarm_right_joint", "r_aj_")):
        return "right_arm"
    if name.startswith(("rj_dg_", "r_hj_")):
        return "tesollo_right"
    if "gripper" in name or "finger" in name or name.startswith("l_hj_"):
        return "gripper"
    return "other"


def group_records(records: dict[str, JointSample]) -> dict[str, list[tuple[str, JointSample]]]:
    """{name: JointSample} → {group: [(name, sample), ...]} (그룹별, 이름 정렬)."""
    grouped: dict[str, list[tuple[str, JointSample]]] = {g: [] for g in GROUP_ORDER}
    for name in sorted(records):
        grouped[classify_joint(name)].append((name, records[name]))
    return grouped


def format_dashboard(
    records: dict[str, JointSample],
    elapsed_sec: float,
    effort_warn: float = 5.0,
) -> str:
    """그룹별 관절 표. effort 절대값이 effort_warn 초과면 '*' 표시(과부하 경보)."""
    grouped = group_records(records)
    lines = [f"===== JOINT MONITOR  t={elapsed_sec:7.2f}s  (|eff|>{effort_warn:g}Nm = *) ====="]
    header = f"  {'joint':<24}{'pos(rad)':>11}{'vel(rad/s)':>12}{'eff(Nm)':>11}"
    for group in GROUP_ORDER:
        rows = grouped[group]
        if not rows:
            continue
        lines.append(f"[{GROUP_LABEL[group]}]")
        lines.append(header)
        for name, s in rows:
            flag = " *" if abs(s.eff) > effort_warn else ""
            lines.append(f"  {name:<24}{s.pos:>11.4f}{s.vel:>12.4f}{s.eff:>11.4f}{flag}")
    return "\n".join(lines)


def csv_header(joint_order: list[str]) -> list[str]:
    """CSV 헤더: t_sec + 각 관절의 pos/vel/eff 열."""
    cols = ["t_sec"]
    for name in joint_order:
        cols += [f"{name}.pos", f"{name}.vel", f"{name}.eff"]
    return cols


def csv_row(elapsed_sec: float, records: dict[str, JointSample], joint_order: list[str]) -> list[float]:
    """joint_order 순서로 t + pos/vel/eff 평탄화. 없는 관절은 NaN."""
    row: list[float] = [round(elapsed_sec, 4)]
    for name in joint_order:
        s = records.get(name)
        if s is None:
            row += [float("nan")] * 3
        else:
            row += [s.pos, s.vel, s.eff]
    return row
