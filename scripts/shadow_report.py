#!/usr/bin/env python3
"""sim 그림자 기록(npz) 과 실기 측정(csv) 을 맞대어 세 층을 각각 판정한다.

    L1  FK(fabric_q) vs 지령 palm pose   Fabrics attractor 가 목표에 수렴하나
    L2  sim 물리 TCP  vs FK(fabric_q)     sim PD 가 fabric 해를 따라가나
    L3  실기 measured vs arm_target       실팔이 그 관절 목표를 따라가나 ← 본 질문

L1·L2 는 sim npz 만으로 나온다. 실기 csv 를 주면 L3 와 지연·지터가 더해진다.

**지연을 왜 따로 재는가.** 추종오차 한 덩어리로는 "느려서 뒤처진 것"과 "덜 가서 뒤처진
것"을 못 가른다. 앞은 대역폭 문제이고 뒤는 중력 처짐(정적)이라 고치는 노브가 다르다.
지연은 지령·측정 시계열의 교차상관 최대점으로 잡고, **그 지연만큼 밀어 정렬한 뒤 남는
오차**를 정적 성분으로 본다.

실행:
    python3 scripts/shadow_report.py --sim logs/shadow/sim_fab_test16_gcON.npz
    python3 scripts/shadow_report.py --sim ... --real logs/shadow/real_x1.csv --md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행 단위 두 쿼터니언 사이 각[deg]. 성분 순서와 무관하고, 부호 모호성은 abs 로 접는다.

    q 와 −q 는 같은 자세다 — abs 를 빼면 같은 자세를 180° 로 보고한다.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def lag_by_cross_correlation(command: np.ndarray, measured: np.ndarray,
                             max_lag: int) -> int:
    """측정이 지령보다 몇 스텝 뒤인가. 두 신호를 평균제거·정규화해 상관 최대점을 찾는다.

    변화가 없는 신호에서는 상관이 무의미하므로 0 을 돌려주고 호출측이 신호세기를
    같이 보게 한다 — "지연 0" 과 "잴 수 없었다" 는 다르다.
    """
    c = command - command.mean()
    m = measured - measured.mean()
    if np.linalg.norm(c) < 1e-9 or np.linalg.norm(m) < 1e-9:
        return 0
    best_lag, best_score = 0, -np.inf
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x, y = c, m
        else:
            x, y = c[:-lag], m[lag:]
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom < 1e-12:
            continue
        score = float(np.dot(x, y) / denom)
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def load_real(path: Path) -> dict[str, np.ndarray]:
    """재생기가 남긴 csv. `step_idx` 로 sim 프레임과 맞춘다."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path}: 비어 있다")
    out: dict[str, list] = {}
    for row in rows:
        for key, value in row.items():
            out.setdefault(key, []).append(float(value) if value not in ("", None) else np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


def _arm_gains(sim) -> str:
    """기록이 적어 둔 팔 게인. **하드코딩 금지** — 게인은 태스크마다 바뀐다.

    좌 그리퍼 트랙은 실기 effort 한계(j5~j7 = 7 N·m)에 맞춰 kp 80 / kd 4 로 낮춰 두었다.
    우팔(유휴)만 kp 400 이다. 보고서가 400 이라고 적으면 추종오차를 **완전히 반대로**
    읽게 된다 — "강성이 높은데도 이만큼 틀어졌다"가 "강성이 낮으니 당연하다"가 된다.
    """
    kp = sim.get("meta_arm_stiffness")
    kd = sim.get("meta_arm_damping")
    if kp is None or kd is None:
        return "게인 미상 — 기록에 meta_arm_stiffness/damping 이 없다"
    return f"kp {float(np.asarray(kp).reshape(-1)[0]):.0f} / kd {float(np.asarray(kd).reshape(-1)[0]):.0f}"


def report_sim(sim, lines):
    palm_cmd = sim["palm_cmd_pos"][:, 0]
    palm_fk = sim["palm_fk_pos"][:, 0]
    tcp = sim["tcp_pos"][:, 0]
    l1 = np.linalg.norm(palm_fk - palm_cmd, axis=-1) * 1000.0
    l1_rot = quat_angle(sim["palm_fk_quat_wxyz"][:, 0], sim["palm_cmd_quat_wxyz"][:, 0])
    l2 = np.linalg.norm(tcp - palm_fk, axis=-1) * 1000.0
    joints = [str(x) for x in sim["meta_joint_names"]]
    track = np.abs(sim["arm_meas"][:, 0] - sim["arm_target"][:, 0])

    lines.append(f"스텝 {sim['action'].shape[0]}  ·  step_dt {float(sim['meta_step_dt'][0]):.4f} s  "
                 f"·  중력보상 {str(sim['meta_gravity_comp'][0])}"
                 + ("" if "meta_gravity_comp_requested" in sim
                    else "  ⚠(구 기록: **요청값**이라 실제와 다를 수 있다)"))
    lines.append(f"fabrics_sim {str(sim['meta_fabrics'][0])}")
    lines.append("")
    lines.append("## L1 — Fabrics attractor 가 지령을 실현하나 (FK(fabric_q) vs palm 지령)")
    s = stats(l1); r = stats(l1_rot)
    lines.append(f"  위치 mean {s['mean']:7.2f}  p95 {s['p95']:7.2f}  max {s['max']:7.2f}  mm")
    lines.append(f"  자세 mean {r['mean']:7.2f}  p95 {r['p95']:7.2f}  max {r['max']:7.2f}  deg")
    lines.append("")
    lines.append("## L2 — sim 물리가 fabric 해를 따라가나 (물리 TCP vs FK)")
    s = stats(l2)
    lines.append(f"  위치 mean {s['mean']:7.2f}  p95 {s['p95']:7.2f}  max {s['max']:7.2f}  mm")
    lines.append("")
    lines.append(f"## sim 관절 추종오차 ({_arm_gains(sim)} — 실기 대비 기준선)")
    lines.append(f"  {'관절':10s} {'mean[mrad]':>11s} {'p95':>8s} {'max':>8s}")
    for i, name in enumerate(joints):
        c = track[:, i] * 1000.0
        lines.append(f"  {name:10s} {c.mean():11.2f} {np.percentile(c,95):8.2f} {c.max():8.2f}")
    lines.append("")
    lines.append("## 정책이 요구하는 것 (실기 능력과 대볼 값)")
    dt = float(sim["meta_step_dt"][0])
    vel = np.abs(np.diff(sim["arm_target"][:, 0], axis=0)) / dt
    lines.append(f"  {'관절':10s} {'mean[rad/s]':>12s} {'p95':>8s} {'max':>8s}")
    for i, name in enumerate(joints):
        c = vel[:, i]
        lines.append(f"  {name:10s} {c.mean():12.3f} {np.percentile(c,95):8.3f} {c.max():8.3f}")
    # ★palm 지령 이동량은 **기록에서 직접 유도**한다. 액션 항의 `cmd_step_norm` 은
    #   변화율 리미터가 꺼지면 갱신되지 않아 **정확히 0** 이 되는데, 그걸 그대로 실으면
    #   "지령이 전혀 안 움직인다"는 측정값처럼 읽힌다. 죽은 채널은 값이 아니다.
    step_mm = np.linalg.norm(np.diff(palm_cmd, axis=0), axis=-1) * 1000.0
    lines.append(f"  palm 지령 이동량  mean {step_mm.mean():.2f}  p95 "
                 f"{np.percentile(step_mm, 95):.2f}  max {step_mm.max():.2f} mm/step  (기록에서 유도)")
    counter = sim["cmd_step_norm"][:, 0]
    if not np.any(counter):
        lines.append("  ⚠ env 의 cmd_step_norm 은 전 구간 0 이다 — 리미터가 꺼져 갱신되지 "
                     "않는 죽은 채널이므로 쓰지 않는다.")
    if "droop" in sim:
        droop = np.abs(sim["droop"][:, 0]) * 1000.0
        lines.append("")
        lines.append(f"## 중력 처짐 보상분 (★sim {_arm_gains(sim)} 기준으로 상한이 잡혀 있다)")
        lines.append(f"  {'관절':10s} {'mean[mrad]':>11s} {'max':>8s}")
        for i, name in enumerate(joints):
            lines.append(f"  {name:10s} {droop[:, i].mean():11.2f} {droop[:, i].max():8.2f}")
    return joints


def report_real(sim, real, joints, lines, max_lag):
    # ★재생은 `--rate-scale` 로 느리게 돌 수 있다. 기록의 step_dt 를 그대로 쓰면 지연이
    #   그 배수만큼 틀린 ms 로 나온다. csv 가 자기 발행주기를 담고 있으면 그것을 쓴다.
    if "publish_dt" in real and np.isfinite(real["publish_dt"]).all():
        dt = float(np.median(real["publish_dt"]))
        scale = float(sim["meta_step_dt"][0]) / dt
        lines.append("")
        lines.append(f"재생 발행주기 {dt*1000:.2f} ms (기록 {float(sim['meta_step_dt'][0])*1000:.2f} ms "
                     f"· rate-scale {scale:.2f})")
    else:
        dt = float(sim["meta_step_dt"][0])
        lines.append("")
        lines.append("⚠ csv 에 publish_dt 가 없다 — 기록 step_dt 로 대신한다. "
                     "느리게 재생했다면 아래 지연 ms 는 그 배수만큼 작게 나온다.")
    idx = real["step_idx"].astype(int)
    keep = (idx >= 0) & (idx < sim["arm_target"].shape[0])
    idx = idx[keep]
    if idx.size == 0:
        raise SystemExit("실기 csv 의 step_idx 가 sim 기록 범위 밖이다 — 정렬 불가")
    target = sim["arm_target"][idx, 0]

    lines.append("")
    lines.append("## L3 — 실팔이 관절 목표를 따라가나")
    lines.append(f"  {'관절':10s} {'지연[스텝]':>10s} {'지연[ms]':>9s} "
                 f"{'RMSE[mrad]':>11s} {'정렬후 RMSE':>12s} {'max':>9s}")
    for i, name in enumerate(joints):
        column = f"meas_{name}"
        if column not in real:
            lines.append(f"  {name:10s} {'(csv 에 없음)':>50s}")
            continue
        measured = real[column][keep]
        lag = lag_by_cross_correlation(target[:, i], measured, max_lag)
        raw = (measured - target[:, i]) * 1000.0
        aligned = (measured[lag:] - target[: len(measured) - lag, i]) * 1000.0 if lag else raw
        lines.append(
            f"  {name:10s} {lag:10d} {lag*dt*1000:9.1f} "
            f"{np.sqrt(np.mean(raw**2)):11.2f} {np.sqrt(np.mean(aligned**2)):12.2f} "
            f"{np.abs(raw).max():9.2f}")
    lines.append("")
    lines.append("  ★정렬 후에도 남는 오차가 정적 성분(중력 처짐)이고, 정렬로 사라진 몫이")
    lines.append("    대역폭 성분이다. 두 성분은 고치는 노브가 다르다.")

    if "t_send" in real:
        send = real["t_send"][keep]
        gaps = np.diff(send) * 1000.0
        lines.append("")
        lines.append("## 발행 지터 (재생기가 실제로 낸 주기)")
        lines.append(f"  목표 {dt*1000:.2f} ms  ·  실측 mean {gaps.mean():.2f}  "
                     f"p95 {np.percentile(gaps,95):.2f}  max {gaps.max():.2f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--real", type=Path, default=None)
    parser.add_argument("--max-lag", type=int, default=30, help="교차상관 탐색 상한[스텝]")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sim = dict(np.load(args.sim, allow_pickle=False))
    lines = [f"# 그림자 판정 — {args.sim.name}", ""]
    joints = report_sim(sim, lines)
    if args.real:
        report_real(sim, load_real(args.real), joints, lines, args.max_lag)
    else:
        lines.append("")
        lines.append("실기 csv 없음 — L3·지연·지터는 재생 뒤에 채운다.")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
