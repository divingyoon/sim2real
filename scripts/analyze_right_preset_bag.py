#!/usr/bin/env python3
"""우팔 preset 재생 bag 분석 — SIM 대 REAL, 튜닝 입력용.

`record_right_shadow_bag.sh` 가 구운 bag 하나만 읽는다. 세 신호를 **구분해서** 본다:

  A `/shadow/sim_target`  sim 이 원한 관절 목표
  B `/isaacsim/right_arm_cmd`  우리가 실제로 보낸 세트포인트(리미터 통과 후)
  C `/joint_states`  실기 실측
  (참고) `/shadow/sim_meas`  sim 물리가 실현한 값 — sim 자체 처짐이 여기 들어 있다

 · **B−A** = 우리가 붙잡은 몫 (리미터/rate). 팔 탓이 아니다.
 · **C−B** = 실기 추종오차. ← 튜닝 대상
 · **C−D**(D=sim_meas) = sim↔실기 물리 차이. 그림자 목표를 sim 실측으로 두는 근거
   (08.31 좌팔: 지령 추종이 오히려 테이블을 긁었다).

지연 보정: C 를 B 에 대해 교차상관으로 정렬한 뒤 오차를 잰다. 안 하면 지연이
추종오차로 둔갑한다(좌팔 실측 +180 ms).

    python3 analyze_right_preset_bag.py logs/rosbags/right_preset_HHMMSS
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

ARM_CANON = [f"r_aj_{i}" for i in range(1, 8)]
ARM_SRC = [f"openarm_right_joint{i}" for i in range(1, 8)]
HAND_SRC = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]


def _read_topic(con: sqlite3.Connection, name: str, msg_type):
    row = con.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    from rclpy.serialization import deserialize_message
    out = []
    for stamp, data in con.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (row[0],)):
        out.append((stamp * 1e-9, deserialize_message(data, msg_type)))
    return out or None


def _stack_array(rows, n: int):
    """Float64MultiArray 스트림 → (t, values[n])."""
    t = np.array([r[0] for r in rows])
    v = np.array([list(r[1].data)[:n] for r in rows], dtype=float)
    return t, v


def _joint_states(rows, names):
    t, v = [], []
    for stamp, m in rows:
        pos = dict(zip(m.name, m.position))
        if not all(n in pos for n in names):
            continue
        t.append(stamp)
        v.append([pos[n] for n in names])
    if not t:
        return None, None
    return np.array(t), np.array(v, dtype=float)


def _interp(t_src, v_src, t_at):
    return np.stack([np.interp(t_at, t_src, v_src[:, k])
                     for k in range(v_src.shape[1])], axis=1)


def _lag_seconds(t_a, v_a, t_b, v_b, max_lag=1.0, step=0.005):
    """v_b 가 v_a 보다 얼마나 늦는가 — 잔차 최소가 되는 이동량."""
    grid = np.arange(t_a[0], t_a[-1], 0.02)
    a = _interp(t_a, v_a, grid)
    best, best_err = 0.0, None
    for lag in np.arange(0.0, max_lag, step):
        at = grid + lag
        keep = (at >= t_b[0]) & (at <= t_b[-1])
        if keep.sum() < 50:
            continue
        b = _interp(t_b, v_b, at[keep])
        err = float(np.sqrt(np.mean((b - a[keep]) ** 2)))
        if best_err is None or err < best_err:
            best, best_err = float(lag), err
    return best, best_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--max-lag", type=float, default=1.0)
    args = parser.parse_args()

    dbs = sorted(args.bag.glob("*.db3"))
    if not dbs:
        raise SystemExit(f"{args.bag} 에 .db3 가 없다")
    con = sqlite3.connect(str(dbs[0]))

    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    js = _read_topic(con, "/joint_states", JointState)
    if js is None:
        raise SystemExit("bag 에 /joint_states 가 없다 — 기록 범위를 확인할 것")
    cmd = _read_topic(con, "/isaacsim/right_arm_cmd", Float64MultiArray)
    tgt = _read_topic(con, "/shadow/sim_target", Float64MultiArray)
    smeas = _read_topic(con, "/shadow/sim_meas", Float64MultiArray)

    t_real, q_real = _joint_states(js, ARM_SRC)
    if t_real is None:
        raise SystemExit("/joint_states 에 우팔 관절이 없다")
    span = t_real[-1] - t_real[0]
    print(f"bag {args.bag.name} · /joint_states {len(t_real)} 샘플 · {span:.1f} s "
          f"({len(t_real)/max(span,1e-9):.0f} Hz)")
    if cmd is None:
        raise SystemExit("/isaacsim/right_arm_cmd 가 없다 — 재생이 이 bag 구간에 없었다")

    t_cmd, q_cmd = _stack_array(cmd, 7)
    print(f"보낸 세트포인트 {len(t_cmd)} 프레임 · {t_cmd[-1]-t_cmd[0]:.1f} s")
    # ★커버리지 확인 — bag 이 재생 전에 죽으면 interp 가 옛값을 클램프해 거짓 수치를 만든다
    overlap = min(t_cmd[-1], t_real[-1]) - max(t_cmd[0], t_real[0])
    if overlap < 0.5 * (t_cmd[-1] - t_cmd[0]):
        print(f"⚠ bag 이 재생 구간을 절반도 못 덮는다(겹침 {overlap:.1f} s) — 수치를 믿지 말 것")

    lag, lag_err = _lag_seconds(t_cmd, q_cmd, t_real, q_real, args.max_lag)
    print(f"\n실기 지연 {lag*1000:.0f} ms (정렬 후 RMSE {np.degrees(lag_err):.2f}°)")

    grid = np.arange(max(t_cmd[0], t_real[0] - lag), min(t_cmd[-1], t_real[-1] - lag), 0.02)
    c = _interp(t_cmd, q_cmd, grid)
    r = _interp(t_real, q_real, grid + lag)

    print("\n═══ C−B: 실기 추종오차 (지연 보정 후) ═══   ← 튜닝 대상")
    print(f"{'관절':10s} {'mean':>9s} {'RMSE':>9s} {'max':>9s}")
    err = r - c
    for k, n in enumerate(ARM_CANON):
        e = err[:, k]
        print(f"{n:10s} {np.degrees(e.mean()):+8.2f}° {np.degrees(np.sqrt((e**2).mean())):8.2f}° "
              f"{np.degrees(np.abs(e).max()):8.2f}°")
    print(f"{'전체':10s} {np.degrees(err.mean()):+8.2f}° "
          f"{np.degrees(np.sqrt((err**2).mean())):8.2f}° "
          f"{np.degrees(np.abs(err).max()):8.2f}°")

    if tgt is not None:
        t_t, q_t = _stack_array(tgt, 7)
        tt = _interp(t_t, q_t, grid)
        hold = c - tt
        print("\n═══ B−A: 우리가 리미터로 붙잡은 몫 (팔 탓이 아니다) ═══")
        print(f"  max {np.degrees(np.abs(hold).max()):.2f}° · "
              f"RMSE {np.degrees(np.sqrt((hold**2).mean())):.2f}°")

    if smeas is not None:
        t_s, q_s = _stack_array(smeas, 7)
        sm = _interp(t_s, q_s, grid)
        d = r - sm
        print("\n═══ C−D: 실기 vs **sim 실측** (그림자 목표 후보) ═══")
        print(f"{'관절':10s} {'mean':>9s} {'RMSE':>9s} {'max':>9s}")
        for k, n in enumerate(ARM_CANON):
            e = d[:, k]
            print(f"{n:10s} {np.degrees(e.mean()):+8.2f}° "
                  f"{np.degrees(np.sqrt((e**2).mean())):8.2f}° "
                  f"{np.degrees(np.abs(e).max()):8.2f}°")
        print("\n★선보상 후보 (부호 반전, canonical 순서 — --arm-offset 에 그대로)")
        print("  " + ",".join(f"{-v:+.4f}" for v in err.mean(axis=0)))

    jh = _read_topic(con, "/dg5f_right/joint_states", JointState)
    if jh is not None:
        t_h, q_h = _joint_states(jh, HAND_SRC)
        if t_h is not None:
            print(f"\n손 실측 {len(t_h)} 샘플 · 관절 {q_h.shape[1]} · "
                  f"이동폭 {np.degrees(np.abs(q_h[-1]-q_h[0]).max()):.1f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
