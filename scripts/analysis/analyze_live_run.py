#!/usr/bin/env python3
"""라이브 그림자 런 분석 — sim 대 실기.

입력: ①sim npz(probe_v2_shadow_record, 20ms/스텝) ②어댑터 CSV(t_send/t_recv·지령)
      ③rosbag(/joint_states 실측). 라이브는 scale s 로 시간이 1/s 배 — sim 스텝 k 의
      지령이 실기에 t0 + k·(0.02/s) 에 도착한다. 정렬 키는 어댑터 CSV 의 실측 시각.

출력: 관절별 L3(실측 vs 지령) mean/RMSE/max · 지연 · palm z 궤적(미러 FK) 비교.
★좌팔 FK 는 없다 — 좌팔 q 를 부호 미러해 우팔 FK 에 넣으면 palm (x,−y,z): z 는 동일
  (URDF y-미러). 홈 z≈0.41 로 자기검증 후 신뢰.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from palm_fk import palm_pose                    # 우팔 FK (base 프레임)

ARM_SRC = [f"openarm_left_joint{i}" for i in range(1, 8)]


def read_bag_joint_states(bag_dir: Path):
    db = sorted(bag_dir.glob("*.db3"))[0]
    con = sqlite3.connect(str(db))
    cur = con.execute(
        "SELECT t.id, t.name FROM topics t WHERE t.name='/joint_states'")
    row = cur.fetchone()
    if row is None:
        raise SystemExit("bag 에 /joint_states 없음")
    tid = row[0]
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    ts, qs = [], []
    for stamp, data in con.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
        m = deserialize_message(data, JointState)
        pos = dict(zip(m.name, m.position))
        if not all(n in pos for n in ARM_SRC):
            continue
        ts.append(stamp * 1e-9)
        qs.append([pos[n] for n in ARM_SRC])
    return np.array(ts), np.array(qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--bag", type=Path, required=True)
    args = ap.parse_args()

    # ① 지령 스트림 (어댑터 CSV — 실제 벽시계에 실린 지령)
    rows = list(csv.DictReader(open(args.adapter)))
    t_cmd = np.array([float(r["t_recv"]) for r in rows])
    q_cmd = np.array([[float(r[f"arm_target_{i}"]) for i in range(7)] for r in rows])
    # ② 실측
    t_ms, q_ms = read_bag_joint_states(args.bag)
    # 런 구간만: 지령 시작~끝 ±1s
    sel = (t_ms >= t_cmd[0] - 0.5) & (t_ms <= t_cmd[-1] + 1.0)
    t_ms, q_ms = t_ms[sel], q_ms[sel]
    print(f"지령 {len(t_cmd)} 프레임 ({t_cmd[-1]-t_cmd[0]:.1f}s) · 실측 {len(t_ms)} 샘플")

    # ③ L3: 실측 시각마다 최신 지령과 비교 (선보상은 브리지가 더하므로 지령은 순수값 —
    #    실측도 순수 지령에 수렴해야 맞다. 즉 여기 오차에는 보상 효과가 이미 반영된다.)
    idx = np.searchsorted(t_cmd, t_ms, side="right") - 1
    ok = idx >= 0
    err = q_ms[ok] - q_cmd[idx[ok]]
    canon = [f"l_aj_{i}" for i in range(1, 8)]
    print(f"\n{'관절':8s}{'mean[mrad]':>11s}{'RMSE':>8s}{'max':>8s}")
    for j, c in enumerate(canon):
        e = err[:, j] * 1000
        print(f"{c:8s}{e.mean():11.1f}{np.sqrt((e**2).mean()):8.1f}{np.abs(e).max():8.1f}")
    # 지연: 대표 관절(l_aj_2)의 교차상관
    tt = np.arange(t_cmd[0], t_cmd[-1], 0.02)
    a = np.interp(tt, t_cmd, q_cmd[:, 1]); b = np.interp(tt, t_ms, q_ms[:, 1])
    a -= a.mean(); b -= b.mean()
    lag = (np.argmax(np.correlate(b, a, "full")) - (len(a) - 1)) * 0.02
    print(f"\n지연(l_aj_2 교차상관): {lag*1000:+.0f} ms")

    # ④ palm z (미러 FK)
    # ★mirror sign 을 소스에서 못 읽는다(manager-based 는 env 파일이 없다) —
    #   grasp_s2r robot_profiles 의 좌우 rest 대응(l=-r, j4 만 동부호)에서 실측 확정된
    #   패턴을 쓴다. 아래 홈 z 자기검증(≈0.41)이 틀리면 이 가정부터 의심할 것.
    sign = np.array([-1, -1, -1, +1, -1, -1, -1], float)
    def palm_z(q7):
        pos, _ = palm_pose(list(sign * np.asarray(q7)), apply_offset=False)
        return float(pos[2])
    z_home = palm_z([-0.0136, -0.3757, -0.0010, 0.9336, -0.4655, 0.0003, -0.3306])
    print(f"\n[자기검증] 미러 FK 홈 palm z = {z_home:.4f} m (기대 ≈0.41)")
    z_cmd = np.array([palm_z(q) for q in q_cmd[::5]])
    z_ms = np.array([palm_z(q) for q in q_ms[::20]])
    print(f"palm z — 지령 min {z_cmd.min():.4f} · 실측 min {z_ms.min():.4f} "
          f"· 차이(처짐) {1000*(z_cmd.min()-z_ms.min()):+.1f} mm")
    print(f"palm z — 지령 mean {z_cmd.mean():.4f} · 실측 mean {z_ms.mean():.4f}")


if __name__ == "__main__":
    main()
