#!/usr/bin/env python3
"""4 노드의 /policy_control/status/* (JSON) 와 /policy_control/pd/applied 를 seq 로 join → CSV.

지연(obs 발행 → pd 적용), 홉별 proc_ms, seq 결손, HOLD 사유가 한 표에 남는다.
라이브 구독 또는 rosbag2 재생 어느 쪽이든 같은 토픽이므로 같은 코드다.

    python3 policy_control/tools/status_to_csv.py --seconds 60 --out logs/policy_control/run.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

NODES = ("obs", "policy", "fabric", "pd")


def summarize(rows: list[dict], policy_dt: float | None) -> str:
    if not rows:
        return "no rows"
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    seqs = sorted(r["seq"] for r in rows)
    missing = (seqs[-1] - seqs[0] + 1 - len(seqs)) if seqs else 0
    p50 = lat[len(lat) // 2] if lat else float("nan")
    p95 = lat[int(0.95 * (len(lat) - 1))] if lat else float("nan")
    budget = "" if policy_dt is None else f" · budget 0.5·dt = {500 * policy_dt:.1f} ms"
    return f"rows {len(rows)} · latency p50 {p50:.2f} / p95 {p95:.2f} ms{budget} · seq missing {missing}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy-dt", type=float, default=None, help="예산 표시용 (s)")
    ap.add_argument("--jsonl", type=Path, default=None, help="모든 status 메시지를 그대로(한 줄 JSON) 남긴다 — 디버그용")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("status_to_csv")
    by_seq: dict[int, dict] = defaultdict(dict)
    raw = open(args.jsonl, "w") if args.jsonl else None

    def on_status(name):
        def cb(msg):
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            if raw is not None:
                raw.write(json.dumps({"topic": name, **d}, ensure_ascii=False) + "\n")
            seq = d.get("seq")
            if seq is None:
                return
            by_seq[int(seq)][name] = d
        return cb

    for n in NODES:
        node.create_subscription(String, f"/policy_control/status/{n}", on_status(n), 50)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if raw is not None:
        raw.close()

    rows = []
    for seq in sorted(by_seq):
        d = by_seq[seq]
        row = {"seq": seq}
        for n in NODES:
            s = d.get(n, {})
            row[f"{n}_ok"] = s.get("ok")
            row[f"{n}_proc_ms"] = s.get("proc_ms")
            row[f"{n}_t_pub_ns"] = s.get("t_pub_ns")
            row[f"{n}_reasons"] = "|".join(s.get("reasons", []))
        t_obs, t_pd = d.get("obs", {}).get("t_pub_ns"), d.get("pd", {}).get("t_pub_ns")
        row["latency_ms"] = None if (t_obs is None or t_pd is None) else (t_pd - t_obs) * 1e-6
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["seq"])
        w.writeheader()
        w.writerows(rows)
    print(summarize(rows, args.policy_dt), "→", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
