#!/usr/bin/env python3
"""체인 토픽 기록기 — obs/action/joint_target/pd applied/joint_states 를 seq 기준으로 npz 에 남긴다(bag 대용).

    python3 policy_control/tools/chain_recorder.py --contract <json> --seconds 30 --out logs/policy_control/run.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import codec  # noqa: E402
from policy_control import contract as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side", choices=("left", "right"), default=None,
                    help="양팔 계약에서 기록할 팔(기본 = 계약 primary/obs 팔)")
    args = ap.parse_args()
    contract = C.load_contract(args.contract)
    segments = [(s.name, s.dim) for s in contract.obs.segments]
    arm = list(contract.side(args.side).arm_joints if args.side else contract.obs.joint_orders["arm"])
    side = "left" if arm[0].startswith("l_") else "right"
    src = [f"openarm_{side}_joint{j.split('_')[-1]}" for j in arm]
    hand = list(contract.side(args.side).hand_joints) if args.side else []   # 손 목표는 있으면 따로(제어 전용 direct 모드)
    tgt_hand: dict = {}

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = Node("chain_recorder")
    obs, act, tgt, app, js = {}, {}, {}, {}, []

    def on_obs(m):
        try:
            v, seq = codec.decode_obs(m, segments)
            obs[seq] = (time.time(), np.asarray(v, float))
        except codec.CodecError:
            pass

    def on_act(m):
        try:
            v, seq = codec.decode_action(m, contract.policy.action_dim)
            act[seq] = np.asarray(v, float)
        except codec.CodecError:
            pass

    def _arm_only(names, *vecs):
        idx = {n: i for i, n in enumerate(names)}
        if not all(a in idx for a in arm):
            return None
        return tuple(np.array([float(v[idx[a]]) for a in arm]) if len(v) == len(names) else np.zeros(len(arm))
                     for v in vecs)

    def on_tgt(m):
        try:
            t = codec.decode_joint_target(m, list(m.name))
        except codec.CodecError:
            return
        sel = _arm_only(list(t.names), t.position, t.velocity)
        if sel is not None:
            tgt[t.seq] = (arm, *sel)
        idx = {n: i for i, n in enumerate(t.names)}
        if hand and all(h in idx for h in hand):
            tgt_hand[t.seq] = np.array([float(t.position[idx[h]]) for h in hand])

    def on_app(m):
        sel = _arm_only(list(m.name), m.position, m.velocity, m.effort)
        if sel is not None:
            app[time.time()] = (arm, *sel)

    def on_js(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if all(s in idx for s in src):
            js.append((time.time(), np.array([m.position[idx[s]] for s in src]),
                       np.array([m.velocity[idx[s]] for s in src]) if len(m.velocity) == len(m.name) else np.zeros(len(src))))

    node.create_subscription(Float64MultiArray, "/policy_control/obs", on_obs, 100)
    node.create_subscription(Float64MultiArray, "/policy_control/action", on_act, 100)
    node.create_subscription(JointState, "/policy_control/joint_target", on_tgt, 100)
    node.create_subscription(JointState, "/policy_control/pd/applied", on_app, 100)
    node.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()

    seqs = sorted(obs) if obs else sorted(tgt)      # obs 노드가 없는 제어 전용 런은 joint_target seq 로 정렬
    out = {
        "seq": np.array(seqs), "t_obs": np.array([obs[s][0] if s in obs else np.nan for s in seqs]),
        "obs": np.stack([obs[s][1] for s in seqs]) if obs else np.zeros((0, contract.policy.obs_dim)),
        "segments": np.array([f"{n}:{d}" for n, d in segments]),
        "action": np.stack([act.get(s, np.full(contract.policy.action_dim, np.nan)) for s in seqs]) if seqs and act else np.zeros((0, 1)),
        "target_names": np.array(arm),
        "target_q": np.stack([tgt[s][1] if s in tgt else np.full(len(arm), np.nan) for s in seqs]) if seqs else np.zeros((0, len(arm))),
        "target_qd": np.stack([tgt[s][2] if s in tgt else np.full(len(arm), np.nan) for s in seqs]) if seqs else np.zeros((0, len(arm))),
        "hand_target_names": np.array(hand),
        "hand_target_q": (np.stack([tgt_hand[s] if s in tgt_hand else np.full(len(hand), np.nan) for s in seqs])
                          if seqs and hand else np.zeros((0, len(hand)))),
        "t_js": np.array([r[0] for r in js]), "js_q": np.stack([r[1] for r in js]) if js else np.zeros((0, len(src))),
        "js_qd": np.stack([r[2] for r in js]) if js else np.zeros((0, len(src))),
        "t_app": np.array(sorted(app)), "app_q": np.stack([app[t][1] for t in sorted(app)]) if app else np.zeros((0, 1)),
        "app_qd": np.stack([app[t][2] for t in sorted(app)]) if app else np.zeros((0, 1)),
        "app_tau": np.stack([app[t][3] for t in sorted(app)]) if app else np.zeros((0, 1)),
        "app_names": np.array(arm),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[recorder] obs {len(seqs)} · js {len(js)} · applied {len(app)} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
