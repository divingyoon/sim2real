#!/usr/bin/env python3
"""에피소드 성공 판정 — /policy_control/obs·status/pd·joint_target 를 구독해 계약 세그먼트로 판정한다.

좌(v2B25) 기준(플랜 §6 M8): ① gripper_gate 가 1 이 된 뒤 ≤ close_steps 안에 그리퍼 목표 ≤ closed_m
② 부착(게이트) 이후 물체 z 가 시작 대비 ≥ lift_m 상승을 hold_steps 동안 유지 ③ pd HOLD 0회
④ obs 세그먼트 값이 유한. 결과는 JSON 으로 stdout + --out.

    python3 policy_control/tools/episode_judge.py --contract logs/policy/left_v2B25/deploy_contract.json --seconds 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import codec  # noqa: E402
from policy_control import contract as C  # noqa: E402


def judge(obs_rows: list[np.ndarray], grip_cmds: list[float], pd_phases: list[str], contract: C.DeployContract,
          close_steps: int, closed_m: float, lift_m: float, hold_steps: int) -> dict:
    seg = {s.name: s for s in contract.obs.segments}
    off, offsets = 0, {}
    for s in contract.obs.segments:
        offsets[s.name] = (off, off + s.dim)
        off += s.dim
    if not obs_rows:
        return {"success": False, "reasons": ["no obs received"]}
    obs = np.stack(obs_rows)
    reasons = []
    gate = obs[:, offsets["gripper_gate"][0]] if "gripper_gate" in seg else None
    obj_z = obs[:, offsets["object_position"][0] + 2] if "object_position" in seg else None
    if not np.isfinite(obs).all():
        reasons.append("non-finite obs")
    gate_on = int(np.argmax(gate > 0.5)) if gate is not None and gate.max() > 0.5 else None
    if gate_on is None:
        reasons.append("gate never opened")
    else:
        closes = [k for k, g in enumerate(grip_cmds) if g <= closed_m]
        first_close = next((k for k in closes if k >= gate_on), None)
        if first_close is None or first_close - gate_on > close_steps:
            reasons.append(f"gripper did not close within {close_steps} steps of gate (gate {gate_on}, close {first_close})")
        if obj_z is not None:
            rise = obj_z - obj_z[0]
            held = np.convolve((rise >= lift_m).astype(float), np.ones(hold_steps), "valid") >= hold_steps if len(rise) >= hold_steps else np.array([False])
            if not held.any():
                reasons.append(f"object z rise {rise.max():.3f} m < {lift_m} m for {hold_steps} steps")
    holds = sum(1 for p in pd_phases if p == "HOLD")
    if holds:
        reasons.append(f"pd HOLD seen {holds}×")
    return {"success": not reasons, "reasons": reasons, "steps": int(len(obs)), "gate_on": gate_on,
            "obj_z_rise_max": None if obj_z is None else float((obj_z - obj_z[0]).max()),
            "pd_phases": sorted(set(pd_phases))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--close-steps", type=int, default=40)
    ap.add_argument("--closed-m", type=float, default=0.01)
    ap.add_argument("--lift-m", type=float, default=0.05)
    ap.add_argument("--hold-steps", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    contract = C.load_contract(args.contract)
    segments = [(s.name, s.dim) for s in contract.obs.segments]

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    rclpy.init()
    node = Node("episode_judge")
    obs_rows, grip_cmds, phases = [], [], []

    def on_obs(msg):
        try:
            vec, _seq = codec.decode_obs(msg, segments)
        except codec.CodecError as exc:
            node.get_logger().warning(f"obs decode: {exc}")
            return
        obs_rows.append(np.asarray(vec, dtype=float))

    def on_target(msg: JointState):
        names = list(msg.name)
        grip = [n for n in names if "gripper" in n]
        if grip:
            grip_cmds.append(float(msg.position[names.index(grip[0])]))

    def on_pd(msg: String):
        try:
            phases.append(str(json.loads(msg.data).get("phase")))
        except json.JSONDecodeError:
            pass

    node.create_subscription(Float64MultiArray, "/policy_control/obs", on_obs, 50)
    node.create_subscription(JointState, "/policy_control/joint_target", on_target, 50)
    node.create_subscription(String, "/policy_control/status/pd", on_pd, 50)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    verdict = judge(obs_rows, grip_cmds, phases, contract, args.close_steps, args.closed_m, args.lift_m, args.hold_steps)
    text = json.dumps(verdict, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if verdict["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
