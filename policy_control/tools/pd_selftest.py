#!/usr/bin/env python3
"""pd 노드 셀프테스트 — lowlevel_check TEST0/1/2 를 **forward 백엔드 경로**로 재현한다.

pd 노드가 engage 된 상태(phase RAMPING|TRACKING)에서 /policy_control/joint_target 으로
① TEST0 기동 자세 기록 → ② TEST1 hold(목표 = 시작 실측, hold_s) 드리프트 → ③ TEST2 관절별
±진폭 스텝(목표 스트림은 0.1 rad/s 로 전진, pd 도 자기 rate-limit 을 건다) 을 보내고
/joint_states 로 sign/ratio/crosstalk 표를 만든다(lowlevel_check_core 재사용). 결과 JSON.

    python3 policy_control/tools/pd_selftest.py --contract logs/policy/left_v2B25/deploy_contract.json \
        --robot policy_control/config/robots/left_gripper_real.yaml --execute --out logs/measure/pd_selftest.json
★실기 동작이다 — 사용자 승인 후에만 --execute. 승인 없이는 계획만 출력한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_control import _paths  # noqa: E402,F401
from policy_control import codec  # noqa: E402
from policy_control import contract as C  # noqa: E402
from policy_control.sources import load_robot_cfg  # noqa: E402

from jtc_bridge_core import velocity_limited_target  # noqa: E402
from lowlevel_check_core import (build_step_plan, evaluate_hold, evaluate_step,  # noqa: E402
                                 summarize_sign_table)

TARGET_TOPIC = "/policy_control/joint_target"
STATE_TOPIC = "/joint_states"
PD_STATUS = "/policy_control/status/pd"
EPISODE_TOPIC = "/policy_control/episode"
STOP_WAIT_S = 1.0         # episode stop 이 pd 에 닿을 때까지(< watchdog 0.25 s 가 정상, 여유 포함)
STREAM_HZ = 50.0
PARK_SPEED = 0.1          # rad/s — robotctl/shadow_replay 와 같은 값·이유
ABORT_DEVIATION_RAD = 0.45


def plan_text(joints, plan) -> str:
    lines = [f"[selftest] joints {joints}", f"[selftest] {len(plan)} segments:"]
    for s in plan:
        lines.append(f"  {s.phase:5s} {s.joint or '-':8s} {s.amplitude:+.3f} rad {s.duration_s:.1f} s")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--robot", type=Path, required=True)
    ap.add_argument("--amplitudes", default="0.02,0.05,0.10")
    ap.add_argument("--hold-s", type=float, default=10.0)
    ap.add_argument("--dwell-s", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--side", choices=("left", "right"), default=None,
                    help="양팔 계약에서 시험할 팔(기본 = 계약 primary/obs 팔)")
    ap.add_argument("--joints", default=None,
                    help="스텝을 줄 관절 부분집합 CSV(canonical, 예 l_aj_4,l_aj_6). 기본 = 팔 7관절 전부. 목표는 항상 7관절 전부 싣는다")
    ap.add_argument("--execute", action="store_true", help="★없으면 계획만 출력")
    ap.add_argument("--no-stop-event", action="store_true", default=False,
                    help="끝에 /policy_control/episode stop 을 내지 않는다(그러면 스트림 두절로 pd 가 watchdog HOLD 로 간다)")
    args = ap.parse_args()
    contract = C.load_contract(args.contract)
    cfg = load_robot_cfg(args.robot)
    joints = list(contract.side(args.side).arm_joints if args.side else contract.obs.joint_orders["arm"])
    step_joints = [j.strip() for j in args.joints.split(",")] if args.joints else list(joints)
    unknown = [j for j in step_joints if j not in joints]
    if unknown:
        raise SystemExit(f"[selftest] --joints {unknown} 는 팔 관절 {joints} 에 없다")
    amps = tuple(float(a) for a in args.amplitudes.split(","))
    plan = build_step_plan(step_joints, amps, dwell_s=args.dwell_s, hold_s=args.hold_s)
    print(plan_text(joints, plan))
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute (pd 가 engage 되어 있어야 한다)")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    rclpy.init()
    node = Node("pd_selftest")
    src_of = {c: v["source"] for c, v in cfg.joint_limits.items()} if hasattr(cfg, "joint_limits") else None
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    state: dict = {}
    phase: dict = {"pd": None, "target": None}

    def on_js(m):
        state.update(zip(m.name, m.position))
        state["_t"] = time.time()

    def on_pd(m):
        try:
            body = json.loads(m.data)
        except json.JSONDecodeError:
            return
        phase["pd"], phase["target"] = body.get("phase"), body.get("target")

    node.create_subscription(JointState, STATE_TOPIC, on_js, qos_profile_sensor_data)
    node.create_subscription(String, PD_STATUS, on_pd, 10)
    pub = node.create_publisher(JointState, TARGET_TOPIC, 10)
    # episode stop 발행자는 **지금** 만든다 — 끝에 만들면 discovery 전에 프로세스가 끝나 latched 메시지가 사라진다(run right1)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ep_pub = node.create_publisher(String, EPISODE_TOPIC, latched)
    side = "left" if joints[0].startswith("l_") else "right"
    source = [f"openarm_{side}_joint{j.split('_')[-1]}" for j in joints]

    def measured() -> np.ndarray:
        return np.array([state[s] for s in source], dtype=float)

    t0 = time.time()
    while time.time() - t0 < 5.0 and not (all(s in state for s in source) and phase["pd"]):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not all(s in state for s in source):
        raise SystemExit(f"[selftest] {STATE_TOPIC} 수신 없음")
    if phase["pd"] not in ("RAMPING", "TRACKING"):
        raise SystemExit(f"[selftest] pd phase {phase['pd']!r} — engage 먼저(episode_ctl --approve pd_engage)")

    dt = 1.0 / STREAM_HZ
    seq = 0

    def send(q_target: np.ndarray, qd: np.ndarray | None = None) -> None:
        nonlocal seq
        msg = codec.encode_joint_target(joints, q_target, np.zeros(len(joints)) if qd is None else qd,
                                        episode="selftest", seq=seq)
        pub.publish(msg)
        seq += 1

    def stream(target: np.ndarray, duration_s: float, sp: np.ndarray) -> tuple[list, np.ndarray]:
        samples = []
        n = max(1, int(duration_s / dt))
        for _ in range(n):
            sp = velocity_limited_target(target, sp, PARK_SPEED, dt)
            send(sp)
            rclpy.spin_once(node, timeout_sec=dt)
            q = measured()
            if float(np.abs(q - sp).max()) > ABORT_DEVIATION_RAD:
                raise SystemExit(f"[selftest] 추종 편차 {float(np.abs(q - sp).max()):.3f} rad > {ABORT_DEVIATION_RAD} — 중단")
            samples.append(q)
        return samples, sp

    base = measured()
    report = {"joints": joints, "start_q": base.tolist(), "hold": None, "steps": [], "table": None}
    sp = base.copy()
    verdicts = []
    for spec in plan:
        if spec.phase == "hold":
            samples, sp = stream(base, spec.duration_s, sp)
            report["hold"] = evaluate_hold(joints, samples)
            print(f"[selftest] hold drift worst: " + ", ".join(f"{j} {v:+.4f}" for j, v in report["hold"].items()))
            continue
        target = base.copy()
        target[joints.index(spec.joint)] += spec.amplitude
        samples, sp = stream(target, spec.duration_s, sp)
        v = evaluate_step(joints, base, samples[-1], spec)
        verdicts.append(v)
        report["steps"].append(v.__dict__)
        print(f"[selftest] {spec.joint} {spec.amplitude:+.2f}: measured {v.measured:+.4f} ratio {v.ratio:.2f} "
              f"{'ok' if v.ok else v.reason}")
        _, sp = stream(base, spec.duration_s, sp)          # 기준 자세로 복귀
    report["table"] = summarize_sign_table(verdicts, joints)
    report["ok"] = all(v.ok for v in verdicts)
    send(base)
    if not args.no_stop_event:
        # 스트림이 끊기면 pd 는 watchdog HOLD 로 간다(가드). 시험이 끝났다는 뜻으로 episode stop 을 내어
        # pd 가 마지막 세트포인트를 내부 목표로 **의도적으로** 붙들게 한다(episode_ctl 의 ep_stop 과 같은 경로).
        # pd status 의 target 이 internal 로 바뀔 때까지(≤ watchdog) 목표를 계속 보내며 기다린다.
        ep_pub.publish(String(data=json.dumps({"episode": 0, "event": "stop", "object_anchor": None, "home_q": {},
                                               "reasons": ["pd_selftest done"], "t_ns": time.time_ns()})))
        t1 = time.time()
        while time.time() - t1 < STOP_WAIT_S and phase["target"] != "internal":
            rclpy.spin_once(node, timeout_sec=dt)
        print(f"[selftest] episode stop → pd target {phase['target']!r}")
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"[selftest] → {args.out}")
    print(f"[selftest] {'OK' if report['ok'] else 'FAIL'}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
