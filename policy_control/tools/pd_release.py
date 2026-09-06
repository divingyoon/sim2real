#!/usr/bin/env python3
"""pd 노드 밖의 복구 CLI — forward 컨트롤러에 0 을 보내고 JTC 로 되돌린다.

pd 노드가 죽으면 하드웨어는 마지막 (q*, q̇*, τ_ff) 를 **영구 유지**한다(워치독 없음). 이
스크립트는 (1) velocity/effort 에 0 을 5회 송출하고 (2) switch_controller 로 JTC 를 다시
활성화한다. position 은 건드리지 않는다 — JTC 가 현재 위치를 이어받아 홀딩한다.

    python3 policy_control/tools/pd_release.py --side left --execute
"""
from __future__ import annotations

import argparse
import time

ZERO_TICKS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", required=True, help="left | right | both | left,right (팔마다 차례로)")
    ap.add_argument("--execute", action="store_true", help="★없으면 계획만 출력한다")
    args = ap.parse_args()
    sides = parse_sides(args.side)
    plans = {s: ([f"{s}_forward_{k}_controller" for k in ("position", "velocity", "effort")],
                 f"{s}_joint_trajectory_controller") for s in sides}
    for s, (fwd, jtc) in plans.items():
        print(f"[pd_release] {s}: 0 ×{ZERO_TICKS} → /{fwd[1]}/commands, /{fwd[2]}/commands · switch → {jtc}")
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute")
        return 0

    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("pd_release")
    rc = 0
    for s, (fwd, jtc) in plans.items():
        rc = max(rc, release_one(node, fwd, jtc))
    node.destroy_node()
    rclpy.shutdown()
    return rc


def parse_sides(text: str) -> list[str]:
    t = text.strip().lower()
    sides = ["right", "left"] if t in ("both", "all") else [s.strip() for s in t.split(",") if s.strip()]
    bad = [s for s in sides if s not in ("left", "right")]
    if bad or not sides or len(set(sides)) != len(sides):
        raise SystemExit(f"--side 는 left | right | both | left,right — 받은 값 {text!r}")
    return sides


def release_one(node, fwd: list[str], jtc: str) -> int:
    import rclpy
    from controller_manager_msgs.srv import SwitchController
    from std_msgs.msg import Float64MultiArray

    pubs = [node.create_publisher(Float64MultiArray, f"/{n}/commands", 10) for n in fwd[1:]]
    time.sleep(0.3)
    for _ in range(ZERO_TICKS):
        for p in pubs:
            p.publish(Float64MultiArray(data=[0.0] * 7))
        rclpy.spin_once(node, timeout_sec=0.02)
    client = node.create_client(SwitchController, "/controller_manager/switch_controller")
    if not client.wait_for_service(3.0):
        print("[pd_release] controller_manager 없음 — 0 송출만 했다")
        return 2
    req = SwitchController.Request()
    req.activate_controllers, req.deactivate_controllers = [jtc], fwd
    req.strictness, req.activate_asap = req.BEST_EFFORT, True
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    ok = bool(fut.result() and fut.result().ok)
    print(f"[pd_release] switch → {jtc}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
