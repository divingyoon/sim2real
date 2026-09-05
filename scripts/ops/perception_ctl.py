#!/usr/bin/env python3
"""노드 2 의 사용자 면 — 물체를 골라 인지 체인을 켜고 끈다.

  perception_ctl.py start shaker_closed cup_big_s100 [--viewer]
  perception_ctl.py stop [--camera]
  perception_ctl.py viewer on|off
  perception_ctl.py status
  perception_ctl.py list

이름은 레지스트리(config/objects.yaml)로 검증·alias 해석 후 /perception/cmd 에 발행한다.
perception_launcher_node.py 가 떠 있어야 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))


from object_registry import load_registry  # noqa: E402


def build_payload(args, registry) -> dict | None:
    if args.op == "start":
        payload = {"op": "start", "objects": [registry.resolve(n) for n in args.objects]}
        if args.viewer:
            payload["viewer"] = True      # 키를 빼면 런처는 뷰어를 건드리지 않는다(False 는 '내려라')
        return payload
    if args.op == "stop":
        return {"op": "stop", "camera": bool(args.camera)}
    if args.op == "viewer":
        return {"op": "viewer", "on": args.on == "on"}
    return None


def print_status(payload: dict) -> None:
    print(f"camera: {'up' if payload['camera_up'] else 'down'} ({payload['camera_hz']} Hz)"
          f" · viewer: {payload['viewer']} · busy: {payload['busy']}")
    for name, info in payload["objects"].items():
        age = info["pose_age_s"]
        print(f"  {name:16s} container={info['container'] or '-':22s} "
              f"pose={'-' if age is None else f'{age:.2f}s ago'}")
    if payload.get("error"):
        print(f"  ERROR: {payload['error']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)
    s = sub.add_parser("start")
    s.add_argument("objects", nargs="+")
    s.add_argument("--viewer", action="store_true")
    st = sub.add_parser("stop")
    st.add_argument("--camera", action="store_true", help="카메라까지 내린다")
    v = sub.add_parser("viewer")
    v.add_argument("on", choices=("on", "off"))
    sub.add_parser("status")
    sub.add_parser("list")
    args = ap.parse_args()
    registry = load_registry()
    if args.op == "list":
        for name in registry.names():
            print(f"{name:16s} {registry.get(name).real}")
        for alias, target in registry.aliases.items():
            print(f"{alias:16s} → {target}")
        return 0
    payload = build_payload(args, registry)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("perception_ctl")
    if payload is not None:
        pub = node.create_publisher(String, "/perception/cmd", 10)
        deadline = time.monotonic() + 3.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            print("perception_launcher_node 가 안 떠 있다 (/perception/cmd 구독자 0)", file=sys.stderr)
            return 1
        pub.publish(String(data=json.dumps(payload)))
        print(f"sent {payload}")
    box: list[str] = []
    node.create_subscription(String, "/perception/status", lambda m: box.append(m.data), 10)
    deadline = time.monotonic() + 3.0
    while not box and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if box:
        print_status(json.loads(box[-1]))
    else:
        print("/perception/status 가 안 온다", file=sys.stderr)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
