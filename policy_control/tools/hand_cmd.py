#!/usr/bin/env python3
"""hand_cmd — control-only fabric 노드에 손 관절 목표를 보낸다 (/policy_control/hand_cmd, JointState canonical).

    python3 policy_control/tools/hand_cmd.py --contract logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json \
        --side left --close 0.0                       # 계약 open pose (home_hand)
    python3 policy_control/tools/hand_cmd.py --contract … --side right --close 0.6      # open→grip 60 %
    python3 policy_control/tools/hand_cmd.py --contract … --side right --close 0 --joint r_hj_index_2=0.8
    python3 policy_control/tools/hand_cmd.py --contract … --side left --close 1 --dry-run

규약
  · open = 계약 `sides[side].home_hand`(자산 계약: 우 HAND_OPEN_POSE, 좌는 _HAND_SIGN 미러).
    grip = grasp_s2r_synergy.HAND_GRIP_POSE(우) / _HAND_SIGN 미러(좌) — 학습 트랙과 같은 폐쇄 자세.
    --close f ∈ [0,1] 는 open→grip 선형 보간, --joint name=value 가 개별 관절을 덮어쓴다.
  · 1.8 rad 같은 과지령은 fabric 노드의 DirectDecoder 가 자산 soft limit 으로 자른다.
  · ROS_DOMAIN_ID 0/unset(실기 기본 도메인)은 거부한다 — --allow-domain-0 로만 통과한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # sim2real/policy_control → `import policy_control`

from palm_cmd import check_domain  # noqa: E402  (same tools/ dir)
from policy_control import contract as C  # noqa: E402
from policy_control.contract_assets import _mirror_signs  # noqa: E402

NS = "/policy_control"
TOPIC_HAND_CMD = f"{NS}/hand_cmd"
HOLD_HZ = 10.0


def grip_pose(side: str, hand_joints: list) -> dict:
    """학습 트랙의 완전 파지 자세(우 HAND_GRIP_POSE), 좌는 _HAND_SIGN 미러."""
    from grasp_s2r_synergy import HAND_GRIP_POSE, HAND_JOINT_NAMES

    right = dict(zip(HAND_JOINT_NAMES, HAND_GRIP_POSE))
    if side == "right":
        vals = right
    else:
        _, sign = _mirror_signs()
        vals = {f"l{n[1:]}": s * v for (n, v), s in zip(right.items(), sign)}
    missing = [j for j in hand_joints if j not in vals]
    if missing:
        raise ValueError(f"grip pose lacks {missing}")
    return {j: float(vals[j]) for j in hand_joints}


def hand_targets(open_pose: dict, grip: dict, close: float, overrides: dict | None = None) -> dict:
    """open→grip 보간 후 개별 관절 덮어쓰기. 새 dict 를 돌려준다."""
    if not 0.0 <= close <= 1.0:
        raise ValueError(f"--close 는 0..1, got {close}")
    out = {j: (1.0 - close) * open_pose[j] + close * grip[j] for j in open_pose}
    for j, v in (overrides or {}).items():
        if j not in out:
            raise ValueError(f"--joint {j}: 이 손의 관절이 아니다 (허용 {sorted(out)[:3]}…)")
        out[j] = float(v)
    return out


def parse_overrides(items: list) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--joint 형식은 name=value, got {it!r}")
        name, val = it.split("=", 1)
        out[name.strip()] = float(val)
    return out


# ---------------------------------------------------------------- ROS
class Runner:
    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        rclpy.init()
        self.rclpy, self.JointState = rclpy, JointState
        self.node = Node("hand_cmd")
        self.pub = self.node.create_publisher(JointState, TOPIC_HAND_CMD, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def wait_subscriber(self, timeout: float = 3.0) -> int:
        """fabric 노드가 이 토픽을 구독할 때까지 기다린다 — RELIABLE 이어도 매칭 전 발행은 버려진다
        (fake_direct_right_run2: 두 번째 hand_cmd 가 조용히 사라졌다)."""
        deadline = time.monotonic() + timeout
        while self.pub.get_subscription_count() < 1 and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.pub.get_subscription_count()

    def publish(self, targets: dict, hold: float) -> int:
        n, deadline = 0, time.monotonic() + max(hold, 0.0)
        while True:
            msg = self.JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = list(targets)
            msg.position = [float(v) for v in targets.values()]
            self.pub.publish(msg)
            n += 1
            if time.monotonic() >= deadline:
                return n
            self.rclpy.spin_once(self.node, timeout_sec=1.0 / HOLD_HZ)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def _parse(argv) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", type=Path, required=True, help="deploy_contract.json (sides[side].home_hand = open)")
    ap.add_argument("--side", required=True, choices=("left", "right"))
    ap.add_argument("--close", type=float, default=0.0, help="0 = open, 1 = grip (선형 보간)")
    ap.add_argument("--joint", action="append", default=[], metavar="NAME=VALUE", help="개별 관절 덮어쓰기(canonical)")
    ap.add_argument("--hold", type=float, default=0.0, help=f"이 시간(s) 동안 {HOLD_HZ:.0f} Hz 로 재발행 (기본 1회)")
    ap.add_argument("--allow-domain-0", action="store_true", help="★실기 기본 도메인에서도 발행한다")
    ap.add_argument("--dry-run", action="store_true", help="목표만 계산·출력하고 발행하지 않는다")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    refuse = check_domain(dict(os.environ), args.allow_domain_0)
    if refuse:
        print(f"✗ {refuse}", file=sys.stderr)
        return 3
    try:
        side = C.load_contract(args.contract).side(args.side)
        if side.ee_kind != "dg5f":
            raise ValueError(f"side {args.side} ee_kind {side.ee_kind!r}: hand_cmd 는 DG-5F 손만 안다")
        targets = hand_targets(dict(side.home_hand), grip_pose(args.side, list(side.hand_joints)), args.close,
                               parse_overrides(args.joint))
    except (C.ContractError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    for j, v in targets.items():
        print(f"  {j:16s} {v:+.3f}")
    if args.dry_run:
        print("DRY RUN — 발행하지 않았다")
        return 0
    runner = Runner()
    try:
        if runner.wait_subscriber() < 1:
            print(f"✗ {TOPIC_HAND_CMD} 구독자가 없다 — fabric_node(control-only, direct 모드) 가 떠 있나?", file=sys.stderr)
            return 2
        n = runner.publish(targets, args.hold)
        print(f"published {TOPIC_HAND_CMD} ×{n} ({len(targets)} joints, close {args.close:.2f})")
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
