#!/usr/bin/env python3

"""Publish manual test commands to the Isaac Sim bridge topics from a terminal."""

from __future__ import annotations

import argparse

import rclpy

import sys
from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabrics_ros_interface import Sim2RealCommandPublisher


def _float_list(raw_values: list[str]) -> list[float]:
    return [float(value) for value in raw_values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish direct commands to /isaacsim/* topics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    left_arm_parser = subparsers.add_parser("left-arm")
    left_arm_parser.add_argument("values", nargs=7)

    left_gripper_parser = subparsers.add_parser("left-gripper")
    left_gripper_parser.add_argument("value")

    left_full_parser = subparsers.add_parser("left-full")
    left_full_parser.add_argument("arm", nargs=7)
    left_full_parser.add_argument("gripper")

    right_arm_parser = subparsers.add_parser("right-arm")
    right_arm_parser.add_argument("values", nargs=7)

    right_hand_parser = subparsers.add_parser("right-hand")
    right_hand_parser.add_argument("values", nargs=20)

    right_full_parser = subparsers.add_parser("right-full")
    right_full_parser.add_argument("arm", nargs=7)
    right_full_parser.add_argument("hand", nargs=20)

    args = parser.parse_args()

    rclpy.init()
    node = Sim2RealCommandPublisher()

    if args.command == "left-arm":
        values = _float_list(args.values)
        node.send_left_arm(values)
        print(f"Published /isaacsim/left_arm_cmd: {values}")
    elif args.command == "left-gripper":
        value = float(args.value)
        node.send_left_gripper(value)
        print(f"Published /isaacsim/left_gripper_cmd: {value}")
    elif args.command == "left-full":
        arm = _float_list(args.arm)
        gripper = float(args.gripper)
        node.send_left_full(arm, gripper)
        print(f"Published left full command: arm={arm}, gripper={gripper}")
    elif args.command == "right-arm":
        values = _float_list(args.values)
        node.send_right_arm(values)
        print(f"Published /isaacsim/right_arm_cmd: {values}")
    elif args.command == "right-hand":
        values = _float_list(args.values)
        node.send_right_hand(values)
        print(f"Published /isaacsim/right_hand_cmd: {values}")
    elif args.command == "right-full":
        arm = _float_list(args.arm)
        hand = _float_list(args.hand)
        node.send_right_full(arm, hand)
        print(f"Published right full command: arm={arm}, hand={hand}")

    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
