#!/usr/bin/env python3

"""ROS 2 relay that republishes commands written by file_command_transport.py."""

from __future__ import annotations

import argparse
import json
import os
import time

import sys
from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_command_transport import DEFAULT_COMMAND_FILE
from fabrics_ros_interface import create_publisher


# Default control pose is the grasp-ready initialization used by the
# SkillBlender/OpenArm manipulation stack, not the raw all-zero joint pose.
DEFAULT_LEFT_ARM = [-0.5, -0.5, 0.6, 0.7, 0.0, 0.0, -1.0]
DEFAULT_LEFT_GRIPPER = 0.015
DEFAULT_RIGHT_ARM = [0.5, 0.5, -0.6, 0.7, 0.0, 0.0, 1.0]
DEFAULT_RIGHT_HAND = [
    0.0, -1.571, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
]


def _float_list(values: list[float] | None, expected_len: int, name: str) -> list[float] | None:
    if values is None:
        return None
    if len(values) != expected_len:
        raise ValueError(f"{name} must have exactly {expected_len} values")
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Relay file-based sim2real commands into ROS 2 topics")
    parser.add_argument("--command-file", default=DEFAULT_COMMAND_FILE)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--left-arm", type=float, nargs=7, help="Override default left arm pose")
    parser.add_argument("--left-gripper", type=float, help="Override default left gripper opening")
    parser.add_argument("--right-arm", type=float, nargs=7, help="Override default right arm pose")
    parser.add_argument("--right-hand", type=float, nargs=20, help="Override default right hand pose")
    args = parser.parse_args()

    pub = create_publisher()
    last_seq = None
    period = 1.0 / args.hz
    current_left_arm = _float_list(args.left_arm, 7, "left-arm") or list(DEFAULT_LEFT_ARM)
    current_left_gripper = float(args.left_gripper) if args.left_gripper is not None else float(DEFAULT_LEFT_GRIPPER)
    current_right_arm = _float_list(args.right_arm, 7, "right-arm") or list(DEFAULT_RIGHT_ARM)
    current_right_hand = _float_list(args.right_hand, 20, "right-hand") or list(DEFAULT_RIGHT_HAND)

    print(f"Watching command file: {args.command_file}")
    print(f"Default left arm: {current_left_arm}")
    print(f"Default left gripper: {current_left_gripper}")
    print(f"Default right arm: {current_right_arm}")
    print(f"Default right hand: {current_right_hand}")
    try:
        while True:
            if os.path.exists(args.command_file):
                try:
                    with open(args.command_file, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except (json.JSONDecodeError, OSError):
                    payload = None

                if payload is not None:
                    seq = payload.get("seq")
                    if seq != last_seq:
                        if "left_arm" in payload:
                            current_left_arm = list(payload["left_arm"])
                        if "left_gripper" in payload:
                            current_left_gripper = float(payload["left_gripper"])
                        if "right_arm" in payload:
                            current_right_arm = list(payload["right_arm"])
                        if "right_hand" in payload:
                            current_right_hand = list(payload["right_hand"])
                        last_seq = seq

            # Continuously republish the latest command so the sim shadow robot
            # stays pinned at a known pose even before higher-level control starts.
            pub.send_left_full(current_left_arm, current_left_gripper)
            pub.send_right_full(current_right_arm, current_right_hand)
            pub.spin_once()
            time.sleep(period)
    finally:
        pub.close()


if __name__ == "__main__":
    main()
