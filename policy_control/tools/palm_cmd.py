#!/usr/bin/env python3
"""palm_cmd — control-only fabric 노드에 palm 목표를 보낸다 (/policy_control/palm_cmd, PoseStamped).

    python3 policy_control/tools/palm_cmd.py --rel 0.05 0 0                    # 현재 palm 에서 +5 cm x
    python3 policy_control/tools/palm_cmd.py --rel 0 0 0 0 0 0.1               # roll pitch yaw 델타 [rad]
    python3 policy_control/tools/palm_cmd.py --abs 0.30 -0.30 0.40 1.571 0 1.571   # x y z roll pitch yaw
    python3 policy_control/tools/palm_cmd.py --rel 0 0 0.03 --hold 2.0         # 2 s 동안 10 Hz 재발행
    python3 policy_control/tools/palm_cmd.py --rel 0.02 0 0 --dry-run          # 계산만, 발행 없음

규약
  · 현재 자세는 /policy_control/palm_pose(fabric 현재 palm FK, latched) 를 **한 번** 읽는다 — 상대 이동의 기준.
    fabric 노드가 떠 있지 않으면(타임아웃) 아무것도 발행하지 않는다.
  · 각도 규약은 fabric 의 euler_zyx = (yaw, pitch, roll). CLI 는 사람 순서 roll pitch yaw 로 받는다(--deg 면 도).
    --rel 의 회전 델타는 base 프레임에서 합성한다: R_target = Rz(dyaw)·Ry(dpitch)·Rx(droll) · R_current.
  · ROS_DOMAIN_ID 0/unset(실기 기본 도메인)은 거부한다 — --allow-domain-0 로만 통과한다.
  · control-only 계약에는 박스·속도 제한이 없다. 작은 델타로 움직여라.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # sim2real/policy_control → `import policy_control`

from policy_control.decoder_core import euler_zyx_from_quat, euler_zyx_from_rot, rot_euler_zyx  # noqa: E402

NS = "/policy_control"
TOPIC_PALM_POSE = f"{NS}/palm_pose"
TOPIC_PALM_CMD = f"{NS}/palm_cmd"
DEFAULT_FRAME = "base_link"
DEFAULT_TIMEOUT = 3.0
HOLD_HZ = 10.0
POS_DIM = 3
POSE_DIM = 6


def check_domain(env: dict, allow_domain_0: bool) -> str | None:
    """실기 기본 도메인(0/unset) 거부 사유 — 통과하면 None."""
    domain = str(env.get("ROS_DOMAIN_ID", "")).strip()
    if domain in ("", "0") and not allow_domain_0:
        return f"ROS_DOMAIN_ID={domain or 'unset'} 은 실기 기본 도메인이다 — --allow-domain-0 없이는 거부"
    return None


def rpy_to_euler_zyx(rpy, deg: bool = False) -> np.ndarray:
    """(roll, pitch, yaw) → fabric euler_zyx (yaw, pitch, roll); --deg 면 도 → rad."""
    r, p, y = (float(v) for v in np.asarray(rpy, dtype=float).reshape(3))
    scale = np.pi / 180.0 if deg else 1.0
    return np.array([y, p, r]) * scale


def euler_zyx_to_rpy(e, deg: bool = False) -> np.ndarray:
    ez, ey, ex = (float(v) for v in np.asarray(e, dtype=float).reshape(3))
    scale = 180.0 / np.pi if deg else 1.0
    return np.array([ex, ey, ez]) * scale


def compose_target(current6, rel=None, abs_=None, deg: bool = False) -> np.ndarray:
    """현재 palm(pos3 + euler_zyx3) + --rel(dx dy dz [droll dpitch dyaw]) 또는 --abs(x y z roll pitch yaw) → 목표 6D."""
    if (rel is None) == (abs_ is None):
        raise ValueError("--rel 과 --abs 중 정확히 하나")
    if abs_ is not None:
        a = np.asarray(abs_, dtype=float).reshape(-1)
        if a.size != POSE_DIM:
            raise ValueError(f"--abs 는 x y z roll pitch yaw 6개 ({a.size}개 받음)")
        return np.concatenate([a[:POS_DIM], rpy_to_euler_zyx(a[POS_DIM:], deg)])
    cur = np.asarray(current6, dtype=float).reshape(POSE_DIM)
    d = np.asarray(rel, dtype=float).reshape(-1)
    if d.size not in (POS_DIM, POSE_DIM):
        raise ValueError(f"--rel 은 dx dy dz [droll dpitch dyaw] 3개 또는 6개 ({d.size}개 받음)")
    pos = cur[:POS_DIM] + d[:POS_DIM]
    if d.size == POS_DIM:
        return np.concatenate([pos, cur[POS_DIM:]])
    R = rot_euler_zyx(rpy_to_euler_zyx(d[POS_DIM:], deg)) @ rot_euler_zyx(cur[POS_DIM:])
    return np.concatenate([pos, euler_zyx_from_rot(R)])


def quat_from_euler_zyx(e) -> np.ndarray:
    """euler_zyx → 쿼터니언 wxyz (grasp_s2r_core._quat_from_matrix 와 같은 규약)."""
    from grasp_s2r_core import _quat_from_matrix

    return _quat_from_matrix(rot_euler_zyx(e))


def fmt(palm6, deg: bool) -> str:
    p = np.asarray(palm6, dtype=float)
    rpy = euler_zyx_to_rpy(p[POS_DIM:], deg)
    unit = "deg" if deg else "rad"
    return (f"xyz ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m · rpy ({rpy[0]:+.3f}, {rpy[1]:+.3f}, {rpy[2]:+.3f}) {unit}")


# ---------------------------------------------------------------- ROS
class Runner:
    """palm_pose 한 번 읽기 + palm_cmd 발행. rclpy 는 여기서만 import 된다."""

    def __init__(self, frame: str) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("palm_cmd")
        self.frame = frame
        self.current: np.ndarray | None = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(PoseStamped, TOPIC_PALM_POSE, self._on_pose, latched)
        self.pub = self.node.create_publisher(PoseStamped, TOPIC_PALM_CMD, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def _on_pose(self, msg) -> None:
        from policy_control import codec

        pose = codec.decode_pose(msg)
        if pose.frame and pose.frame != self.frame:
            print(f"    ✗ palm_pose frame {pose.frame!r} ≠ {self.frame!r} — 무시", file=sys.stderr)
            return
        self.current = np.concatenate([pose.pos, euler_zyx_from_quat(pose.quat)])

    def read_pose(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        while self.current is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.current

    def wait_subscriber(self, timeout: float = 3.0) -> int:
        """fabric 노드가 이 토픽을 구독할 때까지 기다린다 — RELIABLE 이어도 매칭 전 발행은 버려진다
        (fake_direct_right_run2: 두 번째 hand_cmd 가 조용히 사라졌다)."""
        deadline = time.monotonic() + timeout
        while self.pub.get_subscription_count() < 1 and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.pub.get_subscription_count()

    def publish(self, target6: np.ndarray, hold: float) -> int:
        from policy_control import codec

        n, deadline = 0, time.monotonic() + max(hold, 0.0)
        while True:
            self.pub.publish(codec.encode_pose(target6[:POS_DIM], quat_from_euler_zyx(target6[POS_DIM:]), self.frame,
                                               stamp=time.time()))
            n += 1
            if time.monotonic() >= deadline:
                return n
            self.rclpy.spin_once(self.node, timeout_sec=1.0 / HOLD_HZ)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def _parse(argv) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rel", type=float, nargs="+", metavar="D", help="dx dy dz [droll dpitch dyaw] (현재 palm 기준)")
    g.add_argument("--abs", type=float, nargs=6, metavar="V", help="x y z roll pitch yaw (절대)")
    ap.add_argument("--deg", action="store_true", help="각도를 도로 해석")
    ap.add_argument("--hold", type=float, default=0.0, help=f"이 시간(s) 동안 {HOLD_HZ:.0f} Hz 로 재발행 (기본 1회)")
    ap.add_argument("--frame", default=DEFAULT_FRAME)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="palm_pose 대기 시간(s)")
    ap.add_argument("--allow-domain-0", action="store_true", help="★실기 기본 도메인에서도 발행한다")
    ap.add_argument("--dry-run", action="store_true", help="목표만 계산·출력하고 발행하지 않는다")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    refuse = check_domain(dict(os.environ), args.allow_domain_0)
    if refuse:
        print(f"✗ {refuse}", file=sys.stderr)
        return 3
    runner = Runner(args.frame)
    try:
        current = runner.read_pose(args.timeout)
        if current is None:
            print(f"✗ {TOPIC_PALM_POSE} 를 {args.timeout:.1f}s 안에 못 받았다 — fabric_node(control-only) 가 떠 있나?",
                  file=sys.stderr)
            return 2
        target = compose_target(current, rel=args.rel, abs_=args.abs, deg=args.deg)
        print(f"current  {fmt(current, args.deg)}")
        print(f"target   {fmt(target, args.deg)}")
        if args.dry_run:
            print("DRY RUN — 발행하지 않았다")
            return 0
        if runner.wait_subscriber() < 1:
            print(f"✗ {TOPIC_PALM_CMD} 구독자가 없다 — fabric_node(control-only, direct 모드) 가 떠 있나?", file=sys.stderr)
            return 2
        n = runner.publish(target, args.hold)
        print(f"published {TOPIC_PALM_CMD} ×{n} (frame {args.frame})")
        return 0
    except ValueError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
