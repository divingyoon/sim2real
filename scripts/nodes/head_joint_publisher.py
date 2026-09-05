#!/usr/bin/env python3
"""목 관절 각도를 ROS 로 발행한다 — `cup_pose_relay` 가 카메라 자세를 매번 계산하도록.

`global_camera_extrinsics.yaml` 의 `camera:` 블록은 **한 목 자세의 정적 스냅샷**이라
목이 돌면 통째로 틀린다(pan 15° 면 카메라가 11 mm 이동 + 큰 회전 → 0.6 m 컵에서 수 cm).
이 노드가 목 각도를 흘려 주면 릴레이가

    T_base_cam(pan, tilt) = T_base_neck(pan, tilt) ∘ T_neck_cam

로 매번 다시 만든다.

★**URDF 규약(라디안)으로 발행한다.** 토픽이 `head_j_pan`/`head_j_tilt` 라는 URDF 관절
이름을 쓰므로 그게 표준 의미다. 인코더 각을 그대로 실으면 pan 이 반대로 흐른다.
변환은 `head_fk_chain.urdf_from_encoder` 가 한다.

★**읽기 전용**이다. 토크·게인·목표를 건드리지 않는다 — 다른 도구가 목을 잡고 있어도
안전하게 같이 돌릴 수 있다.

    python head_joint_publisher.py                       # 20 Hz 로 /head/joint_states
    python head_joint_publisher.py --dry-run             # 하드웨어 없이 계획만
"""

from __future__ import annotations

import argparse
import math
import sys

from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_compliant_hold import (
    DEFAULT_BAUD,
    CompliantController,
    autodetect_port,
    discover_ports,
    parse_motor_ids,
    tick_to_deg,
)
from head_fk_chain import urdf_from_encoder

JOINT_NAMES = ["head_j_pan", "head_j_tilt"]
DEFAULT_TOPIC = "/head/joint_states"
DEFAULT_RATE_HZ = 20.0


def joint_state_fields(pan_encoder_deg: float, tilt_encoder_deg: float
                       ) -> tuple[list[str], list[float]]:
    """(이름, 위치[rad]) — **URDF 규약**으로 변환해 실는다."""
    pan_urdf, tilt_urdf = urdf_from_encoder(pan_encoder_deg, tilt_encoder_deg)
    return list(JOINT_NAMES), [math.radians(pan_urdf), math.radians(tilt_urdf)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ids", default="1,2", help="pan,tilt 순서")
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        ids = parse_motor_ids(args.ids)
        if len(ids) != 2:
            raise ValueError(f"pan,tilt 두 개여야 한다: {ids}")
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    print(f"포트 {port} @ {args.baud} · ids {ids} → {args.topic} @ {args.rate_hz} Hz")
    print(f"관절 이름 {JOINT_NAMES} · 단위 라디안 · **URDF 규약**(pan 부호 반전 적용)")
    print("읽기 전용 — 토크·게인·목표를 건드리지 않는다")
    if args.dry_run:
        names, pos = joint_state_fields(0.0, -20.0)
        print(f"\n예시(인코더 pan 0 / tilt -20): {names} = "
              f"{[round(v, 6) for v in pos]} rad")
        print("--dry-run — 포트를 열지 않았다")
        return 0

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    controller = CompliantController(port, args.baud)
    rclpy.init()
    node = Node("head_joint_publisher")
    pub = node.create_publisher(JointState, args.topic, 10)
    count = {"n": 0}

    def tick() -> None:
        try:
            pan = tick_to_deg(controller.read_present_tick(ids[0]))
            tilt = tick_to_deg(controller.read_present_tick(ids[1]))
        except Exception as exc:                    # 한 번 실패해도 노드를 죽이지 않는다
            node.get_logger().warning(f"목 각도 읽기 실패: {exc}")
            return
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = joint_state_fields(pan, tilt)
        pub.publish(msg)
        count["n"] += 1
        if count["n"] % (round(args.rate_hz) * 10) == 1:
            node.get_logger().info(
                f"#{count['n']} 인코더 pan {pan:+.2f}° tilt {tilt:+.2f}°")

    node.create_timer(1.0 / args.rate_hz, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        controller.close()
        print("포트 닫음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
