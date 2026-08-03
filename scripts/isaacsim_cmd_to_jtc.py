#!/usr/bin/env python3
"""브리지: /isaacsim/right_{arm,hand}_cmd (canonical) → robot_control JTC.

정책 노드(grasp_inference/pour_inference)가 발행하는 canonical 관절 명령을 robot_control
컨트롤러가 받는 source 관절 순서·부호·한계로 변환해 단일포인트 JointTrajectory 로 스트리밍한다.
robot_control 코드 무수정(Option B). robot PC(controllers 와 co-locate) 에서 실행 권장.

실행 (robot PC):
    python3 isaacsim_cmd_to_jtc.py \
        [--profile ~/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml] \
        [--arm-topic /right_joint_trajectory_controller/joint_trajectory] \
        [--hand-topic /dg5f_right/dg5f_right_controller/joint_trajectory] \
        [--control-dt 0.0167] [--max-vel 0.1]

구독: /isaacsim/right_arm_cmd (Float64MultiArray, 7 canonical r_aj_1..7)
      /isaacsim/right_hand_cmd (Float64MultiArray, 20 canonical r_hj_* finger-major)
발행: <arm-topic>, <hand-topic> (trajectory_msgs/JointTrajectory, 단일포인트)

★ 컨트롤러 interpolation_method="none"(openarm_bimanual_controllers.yaml, 고주파 서보) 전제:
  time_from_start=0 으로 목표를 **즉시 적용**하고, 속도제한은 위치 클램프
  (velocity_limited_target, 실제 위치 ±max_vel·dt)로 한다. 미래 시각 tfs 를 주면 스트림에서
  포인트가 영영 적용 안 돼 로봇이 전혀 안 움직인다. [[jtc-none-interpolation-silent-stall]]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from jtc_bridge_core import (
    JointRemap,
    load_profile_joints,
    velocity_limited_target,
)

ARM_CANON = [f"r_aj_{i}" for i in range(1, 8)]
ARM_SOURCE = [f"openarm_right_joint{i}" for i in range(1, 8)]
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HAND_CANON = [f"r_hj_{f}_{j}" for f in _FINGERS for j in range(1, 5)]
HAND_SOURCE = [f"rj_dg_{fi}_{j}" for fi in range(1, 6) for j in range(1, 5)]

DEFAULT_PROFILE = str(
    Path.home() / "rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml"
)


def _duration(sec_float: float) -> Duration:
    sec = int(sec_float)
    nanosec = int(round((sec_float - sec) * 1e9))
    return Duration(sec=sec, nanosec=nanosec)


class IsaacsimCmdToJtc(Node):
    def __init__(
        self,
        profile_path: str,
        arm_topic: str,
        hand_topic: str,
        control_dt: float,
        max_vel: float,
        arm_state_topic: str,
        hand_state_topic: str,
    ) -> None:
        super().__init__("isaacsim_cmd_to_jtc")
        prof = load_profile_joints(profile_path)
        self.arm_remap = JointRemap(ARM_CANON, ARM_SOURCE, prof)
        self.hand_remap = JointRemap(HAND_CANON, HAND_SOURCE, prof)
        self.control_dt = float(control_dt)
        self.max_vel = float(max_vel)

        # 실제 관절 위치(source명 → pos). 세트포인트 첫 초기화(점프 방지)에만 사용.
        self.actual: dict[str, float] = {}
        # 직전 세트포인트(label→source순 위치). rate-limit 을 이 값 기준 전진(fabric_q 추종).
        self._last_setpoint: dict[str, np.ndarray] = {}
        # ★최신 목표(label→source순). 콜백은 저장만 하고, 전진·발행은 고정 주기 타이머가 한다.
        #   (수신 시점에 전진하면 명령 rate 가 낮을 때 — APPROACHING 10Hz — 실효 속도가
        #    max_vel×(cmd_rate×dt) 로 줄어 팔이 목표에 영영 못 도달. 08.03 실기 정체 근본원인)
        self._target: dict[str, np.ndarray] = {}

        self.arm_pub = self.create_publisher(JointTrajectory, arm_topic, 10)
        self.hand_pub = self.create_publisher(JointTrajectory, hand_topic, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_arm_cmd", self._arm_cb, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_hand_cmd", self._hand_cb, 10)
        self.create_subscription(JointState, arm_state_topic, self._state_cb, 20)
        self.create_subscription(JointState, hand_state_topic, self._state_cb, 20)
        self.create_timer(self.control_dt, self._tick)

        self.get_logger().info(
            f"브리지 준비: arm→{arm_topic}, hand→{hand_topic}\n"
            f"  fabric_q 직접 추종 (세트포인트 rate-limit max_vel={self.max_vel} rad/s, time_from_start=0)\n"
            f"  제어주기 dt={self.control_dt*1000:.1f}ms · interpolation=none 전제 · 실제위치 클램프 없음\n"
            f"  상태구독: {arm_state_topic}, {hand_state_topic}\n"
            f"  profile={profile_path}"
        )

    def _state_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            self.actual[name] = msg.position[i]

    def _store_target(self, remap: JointRemap, values, n: int, label: str) -> None:
        if len(values) != n:
            self.get_logger().warn(f"{label} cmd 길이 {len(values)} != {n}, 무시")
            return
        self._target[label] = remap.apply(list(values))

    def _arm_cb(self, msg: Float64MultiArray) -> None:
        self._store_target(self.arm_remap, msg.data, 7, "arm")

    def _hand_cb(self, msg: Float64MultiArray) -> None:
        self._store_target(self.hand_remap, msg.data, 20, "hand")

    def _tick(self) -> None:
        """고정 주기(control_dt) 세트포인트 전진+발행 — 명령 rate 와 무관하게 max_vel 보장."""
        for label, pub, remap in (
            ("arm", self.arm_pub, self.arm_remap),
            ("hand", self.hand_pub, self.hand_remap),
        ):
            target = self._target.get(label)
            if target is None:
                continue   # 첫 명령 수신 전엔 발행하지 않음
            last = self._last_setpoint.get(label)
            if last is None or last.shape != target.shape:
                # 첫 명령: 세트포인트를 실제 위치에서 시작(점프 방지).
                cur = [self.actual.get(src) for src in remap.output_source]
                if any(c is None for c in cur):
                    # 실제 위치 미수신(예: 손 상태 로컬 DDS 수신 실패) → target 에서 초기화해
                    # 명령이 나가게 한다. 손은 첫 target 이 APPROACH(열림)라 점프가 작아 안전.
                    self.get_logger().warn(
                        f"{label}: 실제 위치 미수신 → target 에서 세트포인트 초기화(상태없이 진행)",
                        throttle_duration_sec=5.0,
                    )
                    last = target
                else:
                    last = np.array(cur, dtype=np.float64)
            # 이후는 실제 위치와 무관하게 fabric_q(target) 를 rate-limit 로 추종 → 홀딩 토크 정상.
            positions = velocity_limited_target(target, last, self.max_vel, self.control_dt)
            self._last_setpoint[label] = positions
            jt = JointTrajectory()
            jt.joint_names = list(remap.output_source)
            pt = JointTrajectoryPoint()
            pt.positions = positions.tolist()
            pt.time_from_start = _duration(0.0)   # none 보간 → 즉시 적용(미래 시각이면 무동작)
            jt.points = [pt]
            pub.publish(jt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--arm-topic", default="/right_joint_trajectory_controller/joint_trajectory")
    parser.add_argument("--hand-topic", default="/dg5f_right/dg5f_right_controller/joint_trajectory")
    parser.add_argument("--control-dt", type=float, default=1.0 / 60.0,
                        help="제어 주기[s]. 위치클램프 속도제한 step = max_vel·control_dt")
    parser.add_argument("--max-vel", type=float, default=0.5,
                        help="세트포인트 rate-limit [rad/s]. fabric_q 를 이 속도로 추종(큰 점프만 부드럽게)")
    parser.add_argument("--arm-state-topic", default="/joint_states")
    parser.add_argument("--hand-state-topic", default="/dg5f_right/joint_states")
    args = parser.parse_args()

    rclpy.init()
    node = IsaacsimCmdToJtc(
        profile_path=args.profile,
        arm_topic=args.arm_topic,
        hand_topic=args.hand_topic,
        control_dt=args.control_dt,
        max_vel=args.max_vel,
        arm_state_topic=args.arm_state_topic,
        hand_state_topic=args.hand_state_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
