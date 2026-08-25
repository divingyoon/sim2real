#!/usr/bin/env python3
"""sim 이 기록한 관절 목표를 실기로 다시 흘려보낸다 (ROS2 노드).

**이 노드는 정책을 돌리지 않는다.** 기록을 재생할 뿐이다. 그래서 실기가 이상하면 그냥
멈추면 되고, 관측이 어긋나 정책이 발산할 여지가 없다. 카메라도 물체도 없는 첫 실기
투입에 맞는 형태다.

계획 수립·검증은 `shadow_replay_core.py` 가 한다(로봇 없이 테스트된다). 여기는 발행과
기록만 담당한다.

지키는 것
  · `--execute` 없으면 **아무것도 발행하지 않는다**. robotctl 과 같은 규약이다.
  · 첫 프레임으로 도약하지 않는다 — 실측 자세에서 0.1 rad/s 로 램프해 들어간다.
  · 고정 주기 타이머로 발행한다. 수신 시점에 전진시키면 명령 rate 가 낮을 때 실효속도가
    줄어 목표에 영영 못 간다(08.03 정체의 근본원인).
  · 중단 조건을 넘으면 **즉시 멈춘다**(추종오차·effort·상태 두절).
  · 매 발행마다 `t_send`·`step_idx` 를 csv 로 남긴다 — `shadow_report.py` 의 정렬 키다.

시각 규약: `time_from_start=0`. 컨트롤러가 `interpolation_method="none"` 이라 미래 시각
포인트는 스트림에서 영영 적용되지 않고 로봇이 **무경고로 안 움직인다**.
[[jtc-none-interpolation-silent-stall]]

실행 (로봇 PC):
    python3 shadow_replay.py --sim logs/shadow/sim_fab_test16_gcON.npz \\
        --robot gripper_left --rate-scale 0.25 --log logs/shadow/real_x025.csv
    # 위는 dry-run. 실제로 내보내려면 --execute 를 붙인다.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from jtc_bridge_core import JointRemap, velocity_limited_target  # noqa: E402
from robot_profile import load_robot_profile                     # noqa: E402
from shadow_replay_core import ReplayPlan                        # noqa: E402

#: 중단 조건. 넘으면 즉시 멈춘다 — 재생은 언제든 다시 하면 되고 팔은 하나뿐이다.
ABORT_TRACKING_ERR_RAD = 0.30
ABORT_EFFORT_NM = {"l_aj_5": 5.0, "l_aj_6": 5.0, "l_aj_7": 5.0}
ABORT_STATE_STALE_SEC = 1.0


def build_plan(sim_npz: Path, rate_scale: float, profile) -> ReplayPlan:
    data = np.load(sim_npz, allow_pickle=False)
    joint_names = [str(x) for x in data["meta_joint_names"]]
    if tuple(joint_names) != tuple(profile.arm_canonical):
        raise SystemExit(
            f"기록의 팔 관절 순서가 프로필과 다르다\n  기록 {joint_names}\n"
            f"  프로필 {list(profile.arm_canonical)}"
        )
    grip_names = [str(x) for x in data["meta_grip_names"]]
    if profile.ee_canonical[0] not in grip_names:
        raise SystemExit(
            f"프로필 그리퍼 {profile.ee_canonical[0]} 이 기록 {grip_names} 에 없다"
        )
    # sim 은 USD 가 mimic 을 잃어 두 조를 다 지령하지만 실기는 한 조면 따라온다.
    jaw = grip_names.index(profile.ee_canonical[0])
    return ReplayPlan(
        arm_target=data["arm_target"][:, 0],
        grip_target=data["grip_cmd"][:, 0, jaw],
        step_dt=float(data["meta_step_dt"][0]),
        rate_scale=rate_scale,
        joint_names=joint_names,
        gripper_name=profile.ee_canonical[0],
    )


def describe(plan: ReplayPlan, profile) -> str:
    """재생 **전에** 알아야 할 것: 얼마나 걸리고, 실기 한계 대비 무엇을 요구하는가."""
    lines = [
        f"프레임 {plan.n_frames}  ·  발행주기 {plan.publish_dt*1000:.2f} ms "
        f"({1/plan.publish_dt:.1f} Hz)  ·  총 {plan.schedule[-1]:.1f} s",
    ]
    limits = [profile.joint_limits[j].get("velocity") for j in plan.joint_names]
    if any(v is None for v in limits):
        unknown = [j for j, v in zip(plan.joint_names, limits) if v is None]
        lines.append(f"요구 최대 관절속도 {plan.peak_joint_speed:.3f} rad/s  ·  "
                     f"⚠ 한계를 모른다(프로필에 velocity 없음): {unknown}")
        return "\n".join(lines)
    limit = min(limits)
    lines.append(
        f"요구 최대 관절속도 {plan.peak_joint_speed:.3f} rad/s  vs  프로필 한계 {limit:.3f}")
    if plan.peak_joint_speed > limit:
        lines.append(f"  ⚠ 요구가 한계를 넘는다 — --rate-scale 을 "
                     f"{limit/plan.peak_joint_speed:.2f} 이하로 낮출 것")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="probe_fab_shadow_record 의 npz")
    parser.add_argument("--robot", default="gripper_left", help="config/robots 의 구성")
    parser.add_argument("--rate-scale", type=float, default=0.25,
                        help="(0,1]. 낮출수록 느리게 재생한다. 처음에는 0.25 로 시작할 것.")
    parser.add_argument("--max-vel", type=float, default=0.5,
                        help="세트포인트 전진 상한[rad/s]. 재생 요구와 별개인 안전 캡.")
    parser.add_argument("--frames", type=int, default=0, help=">0 이면 앞에서 그만큼만")
    parser.add_argument("--log", type=Path, default=None, help="발행·측정 csv")
    parser.add_argument("--execute", action="store_true",
                        help="이것 없으면 계획만 출력하고 아무것도 발행하지 않는다")
    args = parser.parse_args()

    profile = load_robot_profile(args.robot)
    plan = build_plan(args.sim, args.rate_scale, profile)
    if args.frames > 0:
        plan = ReplayPlan(
            arm_target=plan.arm_target[: args.frames],
            grip_target=plan.grip_target[: args.frames],
            step_dt=plan.step_dt, rate_scale=plan.rate_scale,
            joint_names=plan.joint_names, gripper_name=plan.gripper_name,
        )

    print(f"구성 {profile.name}  ·  기록 {args.sim.name}")
    print(describe(plan, profile))
    if not args.execute:
        print("\nDRY RUN: 아무것도 발행하지 않았다. 실제로 내보내려면 --execute 를 붙일 것.")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class ShadowReplay(Node):
        def __init__(self) -> None:
            super().__init__("shadow_replay")
            self.plan = plan
            self.profile = profile
            self.arm_remap = JointRemap(
                list(plan.joint_names), list(profile.arm_source), profile.joint_limits)
            self.grip_remap = JointRemap(
                [plan.gripper_name], list(profile.ee_source), profile.joint_limits)
            self.arm_pub = self.create_publisher(
                Float64MultiArray, profile.topics["arm_cmd"], 10)
            self.grip_pub = self.create_publisher(Float64, profile.topics["ee_cmd"], 10)
            self.arm_traj = self.create_publisher(
                JointTrajectory, profile.topics["arm_traj"], 10)
            self.grip_traj = self.create_publisher(
                JointTrajectory, profile.topics["ee_traj"], 10)
            self.create_subscription(JointState, profile.topics["arm_state"],
                                     self._state_cb, qos_profile_sensor_data)

            self.measured = np.zeros(len(plan.joint_names))
            self.effort = np.zeros(len(plan.joint_names))
            self.last_state = 0.0
            self._src_index: dict[str, int] = {}
            self.setpoint: np.ndarray | None = None
            self.ramp: np.ndarray | None = None
            self.cursor = 0
            self.rows: list[dict] = []
            self.aborted: str | None = None
            self.timer = None

        def _state_cb(self, msg: JointState) -> None:
            if not self._src_index:
                self._src_index = {n: i for i, n in enumerate(msg.name)}
            for k, src in enumerate(self.profile.arm_source):
                i = self._src_index.get(src)
                if i is None:
                    continue
                sign = self.profile.joint_limits[self.plan.joint_names[k]]["sign"]
                self.measured[k] = msg.position[i] * sign
                if msg.effort and i < len(msg.effort):
                    self.effort[k] = msg.effort[i]
            self.last_state = time.monotonic()

        def start(self) -> None:
            deadline = time.monotonic() + 5.0
            while self.last_state == 0.0 and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_state == 0.0:
                raise SystemExit(
                    f"{self.profile.topics['arm_state']} 를 못 받았다 — bringup 확인"
                )
            self.ramp = self.plan.ramp_from(self.measured.copy())
            self.setpoint = self.measured.copy()
            self.get_logger().info(
                f"실측에서 첫 프레임까지 램프 {len(self.ramp)} 프레임 "
                f"({len(self.ramp)*self.plan.publish_dt:.1f} s)")
            self.timer = self.create_timer(self.plan.publish_dt, self._tick)

        def _abort(self, why: str) -> None:
            self.aborted = why
            self.get_logger().error(f"중단: {why}")
            if self.timer is not None:
                self.timer.cancel()

        def _tick(self) -> None:
            now = time.monotonic()
            if now - self.last_state > ABORT_STATE_STALE_SEC:
                return self._abort(f"상태 두절 {now - self.last_state:.2f} s")

            in_ramp = self.cursor < len(self.ramp)
            if in_ramp:
                target = self.ramp[self.cursor]
                grip = float(self.plan.grip_target[0])
                step_idx = -1
            else:
                frame = self.cursor - len(self.ramp)
                if frame >= self.plan.n_frames:
                    self.get_logger().info("재생 완료")
                    return self._abort("완료")
                target = self.plan.arm_target[frame]
                grip = float(self.plan.grip_target[frame])
                step_idx = frame

            err = np.abs(self.measured - target)
            if err.max() > ABORT_TRACKING_ERR_RAD:
                worst = self.plan.joint_names[int(np.argmax(err))]
                return self._abort(f"{worst} 추종오차 {err.max():.3f} rad")
            for k, name in enumerate(self.plan.joint_names):
                cap = ABORT_EFFORT_NM.get(name)
                if cap is not None and abs(self.effort[k]) > cap:
                    return self._abort(f"{name} effort {self.effort[k]:.2f} N·m")

            self.setpoint = velocity_limited_target(
                target, self.setpoint, args.max_vel, self.plan.publish_dt)
            self._publish(self.setpoint, grip)
            self.rows.append({
                "t_send": now, "step_idx": step_idx,
                **{f"cmd_{n}": float(self.setpoint[k])
                   for k, n in enumerate(self.plan.joint_names)},
                **{f"meas_{n}": float(self.measured[k])
                   for k, n in enumerate(self.plan.joint_names)},
                **{f"eff_{n}": float(self.effort[k])
                   for k, n in enumerate(self.plan.joint_names)},
                "cmd_gripper": grip,
            })
            self.cursor += 1

        def _publish(self, arm: np.ndarray, grip: float) -> None:
            self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in arm]))
            self.grip_pub.publish(Float64(data=grip))
            self.arm_traj.publish(self._traj(
                list(self.profile.arm_source), self.arm_remap.apply(arm)))
            self.grip_traj.publish(self._traj(
                list(self.profile.ee_source), self.grip_remap.apply(np.array([grip]))))

        @staticmethod
        def _traj(names: list[str], positions: np.ndarray) -> "JointTrajectory":
            msg = JointTrajectory()
            msg.joint_names = names
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in positions]
            # ★0 이어야 한다. 미래 시각이면 포인트가 영영 적용되지 않는다.
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 0
            msg.points = [point]
            return msg

    rclpy.init()
    node = ShadowReplay()
    try:
        node.start()
        while node.aborted is None:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.aborted = "사용자 중단"
    finally:
        if args.log and node.rows:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(node.rows[0]))
                writer.writeheader()
                writer.writerows(node.rows)
            print(f"-> {args.log}  ({len(node.rows)} 행)")
        node.destroy_node()
        rclpy.shutdown()
    print(f"종료: {node.aborted}")
    return 0 if node.aborted == "완료" else 1


if __name__ == "__main__":
    raise SystemExit(main())
