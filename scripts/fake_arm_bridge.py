#!/usr/bin/env python3
"""컨트롤러 자리에 서는 루프백 리그 — 실기 없이 **ROS 경로 전체**를 돌린다.

무엇을 대신하나. 로봇 PC 에서는 `openarm_bringup` 이 JointTrajectory 를 받아 컨트롤러로
넘기고 `joint_state_broadcaster` 가 `/joint_states` 를 낸다. 이 노드는 그 두 역할만 흉내
낸다 — 궤적을 받아 팔 모델로 적분하고 `/joint_states` 를 낸다.

**왜 필요한가.** 새로 쓴 것은 재생기·브리지·리포트이고, 벤더 스택은 이미 검증된 코드다.
그런데 새 코드의 ROS 부분(발행·구독·리맵·타이머·게이트·csv)은 로봇 없이는 한 번도
돌지 않는다. 여기서 돌려 두면 실기에서 남는 미지수는 **로봇 자체**뿐이다.

⚠ `effort` 는 0 을 싣는다. 모델이 토크를 모르기 때문이다 — 재생기의 effort 중단 조건은
  이 리그에서 **한 번도 발화하지 않는다**. 그 게이트는 실기에서만 유효하다.

⚠ **이것은 추종 검증이 아니다.** `--model rate` 는 속도제한만 있는 완벽 추종이라
`use_fake_hardware:=true` 와 같은 한계를 갖는다 — droop 이 없어 서보 버그를 숨긴다.
`--model pd` 는 실측 게인 PD 로 지연·오버슛을 **예측**하지만 그것도 모델이지 로봇이 아니다.
숫자를 실측으로 인용하지 말 것.

실행:
    source /opt/ros/humble/setup.bash && . .venv/bin/activate
    python3 scripts/fake_arm_bridge.py --robot gripper_left --model pd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm_inertia import effective_inertia          # noqa: E402
from arm_pd_model import MockArm, load_arm_pd      # noqa: E402
from robot_profile import idle_arm_rest_pose, load_robot_profile  # noqa: E402

#: 실측 캘리브레이션. `right_arm_best_calibration.json` 은 **우팔** 값이다 —
#  좌팔에는 식별 캘리브가 없다(그 파일의 openarm_left_arm 400/80 은 sim 기본값이 남은 것).
#  같은 팔 하드웨어라 예측용으로 쓰되, 실측이라고 부르지 않는다.
CALIBRATION = Path.home() / "rl_ws/hdgp/log/logs/r2s_autotune/results/right_arm_best_calibration.json"
ASSET_URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="gripper_left")
    parser.add_argument("--model", choices=["rate", "pd"], default="pd")
    parser.add_argument("--max-vel", type=float, default=20.0,
                        help="MockArm 내부 세트포인트 leash[rad/s]. ★기본을 크게 둔 이유: "
                             "브리지 rate-limit 은 **재생기가 이미** 걸었다(그것도 실제 "
                             "위치가 아니라 직전 세트포인트 기준으로). 여기서 또 걸면 "
                             "leash 가 실제 위치를 참조하므로 스프링 토크가 kp·max_vel·dt "
                             "로 묶여 종단속도가 (kp·max_vel·dt − fc)/kd 에 갇힌다 — "
                             "팔이 아니라 모델이 만든 정체다.")
    parser.add_argument("--rate-hz", type=float, default=50.0,
                    help="적분·발행 주기. ★높일수록 세트포인트 선행량이 작아져 "
                         "마찰 데드밴드에 걸린다 — MockArm 이 거부하면 낮출 것.")
    parser.add_argument("--idle-arm-offset", type=float, default=0.0,
                        help="유휴 팔을 rest 에서 이만큼 어긋나게 둔다(게이트 시험용)")
    args = parser.parse_args()

    profile = load_robot_profile(args.robot)
    arm_canonical = list(profile.arm_canonical)
    idle_canonical = list(profile.idle_arm_canonical)
    ee_canonical = list(profile.ee_canonical)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory

    dt = 1.0 / args.rate_hz
    q0 = np.zeros(7)
    inertia = np.asarray(
        effective_inertia(str(ASSET_URDF), arm_canonical, dict(zip(arm_canonical, q0.tolist()))),
        dtype=float)
    # `load_arm_pd` 가 이미 관절군을 관절별 배열로 펴 준다 — 여기서 다시 펴면 그 분할이
    # 두 벌이 되고, 캘리브 쪽이 군을 바꾸면 조용히 어긋난다.
    kp, kd, fc, dataset = load_arm_pd(CALIBRATION)

    class FakeArmBridge(Node):
        def __init__(self) -> None:
            super().__init__("fake_arm_bridge")
            # ★substeps 를 줄이면 마찰 데드밴드가 부풀어 저관성 관절이 조용히 얼어붙는다
            #   (`MockArm` 이 이제 거부한다). 기본값을 쓴다.
            self.arm = MockArm(q0=q0.copy(), model=args.model, max_vel=args.max_vel,
                               dt=dt, kp=kp, kd=kd, fc=fc, inertia=inertia)
            self.arm_cmd = q0.copy()
            self.grip = 0.0
            self.idle = np.asarray(idle_arm_rest_pose(profile)) + args.idle_arm_offset
            self.pub = self.create_publisher(JointState, profile.topics["arm_state"], 10)
            self.create_subscription(JointTrajectory, profile.topics["arm_traj"],
                                     self._arm_cb, 10)
            self.create_subscription(JointTrajectory, profile.topics["ee_traj"],
                                     self._grip_cb, 10)
            self.create_timer(dt, self._tick)
            self.received = 0
            self.get_logger().info(
                f"모델 {args.model} · {args.rate_hz:.0f} Hz · "
                f"{profile.topics['arm_traj']} 수신 → {profile.topics['arm_state']} 발행")
            if args.model == "pd":
                self.get_logger().info(
                    f"게인 출처 {CALIBRATION.name} (dataset={dataset}) — "
                    "★우팔 식별값이다. 좌팔 실측 캘리브는 없다.")

        def _index_of(self, msg, source_names):
            return [msg.joint_names.index(n) if n in msg.joint_names else None
                    for n in source_names]

        def _arm_cb(self, msg: JointTrajectory) -> None:
            if not msg.points:
                return
            positions = msg.points[0].positions
            for k, i in enumerate(self._index_of(msg, profile.arm_source)):
                if i is not None and i < len(positions):
                    sign = profile.joint_limits[arm_canonical[k]]["sign"]
                    self.arm_cmd[k] = positions[i] * sign
            self.received += 1

        def _grip_cb(self, msg: JointTrajectory) -> None:
            if msg.points and msg.points[0].positions:
                self.grip = float(msg.points[0].positions[0])

        def _tick(self) -> None:
            # MockArm.step 은 상태를 갱신하고 아무것도 돌려주지 않는다 — 결과는 .q 다.
            self.arm.step(self.arm_cmd)
            q, qd = self.arm.q, self.arm.qd
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            # 실기 `/joint_states` 는 **양팔 + 좌 그리퍼**를 한 메시지에 싣는다. 한쪽만
            # 실으면 재생기의 유휴 팔 게이트가 확인할 값을 못 찾는다.
            names, positions = [], []
            for k, canonical in enumerate(arm_canonical):
                spec = profile.joint_limits[canonical]
                names.append(spec["source"]); positions.append(float(q[k]) * spec["sign"])
            for k, canonical in enumerate(idle_canonical):
                spec = profile.joint_limits[canonical]
                names.append(spec["source"]); positions.append(float(self.idle[k]) * spec["sign"])
            for canonical in ee_canonical:
                spec = profile.joint_limits[canonical]
                names.append(spec["source"]); positions.append(float(self.grip) * spec["sign"])
            msg.name = names
            msg.position = positions
            # 속도는 팔만 실값, 나머지는 0. effort 는 **모델이 모른다** — 0 을 실으면
            # 재생기의 effort 중단 조건이 "항상 안전"으로 읽힌다. 실기에서만 유효한
            # 게이트라는 것을 로그로 남기고 0 을 싣는다(메시지 형식은 맞춰야 한다).
            msg.velocity = [0.0] * len(names)
            for k in range(len(arm_canonical)):
                msg.velocity[k] = float(qd[k]) * profile.joint_limits[arm_canonical[k]]["sign"]
            msg.effort = [0.0] * len(names)
            self.pub.publish(msg)

    rclpy.init()
    node = FakeArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"수신한 궤적 {node.received} 건")
        node.destroy_node()
        # Ctrl-C 는 rclpy 가 컨텍스트를 이미 내린 뒤에 여기로 온다 — 두 번 내리면
        # RCLError 가 나고, 그게 종료 코드를 오염시켜 "실패"처럼 보인다.
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
