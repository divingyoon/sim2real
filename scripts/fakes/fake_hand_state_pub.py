#!/usr/bin/env python3
"""fake 손 상태 퍼블리셔 — 테솔로 손 분리 상태에서 풀 파이프라인 플러밍 검증용.

grasp_inference / policy_control obs 는 /dg5f_<side>/joint_states(손 20관절)와 /dg5f_<side>/tip_forces_xyz
(tip 5×3) 가 있어야 start 게이트를 통과하고 obs 를 조립한다. 손을 분리했을 때 이 노드가 **정적 손 상태**
(APPROACH/open 자세)와 **접촉 0**을 발행해, 지각→정책→fabric→pd 경로를 손 없이 흐르게 한다.

⚠️ 실제 손 구동은 없음(손 분리). 정책의 손 명령은 무시된다 — 팔 궤적만 검증하는 용도.

--echo 모드(08.03, RUNNING 팔 후퇴 진단): 정적 자세 대신 정책의 손 명령을 그대로 관절상태로 되돌려
발행한다. sim 에서는 손이 명령을 즉시 추종하므로, echo 는 "sim 처럼 진화하는 손 obs"를 손 없이 재현한다.
--echo-topic /policy_control/joint_target (JointState, canonical 이름): 이 손의 관절이 **모두** 있는 메시지만
반사한다 — 양팔 fabric/pd 가 같은 토픽에 낼 때 다른 팔의 목표는 자연히 걸러진다.

두 모드:
  레거시 프로필  --robot tesollo_sensor__right (scripts/robot_profile, hdgp preset 의 HAND_APPROACH_POSE)
  계약          --contract <deploy_contract.json> --robot-yaml <policy_control robots yaml> --side left|right
                — 관절 = 계약 sides[side].hand_joints, source 이름 = 합친 프로필(좌손 보충 포함), 자세 = 계약 home_hand,
                  토픽 = /dg5f_<side>/{joint_states, tip_forces_xyz, contact_forces}

★자세·관절명을 하드코딩하지 않는다: 좌측은 우측의 부호 미러라 복제하면 조용히 틀린다.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

_HERE = Path(__file__).resolve().parent
_SIM2REAL = _HERE.parents[1]
for _p in (_SIM2REAL / "scripts", _SIM2REAL / "policy_control"):   # ★`scripts/` 는 한 단계 위, policy_control 은 계약 모드용
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NUM_TIPS = 5


@dataclass(frozen=True)
class HandSpec:
    """어느 손을 흉내내나: 이름(canonical/source)·초기 자세·토픽."""
    name: str
    canonical: tuple
    source: tuple
    pose: tuple                 # source 순 초기 자세(부호 +1: canonical == source 값)
    js_topic: str
    xyz_topic: str
    norm_topic: str


def spec_from_profile(name: str) -> HandSpec:
    from robot_profile import load_hdgp_module, load_robot_profile

    profile = load_robot_profile(name)
    # 좌측은 우측의 부호 미러 — preset 에서 가져오지 않으면 조용히 틀린다
    pose = tuple(float(v) for v in load_hdgp_module(profile, "preset").HAND_APPROACH_POSE)
    if len(pose) != len(profile.ee_source):
        raise ValueError(f"APPROACH 자세 {len(pose)}D != EE 관절 {len(profile.ee_source)}개")
    return HandSpec(name=profile.name, canonical=tuple(profile.ee_canonical), source=tuple(profile.ee_source), pose=pose,
                    js_topic=profile.topics["ee_state"], xyz_topic=profile.topics["tip_force_xyz"],
                    norm_topic=profile.topics["tip_force_norm"])


def spec_from_contract(contract_path: Path, robot_yaml: Path, side: str) -> HandSpec:
    from policy_control.contract import load_contract
    from policy_control.sources import load_profile, load_robot_cfg

    contract = load_contract(contract_path)
    profile = load_profile(load_robot_cfg(robot_yaml).joint_profiles)
    s = contract.side(side)
    if not s.hand_joints:
        raise ValueError(f"contract side {side} has no hand joints")
    missing = [j for j in s.hand_joints if j not in s.home_hand]
    if missing:
        raise ValueError(f"contract side {side} home_hand lacks {missing}")
    ns = f"/dg5f_{side}"
    return HandSpec(name=f"{contract.run.task}:{side}", canonical=tuple(s.hand_joints),
                    source=tuple(profile[j]["source"] for j in s.hand_joints),
                    pose=tuple(float(s.home_hand[j]) * float(profile[j]["sign"]) for j in s.hand_joints),
                    js_topic=f"{ns}/joint_states", xyz_topic=f"{ns}/tip_forces_xyz", norm_topic=f"{ns}/contact_forces")


DRIVER_PID_P = 1.5          # dg5f 드라이버 JTC 기본 PID p = 벤더값 (09.06 이후 pd 노드도 같은 값을 기대한다)
DRIVER_PID_D = 0.0


class FakeHandController(Node):
    """드라이버 JTC 컨트롤러 노드 자리(`/dg5f_<side>/dg5f_<side>_controller`) — gains.<joint>.{p,d} 파라미터만 흉내낸다.
    pd 노드의 HandGainsClient 가 GetParameters/SetParameters 로 PID 를 대조·적용하는 경로를 fake 플랜트에서 돌린다."""

    def __init__(self, namespace: str, joints: tuple) -> None:
        ns = namespace.strip("/")
        super().__init__(f"{ns}_controller", namespace=f"/{ns}")
        for j in joints:
            self.declare_parameter(f"gains.{j}.p", DRIVER_PID_P)
            self.declare_parameter(f"gains.{j}.d", DRIVER_PID_D)
        self.get_logger().info(f"fake 손 컨트롤러 파라미터 {self.get_fully_qualified_name()}: gains.<joint>.p={DRIVER_PID_P} d={DRIVER_PID_D}")


class FakeHandState(Node):
    def __init__(self, spec: HandSpec, rate_hz: float, echo_topic: str | None = None,
                 cmd_topic: str | None = None) -> None:
        super().__init__("fake_hand_state_pub")
        self.spec = spec
        self.echo = bool(echo_topic or cmd_topic)
        self.rate_hz = rate_hz
        self.js_pub = self.create_publisher(JointState, spec.js_topic, 10)
        self.xyz_pub = self.create_publisher(Float64MultiArray, spec.xyz_topic, 10)
        self.ct_pub = self.create_publisher(Float64MultiArray, spec.norm_topic, 10)
        self._last_cmd: list[float] = list(spec.pose)
        self._prev_pub: list[float] = list(spec.pose)
        if echo_topic:
            # policy_control: fabric/pd 의 JointState 목표(canonical 이름)를 이름으로 반사
            self.create_subscription(JointState, echo_topic, self._joint_target_cb, 10)
        elif cmd_topic:
            self.create_subscription(Float64MultiArray, cmd_topic, self._cmd_cb, 10)
        self.create_timer(1.0 / rate_hz, self._tick)
        mode = f"echo({echo_topic or cmd_topic} 반사)" if self.echo else "정적 자세"
        self.get_logger().info(
            f"fake 손 상태 발행[{spec.name} · {mode}]: {spec.js_topic}, {spec.xyz_topic} (15×0) + "
            f"{spec.norm_topic} (5×0), {rate_hz:g}Hz\n  ⚠️ 손 분리 상태 플러밍 검증용 — 실제 손 구동 아님")

    def _joint_target_cb(self, msg: JointState) -> None:
        """canonical 이름 → source 순서로 재배열해 반사. 이 손의 관절이 하나라도 빠지면 무시(다른 팔의 목표)."""
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(c in idx for c in self.spec.canonical):
            return
        self._last_cmd = [float(msg.position[idx[c]]) for c in self.spec.canonical]

    def _cmd_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= len(self.spec.source):
            self._last_cmd = list(msg.data[: len(self.spec.source)])

    def _tick(self) -> None:
        pos = self._last_cmd if self.echo else list(self.spec.pose)
        vel = [(p - q) * self.rate_hz for p, q in zip(pos, self._prev_pub)]
        self._prev_pub = list(pos)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.spec.source)
        js.position = list(pos)
        js.velocity = vel
        self.js_pub.publish(js)

        xyz = Float64MultiArray()
        xyz.data = [0.0] * (NUM_TIPS * 3)
        self.xyz_pub.publish(xyz)

        ct = Float64MultiArray()
        ct.data = [0.0] * NUM_TIPS
        self.ct_pub.publish(ct)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", default=None, help="레거시: config/robots 의 구성 프로필 이름 (기본 tesollo_bi_s__right)")
    parser.add_argument("--contract", type=Path, default=None, help="계약 모드: deploy_contract.json")
    parser.add_argument("--robot-yaml", type=Path, default=None, help="계약 모드: policy_control/config/robots/*.yaml")
    parser.add_argument("--side", choices=("left", "right"), default=None, help="계약 모드: 어느 손")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--echo-topic", default=None, help="policy_control 의 /policy_control/joint_target(JointState)을 반사")
    parser.add_argument("--echo", action="store_true", default=False,
                        help="레거시: 정책 손 명령(<ee_cmd>)을 관절상태로 반사 — 진화하는 손 obs 재현")
    parser.add_argument("--controller-node", action="store_true", default=False,
                        help="드라이버 JTC 컨트롤러 노드(/dg5f_<side>/dg5f_<side>_controller, gains.<joint>.p/d 파라미터)도 띄운다")
    args = parser.parse_args()
    if args.contract is not None:
        if args.robot_yaml is None or args.side is None:
            raise SystemExit("--contract 에는 --robot-yaml 과 --side 가 필요하다")
        spec, cmd_topic = spec_from_contract(args.contract, args.robot_yaml, args.side), None
    else:
        from robot_profile import load_robot_profile

        name = args.robot or "tesollo_bi_s__right"
        spec = spec_from_profile(name)
        cmd_topic = load_robot_profile(name).topics["ee_cmd"] if args.echo and not args.echo_topic else None
    rclpy.init()
    node = FakeHandState(spec, args.rate, echo_topic=args.echo_topic, cmd_topic=cmd_topic)
    nodes = [node]
    if args.controller_node:
        nodes.append(FakeHandController(spec.js_topic.rsplit("/", 1)[0], spec.source))
    from rclpy.executors import SingleThreadedExecutor

    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for n in nodes:
            n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
