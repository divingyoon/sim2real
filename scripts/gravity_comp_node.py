#!/usr/bin/env python3
"""**연속** 중력보상 — 궤적을 따라가는 **동안** tau_ff 를 실시간으로 낸다.

`robotctl pose gravity` 는 한 자세를 붙잡는 도구다(정지 유지·스케일 스윕용). 재생
중에는 관절각이 매 순간 바뀌므로 중력토크도 매 순간 다시 계산해야 한다. 이 노드가
그 일을 한다: `/joint_states` 를 구독해 실측 자세로 중력토크를 계산하고
`/<effort_controller>/commands` 로 발행한다.

왜 필요한가(08.31 실측). preset 궤적은 sim 에서 접촉 0 건으로 검증됐는데 실기에서는
테이블과 겹쳤다. 궤적이 틀린 게 아니라 **실기가 그 궤적을 못 따라가서** 실제 경로가
아래로 처진 것이다(j7 −12.8°, palm −50 mm). 보상이 켜지면 sim 궤적 = 실기 궤적이 되고,
그러면 **궤적을 보정할 필요가 없어진다** — j2 를 올린 safe 변형도 필요 없다.

  · JTC 는 계속 position 을 잡는다. 이 노드는 그 위에 더해지는 **피드포워드**다
    (`tau = kp(q_des−q) + kd(qd_des−qd) + tau_ff`). 팔을 놓아버리는 것이 아니다.
  · 실측 자세로 계산한다(지령이 아니라) — 처진 자세에서 필요한 토크가 진짜 필요량이다.
  · `--execute` 없으면 계산만 하고 아무것도 발행하지 않는다.
  · Ctrl-C 로 끝낼 때 **0 을 한 번 보내고** 나간다. 마지막 토크가 남아 있으면 팔이
    그 힘을 계속 받는다.

실행 (★사용자 승인 후):
    # effort 컨트롤러가 먼저 떠 있어야 한다
    ~/rl_ws/robot_control/ros_ws/load_effort_controllers.sh right

    python3 gravity_comp_node.py --scale 1.0 \\
        --payload 0.9130,-0.00450,-0.01723,0.22147 --execute
    # 다른 터미널에서 shadow_replay 를 돌리면 그 궤적을 따라가며 보상이 유지된다
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, "/home/user/rl_ws/robot_control/src")

DEFAULT_URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
#: 08.31 관절별 스윕 결과. 1 을 넘는 값은 URDF 손 질량이 실제보다 가벼워서였다 —
#  페이로드를 실질량에 맞추면 1.0 근처로 내려와야 한다(검증 대기).
DEFAULT_SCALE = "1.0"
PUBLISH_HZ = 50.0
#: 모델이 미친 값을 내면 팔에 그대로 간다. 관절별 상한을 넘으면 발행을 멈춘다.
TORQUE_CAP_NM = 20.0


def _parse_payload(text: str | None):
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--payload 는 MASS,X,Y,Z 네 값 — 받은 {len(parts)}개")
    values = [float(p) for p in parts]
    return values[0], values[1:]


def _parse_scale(text: str, n: int) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 1:
        return np.full(n, float(parts[0]))
    if len(parts) != n:
        raise SystemExit(f"--scale 은 1개 또는 {n}개 — 받은 {len(parts)}개")
    return np.array([float(p) for p in parts])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="tesollo_sensor__right")
    parser.add_argument("--group", default="openarm_right_arm")
    parser.add_argument("--profile", default="openarm_tesollo")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--payload", help="MASS,X,Y,Z — 중력 모델이 빠뜨린 손 몫")
    parser.add_argument("--scale", default=DEFAULT_SCALE,
                        help="1개 또는 관절 수만큼. 08.31 실측 최적 1.1(질량 오차 보정 전)")
    parser.add_argument("--cap", type=float, default=TORQUE_CAP_NM)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from robot_control.kinematics import chain_from_urdf, with_payload
    from robot_control.profile import load_builtin_profile

    profile = load_builtin_profile(args.profile)
    group = profile.groups[args.group]
    canonical = list(group.joints)
    source_of = {j.canonical: j.source for j in profile.joints}
    sources = [source_of[c] for c in canonical]
    sign_of = {j.canonical: j.sign for j in profile.joints}

    urdf = args.urdf.read_text()
    tip = group.asset_tip_link or group.tip_link
    chain = chain_from_urdf(urdf, canonical, tip)
    payload = _parse_payload(args.payload)
    if payload is not None:
        chain = with_payload(chain, payload[0], payload[1])
        print(f"payload {payload[0]:.4f} kg at {np.round(payload[1], 5).tolist()} "
              f"→ {chain.links[-1].name}")
    scale = _parse_scale(args.scale, len(canonical))
    total = sum(link.mass for link in chain.links)
    print(f"{args.group}: {len(canonical)} joints · {total:.3f} kg modelled · "
          f"scale {np.round(scale, 2).tolist()}")
    if not args.execute:
        print("DRY RUN — 실제로 발행하려면 --execute")
        return 0
    if group.effort_controller is None:
        raise SystemExit(f"group {args.group!r} 에 effort_controller 선언이 없다")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    topic = f"/{group.effort_controller}/commands"

    class GravityComp(Node):
        def __init__(self) -> None:
            super().__init__("gravity_comp")
            self.pub = self.create_publisher(Float64MultiArray, topic, 10)
            self.create_subscription(JointState, "/joint_states", self._cb,
                                     qos_profile_sensor_data)
            self.q = np.zeros(len(canonical))
            self.have = False
            self.n = 0
            self.stopped: str | None = None
            self.create_timer(1.0 / PUBLISH_HZ, self._tick)

        def _cb(self, msg: JointState) -> None:
            index = {n: i for i, n in enumerate(msg.name)}
            for k, src in enumerate(sources):
                i = index.get(src)
                if i is not None:
                    self.q[k] = msg.position[i] * sign_of[canonical[k]]
            self.have = True

        def _tick(self) -> None:
            if not self.have or self.stopped:
                return
            tau = chain.gravity_torque(self.q) * scale
            worst = float(np.max(np.abs(tau)))
            if worst > args.cap:
                self.stopped = f"모델 토크 {worst:.1f} N·m 가 상한 {args.cap} 초과"
                self.get_logger().error(self.stopped + " — 발행 중단, 0 송출")
                self.pub.publish(Float64MultiArray(data=[0.0] * len(canonical)))
                return
            self.pub.publish(Float64MultiArray(data=[float(v) for v in tau]))
            self.n += 1
            if self.n % (int(PUBLISH_HZ) * 5) == 0:
                self.get_logger().info(
                    f"{self.n} 발행 · tau " + " ".join(f"{v:+.2f}" for v in tau))

    rclpy.init()
    node = GravityComp()

    # ★Ctrl-C 를 **우리가** 받는다. rclpy 의 기본 핸들러가 먼저 받으면 컨텍스트를
    #   내리고, 그 뒤로는 0 을 발행할 방법이 없다 — 09.01 에 실제로 그렇게 실패해
    #   ("Context.init() must only be called once") effort 컨트롤러에 마지막 토크가
    #   남았다. 핸들러를 덮어써서 루프를 정상적으로 빠져나오게 하면, 컨텍스트가
    #   살아 있는 채로 해제 publish 를 할 수 있다.
    interrupted = False

    def _on_signal(signum, frame):        # noqa: ARG001
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"→ {topic} @ {PUBLISH_HZ:.0f} Hz · Ctrl-C 로 종료(0 을 보내고 나간다)")
    try:
        while rclpy.ok() and node.stopped is None and not interrupted:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:             # 핸들러가 놓친 경우의 그물
        interrupted = True
    finally:
        released = _release(node, len(canonical), Float64MultiArray, rclpy)
        print(f"\n종료 · {node.n} 프레임 발행"
              + (f" · 중단: {node.stopped}" if node.stopped else "")
              + (" · 0 송출 완료" if released else ""))
    return 0


def _release(node, width: int, message_type, rclpy) -> bool:
    """마지막 토크를 남기지 않는다 — 남으면 팔이 그 힘을 계속 받는다.

    컨텍스트가 살아 있으면 그대로 보내고, 이미 내려갔으면 **새 Context** 로 보낸다.
    `rclpy.init()` 을 다시 부르는 것은 안 된다(전역 컨텍스트는 한 번뿐이다).
    """
    try:
        if rclpy.ok():
            for _ in range(5):
                node.pub.publish(message_type(data=[0.0] * width))
                rclpy.spin_once(node, timeout_sec=0.02)
            return True
    except Exception as exc:              # noqa: BLE001 — 아래 경로로 한 번 더 시도한다
        print(f"⚠ 0 송출 1차 실패({exc}) — 새 컨텍스트로 재시도")

    try:
        from rclpy.context import Context
        from rclpy.node import Node

        context = Context()
        rclpy.init(context=context)
        try:
            spare = Node("gravity_comp_release", context=context)
            pub = spare.create_publisher(message_type, node.pub.topic_name, 10)
            time.sleep(0.3)               # 구독자가 붙을 틈
            for _ in range(5):
                pub.publish(message_type(data=[0.0] * width))
                rclpy.spin_once(spare, timeout_sec=0.02)
            return True
        finally:
            rclpy.shutdown(context=context)
    except Exception as exc:              # noqa: BLE001 — 여기서 죽으면 토크가 남는다
        print(f"⚠ 0 송출 실패({exc}) — effort 컨트롤러를 unload 해서 풀 것:\n"
              "  ~/rl_ws/robot_control/ros_ws/load_effort_controllers.sh --unload right")
        return False


if __name__ == "__main__":
    raise SystemExit(main())
