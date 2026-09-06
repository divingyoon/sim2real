"""episode_master — 제어 전용(control_only) 계약에서 에피소드 이벤트를 내는 작은 마스터 노드.

정책 계약에서는 obs 노드가 에피소드 마스터다(services episode/{reset,start,stop,abort} + latched
/policy_control/episode). 제어 전용 계약(정책 없음)에서는 obs 노드가 뜨지 않으므로 이 노드가 같은
서비스와 같은 JSON 이벤트를 낸다 — fabric 노드(direct 모드)와 pd 노드는 마스터가 누구인지 모른다.

  services std_srvs/Trigger /policy_control/episode/{reset,start,stop,abort}
  publish  /policy_control/episode (String JSON, reliable+transient_local depth 1: EpisodeEvent + t_ns)
  publish  /policy_control/status/episode_master (String JSON)

정책 계약을 주면 기동을 거부한다(마스터가 둘이면 seq/episode 가 갈린다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):          # `python policy_control/policy_control/<node>.py` (launch use_source 모드)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import _paths  # noqa: F401,E402
from policy_control.chain import EpisodeEvent  # noqa: E402
from policy_control.contract import ContractError, DeployContract, load_contract  # noqa: E402

NS = "/policy_control"
NODE = "episode_master"
EVENTS = ("reset", "start", "stop", "abort")


class EpisodeMasterError(RuntimeError):
    """계약이 제어 전용이 아니거나 읽을 수 없다."""


def contract_home(contract: DeployContract) -> dict:
    """모든 side 의 로봇 홈(팔 + 손) — 이벤트 home_q 필드(정보용; fabric 은 계약 fabric.home_q 를 쓴다)."""
    home: dict = {}
    for name in contract.side_names:
        s = contract.side(name)
        home.update(dict(zip(s.arm_joints, s.home_arm)))
        home.update(s.home_hand)
    return home


class EpisodeBook:
    """에피소드 번호·상태 — ROS 없는 순수 상태기계. 메서드는 (event | None, reasons) 를 돌려준다."""

    def __init__(self, home_q: dict) -> None:
        self.home_q = dict(home_q)
        self.episode = 0
        self.phase = "idle"              # idle | armed | running | stopped

    def reset(self) -> tuple[EpisodeEvent, list]:
        self.episode += 1
        self.phase = "armed"
        return EpisodeEvent(episode=self.episode, event="reset", object_anchor=None, home_q=self.home_q), []

    def start(self) -> tuple[EpisodeEvent | None, list]:
        if self.phase != "armed":
            return None, [f"no reset episode (phase {self.phase}) — call episode/reset first"]
        self.phase = "running"
        return EpisodeEvent(episode=self.episode, event="start", object_anchor=None, home_q=self.home_q), []

    def end(self, event: str, reason: str) -> tuple[EpisodeEvent, list]:
        self.phase = "stopped"
        return EpisodeEvent(episode=self.episode, event=event, object_anchor=None, home_q=self.home_q,
                            reasons=(reason,)), []


def _trigger(res, ok: bool, reasons: list):
    res.success = bool(ok)
    res.message = json.dumps({"ok": bool(ok), "reasons": [str(r) for r in reasons]}, ensure_ascii=False)
    return res


def load_control_contract(path: Path) -> DeployContract:
    try:
        contract = load_contract(path)
    except ContractError as exc:
        raise EpisodeMasterError(f"contract: {exc}") from exc
    if not contract.control_only:
        raise EpisodeMasterError(f"{path}: not a control_only contract — obs_node is the episode master there")
    return contract


try:
    from rclpy.node import Node
except ImportError:                      # 순수 로직(EpisodeBook)은 ROS 없이도 import 된다
    Node = object                        # type: ignore[misc,assignment]


class EpisodeMaster(Node):
    def __init__(self, **kw) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        super().__init__(NODE, **kw)
        self.declare_parameter("contract", "")
        path = Path(str(self.get_parameter("contract").value))
        if not path.is_file():
            raise EpisodeMasterError(f"contract file not found: {path!r}")
        self.contract = load_control_contract(path)
        self.book = EpisodeBook(contract_home(self.contract))
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(String, f"{NS}/episode", latched)
        self._pub_status = self.create_publisher(String, f"{NS}/status/{NODE}", QoSProfile(depth=10))
        self._String = String
        for name in EVENTS:
            self.create_service(Trigger, f"{NS}/episode/{name}", getattr(self, f"_srv_{name}"))
        self.get_logger().info(f"episode master for control-only contract {self.contract.asset.name if self.contract.asset else path.name}")

    def _srv_reset(self, _req, res):
        return self._emit(*self.book.reset(), res)

    def _srv_start(self, _req, res):
        return self._emit(*self.book.start(), res)

    def _srv_stop(self, _req, res):
        return self._emit(*self.book.end("stop", "user stop"), res)

    def _srv_abort(self, _req, res):
        return self._emit(*self.book.end("abort", "abort requested"), res)

    def _emit(self, event: EpisodeEvent | None, reasons: list, res):
        if event is None:
            return _trigger(res, False, reasons)
        body = {**event.as_dict(), "t_ns": self.get_clock().now().nanoseconds}
        self._pub.publish(self._String(data=json.dumps(body)))
        status = {"node": NODE, "phase": self.book.phase, "episode": self.book.episode, "event": event.event,
                  "ok": True, "reasons": list(event.reasons), "t_pub_ns": body["t_ns"]}
        self._pub_status.publish(self._String(data=json.dumps(status)))
        self.get_logger().info(f"episode {event.episode} {event.event} {list(event.reasons)}")
        return _trigger(res, True, [])


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = EpisodeMaster()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
