"""obs_node — `chain.ObsStage` 를 감싸는 얇은 rclpy 껍질 (에피소드 마스터).

    구독  robot yaml `sources:` 의 토픽(joint_state → JointState, pose → PoseStamped,
          float_array → Float64MultiArray; 센서 QoS, /policy_control/* 는 reliable 10)
          /policy_control/action (Float64MultiArray) — 다음 tick 의 last_action(seq 매칭)
    발행  /policy_control/obs (Float64MultiArray, dim.label=세그먼트, data_offset=seq)
          /policy_control/status/obs (String JSON: StageStatus + t_pub_ns/started/action_seq…)
          /policy_control/episode (String JSON, reliable+transient_local depth 1: EpisodeEvent + t_ns)
    서비스 std_srvs/Trigger /policy_control/episode/{reset,start,stop,abort}
          응답 message = JSON {"ok": bool, "reasons": [...]}, success = ok
    타이머 contract.rate.policy_hz (파라미터 policy_hz > 0 이면 덮어씀)

규약: reset = 스냅샷으로 ObsStage.reset(앵커·seq 0 준비, 아직 tick 안 함) → start = tick 시작 →
stop/abort = 종료. 제어 논리는 전부 chain/obs_core 에 있고 여기엔 배선·디코딩·시각만 있다.
잘못된 메시지는 status 에 사유로 남기고 계속 돈다(노드는 죽지 않는다).
파라미터 `side`(left|right, 기본 = 계약 primary_side): 관절 순서·바디는 `contract.side(side)`,
센서는 robot yaml 에서 그 팔 것만(`sources.select_side` — 양팔 yaml 은 `<역할>_<side>`). 거부:
control-only 계약(세그먼트 없음), yaml 이 그 팔을 안 다룸, fk.kind=fabric(FK 제공자를 fk= 로 주지 않는 한 —
fabric 노드가 palm_pose/tips 를 든다). fk.kind ∈ {left_gripper, urdf_chain} 은 노드가 직접 만든다.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):                                  # `python policy_control/policy_control/obs_node.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"

import json                                                    # noqa: E402
import time                                                    # noqa: E402
from dataclasses import dataclass, replace                     # noqa: E402

import numpy as np                                             # noqa: E402
import rclpy                                                   # noqa: E402
from geometry_msgs.msg import PoseStamped                      # noqa: E402
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import JointState                         # noqa: E402
from std_msgs.msg import Float64MultiArray, String             # noqa: E402
from std_srvs.srv import Trigger                               # noqa: E402

from . import _paths                                           # noqa: E402
from .chain import ChainError, EpisodeEvent, ObsStage, StageStatus  # noqa: E402
from .codec import (CodecError, decode_action, decode_float_array, decode_joint_state, decode_pose,  # noqa: E402
                    encode_obs, encode_status)
from .contract import ContractError, DeployContract, load_contract  # noqa: E402
from .fk_numpy import FKError, make_fk                         # noqa: E402
from .obs_core import ObsCore, ObsError                        # noqa: E402
from .sources import RobotCfgError, SourceCfg, SourceSet, load_robot_cfg, select_side  # noqa: E402

NS = "/policy_control"
CHAIN_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
EPISODE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
MSG_BY_TYPE = {"joint_state": JointState, "pose": PoseStamped, "float_array": Float64MultiArray}
SUPPORTED_FK_KINDS = ("left_gripper", "urdf_chain")
_NODE_ERRORS = (CodecError, ChainError, ObsError, RobotCfgError, FKError, ValueError)


class ObsNodeError(RuntimeError):
    """노드를 만들 수 없다(파라미터/계약/로봇 yaml/FK)."""


@dataclass(frozen=True)
class _Params:
    contract: Path
    robot: Path
    device: str
    policy_hz: float
    side: str


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e3


def resolve_path(text: str) -> Path:
    """절대경로 그대로; 상대경로는 cwd → sim2real → rl_ws 순으로 존재하는 첫 것(launch 는 절대경로를 준다)."""
    p = Path(text).expanduser()
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, _paths.SIM2REAL / p, _paths.RL_WS / p]
    for c in candidates:
        if c.exists():
            return c
    raise ObsNodeError(f"{text!r} not found (tried {[str(c) for c in candidates]})")


def _trigger(res, ok: bool, reasons: list[str]):
    res.success = bool(ok)
    res.message = json.dumps({"ok": bool(ok), "reasons": [str(r) for r in reasons]}, ensure_ascii=False)
    return res


class ObsNode(Node):
    """subscribe → codec.decode → SourceSet → ObsStage.tick → codec.encode → publish."""

    def __init__(self, *, context=None, parameter_overrides=None, fk=None) -> None:
        super().__init__("obs_node", context=context, parameter_overrides=parameter_overrides)
        p = self._read_params()
        self.contract = self._load(load_contract, p.contract, ContractError, "contract")
        if self.contract.control_only:
            raise ObsNodeError(f"contract {p.contract} is control-only (no policy, no obs segments) — "
                               "the obs node has nothing to build; give it a policy contract")
        self.side = p.side or str(self.contract.primary_side)
        self.robot_cfg = self._select_side(self._load(load_robot_cfg, p.robot, RobotCfgError, "robot yaml"))
        self.fk = fk if fk is not None else self._make_fk(self.contract, self.side)
        self.sources = SourceSet(self.robot_cfg)
        self.stage = ObsStage(self.contract, self.robot_cfg, self.fk, core=self._make_core())
        self.segments = [(s.name, s.dim) for s in self.contract.obs.segments]
        self._started = False
        self._last_event: EpisodeEvent | None = None
        self._action: tuple[np.ndarray, int] | None = None
        self._action_matched = False
        self._source_errors: dict[str, str] = {}
        self._pub_obs = self.create_publisher(Float64MultiArray, f"{NS}/obs", CHAIN_QOS)
        self._pub_status = self.create_publisher(String, f"{NS}/status/obs", CHAIN_QOS)
        self._pub_episode = self.create_publisher(String, f"{NS}/episode", EPISODE_QOS)
        self._subscribe_sources()
        self.create_subscription(Float64MultiArray, f"{NS}/action", self._on_action, CHAIN_QOS)
        for name, fn in (("reset", self._srv_reset), ("start", self._srv_start),
                         ("stop", self._srv_stop), ("abort", self._srv_abort)):
            self.create_service(Trigger, f"{NS}/episode/{name}", fn)
        self.hz = p.policy_hz if p.policy_hz > 0.0 else float(self.contract.rate.policy_hz)
        self.create_timer(1.0 / self.hz, self._on_tick)
        self.get_logger().info(f"obs_node: {self.contract.run.experiment} side {self.side} @ {self.hz:g} Hz, "
                               f"robot {self.robot_cfg.robot}, sources {sorted(self.robot_cfg.sources)}")

    # ---------------------------------------------------------------- setup
    def _read_params(self) -> _Params:
        self.declare_parameter("contract", "")
        self.declare_parameter("robot", "")
        self.declare_parameter("device", "cuda:0")            # launch 가 넘긴다 — obs 노드는 쓰지 않는다
        self.declare_parameter("policy_hz", 0.0)
        self.declare_parameter("side", "")                    # left | right; '' = 계약 primary_side
        contract = str(self.get_parameter("contract").value)
        robot = str(self.get_parameter("robot").value)
        if not contract or not robot:
            raise ObsNodeError("parameters 'contract' and 'robot' are required (-p contract:=… -p robot:=…)")
        return _Params(contract=resolve_path(contract), robot=resolve_path(robot),
                       device=str(self.get_parameter("device").value),
                       policy_hz=float(self.get_parameter("policy_hz").value),
                       side=str(self.get_parameter("side").value))

    def _select_side(self, cfg):
        try:
            return select_side(cfg, self.side)
        except RobotCfgError as exc:
            raise ObsNodeError(f"robot yaml {cfg.robot} / side {self.side!r}: {exc}") from exc

    def _make_core(self) -> ObsCore:
        try:
            return ObsCore(self.contract, self.robot_cfg, self.fk, side=self.side)
        except (ObsError, ContractError) as exc:
            raise ObsNodeError(f"obs core (side {self.side!r}): {exc}") from exc

    @staticmethod
    def _load(loader, path: Path, err_type, what: str):
        try:
            return loader(path)
        except err_type as exc:
            raise ObsNodeError(f"{what} {path}: {exc}") from exc

    @staticmethod
    def _make_fk(contract: DeployContract, side: str):
        kind = contract.obs.fk.get("kind")
        if kind not in SUPPORTED_FK_KINDS:
            raise ObsNodeError(f"obs_node: fk.kind {kind!r} needs an FK provider (fabric palm_pose/tips) — "
                               f"the node builds only {SUPPORTED_FK_KINDS} itself; pass fk= to ObsNode")
        try:
            return make_fk(contract, _paths.RL_WS, side=side)
        except (FKError, ContractError) as exc:
            raise ObsNodeError(f"fk (side {side!r}): {exc}") from exc

    def _subscribe_sources(self) -> None:
        by_topic: dict[str, list[SourceCfg]] = {}
        for s in self.robot_cfg.sources.values():
            by_topic.setdefault(s.topic, []).append(s)
        for topic, srcs in by_topic.items():
            types = {s.type for s in srcs}
            if len(types) != 1:
                raise ObsNodeError(f"topic {topic}: sources of different types share it ({sorted(types)})")
            qos = CHAIN_QOS if topic.startswith(NS + "/") else qos_profile_sensor_data
            self.create_subscription(MSG_BY_TYPE[srcs[0].type], topic, self._source_cb(srcs), qos)

    # ---------------------------------------------------------------- callbacks (tiny)
    def _source_cb(self, srcs: list[SourceCfg]):
        def cb(msg) -> None:
            now = time.monotonic()
            for s in srcs:
                self._update_source(s, msg, now)
        return cb

    def _update_source(self, s: SourceCfg, msg, now: float) -> None:
        try:
            if s.type == "joint_state":
                self.sources.update_from_joint_state(s.name, decode_joint_state(msg), now)
            elif s.type == "pose":
                self.sources.update_from_pose(s.name, decode_pose(msg), now)
            else:
                self.sources.update_from_float_array(s.name, decode_float_array(msg), now)
            self._source_errors.pop(s.name, None)
        except (CodecError, RobotCfgError) as exc:
            self._source_errors[s.name] = str(exc)
            self.get_logger().warning(f"source {s.name} ({s.topic}): {exc}", throttle_duration_sec=1.0)

    def _on_action(self, msg) -> None:
        try:
            self._action = decode_action(msg, self.contract.policy.action_dim)
        except CodecError as exc:
            self.get_logger().warning(f"action: {exc}", throttle_duration_sec=1.0)

    def _on_tick(self) -> None:
        t0 = time.perf_counter()
        if not self._started:
            self._publish_status(self._idle_status(), t0)
            return
        state = self.sources.snapshot(time.monotonic())
        try:
            tick = self.stage.tick(state, last_action=self._matched_action())
        except _NODE_ERRORS as exc:
            self.get_logger().error(f"tick: {exc}", throttle_duration_sec=1.0)
            self._publish_status(self._error_status(str(exc)), t0)
            return
        if tick.out.valid:
            self._pub_obs.publish(encode_obs(tick.out.obs, self.segments, tick.out.seq))
        if tick.abort:
            self._started = False
            self._publish_event(replace(self._last_event, event="abort", object_anchor=None,
                                        reasons=tuple(tick.out.reasons)))
        self._publish_status(tick.status, t0)

    # ---------------------------------------------------------------- services
    def _srv_reset(self, _req, res):
        state = self.sources.snapshot(time.monotonic())
        reasons = [f"missing source {m}" for m in state.missing] + [f"stale source {s}" for s in state.stale]
        if reasons:
            return _trigger(res, False, reasons)
        try:
            event = self.stage.reset(state)
        except _NODE_ERRORS as exc:
            return _trigger(res, False, [f"reset: {exc}"])
        self._started, self._action, self._last_event = False, None, event
        self._publish_event(event)
        return _trigger(res, True, [])

    def _srv_start(self, _req, res):
        if self.stage.phase != "running" or self._last_event is None:
            return _trigger(res, False, [f"no reset episode (phase {self.stage.phase}) — call episode/reset first"])
        self._started = True
        self._publish_event(replace(self._last_event, event="start", reasons=()))
        return _trigger(res, True, [])

    def _srv_stop(self, _req, res):
        self._started = False
        self._publish_event(self.stage.stop("user stop"))
        return _trigger(res, True, [])

    def _srv_abort(self, _req, res):
        self._started = False
        self._publish_event(self.stage.abort("user abort"))
        return _trigger(res, True, [])

    # ---------------------------------------------------------------- helpers
    def _matched_action(self) -> np.ndarray | None:
        """다음 seq n 의 obs 에는 seq n−1 의 액션만 들어간다(늦게 온 액션은 버린다 — ObsCore 가 직전 값 유지)."""
        self._action_matched = self._action is not None and self._action[1] == self.stage.core.seq - 1
        return self._action[0] if self._action_matched else None

    def _idle_status(self) -> StageStatus:
        phase = self.stage.phase
        reason = ("episode not started (call episode/start)" if phase == "running"
                  else f"no running episode (phase {phase})")
        return StageStatus(node=ObsStage.NODE, phase=phase, episode=self.stage.episode, seq=self.stage.core.seq,
                           ok=False, reasons=(reason,), proc_ms=0.0, extras={"gap": self.stage.gap})

    def _error_status(self, reason: str) -> StageStatus:
        return StageStatus(node=ObsStage.NODE, phase=self.stage.phase, episode=self.stage.episode,
                           seq=self.stage.core.seq, ok=False, reasons=(f"tick error: {reason}",), proc_ms=0.0)

    def _publish_status(self, status: StageStatus, t0: float) -> None:
        body = {**status.as_dict(), "stage_ms": status.proc_ms, "proc_ms": _ms(t0),
                "t_pub_ns": self.get_clock().now().nanoseconds, "started": self._started,
                "action_seq": None if self._action is None else self._action[1],
                "action_matched": self._action_matched, "source_errors": dict(self._source_errors)}
        self._pub_status.publish(encode_status(body))

    def _publish_event(self, event: EpisodeEvent) -> None:
        body = {**event.as_dict(), "t_ns": self.get_clock().now().nanoseconds}
        self._pub_episode.publish(encode_status(body))
        self.get_logger().info(f"episode {event.episode} {event.event} {list(event.reasons)}")


def spin_until_shutdown(executor) -> None:
    """SIGINT(KeyboardInterrupt)·SIGTERM(rclpy 핸들러 → ExternalShutdownException) 모두 조용히 끝낸다.
    Humble 은 종료 직후 wait-set 생성에서 RCLError 를 낼 수 있다(레이스) — 컨텍스트가 이미 닫혔을 때만 삼킨다."""
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise


def main(argv=None) -> int:
    rclpy.init(args=argv)
    try:
        node = ObsNode()
    except ObsNodeError as exc:
        print(f"obs_node: {exc}", file=sys.stderr)
        rclpy.try_shutdown()
        return 2
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        spin_until_shutdown(executor)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
