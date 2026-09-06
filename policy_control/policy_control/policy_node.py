"""policy_node — `chain.PolicyStage` 를 감싸는 얇은 rclpy 껍질 ("벡터 in → 벡터 out").

    구독  /policy_control/obs (Float64MultiArray; 계약 세그먼트 라벨 검증 → decode_obs)
          /policy_control/episode (String JSON, transient_local) — event 'reset' 이면 PolicyStage.reset
    발행  /policy_control/action (Float64MultiArray, seq 사본)
          /policy_control/status/policy (String JSON: StageStatus + t_pub_ns)
    파라미터 contract / robot(미사용, launch 규약) / device

seq 규칙(0 = 리셋+forward, 중복·역행 = 직전 액션 사본, 미시작 seq>0 = 거부)은 PolicyCore 가 가진다.
체크포인트 적재(torch)는 생성자 인자 core 가 None 일 때만 — 테스트는 duck-typed core(act/reset)를 준다.
잘못된 obs/seq 는 status ok:false 사유로 남기고 액션을 내지 않는다(노드는 죽지 않는다).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):                                  # `python policy_control/policy_control/policy_node.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"

import time                                                    # noqa: E402

import rclpy                                                   # noqa: E402
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Float64MultiArray, String             # noqa: E402

from . import _paths                                           # noqa: E402
from .chain import ChainError, PolicyStage, StageStatus        # noqa: E402
from .codec import CodecError, decode_obs, decode_status, encode_action, encode_status  # noqa: E402
from .contract import ContractError, load_contract             # noqa: E402
from .policy_core import PolicyCore, PolicyIOError, SeqError   # noqa: E402

NS = "/policy_control"
CHAIN_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
EPISODE_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)  # 구독 깊이 10: reset 직후 start 가 오면 depth 1 은 reset 을 버린다(run15)
_TICK_ERRORS = (CodecError, SeqError, PolicyIOError, ChainError, ValueError)


class PolicyNodeError(RuntimeError):
    """노드를 만들 수 없다(파라미터/계약/체크포인트)."""


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
    raise PolicyNodeError(f"{text!r} not found (tried {[str(c) for c in candidates]})")


class PolicyNode(Node):
    """subscribe obs → codec.decode_obs → PolicyStage.tick → codec.encode_action → publish."""

    def __init__(self, *, context=None, parameter_overrides=None, core=None) -> None:
        super().__init__("policy_node", context=context, parameter_overrides=parameter_overrides)
        contract_path, device = self._read_params()
        try:
            self.contract = load_contract(contract_path)
        except ContractError as exc:
            raise PolicyNodeError(f"contract {contract_path}: {exc}") from exc
        self.segments = [(s.name, s.dim) for s in self.contract.obs.segments]
        self.stage = PolicyStage(core if core is not None else self._load_core(device))
        self._phase = "idle"
        self._pub_action = self.create_publisher(Float64MultiArray, f"{NS}/action", CHAIN_QOS)
        self._pub_status = self.create_publisher(String, f"{NS}/status/policy", CHAIN_QOS)
        self.create_subscription(Float64MultiArray, f"{NS}/obs", self._on_obs, CHAIN_QOS)
        self.create_subscription(String, f"{NS}/episode", self._on_episode, EPISODE_QOS)
        self.get_logger().info(f"policy_node: {self.contract.run.experiment} obs {self.contract.policy.obs_dim} → "
                               f"action {self.contract.policy.action_dim}, rnn {self.contract.policy.rnn}")

    # ---------------------------------------------------------------- setup
    def _read_params(self) -> tuple[Path, str]:
        self.declare_parameter("contract", "")
        self.declare_parameter("robot", "")                    # launch 규약 — policy 노드는 쓰지 않는다
        self.declare_parameter("device", "cuda:0")
        contract = str(self.get_parameter("contract").value)
        if not contract:
            raise PolicyNodeError("parameter 'contract' is required (-p contract:=…)")
        return resolve_path(contract), str(self.get_parameter("device").value)

    def _load_core(self, device: str) -> PolicyCore:
        run_dir = _paths.SIM2REAL / self.contract.run.dir
        try:
            return PolicyCore(self.contract, run_dir, device)
        except (ContractError, PolicyIOError, OSError, RuntimeError) as exc:
            raise PolicyNodeError(f"checkpoint ({run_dir}, {device}): {exc}") from exc

    # ---------------------------------------------------------------- callbacks (tiny)
    def _on_obs(self, msg) -> None:
        t0 = time.perf_counter()
        seq = int(msg.layout.data_offset)
        try:
            obs, seq = decode_obs(msg, self.segments)
            tick = self.stage.tick(obs, seq)
        except _TICK_ERRORS as exc:
            self.get_logger().warning(f"obs seq {seq}: {exc}", throttle_duration_sec=1.0)
            self._publish_status(self._error_status(seq, str(exc)), t0)
            return
        self._phase = "running"
        self._pub_action.publish(encode_action(tick.action, tick.seq))
        self._publish_status(tick.status, t0)

    def _on_episode(self, msg) -> None:
        try:
            body = decode_status(msg)
        except CodecError as exc:
            self.get_logger().warning(f"episode: {exc}")
            return
        if body.get("event") == "reset":
            episode = body.get("episode")
            self.stage.reset(episode=None if episode is None else int(episode))
            self._phase = "reset"
            self.get_logger().info(f"episode {episode} reset → policy state cleared (next obs must be seq 0)")

    # ---------------------------------------------------------------- helpers
    def _error_status(self, seq: int, reason: str) -> StageStatus:
        return StageStatus(node=PolicyStage.NODE, phase=self._phase, episode=self.stage.episode, seq=seq,
                           ok=False, reasons=(reason,), proc_ms=0.0)

    def _publish_status(self, status: StageStatus, t0: float) -> None:
        body = {**status.as_dict(), "stage_ms": status.proc_ms, "proc_ms": _ms(t0),
                "t_pub_ns": self.get_clock().now().nanoseconds}
        self._pub_status.publish(encode_status(body))


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
        node = PolicyNode()
    except PolicyNodeError as exc:
        print(f"policy_node: {exc}", file=sys.stderr)
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
