"""policy_node — rclpy 껍질 배선 테스트 (ros 마커, 격리 도메인, 인프로세스).

체크포인트를 적재하지 않는다: `PolicyCore.with_backend(FakeBackend)` 를 생성자 인자 `core` 로
주입한다(seq 규칙은 진짜 PolicyCore 것). 프로브가 /policy_control/obs 를 발행하고 action/status 를
받는다. 확인: seq 사본, seq 0 리셋, episode 'reset' 이벤트 리셋, 미시작 seq>0 거부, 라벨 불일치
obs → status ok:false 후 생존.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ros

SIM2REAL = Path(__file__).resolve().parents[2]
CONTRACT = SIM2REAL / "logs" / "policy" / "left_v2B25" / "deploy_contract.json"
ROBOT = SIM2REAL / "policy_control" / "config" / "robots" / "left_gripper_fake.yaml"
NS = "/policy_control"


class FakeBackend:
    """forward = obs 앞 action_dim 칸(결정론), reset 횟수를 센다."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.resets = 0
        self.forwards = 0

    def forward(self, obs: np.ndarray) -> np.ndarray:
        self.forwards += 1
        return np.asarray(obs[: self.action_dim], dtype=np.float32).copy()

    def reset(self) -> None:
        self.resets += 1


class _Spinner:
    def __init__(self, context, *nodes):
        from rclpy.executors import SingleThreadedExecutor

        self.exec = SingleThreadedExecutor(context=context)
        for n in nodes:
            self.exec.add_node(n)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            self.exec.spin_once(timeout_sec=0.02)

    def close(self):
        self._stop.set()
        self._th.join(timeout=2.0)
        self.exec.shutdown()


class _Probe:
    def __init__(self, context, segments):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float64MultiArray, String

        self.segments = segments
        self.node = Node("policy_probe", context=context)
        self.obs_pub = self.node.create_publisher(Float64MultiArray, f"{NS}/obs", 10)
        ep_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ep_pub = self.node.create_publisher(String, f"{NS}/episode", ep_qos)
        self.actions, self.status = [], []
        self.node.create_subscription(Float64MultiArray, f"{NS}/action", self.actions.append, 10)
        self.node.create_subscription(String, f"{NS}/status/policy",
                                      lambda m: self.status.append(json.loads(m.data)), 10)

    def send_obs(self, seq: int, scale: float = 1.0):
        from policy_control import codec

        n = sum(d for _, d in self.segments)
        obs = np.arange(n, dtype=np.float64) * 0.01 * scale
        self.obs_pub.publish(codec.encode_obs(obs, self.segments, seq))
        return obs

    def send_episode(self, event: str, episode: int):
        from std_msgs.msg import String

        self.ep_pub.publish(String(data=json.dumps({"episode": episode, "event": event, "t_ns": 0})))

    def close(self):
        self.node.destroy_node()


def _wait(pred, timeout: float = 3.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def contract():
    from policy_control import contract as C

    return C.load_contract(CONTRACT)


@pytest.fixture
def rig(ros, contract):
    from rclpy.parameter import Parameter

    from policy_control.policy_core import PolicyCore
    from policy_control.policy_node import PolicyNode

    backend = FakeBackend(contract.policy.action_dim)
    core = PolicyCore.with_backend(backend, obs_dim=contract.policy.obs_dim, action_dim=contract.policy.action_dim,
                                   action_clip=contract.policy.action_clip, contract=contract)
    overrides = [Parameter("contract", value=str(CONTRACT)), Parameter("robot", value=str(ROBOT)),
                 Parameter("device", value="cpu")]
    node = PolicyNode(context=ros, parameter_overrides=overrides, core=core)
    probe = _Probe(ros, [(s.name, s.dim) for s in contract.obs.segments])
    spin = _Spinner(ros, node, probe.node)
    _wait(lambda: probe.node.count_subscribers(f"{NS}/obs") >= 1 and probe.node.count_subscribers(f"{NS}/episode") >= 1
          and probe.node.count_publishers(f"{NS}/action") >= 1)
    try:
        yield node, probe, backend
    finally:
        spin.close()
        probe.close()
        node.destroy_node()


# ------------------------------------------------------------------ tests
def test_action_copies_seq_and_seq0_resets(rig, contract):
    node, probe, backend = rig
    obs0 = probe.send_obs(0)
    assert _wait(lambda: len(probe.actions) >= 1)
    a0 = probe.actions[0]
    assert a0.layout.data_offset == 0 and [d.label for d in a0.layout.dim] == ["action"]
    np.testing.assert_allclose(a0.data, obs0[: contract.policy.action_dim], atol=1e-6)
    assert backend.resets == 1 and backend.forwards == 1

    probe.send_obs(1, scale=2.0)
    assert _wait(lambda: len(probe.actions) >= 2)
    assert probe.actions[1].layout.data_offset == 1 and backend.forwards == 2
    probe.send_obs(1, scale=3.0)                    # 중복 seq → forward 없이 직전 액션 사본
    assert _wait(lambda: len(probe.actions) >= 3)
    assert backend.forwards == 2
    np.testing.assert_allclose(probe.actions[2].data, probe.actions[1].data)
    probe.send_obs(0)                               # 반복된 seq 0 → 재리셋
    assert _wait(lambda: backend.resets == 2)
    st = [s for s in probe.status if s.get("ok")]
    assert st and st[-1]["node"] == "policy" and "t_pub_ns" in st[-1] and st[-1]["proc_ms"] >= 0.0


def test_episode_reset_event_resets_and_unstarted_seq_is_refused(rig, contract):
    node, probe, backend = rig
    probe.send_obs(0)
    probe.send_obs(1)
    assert _wait(lambda: len(probe.actions) >= 2)
    probe.send_episode("reset", 7)
    assert _wait(lambda: backend.resets == 2)
    probe.send_obs(2)                               # 리셋 뒤 seq 0 없이 → SeqError → ok:false, 액션 없음
    assert _wait(lambda: probe.status and probe.status[-1]["ok"] is False and probe.status[-1]["seq"] == 2)
    assert any("seq 0" in r for r in probe.status[-1]["reasons"])
    assert len(probe.actions) == 2
    probe.send_obs(0)
    assert _wait(lambda: len(probe.actions) >= 3)
    assert probe.actions[-1].layout.data_offset == 0 and probe.status[-1]["episode"] == 7


def test_malformed_obs_reports_not_ok_and_node_survives(rig, contract):
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

    node, probe, backend = rig
    bad = Float64MultiArray(layout=MultiArrayLayout(dim=[MultiArrayDimension(label="junk", size=3, stride=0)],
                                                    data_offset=0), data=[1.0, 2.0, 3.0])
    probe.obs_pub.publish(bad)
    assert _wait(lambda: probe.status and probe.status[-1]["ok"] is False)
    assert any("레이아웃" in r or "layout" in r for r in probe.status[-1]["reasons"])
    assert not probe.actions and backend.forwards == 0
    probe.send_obs(0)
    assert _wait(lambda: len(probe.actions) >= 1)
    assert probe.status[-1]["ok"] is True


def test_missing_contract_parameter_is_a_clear_error(ros):
    from policy_control.policy_node import PolicyNode, PolicyNodeError

    with pytest.raises(PolicyNodeError, match="contract"):
        PolicyNode(context=ros, core=object())
