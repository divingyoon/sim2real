"""obs_node — rclpy 껍질 배선 테스트 (ros 마커, 격리 도메인, 인프로세스).

가짜 플랜트(/joint_states + /objects/cup_big_s100/pose 발행)와 프로브(obs/status/episode 구독,
episode 서비스 클라이언트)를 같은 executor 에 태워 obs 노드의 프로토콜을 확인한다:
라벨 = 계약 세그먼트, seq 0,1,2…, episode 이벤트 본문, 스테일 → ok:false·미발행, stop → 미발행,
액션 되먹임(seq 매칭). 좌 left_gripper_fake yaml + left_v2B25 계약. 하드웨어 없음.
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
ARM_SRC = [f"openarm_left_joint{i}" for i in range(1, 8)]
GRIP_SRC = "openarm_left_finger_joint1"
CUP_TOPIC = "/objects/cup_big_s100/pose"
CUP = (0.40, 0.20, 0.30)


# ------------------------------------------------------------------ harness
class _Spinner:
    """노드들을 한 executor 에 태워 백그라운드에서 돌린다."""

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


class _Plant:
    """가짜 센서 발행 + 프로브 구독 + 서비스 클라이언트 (테스트 스레드에서 호출)."""

    def __init__(self, context):
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger

        self.node = Node("fake_plant_probe", context=context)
        self.js_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.cup_pub = self.node.create_publisher(PoseStamped, CUP_TOPIC, 10)
        self.act_pub = self.node.create_publisher(Float64MultiArray, f"{NS}/action", 10)
        self.obs, self.status, self.episode = [], [], []
        self.node.create_subscription(Float64MultiArray, f"{NS}/obs", self.obs.append, 10)
        self.node.create_subscription(String, f"{NS}/status/obs", self._json(self.status), 10)
        ep_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(String, f"{NS}/episode", self._json(self.episode), ep_qos)
        self.clients = {n: self.node.create_client(Trigger, f"{NS}/episode/{n}")
                        for n in ("reset", "start", "stop", "abort")}

    @staticmethod
    def _json(sink: list):
        return lambda m: sink.append(json.loads(m.data))

    def feed(self, q, grip: float = 0.044, cup=CUP) -> None:
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        stamp = self.node.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = [*ARM_SRC, GRIP_SRC]
        js.position = [float(v) for v in q] + [float(grip)]
        js.velocity = [0.0] * 8
        self.js_pub.publish(js)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in cup)
        pose.pose.orientation.w = 1.0
        self.cup_pub.publish(pose)

    def feed_for(self, seconds: float, q, hz: float = 50.0, **kw) -> None:
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            self.feed(q, **kw)
            time.sleep(1.0 / hz)

    def call(self, name: str, timeout: float = 3.0) -> tuple[bool, list[str]]:
        from std_srvs.srv import Trigger

        cli = self.clients[name]
        assert cli.wait_for_service(timeout_sec=timeout), f"service {name} unavailable"
        fut = cli.call_async(Trigger.Request())
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        assert fut.done(), f"service {name} timeout"
        resp = fut.result()
        body = json.loads(resp.message)
        assert body["ok"] == resp.success
        return bool(resp.success), list(body["reasons"])

    def close(self):
        self.node.destroy_node()


@pytest.fixture
def contract():
    from policy_control import contract as C

    return C.load_contract(CONTRACT)


@pytest.fixture
def rig(ros, request):
    """(node, plant, spinner). policy_hz 는 request.param 로 덮어쓴다(기본 계약값)."""
    from rclpy.parameter import Parameter

    from policy_control.obs_node import ObsNode

    hz = getattr(request, "param", 0.0)
    overrides = [Parameter("contract", value=str(CONTRACT)), Parameter("robot", value=str(ROBOT)),
                 Parameter("policy_hz", value=float(hz))]
    node = ObsNode(context=ros, parameter_overrides=overrides)
    plant = _Plant(ros)
    spin = _Spinner(ros, node, plant.node)
    _wait(lambda: plant.node.count_subscribers("/joint_states") >= 1 and plant.node.count_publishers(f"{NS}/obs") >= 1)
    try:
        yield node, plant, spin
    finally:
        spin.close()
        plant.close()
        node.destroy_node()


def _wait(pred, timeout: float = 3.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _labels(msg) -> list[tuple[str, int]]:
    return [(d.label, d.size) for d in msg.layout.dim]


# ------------------------------------------------------------------ tests
def test_reset_start_publishes_obs_with_contract_labels_and_consecutive_seq(rig, contract):
    node, plant, _ = rig
    home = np.array(contract.pd.home_arm)
    plant.feed_for(0.2, home)
    ok, reasons = plant.call("reset")
    assert ok, reasons
    assert _wait(lambda: any(e["event"] == "reset" for e in plant.episode))
    ev = next(e for e in plant.episode if e["event"] == "reset")
    assert ev["episode"] == 1 and "t_ns" in ev and set(ev["home_q"]) >= set(contract.obs.joint_orders["arm"])
    np.testing.assert_allclose(ev["object_anchor"], CUP, atol=1e-9)
    assert not plant.obs, "start 전에는 obs 를 내면 안 된다"

    ok, reasons = plant.call("start")
    assert ok, reasons
    plant.feed_for(0.4, home)
    assert _wait(lambda: len(plant.obs) >= 5)
    want = [(s.name, s.dim) for s in contract.obs.segments]
    assert all(_labels(m) == want for m in plant.obs)
    seqs = [m.layout.data_offset for m in plant.obs]
    assert seqs == list(range(len(seqs)))
    assert all(len(m.data) == contract.policy.obs_dim for m in plant.obs)
    assert any(e["event"] == "start" and e["episode"] == 1 for e in plant.episode)
    st = [s for s in plant.status if s.get("ok")]
    assert st and st[-1]["node"] == "obs" and st[-1]["phase"] == "running" and st[-1]["episode"] == 1
    assert "t_pub_ns" in st[-1] and st[-1]["proc_ms"] >= 0.0 and st[-1]["seq"] == seqs[-1]


def test_stop_ends_publishing_and_reports_phase(rig, contract):
    node, plant, _ = rig
    home = np.array(contract.pd.home_arm)
    plant.feed_for(0.2, home)
    assert plant.call("reset")[0] and plant.call("start")[0]
    plant.feed_for(0.3, home)
    assert _wait(lambda: len(plant.obs) >= 3)
    ok, _ = plant.call("stop")
    assert ok
    n_before = len(plant.obs)
    time.sleep(0.1)
    n_settled = len(plant.obs)
    plant.feed_for(0.3, home)
    assert len(plant.obs) == n_settled and n_settled - n_before <= 1
    assert any(e["event"] == "stop" for e in plant.episode)
    assert _wait(lambda: plant.status and plant.status[-1]["phase"] == "stopped" and not plant.status[-1]["ok"])
    assert any("stopped" in r for r in plant.status[-1]["reasons"])


def test_stale_source_reports_not_ok_and_holds_obs(rig, contract):
    node, plant, _ = rig
    home = np.array(contract.pd.home_arm)
    plant.feed_for(0.2, home)
    assert plant.call("reset")[0] and plant.call("start")[0]
    plant.feed_for(0.3, home)
    assert _wait(lambda: len(plant.obs) >= 3)
    time.sleep(0.8)                      # stale_sec 0.5 초과 — 아무것도 안 보낸다
    n = len(plant.obs)
    last = plant.status[-1]
    assert last["ok"] is False and any("stale source" in r for r in last["reasons"])
    assert last["phase"] == "running" and last["seq"] > 0     # MLP: max_gap None → abort 없음, seq 는 계속
    time.sleep(0.2)
    assert len(plant.obs) == n
    plant.feed_for(0.3, home)            # 재개
    assert _wait(lambda: len(plant.obs) > n)
    assert plant.status[-1]["ok"] is True


def test_start_before_reset_and_reset_without_sources_are_refused(rig):
    node, plant, _ = rig
    ok, reasons = plant.call("start")
    assert not ok and reasons
    ok, reasons = plant.call("reset")
    assert not ok and any("missing" in r for r in reasons)
    assert not plant.obs and not any(e["event"] == "reset" for e in plant.episode)
    assert _wait(lambda: bool(plant.status)) and plant.status[-1]["ok"] is False
    assert plant.status[-1]["phase"] == "idle"


def test_abort_publishes_event_and_stops(rig, contract):
    node, plant, _ = rig
    home = np.array(contract.pd.home_arm)
    plant.feed_for(0.2, home)
    assert plant.call("reset")[0] and plant.call("start")[0]
    plant.feed_for(0.2, home)
    ok, _ = plant.call("abort")
    assert ok
    assert _wait(lambda: any(e["event"] == "abort" and e["reasons"] for e in plant.episode))
    n = len(plant.obs)
    plant.feed_for(0.2, home)
    assert len(plant.obs) <= n + 1
    assert plant.status[-1]["phase"] == "aborted"


@pytest.mark.parametrize("rig", [5.0], indirect=True)
def test_action_feedback_matched_by_seq(rig, contract):
    """policy_hz 5 Hz 로 늦춰 액션이 다음 tick 전에 도착하게 한다: obs[k+1].actions == action(seq k)."""
    from policy_control import codec
    from policy_control.obs_core import split_segments

    node, plant, _ = rig
    home = np.array(contract.pd.home_arm)
    plant.feed_for(0.3, home, hz=20)
    assert plant.call("reset")[0] and plant.call("start")[0]
    assert _wait(lambda: len(plant.obs) >= 1, timeout=2.0)
    k = plant.obs[-1].layout.data_offset
    action = np.linspace(-1.0, 1.0, contract.policy.action_dim)
    plant.act_pub.publish(codec.encode_action(action, k))
    plant.feed_for(0.5, home, hz=20)
    assert _wait(lambda: any(m.layout.data_offset == k + 2 for m in plant.obs), timeout=2.0)
    by_seq = {m.layout.data_offset: np.asarray(m.data) for m in plant.obs}
    np.testing.assert_allclose(split_segments(by_seq[k + 1], contract)["actions"], action)
    np.testing.assert_allclose(split_segments(by_seq[k], contract)["actions"], 0.0)
    # seq k+2 에는 새 액션이 없다 — ObsCore 는 직전 last_action 을 유지한다
    np.testing.assert_allclose(split_segments(by_seq[k + 2], contract)["actions"], action)
    st = [s for s in plant.status if s.get("seq") == k + 1]
    assert st and st[-1]["action_seq"] == k and st[-1]["action_matched"] is True


def test_right_contract_without_fk_provider_is_refused(ros):
    from rclpy.parameter import Parameter

    from policy_control.obs_node import ObsNode, ObsNodeError

    right = SIM2REAL / "logs" / "policy" / "right_g1" / "deploy_contract.json"
    if not right.exists():
        pytest.skip("우 계약 없음")
    overrides = [Parameter("contract", value=str(right)),
                 Parameter("robot", value=str(ROBOT.with_name("right_dg5f_fake.yaml")))]
    with pytest.raises(ObsNodeError, match="fk"):
        ObsNode(context=ros, parameter_overrides=overrides)


def test_missing_contract_parameter_is_a_clear_error(ros):
    from policy_control.obs_node import ObsNode, ObsNodeError

    with pytest.raises(ObsNodeError, match="contract"):
        ObsNode(context=ros)


# ------------------------------------------------------------------ 09.06 side 파라미터 / 양팔 yaml / control-only 거부
DG5FM_CONTRACT = SIM2REAL / "logs/policy/right_g1/deploy_contract.dg5f-m.json"
ASSET_CONTRACT = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
BI_ROBOT = ROBOT.with_name("dg5f_m_bi_fake.yaml")
RIGHT_ARM_SRC = [f"openarm_right_joint{i}" for i in range(1, 8)]
RIGHT_HAND_SRC = [f"rj_dg_{f}_{i}" for f in range(1, 6) for i in range(1, 5)]
needs_dg5fm = pytest.mark.skipif(not (DG5FM_CONTRACT.exists() and ASSET_CONTRACT.exists()), reason="dg5f-m 계약 없음")


def _overrides(contract, robot, side: str = "", hz: float = 0.0):
    from rclpy.parameter import Parameter

    return [Parameter("contract", value=str(contract)), Parameter("robot", value=str(robot)),
            Parameter("side", value=side), Parameter("policy_hz", value=float(hz))]


@needs_dg5fm
def test_control_only_contract_is_refused_with_a_clear_error(ros):
    from policy_control.obs_node import ObsNode, ObsNodeError

    with pytest.raises(ObsNodeError, match="control-only"):
        ObsNode(context=ros, parameter_overrides=_overrides(ASSET_CONTRACT, BI_ROBOT, "right"))


@needs_dg5fm
def test_side_the_yaml_does_not_have_is_refused(ros):
    from policy_control.obs_node import ObsNode, ObsNodeError

    # 한 팔(우) yaml 에 좌팔 요청
    with pytest.raises(ObsNodeError, match="side 'left'"):
        ObsNode(context=ros, parameter_overrides=_overrides(DG5FM_CONTRACT, ROBOT.with_name("dg5f_m_right_fake.yaml"), "left"))
    # 좌 그리퍼 계약을 DG-5F 양팔 yaml 에 — 손 관절이 없다
    with pytest.raises(ObsNodeError, match="obs core"):
        ObsNode(context=ros, parameter_overrides=_overrides(CONTRACT, BI_ROBOT, "left"))


def _urdf_chain_contract(tmp_path: Path) -> Path:
    """dg5f-m 우 g1 계약을 fk.kind=urdf_chain 으로 (노드가 자산 URDF 로 직접 FK)."""
    raw = json.loads(DG5FM_CONTRACT.read_text())
    raw["obs"]["fk"] = {**raw["obs"]["fk"], "kind": "urdf_chain"}
    p = tmp_path / "deploy_contract.urdf_chain.json"
    p.write_text(json.dumps(raw))
    return p


class _RightPlant(_Plant):
    """양팔 yaml 의 우팔 소스 4종을 먹인다: /joint_states(양팔 이름), /dg5f_right/joint_states, tip_forces, cup."""

    def __init__(self, context):
        super().__init__(context)
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        self.hand_pub = self.node.create_publisher(JointState, "/dg5f_right/joint_states", 10)
        self.tip_pub = self.node.create_publisher(Float64MultiArray, "/dg5f_right/tip_forces_xyz", 10)

    def feed(self, q, hand=None, cup=(0.36, -0.16, 0.30)) -> None:
        from policy_control import codec

        hand = np.zeros(20) if hand is None else np.asarray(hand, dtype=float)
        names = [*RIGHT_ARM_SRC, *ARM_SRC]
        self.js_pub.publish(codec.encode_joint_state(names, [*q, *np.zeros(7)], velocity=np.zeros(14)))
        self.hand_pub.publish(codec.encode_joint_state(RIGHT_HAND_SRC, hand, velocity=np.zeros(20)))
        self.tip_pub.publish(codec.encode_float_array(np.zeros(15), ("tip", "axis"), (5, 3), seq=0))
        self.cup_pub.publish(codec.encode_pose(cup, (1.0, 0.0, 0.0, 0.0), "base_link"))


@needs_dg5fm
def test_right_side_of_bimanual_yaml_publishes_155d_obs_with_urdf_chain_fk(ros, tmp_path):
    from policy_control import contract as C
    from policy_control.obs_core import split_segments
    from policy_control.obs_node import ObsNode
    from policy_control import fk_numpy

    contract_path = _urdf_chain_contract(tmp_path)
    contract = C.load_contract(contract_path)
    node = ObsNode(context=ros, parameter_overrides=_overrides(contract_path, BI_ROBOT, "right"))
    plant = _RightPlant(ros)
    spin = _Spinner(ros, node, plant.node)
    try:
        assert node.side == "right"
        assert set(node.robot_cfg.sources) == {"arm", "ee", "tip_force", "decoder_target", "object", "head"}
        assert isinstance(node.fk, fk_numpy.UrdfChainFK) and node.fk.palm_body == "r_hl_palm"
        _wait(lambda: plant.node.count_subscribers("/dg5f_right/joint_states") >= 1)
        home = np.array(contract.pd.home_arm)
        open_pose = np.array(contract.action.hand.params["open_pose"])
        plant.feed_for(0.3, home, hand=open_pose)
        ok, reasons = plant.call("reset")
        assert ok, reasons
        ok, reasons = plant.call("start")
        assert ok, reasons
        plant.feed_for(0.4, home, hand=open_pose)
        assert _wait(lambda: len(plant.obs) >= 3)
        want = [(s.name, s.dim) for s in contract.obs.segments]
        assert all(_labels(m) == want for m in plant.obs) and all(len(m.data) == 155 for m in plant.obs)
        seg = split_segments(np.asarray(plant.obs[-1].data), contract)
        pose = node.fk.palm_pose(home, open_pose)
        np.testing.assert_allclose(seg["palm_pos"], pose.palm_pos, atol=1e-9)
        np.testing.assert_allclose(seg["arm_q"], home, atol=1e-9)
        np.testing.assert_allclose(seg["joint_err"], 0.0, atol=1e-9)
        assert plant.status and plant.status[-1]["ok"] is True
    finally:
        spin.close()
        plant.close()
        node.destroy_node()
