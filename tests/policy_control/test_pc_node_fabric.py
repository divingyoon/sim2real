"""fabric_node — rclpy 껍질 배선 테스트 (ros 마커, 격리 도메인, 가짜 fabric backend, CUDA 불요).

노드 밖에서 /joint_states · /objects/<name>/pose · /policy_control/{episode,obs,action} 을 흘리고
/policy_control/{joint_target,palm_target,status/fabric} 과 episode/abort 서비스 호출을 확인한다.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.ros

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
LEFT_ROBOT = SIM2REAL / "policy_control/config/robots/left_gripper_fake.yaml"
needs_left = pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")

NS = "/policy_control"
ARM_SRC = [f"openarm_left_joint{i}" for i in range(1, 8)]
ARM_CAN = [f"l_aj_{i}" for i in range(1, 8)]
GRIP_SRC = "openarm_left_finger_joint1"
OBJECT_TOPIC = "/objects/cup_big_s100/pose"


# ------------------------------------------------------------------ fakes
class FakeFabric:
    def __init__(self, num_joints: int) -> None:
        self.num_joints = num_joints
        self.default_config = torch.zeros(1, num_joints)

    def set_features(self, hand, palm, convention, q, qd, obj_ids, obj_ind, damping):
        pass

    def get_palm_pose(self, q, convention):
        return torch.cat([q[:, :3], q[:, :3] * 0.5], dim=1)

    def get_fingertip_positions(self, q):
        return q[:, :5].unsqueeze(-1).repeat(1, 1, 3)


class FakeIntegrator:
    def step(self, q, qd, qdd, dt):
        return q + 0.001, torch.ones_like(qd) * 0.1, qdd


def _fake_fabric_core(contract):
    from policy_control import fabric_core as F

    n = len(contract.fabric.joint_order)
    be = F.FabricBackend(fabric=FakeFabric(n), integrator=FakeIntegrator(), object_ids=None,
                         object_indicator=None, device="cpu")
    return F.FabricCore(contract, "cpu", backend=be)


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


class Peer:
    """노드 상대편: 센서/상류 발행자 + 하류 프로브 + episode/abort 서비스 서버."""

    def __init__(self, context, contract):
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger

        self.contract = contract
        self.node = Node("fabric_peer", context=context)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_js = self.node.create_publisher(JointState, "/joint_states", 10)
        self.pub_obj = self.node.create_publisher(PoseStamped, OBJECT_TOPIC, 10)
        self.pub_ep = self.node.create_publisher(String, f"{NS}/episode", latched)
        self.pub_obs = self.node.create_publisher(Float64MultiArray, f"{NS}/obs", 10)
        self.pub_act = self.node.create_publisher(Float64MultiArray, f"{NS}/action", 10)
        self.got = {"joint_target": [], "palm_target": [], "status": []}
        for key, typ, topic in (("joint_target", JointState, f"{NS}/joint_target"),
                                ("palm_target", PoseStamped, f"{NS}/palm_target"),
                                ("status", String, f"{NS}/status/fabric")):
            self.node.create_subscription(typ, topic, (lambda k: lambda m: self.got[k].append(m))(key), 10)
        self.aborts = []
        self.node.create_service(Trigger, f"{NS}/episode/abort", self._abort)
        self.q = np.asarray(contract.pd.home_arm, dtype=float)
        self.grip = 0.044

    def _abort(self, req, res):
        self.aborts.append(time.monotonic())
        res.success = True
        res.message = json.dumps({"ok": True, "reasons": []})
        return res

    def sensors(self):
        from policy_control import codec

        js = codec.encode_joint_state(ARM_SRC + [GRIP_SRC], np.concatenate([self.q, [self.grip]]),
                                      velocity=np.zeros(8), stamp=time.time())
        self.pub_js.publish(js)
        self.pub_obj.publish(codec.encode_pose([0.40, -0.16, 0.30], [1.0, 0.0, 0.0, 0.0], "base_link",
                                               stamp=time.time()))

    def episode(self, event: str, episode: int = 1, anchor=(0.40, -0.16, 0.30)):
        from std_msgs.msg import String

        home = {**dict(zip(ARM_CAN, self.contract.pd.home_arm)), **self.contract.pd.home_hand}
        body = {"episode": episode, "event": event, "object_anchor": list(anchor) if event == "reset" else None,
                "home_q": home, "reasons": [], "t_ns": time.time_ns()}
        self.pub_ep.publish(String(data=json.dumps(body)))

    def obs(self, seq: int, gate_open: bool, object_pos=(0.40, -0.16, 0.30)):
        from policy_control import codec

        segs = [(s.name, s.dim) for s in self.contract.obs.segments]
        vec = np.zeros(sum(d for _, d in segs))
        off = 0
        for name, dim in segs:
            if name == "gripper_gate":
                vec[off] = 1.0 if gate_open else 0.0
            if name == "object_position":
                vec[off:off + 3] = object_pos
            off += dim
        self.pub_obs.publish(codec.encode_obs(vec, segs, seq))

    def action(self, seq: int, gripper: float = -0.9, dim: int | None = None):
        from policy_control import codec

        n = self.contract.policy.action_dim if dim is None else dim
        a = np.zeros(n)
        if n >= 7:
            a[6] = gripper
        time.sleep(0.005)                       # 실제 체인의 정책 지연(obs → action) 흉내
        self.pub_act.publish(codec.encode_action(a, seq))

    def wait(self, key: str, n: int, timeout: float = 3.0):
        t0 = time.monotonic()
        while len(self.got[key]) < n and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        return list(self.got[key])

    def statuses(self):
        return [json.loads(m.data) for m in self.got["status"]]

    def last_status(self, timeout: float = 3.0, min_count: int = 1):
        self.wait("status", min_count, timeout)
        return self.statuses()[-1]

    def mark(self) -> int:
        return len(self.got["status"])

    def status_after(self, mark: int, timeout: float = 3.0):
        """mark 이후 새 status 한 건(이전 status 를 답으로 오인하지 않게)."""
        got = self.wait("status", mark + 1, timeout)
        assert len(got) > mark, "status 미수신"
        return json.loads(got[mark].data)

    def close(self):
        self.node.destroy_node()


def _wait_discovery(node, topic: str, min_subs: int = 1, timeout: float = 3.0):
    t0 = time.monotonic()
    while node.count_subscribers(topic) < min_subs and time.monotonic() - t0 < timeout:
        time.sleep(0.02)


@pytest.fixture(scope="module")
def left():
    from policy_control import contract as C

    return C.load_contract(LEFT_JSON)


def _make_node(context, contract, robot_yaml: Path):
    from rclpy.parameter import Parameter

    from policy_control.fabric_node import FabricNode

    overrides = [Parameter("contract", value=str(LEFT_JSON)), Parameter("robot", value=str(robot_yaml)),
                 Parameter("device", value="cpu")]
    return FabricNode(context=context, fabric=_fake_fabric_core(contract), parameter_overrides=overrides)


@pytest.fixture
def rig(ros, left):
    """fabric 노드(가짜 fabric) + 상대편, 백그라운드 spin."""
    node = _make_node(ros, left, LEFT_ROBOT)
    peer = Peer(ros, left)
    spin = _Spinner(ros, node, peer.node)
    for t in ("/joint_states", OBJECT_TOPIC, f"{NS}/obs", f"{NS}/action", f"{NS}/episode"):
        _wait_discovery(peer.node, t)
    try:
        yield node, peer
    finally:
        spin.close()
        peer.close()
        node.destroy_node()


def _start_episode(peer, episode=1):
    """reset 이벤트 → 노드의 reset status 수신까지 기다린다(디스커버리 완료의 증거)."""
    n0 = len(peer.got["status"])
    for _ in range(3):
        peer.sensors()
        time.sleep(0.02)
    peer.episode("reset", episode)
    st = peer.last_status(min_count=n0 + 1)
    assert st["ok"] is True and st["episode"] == episode, st
    peer.sensors()
    time.sleep(0.05)


# ------------------------------------------------------------------ tests
@needs_left
def test_action_before_reset_reports_not_running_and_publishes_nothing(rig):
    node, peer = rig
    peer.sensors()
    m = peer.mark()
    peer.obs(0, True)
    peer.action(0)
    st = peer.status_after(m)
    assert st["node"] == "fabric" and st["ok"] is False
    assert any("episode" in r for r in st["reasons"])
    assert peer.got["joint_target"] == []


@needs_left
def test_reset_then_action_publishes_joint_and_palm_targets(rig, left):
    from policy_control import codec

    node, peer = rig
    _start_episode(peer, episode=1)
    m = peer.mark()
    peer.obs(0, True)
    peer.action(0, gripper=-0.9)
    jt = peer.wait("joint_target", 1)
    assert [j.header.frame_id for j in jt] == ["1:0"]
    tgt = codec.decode_joint_target(jt[0], ARM_CAN + ["l_hj_gripper_1"])
    assert tgt.episode == "1" and tgt.seq == 0
    assert tuple(jt[0].name) == tuple(ARM_CAN) + ("l_hj_gripper_1",)
    assert tgt.position[7] == pytest.approx(left.action.hand.params["close"])
    assert jt[0].velocity[0] == pytest.approx(0.1 * left.fabric.vel_ff_scale)
    pt = peer.wait("palm_target", 1)
    assert pt[0].header.frame_id == "base_link"
    assert abs(np.linalg.norm([pt[0].pose.orientation.w, pt[0].pose.orientation.x, pt[0].pose.orientation.y,
                               pt[0].pose.orientation.z]) - 1.0) < 1e-6
    st = peer.status_after(m)
    assert st["ok"] is True and st["seq"] == 0 and st["episode"] == 1 and st["phase"] == "running"
    assert st["proc_ms"] >= 0.0 and "t_pub_ns" in st
    # 다음 스텝: seq 1, frame_id 는 '1:1'
    peer.sensors()
    peer.obs(1, True)
    peer.action(1, gripper=-0.9)
    jt = peer.wait("joint_target", 2)
    assert jt[1].header.frame_id == "1:1"


@needs_left
def test_gate_closed_forces_gripper_open(rig, left):
    node, peer = rig
    _start_episode(peer)
    peer.obs(0, False)
    peer.action(0, gripper=-0.9)
    jt = peer.wait("joint_target", 1)
    assert jt[0].position[7] == pytest.approx(left.action.hand.params["open"])


@needs_left
def test_missing_obs_for_seq_is_reported_not_crashed(rig):
    node, peer = rig
    _start_episode(peer)
    m = peer.mark()
    peer.action(5)                                    # obs seq 5 없음
    st = peer.status_after(m)
    assert st["ok"] is False and any("obs" in r and "5" in r for r in st["reasons"])
    assert peer.got["joint_target"] == []
    peer.obs(6, True)
    peer.action(6)
    assert len(peer.wait("joint_target", 1)) == 1     # 노드는 살아 있다


@needs_left
def test_bad_action_layout_is_reported_not_crashed(rig):
    node, peer = rig
    _start_episode(peer)
    m = peer.mark()
    peer.obs(0, True)
    peer.action(0, dim=3)
    st = peer.status_after(m)
    assert st["ok"] is False and any("action" in r for r in st["reasons"])
    assert peer.got["joint_target"] == []
    peer.action(0)
    assert len(peer.wait("joint_target", 1)) == 1


@needs_left
def test_stop_event_disarms_until_next_reset(rig):
    node, peer = rig
    _start_episode(peer)
    peer.obs(0, True)
    peer.action(0)
    assert len(peer.wait("joint_target", 1)) == 1
    m = peer.mark()
    peer.episode("stop", 1)
    peer.status_after(m)
    m = peer.mark()
    peer.obs(1, True)
    peer.action(1)
    st = peer.status_after(m)
    assert st["ok"] is False and peer.got["joint_target"][-1].header.frame_id == "1:0"
    _start_episode(peer, episode=2)
    peer.obs(0, True)
    peer.action(0)
    jt = peer.wait("joint_target", 2)
    assert jt[-1].header.frame_id == "2:0"


@needs_left
def test_clearance_violation_calls_abort_and_stops_targets(ros, left, tmp_path):
    raw = yaml.safe_load(LEFT_ROBOT.read_text())
    raw["table"]["clearance_min"] = 10.0                          # 어떤 자세도 위반
    raw["table"].pop("center_xy", None)                           # 판 xy 범위 없음 → 어디서나 검사
    raw["table"].pop("size_xy", None)
    robot_yaml = tmp_path / "left_huge_clearance.yaml"
    robot_yaml.write_text(yaml.safe_dump(raw))
    node = _make_node(ros, left, robot_yaml)
    peer = Peer(ros, left)
    spin = _Spinner(ros, node, peer.node)
    try:
        for t in ("/joint_states", OBJECT_TOPIC, f"{NS}/obs", f"{NS}/action", f"{NS}/episode"):
            _wait_discovery(peer.node, t)
        _start_episode(peer)
        m = peer.mark()
        peer.obs(0, True)
        peer.action(0)
        st = peer.status_after(m)
        assert st["ok"] is False and any("clearance" in r for r in st["reasons"])
        t0 = time.monotonic()
        while not peer.aborts and time.monotonic() - t0 < 3.0:
            time.sleep(0.02)
        assert len(peer.aborts) == 1
        assert peer.got["joint_target"] == []
        m = peer.mark()
        peer.obs(1, True)
        peer.action(1)                                             # abort 뒤 → 발행 없음, abort 재호출 없음
        st = peer.status_after(m)
        assert st["ok"] is False and peer.got["joint_target"] == []
        time.sleep(0.2)
        assert len(peer.aborts) == 1
    finally:
        spin.close()
        peer.close()
        node.destroy_node()


@needs_left
def test_stale_sensors_are_reported(rig):
    node, peer = rig
    _start_episode(peer)
    time.sleep(0.7)                                   # arm/ee stale_sec 0.5 초과
    m = peer.mark()
    peer.obs(0, True)
    peer.action(0)
    st = peer.status_after(m)
    assert st["ok"] is False and any("stale" in r for r in st["reasons"])
    assert peer.got["joint_target"] == []


# ================================================================== control-only(direct) 모드 · side 파라미터 · palm_cmd 도구
import contextlib
import importlib.util
import os

from policy_control import fabric_core as F
from policy_control import sources
from policy_control.fabric_node import palm6_to_quat

ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
BI_ROBOT = SIM2REAL / "policy_control/config/robots/dg5f_m_bi_fake.yaml"
LEFT_DG5F_ROBOT = SIM2REAL / "policy_control/config/robots/dg5f_m_left_fake.yaml"
PALM_CMD_TOOL = SIM2REAL / "policy_control/tools/palm_cmd.py"
needs_asset = pytest.mark.skipif(not ASSET_JSON.exists(), reason="asset_openarm_dg5f-m_bi_rl contract 없음")
SIDES = ("left", "right")
PUMP_PERIOD = 0.1


@pytest.fixture(scope="module")
def asset():
    from policy_control import contract as C

    return C.load_contract(ASSET_JSON)


@pytest.fixture(scope="module")
def bi_robot_yaml(tmp_path_factory) -> Path:
    """양팔 fake yaml 사본 — 가짜 fabric 의 FK(손끝 z = q[:5] ≈ 0)가 판 아래라 실제 table 값이면 가드가 abort 한다.
    여기서는 배선만 보므로 판을 −10 m 로 내려 가드를 무력화한다(가드 자체는 test_clearance_violation… 이 본다)."""
    raw = yaml.safe_load(BI_ROBOT.read_text())
    raw["table"] = {"top": -10.0, "clearance_min": 0.03}
    path = tmp_path_factory.mktemp("robots") / "dg5f_m_bi_fake_no_table.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _fake_fabric_core_side(contract, side):
    n = len(contract.side(side).fabric.joint_order)
    be = F.FabricBackend(fabric=FakeFabric(n), integrator=FakeIntegrator(), object_ids=None,
                         object_indicator=None, device="cpu")
    return F.FabricCore(contract, "cpu", side=side, backend=be)


class DirectPeer(Peer):
    """control-only 상대편: 양팔 센서(/joint_states 14관절·/dg5f_<side>/joint_states 20관절, 프로필 source 이름) +
    물체 + episode + palm_cmd/hand_cmd 발행, joint_target/palm_target/palm_pose/status 프로브."""

    def __init__(self, context, contract, robot_cfg):
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.contract = contract
        profile = sources.load_profile(robot_cfg.joint_profiles)
        self.arm_src = {s: [profile[j]["source"] for j in contract.side(s).arm_joints] for s in SIDES}
        self.hand_src = {s: [profile[j]["source"] for j in contract.side(s).hand_joints] for s in SIDES}
        self.hand_home = {s: np.array([contract.side(s).home_hand[j] for j in contract.side(s).hand_joints]) for s in SIDES}
        self.node = Node("direct_peer", context=context)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_js = self.node.create_publisher(JointState, "/joint_states", 10)
        self.pub_hand = {s: self.node.create_publisher(JointState, f"/dg5f_{s}/joint_states", 10) for s in SIDES}
        self.pub_obj = self.node.create_publisher(PoseStamped, OBJECT_TOPIC, 10)
        self.pub_ep = self.node.create_publisher(String, f"{NS}/episode", latched)
        self.pub_palm_cmd = self.node.create_publisher(PoseStamped, f"{NS}/palm_cmd", 10)
        self.pub_hand_cmd = self.node.create_publisher(JointState, f"{NS}/hand_cmd", 10)
        self.got = {"joint_target": [], "palm_target": [], "palm_pose": [], "status": []}
        for key, typ, topic, qos in (("joint_target", JointState, f"{NS}/joint_target", 10),
                                     ("palm_target", PoseStamped, f"{NS}/palm_target", 10),
                                     ("palm_pose", PoseStamped, f"{NS}/palm_pose", latched),
                                     ("status", String, f"{NS}/status/fabric", 10)):
            self.node.create_subscription(typ, topic, (lambda k: lambda m: self.got[k].append(m))(key), qos)

    def sensors(self):
        from policy_control import codec

        now = time.time()
        names = self.arm_src["left"] + self.arm_src["right"]
        self.pub_js.publish(codec.encode_joint_state(names, np.zeros(14), velocity=np.zeros(14), stamp=now))
        for s in SIDES:
            self.pub_hand[s].publish(codec.encode_joint_state(self.hand_src[s], self.hand_home[s],
                                                              velocity=np.zeros(20), stamp=now))
        self.pub_obj.publish(codec.encode_pose([0.40, -0.16, 0.30], [1.0, 0.0, 0.0, 0.0], "base_link", stamp=now))

    def episode(self, event: str, episode: int = 1):
        from std_msgs.msg import String

        body = {"episode": episode, "event": event, "object_anchor": None, "home_q": {}, "reasons": [],
                "t_ns": time.time_ns()}
        self.pub_ep.publish(String(data=json.dumps(body)))

    def palm_cmd(self, palm6, frame: str = "base_link"):
        from policy_control import codec

        p = np.asarray(palm6, dtype=float)
        self.pub_palm_cmd.publish(codec.encode_pose(p[:3], palm6_to_quat(p), frame, stamp=time.time()))

    def hand_cmd(self, names, values):
        from policy_control import codec

        self.pub_hand_cmd.publish(codec.encode_joint_state(list(names), np.asarray(values, dtype=float), stamp=time.time()))

    def pose_xyz(self, key: str, index: int = -1) -> np.ndarray:
        p = self.got[key][index].pose.position
        return np.array([p.x, p.y, p.z])


def _pump(peer, seconds: float):
    """센서를 stale_sec(0.5) 안에 계속 흘리며 기다린다(direct 타이머는 매 tick 실측을 요구한다)."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        peer.sensors()
        time.sleep(PUMP_PERIOD)


def _pump_until(peer, stop: threading.Event):
    while not stop.is_set():
        peer.sensors()
        time.sleep(PUMP_PERIOD)


def _direct_node(context, contract, robot_yaml: Path, side: str):
    from rclpy.parameter import Parameter

    from policy_control.fabric_node import FabricNode

    overrides = [Parameter("contract", value=str(ASSET_JSON)), Parameter("robot", value=str(robot_yaml)),
                 Parameter("device", value="cpu"), Parameter("side", value=side)]
    return FabricNode(context=context, fabric=_fake_fabric_core_side(contract, side), parameter_overrides=overrides)


@contextlib.contextmanager
def _direct_rig(ros, asset, side: str, robot_yaml: Path):
    node = _direct_node(ros, asset, robot_yaml, side)
    peer = DirectPeer(ros, asset, sources.load_robot_cfg(robot_yaml))
    spin = _Spinner(ros, node, peer.node)
    try:
        for t in ("/joint_states", f"/dg5f_{side}/joint_states", OBJECT_TOPIC, f"{NS}/episode", f"{NS}/palm_cmd",
                  f"{NS}/hand_cmd"):
            _wait_discovery(peer.node, t)
        yield node, peer
    finally:
        spin.close()
        peer.close()
        node.destroy_node()


def _reset_direct(peer, episode: int = 1) -> dict:
    n0 = len(peer.got["status"])
    _pump(peer, 0.15)
    peer.episode("reset", episode)
    st = peer.last_status(min_count=n0 + 1)
    assert st["ok"] is True and st["armed"] is True and st["running"] is False and st["episode"] == episode, st
    return st


@needs_asset
@pytest.mark.parametrize("side", list(SIDES))
def test_control_only_palm_cmd_and_hand_cmd_drive_joint_target_for_side(ros, asset, bi_robot_yaml, side):
    s = asset.side(side)
    hand_joints = tuple(s.hand_joints)
    with _direct_rig(ros, asset, side, bi_robot_yaml) as (node, peer):
        assert node.mode == "direct" and node.side == side
        _reset_direct(peer)
        pp = peer.wait("palm_pose", 1)                                  # 리셋 직후 latched palm_pose
        assert pp[0].header.frame_id == "base_link"
        _pump(peer, 0.25)
        assert peer.got["joint_target"] == []                           # start 전에는 적분하지 않는다
        peer.episode("start", 1)
        _pump(peer, 0.3)
        jt = peer.wait("joint_target", 3)
        assert tuple(jt[0].name) == tuple(s.arm_joints) + hand_joints  # 이 팔의 관절만
        assert jt[0].header.frame_id == "1:0" and jt[1].header.frame_id == "1:1"
        np.testing.assert_allclose(jt[0].position[7:], peer.hand_home[side])    # 손 = 리셋 실측 유지
        home_palm = node.fabric.palm_pose(np.asarray(node.fabric.cfg.home_q))
        np.testing.assert_allclose(peer.pose_xyz("palm_target", 0), home_palm[:3], atol=1e-9)  # 첫 목표 = 홈 palm
        target = np.array([0.3, 0.1 if side == "left" else -0.1, 0.4, 0.0, 0.0, 0.0])
        peer.palm_cmd(target)
        _pump(peer, 0.25)
        np.testing.assert_allclose(peer.pose_xyz("palm_target"), target[:3], atol=1e-9)
        cmd = peer.hand_home[side] + 0.1
        peer.hand_cmd(hand_joints, cmd)
        _pump(peer, 0.25)
        np.testing.assert_allclose(peer.got["joint_target"][-1].position[7:], cmd, atol=1e-9)
        assert len(peer.got["palm_pose"]) > 3                             # tick 마다 현재 palm 발행
        st = peer.last_status()
        assert st["side"] == side and st["mode"] == "direct" and st["running"] is True and st["ok"] is True
        peer.episode("stop", 1)
        _pump(peer, 0.15)
        n = len(peer.got["joint_target"])
        _pump(peer, 0.3)
        assert len(peer.got["joint_target"]) == n                          # stop → 발행 중단


@needs_asset
def test_control_only_bad_palm_cmd_and_hand_cmd_are_reported_not_applied(ros, asset, bi_robot_yaml):
    with _direct_rig(ros, asset, "left", bi_robot_yaml) as (node, peer):
        _reset_direct(peer)
        peer.episode("start", 1)
        _pump(peer, 0.2)
        before = peer.pose_xyz("palm_target")
        m = peer.mark()
        peer.palm_cmd(np.array([0.3, 0.1, 0.4, 0.0, 0.0, 0.0]), frame="odom")
        st = peer.status_after(m)
        assert st["ok"] is False and any("palm_cmd" in r and "frame" in r for r in st["reasons"])
        _pump(peer, 0.2)
        np.testing.assert_allclose(peer.pose_xyz("palm_target"), before, atol=1e-9)   # 목표 불변
        m = peer.mark()
        peer.hand_cmd(("l_hj_thumb_1",), [0.3])                                      # 손 관절 결손
        st = peer.status_after(m)
        assert st["ok"] is False and any("hand_cmd" in r for r in st["reasons"])
        _pump(peer, 0.2)
        np.testing.assert_allclose(peer.got["joint_target"][-1].position[7:], peer.hand_home["left"])
        assert len(peer.wait("joint_target", 5)) >= 5                                # 노드는 살아 있다


@needs_asset
def test_side_param_must_match_a_single_arm_yaml(ros, asset):
    from policy_control.sources import RobotCfgError

    with pytest.raises(RobotCfgError):
        _direct_node(ros, asset, LEFT_DG5F_ROBOT, "right")
    node = _direct_node(ros, asset, LEFT_DG5F_ROBOT, "left")
    try:
        assert node.side == "left" and node.robot_cfg.sources["ee"].topic == "/dg5f_left/joint_states"
        assert node.stage.hand_joints == tuple(asset.side("left").hand_joints)
    finally:
        node.destroy_node()


@needs_asset
def test_side_param_defaults_to_primary(ros, asset):
    from rclpy.parameter import Parameter

    from policy_control.fabric_node import FabricNode

    overrides = [Parameter("contract", value=str(ASSET_JSON)), Parameter("robot", value=str(BI_ROBOT)),
                 Parameter("device", value="cpu")]
    node = FabricNode(context=ros, fabric=_fake_fabric_core_side(asset, asset.primary_side), parameter_overrides=overrides)
    try:
        assert node.side == asset.primary_side == "right" and node.robot_cfg.sources["arm"].joints[0] == "r_aj_1"
    finally:
        node.destroy_node()


# ---------------------------------------------------------------- tools/palm_cmd.py
def _load_palm_cmd_tool():
    spec = importlib.util.spec_from_file_location("palm_cmd_tool", PALM_CMD_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_palm_cmd_tool_pure_helpers():
    T = _load_palm_cmd_tool()
    assert T.check_domain({}, False) and T.check_domain({"ROS_DOMAIN_ID": "0"}, False)
    assert T.check_domain({"ROS_DOMAIN_ID": "99"}, False) is None and T.check_domain({}, True) is None
    cur = np.array([0.1, 0.2, 0.3, 0.5, 0.1, -0.4])
    np.testing.assert_allclose(T.compose_target(cur, rel=[0.05, 0.0, -0.02]), [0.15, 0.2, 0.28, 0.5, 0.1, -0.4])
    yaw_only = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(T.compose_target(yaw_only, rel=[0, 0, 0, 0, 0, 0.3])[3:], [0.8, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T.compose_target(cur, abs_=[0.3, -0.3, 0.4, 90, 0, 90], deg=True),
                               [0.3, -0.3, 0.4, np.pi / 2, 0.0, np.pi / 2])
    np.testing.assert_allclose(T.rpy_to_euler_zyx([1.0, 2.0, 3.0]), [3.0, 2.0, 1.0])
    np.testing.assert_allclose(T.euler_zyx_to_rpy(T.rpy_to_euler_zyx([10, 20, 30], deg=True), deg=True), [10, 20, 30])
    for bad in (dict(rel=[1.0, 2.0]), dict(), dict(rel=[0.0] * 3, abs_=[0.0] * 6), dict(abs_=[0.0] * 5)):
        with pytest.raises(ValueError):
            T.compose_target(cur, **bad)
    assert T.main(["--rel", "0.01", "0", "0"]) == 3 if "ROS_DOMAIN_ID" not in os.environ else True   # 도메인 0/unset 거부


@needs_asset
def test_palm_cmd_tool_relative_move_targets_the_running_fabric(ros, asset, bi_robot_yaml, monkeypatch):
    T = _load_palm_cmd_tool()
    monkeypatch.setenv("ROS_DOMAIN_ID", os.environ.get("PC_TEST_DOMAIN", "99"))
    with _direct_rig(ros, asset, "left", bi_robot_yaml) as (node, peer):
        _reset_direct(peer)
        peer.episode("start", 1)
        _pump(peer, 0.3)
        assert peer.wait("joint_target", 2)
        x_before = peer.pose_xyz("palm_pose")[0]
        stop = threading.Event()
        th = threading.Thread(target=_pump_until, args=(peer, stop), daemon=True)
        th.start()
        try:
            rc = T.main(["--rel", "0.05", "0", "0", "--timeout", "5"])
        finally:
            stop.set()
            th.join(2.0)
        assert rc == 0
        _pump(peer, 0.3)
        x_after = peer.pose_xyz("palm_pose")[0]
        x_target = peer.pose_xyz("palm_target")[0]
        # 도구는 발행 시점의 palm_pose 를 기준으로 +5 cm 를 더한다 (가짜 fabric 의 q 는 tick 마다 조금씩 흐른다)
        assert x_before + 0.05 - 1e-9 <= x_target <= x_after + 0.05 + 1e-9, (x_before, x_target, x_after)
        assert T.main(["--rel", "0", "0", "0", "--dry-run", "--timeout", "3"]) == 0
