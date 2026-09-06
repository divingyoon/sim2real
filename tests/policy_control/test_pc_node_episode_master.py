"""episode_master — 제어 전용 계약의 에피소드 마스터(순수 상태기계 + ROS 서비스/latched 이벤트)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from policy_control import contract as C
from policy_control import episode_master as M

SIM2REAL = Path(__file__).resolve().parents[2]
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
RIGHT_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
NS = M.NS
needs_asset = pytest.mark.skipif(not ASSET_JSON.exists(), reason="자산 계약 없음")


# ---------------------------------------------------------------- 순수 로직
def test_book_reset_start_stop_sequence():
    book = M.EpisodeBook({"r_aj_1": 0.0})
    ev, why = book.start()
    assert ev is None and "reset first" in why[0]
    ev, _ = book.reset()
    assert (ev.episode, ev.event, book.phase) == (1, "reset", "armed") and ev.home_q == {"r_aj_1": 0.0}
    ev, _ = book.start()
    assert (ev.episode, ev.event, book.phase) == (1, "start", "running")
    ev, _ = book.start()                        # 두 번째 start 는 거부
    assert ev is None
    ev, _ = book.end("stop", "user stop")
    assert ev.event == "stop" and ev.reasons == ("user stop",) and book.phase == "stopped"
    ev, _ = book.reset()
    assert ev.episode == 2
    d = ev.as_dict()
    assert set(d) == {"episode", "event", "object_anchor", "home_q", "reasons"} and d["object_anchor"] is None


@needs_asset
def test_contract_home_covers_both_sides_and_policy_contract_refused():
    c = C.load_contract(ASSET_JSON)
    home = M.contract_home(c)
    assert len(home) == 2 * 27 and home["l_hj_thumb_2"] == pytest.approx(1.57) and home["r_aj_4"] == 0.0
    assert M.load_control_contract(ASSET_JSON).control_only
    if RIGHT_JSON.exists():
        with pytest.raises(M.EpisodeMasterError):
            M.load_control_contract(RIGHT_JSON)
    with pytest.raises(M.EpisodeMasterError):
        M.load_control_contract(SIM2REAL / "nope.json")


# ---------------------------------------------------------------- ROS
class _Spinner:
    def __init__(self, context, *nodes):
        from rclpy.executors import SingleThreadedExecutor

        self.executor = SingleThreadedExecutor(context=context)
        for n in nodes:
            self.executor.add_node(n)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.executor.spin_once(timeout_sec=0.05)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=2.0)
        self.executor.shutdown()


@pytest.fixture
def rig(ros):
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    master = M.EpisodeMaster(context=ros, parameter_overrides=[Parameter("contract", value=str(ASSET_JSON))])
    peer = Node("peer", context=ros)
    got = {"episode": [], "status": []}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    peer.create_subscription(String, f"{NS}/episode", lambda m: got["episode"].append(json.loads(m.data)), latched)
    peer.create_subscription(String, f"{NS}/status/{M.NODE}", lambda m: got["status"].append(json.loads(m.data)), 10)
    clients = {n: peer.create_client(Trigger, f"{NS}/episode/{n}") for n in M.EVENTS}
    spin = _Spinner(ros, master, peer)

    def call(name: str, timeout: float = 3.0) -> dict:
        assert clients[name].wait_for_service(timeout_sec=timeout), name
        fut = clients[name].call_async(Trigger.Request())
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        assert fut.done(), f"{name}: no response"
        return {"success": fut.result().success, **json.loads(fut.result().message)}

    def wait_events(n: int, timeout: float = 3.0):
        t0 = time.monotonic()
        while len(got["episode"]) < n and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        return list(got["episode"])

    yield {"call": call, "wait": wait_events, "got": got, "master": master}
    spin.close()
    master.destroy_node()
    peer.destroy_node()


@needs_asset
def test_services_publish_latched_events(rig):
    call, wait = rig["call"], rig["wait"]
    r = call("start")
    assert r["success"] is False and "reset first" in r["reasons"][0] and rig["got"]["episode"] == []
    assert call("reset")["success"] is True
    ev = wait(1)[-1]
    assert (ev["episode"], ev["event"]) == (1, "reset") and ev["object_anchor"] is None and "t_ns" in ev
    assert len(ev["home_q"]) == 54 and ev["home_q"]["l_hj_thumb_2"] == pytest.approx(1.57)
    assert call("start")["success"] is True
    assert wait(2)[-1]["event"] == "start"
    assert call("stop")["success"] is True
    assert wait(3)[-1]["reasons"] == ["user stop"]
    assert call("reset")["success"] and wait(4)[-1]["episode"] == 2
    assert call("abort")["success"] and wait(5)[-1]["event"] == "abort"
    assert [s["phase"] for s in rig["got"]["status"]][-2:] == ["armed", "stopped"]


@needs_asset
def test_policy_contract_is_refused_at_construction(ros):
    from rclpy.parameter import Parameter

    if not RIGHT_JSON.exists():
        pytest.skip("right_g1 계약 없음")
    with pytest.raises(M.EpisodeMasterError):
        M.EpisodeMaster(context=ros, parameter_overrides=[Parameter("contract", value=str(RIGHT_JSON))])
