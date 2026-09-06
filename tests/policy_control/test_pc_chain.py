"""M6 — chain 스테이지 단위 테스트(GPU·ROS 없음, 가짜 fabric/정책/FK).

잠그는 규칙:
  · ObsStage: reset 마다 episode +1·seq 0, 미발행 tick 에도 seq 증가, 스테일 → valid False,
    LSTM 계약은 max_gap_ticks 초과 시 abort(재개 없음), MLP 계약은 기본 abort 없음
  · PolicyStage: seq 규칙은 PolicyCore 안(중복 seq → 직전 액션, forward 없음)
  · FabricStage: 게이트 닫힘 → 강제 개방, 판 여유 가드(목표 q 기준, 판 xy 밖은 면제),
    hand_sync 'syn_target'|'measured'|None 에 따라 fabric 손 슬롯에 들어가는 벡터
  · PdStage: engage → RAMPING → ramp_done → TRACKING, 목표당 new_policy_step 1회,
    watchdog → HOLD(세트포인트 동결·q̇ 0), estop → HOLD, release → RELEASING → IDLE
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from policy_control import chain as CH
from policy_control import contract as C
from policy_control import contract_build as B
from policy_control import lockstep as LS
from policy_control import decoder_core as D
from policy_control import fabric_core as F
from policy_control import fk_numpy, pd_law, policy_core, sources
from policy_control.pd_state import Phase

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
ROBOTS = SIM2REAL / "policy_control/config/robots"
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
RIGHT_E1_RUN = SIM2REAL / "logs/policy/right_e1"
needs_e1 = pytest.mark.skipif(not (RIGHT_E1_RUN / "params").exists(), reason="right_e1 run dir 없음")
needs_left = pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")
needs_right = pytest.mark.skipif(not RIGHT_JSON.exists(), reason="right_g1 contract 없음")

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
HAND_PROFILE = tuple(f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5))
N = 7


# ------------------------------------------------------------------ fakes
class FakeFabric:
    def __init__(self, num_joints: int) -> None:
        self.num_joints = num_joints
        self.default_config = torch.zeros(1, num_joints)
        self.calls: list[dict] = []

    def set_features(self, hand, palm, convention, q, qd, obj_ids, obj_ind, damping):
        self.calls.append({"palm": palm.clone(), "q": q.clone()})

    def get_palm_pose(self, q, convention):
        return torch.cat([q[:, :3], q[:, :3] * 0.5], dim=1)

    def get_fingertip_positions(self, q):
        return q[:, :5].unsqueeze(-1).repeat(1, 1, 3)


class FakeIntegrator:
    def step(self, q, qd, qdd, dt):
        return q + 0.01, torch.ones_like(qd), qdd


def _fake_fabric(contract: C.DeployContract) -> F.FabricCore:
    n = len(contract.fabric.joint_order)
    be = F.FabricBackend(fabric=FakeFabric(n), integrator=FakeIntegrator(),
                         object_ids=None, object_indicator=None, device="cpu")
    return F.FabricCore(contract, "cpu", backend=be)


class FakeLeftFK:
    """좌 FK 대역: palm/tcp z 를 arm_q[0] 로 조종한다(가드 테스트용)."""

    palm_body = "l_hl_gripper_base"
    hand_joints = ("l_hj_gripper_1", "l_hj_gripper_2")
    tip_names = ("l_hl_gripper_left_finger", "l_hl_gripper_right_finger")

    def palm_pose(self, arm_q, hand_q):
        q = np.asarray(arm_q, dtype=float)
        base = np.array([0.4, 0.2, float(q[0])])
        tips = np.stack([base + [0.0, 0.02, -0.05], base + [0.0, -0.02, -0.05]])
        return fk_numpy.FKPose(palm_body=self.palm_body, palm_pos=base, palm_quat=IDENT.copy(), tips=tips,
                               tip_names=self.tip_names, extra={"tcp": base + [0.0, 0.0, -0.08]})


class RecordedBackend:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        self.i = 0
        self.forwards = 0

    def forward(self, obs):
        a = self.actions[self.i % len(self.actions)]
        self.i += 1
        self.forwards += 1
        return a

    def reset(self):
        self.i = 0


def _left_state(**kw) -> sources.RobotState:
    base = dict(arm_q=np.array([0.30, -0.3, 0.0, 0.5, -0.4, 0.0, -0.8]), arm_qd=np.zeros(7),
                ee_names=("l_hj_gripper_1", "l_hj_gripper_2"), ee_q=np.full(2, 0.044), ee_qd=np.zeros(2),
                object_pos=np.array([0.4, 0.2, 0.25]), object_quat=IDENT, tip_force=None, tip_names=(),
                head=None, decoder_target=None, stamps={}, stale=(), missing=())
    base.update(kw)
    return sources.RobotState(**base)


@pytest.fixture(scope="module")
def left() -> C.DeployContract:
    return C.load_contract(LEFT_JSON)


@pytest.fixture(scope="module")
def right() -> C.DeployContract:
    return C.load_contract(RIGHT_JSON)


@pytest.fixture(scope="module")
def left_cfg() -> sources.RobotCfg:
    return sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml"), "live")


def _obs_stage(contract, cfg, **kw) -> CH.ObsStage:
    return CH.ObsStage(contract, cfg, FakeLeftFK(), **kw)


# ================================================================== ObsStage
@needs_left
def test_obs_stage_episode_and_seq_bookkeeping(left, left_cfg):
    st = _obs_stage(left, left_cfg)
    assert st.episode == 0 and st.phase == "idle"
    ev = st.reset(_left_state())
    assert ev.episode == 1 and ev.event == "reset" and st.phase == "running"
    assert ev.object_anchor is not None and set(ev.home_q) >= set(left.obs.joint_orders["arm"])
    ticks = [st.tick(_left_state(), last_action=None if t == 0 else np.zeros(7)) for t in range(3)]
    assert [t.out.seq for t in ticks] == [0, 1, 2]
    assert all(t.out.valid for t in ticks)
    assert [t.status.seq for t in ticks] == [0, 1, 2]
    assert ticks[0].status.node == "obs" and ticks[0].status.episode == 1 and ticks[0].status.ok
    ev2 = st.reset(_left_state())
    assert ev2.episode == 2
    assert st.tick(_left_state()).out.seq == 0


@needs_left
def test_obs_stage_default_gap_rule_follows_contract_rnn(left, left_cfg):
    mlp = _obs_stage(left, left_cfg)
    assert mlp.max_gap_ticks is None
    lstm_contract = dataclasses.replace(left, policy=dataclasses.replace(left.policy, rnn={"type": "lstm"}))
    lstm = _obs_stage(lstm_contract, left_cfg)
    assert lstm.max_gap_ticks == CH.DEFAULT_RNN_MAX_GAP_TICKS == 3
    assert _obs_stage(left, left_cfg, max_gap_ticks=5).max_gap_ticks == 5


@needs_left
def test_obs_stage_stale_is_invalid_and_aborts_after_max_gap(left, left_cfg):
    st = _obs_stage(left, left_cfg, max_gap_ticks=2)
    st.reset(_left_state())
    ok = st.tick(_left_state())
    assert ok.out.valid and ok.gap == 0 and not ok.abort
    stale = _left_state(stale=("arm",))
    t1 = st.tick(stale)
    t2 = st.tick(stale)
    assert not t1.out.valid and not t2.out.valid
    assert (t1.gap, t2.gap) == (1, 2) and not t1.abort and not t2.abort
    assert any("stale" in r for r in t1.out.reasons) and not t1.status.ok
    t3 = st.tick(stale)
    assert t3.abort and t3.gap == 3 and st.phase == "aborted"
    assert any("max_gap_ticks" in r for r in t3.out.reasons)
    # 재개 없음: 센서가 살아나도 다음 reset 까지 미발행, seq 는 계속 증가
    t4 = st.tick(_left_state())
    assert not t4.out.valid and any("aborted" in r for r in t4.out.reasons)
    assert [t.out.seq for t in (ok, t1, t2, t3, t4)] == [0, 1, 2, 3, 4]
    st.reset(_left_state())
    assert st.phase == "running" and st.tick(_left_state()).out.valid


@needs_left
def test_obs_stage_without_gap_rule_recovers_from_stale(left, left_cfg):
    st = _obs_stage(left, left_cfg)          # MLP: max_gap None
    st.reset(_left_state())
    for _ in range(10):
        assert not st.tick(_left_state(stale=("arm",))).out.valid
    t = st.tick(_left_state())
    assert t.out.valid and t.gap == 0 and st.phase == "running"


@needs_left
def test_obs_stage_tick_before_reset_and_after_stop(left, left_cfg):
    st = _obs_stage(left, left_cfg)
    t = st.tick(_left_state())
    assert not t.out.valid and any("idle" in r for r in t.out.reasons)
    st.reset(_left_state())
    ev = st.stop("user")
    assert ev.event == "stop" and st.phase == "stopped"
    assert not st.tick(_left_state()).out.valid
    ev = st.abort("fabric: table clearance")
    assert ev.event == "abort" and "fabric: table clearance" in ev.reasons


# ================================================================== PolicyStage
def test_policy_stage_wraps_seq_rule():
    be = RecordedBackend(np.arange(3 * 7, dtype=float).reshape(3, 7))
    core = policy_core.PolicyCore.with_backend(be, obs_dim=4, action_dim=7, action_clip=None)
    stage = CH.PolicyStage(core)
    obs = np.ones(4)
    a0 = stage.tick(obs, 0)
    a1 = stage.tick(obs, 1)
    dup = stage.tick(obs, 1)                    # 중복 seq: forward 없음, 직전 액션
    assert be.forwards == 2
    np.testing.assert_array_equal(a0.action, be.actions[0])
    np.testing.assert_array_equal(a1.action, be.actions[1])
    np.testing.assert_array_equal(dup.action, a1.action)
    assert a1.seq == 1 and a1.status.node == "policy" and a1.status.ok
    stage.reset()
    assert be.i == 0
    with pytest.raises(policy_core.SeqError):
        stage.tick(obs, 3)                       # seq 0 을 보기 전의 seq>0


# ================================================================== FabricStage (left)
def _left_fabric_stage(left, table=None) -> CH.FabricStage:
    table = table or CH.TableGuard(top=0.205, clearance_min=0.03)
    return CH.FabricStage(left, D.ActionDecoder(left), _fake_fabric(left), table, fk=FakeLeftFK())


def _fab_in(action, seq, gate_open=True, arm_q=None, ee_q=None):
    return CH.FabricIn(action=np.asarray(action, dtype=float), seq=seq, gate_open=gate_open,
                       object_pos=np.array([0.4, 0.2, 0.25]),
                       arm_q_meas=np.zeros(7) if arm_q is None else np.asarray(arm_q, dtype=float),
                       ee_names=("l_hj_gripper_1", "l_hj_gripper_2"),
                       ee_q=np.full(2, 0.044) if ee_q is None else np.asarray(ee_q, dtype=float))


@needs_left
def test_left_fabric_stage_reset_and_tick_outputs(left):
    st = _left_fabric_stage(left)
    rs = st.reset(arm_q_meas=np.asarray(left.pd.home_arm), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"),
                  ee_q=np.full(2, 0.044))
    np.testing.assert_allclose(rs.home_q, left.fabric.home_q)
    out = st.tick(_fab_in(np.array([0.0] * 6 + [-0.9]), seq=0, gate_open=True, arm_q=np.full(7, 0.5)))
    assert out.seq == 0 and out.status.node == "fabric" and not out.status.ok    # 가드가 걸린 tick
    assert tuple(out.joint_names) == tuple(left.obs.joint_orders["arm"]) + tuple(left.action.hand.joints)
    assert out.q_arm.shape == (7,) and out.qd_arm.shape == (7,) and out.q.shape == (8,) and out.qd.shape == (8,)
    assert out.gripper_cmd == left.action.hand.params["close"] and out.hand_target is None
    assert out.q[7] == out.gripper_cmd and out.qd[7] == 0.0
    np.testing.assert_allclose(out.qd_arm, np.ones(7) * left.fabric.vel_ff_scale)
    # 목표 q[0] = home[0] + 2×0.01, tcp z = q[0] − 0.08 → 판 위 여유 ≪ 0: 가드
    assert out.clearance == pytest.approx(left.fabric.home_q[0] + 0.02 - 0.08 - 0.205)
    assert not out.clearance_ok and out.abort and any("clearance" in r for r in out.reasons)
    assert out.palm6.shape == (6,)


@needs_left
def test_left_fabric_stage_gate_closed_forces_open(left):
    st = _left_fabric_stage(left)
    st.reset(arm_q_meas=np.asarray(left.pd.home_arm), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"),
             ee_q=np.full(2, 0.044))
    close = np.array([0.0] * 6 + [-0.9])
    assert st.tick(_fab_in(close, 0, gate_open=False)).gripper_cmd == left.action.hand.params["open"]
    assert st.tick(_fab_in(close, 1, gate_open=True)).gripper_cmd == left.action.hand.params["close"]
    assert st.tick(_fab_in(close, 2, gate_open=True)).close_gate == 1.0


@needs_left
def test_left_clearance_guard_uses_target_q_and_table_extent(left):
    # 판 xy 범위를 주면 판 밖 목표는 낮아도 통과, 판 위는 여유 하한을 지켜야 한다
    inside = CH.TableGuard(top=0.205, clearance_min=0.03, center_xy=(0.4, 0.2), size_xy=(0.5, 0.9))
    outside = CH.TableGuard(top=0.205, clearance_min=0.03, center_xy=(2.0, 2.0), size_xy=(0.1, 0.1))
    for table, want_ok in ((inside, False), (outside, True)):
        st = _left_fabric_stage(left, table)
        st.reset(arm_q_meas=np.zeros(7), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"), ee_q=np.full(2, 0.044))
        out = st.tick(_fab_in(np.zeros(7), 0))
        assert out.clearance_ok is want_ok, table
        # 목표 q[0](= fabric home[0] + 2×0.01) 기준 — 실측 q(0) 가 아니다
        assert out.clearance == pytest.approx(left.fabric.home_q[0] + 0.02 - 0.08 - 0.205)
    # 여유가 충분하면 통과: home_q 로 fabric 을 높은 q[0] 에서 시작
    st = _left_fabric_stage(left, inside)
    st.reset(arm_q_meas=np.zeros(7), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"), ee_q=np.full(2, 0.044),
             home_q=np.array([1.0, 0, 0, 0, 0, 0, 0]))
    out = st.tick(_fab_in(np.zeros(7), 0))
    assert out.clearance_ok and out.clearance == pytest.approx(1.02 - 0.08 - 0.205)


@needs_left
def test_left_fabric_stage_requires_fk_for_left_gripper_kind(left):
    with pytest.raises(CH.ChainError):
        CH.FabricStage(left, D.ActionDecoder(left), _fake_fabric(left), CH.TableGuard(0.2, 0.03), fk=None)


@needs_left
def test_fabric_stage_seq_and_new_objects(left):
    st = _left_fabric_stage(left)
    st.reset(arm_q_meas=np.zeros(7), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"), ee_q=np.full(2, 0.044))
    a = np.zeros(7)
    o1 = st.tick(_fab_in(a, 0))
    o2 = st.tick(_fab_in(a, 1))
    assert o1 is not o2 and o1.q_arm is not o2.q_arm
    assert o2.seq == 1 and not np.shares_memory(o1.q_arm, o2.q_arm)


# ================================================================== FabricStage (right) hand sync
def _right_stage(right, hand_sync) -> tuple[CH.FabricStage, FakeFabric]:
    c = dataclasses.replace(right, fabric=dataclasses.replace(right.fabric, hand_sync=hand_sync))
    fab = _fake_fabric(c)
    dec = D.ActionDecoder(c, hand_soft_limits=np.asarray(c.action.hand.params["soft_limits"]))
    st = CH.FabricStage(c, dec, fab, CH.TableGuard(top=0.2, clearance_min=0.03))
    return st, fab.backend.fabric


def _right_in(seq, hand_q):
    return CH.FabricIn(action=np.full(21, 0.3), seq=seq, gate_open=None, object_pos=np.array([0.35, -0.17, 0.28]),
                       arm_q_meas=np.asarray(C.load_contract(RIGHT_JSON).pd.home_arm), ee_names=HAND_PROFILE,
                       ee_q=np.asarray(hand_q, dtype=float))


@needs_right
@pytest.mark.parametrize("mode", ["syn_target", "measured", None])
def test_right_hand_sync_modes_feed_the_fabric_hand_slot(right, mode):
    from grasp_s2r_fabric import permutation

    st, fake = _right_stage(right, mode)
    open_pose = np.asarray(right.action.hand.params["open_pose"])
    home = np.asarray(right.pd.home_arm)
    st.reset(arm_q_meas=home, ee_names=HAND_PROFILE, ee_q=open_pose, object_anchor=np.array([0.35, -0.17, 0.28]))
    meas = open_pose + 0.123
    out = st.tick(_right_in(0, meas))
    assert out.hand_target is not None and out.hand_target.shape == (20,) and out.gripper_cmd is None
    assert 0.0 <= out.close_gate <= 1.0
    assert out.q.shape == (27,) and out.qd.shape == (27,)
    np.testing.assert_allclose(out.q[7:], out.hand_target)
    perm = permutation(right.action.hand.joints, right.fabric.joint_order[7:])
    slot = fake.calls[-1]["q"][0, 7:].numpy()
    if mode == "syn_target":
        np.testing.assert_allclose(slot, out.hand_target[perm], atol=1e-6)
    elif mode == "measured":
        np.testing.assert_allclose(slot, meas[perm], atol=1e-6)
    else:
        np.testing.assert_allclose(slot, np.asarray(right.fabric.home_q)[7:], atol=1e-6)   # 손 슬롯 불변


@needs_right
def test_right_fabric_stage_reset_calibrates_cage_and_clearance_uses_tips(right):
    st, _ = _right_stage(right, "syn_target")
    home = np.asarray(right.pd.home_arm)
    open_pose = np.asarray(right.action.hand.params["open_pose"])
    rs = st.reset(arm_q_meas=home, ee_names=HAND_PROFILE, ee_q=open_pose, object_anchor=np.array([0.35, -0.17, 0.28]))
    assert st.decoder.cage is not None and rs.object_anchor is not None
    out = st.tick(_right_in(0, open_pose))
    # 가짜 FK: tips z = q[:5] → 최저 = min(q_full[:5]) − top
    assert out.clearance == pytest.approx(float(out.q_full[:5].min()) - 0.2)
    with pytest.raises(D.DecoderError):
        CH.FabricStage(right, D.ActionDecoder(right), _fake_fabric(right), CH.TableGuard(0.2, 0.03)).reset(
            arm_q_meas=home, ee_names=HAND_PROFILE, ee_q=open_pose)   # spawn 앵커 없음


# ================================================================== PdStage
def _law(**over) -> pd_law.PdLawCfg:
    base = dict(max_vel=2.0, lead_sec=0.1, lead_vel=2.0, vel_ff_scale=1.0, vel_ff_cap=0.5,
                effort_cap=20.0, lower=np.full(N, -3.0), upper=np.full(N, 3.0), gravity_mode="off")
    base.update(over)
    return pd_law.PdLawCfg(**base)


def _pd(**over) -> CH.PdStage:
    kw = dict(ramp_cfg=_law(max_vel=0.1), track_cfg=_law(), watchdog_sec=0.25, abort_tracking=0.3,
              ramp_tol=0.01, release_zero_ticks=3, dt=0.01)
    kw.update(over)
    return CH.PdStage(**kw)


def _target(q, seq, t, qd=None) -> CH.PdTarget:
    return CH.PdTarget(q=np.full(N, float(q)), qd=np.zeros(N) if qd is None else np.full(N, float(qd)),
                       tau_ff=np.zeros(N), seq=seq, t_recv=float(t))


def test_pd_stage_idle_emits_nothing_then_engage_ramps_to_tracking():
    pd = _pd()
    assert pd.state.fsm.phase is Phase.IDLE
    out = pd.tick(None, np.zeros(N), np.zeros(N), now=0.0)
    assert out.cmd is None and out.status.phase == "IDLE" and out.status.node == "pd"
    pd.engage(np.zeros(N))
    assert pd.state.fsm.phase is Phase.RAMPING
    # 첫 목표 전: 시드 세트포인트를 유지한다
    out = pd.tick(None, np.zeros(N), np.zeros(N), now=0.01)
    np.testing.assert_array_equal(out.cmd.q, np.zeros(N))
    assert out.target_age is None and not out.new_policy_step
    # 첫 목표(0.05 rad 떨어짐)가 매 tick 재수신(같은 seq, 새 t_recv → watchdog 갱신):
    # 램프 0.1 rad/s × 0.01 s = 0.001/tick, 50 tick 뒤 도달 → TRACKING
    phases = []
    for k in range(60):
        tgt = _target(0.05, seq=0, t=0.02 + k * 0.01)
        out = pd.tick(tgt, np.zeros(N), np.zeros(N), now=0.02 + k * 0.01)
        assert out.new_policy_step is (k == 0)
        phases.append(out.state.fsm.phase)
        assert out.cmd.q[0] <= 0.05 + 1e-12
    assert phases[0] is Phase.RAMPING and phases[-1] is Phase.TRACKING
    assert out.cmd.q[0] == pytest.approx(0.05)


def test_pd_stage_new_policy_step_once_per_target_seq():
    pd = _pd()
    pd.engage(np.zeros(N))
    t0 = _target(0.0, seq=0, t=0.0)
    flags = [pd.tick(t0, np.zeros(N), np.zeros(N), now=k * 0.01).new_policy_step for k in range(3)]
    assert flags == [True, False, False]
    t1 = _target(0.0, seq=1, t=0.03)
    assert pd.tick(t1, np.zeros(N), np.zeros(N), now=0.03).new_policy_step
    assert not pd.tick(t1, np.zeros(N), np.zeros(N), now=0.04).new_policy_step
    # 목표 None 이면 직전 목표를 유지(new step 아님)
    assert not pd.tick(None, np.zeros(N), np.zeros(N), now=0.05).new_policy_step


def test_pd_stage_watchdog_holds_and_freezes_setpoint():
    pd = _pd()
    pd.engage(np.zeros(N))
    pd.tick(_target(0.0, 0, 0.0), np.zeros(N), np.zeros(N), now=0.0)       # ramp_done(이미 도달)
    assert pd.state.fsm.phase is Phase.TRACKING
    moving = _target(1.0, seq=1, t=0.01, qd=0.3)
    out = pd.tick(moving, np.zeros(N), np.zeros(N), now=0.01)
    assert out.cmd.q[0] == pytest.approx(0.02) and out.cmd.qd[0] == pytest.approx(0.3)   # 2 rad/s·0.01
    out = pd.tick(moving, np.zeros(N), np.zeros(N), now=0.02)
    q_frozen = out.cmd.q.copy()
    late = pd.tick(moving, np.zeros(N), np.zeros(N), now=0.01 + 0.3)        # 목표 두절 0.3 s > 0.25
    assert late.state.fsm.phase is Phase.HOLD and "watchdog" in late.state.fsm.hold_reason
    assert any("watchdog" in r for r in late.faults) and not late.status.ok
    np.testing.assert_array_equal(late.cmd.q, q_frozen)
    np.testing.assert_array_equal(late.cmd.qd, np.zeros(N))
    # HOLD 는 새 목표가 와도 풀리지 않는다
    again = pd.tick(_target(1.0, seq=2, t=0.35), np.zeros(N), np.zeros(N), now=0.35)
    assert again.state.fsm.phase is Phase.HOLD
    np.testing.assert_array_equal(again.cmd.q, q_frozen)


def test_pd_stage_estop_and_tracking_error_hold():
    pd = _pd()
    pd.engage(np.zeros(N))
    out = pd.tick(_target(0.0, 0, 0.0), np.zeros(N), np.zeros(N), now=0.0, estop=True)
    assert out.state.fsm.phase is Phase.HOLD and "estop" in out.state.fsm.hold_reason
    pd2 = _pd()
    pd2.engage(np.zeros(N))
    out = pd2.tick(_target(0.0, 0, 0.0), np.full(N, 0.5), np.zeros(N), now=0.0)   # |sp − meas| 0.5 > 0.3
    assert out.state.fsm.phase is Phase.HOLD and "tracking error" in out.state.fsm.hold_reason


def test_pd_stage_release_zero_ticks_to_idle_and_new_episode_resets_droop():
    pd = _pd(track_cfg=_law(gravity_mode="integral_droop", droop_gain=0.5, droop_limit=np.full(N, 1.0)))
    pd.engage(np.zeros(N))
    pd.tick(_target(0.0, 0, 0.0), np.zeros(N), np.zeros(N), now=0.0)
    assert pd.state.fsm.phase is Phase.TRACKING
    out = pd.tick(_target(0.1, 1, 0.01), np.zeros(N), np.zeros(N), now=0.01)
    assert out.state.law.droop[0] == pytest.approx(0.05)
    pd.new_episode(episode=2)
    assert np.all(pd.state.law.droop == 0.0) and pd.state.fsm.phase is Phase.TRACKING
    pd.release()
    assert pd.state.fsm.phase is Phase.RELEASING
    outs = [pd.tick(None, np.zeros(N), np.zeros(N), now=0.1 + k * 0.01) for k in range(3)]
    for o in outs:
        np.testing.assert_array_equal(o.cmd.qd, np.zeros(N))
        np.testing.assert_array_equal(o.cmd.tau, np.zeros(N))
    assert outs[-1].state.fsm.phase is Phase.IDLE
    assert pd.tick(None, np.zeros(N), np.zeros(N), now=0.2).cmd is None


def test_pd_stage_returns_new_state_objects():
    pd = _pd()
    pd.engage(np.zeros(N))
    s0 = pd.state
    out = pd.tick(_target(0.0, 0, 0.0), np.zeros(N), np.zeros(N), now=0.0)
    assert out.state is not s0 and out.state.law is not s0.law
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.state.fsm = None  # type: ignore[misc]


# ================================================================== lockstep (CPU, 가짜 fabric)
@needs_left
def test_stream_states_left_row_semantics(left, fixtures_dir):
    stream = LS.load_stream(fixtures_dir / "stream_left_v2b25.npz")
    assert stream.n == 65 and stream.step_dt == pytest.approx(left.rate.step_dt)
    states = LS.stream_states(stream, left)
    assert len(states) == stream.n
    s0, s1 = states[0], states[1]
    np.testing.assert_allclose(s0.arm_q, left.pd.home_arm)
    np.testing.assert_allclose(s0.ee_q, np.full(2, left.action.hand.params["open"]))
    np.testing.assert_allclose(s0.object_pos, stream.cup_spawn)
    np.testing.assert_allclose(s1.arm_q, stream.arm_meas[0])          # t 는 t−1 의 물리 뒤 실측
    np.testing.assert_allclose(s1.ee_q, stream.hand_meas[0])
    np.testing.assert_allclose(s1.object_pos, stream.cup_pos3[0])
    np.testing.assert_allclose(s1.arm_qd, stream.obs[1, 9:16])          # 속도는 obs 칸 그대로
    np.testing.assert_allclose(s1.ee_qd, stream.obs[1, 16:18])
    assert s1.ee_names == ("l_hj_gripper_1", "l_hj_gripper_2") and s1.tip_force is None
    uni = LS.stream_states(stream, left, unify_ee=True)[1]
    assert uni.ee_q[0] == uni.ee_q[1] == stream.hand_meas[0][0]


@needs_e1
def test_stream_states_right_hand_order_and_tip_force(fixtures_dir):
    contract = B.build_contract(RIGHT_E1_RUN)
    stream = LS.load_stream(fixtures_dir / "stream_right_e1_v2.npz")
    states = LS.stream_states(stream, contract)
    s0, s1 = states[0], states[1]
    assert s0.ee_names == HAND_PROFILE
    np.testing.assert_allclose(s0.ee_q, contract.action.hand.params["open_pose"])
    hand_obs = list(contract.obs.joint_orders["hand_obs"])
    dof_to_prof = [hand_obs.index(n) for n in HAND_PROFILE]
    np.testing.assert_allclose(s1.ee_qd, stream.obs[1, 34:54][dof_to_prof])   # DOF 순 → 프로필 순(이름)
    np.testing.assert_allclose(s1.tip_force, stream.obs[1, 96:111].reshape(5, 3) * 10.0)
    assert s1.tip_names == tuple(contract.obs.joint_orders["tips"])


@needs_left
def test_run_lockstep_with_fake_fabric_records_every_stage(left, left_cfg, fixtures_dir):
    stream = LS.load_stream(fixtures_dir / "stream_left_v2b25.npz")
    states = LS.stream_states(stream, left)[:8]
    policy = LS.recorded_policy(left, stream.actions[:8])
    lower, upper, _ = pd_law.limits_from_profile(RL_WS / "robot_control/src/robot_control/profiles/openarm_tesollo.yaml",
                                                 left.pd.sim_gains.joints)
    cfg = pd_law.load_pd_config(SIM2REAL / "policy_control/config/pd_left.yaml")
    pd = CH.PdStage(ramp_cfg=pd_law.law_cfg_from_config(cfg, left, "ramp", lower, upper),
                    track_cfg=pd_law.law_cfg_from_config(cfg, left, "full", lower, upper),
                    watchdog_sec=cfg.watchdog_sec, abort_tracking=cfg.abort_tracking, ramp_tol=cfg.settle.tol,
                    release_zero_ticks=cfg.release_zero_ticks, dt=1.0 / cfg.pd_hz)
    table = CH.TableGuard(top=-10.0, clearance_min=0.03)               # 가짜 fabric 의 q 는 기하가 아니다 — 가드 무력화
    trace = LS.run_lockstep(left, left_cfg, states, policy, _fake_fabric(left), table, fk=FakeLeftFK(), pd=pd,
                            goal=stream.goal[0])
    assert trace.episode == 1 and not trace.aborted and len(trace.steps) == 8
    assert [s.seq for s in trace.steps] == list(range(8)) and all(s.valid for s in trace.steps)
    np.testing.assert_allclose(trace.column("action"), stream.actions[:8], atol=0)   # clip 100 = 무클립
    assert trace.column("obs").shape == (8, left.policy.obs_dim)
    assert all(len(s.pd) == 2 for s in trace.steps)                    # 50 Hz 정책 · 100 Hz pd
    ticks = [pt for s in trace.steps for pt in s.pd]
    assert all(pt.phase is Phase.TRACKING for pt in ticks)
    assert [pt.new_policy_step for pt in ticks] == [True, False] * 8
    assert set(trace.steps[0].status) == {"obs", "policy", "fabric"}
    assert trace.steps[0].status["obs"]["seq"] == 0 and trace.steps[3].status["fabric"]["seq"] == 3


@needs_left
def test_run_lockstep_stops_on_clearance_abort_or_records_it(left, left_cfg, fixtures_dir):
    stream = LS.load_stream(fixtures_dir / "stream_left_v2b25.npz")
    states = LS.stream_states(stream, left)[:5]
    too_strict = CH.TableGuard(top=0.205, clearance_min=10.0)
    trace = LS.run_lockstep(left, left_cfg, states, LS.recorded_policy(left, stream.actions), _fake_fabric(left),
                            too_strict, fk=FakeLeftFK(), goal=stream.goal[0])
    assert trace.aborted and len(trace.steps) == 1 and any("clearance" in r for r in trace.abort_reasons)
    trace = LS.run_lockstep(left, left_cfg, states, LS.recorded_policy(left, stream.actions), _fake_fabric(left),
                            too_strict, fk=FakeLeftFK(), goal=stream.goal[0], stop_on_abort=False)
    assert not trace.aborted and len(trace.steps) == 5 and not any(s.clearance_ok for s in trace.steps)


@needs_left
def test_run_lockstep_marks_invalid_ticks_and_aborts_on_gap(left, left_cfg, fixtures_dir):
    stream = LS.load_stream(fixtures_dir / "stream_left_v2b25.npz")
    states = list(LS.stream_states(stream, left)[:6])
    states[2:5] = [dataclasses.replace(s, stale=("object",)) for s in states[2:5]]
    trace = LS.run_lockstep(left, left_cfg, states, LS.recorded_policy(left, stream.actions), _fake_fabric(left),
                            CH.TableGuard(top=-10.0, clearance_min=0.03), fk=FakeLeftFK(), goal=stream.goal[0],
                            max_gap_ticks=2)
    assert [s.valid for s in trace.steps] == [True, True, False, False, False]      # 3번째 스테일에서 abort
    assert trace.aborted and any("max_gap_ticks" in r for r in trace.abort_reasons)
    assert [s.seq for s in trace.steps] == [0, 1, 2, 3, 4]


def test_recorded_backend_exhaustion_is_an_error():
    be = LS.RecordedBackend(np.zeros((2, 7)))
    be.forward(np.zeros(4))
    be.forward(np.zeros(4))
    with pytest.raises(CH.ChainError):
        be.forward(np.zeros(4))
    with pytest.raises(CH.ChainError):
        LS.RecordedBackend(np.zeros(7))


# ================================================================== FabricStage (control-only, direct palm target)
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
needs_asset = pytest.mark.skipif(not ASSET_JSON.exists(), reason="asset contract 없음")


@pytest.fixture(scope="module")
def asset() -> C.DeployContract:
    return C.load_contract(ASSET_JSON)


def _fake_fabric_side(contract: C.DeployContract, side: str) -> F.FabricCore:
    n = len(contract.side(side).fabric.joint_order)
    be = F.FabricBackend(fabric=FakeFabric(n), integrator=FakeIntegrator(),
                         object_ids=None, object_indicator=None, device="cpu")
    return F.FabricCore(contract, "cpu", side=side, backend=be)


def _direct_stage(asset, side) -> tuple[CH.FabricStage, FakeFabric]:
    fab = _fake_fabric_side(asset, side)
    st = CH.FabricStage(asset, D.make_decoder(asset, side=side), fab, CH.TableGuard(top=-10.0, clearance_min=0.03))
    return st, fab.backend.fabric


def _direct_in(palm, seq, names, ee_q, hand_cmd=None) -> CH.FabricIn:
    return CH.FabricIn(action=np.asarray(palm, dtype=float), seq=seq, gate_open=None, object_pos=None,
                       arm_q_meas=np.zeros(7), ee_names=names, ee_q=np.asarray(ee_q, dtype=float), hand_cmd=hand_cmd)


@needs_asset
@pytest.mark.parametrize("side", ["left", "right"])
def test_direct_stage_holds_hand_then_follows_hand_cmd_and_names_side_joints(asset, side):
    s = asset.side(side)
    st, fake = _direct_stage(asset, side)
    names = tuple(s.hand_joints)
    meas = np.linspace(0.0, 1.0, 20)
    rs = st.reset(arm_q_meas=np.zeros(7), ee_names=names, ee_q=meas)
    np.testing.assert_allclose(rs.home_q, s.fabric.home_q)
    assert st.side == side and st.kind == "fabric" and st.decoder.kind == D.KIND_DIRECT
    palm = np.array([0.3, 0.1, 0.4, 0.0, 0.2, 0.0])
    out = st.tick(_direct_in(palm, 0, names, meas))
    assert out.joint_names == tuple(s.arm_joints) + names and out.q.shape == (27,) and out.qd.shape == (27,)
    np.testing.assert_array_equal(out.q[7:], meas)                                   # 손 = 리셋 실측 유지
    np.testing.assert_allclose(fake.calls[-1]["q"][0, 7:].numpy(), meas, atol=1e-6)   # syn_target 동기화
    np.testing.assert_allclose(fake.calls[-1]["palm"][0].numpy(), palm, atol=1e-6)   # 절대 palm 이 attractor 목표
    np.testing.assert_array_equal(out.palm6, palm)
    np.testing.assert_allclose(out.palm6_now, st.fabric.palm_pose(out.q_full))
    assert out.gripper_cmd is None and out.close_gate == 1.0 and out.status.ok
    cmd = meas + 0.3
    out2 = st.tick(_direct_in(palm, 1, names, meas, hand_cmd=cmd))
    np.testing.assert_array_equal(out2.q[7:], cmd)
    np.testing.assert_allclose(out2.qd[7:], np.full(20, 0.3 / asset.rate.step_dt * s.fabric.hand_vel_ff_scale))
    out3 = st.tick(_direct_in(palm, 2, names, meas))
    np.testing.assert_array_equal(out3.q[7:], cmd)                                    # hand_cmd 없으면 유지
    np.testing.assert_array_equal(out3.qd[7:], np.zeros(20))


@needs_asset
def test_direct_stage_rejects_wrong_side_decoder_and_bad_hand_cmd(asset):
    fab = _fake_fabric_side(asset, "left")
    with pytest.raises(CH.ChainError):
        CH.FabricStage(asset, D.make_decoder(asset, side="right"), fab, CH.TableGuard(0.2, 0.03))
    st, _ = _direct_stage(asset, "left")
    names = tuple(asset.side("left").hand_joints)
    st.reset(arm_q_meas=np.zeros(7), ee_names=names, ee_q=np.zeros(20))
    with pytest.raises(CH.ChainError):
        st.tick(_direct_in(np.zeros(6), 0, names, np.zeros(20), hand_cmd=np.zeros(3)))
    with pytest.raises(CH.ChainError):
        st.reset(arm_q_meas=np.zeros(7), ee_names=names[:19], ee_q=np.zeros(19))     # 손 관절 결손


def test_fabric_in_hand_cmd_defaults_to_none():
    inp = CH.FabricIn(action=np.zeros(6), seq=0, gate_open=None, object_pos=None, arm_q_meas=np.zeros(7),
                      ee_names=(), ee_q=np.zeros(0))
    assert inp.hand_cmd is None


@needs_right
def test_policy_stage_reports_palm6_now_of_target_q(right):
    st, _ = _right_stage(right, "syn_target")
    open_pose = np.asarray(right.action.hand.params["open_pose"])
    st.reset(arm_q_meas=np.asarray(right.pd.home_arm), ee_names=HAND_PROFILE, ee_q=open_pose,
             object_anchor=np.array([0.35, -0.17, 0.28]))
    out = st.tick(_right_in(0, open_pose))
    np.testing.assert_allclose(out.palm6_now, st.fabric.palm_pose(out.q_full))
    assert st.side == "right" and st.hand_joints == tuple(right.action.hand.joints)
