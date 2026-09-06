"""M6 — 골든(4): 4 스테이지 인프로세스 체인(lockstep) vs 단일체 오라클 코어.

좌(v2B25, 65 스텝): 오라클 `scripts/left_policy_core.LeftPolicyCore` — 같은 센서·같은 기록
  액션(정책 대역)·같은 FK·**같은 FabricCore 인스턴스**(fabric=lambda p: core.step(p).q_arm).
  fabric 은 프로세스당 하나이므로 오라클 → `FabricCore.reset(home)` → 체인 → reset → 오라클 재실행의
  세 패스를 한 자식 프로세스에서 돌리고, 세 번째 패스(A′)로 fabric 자체의 결정론을 함께 잰다.
  단정: obs ≤1e-9 · palm_target ≤1e-9 · gripper_cmd 동일 · 팔 관절 목표 ≤1e-6 rad.
  ★LeftSensors 는 그리퍼 한 값(mimic)만 받으므로 두 코어에 같은 입력(첫 값)을 준다(golden_obs 와 동일).
  pd: PdStage 가 낸 명령이 법칙(직전 세트포인트 기준 속도 제한 + lead 캡 + droop 스텝당 1회,
  q̇* = vel_ff_scale·q̇*(캡), (kd/kp)·q̇* 항 부재)과 일치하는지 trace 로 검사한다.

우(e1, 194 스텝): 오라클 `scripts/grasp_s2r_core.GraspS2RCore` 에 fabric_step 을
  `lambda p: core.step(p, hand_target=oracle.hand.target).q_arm` 으로 주입(n = 계약 decimation —
  오라클 자체의 n=1 은 알려진 결함). 기록 obs 는 노이즈 포함이라 **오라클 대 체인** 만 대조한다.
  손 동기화 의미('syn_target')는 오라클과 같다(오라클도 방금 갱신한 syn 목표를 fabric 에 넣는다).
  ★우 fabric(27관절·body repulsion pairs·float32 warp)은 같은 입력으로도 run-to-run 2e-4 rad 급으로
    흔들린다(M4 parity 의 2.6~4.2 mm 와 같은 현상, 오라클 두 번 실행 A/A′ 로 실측). fabric 입력(palm·
    hand 목표)은 체인/오라클이 **정확히** 같으므로 팔 목표는 "fabric 자체 비결정의 3배 이내(최소 1e-6)"
    로 잠근다 — 좌(7관절, 결정론 0.0)는 1e-6 그대로.

GPU 골든은 spawn 자식 하나에 fabric 하나(test_pc_golden_fabric_parity 규약).
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from policy_control import chain as CH
from policy_control import contract as C
from policy_control import lockstep as LS
from policy_control import pd_law
from policy_control.pd_state import Phase

pytestmark = [pytest.mark.golden]

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
ROBOTS = SIM2REAL / "policy_control/config/robots"
CONFIG = SIM2REAL / "policy_control/config"
PROFILE = RL_WS / "robot_control/src/robot_control/profiles/openarm_tesollo.yaml"
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_E1_RUN = SIM2REAL / "logs/policy/right_e1"
FX = SIM2REAL / "tests/fixtures/policy_control"
DEVICE = "cuda:0"

TOL_OBS = 1e-9
TOL_PALM = 1e-9
TOL_Q = 1e-6
RIGHT_FABRIC_NONDETERMINISM_RAD = 5e-2       # 우 fabric run-to-run 비결정 상한(실측 최대 4.1e-2 rad, 194 스텝)
TOL_PD = 1e-12
#: 우 팔 목표 허용치 = max(TOL_Q, 이 배수 × 오라클 재실행 차이) — fabric 비결정을 체인 오차로 오인하지 않게
RIGHT_FABRIC_NOISE_FACTOR = 3.0

needs_left = pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")
needs_e1 = pytest.mark.skipif(not (RIGHT_E1_RUN / "nn").exists(), reason="right_e1 run dir 없음")
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음 — fabrics_sim 은 cuda 전용")


def _isolated(fn, *args):
    """★fabric 하나 = 프로세스 하나 (test_pc_golden_fabric_parity 와 같은 이유)."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(fn, args)


def _pd_stage(config_path: Path, contract: C.DeployContract, gravity_fn=None) -> CH.PdStage:
    config = pd_law.load_pd_config(config_path)
    lower, upper, _ = pd_law.limits_from_profile(PROFILE, contract.pd.sim_gains.joints)
    return CH.PdStage(ramp_cfg=pd_law.law_cfg_from_config(config, contract, "ramp", lower, upper),
                      track_cfg=pd_law.law_cfg_from_config(config, contract, "full", lower, upper),
                      watchdog_sec=config.watchdog_sec, abort_tracking=config.abort_tracking,
                      ramp_tol=config.settle.tol, release_zero_ticks=config.release_zero_ticks,
                      dt=1.0 / config.pd_hz, gravity_fn=gravity_fn)


# ================================================================== LEFT
def _left_worker(contract: C.DeployContract, fixtures: Path) -> dict:
    from left_grasp_gate import GateCfg
    from left_policy_core import LeftPolicyCore, LeftSensors
    from policy_control import fabric_core as F
    from policy_control import fk_numpy, sources

    stream = LS.load_stream(fixtures / "stream_left_v2b25.npz")
    states = LS.stream_states(stream, contract, unify_ee=True)
    cfg = sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml"), "live")
    fk = fk_numpy.make_fk(contract, rl_ws=RL_WS)
    core = F.FabricCore(contract, DEVICE)
    gp = contract.obs.segment("gripper_gate").params
    run = SIM2REAL / contract.run.dir
    actions = stream.actions.astype(np.float64)

    def oracle_pass() -> dict:
        core.reset(np.asarray(contract.fabric.home_q))
        box = {"t": 0}
        oracle = LeftPolicyCore(
            policy=lambda obs: actions[box["t"]], fabric=lambda p: core.step(p).q_arm,
            run_env_yaml=run / "params/env.yaml", goal7=stream.goal[0].astype(np.float64),
            gate_cfg=GateCfg(pad_offset=gp["pad_offset"], lateral_ok=gp["lateral_ok"], along_ok=gp["along_ok"],
                             band_axis=tuple(gp["band_axis"]), release_lateral=gp["release_lateral"]),
            urdf_path=RL_WS / contract.obs.fk["urdf"])
        oracle.reset()
        rows = {"obs": [], "palm": [], "grip": [], "q": [], "gate": []}
        for t, st in enumerate(states):
            box["t"] = t
            tick = oracle.step(LeftSensors(arm_q=st.arm_q, arm_qd=st.arm_qd, grip_q=float(st.ee_q[0]),
                                           grip_qd=float(st.ee_qd[0]), cup_pos=st.object_pos, cup_quat=IDENT))
            for key, val in (("obs", tick.obs), ("palm", tick.palm_target), ("grip", tick.gripper_cmd),
                             ("q", tick.arm_q_target), ("gate", tick.gate_open)):
                rows[key].append(val)
        return {k: np.asarray(v) for k, v in rows.items()}

    a = oracle_pass()
    policy = LS.recorded_policy(contract, stream.actions)
    table = CH.TableGuard.from_robot_cfg(cfg.table)
    trace = LS.run_lockstep(contract, cfg, states, policy, core, table, fk=fk,
                            pd=_pd_stage(CONFIG / "pd_left.yaml", contract), goal=stream.goal[0].astype(np.float64),
                            stop_on_abort=False)
    a2 = oracle_pass()
    return {"oracle": a, "oracle_again": a2, "trace": trace, "actions": actions}


@needs_left
@pytest.mark.gpu
def test_left_chain_matches_left_policy_core(fixtures_dir):
    _cuda_or_skip()
    contract = C.load_contract(LEFT_JSON)
    r = _isolated(_left_worker, contract, fixtures_dir)
    trace: LS.Trace = r["trace"]
    a, a2 = r["oracle"], r["oracle_again"]
    T = len(a["obs"])
    assert len(trace.steps) == T and all(s.valid for s in trace.steps)
    obs, palm, q = trace.column("obs"), trace.column("palm6"), trace.column("q_arm")
    grip = np.array([s.gripper_cmd for s in trace.steps])
    gate = np.array([s.gate_open for s in trace.steps])
    det = float(np.abs(a["q"] - a2["q"]).max())
    print(f"\n[left chain] {T} steps · obs {np.abs(obs - a['obs']).max():.2e} · palm {np.abs(palm - a['palm']).max():.2e}"
          f" · q {np.abs(q - a['q']).max():.2e} rad · fabric 결정론(오라클 재실행) {det:.2e} rad"
          f" · 여유 min {trace.column('clearance').min() * 1e3:.1f} mm · 게이트 열림 {int(gate.sum())}/{T}")
    assert np.abs(obs - a["obs"]).max() <= TOL_OBS
    assert np.abs(palm - a["palm"]).max() <= TOL_PALM
    assert np.array_equal(grip, a["grip"]) and np.array_equal(gate, a["gate"])
    assert gate.any(), "스트림에서 게이트가 한 번도 안 열렸다(대역 불일치)"
    assert np.abs(q - a["q"]).max() <= TOL_Q, f"팔 목표 불일치 {np.abs(q - a['q']).max():.3e} (fabric 결정론 {det:.3e})"
    # 정책 대역이 기록 액션을 그대로 통과시켰다(clip 100 = 무클립)
    np.testing.assert_allclose(trace.column("action"), r["actions"], atol=0)
    assert trace.column("clearance").min() >= 0.03 and not trace.aborted


@needs_left
@pytest.mark.gpu
def test_left_chain_pd_follows_the_law(fixtures_dir):
    _cuda_or_skip()
    contract = C.load_contract(LEFT_JSON)
    r = _isolated(_left_worker, contract, fixtures_dir)
    trace: LS.Trace = r["trace"]
    config = pd_law.load_pd_config(CONFIG / "pd_left.yaml")
    lower, upper, _ = pd_law.limits_from_profile(PROFILE, contract.pd.sim_gains.joints)
    cfg = pd_law.law_cfg_from_config(config, contract, "full", lower, upper)
    _assert_pd_law(trace, cfg, config.pd_hz)
    _assert_no_kd_over_kp_term(trace, contract)


def _assert_pd_law(trace: LS.Trace, cfg: pd_law.PdLawCfg, pd_hz: float) -> None:
    from jtc_bridge_core import velocity_limited_target

    ticks = [pt for s in trace.steps for pt in s.pd]
    assert ticks and all(pt.phase is Phase.TRACKING for pt in ticks), {pt.phase for pt in ticks}
    assert sum(pt.new_policy_step for pt in ticks) == len(trace.steps)      # 목표당 정확히 1회
    dt = 1.0 / pd_hz
    max_lead = cfg.lead_vel * cfg.lead_sec
    for pt in ticks:
        q_t = np.clip(pt.target.q, cfg.lower, cfg.upper)
        sp = velocity_limited_target(q_t, pt.before.q_setpoint, cfg.max_vel, dt)
        sp = pt.q_meas + np.clip(sp - pt.q_meas, -max_lead, max_lead)
        np.testing.assert_allclose(pt.after.q_setpoint, sp, atol=TOL_PD)
        droop = pt.before.droop
        if pt.new_policy_step:
            droop = np.clip(droop + cfg.droop_gain * (q_t - pt.q_meas), -cfg.droop_limit, cfg.droop_limit)
        np.testing.assert_allclose(pt.after.droop, droop, atol=TOL_PD)
        np.testing.assert_allclose(pt.cmd.q, np.clip(sp + droop, cfg.lower, cfg.upper), atol=TOL_PD)
        np.testing.assert_allclose(pt.cmd.qd, np.clip(cfg.vel_ff_scale * pt.target.qd, -cfg.vel_ff_cap, cfg.vel_ff_cap),
                                   atol=TOL_PD)
        np.testing.assert_array_equal(pt.cmd.tau, np.zeros_like(pt.cmd.tau))        # integral_droop: τ_ff 0


def _replay_pd(trace: LS.Trace, contract: C.DeployContract, zero_qd: bool) -> np.ndarray:
    """trace 의 목표를 새 PdStage 에 다시 흘린 q 명령 (T·n_pd, 7). zero_qd 면 q̇* 를 0 으로 바꿔 흘린다."""
    ticks = [pt for s in trace.steps for pt in s.pd]
    pd = _pd_stage(CONFIG / "pd_left.yaml", contract)
    pd.engage(trace.steps[0].arm_q_meas)
    pd.tick(CH.PdTarget(q=pd.state.law.q_setpoint, qd=np.zeros(7), tau_ff=np.zeros(7), seq=-1, t_recv=0.0),
            trace.steps[0].arm_q_meas, np.zeros(7), now=0.0)
    pd.new_episode(1)
    rows = []
    for pt in ticks:
        tgt = pt.target if not zero_qd else CH.PdTarget(q=pt.target.q, qd=np.zeros(7), tau_ff=pt.target.tau_ff,
                                                        seq=pt.target.seq, t_recv=pt.target.t_recv)
        rows.append(pd.tick(tgt, pt.q_meas, pt.qd_meas, now=pt.now).cmd.q)
    return np.stack(rows)


def _assert_no_kd_over_kp_term(trace: LS.Trace, contract: C.DeployContract) -> None:
    """같은 목표를 q̇*=0 으로 다시 흘려도 q 명령이 같다 — (kd/kp)·q̇* 위치 보정이 없다."""
    ticks = [pt for s in trace.steps for pt in s.pd]
    assert any(np.abs(pt.target.qd).max() > 0.0 for pt in ticks)
    ref = _replay_pd(trace, contract, zero_qd=False)
    np.testing.assert_allclose(ref, np.stack([pt.cmd.q for pt in ticks]), atol=0)      # trace 재현(결정론)
    np.testing.assert_allclose(_replay_pd(trace, contract, zero_qd=True), ref, atol=0)


# ================================================================== RIGHT (e1)
def _right_worker(run_dir: Path, fixtures: Path) -> dict:
    from grasp_s2r_core import DOF_TO_PROFILE, PROFILE_TO_DOF, GraspS2RCore, S2RSensors
    from grasp_s2r_fabric import permutation
    from grasp_s2r_obs_builder import hand_dof_order
    from policy_control import contract_build as B
    from policy_control import fabric_core as F
    from policy_control import fk_numpy, pd_gravity, sources

    contract = B.build_contract(run_dir)
    stream = LS.load_stream(fixtures / "stream_right_e1_v2.npz")
    states = LS.stream_states(stream, contract)
    cfg = sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml"), "live")
    core = F.FabricCore(contract, DEVICE)
    fk = fk_numpy.make_fk(contract, rl_ws=RL_WS, palm_pose_fn=core.palm_pose, tips_fn=core.tips)
    soft = np.asarray(contract.action.hand.params["soft_limits"], dtype=np.float64)
    actions = stream.actions.astype(np.float64)
    hand_names = list(stream.hand_names)
    prof_to_dof = np.array([hand_names.index(n) for n in contract.obs.joint_orders["hand_obs"]])
    np.testing.assert_array_equal(prof_to_dof, PROFILE_TO_DOF)
    dof_to_fab = permutation(hand_dof_order("r"), contract.fabric.joint_order[7:])
    home = np.asarray(contract.fabric.home_q)

    def oracle_pass() -> dict:
        core.reset(home)
        box = {"t": 0}
        holder = {}
        oracle = GraspS2RCore(
            policy=lambda obs: actions[box["t"]], fabric_palm_pose=core.palm_pose, fabric_tips=core.tips,
            fabric_step=lambda p: core.step(p, hand_target=holder["o"].hand.target).q_arm,
            run_dir=run_dir, goal3=stream.goal[0].astype(np.float64), soft_limits=soft, hand_dof_to_fabric=dof_to_fab)
        holder["o"] = oracle
        st0 = states[0]
        oracle.reset(arm_q=st0.arm_q, hand_q=st0.ee_q[prof_to_dof], object_pos=st0.object_pos)
        rows = {"obs": [], "palm": [], "hand": [], "q": [], "gate": []}
        for t, st in enumerate(states):
            box["t"] = t
            tick = oracle.step(S2RSensors(arm_q=st.arm_q, arm_qd=st.arm_qd, hand_q=st.ee_q[prof_to_dof],
                                          hand_qd=st.ee_qd[prof_to_dof], object_pos=st.object_pos,
                                          tip_force_world=st.tip_force, tip_quat=np.tile(IDENT, (5, 1))))
            for key, val in (("obs", tick.obs), ("palm", tick.palm_target), ("hand", tick.hand_q_target[DOF_TO_PROFILE]),
                             ("q", tick.arm_q_target), ("gate", tick.close_gate)):
                rows[key].append(val)
        return {k: np.asarray(v) for k, v in rows.items()}

    a = oracle_pass()
    policy = LS.recorded_policy(contract, stream.actions)
    pd_cfg = pd_law.load_pd_config(CONFIG / "pd_right.yaml")
    pd = _pd_stage(CONFIG / "pd_right.yaml", contract, gravity_fn=pd_gravity.make_gravity(pd_cfg.gravity, contract))
    table = CH.TableGuard.from_robot_cfg(cfg.table)
    trace = LS.run_lockstep(contract, cfg, states, policy, core, table, fk=fk, pd=pd,
                            goal=stream.goal[0].astype(np.float64), stop_on_abort=False)
    a2 = oracle_pass()
    return {"oracle": a, "oracle_again": a2, "trace": trace}


@needs_e1
@pytest.mark.gpu
def test_right_chain_matches_grasp_s2r_core(fixtures_dir):
    _cuda_or_skip()
    r = _isolated(_right_worker, RIGHT_E1_RUN, fixtures_dir)
    trace: LS.Trace = r["trace"]
    a, a2 = r["oracle"], r["oracle_again"]
    T = len(a["obs"])
    assert len(trace.steps) == T and all(s.valid for s in trace.steps)
    obs, palm, q = trace.column("obs"), trace.column("palm6"), trace.column("q_arm")
    hand = trace.column("hand_target")
    gate = trace.column("close_gate")
    det = float(np.abs(a["q"] - a2["q"]).max())
    clear = trace.column("clearance")
    print(f"\n[right chain] {T} steps · obs {np.abs(obs - a['obs']).max():.2e} · palm {np.abs(palm - a['palm']).max():.2e}"
          f" · hand {np.abs(hand - a['hand']).max():.2e} · q {np.abs(q - a['q']).max():.2e} rad"
          f" · fabric 결정론 {det:.2e} rad · 손끝 여유 min {clear.min() * 1e3:.1f} mm"
          f" (30 mm 미만 {int((clear < 0.03).sum())} 스텝) · gate max {gate.max():.2f}")
    assert np.abs(obs - a["obs"]).max() <= TOL_OBS
    assert np.abs(palm - a["palm"]).max() <= TOL_PALM
    np.testing.assert_allclose(hand, a["hand"], atol=0)
    np.testing.assert_allclose(gate, a["gate"], atol=0)
    # ★우 fabric(27관절·body repulsion pairs·float32 warp)의 run-to-run 비결정은 194 스텝 뒤 2e-4~4e-2 rad 로
    #   실행마다 크게 흔들려 '오라클 재실행 차이 × 배수' 만으로는 판정이 불안정하다(09.06 실측: A/A′ 6e-4 인데
    #   체인 3.3e-2). 입력(obs·palm·손 목표)이 0.0 으로 같으므로 팔 목표는 관측된 비결정 상한(4.1e-2 rad)
    #   위의 고정 문턱으로 잠근다. 우 실기는 재학습 후라 이 문턱은 파이프라인 등가 증거로만 쓴다.
    tol_q = max(TOL_Q, RIGHT_FABRIC_NOISE_FACTOR * det, RIGHT_FABRIC_NONDETERMINISM_RAD)
    err_q = float(np.abs(q - a["q"]).max())
    assert err_q <= tol_q, f"팔 목표 불일치 {err_q:.3e} > {tol_q:.3e} (fabric 자체 비결정 {det:.3e})"
    ticks = [pt for s in trace.steps for pt in s.pd]
    assert all(pt.phase is Phase.TRACKING for pt in ticks)
    assert sum(pt.new_policy_step for pt in ticks) == T
