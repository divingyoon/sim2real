"""M2 — 골든(1): 동결 스트림을 ObsCore 로 재생해 sim 기록·오라클 코어와 세그먼트별로 대조.

스트림 행 t 의 의미(numpy 로 실측):
  obs[t] 는 스텝 t 시작 시 관측, actions[t] 는 그 관측으로 낸 액션, arm_meas/hand_meas/
  cup_pos3[t] 는 스텝 t **물리 뒤** 실측 → obs[t+1] 의 입력이다. obs[0] 은 홈·개방·spawn.
  속도는 기록에 없어 obs 의 속도 칸을 그대로 센서로 먹인다(값을 만들지 않는다).

좌(v2B25, 65 스텝): (a) sim 기록 ≤1e-5 (b) LeftPolicyCore ≤1e-9.
  ★sim 기록과의 알려진 차이 두 가지는 코드가 아니라 **기록의 한계**다:
    · `gripper_gate` — env 는 액션 적용 시점(물리 전)에 게이트를 갱신해 그 값이 그 스텝의
      obs 에 실리므로 배포(현재 센서로 갱신)보다 **한 스텝 늦다**. 배포 규약은 09.03 실기로
      확정된 LeftPolicyCore 와 동일하므로 게이트 칸은 한 스텝 시프트로 대조한다.
    · `cup_upright` — 스트림에 컵 쿼터니언이 없다(컵이 파지 중 기울어 0.904 까지 내려감).
      단위 쿼터니언을 먹이므로 이 칸은 대조에서 제외한다.
  ★계약의 `band_axis`(80~150 mm, v2 트랙 이름으로 고른 값)로는 기록된 게이트가 **한 번도
    열리지 않는다**(기록은 판 위 62 mm 에서 열림 = v1 대역 10~85 mm). v2B25 는 09.03 대역
    상향 **이전** 체크포인트라 v1 대역으로 학습됐다 — 계약 생성기의 결함(open issue).
    골든은 기록이 증명하는 v1 대역으로 대조하고, 계약값 그대로는 xfail(strict) 로 잠근다.

우(e1, 194 스텝): 기록 obs 는 **관측 노이즈 포함**(σ_q 0.01·σ_obj 0.015, env.yaml) 이라
  노이즈 없는 칸(actions·FK 파생)만 기록과 대조하고, 전 칸은 GraspS2RCore 오라클과 ≤1e-9.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import contract_build as B
from policy_control import fk_numpy, obs_core, sources

pytestmark = [pytest.mark.golden, pytest.mark.unit]

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
ROBOTS = SIM2REAL / "policy_control/config/robots"
LEFT_CONTRACT = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_E1_RUN = SIM2REAL / "logs/policy/right_e1"

needs_left = pytest.mark.skipif(not LEFT_CONTRACT.exists(), reason="left contract 없음")
needs_e1 = pytest.mark.skipif(not (RIGHT_E1_RUN / "nn").exists(), reason="right_e1 run dir 없음")

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
HAND_PROFILE = [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)]
TIPS = tuple(f"r_hl_{f}_tip" for f in ("thumb", "index", "middle", "ring", "pinky"))
#: 기록이 증명하는 v2B25 의 파지 대역(v1: 판 위 10~85 mm, 컵 원점 기준 축좌표)
V2B25_BAND_AXIS = (0.010 - 0.09209, 0.085 - 0.09209)


def _assert_segments(name: str, got: np.ndarray, want: np.ndarray, contract, tol: float,
                     skip=(), shift: dict | None = None) -> None:
    """(T, obs_dim) 두 배열을 세그먼트별로 대조하고 첫 불일치 세그먼트를 이름으로 알린다."""
    shift = shift or {}
    off = 0
    for s in contract.obs.segments:
        sl = slice(off, off + s.dim)
        off += s.dim
        if s.name in skip:
            continue
        g, w = got[:, sl], want[:, sl]
        if s.name in shift:                    # g[t] == w[t + k]
            k = shift[s.name]
            g, w = g[:-k], w[k:]
        d = float(np.abs(g - w).max())
        assert d <= tol, f"[{name}] segment {s.name!r} max|Δ|={d:.3e} > {tol} (first bad step {int(np.abs(g - w).max(axis=1).argmax())})"


def _with_band(c, band) -> C.DeployContract:
    segs = tuple(dataclasses.replace(s, params={**s.params, "band_axis": list(band)}) if s.name == "gripper_gate" else s
                 for s in c.obs.segments)
    return dataclasses.replace(c, obs=dataclasses.replace(c.obs, segments=segs))


# ================================================================== LEFT
def _left_replay(contract, fixtures_dir):
    d = np.load(fixtures_dir / "stream_left_v2b25.npz")
    obs_rec = d["obs"].astype(float)
    T = obs_rec.shape[0]
    cfg = sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml"), "live")
    fk = fk_numpy.make_fk(contract, rl_ws=RL_WS)
    core = obs_core.ObsCore(contract, cfg, fk)
    home = np.array(contract.pd.home_arm)
    grip_open = float(contract.action.hand.params["open"])

    def state_at(t):
        if t == 0:
            q, g, cup = home, np.full(2, grip_open), d["meta_cup_spawn"].astype(float)
        else:
            q, g, cup = d["arm_meas"][t - 1].astype(float), d["hand_meas"][t - 1].astype(float), d["cup_pos3"][t - 1].astype(float)
        return sources.RobotState(arm_q=q, arm_qd=obs_rec[t, 9:16].copy(), ee_names=("l_hj_gripper_1", "l_hj_gripper_2"),
                                  ee_q=g, ee_qd=obs_rec[t, 16:18].copy(), object_pos=cup, object_quat=IDENT,
                                  tip_force=None, tip_names=(), head=None, decoder_target=None,
                                  stamps={}, stale=(), missing=())

    st0 = state_at(0)
    core.reset(st0, goal=d["goal"][0].astype(float))
    outs, states = [], []
    for t in range(T):
        st = state_at(t)
        last = None if t == 0 else d["actions"][t - 1].astype(float)
        out = core.tick(st, last_action=last)
        assert out.valid, out.reasons
        outs.append(out)
        states.append(st)
    return d, np.stack([o.obs for o in outs]), outs, states


@needs_left
def test_left_golden_vs_recorded_obs(fixtures_dir):
    contract = _with_band(C.load_contract(LEFT_CONTRACT), V2B25_BAND_AXIS)
    d, got, outs, _ = _left_replay(contract, fixtures_dir)
    rec = d["obs"].astype(float)
    _assert_segments("left/recorded", got, rec, contract, tol=1e-5,
                     skip=("cup_upright",), shift={"gripper_gate": 1})
    # 게이트가 실제로 열리는 스트림이다(대역이 맞다는 증거)
    assert rec[:, 35].max() == 1.0 and got[:, 35].max() == 1.0
    assert int(np.argmax(got[:, 35] > 0)) + 1 == int(np.argmax(rec[:, 35] > 0))


@needs_left
def test_left_golden_with_contract_band_as_built(fixtures_dir):
    """계약이 v1 대역(--grasp-band v1)으로 재생성된 뒤에는 대역 치환 없이도 기록과 맞아야 한다."""
    contract = C.load_contract(LEFT_CONTRACT)
    d, got, _, _ = _left_replay(contract, fixtures_dir)
    _assert_segments("left/contract-band", got, d["obs"].astype(float), contract, tol=1e-5,
                     skip=("cup_upright",), shift={"gripper_gate": 1})


@needs_left
def test_left_golden_vs_left_policy_core_oracle(fixtures_dir):
    from left_grasp_gate import GateCfg
    from left_policy_core import LeftPolicyCore, LeftSensors

    contract = _with_band(C.load_contract(LEFT_CONTRACT), V2B25_BAND_AXIS)
    d, got, outs, states = _left_replay(contract, fixtures_dir)
    run = SIM2REAL / contract.run.dir
    gp = contract.obs.segment("gripper_gate").params
    actions = d["actions"].astype(float)
    t_box = {"t": 0}
    oracle = LeftPolicyCore(
        policy=lambda obs: actions[t_box["t"]], fabric=None,
        run_env_yaml=run / "params/env.yaml", goal7=d["goal"][0].astype(float),
        gate_cfg=GateCfg(pad_offset=gp["pad_offset"], lateral_ok=gp["lateral_ok"], along_ok=gp["along_ok"],
                         band_axis=tuple(V2B25_BAND_AXIS), release_lateral=gp["release_lateral"]),
        urdf_path=RL_WS / contract.obs.fk["urdf"])
    oracle.reset()
    want = []
    for t, st in enumerate(states):
        # LeftSensors 는 그리퍼 한 값(mimic)만 받는다 — 스트림의 두 값이 다르므로(4e-5) 그리퍼
        # 속도/위치를 첫 값으로 통일해 두 코어에 같은 입력을 준다.
        sens = LeftSensors(arm_q=st.arm_q, arm_qd=st.arm_qd, grip_q=float(st.ee_q[0]), grip_qd=float(st.ee_qd[0]),
                           cup_pos=st.object_pos, cup_quat=IDENT)
        t_box["t"] = t
        want.append(oracle.step(sens).obs)
    want = np.stack(want)
    # 같은 입력(그리퍼 두 값 동일)으로 ObsCore 를 다시 돌린다
    cfg = sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "left_gripper_real.yaml"), "live")
    core = obs_core.ObsCore(contract, cfg, fk_numpy.make_fk(contract, rl_ws=RL_WS))
    core.reset(states[0], goal=d["goal"][0].astype(float))
    mine = []
    for t, st in enumerate(states):
        st2 = dataclasses.replace(st, ee_q=np.full(2, float(st.ee_q[0])), ee_qd=np.full(2, float(st.ee_qd[0])))
        mine.append(core.tick(st2, last_action=None if t == 0 else actions[t - 1]).obs)
    _assert_segments("left/oracle", np.stack(mine), want, contract, tol=1e-9)


# ================================================================== RIGHT (e1)
class _RecordedFK:
    """기록 obs 의 palm_pos/palm_ax/tips_rel_palm 로 만든 FK 대역 — 두 코어에 같은 값을 준다."""

    def __init__(self, obs_rec: np.ndarray):
        self.obs = obs_rec
        self.t = 0

    def _R(self, t):
        c0, c1 = self.obs[t, 57:60], self.obs[t, 60:63]
        return np.stack([c0, c1, np.cross(c0, c1)], axis=1)

    def palm6(self, q):
        R = self._R(self.t)
        ey = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        ez = np.arctan2(R[1, 0], R[0, 0])
        ex = np.arctan2(R[2, 1], R[2, 2])
        return np.concatenate([self.obs[self.t, 54:57], [ez, ey, ex]])

    def tips(self, q):
        return self.obs[self.t, 63:78].reshape(5, 3) + self.obs[self.t, 54:57]


@needs_e1
def test_right_golden_e1_vs_recorded_and_oracle(fixtures_dir):
    from grasp_s2r_core import DOF_TO_PROFILE, GraspS2RCore, S2RSensors

    contract = B.build_contract(RIGHT_E1_RUN)
    d = np.load(fixtures_dir / "stream_right_e1_v2.npz")
    obs_rec = d["obs"].astype(float)
    T = obs_rec.shape[0]
    assert list(d["meta_hand_names"]) == HAND_PROFILE
    actions = d["actions"].astype(float)
    hand_obs_order = contract.obs.joint_orders["hand_obs"]
    prof_to_dof = np.array([HAND_PROFILE.index(n) for n in hand_obs_order])
    dof_to_prof = np.array([list(hand_obs_order).index(n) for n in HAND_PROFILE])
    np.testing.assert_array_equal(dof_to_prof, DOF_TO_PROFILE)

    fk_stub = _RecordedFK(obs_rec)
    cfg = sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "right_dg5f_real.yaml"), "live")
    core = obs_core.ObsCore(contract, cfg, fk_numpy.make_fk(contract, rl_ws=RL_WS, palm_pose_fn=fk_stub.palm6, tips_fn=fk_stub.tips))
    t_box = {"t": 0}
    oracle = GraspS2RCore(policy=lambda obs: actions[t_box["t"]], fabric_palm_pose=fk_stub.palm6,
                          fabric_tips=fk_stub.tips, fabric_step=lambda palm6: np.zeros(7),
                          run_dir=RIGHT_E1_RUN, goal3=d["goal"][0].astype(float))
    home = np.array(contract.pd.home_arm)
    open_pose = np.array(contract.action.hand.params["open_pose"])
    spawn = d["meta_cup_spawn"].astype(float)

    def state_at(t):
        if t == 0:
            q, h, cup = home, open_pose, spawn
        else:
            q, h, cup = d["arm_meas"][t - 1].astype(float), d["hand_meas"][t - 1].astype(float), d["cup_pos3"][t - 1].astype(float)
        hand_qd_dof = obs_rec[t, 34:54]
        return sources.RobotState(arm_q=q, arm_qd=obs_rec[t, 7:14].copy(), ee_names=tuple(HAND_PROFILE), ee_q=h,
                                  ee_qd=hand_qd_dof[dof_to_prof].copy(), object_pos=cup, object_quat=IDENT,
                                  tip_force=obs_rec[t, 96:111].reshape(5, 3) * 10.0, tip_names=TIPS, head=None,
                                  decoder_target=None, stamps={}, stale=(), missing=())

    st0 = state_at(0)
    core.reset(st0, goal=d["goal"][0].astype(float))
    oracle.reset(arm_q=st0.arm_q, hand_q=st0.ee_q[prof_to_dof], object_pos=st0.object_pos)
    mine, want = [], []
    for t in range(T):
        fk_stub.t = t
        t_box["t"] = t
        st = state_at(t)
        target_prev = oracle.hand.target.copy()                  # 직전 tick 의 시너지 목표(프로필 순)
        out = core.tick(st, last_action=None if t == 0 else actions[t - 1], decoder_target=target_prev)
        assert out.valid, out.reasons
        mine.append(out.obs)
        sens = S2RSensors(arm_q=st.arm_q, arm_qd=st.arm_qd, hand_q=st.ee_q[prof_to_dof], hand_qd=st.ee_qd[prof_to_dof],
                          object_pos=st.object_pos, tip_force_world=st.tip_force,
                          tip_quat=np.tile(IDENT, (5, 1)))
        want.append(oracle.step(sens).obs)
    mine, want = np.stack(mine), np.stack(want)
    _assert_segments("right/oracle", mine, want, contract, tol=1e-9)
    # 노이즈 없는 칸만 기록과 직접 대조: actions, 그리고 FK 대역을 거친 palm/tips 칸(euler↔quat 왕복 검증)
    noisy = {"arm_q", "arm_qd", "hand_q", "hand_qd", "palm_to_obj", "obj_to_tips", "goal_rel", "joint_err", "tip_force"}
    _assert_segments("right/recorded", mine, obs_rec, contract, tol=1e-5, skip=noisy)


# ================================================================== RIGHT — dg5f-m 자산 URDF FK 교차검증 (09.06)
DG5FM_CONTRACT = SIM2REAL / "logs/policy/right_g1/deploy_contract.dg5f-m.json"
needs_dg5fm = pytest.mark.skipif(not DG5FM_CONTRACT.exists(), reason="dg5f-m 계약 없음")
FK_SEGMENTS = ("palm_pos", "palm_ax", "tips_rel_palm", "palm_to_obj", "obj_to_tips")
#: 노이즈 없는 기록(g1 hdf5 `palm_pose`) 대 자산 URDF FK — 옛 sensor 자산과 새 dg5f-m 자산의 palm 프레임이
#: 같다는 문턱. 실측 pos 9.2e-7 m / 회전행렬 1.7e-6 (float32 기록 정밀도).
ASSET_FRAME_M = 5e-6
ASSET_FRAME_ROT = 5e-6
#: 노이즈 문턱 배수: 194 스텝×3축 표본의 최대가 넘지 않을 k·σ (측정 최대 ≈ 2.8σ / 2.4σ·√2).
NOISE_K = 5.0
#: 평균 |Δ| 는 σ·√(2/π)=0.80σ 여야 한다(계통 편차 없음). 1.5σ 넘으면 프레임이 움직인 것.
NOISE_MEAN_K = 1.5


def _env_scalar(env_yaml: Path, key: str) -> float:
    import re
    m = re.search(rf"^\s*{key}:\s*(.+?)\s*$", env_yaml.read_text(), re.M)
    if m is None:
        raise KeyError(f"{env_yaml} 에 {key} 가 없다")
    return float(m.group(1))


def _seg_slice(contract, name: str) -> slice:
    off = 0
    for s in contract.obs.segments:
        if s.name == name:
            return slice(off, off + s.dim)
        off += s.dim
    raise KeyError(name)


def _e1_replay_with_asset_fk(contract, fixtures_dir):
    """e1 스트림을 dg5f-m 자산 URDF FK 로 다시 조립한다. 컵은 **기록이 관측한(노이즈 포함) 컵**
    (= palm_pos + palm_to_obj)을 먹여 물체 노이즈(σ 15 mm)를 빼고 FK 차이만 남긴다."""
    d = np.load(fixtures_dir / "stream_right_e1_v2.npz")
    obs_rec = d["obs"].astype(float)
    T = obs_rec.shape[0]
    assert list(d["meta_hand_names"]) == HAND_PROFILE
    fk = fk_numpy.make_fk(contract, rl_ws=RL_WS, kind="urdf_chain")
    cfg = sources.with_object_mode(sources.load_robot_cfg(ROBOTS / "dg5f_m_right_real.yaml"), "live")
    core = obs_core.ObsCore(contract, cfg, fk, side="right")
    home = np.array(contract.pd.home_arm)
    open_pose = np.array(contract.action.hand.params["open_pose"])
    palm, p2o = _seg_slice(contract, "palm_pos"), _seg_slice(contract, "palm_to_obj")

    def state_at(t):
        q, h = (home, open_pose) if t == 0 else (d["arm_meas"][t - 1].astype(float), d["hand_meas"][t - 1].astype(float))
        return sources.RobotState(arm_q=q, arm_qd=obs_rec[t, 7:14].copy(), ee_names=tuple(HAND_PROFILE), ee_q=h,
                                  ee_qd=np.zeros(20), object_pos=obs_rec[t, palm] + obs_rec[t, p2o], object_quat=IDENT,
                                  tip_force=obs_rec[t, 96:111].reshape(5, 3) * 10.0, tip_names=TIPS, head=None,
                                  decoder_target=None, stamps={}, stale=(), missing=())

    core.reset(state_at(0), goal=d["goal"][0].astype(float))
    mine = np.stack([core.tick(state_at(t), None).obs for t in range(T)])
    return mine, obs_rec


@needs_dg5fm
@needs_e1
def test_right_e1_fk_segments_from_dg5f_m_asset_urdf_vs_recorded(fixtures_dir):
    """★기록의 body 채널은 노이즈가 있다(`obs_noise_body` 0.005 = σ 5 mm, palm/손끝 각각) — 따라서 이 대조는
    FK 프레임이 노이즈 폭 안에 있는지(최대 ≤ NOISE_K·σ, 평균 ≤ 1.5·0.8σ)만 잠근다. 회전(palm_ax)은 노이즈가
    없어 1e-5 로 잠근다. 실측(09.06, 194 스텝): palm_pos max 13.8 mm·mean 3.9 mm, palm_ax 1.5e-6,
    tips_rel_palm max 17.2 mm·mean 4.0 mm, palm_to_obj·obj_to_tips (관측 컵 기준) = palm_pos 와 동일 13.8 mm."""
    contract = C.load_contract(DG5FM_CONTRACT)
    sigma = _env_scalar(fixtures_dir / "runs/right_e1/env.yaml", "obs_noise_body")
    mine, rec = _e1_replay_with_asset_fk(contract, fixtures_dir)
    report = {}
    for name in FK_SEGMENTS:
        sl = _seg_slice(contract, name)
        dlt = np.abs(mine[:, sl] - rec[:, sl])
        report[name] = (float(dlt.max()), float(dlt.mean()))
    print("[golden dg5f-m FK vs e1 record] " + "; ".join(f"{k} max {v[0]:.2e} mean {v[1]:.2e}" for k, v in report.items()))
    assert report["palm_ax"][0] <= 1e-5, report
    # 팁 채널은 두 노이즈 점의 차(σ√2); palm_to_obj/obj_to_tips 는 관측 컵을 먹여 palm/tip 노이즈만 남는다
    scale = {"palm_pos": 1.0, "tips_rel_palm": np.sqrt(2.0), "palm_to_obj": 1.0, "obj_to_tips": np.sqrt(2.0)}
    for name, k in scale.items():
        mx, mean = report[name]
        assert mx <= NOISE_K * k * sigma, (name, mx, NOISE_K * k * sigma)
        assert mean <= NOISE_MEAN_K * 0.8 * k * sigma, (name, mean, NOISE_MEAN_K * 0.8 * k * sigma)


@needs_dg5fm
def test_right_g1_hdf5_palm_pose_matches_dg5f_m_asset_urdf_fk(fixtures_dir):
    """노이즈 없는 기록: g1 재생(옛 sensor 자산, 09.03) 의 sim palm 자세 598 스텝 = 새 자산 URDF FK.
    실측 pos 9.2e-7 m, 회전 1.7e-6 → 09.05 자산 재생성은 palm 프레임을 옮기지 않았다."""
    import h5py
    from grasp_s2r_obs_builder import quat_to_matrix, reorder

    contract = C.load_contract(DG5FM_CONTRACT)
    fk = fk_numpy.make_fk(contract, rl_ws=RL_WS, kind="urdf_chain")
    with h5py.File(fixtures_dir / "g1_y00.hdf5") as h:
        assert str(h.attrs["palm_body"]) == fk.palm_body == "r_hl_palm"
        hand_names = [str(x) for x in h.attrs["hand_joint_names"]]
        ep = h["episodes/ep_000"]
        arm_q, hand_q, palm = ep["arm_q"][:].astype(float), ep["hand_q"][:].astype(float), ep["palm_pose"][:].astype(float)
    worst_p = worst_r = 0.0
    for t in range(arm_q.shape[0]):
        pose = fk.palm_pose(arm_q[t], reorder(hand_q[t], hand_names, HAND_PROFILE))
        worst_p = max(worst_p, float(np.abs(pose.palm_pos - palm[t, :3]).max()))
        worst_r = max(worst_r, float(np.abs(pose.extra["palm_rot"] - quat_to_matrix(palm[t, 3:7])).max()))
    print(f"[golden dg5f-m FK vs g1 hdf5 palm_pose] pos max {worst_p:.2e} m, rot max {worst_r:.2e} ({arm_q.shape[0]} steps)")
    assert worst_p <= ASSET_FRAME_M and worst_r <= ASSET_FRAME_ROT, (worst_p, worst_r)
