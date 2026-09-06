"""lockstep — 기록된 센서 스트림을 obs → policy → fabric → pd 체인에 **인프로세스·결정론**으로 흘린다.

DDS 없이 네 스테이지(chain.py)를 같은 순서로 부른다. 골든 체인 테스트(오라클 코어 대조)와
bag 오프라인 diff 가 쓴다. 노드 코드에는 결정론 분기가 없다 — 등가는 여기서 증명한다.

스트림 행 t 의 의미(tests/fixtures/policy_control/stream_*.npz, golden_obs 와 동일):
  obs[t] 는 스텝 t 시작 관측, actions[t] 는 그 관측의 액션, arm_meas/hand_meas/cup_pos3[t] 는
  스텝 t **물리 뒤** 실측 → 스텝 t+1 의 입력. 스텝 0 = 홈·개방·spawn. 속도는 기록에 없어
  obs 의 속도 칸을 그대로 센서로 먹인다(값을 만들지 않는다). 손끝 힘은 obs 의 tip_force 칸 ×
  contact_force_max(계약)로 되돌린다.

정책은 실제 체크포인트(PolicyCore)나 `recorded_policy`(기록 액션을 돌려주는 stub) 어느 쪽이든
PolicyCore 인터페이스로 받는다. fabric 은 실제 FabricCore(cuda, 프로세스당 하나) 또는 가짜
backend 를 꽂은 FabricCore.

pd 는 정책 스텝마다 `pd_ticks_per_step`(기본 round(step_dt·pd_hz)) 번 tick 한다. 실측 q 는
그 스텝의 입력 상태(스텝 시작 실측)를 모든 sub-tick 에 쓴다 — 기록에 sub-tick 실측이 없다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import _paths  # noqa: F401
from .chain import (AUTO, ChainError, FabricIn, FabricStage, ObsStage, PdStage, PdTarget, PolicyStage,
                    TableGuard)
from .contract import DeployContract
from .decoder_core import make_decoder, side_soft_limits
from .fabric_core import FabricCore
from .fk_numpy import make_fk
from .pd_law import PdCommand, PdState
from .pd_state import Phase
from .policy_core import PolicyCore
from .sources import RobotCfg, RobotState

from grasp_s2r_obs_builder import reorder  # noqa: E402

__all__ = ["Stream", "load_stream", "stream_states", "RecordedBackend", "recorded_policy", "PdTick",
           "StepRecord", "Trace", "run_lockstep"]

IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_STREAM_KEYS = ("obs", "actions", "arm_meas", "hand_meas", "cup_pos3", "goal", "meta_cup_spawn",
                "meta_arm_names", "meta_hand_names", "meta_step_dt")


# ------------------------------------------------------------------ stream
@dataclass(frozen=True)
class Stream:
    obs: np.ndarray            # (T, obs_dim)
    actions: np.ndarray        # (T, action_dim)
    arm_meas: np.ndarray       # (T, 7)
    hand_meas: np.ndarray      # (T, n_hand) hand_names 순
    cup_pos3: np.ndarray       # (T, 3)
    goal: np.ndarray           # (T, goal_dim)
    cup_spawn: np.ndarray      # (3,)
    arm_names: tuple
    hand_names: tuple
    step_dt: float
    task: str
    checkpoint: str

    @property
    def n(self) -> int:
        return int(self.obs.shape[0])


def load_stream(path: Path) -> Stream:
    d = np.load(Path(path), allow_pickle=False)
    missing = [k for k in _STREAM_KEYS if k not in d.files]
    if missing:
        raise ChainError(f"stream {path} 에 키가 없다: {missing}")
    f = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
    stream = Stream(obs=f("obs"), actions=f("actions"), arm_meas=f("arm_meas"), hand_meas=f("hand_meas"),
                    cup_pos3=f("cup_pos3"), goal=f("goal"), cup_spawn=f("meta_cup_spawn"),
                    arm_names=tuple(str(s) for s in d["meta_arm_names"]),
                    hand_names=tuple(str(s) for s in d["meta_hand_names"]), step_dt=float(d["meta_step_dt"]),
                    task=str(d["meta_task"]) if "meta_task" in d.files else "",
                    checkpoint=str(d["meta_checkpoint"]) if "meta_checkpoint" in d.files else "")
    n = stream.n
    for k in ("actions", "arm_meas", "hand_meas", "cup_pos3", "goal"):
        if getattr(stream, k).shape[0] != n:
            raise ChainError(f"stream {path}: {k} 행 수 {getattr(stream, k).shape[0]} ≠ obs {n}")
    return stream


def _segment_slices(contract: DeployContract) -> dict:
    out, off = {}, 0
    for s in contract.obs.segments:
        out[s.name] = (s, slice(off, off + s.dim))
        off += s.dim
    return out


def _find(slices: dict, builder: str, **params) -> tuple | None:
    for seg, sl in slices.values():
        if seg.builder == builder and all(seg.params.get(k) == v for k, v in params.items()):
            return seg, sl
    return None


def _velocities(contract: DeployContract, obs_row: np.ndarray, hand_names: Sequence[str]) -> tuple:
    """(arm_qd(7,), ee_qd(len(hand_names),)) — obs 행의 속도 칸을 센서로 되돌린다."""
    slices = _segment_slices(contract)
    n_arm = len(contract.obs.joint_orders["arm"])
    rel = _find(slices, "joint_vel_rel")
    if rel is not None:
        seg, sl = rel
        v = obs_row[sl] + np.asarray(seg.params["default"], dtype=np.float64)
        ee_order = list(contract.obs.joint_orders["ee"])
        return v[:n_arm].copy(), np.asarray(reorder(v[n_arm:], ee_order, list(hand_names)), dtype=np.float64)
    arm = _find(slices, "joint_vel_abs", order="arm")
    hand = _find(slices, "joint_vel_abs", order="hand_obs")
    if arm is None or hand is None:
        raise ChainError("계약에 joint_vel_rel 도 joint_vel_abs(arm, hand_obs) 도 없다 — 스트림 속도를 못 만든다")
    hand_obs = list(contract.obs.joint_orders["hand_obs"])
    return obs_row[arm[1]].copy(), np.asarray(reorder(obs_row[hand[1]], hand_obs, list(hand_names)), dtype=np.float64)


def _tip_force(contract: DeployContract, obs_row: np.ndarray) -> tuple:
    found = _find(_segment_slices(contract), "tip_force_local")
    if found is None:
        return None, ()
    seg, sl = found
    tips = tuple(contract.obs.joint_orders["tips"])
    scale = float(seg.params["contact_force_max"])
    return obs_row[sl].reshape(len(tips), -1) * scale, tips


def _open_hand(contract: DeployContract, hand_names: Sequence[str]) -> np.ndarray:
    h = contract.action.hand
    if h.decoder == "synergy":
        return np.asarray(reorder(np.asarray(h.params["open_pose"], dtype=np.float64), list(h.joints), list(hand_names)),
                          dtype=np.float64)
    return np.full(len(hand_names), float(h.params["open"]))


def stream_states(stream: Stream, contract: DeployContract, unify_ee: bool = False) -> tuple:
    """행 규약대로 RobotState 를 만든다. unify_ee: 좌 그리퍼 두 값을 첫 값으로 통일(mimic 오라클과 같은 입력)."""
    if tuple(stream.arm_names) != tuple(contract.obs.joint_orders["arm"]):
        raise ChainError(f"stream arm_names {stream.arm_names} ≠ 계약 obs.joint_orders.arm")
    home = np.asarray(contract.pd.home_arm, dtype=np.float64)
    open_hand = _open_hand(contract, stream.hand_names)
    states = []
    for t in range(stream.n):
        if t == 0:
            q, h, cup = home, open_hand, stream.cup_spawn
        else:
            q, h, cup = stream.arm_meas[t - 1], stream.hand_meas[t - 1], stream.cup_pos3[t - 1]
        arm_qd, ee_qd = _velocities(contract, stream.obs[t], stream.hand_names)
        tip_force, tip_names = _tip_force(contract, stream.obs[t])
        if unify_ee:
            h, ee_qd = np.full(h.shape, float(h[0])), np.full(ee_qd.shape, float(ee_qd[0]))
        states.append(RobotState(arm_q=q.copy(), arm_qd=arm_qd, ee_names=tuple(stream.hand_names), ee_q=h.copy(),
                                 ee_qd=ee_qd, object_pos=cup.copy(), object_quat=IDENT_QUAT.copy(),
                                 tip_force=tip_force, tip_names=tip_names, head=None, decoder_target=None,
                                 stamps={}, stale=(), missing=()))
    return tuple(states)


# ------------------------------------------------------------------ policy stub
class RecordedBackend:
    """PolicyBackend: forward 마다 기록 액션 한 행(원출력 자리). reset → 처음부터."""

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim != 2:
            raise ChainError(f"actions 는 (T, action_dim) 이어야 한다, got {self.actions.shape}")
        self.i = 0

    def forward(self, obs: np.ndarray) -> np.ndarray:
        if self.i >= len(self.actions):
            raise ChainError(f"기록 액션 {len(self.actions)}행을 다 썼다 (forward #{self.i + 1})")
        a = self.actions[self.i].copy()
        self.i += 1
        return a

    def reset(self) -> None:
        self.i = 0


def recorded_policy(contract: DeployContract, actions: np.ndarray) -> PolicyCore:
    """체크포인트 없이 기록 액션을 재생하는 PolicyCore(계약의 action_clip 적용)."""
    return PolicyCore.with_backend(RecordedBackend(actions), obs_dim=contract.policy.obs_dim,
                                   action_dim=contract.policy.action_dim, action_clip=contract.policy.action_clip,
                                   contract=contract)


# ------------------------------------------------------------------ trace
@dataclass(frozen=True)
class PdTick:
    now: float
    target: PdTarget
    q_meas: np.ndarray
    qd_meas: np.ndarray
    before: PdState
    after: PdState
    cmd: PdCommand
    phase: Phase
    new_policy_step: bool


@dataclass(frozen=True)
class StepRecord:
    t: int
    seq: int
    valid: bool
    reasons: tuple
    arm_q_meas: np.ndarray
    obs: np.ndarray | None = None
    action: np.ndarray | None = None
    gate_open: bool | None = None
    object_pos: np.ndarray | None = None
    palm6: np.ndarray | None = None
    q_arm: np.ndarray | None = None
    qd_arm: np.ndarray | None = None
    q_full: np.ndarray | None = None
    hand_target: np.ndarray | None = None
    gripper_cmd: float | None = None
    close_gate: float | None = None
    clearance: float | None = None
    clearance_ok: bool | None = None
    pd: tuple = ()
    status: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    episode: int
    steps: tuple
    aborted: bool
    abort_reasons: tuple

    def column(self, name: str) -> np.ndarray:
        """유효 스텝의 필드를 (T, …) 배열로."""
        rows = [getattr(s, name) for s in self.steps if s.valid]
        if not rows or any(r is None for r in rows):
            raise ChainError(f"trace 에 {name!r} 값이 없다")
        return np.asarray(rows)


# ------------------------------------------------------------------ runner
def _pre_ramp(pd: PdStage, q_seed: np.ndarray, episode: int, max_ticks: int = 10) -> None:
    """engage 뒤 홈을 목표로 램프를 끝내 TRACKING 으로 넘긴다(실기 goto_home 후 시작 규약)."""
    pd.engage(q_seed)
    for k in range(max_ticks):
        out = pd.tick(PdTarget(q=q_seed, qd=np.zeros_like(q_seed), tau_ff=np.zeros_like(q_seed), seq=-1,
                               t_recv=k * pd.dt), q_seed, np.zeros_like(q_seed), now=k * pd.dt)
        if out.state.fsm.phase is Phase.TRACKING:
            break
    if pd.state.fsm.phase is not Phase.TRACKING:
        raise ChainError(f"pd 가 TRACKING 에 못 들어갔다: {pd.state.fsm.phase.value} {pd.state.fsm.hold_reason}")
    pd.new_episode(episode)


def _pd_ticks(pd: PdStage, fo, state: RobotState, t: int, step_dt: float, n: int) -> tuple:
    t_recv = t * step_dt
    target = PdTarget(q=fo.q_arm, qd=fo.qd_arm, tau_ff=np.zeros_like(fo.q_arm), seq=fo.seq, t_recv=t_recv)
    ticks = []
    for k in range(n):
        before = pd.state.law
        now = t_recv + k * pd.dt
        out = pd.tick(target, state.arm_q, state.arm_qd, now=now)
        ticks.append(PdTick(now=now, target=target, q_meas=np.array(state.arm_q), qd_meas=np.array(state.arm_qd),
                            before=before, after=out.state.law, cmd=out.cmd, phase=out.state.fsm.phase,
                            new_policy_step=out.new_policy_step))
    return tuple(ticks)


def _build_stages(contract, robot_cfg, policy, fabric, table, fk, decoder, max_gap_ticks) -> tuple:
    """팔은 fabric(`FabricCore.side`) 이 정한다 — FK·디코더도 같은 팔로 만든다."""
    side = fabric.side
    if fk is None:
        fk = make_fk(contract, rl_ws=_paths.RL_WS, palm_pose_fn=fabric.palm_pose, tips_fn=fabric.tips, side=side)
    if decoder is None:
        decoder = make_decoder(contract, side=side, hand_soft_limits=side_soft_limits(contract, side))
    return (ObsStage(contract, robot_cfg, fk, max_gap_ticks=max_gap_ticks), PolicyStage(policy),
            FabricStage(contract, decoder, fabric, table, fk=fk))


def _run_step(stages: tuple, pd: PdStage | None, st: RobotState, t: int, last_action, dec_target,
              n_pd: int, step_dt: float) -> tuple:
    """한 정책 스텝: obs → policy → fabric → pd. (StepRecord, ObsTick, FabricOut|None)."""
    obs_stage, policy_stage, fabric_stage = stages
    ot = obs_stage.tick(st, last_action=last_action, decoder_target=dec_target)
    if not ot.out.valid:
        rec = StepRecord(t=t, seq=ot.out.seq, valid=False, reasons=tuple(ot.out.reasons),
                         arm_q_meas=np.array(st.arm_q), status={"obs": ot.status.as_dict()})
        return rec, ot, None
    pt = policy_stage.tick(ot.out.obs, ot.out.seq)
    fo = fabric_stage.tick(FabricIn(action=pt.action, seq=ot.out.seq, gate_open=ot.out.aux.get("gate_open"),
                                    object_pos=ot.out.aux.get("object_pos"), arm_q_meas=st.arm_q,
                                    ee_names=st.ee_names, ee_q=st.ee_q))
    ticks = _pd_ticks(pd, fo, st, t, step_dt, n_pd) if pd is not None else ()
    rec = StepRecord(
        t=t, seq=ot.out.seq, valid=True, reasons=fo.reasons, arm_q_meas=np.array(st.arm_q), obs=ot.out.obs.copy(),
        action=pt.action.copy(), gate_open=ot.out.aux.get("gate_open"), object_pos=ot.out.aux.get("object_pos"),
        palm6=fo.palm6, q_arm=fo.q_arm, qd_arm=fo.qd_arm, q_full=fo.q_full, hand_target=fo.hand_target,
        gripper_cmd=fo.gripper_cmd, close_gate=fo.close_gate, clearance=fo.clearance, clearance_ok=fo.clearance_ok,
        pd=ticks, status={"obs": ot.status.as_dict(), "policy": pt.status.as_dict(), "fabric": fo.status.as_dict()})
    return rec, ot, fo


def run_lockstep(contract: DeployContract, robot_cfg: RobotCfg, states: Sequence[RobotState], policy: PolicyCore,
                 fabric: FabricCore, table: TableGuard, *, fk=None, decoder=None,
                 pd: PdStage | None = None, goal=None, home_q=None, pd_ticks_per_step: int | None = None,
                 stop_on_abort: bool = True, max_gap_ticks=AUTO) -> Trace:
    """states[0] 으로 reset(에피소드 1) 한 뒤 states 를 순서대로 한 스텝씩 흘린다."""
    if not states:
        raise ChainError("states 가 비었다")
    stages = _build_stages(contract, robot_cfg, policy, fabric, table, fk, decoder, max_gap_ticks)
    obs_stage, policy_stage, fabric_stage = stages
    st0 = states[0]
    ev = obs_stage.reset(st0, goal=goal)
    policy_stage.reset(ev.episode)
    fabric_stage.reset(st0.arm_q, st0.ee_names, st0.ee_q, object_anchor=ev.object_anchor, home_q=home_q,
                       episode=ev.episode)
    n_pd = 0
    if pd is not None:
        n_pd = pd_ticks_per_step or max(1, int(round(contract.rate.step_dt / pd.dt)))
        _pre_ramp(pd, np.asarray(st0.arm_q, dtype=np.float64), ev.episode)

    steps, last_action, dec_target, abort_reasons = [], None, None, ()
    for t, st in enumerate(states):
        rec, ot, fo = _run_step(stages, pd, st, t, last_action, dec_target, n_pd, contract.rate.step_dt)
        steps.append(rec)
        if not rec.valid:
            if ot.abort:
                abort_reasons = tuple(ot.out.reasons)
                if stop_on_abort:
                    break
            continue
        last_action, dec_target = rec.action, rec.hand_target
        if fo.abort:
            abort_reasons = abort_reasons + fo.reasons
            if stop_on_abort:
                obs_stage.abort("; ".join(fo.reasons))
                break
    return Trace(episode=ev.episode, steps=tuple(steps), aborted=bool(abort_reasons) and stop_on_abort,
                 abort_reasons=abort_reasons)
