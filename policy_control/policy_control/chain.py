"""chain — obs → policy → fabric → pd 네 스테이지의 **순수 tick 논리**(ROS 무의존).

rclpy 노드(M7)와 인프로세스 lockstep(M6)이 같은 스테이지 객체를 부른다 — 노드 껍질에는
배선만 남고 규약(seq/episode/stale/abort/가드/hand sync/watchdog)은 전부 여기 있다.
코어(ObsCore·PolicyCore·ActionDecoder·FabricCore·pd_law/pd_state)는 import 로 재사용하고
이 모듈은 그것들을 **합성**만 한다. 입출력은 frozen dataclass, 배열은 새 사본.

    ObsStage(contract, robot_cfg, fk, core=None, max_gap_ticks=AUTO)
        .reset(state, object_pos=None, goal=None, episode=None) -> EpisodeEvent   (episode 마스터)
        .stop(reason='') / .abort(reason) -> EpisodeEvent
        .tick(state, last_action=None, decoder_target=None) -> ObsTick(out, status, gap, abort)
    PolicyStage(core).reset() / .tick(obs, seq) -> PolicyTick(action, seq, status)
    FabricStage(contract, decoder, fabric, table: TableGuard, fk=None)      (팔 = fabric.side)
        .reset(arm_q_meas, ee_names, ee_q, object_anchor=None, home_q=None) -> FabricReset
        .tick(FabricIn) -> FabricOut     (control-only: action = 절대 palm6, hand_cmd = 손 관절 목표)
    PdStage(ramp_cfg, track_cfg, watchdog_sec, abort_tracking, ramp_tol, release_zero_ticks, dt, gravity_fn=None)
        .engage(q_seed) / .release() / .new_episode(episode) ; .tick(target|None, q_meas, qd_meas, now, ...) -> PdOut

규약 요점:
  · seq 0 = 에피소드 첫 tick(리셋 신호). ObsCore 가 미발행 tick 에도 seq 를 올린다.
  · 스테일/결손 → valid False. 연속 미발행 gap 이 max_gap_ticks 를 넘으면 abort(재개 없음) —
    LSTM 계약(policy.rnn) 기본 3, MLP 기본 None(재개 허용).
  · 판 여유 가드는 **목표 q** 로 잰다(좌: obs FK 의 TCP, 우: fabric 손끝 최저 z). 판 xy 범위가
    주어지면 판 밖은 면제(left/right_inference_node 의 over_table), 없으면 어디서나 검사.
  · hand_sync: 'syn_target' → 디코더 손 목표를 fabric 손 슬롯에, 'measured' → 실측, None → 없음.
  · pd: 목표당 new_policy_step 1회(droop 적분 1회), 목표 나이 > watchdog → HOLD(세트포인트 동결·q̇ 0).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from . import _paths  # noqa: F401
from .contract import DeployContract
from .decoder_core import KIND_DIRECT, DecoderAux
from .fabric_core import FabricCore, name_permutation
from .obs_core import ObsCore, ObsOut
from .pd_law import PdCommand, PdInputs, PdLawCfg, PdState, initial_state, reset_droop
from .pd_law import step as pd_law_step
from .pd_state import FaultInputs, FsmState, Phase, detect_faults, initial_fsm, law_flags, transition
from .policy_core import PolicyCore
from .sources import RobotCfg, RobotState, TableCfg

from grasp_s2r_obs_builder import reorder  # noqa: E402

__all__ = ["ChainError", "StageStatus", "EpisodeEvent", "ObsTick", "ObsStage", "PolicyTick", "PolicyStage",
           "TableGuard", "FabricIn", "FabricReset", "FabricOut", "FabricStage", "PdTarget", "PdStageState",
           "PdOut", "PdStage", "DEFAULT_RNN_MAX_GAP_TICKS", "AUTO"]

#: LSTM 계약의 기본 연속 미발행 상한(플랜 §4.1 max_gap_ticks) — 계약에 아직 필드가 없어 인자로 받는다.
DEFAULT_RNN_MAX_GAP_TICKS = 3
AUTO = object()
_ENGAGED = (Phase.RAMPING, Phase.TRACKING, Phase.HOLD)


class ChainError(RuntimeError):
    """스테이지 배선/입력이 계약과 맞지 않는다."""


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e3


def _vec(a, n: int, what: str) -> np.ndarray:
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    if v.size != n:
        raise ChainError(f"{what}: {v.size}개 — {n}개여야 한다")
    if not np.all(np.isfinite(v)):
        raise ChainError(f"{what}: NaN/inf")
    return v.copy()


# ------------------------------------------------------------------ status
@dataclass(frozen=True)
class StageStatus:
    """`/policy_control/status/<node>` 의 본문(t_pub_ns 는 노드가 붙인다)."""

    node: str
    phase: str
    episode: int
    seq: int
    ok: bool
    reasons: tuple
    proc_ms: float
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"node": self.node, "phase": self.phase, "episode": self.episode, "seq": self.seq,
                "ok": self.ok, "reasons": list(self.reasons), "proc_ms": self.proc_ms, **self.extras}


# ================================================================== obs
@dataclass(frozen=True)
class EpisodeEvent:
    """`/policy_control/episode` JSON 본문(t_ns 는 노드가 붙인다)."""

    episode: int
    event: str                       # reset | start | stop | abort
    object_anchor: np.ndarray | None
    home_q: dict
    reasons: tuple = ()

    def as_dict(self) -> dict:
        return {"episode": self.episode, "event": self.event,
                "object_anchor": None if self.object_anchor is None else [float(v) for v in self.object_anchor],
                "home_q": {k: float(v) for k, v in self.home_q.items()}, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class ObsTick:
    out: ObsOut
    status: StageStatus
    gap: int                         # 연속 미발행 tick 수(이번 tick 포함)
    abort: bool                      # 이번 tick 에서 max_gap_ticks 를 넘겼다


def _home_dict(contract: DeployContract) -> dict:
    arm = dict(zip(contract.obs.joint_orders["arm"], contract.pd.home_arm))
    return {**arm, **{k: float(v) for k, v in contract.pd.home_hand.items()}}


class ObsStage:
    """에피소드 마스터 + ObsCore 합성. phase: idle | running | stopped | aborted."""

    NODE = "obs"

    def __init__(self, contract: DeployContract, robot_cfg: RobotCfg, fk, core: ObsCore | None = None,
                 max_gap_ticks=AUTO) -> None:
        self.contract = contract
        self.core = core if core is not None else ObsCore(contract, robot_cfg, fk)
        if max_gap_ticks is AUTO:
            max_gap_ticks = DEFAULT_RNN_MAX_GAP_TICKS if contract.policy.rnn is not None else None
        if max_gap_ticks is not None and int(max_gap_ticks) < 1:
            raise ChainError(f"max_gap_ticks must be >= 1 or None, got {max_gap_ticks}")
        self.max_gap_ticks = None if max_gap_ticks is None else int(max_gap_ticks)
        self.episode = 0
        self.phase = "idle"
        self.gap = 0
        self._reasons: tuple = ()

    # ---------------------------------------------------------------- episode
    def reset(self, state: RobotState, object_pos=None, goal=None, episode: int | None = None) -> EpisodeEvent:
        self.core.reset(state, object_pos=object_pos, goal=goal)
        self.episode = self.episode + 1 if episode is None else int(episode)
        self.phase, self.gap, self._reasons = "running", 0, ()
        anchor = state.object_pos if object_pos is None else object_pos
        return EpisodeEvent(episode=self.episode, event="reset", object_anchor=_vec(anchor, 3, "object_anchor"),
                            home_q=_home_dict(self.contract))

    def stop(self, reason: str = "") -> EpisodeEvent:
        return self._end("stopped", "stop", reason)

    def abort(self, reason: str) -> EpisodeEvent:
        if not reason:
            raise ChainError("abort needs a reason")
        return self._end("aborted", "abort", reason)

    def _end(self, phase: str, event: str, reason: str) -> EpisodeEvent:
        self.phase = phase
        self._reasons = tuple(r for r in (reason,) if r)
        return EpisodeEvent(episode=self.episode, event=event, object_anchor=None,
                            home_q=_home_dict(self.contract), reasons=self._reasons)

    # ---------------------------------------------------------------- tick
    def tick(self, state: RobotState, last_action=None, decoder_target=None) -> ObsTick:
        t0 = time.perf_counter()
        if self.phase != "running":
            reasons = (f"no running episode (phase {self.phase})", *self._reasons)
            out = ObsOut(obs=None, valid=False, reasons=reasons, aux={}, seq=self.core.seq)
            return ObsTick(out, self._status(out, t0), self.gap, False)
        out = self.core.tick(state, last_action=last_action, decoder_target=decoder_target)
        if out.valid:
            self.gap = 0
            return ObsTick(out, self._status(out, t0), 0, False)
        self.gap += 1
        abort = self.max_gap_ticks is not None and self.gap > self.max_gap_ticks
        if abort:
            reason = f"max_gap_ticks exceeded: {self.gap} consecutive invalid ticks > {self.max_gap_ticks}"
            self._end("aborted", "abort", reason)
            out = replace(out, reasons=(*out.reasons, reason))
        return ObsTick(out, self._status(out, t0), self.gap, abort)

    def _status(self, out: ObsOut, t0: float) -> StageStatus:
        return StageStatus(node=self.NODE, phase=self.phase, episode=self.episode, seq=out.seq, ok=out.valid,
                           reasons=tuple(out.reasons), proc_ms=_ms(t0), extras={"gap": self.gap})


# ================================================================== policy
@dataclass(frozen=True)
class PolicyTick:
    action: np.ndarray
    seq: int
    status: StageStatus


class PolicyStage:
    """obs 벡터 + seq → 액션. seq 규칙(리셋·중복·역행)은 PolicyCore 가 가진다."""

    NODE = "policy"

    def __init__(self, core: PolicyCore, episode: int = 0) -> None:
        self.core = core
        self.episode = int(episode)

    def reset(self, episode: int | None = None) -> None:
        self.core.reset()
        if episode is not None:
            self.episode = int(episode)

    def tick(self, obs: np.ndarray, seq: int) -> PolicyTick:
        t0 = time.perf_counter()
        action = self.core.act(obs, seq)
        status = StageStatus(node=self.NODE, phase="running", episode=self.episode, seq=int(seq), ok=True,
                             reasons=(), proc_ms=_ms(t0))
        return PolicyTick(action=np.array(action, dtype=np.float64), seq=int(seq), status=status)


# ================================================================== fabric
@dataclass(frozen=True)
class TableGuard:
    """판 상면·여유 하한(robot yaml `table`) + 선택적 판 xy 범위(없으면 어디서나 검사)."""

    top: float
    clearance_min: float
    center_xy: tuple | None = None
    size_xy: tuple | None = None

    @classmethod
    def from_robot_cfg(cls, table: TableCfg, center_xy=None, size_xy=None) -> "TableGuard":
        center_xy = table.center_xy if center_xy is None else center_xy
        size_xy = table.size_xy if size_xy is None else size_xy
        return cls(top=float(table.top), clearance_min=float(table.clearance_min),
                   center_xy=None if center_xy is None else tuple(float(v) for v in center_xy),
                   size_xy=None if size_xy is None else tuple(float(v) for v in size_xy))

    def over_table(self, p) -> bool:
        if self.center_xy is None or self.size_xy is None:
            return True
        return (abs(float(p[0]) - self.center_xy[0]) <= self.size_xy[0] / 2
                and abs(float(p[1]) - self.center_xy[1]) <= self.size_xy[1] / 2)


@dataclass(frozen=True)
class FabricIn:
    """한 tick 의 fabric 입력. ee_q 는 ee_names 순(로봇 yaml ee.joints + mirror) — 이름으로 재정렬한다."""

    action: np.ndarray
    seq: int
    gate_open: bool | None           # 좌: obs 'gripper_gate' 슬롯 (aux gate_open)
    object_pos: np.ndarray | None    # 우: close_gate 용 물체 위치(root)
    arm_q_meas: np.ndarray
    ee_names: tuple
    ee_q: np.ndarray
    hand_cmd: np.ndarray | None = None   # control-only(direct): 손 관절 목표(side hand_joints 순), None = 유지


@dataclass(frozen=True)
class FabricReset:
    home_q: np.ndarray               # fabric 전체 관절(contract.fabric.joint_order)
    palm6_home: np.ndarray
    object_anchor: np.ndarray | None


@dataclass(frozen=True)
class FabricOut:
    seq: int
    joint_names: tuple               # arm(canonical) + action.hand.joints → /policy_control/joint_target
    q: np.ndarray                    # (n_arm + n_hand,) 목표
    qd: np.ndarray                   # (n_arm + n_hand,) ×vel_ff_scale (손은 ×hand_vel_ff_scale, 그리퍼 0)
    q_arm: np.ndarray
    qd_arm: np.ndarray
    q_full: np.ndarray               # fabric 상태 전체(우: 팔7+손20 fabric 순)
    palm6: np.ndarray                # 디코더 palm 목표(pos3 + euler_zyx3)
    hand_target: np.ndarray | None   # synergy: action.hand.joints 순
    gripper_cmd: float | None        # binary_gripper [m]
    close_gate: float
    clearance: float                 # 목표 자세의 판 위 여유 [m]
    clearance_ok: bool
    abort: bool
    reasons: tuple
    status: StageStatus
    diag: dict = field(default_factory=dict)
    palm6_now: np.ndarray | None = None  # 목표 q 의 fabric palm FK(pos3 + euler_zyx3) → /policy_control/palm_pose


def _clearance_kind(contract: DeployContract, s, fk) -> str:
    """판 여유를 재는 방법 — 좌 그리퍼는 obs FK 의 TCP(`left_gripper`), DG-5F(fabric·urdf_chain)는 fabric 손끝 최저 z."""
    kind = contract.obs.fk.get("kind")
    if kind == "left_gripper":
        if fk is None:
            raise ChainError("fk.kind=left_gripper: 판 여유(TCP)를 재려면 obs FK 제공자가 필요하다")
        return "left_gripper"
    if s.ee_kind == "gripper":
        raise ChainError(f"gripper side {s.side}: fabric 손끝 FK 가 없다 — obs.fk.kind 는 left_gripper 여야 한다 (got {kind!r})")
    if kind not in ("fabric", "urdf_chain"):
        raise ChainError(f"모르는 obs.fk.kind {kind!r}")
    return "fabric"


class FabricStage:
    """디코더(ActionDecoder | DirectDecoder) + FabricCore + 판 여유 가드.

    팔은 fabric(`FabricCore.side`) 이 정하고 디코더도 같은 팔이어야 한다. 손 동기화는 그 팔의
    `fabric.hand_sync`. joint_target 이름 = 팔 canonical 7 + 디코더 손 관절(`decoder.hand_joints`).
    """

    NODE = "fabric"

    def __init__(self, contract: DeployContract, decoder, fabric: FabricCore, table: TableGuard, fk=None,
                 episode: int = 0) -> None:
        self.contract = contract
        self.decoder = decoder
        self.fabric = fabric
        self.table = table
        self.fk = fk
        self.episode = int(episode)
        self.side_cfg = fabric.side_cfg
        self.side = fabric.side
        if getattr(decoder, "side", self.side) != self.side:
            raise ChainError(f"디코더 팔 {decoder.side!r} ≠ fabric 팔 {self.side!r}")
        self.kind = _clearance_kind(contract, self.side_cfg, fk)
        self.arm_joints = tuple(self.side_cfg.arm_joints)
        self.n_arm = len(self.arm_joints)
        self.hand_joints = tuple(decoder.hand_joints)
        self._sync = fabric.cfg.hand_sync
        self._hand_perm = (name_permutation(self.hand_joints, fabric.joint_names[self.n_arm:])
                           if fabric.n_hand else None)
        self._hand_scale = float(fabric.cfg.hand_vel_ff_scale or 0.0)
        self._anchor: np.ndarray | None = None

    # ---------------------------------------------------------------- helpers
    def _hand_meas(self, ee_names, ee_q) -> np.ndarray:
        try:
            return np.asarray(reorder(np.asarray(ee_q, dtype=float), list(ee_names), list(self.hand_joints)),
                              dtype=np.float64)
        except (KeyError, ValueError) as exc:
            raise ChainError(f"ee_names 에서 계약 손 관절을 못 만든다: {exc}") from exc

    def _fk_hand(self, ee_names, ee_q) -> np.ndarray:
        try:
            return np.asarray(reorder(np.asarray(ee_q, dtype=float), list(ee_names), list(self.fk.hand_joints)),
                              dtype=np.float64)
        except (KeyError, ValueError) as exc:
            raise ChainError(f"ee_names 에서 FK 손 관절을 못 만든다: {exc}") from exc

    def _q_full(self, arm_q, hand_q_contract) -> np.ndarray:
        arm = _vec(arm_q, self.n_arm, "arm_q_meas")
        if self._hand_perm is None:
            return arm
        hand = _vec(hand_q_contract, len(self.hand_joints), "hand_q")
        return np.concatenate([arm, hand[self._hand_perm]])

    # ---------------------------------------------------------------- episode
    def reset(self, arm_q_meas, ee_names, ee_q, object_anchor=None, home_q=None,
              episode: int | None = None) -> FabricReset:
        """fabric_q ← home(영속 상태 시드), 디코더 리셋(앵커 스냅샷·케이지 캘리브는 **실측** 자세)."""
        home = np.asarray(self.fabric.cfg.home_q if home_q is None else home_q, dtype=np.float64).reshape(-1)
        self.fabric.reset(home)
        if episode is not None:
            self.episode = int(episode)
        self._anchor = None if object_anchor is None else _vec(object_anchor, 3, "object_anchor")
        hand_q = self._hand_meas(ee_names, ee_q)
        q_meas = self._q_full(arm_q_meas, hand_q)
        palm6, tips = None, None
        if self.decoder.hand is not None:
            palm6, tips = self.fabric.palm_pose(q_meas), self.fabric.tips(q_meas)
        self.decoder.reset(object_pos=self._anchor, hand_q=hand_q, palm6=palm6, tips=tips)
        return FabricReset(home_q=home.copy(), palm6_home=self.fabric.palm_pose(home), object_anchor=self._anchor)

    # ---------------------------------------------------------------- tick
    def tick(self, inp: FabricIn) -> FabricOut:
        t0 = time.perf_counter()
        hand_q = self._hand_meas(inp.ee_names, inp.ee_q)
        aux = self._aux(inp, hand_q)
        dec = self.decoder.step(inp.action, aux)
        hand_for_fabric = self._hand_for_fabric(dec.hand_target, hand_q)
        jt = self.fabric.step(dec.palm6, hand_target=hand_for_fabric)
        palm_now = self.fabric.palm_pose(jt.q_full)
        clearance, point = self._clearance(jt.q_full, palm_now, inp.ee_names, inp.ee_q)
        ok = not (self.table.over_table(point) and clearance < self.table.clearance_min)
        reasons = () if ok else (f"table clearance {clearance * 1e3:.1f} mm < {self.table.clearance_min * 1e3:.1f} mm",)
        hand_vals, hand_vel = self._hand_slots(dec)
        status = StageStatus(node=self.NODE, phase="running", episode=self.episode, seq=int(inp.seq), ok=ok,
                             reasons=reasons, proc_ms=_ms(t0), extras={"close_gate": dec.close_gate})
        return FabricOut(
            seq=int(inp.seq), joint_names=self.arm_joints + self.hand_joints,
            q=np.concatenate([jt.q_arm, hand_vals]), qd=np.concatenate([jt.qd_arm, hand_vel]),
            q_arm=jt.q_arm.copy(), qd_arm=jt.qd_arm.copy(), q_full=jt.q_full.copy(), palm6=dec.palm6.copy(),
            hand_target=None if dec.hand_target is None else dec.hand_target.copy(),
            gripper_cmd=dec.gripper_cmd, close_gate=float(dec.close_gate), clearance=float(clearance),
            clearance_ok=ok, abort=not ok, reasons=reasons, status=status, diag=dict(dec.diag),
            palm6_now=palm_now.copy())

    def _aux(self, inp: FabricIn, hand_q: np.ndarray) -> DecoderAux:
        if self.decoder.kind == KIND_DIRECT:
            cmd = None if inp.hand_cmd is None else _vec(inp.hand_cmd, len(self.hand_joints), "hand_cmd")
            return DecoderAux(hand_cmd=cmd)
        if self.decoder.hand is None:
            return DecoderAux(gate_open=inp.gate_open)
        if inp.object_pos is None:
            raise ChainError("synergy 디코더는 object_pos 가 필요하다(close_gate)")
        palm6 = self.fabric.palm_pose(self._q_full(inp.arm_q_meas, hand_q))
        return DecoderAux(palm6=palm6, object_pos=_vec(inp.object_pos, 3, "object_pos"), hand_q=hand_q)

    def _hand_for_fabric(self, hand_target, hand_q_meas) -> np.ndarray | None:
        if self._sync is None:
            return None
        if self._sync == "syn_target":
            if hand_target is None:
                raise ChainError("hand_sync=syn_target 인데 디코더가 손 목표를 내지 않는다")
            return hand_target
        return hand_q_meas                                   # 'measured'

    def _hand_slots(self, dec) -> tuple[np.ndarray, np.ndarray]:
        if dec.hand_target is not None:
            vel = np.zeros_like(dec.hand_target) if dec.syn_vel is None else dec.syn_vel * self._hand_scale
            return dec.hand_target.copy(), vel
        n = len(self.hand_joints)
        if n == 0:
            return np.zeros(0), np.zeros(0)
        if dec.gripper_cmd is None:
            raise ChainError("디코더가 손 목표도 그리퍼 지령도 내지 않는다")
        return np.full(n, float(dec.gripper_cmd)), np.zeros(n)

    def _clearance(self, q_full: np.ndarray, palm_now: np.ndarray, ee_names, ee_q) -> tuple[float, np.ndarray]:
        """(판 위 여유 [m], over_table 판정점). 좌: 목표 팔 q 의 TCP, 우: 목표 손끝 최저 z(판정점 = palm)."""
        if self.kind == "left_gripper":
            pose = self.fk.palm_pose(q_full[:self.n_arm], self._fk_hand(ee_names, ee_q))
            tcp = pose.extra.get("tcp")
            if tcp is None:
                raise ChainError("좌 FK 제공자가 extra['tcp'] 를 주지 않는다")
            return float(tcp[2]) - self.table.top, np.asarray(tcp, dtype=float)
        tips = self.fabric.tips(q_full)
        return float(tips[:, 2].min()) - self.table.top, np.asarray(palm_now[:3], dtype=float)


# ================================================================== pd
@dataclass(frozen=True)
class PdTarget:
    """`/policy_control/joint_target` 한 건(팔 그룹 조각). t_recv 는 수신 시각 [s]."""

    q: np.ndarray
    qd: np.ndarray
    tau_ff: np.ndarray
    seq: int
    t_recv: float


@dataclass(frozen=True)
class PdStageState:
    fsm: FsmState
    law: PdState | None              # engage 전 None
    target: PdTarget | None          # 마지막으로 받은 목표
    episode: int = 0


@dataclass(frozen=True)
class PdOut:
    cmd: PdCommand | None            # IDLE 이면 None(송출 없음)
    state: PdStageState
    new_policy_step: bool
    target_age: float | None
    faults: tuple
    status: StageStatus


class PdStage:
    """pd_law.step + pd_state FSM 합성. 상태는 frozen PdStageState 하나(매 tick 새 객체)."""

    NODE = "pd"

    def __init__(self, *, ramp_cfg: PdLawCfg, track_cfg: PdLawCfg, watchdog_sec: float, abort_tracking: float,
                 ramp_tol: float, release_zero_ticks: int, dt: float,
                 gravity_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        for name, v in (("watchdog_sec", watchdog_sec), ("abort_tracking", abort_tracking),
                        ("ramp_tol", ramp_tol), ("dt", dt)):
            if float(v) <= 0.0:
                raise ChainError(f"{name} must be > 0, got {v}")
        self.ramp_cfg, self.track_cfg = ramp_cfg, track_cfg
        self.watchdog_sec, self.abort_tracking = float(watchdog_sec), float(abort_tracking)
        self.ramp_tol, self.dt = float(ramp_tol), float(dt)
        self.gravity_fn = gravity_fn
        self.n = int(track_cfg.lower.shape[0])
        self.state = PdStageState(fsm=initial_fsm(release_zero_ticks), law=None, target=None)

    # ---------------------------------------------------------------- services
    def engage(self, q_seed) -> PdStageState:
        fsm = transition(self.state.fsm, "engage")
        self.state = replace(self.state, fsm=fsm, law=initial_state(_vec(q_seed, self.n, "q_seed")), target=None)
        return self.state

    def release(self) -> PdStageState:
        self.state = replace(self.state, fsm=transition(self.state.fsm, "release"))
        return self.state

    def new_episode(self, episode: int) -> PdStageState:
        law = None if self.state.law is None else reset_droop(self.state.law)
        self.state = replace(self.state, law=law, episode=int(episode))
        return self.state

    # ---------------------------------------------------------------- tick
    def tick(self, target: PdTarget | None, q_meas, qd_meas, now: float, *, estop: bool = False,
             thermal_act: Sequence[str] = (), switch_failed: bool = False) -> PdOut:
        t0 = time.perf_counter()
        st, new_step = self._accept(target)
        if st.fsm.phase is Phase.IDLE:
            self.state = st
            return self._out(None, st, new_step, None, (), t0)
        q_m = _vec(q_meas, self.n, "q_meas")
        qd_m = _vec(qd_meas, self.n, "qd_meas")
        age = None if st.target is None else float(now) - st.target.t_recv
        faults = self._pre_faults(st, q_m, age, estop, thermal_act, switch_failed)
        fsm = self._apply_faults(st.fsm, faults)
        if fsm.phase is Phase.RELEASING:
            cmd = PdCommand(q=st.law.q_setpoint.copy(), qd=np.zeros(self.n), tau=np.zeros(self.n),
                            limited=(), effort_fault=False)
            self.state = replace(st, fsm=transition(fsm, "zero_tick"))
            return self._out(cmd, self.state, new_step, age, tuple(faults), t0)
        law, cmd, post = self._law(st, fsm, q_m, qd_m, age, new_step)
        fsm = self._apply_faults(fsm, post)
        if post:
            cmd = replace(cmd, qd=np.zeros(self.n))
        if fsm.phase is Phase.RAMPING and st.target is not None and self._reached(law, st.target):
            fsm = transition(fsm, "ramp_done")
        self.state = replace(st, fsm=fsm, law=law)
        return self._out(cmd, self.state, new_step, age, tuple(faults) + tuple(post), t0)

    def _accept(self, target: PdTarget | None) -> tuple[PdStageState, bool]:
        st = self.state
        if target is None:
            return st, False
        new = st.target is None or int(target.seq) != int(st.target.seq)
        if new:
            return replace(st, target=target), True
        # 같은 seq 재수신: 목표는 그대로, 수신 시각만 갱신(watchdog 은 메시지 흐름을 본다)
        # 같은 seq 재수신 = 새 정책 스텝이 아니다(droop 적분 금지)지만 **값은 최신으로** 바꾼다 —
        # 내부 목표(goto_home 의 홈+settle bias, seq 고정)가 tick 마다 갱신되는 것을 법칙이 봐야 한다.
        return replace(st, target=replace(target, seq=st.target.seq)), False

    def _pre_faults(self, st, q_m, age, estop, thermal_act, switch_failed) -> list[str]:
        err = float(np.abs(st.law.q_setpoint - q_m).max())
        return detect_faults(FaultInputs(target_age_sec=age, watchdog_sec=self.watchdog_sec, tracking_err=err,
                                         abort_tracking=self.abort_tracking, target_clipped=False,
                                         effort_fault=False, estop_latched=bool(estop),
                                         thermal_act=tuple(thermal_act), switch_failed=bool(switch_failed)))

    @staticmethod
    def _apply_faults(fsm: FsmState, faults: Sequence[str]) -> FsmState:
        if not faults or fsm.phase not in _ENGAGED:
            return fsm
        for reason in faults:
            fsm = transition(fsm, "fault", reason)
        return fsm

    def _law(self, st, fsm, q_m, qd_m, age, new_step):
        advance, tracking = law_flags(fsm.phase)
        cfg = self.ramp_cfg if fsm.phase is Phase.RAMPING else self.track_cfg
        tgt = st.target
        q_t = st.law.q_setpoint if tgt is None else _vec(tgt.q, self.n, "target.q")
        qd_t = np.zeros(self.n) if tgt is None else _vec(tgt.qd, self.n, "target.qd")
        tau = np.zeros(self.n) if tgt is None else _vec(tgt.tau_ff, self.n, "target.tau_ff")
        inp = PdInputs(q_target=q_t, qd_target=qd_t, tau_request=tau, q_meas=q_m, qd_meas=qd_m,
                       target_fresh=age is not None and age <= self.watchdog_sec, new_policy_step=new_step,
                       advance=advance, tracking=tracking, dt=self.dt)
        law, cmd = pd_law_step(st.law, inp, cfg, self.gravity_fn)
        post = detect_faults(FaultInputs(target_age_sec=None, watchdog_sec=self.watchdog_sec, tracking_err=0.0,
                                         abort_tracking=self.abort_tracking,
                                         target_clipped="position" in cmd.limited, effort_fault=cmd.effort_fault,
                                         estop_latched=False, thermal_act=(), switch_failed=False))
        return law, cmd, post

    def _reached(self, law: PdState, target: PdTarget) -> bool:
        q_t = np.clip(_vec(target.q, self.n, "target.q"), self.track_cfg.lower, self.track_cfg.upper)
        return float(np.abs(law.q_setpoint - q_t).max()) <= self.ramp_tol

    def _out(self, cmd, st, new_step, age, faults, t0) -> PdOut:
        hold = () if st.fsm.hold_reason is None else tuple(st.fsm.hold_reason.split("; "))
        status = StageStatus(node=self.NODE, phase=st.fsm.phase.value, episode=st.episode,
                             seq=-1 if st.target is None else int(st.target.seq),
                             ok=st.fsm.phase is not Phase.HOLD, reasons=hold, proc_ms=_ms(t0),
                             extras={"target_age": age, "new_policy_step": new_step})
        return PdOut(cmd=cmd, state=st, new_policy_step=new_step, target_age=age, faults=faults, status=status)
