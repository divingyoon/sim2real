"""obs_core — 에피소드 상태를 들고 계약 세그먼트를 순서대로 조립하는 순수 코어.

    ObsCore(contract, robot_cfg, fk, side=None)
      .reset(state, object_pos=None, goal=None)   # 앵커 스냅샷·게이트/부착/seq/last_action 초기화
      .tick(state, last_action=None, decoder_target=None) -> ObsOut(obs, valid, reasons, aux, seq)

팔(`side`, 기본 = 계약 primary_side): 관절 순서·palm/손끝 바디·손 디코더는 `contract.side(side)` 에서,
센서 배선은 robot yaml 에서 `sources.select_side` 로 그 팔 것만 고른다(양팔 yaml 은 `<역할>_<side>`,
한 팔 yaml 은 접미사 없는 역할). 거부: control-only 계약(세그먼트 없음), 액션이 없는 팔(정책 없음),
obs 레이아웃이 다른 팔의 것, yaml 이 다른 팔, FK 의 palm 바디가 side 와 다름.

에피소드 상태: last_action(직전 액션, 첫 tick 은 0), 그리퍼 게이트 래치(좌), 물체 앵커/부착,
seq(미발행 tick 에도 증가). 조립 = 계약 `obs.segments` 순 concat, 총 차원은 `policy.obs_dim`
과 대조, NaN 은 에러. 정규화는 체크포인트 안에 있으므로 여기서는 raw 다.

물체 모드(robot yaml `sources.object.mode`):
  · latch_at_reset    — reset 때 스냅샷한 위치를 에피소드 내내 쓴다.
  · attach_after_gate — 게이트가 처음 열린 순간 컵−턱 상대위치를 **그리퍼 프레임**으로 굳혀
                        이후 FK 로 같이 움직인다(09.03 좌 실기 실증, `left_inference_node` 와 같은 식).
                        접근 구간은 latch 와 같다(인식 지터 차단). 게이트가 닫히면 부착을 푼다.
  · live              — 매 tick 센서값.
우팔 `joint_err` 는 직전 tick 의 손 목표(`decoder_target`, action.hand.joints 순)로 만든다.
첫 목표 전에는 계약 open pose — sim 리셋(`_syn_target = q0`)과 같다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _paths  # noqa: F401
from .contract import ContractError, DeployContract, SideCfg
from .obs_segments import ObsBuildError, ObsError, ObsInputs, builder
from .sources import RobotCfg, RobotCfgError, RobotState, select_side
from left_grasp_gate import GateCfg, GraspGate
from left_obs_builder import quat_to_matrix

__all__ = ["ObsCore", "ObsOut", "ObsError", "split_segments"]

IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class ObsOut:
    obs: np.ndarray | None        # valid 가 아니면 None (미발행)
    valid: bool
    reasons: tuple
    aux: dict = field(default_factory=dict)
    seq: int = 0


def split_segments(obs: np.ndarray, contract: DeployContract) -> dict:
    """obs 벡터 → {세그먼트 이름: 조각} (진단·테스트용, 새 배열)."""
    out, off = {}, 0
    for s in contract.obs.segments:
        out[s.name] = np.asarray(obs)[off:off + s.dim].copy()
        off += s.dim
    return out


def _vec(a, n: int, what: str) -> np.ndarray:
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    if v.size != n:
        raise ObsError(f"{what}: {v.size}개 — {n}개여야 한다")
    if not np.all(np.isfinite(v)):
        raise ObsError(f"{what}: NaN/inf")
    return v


def resolve_side(contract: DeployContract, side: str | None) -> SideCfg:
    """관측을 만들 팔 — 정책이 있는(액션을 소비하는) 팔이어야 하고 obs 레이아웃이 그 팔 것이어야 한다."""
    if contract.control_only:
        raise ObsError("control-only 계약(정책 없음)에는 obs 세그먼트가 없다 — obs 노드가 만들 것이 없다")
    try:
        s = contract.side(side or contract.primary_side)
    except ContractError as exc:
        raise ObsError(str(exc)) from exc
    if not s.action_groups or s.hand is None:
        raise ObsError(f"side {s.side!r} 는 액션을 소비하지 않는다(정책 없음) — 관측을 만들 수 없다")
    if list(s.arm_joints) != list(contract.obs.joint_orders["arm"]):
        raise ObsError(f"계약 obs 레이아웃(arm {contract.obs.joint_orders['arm']})은 side {s.side!r} 의 것이 아니다")
    return s


def _check_yaml_side(cfg: RobotCfg, side: SideCfg, fk) -> None:
    arm_yaml = list(cfg.sources["arm"].joints)
    if arm_yaml != list(side.arm_joints):
        raise ObsError(f"robot yaml arm.joints {arm_yaml} ≠ 계약 side {side.side!r} arm_joints {list(side.arm_joints)}")
    ee = cfg.sources["ee"]
    have = set(ee.joints) | set(ee.mirror)
    missing = [j for j in side.hand_joints if j not in have]
    if missing:
        raise ObsError(f"robot yaml ee 소스에 side {side.side!r} 손 관절 {missing} 이 없다")
    if str(fk.palm_body) != str(side.palm_body):
        raise ObsError(f"FK palm_body {fk.palm_body!r} ≠ 계약 side {side.side!r} palm_body {side.palm_body!r}")


class ObsCore:
    """계약 + robot yaml + FK 로 한 tick 의 관측을 만든다."""

    def __init__(self, contract: DeployContract, robot_cfg: RobotCfg, fk, side: str | None = None) -> None:
        self.contract = contract
        self.side = resolve_side(contract, side)
        try:
            self.cfg = select_side(robot_cfg, self.side.side)
        except RobotCfgError as exc:
            raise ObsError(f"robot yaml: {exc}") from exc
        self.fk = fk
        _check_yaml_side(self.cfg, self.side, fk)
        self._builders = [(s, builder(s.builder)) for s in contract.obs.segments]
        self._gate = self._make_gate()
        self._hand_open = self._hand_open_pose()
        self.mode = self.cfg.object_mode
        self._anchor: np.ndarray | None = None
        self._anchor_quat: np.ndarray = IDENT_QUAT
        self._attach_local: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._last_action = np.zeros(contract.policy.action_dim)
        self.seq = 0

    # ---------------------------------------------------------------- setup
    def _make_gate(self) -> GraspGate | None:
        seg = next((s for s in self.contract.obs.segments if s.builder == "gripper_gate"), None)
        if seg is None:
            return None
        p = seg.params
        try:
            cfg = GateCfg(pad_offset=float(p["pad_offset"]), lateral_ok=float(p["lateral_ok"]),
                          along_ok=float(p["along_ok"]), band_axis=tuple(float(v) for v in p["band_axis"]),
                          release_lateral=None if p.get("release_lateral") is None else float(p["release_lateral"]))
        except KeyError as exc:
            raise ObsError(f"gripper_gate params 에 {exc} 가 없다") from exc
        return GraspGate(cfg)

    def _hand_open_pose(self) -> np.ndarray:
        h = self.side.hand
        if h.decoder == "synergy":
            return _vec(h.params["open_pose"], len(h.joints), "hand.params.open_pose")
        return np.full(len(h.joints), float(h.params["open"]))

    # ---------------------------------------------------------------- episode
    def reset(self, state: RobotState, object_pos=None, goal=None) -> None:
        """에피소드 시작 — 물체 앵커·목표 스냅샷, 래치/부착/seq/last_action 초기화."""
        anchor = state.object_pos if object_pos is None else object_pos
        if anchor is None:
            raise ObsError("reset: 물체 위치가 없다 (object 소스 결손)")
        self._anchor = _vec(anchor, 3, "object_pos")
        self._anchor_quat = IDENT_QUAT if state.object_quat is None else _vec(state.object_quat, 4, "object_quat")
        self._goal = self._resolve_goal(goal)
        self._attach_local = None
        self._last_action = np.zeros(self.contract.policy.action_dim)
        self.seq = 0
        if self._gate is not None:
            self._gate.reset()

    def _resolve_goal(self, goal) -> np.ndarray:
        segs = {s.builder: s for s in self.contract.obs.segments}
        if "goal_pose" in segs:
            want = segs["goal_pose"].dim
            return _vec(segs["goal_pose"].params["goal"] if goal is None else goal, want, "goal")
        gmo = segs.get("goal_minus_object")
        if gmo is None:
            return np.zeros(3) if goal is None else _vec(goal, 3, "goal")
        if goal is not None:
            return _vec(goal, 3, "goal")
        offset = gmo.params.get("goal_offset")
        if offset is None:
            raise ObsError("goal 이 없다: goal 인자도, goal_minus_object.goal_offset 도 없다")
        return self._anchor + _vec(offset, 3, "goal_offset")

    # ---------------------------------------------------------------- tick
    def tick(self, state: RobotState, last_action=None, decoder_target=None) -> ObsOut:
        if self._anchor is None:
            raise ObsError("tick 전에 reset() 을 불러야 한다")
        seq, self.seq = self.seq, self.seq + 1
        reasons = self._invalid_reasons(state)
        if reasons:
            return ObsOut(obs=None, valid=False, reasons=tuple(reasons), aux={"seq": seq}, seq=seq)
        if last_action is not None:
            self._last_action = _vec(last_action, self.contract.policy.action_dim, "last_action")
        hand_for_fk = self._hand_for_fk(state)
        fk = self.fk.palm_pose(_vec(state.arm_q, len(self.contract.obs.joint_orders["arm"]), "arm_q"), hand_for_fk)
        obj_pos, obj_quat = self._object(state, fk)
        gate = self._update_gate(fk, obj_pos, obj_quat)
        attached = self._update_attach(fk, obj_pos, gate)
        target = self._decoder_target(state, decoder_target)
        inp = ObsInputs(contract=self.contract, state=state, fk=fk, object_pos=obj_pos, object_quat=obj_quat,
                        goal=self._goal, last_action=self._last_action, gate=gate, decoder_target=target,
                        palm_body=str(self.side.palm_body), hand_joints=tuple(self.side.hand.joints))
        obs = self._assemble(inp)
        aux = {"seq": seq, "gate_open": None if gate is None else bool(gate > 0.5), "attached": attached,
               "object_pos": obj_pos.copy(), "palm_pos": fk.palm_pos.copy(), "palm_quat": fk.palm_quat.copy(),
               "tcp": None if fk.extra.get("tcp") is None else fk.extra["tcp"].copy(), "goal": self._goal.copy()}
        return ObsOut(obs=obs, valid=True, reasons=(), aux=aux, seq=seq)

    def _invalid_reasons(self, state: RobotState) -> list[str]:
        reasons = [f"missing source {m}" for m in state.missing] + [f"stale source {s}" for s in state.stale]
        if state.arm_q is None or state.ee_q is None or state.object_pos is None:
            reasons.append("state has no arm/ee/object")
        return reasons

    def _hand_for_fk(self, state: RobotState) -> np.ndarray:
        from grasp_s2r_obs_builder import reorder
        try:
            return reorder(state.ee_q, list(state.ee_names), list(self.fk.hand_joints))
        except (KeyError, ValueError) as exc:
            raise ObsError(f"FK 손 관절 순서를 이름으로 못 만든다: {exc}") from exc

    def _object(self, state: RobotState, fk) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "live":
            return (_vec(state.object_pos, 3, "object_pos"),
                    IDENT_QUAT if state.object_quat is None else _vec(state.object_quat, 4, "object_quat"))
        if self.mode == "attach_after_gate" and self._attach_local is not None:
            mid, R = _jaw_frame(fk)
            return mid + R @ self._attach_local, self._anchor_quat
        return self._anchor.copy(), self._anchor_quat

    def _update_gate(self, fk, obj_pos, obj_quat) -> float | None:
        if self._gate is None:
            return None
        tips = dict(zip(fk.tip_names, fk.tips))
        try:
            self._gate.update(finger_l_pos=tips["l_hl_gripper_left_finger"], finger_r_pos=tips["l_hl_gripper_right_finger"],
                              gripper_base_quat=fk.palm_quat, cup_pos=obj_pos, cup_quat=obj_quat)
        except KeyError as exc:
            raise ObsError(f"게이트에 필요한 손가락 바디가 FK 에 없다: {exc}") from exc
        return self._gate.obs_value

    def _update_attach(self, fk, obj_pos, gate) -> bool:
        if self.mode != "attach_after_gate" or gate is None:
            return False
        if gate > 0.5 and self._attach_local is None:
            mid, R = _jaw_frame(fk)
            self._attach_local = R.T @ (obj_pos - mid)
        elif gate <= 0.5 and self._attach_local is not None:
            self._attach_local = None
        return self._attach_local is not None

    def _decoder_target(self, state: RobotState, decoder_target) -> np.ndarray:
        hand_joints = list(self.side.hand.joints)
        n = len(hand_joints)
        if decoder_target is not None:
            return _vec(decoder_target, n, "decoder_target")
        if state.decoder_target is not None:
            src = self.cfg.sources.get("decoder_target")
            if src is not None and list(src.joints) == hand_joints:
                return _vec(state.decoder_target, n, "state.decoder_target")
            raise ObsError("robot yaml decoder_target.joints 가 계약 side.hand.joints 와 다르다")
        return self._hand_open.copy()

    def _assemble(self, inp: ObsInputs) -> np.ndarray:
        parts = []
        for seg, fn in self._builders:
            try:
                v = np.asarray(fn(inp, seg), dtype=np.float64).reshape(-1)
            except ObsBuildError:
                raise
            except (KeyError, ValueError, TypeError) as exc:
                raise ObsBuildError(f"segment {seg.name!r}: {exc}") from exc
            if v.size != seg.dim:
                raise ObsError(f"segment {seg.name!r}: {v.size}차원 — 계약은 {seg.dim}")
            if not np.all(np.isfinite(v)):
                raise ObsError(f"segment {seg.name!r}: NaN/inf")
            parts.append(v)
        obs = np.concatenate(parts)
        if obs.size != self.contract.policy.obs_dim:
            raise ObsError(f"obs {obs.size}차원 — 계약 policy.obs_dim {self.contract.policy.obs_dim}")
        return obs


def _jaw_frame(fk) -> tuple[np.ndarray, np.ndarray]:
    """턱 중점(손가락 두 바디 평균)과 그리퍼 base 회전 — `left_inference_node` 부착 수식 그대로."""
    return fk.tips.mean(axis=0), quat_to_matrix(fk.palm_quat)
