"""Action decoder — policy action vector → palm target 6D + hand target, driven by the contract.

Two policy conventions exist and they must never be mixed (the action origin differs):

    palm  absolute_palm  (left v2)  → scripts/gripper_left_palm_command.PalmCommand
          delta_anchor   (right g1) → scripts/grasp_s2r_palm_command.PalmCommand
    hand  binary_gripper (left)     → scripts/left_policy_core.gripper_command
          synergy        (right)    → scripts/grasp_s2r_synergy.SynergyHand + close gate

The reference modules are imported, not copied. What this module adds is only the
glue the contract makes explicit: building the reference cfgs from the **side's**
``palm``/``hand`` sections (``contract.side(side)``, default primary), the action
slices of the side's ``action_groups`` inside the global action vector, the anchor
snapshot at reset (minus ``anchor.fab_to_env``), the cage calibration and
``close_gate`` formula ported from ``scripts/grasp_s2r_core.GraspS2RCore``, and the
frozen in/out records.

Control-only contracts (no policy) have ``palm``/``hand`` = None: ``DirectDecoder``
takes an **absolute palm pose** given from outside (``/policy_control/palm_cmd``) as the
fabric attractor target and optional hand joint targets (``DecoderAux.hand_cmd``);
without one the hand measured at reset is held. ``make_decoder`` picks the class.

★Freeze signal on the real hand: sim freezes on per-phalanx contact, the real DG-5F
  only reports fingertip wrenches. Like ``GraspS2RCore`` we feed the measured hand
  (``DecoderAux.hand_q``) to ``SynergyHand.step`` so its stall/'blocked' judgement
  substitutes for contact — a *known* deviation, documented in grasp_s2r_synergy.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from . import _paths  # noqa: F401  (sibling trees on sys.path)
from .contract import ContractError, DeployContract, HandCfg, PalmCfg, SideCfg

from gripper_left_palm_command import PalmCommand as AbsolutePalm  # noqa: E402
from gripper_left_palm_command import PalmCommandCfg  # noqa: E402
from grasp_s2r_palm_command import PalmCmdCfg  # noqa: E402
from grasp_s2r_palm_command import PalmCommand as DeltaPalm  # noqa: E402
from grasp_s2r_synergy import HAND_JOINT_NAMES, SynergyCfg, SynergyHand  # noqa: E402
from left_policy_core import GRIPPER_CLOSE, GRIPPER_OPEN, gripper_command  # noqa: E402

PALM_DIM = 6
CLOSE_WHEN_NEGATIVE = "a<0"
KIND_GRIPPER = "binary_gripper"
KIND_SYNERGY = "synergy"
KIND_DIRECT = "direct"
GIMBAL_EPS = 1e-9


class DecoderError(ValueError):
    """Runtime input is missing or malformed for the contract's decoder."""


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class DecoderAux:
    """Per-tick side inputs. Which fields are required depends on the decoder kind."""

    gate_open: bool | None = None          # binary_gripper: obs 'gripper_gate' slot
    palm6: np.ndarray | None = None        # synergy: measured palm pose (fabric FK, euler_zyx)
    object_pos: np.ndarray | None = None   # synergy: live object position (root frame)
    hand_q: np.ndarray | None = None       # synergy: measured hand, decoder hand-joint order
    hand_cmd: np.ndarray | None = None     # direct: hand joint targets (side hand_joints order), None = hold


@dataclass(frozen=True)
class CageCalib:
    offset_palm: np.ndarray   # cage centre in the palm frame (3,)
    radius: float             # half thumb↔others distance [m]


@dataclass(frozen=True)
class Decoded:
    palm6: np.ndarray                    # pos3 + euler_zyx3
    hand_target: np.ndarray | None       # synergy/direct: hand joints, decoder hand-joint order
    gripper_cmd: float | None            # binary_gripper: [m]
    syn_vel: np.ndarray | None           # (target − prev) / step_dt
    close_gate: float
    diag: dict = field(default_factory=dict)


# ------------------------------------------------------------------ pure helpers
def _vec(a, n: int, what: str) -> np.ndarray:
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    if v.size != n:
        raise DecoderError(f"{what}: {v.size} values, expected {n}")
    if not np.all(np.isfinite(v)):
        raise DecoderError(f"{what}: NaN/inf")
    return v.copy()


def rot_euler_zyx(e) -> np.ndarray:
    """euler_zyx (ez, ey, ex) → R = Rz·Ry·Rx (fabric palm-pose convention)."""
    ez, ey, ex = (float(v) for v in np.asarray(e, dtype=float).reshape(3))
    cz, sz, cy, sy, cx, sx = np.cos(ez), np.sin(ez), np.cos(ey), np.sin(ey), np.cos(ex), np.sin(ex)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def euler_zyx_from_rot(R) -> np.ndarray:
    """R = Rz(ez)·Ry(ey)·Rx(ex) → (ez, ey, ex); inverse of ``rot_euler_zyx``. Gimbal lock (|ey| = π/2): ex = 0."""
    m = np.asarray(R, dtype=float).reshape(3, 3)
    sy = float(np.clip(-m[2, 0], -1.0, 1.0))
    ey = float(np.arcsin(sy))
    if abs(sy) < 1.0 - GIMBAL_EPS:
        return np.array([np.arctan2(m[1, 0], m[0, 0]), ey, np.arctan2(m[2, 1], m[2, 2])])
    return np.array([np.arctan2(-m[0, 1], m[1, 1]), ey, 0.0])


def rot_from_quat(q_wxyz) -> np.ndarray:
    """Unit quaternion wxyz → rotation matrix."""
    w, x, y, z = (float(v) for v in np.asarray(q_wxyz, dtype=float).reshape(4))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < GIMBAL_EPS:
        raise DecoderError("quaternion has zero norm")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def euler_zyx_from_quat(q_wxyz) -> np.ndarray:
    """PoseStamped orientation (wxyz) → fabric euler_zyx (ez, ey, ex)."""
    return euler_zyx_from_rot(rot_from_quat(q_wxyz))


def calibrate_cage(palm6, tips) -> CageCalib:
    """Cage centre/radius at the home pose (``GraspS2RCore._calibrate_cage``).

    The centre is attached rigidly to the palm: tips[0] is the thumb, the rest are the
    four opposing fingers (tip order = contract ``obs.joint_orders.tips``).
    """
    palm = np.asarray(palm6, dtype=float).reshape(6)
    t = np.asarray(tips, dtype=float).reshape(-1, 3)
    if t.shape[0] < 2:
        raise DecoderError(f"cage needs thumb + ≥1 finger tip, got {t.shape[0]}")
    others = t[1:].mean(axis=0)
    cage = 0.5 * (t[0] + others)
    radius = 0.5 * float(np.linalg.norm(t[0] - others))
    offset = rot_euler_zyx(palm[3:]).T @ (cage - palm[:3])
    return CageCalib(offset_palm=offset, radius=radius)


def banded_dist(delta, deadband: float) -> float:
    """Distance with a z dead-band (grasp height has slack, so z inside the band counts 0)."""
    d = np.asarray(delta, dtype=float).reshape(3)
    dz = max(abs(float(d[2])) - float(deadband), 0.0)
    return float(np.sqrt(d[0] ** 2 + d[1] ** 2 + dz ** 2))


# ------------------------------------------------------------------ contract → reference cfgs
def resolve_side(contract: DeployContract, side: str | None = None) -> SideCfg:
    """The side's config (default primary).

    For the **primary** side the legacy flat sections (``contract.fabric``, ``contract.action.palm|hand``)
    are authoritative: they mirror it on disk (v1 files and every v2 writer) and an in-memory
    ``dataclasses.replace(contract, fabric=…)`` — the single-arm tests/tools — must keep applying.
    Any other side reads its own ``SideCfg``.
    """
    name = side or contract.primary_side
    s = contract.side(name)
    if name != contract.primary_side:
        return s
    return dataclasses.replace(s, fabric=contract.fabric, palm=contract.action.palm, hand=contract.action.hand)


def _side(contract: DeployContract, side: str | None) -> SideCfg:
    return resolve_side(contract, side)


def side_soft_limits(contract: DeployContract, side: str | None = None) -> np.ndarray | None:
    """(n_hand, 2) soft limits of the side's hand decoder, or None (no hand decoder / not in the contract)."""
    s = _side(contract, side)
    lim = None if s.hand is None else s.hand.params.get("soft_limits")
    return None if lim is None else np.asarray(lim, dtype=np.float64)


def _group_slice(contract: DeployContract, s: SideCfg, role: str) -> slice:
    """The side's action group for ``role`` ('palm' | 'hand' | 'gripper', bare or ``<role>_<side>``) → slice."""
    names = [g for g in s.action_groups if g == role or g.startswith(f"{role}_")]
    if len(names) != 1:
        raise ContractError(f"side {s.side}: expected exactly one {role!r} action group, got {names}")
    for g in contract.action.groups:
        if g.name == names[0]:
            return slice(int(g.slice[0]), int(g.slice[1]))
    raise ContractError(f"action group {names[0]!r} missing from contract")


def _absolute_cfg(p: PalmCfg) -> PalmCommandCfg:
    if p.euler_center is None or p.max_pose_angle is None:
        raise ContractError("absolute_palm needs euler_center and max_pose_angle")
    return PalmCommandCfg(
        box_lo=tuple(p.box_lo), box_hi=tuple(p.box_hi), euler_center=tuple(p.euler_center),
        max_pose_angle=float(p.max_pose_angle), pos_rate_limit=p.pos_rate_limit,
        rot_rate_limit=p.rot_rate_limit)


def _delta_cfg(p: PalmCfg) -> PalmCmdCfg:
    need = ("delta_xyz", "delta_rot_deg", "anchor", "rot_center_deg", "rot_half_deg",
            "home_palm", "rot_rate_limit_deg", "pos_rate_limit")
    missing = [k for k in need if getattr(p, k) is None]
    if missing:
        raise ContractError(f"delta_anchor palm is missing {missing}")
    return PalmCmdCfg(
        delta_xyz=tuple(p.delta_xyz), delta_rot_deg=float(p.delta_rot_deg),
        anchor_mode=str(p.anchor["mode"]), anchor_offset_xyz=tuple(p.anchor["offset_xyz"]),
        rate_limit_m=float(p.pos_rate_limit), rate_limit_rot_deg=float(p.rot_rate_limit_deg),
        palm_box_min=tuple(p.box_lo), palm_box_max=tuple(p.box_hi),
        rot_center_deg=tuple(p.rot_center_deg), rot_half_deg=float(p.rot_half_deg),
        home_palm=tuple(p.home_palm))


def _synergy_hand(h: HandCfg, soft_limits) -> SynergyHand:
    params = h.params
    try:
        cfg = SynergyCfg(**{k: params[k] for k in SynergyCfg.__dataclass_fields__})
    except KeyError as exc:
        raise ContractError(f"synergy hand params missing {exc}") from exc
    if tuple(h.joints) != tuple(HAND_JOINT_NAMES):
        raise ContractError("contract hand joints differ from grasp_s2r_synergy.HAND_JOINT_NAMES")
    hand = SynergyHand(cfg, soft_limits=soft_limits)
    for key, table in (("open_pose", hand.open_pose), ("grip_pose", hand.grip_pose)):
        if not np.allclose(np.asarray(params[key], dtype=float), table, atol=1e-9):
            raise ContractError(f"contract {key} != grasp_s2r_synergy table (knobs drifted)")
    return hand


def _check_binary_params(h: HandCfg) -> None:
    params = h.params
    if (float(params["open"]) != GRIPPER_OPEN or float(params["close"]) != GRIPPER_CLOSE
            or params.get("close_when") != CLOSE_WHEN_NEGATIVE
            or not params.get("force_open_when_gate_closed", False)):
        raise ContractError("binary_gripper params differ from left_policy_core.gripper_command")


# ------------------------------------------------------------------ policy decoder
class ActionDecoder:
    """Contract-driven action decoder for one side. One robot arm, one episode at a time.

    Args:
        contract: deploy contract (palm convention + hand decoder of the side).
        side: 'left' | 'right' (default ``contract.primary_side``).
        hand_soft_limits: (20, 2) soft joint limits in decoder hand-joint order —
            required on the real hand (else the 1.8 rad over-command passes through).
        stall_freeze: synergy only — use the measured hand as the freeze signal.
    """

    def __init__(self, contract: DeployContract, *, side: str | None = None, hand_soft_limits=None,
                 stall_freeze: bool = True) -> None:
        self.contract = contract
        self.side_cfg = _side(contract, side)
        s = self.side_cfg
        if s.palm is None or s.hand is None:
            raise ContractError(f"side {s.side}: no action decoders (control-only side) — use DirectDecoder")
        self._palm_slice = _group_slice(contract, s, "palm")
        conv = s.palm.convention
        if conv == "absolute_palm":
            self._palm = AbsolutePalm(_absolute_cfg(s.palm))
        elif conv == "delta_anchor":
            self._palm = DeltaPalm(_delta_cfg(s.palm))
        else:
            raise ContractError(f"unknown palm convention {conv!r}")
        self._fab_to_env = np.zeros(3)
        if s.palm.anchor is not None:
            self._fab_to_env = np.asarray(s.palm.anchor["fab_to_env"], dtype=float).reshape(3)

        self.kind = s.hand.decoder
        self.hand: SynergyHand | None = None
        if self.kind == KIND_GRIPPER:
            _check_binary_params(s.hand)
            self._hand_slice = _group_slice(contract, s, "gripper")
        elif self.kind == KIND_SYNERGY:
            self._hand_slice = _group_slice(contract, s, "hand")
            self.hand = _synergy_hand(s.hand, hand_soft_limits)
        else:
            raise ContractError(f"unknown hand decoder {self.kind!r}")
        self._stall_freeze = bool(stall_freeze) and bool(s.hand.params.get("contact_freeze", False))
        self._gate = dict(s.hand.params.get("close_gate", {"enabled": False}))
        self._cage: CageCalib | None = None
        self._prev_hand: np.ndarray | None = None

    @property
    def side(self) -> str:
        return self.side_cfg.side

    @property
    def hand_joints(self) -> tuple:
        return tuple(self.side_cfg.hand.joints)

    @property
    def cage(self) -> CageCalib | None:
        return self._cage

    # ------------------------------------------------------------------ episode
    def reset(self, *, object_pos=None, hand_q=None, palm6=None, tips=None,
              cage: CageCalib | None = None) -> None:
        """Episode start. Anchor snapshot (spawn mode), hand target = measured, cage calib."""
        if isinstance(self._palm, DeltaPalm):
            self._reset_delta(object_pos)
        else:
            self._palm.reset()
        if self.hand is None:
            return
        if hand_q is None:
            raise DecoderError("synergy reset needs hand_q (decoder hand-joint order)")
        self.hand.reset(hand_q=np.asarray(hand_q, dtype=float).reshape(len(self.hand_joints)))
        self._prev_hand = self.hand.target.copy()
        self._cage = cage
        if self._cage is None and (palm6 is not None and tips is not None):
            self._cage = calibrate_cage(palm6, tips)
        if self._gate.get("enabled", False) and self._cage is None:
            raise DecoderError("close_gate enabled: reset needs (palm6, tips) or cage")

    def _reset_delta(self, object_pos) -> None:
        mode = self.side_cfg.palm.anchor["mode"]
        if mode == "spawn":
            if object_pos is None:
                raise DecoderError("delta_anchor/spawn reset needs object_pos snapshot")
            spawn = np.asarray(object_pos, dtype=float).reshape(3) - self._fab_to_env
            self._palm.reset(object_spawn_pos=spawn)
        else:
            self._palm.reset()

    # ------------------------------------------------------------------ gate
    def close_gate(self, palm6, object_pos) -> float:
        """clip((r − d) / (ramp·r), 0, 1) with the z dead-band distance (GraspS2RCore)."""
        if not self._gate.get("enabled", False):
            return 1.0
        if self._cage is None:
            raise DecoderError("close_gate: no cage — call reset() first")
        p = np.asarray(palm6, dtype=float).reshape(6)
        obj = np.asarray(object_pos, dtype=float).reshape(3)
        cage = p[:3] + rot_euler_zyx(p[3:]) @ self._cage.offset_palm
        d = banded_dist(cage - obj, float(self._gate["z_deadband"]))
        r = self._cage.radius
        ramp = max(float(self._gate["ramp"]) * r, 1e-6)
        return float(np.clip((r - d) / ramp, 0.0, 1.0))

    # ------------------------------------------------------------------ tick
    def step(self, action, aux: DecoderAux) -> Decoded:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != self.contract.policy.action_dim:
            raise DecoderError(f"action dim {a.size} != contract {self.contract.policy.action_dim}")
        palm6 = np.asarray(self._palm.step(a[self._palm_slice].copy()), dtype=float).reshape(6)
        if self.hand is None:
            return self._step_gripper(a, palm6, aux)
        return self._step_synergy(a, palm6, aux)

    def _step_gripper(self, a: np.ndarray, palm6: np.ndarray, aux: DecoderAux) -> Decoded:
        if aux.gate_open is None:
            raise DecoderError("binary_gripper needs aux.gate_open (obs 'gripper_gate' slot)")
        cmd = gripper_command(float(a[self._hand_slice][0]), bool(aux.gate_open))
        return Decoded(palm6=palm6, hand_target=None, gripper_cmd=float(cmd), syn_vel=None,
                       close_gate=1.0 if aux.gate_open else 0.0,
                       diag={"gate_open": bool(aux.gate_open)})

    def _step_synergy(self, a: np.ndarray, palm6: np.ndarray, aux: DecoderAux) -> Decoded:
        if aux.palm6 is None or aux.object_pos is None:
            raise DecoderError("synergy needs aux.palm6 (measured FK) and aux.object_pos")
        gate = self.close_gate(aux.palm6, aux.object_pos)
        hand_q = None
        if self._stall_freeze and aux.hand_q is not None:
            hand_q = np.asarray(aux.hand_q, dtype=float).reshape(len(self.hand_joints)).copy()
        target = np.asarray(self.hand.step(a[self._hand_slice].copy(), close_gate=gate,
                                           hand_q=hand_q), dtype=float).copy()
        prev = self._prev_hand if self._prev_hand is not None else target
        syn_vel = (target - prev) / float(self.contract.rate.step_dt)
        self._prev_hand = target.copy()
        return Decoded(
            palm6=palm6, hand_target=target, gripper_cmd=None, syn_vel=syn_vel, close_gate=gate,
            diag={"box_sat": self._palm.state.box_sat.copy(),
                  "cmd_step_raw": float(self._palm.state.step_raw),
                  "syn_close_mean": float(self.hand.close[self.hand.movable].mean()),
                  "r_cage": self._cage.radius if self._cage else 0.0,
                  "stall_freeze": hand_q is not None})


# ------------------------------------------------------------------ direct (control-only) decoder
class DirectDecoder:
    """Control-only side: the fabric attractor target is an absolute palm pose given from outside.

    ``step(palm6, aux)`` — ``palm6`` is the absolute target (pos3 + euler_zyx3, base frame);
    ``aux.hand_cmd`` (side ``hand_joints`` order) replaces the held hand target, None keeps it.
    The hand starts at the pose measured at ``reset``. No box, no rate limit, no gate —
    the caller (tools/palm_cmd.py) is expected to move in small steps.
    """

    kind = KIND_DIRECT
    hand = None                  # no synergy: FabricStage skips the cage/object plumbing

    def __init__(self, contract: DeployContract, *, side: str | None = None) -> None:
        self.contract = contract
        self.side_cfg = _side(contract, side)
        self._hand: np.ndarray | None = None
        self._prev: np.ndarray | None = None

    @property
    def side(self) -> str:
        return self.side_cfg.side

    @property
    def hand_joints(self) -> tuple:
        return tuple(self.side_cfg.hand_joints)

    @property
    def cage(self) -> CageCalib | None:
        return None

    @property
    def hand_target(self) -> np.ndarray | None:
        return None if self._hand is None else self._hand.copy()

    def reset(self, *, object_pos=None, hand_q=None, palm6=None, tips=None, cage=None) -> None:
        """Episode start: the held hand target = the measured hand (side hand_joints order)."""
        del object_pos, palm6, tips, cage
        n = len(self.hand_joints)
        if n and hand_q is None:
            raise DecoderError("direct reset needs hand_q (measured hand, side hand_joints order)")
        self._hand = np.zeros(0) if n == 0 else _vec(hand_q, n, "hand_q")
        self._prev = self._hand.copy()

    def step(self, palm6, aux: DecoderAux) -> Decoded:
        if self._hand is None:
            raise DecoderError("DirectDecoder.step before reset()")
        target = _vec(palm6, PALM_DIM, "palm target (pos3 + euler_zyx3)")
        n = len(self.hand_joints)
        hand = self._hand if aux.hand_cmd is None else _vec(aux.hand_cmd, n, "hand_cmd")
        syn_vel = (hand - self._prev) / float(self.contract.rate.step_dt)
        self._prev, self._hand = hand.copy(), hand.copy()
        return Decoded(palm6=target, hand_target=hand.copy() if n else None, gripper_cmd=None,
                       syn_vel=syn_vel if n else None, close_gate=1.0,
                       diag={"hand_cmd": aux.hand_cmd is not None})


def make_decoder(contract: DeployContract, *, side: str | None = None, hand_soft_limits=None,
                 stall_freeze: bool = True):
    """ActionDecoder when the side has palm/hand decoders, DirectDecoder for a control-only side."""
    s = _side(contract, side)
    if s.palm is None or s.hand is None:
        return DirectDecoder(contract, side=s.side)
    return ActionDecoder(contract, side=s.side, hand_soft_limits=hand_soft_limits, stall_freeze=stall_freeze)
