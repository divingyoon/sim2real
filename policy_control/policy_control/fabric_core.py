"""Fabric core — one side's Fabrics instance: palm target → arm joint trajectory.

Call order is exactly the training env's (and the deploy references
``scripts/left_inference_dryrun.make_fabric`` / ``scripts/grasp_s2r_fabric.py``):

    initialize_warp → WorldMeshesModel → Fabric(class from contract) →
    DisplacementIntegrator → default_config = home
    per policy step: [hand slot ← target] → set_features **once** → integrate ×decimation

★``fabric_q`` is persistent trajectory-generator state. It is seeded from the home
  pose at reset and never re-synchronised with the measured arm (08.03 freeze root
  cause); only the hand slot is synced (``hand_sync``), as the env does.
★Sides (contract v2): everything is read from ``contract.side(side).fabric`` (default
  the primary side, which the legacy top-level ``contract.fabric`` mirrors — the
  single-arm path is unchanged). Hand vectors enter/leave in the side's command
  order (``side.hand.joints`` when a hand decoder exists, else ``side.hand_joints``)
  and are permuted **by name** into ``fabric.joint_order``.
★Joint names: every fabric URDF — left and right, legacy and dg5f-m — names its
  joints on the right (``openarm_right_joint1..7``, ``rj_dg_<finger>_<segment>``).
  ``fabric.joint_order`` declares the canonical name of each slot; ``joint_key``
  reduces both namings to a side-agnostic key (arm index / finger+segment) so the
  declared order is verified against the URDF the fabric actually loaded.
★fabrics_sim is CUDA-only (warp kernels, 'cuda' hard-coded) and one instance per
  process — ``make_fabric`` refuses a CPU device up front. Unit tests inject a fake
  ``FabricBackend``.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass

import numpy as np
import torch

from . import _paths  # noqa: F401
from .contract import ContractError, DeployContract, SideCfg
from .decoder_core import resolve_side

FABRIC_MODULES = ("fabrics_sim.fabrics.openarm_tesollo_pose_fabric",
                  "fabrics_sim.fabrics.openarm_rh56f1_pose_fabric")
PCA_DIM = 5          # set_features hand slot — unused (use_hand_fabric=False) but required
PALM_DIM = 6
ORIENTATION = "euler_zyx"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_CANON_ARM = re.compile(r"^[lr]_aj_([1-7])$")
_CANON_HAND = re.compile(r"^[lr]_hj_(thumb|index|middle|ring|pinky)_([1-4])$")
_FABRIC_ARM = re.compile(r"^openarm_(?:left|right)_joint([1-7])$")
_FABRIC_HAND = re.compile(r"^[lr]j_dg_([1-5])_([1-4])$")


class FabricError(RuntimeError):
    """Fabric setup or per-step input is invalid."""


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class FabricExtras:
    """Values the training env used that a contract may not carry (overrides win over the contract).

    ``table_z``: table top z for the right's table-obstacle world (env ``table_surface_z``).
    ``use_hand_repulsion`` / ``use_body_repulsion_pairs``: env fabric flags.
    """

    table_z: float | None = None
    use_hand_repulsion: bool = False
    use_body_repulsion_pairs: bool = False


@dataclass(frozen=True)
class FabricBackend:
    """What the core needs from fabrics_sim (duck-typed so tests can fake it)."""

    fabric: object          # set_features / get_palm_pose / get_fingertip_positions / num_joints / default_config
    integrator: object      # step(q, qd, qdd, dt) -> (q, qd, qdd)
    object_ids: object
    object_indicator: object
    device: str


@dataclass(frozen=True)
class JointTarget:
    q_arm: np.ndarray       # (n_arm,) last substep
    qd_arm: np.ndarray      # (n_arm,) last substep × vel_ff_scale
    q_full: np.ndarray      # (n,) full fabric state
    substeps: np.ndarray    # (decimation, n)


# ------------------------------------------------------------------ side / names
def side_cfg(contract: DeployContract, side: str | None = None) -> SideCfg:
    """The side this fabric drives (default primary; ``decoder_core.resolve_side`` mirror rule).

    A side without a fabric section (pd-only) is an error.
    """
    s = resolve_side(contract, side)
    if s.fabric is None:
        raise ContractError(f"side {s.side!r} has no fabric section (pd-only side)")
    return s


def hand_command_joints(s: SideCfg) -> tuple:
    """Hand joint names in the order hand vectors enter/leave this side's fabric.

    With a hand decoder it is the decoder's order (``side.hand.joints`` — the right g1
    synergy profile order); a control-only side uses the asset's canonical order.
    """
    return tuple(s.hand.joints) if s.hand is not None else tuple(s.hand_joints)


def joint_key(name: str) -> tuple:
    """Side-agnostic identity of a joint: ``('arm', i)`` or ``('hand', finger_idx, segment)``.

    Accepts canonical names (``l_aj_3``, ``r_hj_index_2``) and fabric-URDF names
    (``openarm_right_joint3``, ``rj_dg_2_2``). Finger index: thumb=1 … pinky=5.
    """
    m = _CANON_ARM.match(name) or _FABRIC_ARM.match(name)
    if m:
        return ("arm", int(m.group(1)))
    m = _CANON_HAND.match(name)
    if m:
        return ("hand", FINGERS.index(m.group(1)) + 1, int(m.group(2)))
    m = _FABRIC_HAND.match(name)
    if m:
        return ("hand", int(m.group(1)), int(m.group(2)))
    raise ContractError(f"joint {name!r}: neither a canonical nor a fabric-URDF joint name")


def check_joint_names(joint_order, fabric_names) -> None:
    """``fabric.joint_order`` (canonical) must map slot-by-slot onto the URDF the fabric loaded."""
    want = [joint_key(n) for n in joint_order]
    got = [joint_key(n) for n in fabric_names]
    if want == got:
        return
    bad = [(i, c, f) for i, (c, f, w, g) in enumerate(zip(joint_order, fabric_names, want, got)) if w != g]
    if len(want) != len(got):
        raise ContractError(f"fabric.joint_order has {len(want)} joints but the fabric URDF has {len(got)}")
    raise ContractError("fabric.joint_order does not match the fabric URDF order at slots "
                        + ", ".join(f"{i}: {c} vs {f}" for i, c, f in bad[:5]))


def name_permutation(src_names, dst_names) -> np.ndarray:
    """``dst[i] = src[perm[i]]`` (grasp_s2r_fabric.permutation semantics); a missing name is a ContractError."""
    src = list(src_names)
    missing = [n for n in dst_names if n not in src]
    if missing:
        raise ContractError(f"hand permutation: joints {missing} are not in {src}")
    return np.array([src.index(n) for n in dst_names], dtype=int)


# ------------------------------------------------------------------ world
def world_kind(contract: DeployContract, side: str | None = None) -> str:
    w = side_cfg(contract, side).fabric.world
    if set(w) == {"filename"}:
        return "filename"
    if {"table_obstacle", "margin_xy", "thickness"} <= set(w):
        return "table"
    raise ContractError(f"unknown fabric.world spec {sorted(w)}")


def table_world_dict(contract: DeployContract, table_z: float | None, side: str | None = None) -> dict | None:
    """Same box the training env builds (``grasp_s2r_control._build_fabric_world``)."""
    s = side_cfg(contract, side)
    w = s.fabric.world
    if not bool(w["table_obstacle"]):
        return None
    if table_z is None:
        raise FabricError("table-obstacle world needs FabricExtras.table_z (env table_surface_z)")
    if s.palm is None:
        raise ContractError(f"side {s.side}: table-obstacle world needs the palm box (control-only side has none)")
    lo, hi = s.palm.box_lo, s.palm.box_hi
    m = float(w["margin_xy"])
    sx = (hi[0] - lo[0]) + 2.0 * m
    sy = (hi[1] - lo[1]) + 2.0 * m
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    th = float(w["thickness"])
    cz = float(table_z) - 0.5 * th
    return {"table": {"env_index": "all", "type": "box",
                      "scaling": f"{sx} {sy} {th}",
                      "transform": f"{cx} {cy} {cz} 0. 0. 0. 1."}}


def _fabric_class(name: str):
    for mod in FABRIC_MODULES:
        cls = getattr(importlib.import_module(mod), name, None)
        if cls is not None:
            return cls
    raise ContractError(f"fabric class {name!r} not found in {FABRIC_MODULES}")


def _n_hand(s: SideCfg) -> int:
    return len(s.fabric.joint_order) - len(s.arm_joints)


def _verify_backend(backend: FabricBackend, joint_order) -> None:
    """Joint count always; URDF joint names when the backend exposes them (fakes do not)."""
    if int(backend.fabric.num_joints) != len(joint_order):
        raise FabricError(f"fabric num_joints {backend.fabric.num_joints} != "
                          f"contract joint_order {len(joint_order)} — URDF/asset mismatch")
    names = getattr(backend.fabric, "get_joint_names", None)
    if names is not None and list(names()):
        check_joint_names(joint_order, [str(n) for n in names()])


# ------------------------------------------------------------------ factory
def make_fabric(contract: DeployContract, device: str, home_q=None,
                extras: FabricExtras = FabricExtras(), side: str | None = None) -> FabricBackend:
    """Build the real fabrics_sim backend for one side of the contract (CUDA only)."""
    if not str(device).startswith("cuda"):
        raise FabricError(f"fabrics_sim needs a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise FabricError("torch.cuda is not available")
    s = side_cfg(contract, side)
    kind = world_kind(contract, s.side)
    from fabrics_sim.integrator.integrators import DisplacementIntegrator
    from fabrics_sim.utils.utils import initialize_warp
    from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

    f = s.fabric
    initialize_warp(str(device)[-1])
    table_z = extras.table_z if extras.table_z is not None else f.table_z
    world_kw = ({"world_filename": f.world["filename"]} if kind == "filename"
                else {"world_dict": table_world_dict(contract, table_z, s.side)})
    world = WorldMeshesModel(batch_size=1, device=device, max_objects_per_env=int(f.max_objects),
                             **world_kw)
    obj_ids, obj_ind = world.get_object_ids()
    home = np.asarray(f.home_q if home_q is None else home_q, dtype=float).reshape(-1)
    hand_kw = {}
    if _n_hand(s) > 0:
        hand_kw = dict(use_hand_fabric=False, tip_per_finger=False, hand_mode="pca",
                       use_hand_repulsion=bool(extras.use_hand_repulsion or f.use_hand_repulsion),
                       use_body_repulsion_pairs=bool(extras.use_body_repulsion_pairs or f.use_body_repulsion_pairs))
    fab = _fabric_class(f.class_name)(
        batch_size=1, device=device, timestep=float(f.dt), graph_capturable=False,
        robot_dir_name=f.robot_dir, robot_name=f.robot_dir, fabric_params_filename=f.params,
        default_config_override=home.tolist(), **hand_kw)
    backend = FabricBackend(fabric=fab, integrator=DisplacementIntegrator(fab),
                            object_ids=obj_ids, object_indicator=obj_ind, device=str(device))
    _verify_backend(backend, f.joint_order)
    return backend


# ------------------------------------------------------------------ core
class FabricCore:
    """One side's fabric trajectory generator.

    Args:
        contract: deploy contract (v2 sides; the legacy flat sections mirror the primary side).
        device: torch/warp device ('cuda:0'); 'cpu' only with an injected fake backend.
        home_q: cspace rest + initial state in ``side.fabric.joint_order`` (default: contract home_q).
        side: 'left' | 'right' (default ``contract.primary_side``).
        extras: env values missing from the contract (table z, repulsion flags).
        backend: injected fake for tests; otherwise built by ``make_fabric``.
    """

    def __init__(self, contract: DeployContract, device: str, home_q=None, *, side: str | None = None,
                 extras: FabricExtras = FabricExtras(), backend: FabricBackend | None = None) -> None:
        self.contract = contract
        self.side_cfg = side_cfg(contract, side)
        self.side = self.side_cfg.side
        self.cfg = self.side_cfg.fabric
        f = self.cfg
        world_kind(contract, self.side)
        if f.hand_sync not in (None, "syn_target", "measured"):
            raise ContractError(f"unknown fabric.hand_sync {f.hand_sync!r}")
        self.n = len(f.joint_order)
        self.arm_joints = tuple(self.side_cfg.arm_joints)
        self.n_arm = len(self.arm_joints)
        self.n_hand = self.n - self.n_arm
        if list(f.joint_order[:self.n_arm]) != list(self.arm_joints):
            raise ContractError("fabric.joint_order does not start with the side's arm joints")
        self.hand_joints = hand_command_joints(self.side_cfg) if self.n_hand else ()
        self._hand_perm = (name_permutation(self.hand_joints, f.joint_order[self.n_arm:])
                           if self.n_hand else None)
        home = np.asarray(f.home_q if home_q is None else home_q, dtype=float).reshape(-1)
        self.backend = backend or make_fabric(contract, device, home, extras, side=self.side)
        self.device = self.backend.device
        _verify_backend(self.backend, f.joint_order)
        self._pca = torch.zeros(1, PCA_DIM, device=self.device)
        self._damping = float(f.damping) * torch.ones(1, 1, device=self.device)
        self._dt = float(f.dt)
        self._decimation = int(f.decimation)
        if self._decimation < 1:
            raise ContractError("fabric.decimation must be ≥ 1")
        self._q = self._qd = self._qdd = None
        self.reset(home)

    # ------------------------------------------------------------------ state
    @property
    def joint_names(self) -> tuple:
        return tuple(self.cfg.joint_order)

    @property
    def q(self) -> np.ndarray:
        return self._q[0].detach().cpu().numpy().astype(np.float64)

    @property
    def qd(self) -> np.ndarray:
        return self._qd[0].detach().cpu().numpy().astype(np.float64)

    def reset(self, q_home) -> None:
        """Seed fabric_q = home, fabric_qd = qdd = 0 and make home the cspace rest."""
        home = np.asarray(q_home, dtype=np.float32).reshape(-1)
        if home.size != self.n:
            raise FabricError(f"q_home has {home.size} values, fabric has {self.n} joints")
        self._q = torch.tensor(home, device=self.device, dtype=torch.float32).unsqueeze(0).contiguous()
        self._qd = torch.zeros(1, self.n, device=self.device)
        self._qdd = torch.zeros(1, self.n, device=self.device)
        self.backend.fabric.default_config.copy_(self._q)

    def _hand_tensor(self, hand, what: str) -> torch.Tensor:
        if not self.n_hand:
            raise FabricError(f"{what}: this fabric has no hand slot")
        h = np.asarray(hand, dtype=np.float32).reshape(-1)
        if h.size != self.n_hand:
            raise FabricError(f"{what}: expected {self.n_hand} hand joints, got {h.size}")
        return torch.tensor(h[self._hand_perm], device=self.device, dtype=torch.float32)

    def sync_hand(self, hand_q) -> None:
        """Overwrite the hand slot only (arm state stays persistent). ``hand_joints`` order."""
        h = self._hand_tensor(hand_q, "sync_hand")
        q = self._q.clone()
        q[0, self.n_arm:] = h
        self._q = q

    # ------------------------------------------------------------------ FK
    def _to_tensor(self, q) -> torch.Tensor:
        arr = np.asarray(q, dtype=np.float32).reshape(-1)
        if arr.size != self.n:
            raise FabricError(f"q has {arr.size} values, fabric has {self.n} joints")
        return torch.tensor(arr, device=self.device, dtype=torch.float32).unsqueeze(0)

    def palm_pose(self, q) -> np.ndarray:
        """q (n,) → palm pos3 + euler_zyx3 by the fabric's own FK."""
        with torch.inference_mode():
            out = self.backend.fabric.get_palm_pose(self._to_tensor(q), ORIENTATION)
        return out[0].detach().cpu().numpy().astype(np.float64).reshape(PALM_DIM)

    def tips(self, q) -> np.ndarray:
        """q (n,) → fingertip positions (5, 3)."""
        with torch.inference_mode():
            out = self.backend.fabric.get_fingertip_positions(self._to_tensor(q))
        return out[0].detach().cpu().numpy().astype(np.float64).reshape(-1, 3)

    # ------------------------------------------------------------------ step
    def step(self, palm6, hand_target=None) -> JointTarget:
        """One policy step: hand slot sync → set_features once → integrate ×decimation."""
        palm = np.asarray(palm6, dtype=np.float32).reshape(-1)
        if palm.size != PALM_DIM:
            raise FabricError(f"palm target must be 6D (pos3 + euler_zyx3), got {palm.size}")
        if self.cfg.hand_sync is not None and hand_target is None:
            raise FabricError(f"hand_sync={self.cfg.hand_sync!r}: step needs hand_target")
        if hand_target is not None:
            self.sync_hand(hand_target)
        feat = torch.tensor(palm, device=self.device, dtype=torch.float32).unsqueeze(0)
        fab = self.backend.fabric
        fab.set_features(self._pca, feat, ORIENTATION, self._q.detach(), self._qd.detach(),
                         self.backend.object_ids, self.backend.object_indicator, self._damping)
        subs = []
        for _ in range(self._decimation):
            self._q, self._qd, self._qdd = self.backend.integrator.step(
                self._q.detach(), self._qd.detach(), self._qdd.detach(), self._dt)
            subs.append(self._q[0].detach())
        substeps = torch.stack(subs).cpu().numpy().astype(np.float64)
        qd = self._qd[0].detach().cpu().numpy().astype(np.float64)
        return JointTarget(
            q_arm=substeps[-1, :self.n_arm].copy(),
            qd_arm=qd[:self.n_arm] * float(self.cfg.vel_ff_scale),
            q_full=substeps[-1].copy(), substeps=substeps)
