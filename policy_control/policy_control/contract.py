"""deploy contract — the one data file the four nodes read (schema v2).

Everything task-specific (obs layout, action semantics, fabric setup, trained
gains, gravity mode, rates, checkpoint identity) lives here; the nodes carry no
task constants. The file is produced from a training run dump by
``contract_build`` (or, without a policy, from a robot asset manifest by
``contract_assets``) and re-validated on load: dimension sums, action-group
coverage, joint-order lengths, per-side consistency and (optionally) the
checkpoint md5.

v2 adds ``asset`` (which robot the contract is for) and ``sides`` — one
``SideCfg`` per arm so a bimanual asset (``openarm_dg5f-m_bi_rl``) can be driven
one arm at a time. The legacy top-level ``fabric``/``action``/``pd`` sections
still describe the *primary* side, so every v1 consumer keeps working; a v1
file is upgraded on load by deriving its single side from those sections.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "policy_control/deploy_contract/v2"
LEGACY_SCHEMAS = ("policy_control/deploy_contract/v1",)
SIDES = ("left", "right")
ARM_JOINT_RE = re.compile(r"^([lr])_aj_([1-7])$")
#: Damiao MIT packet encoding limits (dm_motor_control.cpp) — a gain outside is unrealisable.
MIT_KP_MAX = 500.0
MIT_KD_MAX = 5.0
GRAVITY_MODES = ("off", "integral_droop", "model_tau_ff")
EE_KINDS = ("dg5f", "gripper", "rh56f1", "none")


class ContractError(ValueError):
    """The contract file is malformed or inconsistent with itself / the checkpoint."""


class GainMismatch(ContractError):
    """Trained kp differs from the real driver kp — the policy cannot be reproduced."""


# ------------------------------------------------------------------ sections
@dataclass(frozen=True)
class RunInfo:
    dir: str                      # run directory, relative to sim2real/ ('' for asset-only contracts)
    task: str                     # e.g. open-grip_l_grasp_sensor_v2
    experiment: str               # full_experiment_name
    checkpoint: str               # relative to run dir, e.g. nn/x.pth ('' if none)
    checkpoint_md5: str
    env_yaml_sha1: str
    agent_yaml_sha1: str


@dataclass(frozen=True)
class RateCfg:
    policy_hz: float
    step_dt: float
    episode_steps: int


@dataclass(frozen=True)
class PolicyCfg:
    obs_dim: int
    action_dim: int
    rnn: dict | None
    mlp_units: list
    normalize_input: bool
    action_clip: float | None
    obs_clip: float | None


@dataclass(frozen=True)
class ObsSegment:
    name: str
    dim: int
    builder: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ObsCfg:
    joint_orders: dict            # arm / hand_obs / hand_profile / tips / ee ...
    fk: dict
    segments: tuple

    def segment(self, name: str) -> ObsSegment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)


@dataclass(frozen=True)
class ActionGroup:
    name: str
    slice: list


@dataclass(frozen=True)
class PalmCfg:
    convention: str               # absolute_palm | delta_anchor
    box_lo: list
    box_hi: list
    pos_rate_limit: float | None
    # absolute_palm
    euler_center: list | None = None
    max_pose_angle: float | None = None
    rot_rate_limit: float | None = None
    # delta_anchor
    delta_xyz: list | None = None
    delta_rot_deg: float | None = None
    anchor: dict | None = None
    rot_center_deg: list | None = None
    rot_half_deg: float | None = None
    home_palm: list | None = None
    rot_rate_limit_deg: float | None = None


@dataclass(frozen=True)
class HandCfg:
    decoder: str                  # binary_gripper | synergy
    joints: list
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ActionCfg:
    groups: tuple
    palm: PalmCfg | None          # None only in control-only (asset) contracts
    hand: HandCfg | None


@dataclass(frozen=True)
class FabricCfg:
    class_name: str
    robot_dir: str
    params: str
    world: dict
    dt: float
    decimation: int
    damping: float
    max_objects: int
    joint_order: list
    home_q: list
    vel_ff_scale: float
    hand_vel_ff_scale: float | None = None
    hand_sync: str | None = None
    #: training-env fabric flags (right: table world needs table_z; body repulsion pairs were ON for g1)
    table_z: float | None = None
    use_hand_repulsion: bool = False
    use_body_repulsion_pairs: bool = False
    #: where home_q came from — the fabric's default_config can differ from the robot's reset pose (left v2!)
    home_source: str | None = None


@dataclass(frozen=True)
class GravityCfg:
    mode: str                     # off | integral_droop | model_tau_ff
    sim_gravity_disabled: bool
    gain: float | None = None
    limit: list | None = None


@dataclass(frozen=True)
class Gains:
    joints: list
    kp: list
    kd: list


@dataclass(frozen=True)
class PdCfg:
    groups: list
    home_arm: list
    home_hand: dict
    gravity: GravityCfg
    sim_gains: Gains


@dataclass(frozen=True)
class AssetInfo:
    """Which robot the contract drives (``hdgp/assets/robot/<name>``); paths relative to rl_ws."""
    name: str
    urdf: str
    manifest: str
    manifest_sha1: str
    ee_kind: str                  # dg5f | gripper | rh56f1 | none


@dataclass(frozen=True)
class SideCfg:
    """Everything one arm needs — the unit the nodes select with their ``side`` parameter."""
    side: str                     # left | right
    arm_joints: list              # canonical l_aj_1..7 / r_aj_1..7
    hand_joints: list             # canonical hand joints in driver/profile order ([] for none)
    ee_kind: str
    palm_body: str                # asset link whose pose is "the palm" (FK + fabric attractor)
    tip_bodies: list              # asset fingertip links, thumb..pinky ([] for a gripper)
    home_arm: list
    home_hand: dict
    pd_groups: list               # robot-yaml group names this side owns (e.g. right_arm, right_hand)
    gravity: GravityCfg
    sim_gains: Gains
    fabric: FabricCfg | None      # None → this side has no Fabrics layer (pd-only)
    palm: PalmCfg | None          # action decoders (None in control-only contracts)
    hand: HandCfg | None
    action_groups: list           # names of ActionCfg.groups this side consumes ([] when control-only)


@dataclass(frozen=True)
class DeployContract:
    schema: str
    run: RunInfo
    rate: RateCfg
    policy: PolicyCfg
    obs: ObsCfg
    action: ActionCfg
    fabric: FabricCfg
    pd: PdCfg
    #: v2 — the bimanual view. ``fabric``/``action``/``pd`` above mirror ``sides[primary_side]``.
    sides: dict = field(default_factory=dict)
    primary_side: str = ""
    asset: AssetInfo | None = None
    control_only: bool = False    # no policy: obs/action are empty, only fabric+pd are meaningful

    def side(self, name: str) -> SideCfg:
        try:
            return self.sides[name]
        except KeyError:
            raise ContractError(f"contract has no side {name!r} (has {sorted(self.sides)})") from None

    @property
    def side_names(self) -> list:
        return sorted(self.sides)


# ------------------------------------------------------------------ (de)serialisation
def to_dict(c: DeployContract) -> dict:
    return _plain(dataclasses.asdict(c))


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item"):          # numpy scalars
        return v.item()
    return v


def _opt(cls, raw):
    return None if raw is None else cls(**raw)


def _pd_from_dict(pd: dict) -> PdCfg:
    return PdCfg(groups=pd["groups"], home_arm=pd["home_arm"], home_hand=pd["home_hand"],
                 gravity=GravityCfg(**pd["gravity"]), sim_gains=Gains(**pd["sim_gains"]))


def _side_from_dict(raw: dict) -> SideCfg:
    return SideCfg(side=raw["side"], arm_joints=raw["arm_joints"], hand_joints=raw["hand_joints"],
                   ee_kind=raw["ee_kind"], palm_body=raw["palm_body"], tip_bodies=raw["tip_bodies"],
                   home_arm=raw["home_arm"], home_hand=raw["home_hand"], pd_groups=raw["pd_groups"],
                   gravity=GravityCfg(**raw["gravity"]), sim_gains=Gains(**raw["sim_gains"]),
                   fabric=_opt(FabricCfg, raw["fabric"]), palm=_opt(PalmCfg, raw["palm"]),
                   hand=_opt(HandCfg, raw["hand"]), action_groups=raw["action_groups"])


def from_dict(raw: dict) -> DeployContract:
    try:
        obs = raw["obs"]
        act = raw["action"]
        base = dict(
            schema=raw["schema"],
            run=RunInfo(**raw["run"]),
            rate=RateCfg(**raw["rate"]),
            policy=PolicyCfg(**raw["policy"]),
            obs=ObsCfg(joint_orders=obs["joint_orders"], fk=obs["fk"],
                       segments=tuple(ObsSegment(**s) for s in obs["segments"])),
            action=ActionCfg(groups=tuple(ActionGroup(**g) for g in act["groups"]),
                             palm=_opt(PalmCfg, act["palm"]), hand=_opt(HandCfg, act["hand"])),
            fabric=FabricCfg(**raw["fabric"]),
            pd=_pd_from_dict(raw["pd"]),
            control_only=bool(raw.get("control_only", False)),
            asset=_opt(AssetInfo, raw.get("asset")),
        )
        if raw.get("sides"):
            sides = {k: _side_from_dict(v) for k, v in raw["sides"].items()}
            primary = raw.get("primary_side") or _infer_primary(sides)
        else:                       # v1 file (or a v2 writer that left sides empty): derive the one side
            sides, primary = legacy_sides(**{k: base[k] for k in ("obs", "action", "fabric", "pd")})
        return DeployContract(**base, sides=sides, primary_side=primary)
    except (KeyError, TypeError) as exc:
        raise ContractError(f"contract is missing or mistyped a field: {exc}") from exc


def _infer_primary(sides: dict) -> str:
    return sorted(sides)[0] if sides else ""


def side_of_joint(joint: str) -> str:
    """'l_aj_3' → 'left'; 'r_hj_thumb_1' → 'right' (canonical prefix)."""
    if joint.startswith("l_"):
        return "left"
    if joint.startswith("r_"):
        return "right"
    raise ContractError(f"joint {joint!r} has no l_/r_ side prefix")


def legacy_sides(obs: ObsCfg, action: ActionCfg, fabric: FabricCfg, pd: PdCfg) -> tuple[dict, str]:
    """A v1 contract describes exactly one arm: rebuild that arm's SideCfg from the flat sections."""
    arm = list(obs.joint_orders["arm"])
    side = side_of_joint(arm[0])
    hand_joints = list(action.hand.joints) if action.hand else []
    if action.hand is not None and action.hand.decoder == "binary_gripper":
        ee, hand_joints = "gripper", list(obs.joint_orders.get("ee", hand_joints))
    elif action.hand is not None:
        ee = "dg5f"
    else:
        ee = "none"
    palm_body = next((s.params["body"] for s in obs.segments if "body" in s.params), "palm_link")
    cfg = SideCfg(side=side, arm_joints=arm, hand_joints=hand_joints, ee_kind=ee, palm_body=palm_body,
                  tip_bodies=list(obs.joint_orders.get("tips", [])), home_arm=list(pd.home_arm),
                  home_hand=dict(pd.home_hand), pd_groups=list(pd.groups), gravity=pd.gravity,
                  sim_gains=pd.sim_gains, fabric=fabric, palm=action.palm, hand=action.hand,
                  action_groups=[g.name for g in action.groups])
    return {side: cfg}, side


# ------------------------------------------------------------------ validation
def validate(c: DeployContract) -> DeployContract:
    if c.schema != SCHEMA and c.schema not in LEGACY_SCHEMAS:
        raise ContractError(f"schema {c.schema!r} not in {(SCHEMA, *LEGACY_SCHEMAS)}")
    _validate_policy_io(c)
    if len(c.fabric.joint_order) != len(c.fabric.home_q):
        raise ContractError("fabric.joint_order and fabric.home_q lengths differ")
    _validate_gains(c.pd.sim_gains, "pd")
    if c.pd.gravity.mode not in GRAVITY_MODES:
        raise ContractError(f"unknown gravity mode {c.pd.gravity.mode!r}")
    if c.rate.policy_hz <= 0 or c.rate.step_dt <= 0:
        raise ContractError("rate must be positive")
    _validate_sides(c)
    return c


def _validate_policy_io(c: DeployContract) -> None:
    total = sum(s.dim for s in c.obs.segments)
    if total != c.policy.obs_dim:
        raise ContractError(f"obs segments sum to {total} but policy.obs_dim is {c.policy.obs_dim}")
    cursor = 0
    for g in c.action.groups:
        if list(g.slice) != [cursor, g.slice[1]] or g.slice[1] <= cursor:
            raise ContractError(f"action group {g.name} slice {g.slice} is not contiguous")
        cursor = g.slice[1]
    if cursor != c.policy.action_dim:
        raise ContractError(f"action groups cover {cursor} but policy.action_dim is {c.policy.action_dim}")
    if c.control_only and (c.policy.obs_dim or c.policy.action_dim or c.run.checkpoint):
        raise ContractError("control_only contracts must have no obs/action/checkpoint")
    if not c.control_only and (c.action.palm is None or c.action.hand is None):
        raise ContractError("action.palm/hand are required unless control_only")


def _validate_gains(g: Gains, where: str) -> None:
    if len(g.joints) != len(g.kp) or len(g.kp) != len(g.kd):
        raise ContractError(f"{where}.sim_gains joints/kp/kd lengths differ")


def _validate_sides(c: DeployContract) -> None:
    if not c.sides:
        raise ContractError("contract has no sides")
    if c.primary_side not in c.sides:
        raise ContractError(f"primary_side {c.primary_side!r} not in sides {sorted(c.sides)}")
    group_names = [g.name for g in c.action.groups]
    claimed: list = []
    for name, s in c.sides.items():
        _validate_side(name, s, group_names)
        claimed += s.action_groups
    if not c.control_only and sorted(claimed) != sorted(group_names):
        raise ContractError(f"sides claim action groups {sorted(claimed)} but the action has {sorted(group_names)}")
    if c.asset is not None and c.asset.ee_kind not in EE_KINDS:
        raise ContractError(f"asset.ee_kind {c.asset.ee_kind!r} not in {EE_KINDS}")


def _validate_side(name: str, s: SideCfg, group_names: list) -> None:
    if name != s.side or s.side not in SIDES:
        raise ContractError(f"side key {name!r} / value {s.side!r} must be one of {SIDES}")
    if len(s.arm_joints) != 7 or any(side_of_joint(j) != s.side for j in s.arm_joints + s.hand_joints):
        raise ContractError(f"side {name}: arm/hand joints must be 7 + same-side canonical names")
    if len(s.home_arm) != 7:
        raise ContractError(f"side {name}: home_arm must have 7 values")
    if s.ee_kind not in EE_KINDS:
        raise ContractError(f"side {name}: ee_kind {s.ee_kind!r} not in {EE_KINDS}")
    if s.gravity.mode not in GRAVITY_MODES:
        raise ContractError(f"side {name}: unknown gravity mode {s.gravity.mode!r}")
    _validate_gains(s.sim_gains, f"sides.{name}")
    if s.fabric is not None and len(s.fabric.joint_order) != len(s.fabric.home_q):
        raise ContractError(f"side {name}: fabric.joint_order and home_q lengths differ")
    if s.fabric is not None and s.fabric.joint_order[:7] != list(s.arm_joints):
        raise ContractError(f"side {name}: fabric.joint_order must start with the side's arm joints")
    unknown = [g for g in s.action_groups if g not in group_names]
    if unknown:
        raise ContractError(f"side {name}: action groups {unknown} are not in action.groups")


def save_contract(c: DeployContract, path: Path) -> None:
    validate(c)
    Path(path).write_text(json.dumps(to_dict(c), indent=1, ensure_ascii=False) + "\n")


def load_contract(path: Path) -> DeployContract:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract {path}: {exc}") from exc
    return validate(from_dict(raw))


# ------------------------------------------------------------------ integrity
def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def verify_checkpoint(c: DeployContract, sim2real_root: Path) -> Path:
    """Re-hash the checkpoint and the run dump the contract was built from."""
    run = Path(sim2real_root) / c.run.dir
    if not c.run.checkpoint:
        raise ContractError("contract has no checkpoint")
    ckpt = run / c.run.checkpoint
    if not ckpt.exists():
        raise ContractError(f"checkpoint missing: {ckpt}")
    if file_md5(ckpt) != c.run.checkpoint_md5:
        raise ContractError(f"checkpoint md5 changed: {ckpt}")
    for name, want in (("env.yaml", c.run.env_yaml_sha1), ("agent.yaml", c.run.agent_yaml_sha1)):
        p = run / "params" / name
        if want and p.exists() and file_sha1(p) != want:
            raise ContractError(f"{name} changed since the contract was built: {p}")
    return ckpt


# ------------------------------------------------------------------ gains
@dataclass(frozen=True)
class GainReport:
    ok: bool
    reasons: list
    kd_note: str
    real_kp: list
    real_kd: list


def load_driver_gains(gains_yaml: Path) -> dict:
    """control_gains.yaml → {joint index 1..7: (kp, kd)}."""
    data = yaml.safe_load(Path(gains_yaml).read_text())
    out = {}
    for key, val in data.items():
        m = re.fullmatch(r"joint(\d+)", str(key))
        if m and isinstance(val, dict) and "kp" in val and "kd" in val:
            out[int(m.group(1))] = (float(val["kp"]), float(val["kd"]))
    if len(out) != 7:
        raise ContractError(f"{gains_yaml}: expected joint1..7 kp/kd, got {sorted(out)}")
    return out


def compare_gains(c: DeployContract, gains_yaml: Path, tol: float = 1e-6, side: str | None = None) -> GainReport:
    """Trained kp **and** kd must both equal the driver's — vendor gains only (2026-09-06).

    Until 09.06 only kp was gated and kd was reported as information, because the right
    arm's run carried an r2s-fitted kd (7.053/4.182/… — the driver's is 2.75/2.5/…).
    That is exactly the hole the user closed: a policy trained on gains the motors never
    receive is a policy for a different robot. An r2s kd is also unrealisable — four of
    those values exceed the MIT packet's kd ceiling of 5.0, so the arm could not have
    been driven that way even in principle.

    ``side`` picks one arm of a bimanual contract; default = the primary side's ``pd.sim_gains``.
    """
    gains = c.pd.sim_gains if side is None else c.side(side).sim_gains
    real = load_driver_gains(gains_yaml)
    reasons, notes, real_kp, real_kd = [], [], [], []
    for joint, kp, kd in zip(gains.joints, gains.kp, gains.kd):
        m = ARM_JOINT_RE.match(joint)
        if m is None:
            continue
        rkp, rkd = real[int(m.group(2))]
        real_kp.append(rkp)
        real_kd.append(rkd)
        if abs(kp - rkp) > tol:
            reasons.append(f"{joint}: trained kp {kp:g} != driver kp {rkp:g}")
        if abs(kd - rkd) > tol:
            reasons.append(f"{joint}: trained kd {kd:g} != driver kd {rkd:g}")
        if kp > MIT_KP_MAX or kd > MIT_KD_MAX:
            notes.append(f"{joint}: kp {kp:g}/kd {kd:g} impossible on the MIT packet")
    return GainReport(ok=not reasons, reasons=reasons, kd_note="; ".join(notes),
                      real_kp=real_kp, real_kd=real_kd)


def require_gains(c: DeployContract, gains_yaml: Path, side: str | None = None) -> GainReport:
    rep = compare_gains(c, gains_yaml, side=side)
    if not rep.ok:
        raise GainMismatch("; ".join(rep.reasons))
    return rep
