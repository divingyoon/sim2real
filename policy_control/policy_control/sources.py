"""sources — robot yaml → 리더 3종(joint_state | float_array | pose) → frozen `RobotState`.

로봇/센서 배선은 `config/robots/*.yaml` 이 정하고(플랜 §4.3), canonical↔source 관절 이름·
부호·한계는 robot_control 프로필(`jtc_bridge_core.load_profile_joints`)에서 온다. 이 모듈은
ROS 를 모른다 — 노드가 `codec` 으로 메시지를 `JointSample/ArraySample/PoseSample` 로 바꿔
`SourceSet.update_from_*` 에 넣고, tick 마다 `snapshot(now)` 로 스냅샷을 받는다.

역할(소스 이름)은 고정 집합이다: arm · ee(그리퍼/손) · object · tip_force · head · decoder_target.
스냅샷은 어느 소스가 스테일/결손인지 **보고**만 한다(값을 지어내지 않는다). 관절은 이름으로
옮기고 결손은 에러다(0 채움 금지).

양팔 yaml(`arm_left`, `ee_right` …)은 노드가 `select_side(cfg, side)` 로 **한 팔을 고른 뒤** 쓴다 —
그 팔의 소스만 남고 이름은 접미사 없는 역할(`arm`, `ee` …)로 돌아오므로 SourceSet/RobotState 는
한 팔 yaml 과 똑같이 동작한다. 한 팔 yaml 은 요청한 팔이 맞는지만 확인한다.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import _paths
from .codec import ArraySample, CodecError, JointSample, PoseSample, select_joints
from .contract import ContractError, side_of_joint
from jtc_bridge_core import load_profile_joints

SOURCE_TYPES = ("joint_state", "float_array", "pose")
OBJECT_MODES = ("latch_at_reset", "attach_after_gate", "live")
REQUIRED_SOURCES = ("arm", "ee", "object")
TYPE_BY_ROLE = {"arm": "joint_state", "ee": "joint_state", "head": "joint_state",
                "decoder_target": "joint_state", "tip_force": "float_array", "object": "pose"}
#: 양팔 yaml 은 역할에 팔 접미사를 붙인다(`arm_left`, `ee_right`, `tip_force_left` …).
#: 접미사 없는 역할은 한 팔 yaml(기존)과 같다. 어느 팔인지는 SourceCfg.side 가 든다.
SIDE_SUFFIXES = ("left", "right")
NAMINGS = ("source", "canonical")
VELOCITY_MODES = ("measured", "zero")


class RobotCfgError(ValueError):
    """robot yaml 이 스키마와 맞지 않거나 프로필과 어긋난다."""


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class SourceCfg:
    name: str
    type: str
    topic: str
    stale_sec: float
    role: str = ""                     # 접미사를 뗀 역할(arm/ee/…); 이름이 `arm_left` 면 role=arm, side=left
    side: str = ""                     # left | right | '' (한 팔 yaml)
    required: bool = True
    joints: tuple = ()                 # joint_state: canonical 이름, 이 순서로 저장
    naming: str = "source"             # 메시지의 이름 체계: source(프로필 변환) | canonical
    mirror: dict = field(default_factory=dict)   # {복제 canonical: 원본 canonical}
    velocity: str = "measured"         # measured | zero
    effort_unit: str | None = None
    tips: tuple = ()                   # float_array: 행 이름(tip-major)
    axes: int = 3
    frame: str = ""                    # pose
    mode: str = ""                     # pose(object): OBJECT_MODES


@dataclass(frozen=True)
class TableCfg:
    top: float
    clearance_min: float
    #: 판 xy 범위(중심, 크기). 없으면 여유 검사를 어디서나 한다(더 보수적). 값은 live 노드의 TABLE 상수.
    center_xy: tuple | None = None
    size_xy: tuple | None = None


@dataclass(frozen=True)
class RobotCfg:
    robot: str
    joint_profile: Path                # 첫 프로필(호환용). 관절 조회는 joint_profiles 전체를 합친 것으로 한다
    sources: dict
    groups: dict
    table: TableCfg
    joint_profiles: tuple = ()         # 합쳐 쓰는 프로필들(robot_control 본 프로필 + 보충 프로필)

    @property
    def object_mode(self) -> str:
        return self.sources["object"].mode

    @property
    def sides(self) -> tuple:
        """yaml 이 다루는 팔들 — 접미사 있는 소스의 side 들, 없으면 groups 의 side, 그것도 없으면 ()."""
        found = sorted({s.side for s in self.sources.values() if s.side})
        if found:
            return tuple(found)
        return tuple(sorted({str(g.get("side")) for g in self.groups.values() if g.get("side")}))


def load_profile(path) -> dict:
    """프로필(들) → {canonical: {source, sign, lower, upper, ...}} (재수출).

    경로 하나 또는 경로 목록. 목록이면 순서대로 합치며 같은 canonical 이 두 파일에 있으면 죽는다
    (좌손 보충 프로필이 본 프로필의 관절을 조용히 덮어쓰지 못하게)."""
    if isinstance(path, (str, Path)):
        return load_profile_joints(path)
    merged: dict = {}
    for p in path:
        part = load_profile_joints(p)
        dup = sorted(set(part) & set(merged))
        if dup:
            raise RobotCfgError(f"joint_profiles: {Path(p).name} 이 이미 있는 canonical 관절을 다시 정의한다 {dup[:5]}")
        merged = {**merged, **part}
    if not merged:
        raise RobotCfgError("joint_profiles 가 비었다")
    return merged


def _resolve(path_txt: str) -> Path:
    p = Path(path_txt).expanduser()
    return p if p.is_absolute() else _paths.RL_WS / p


def _source_cfg(name: str, raw: dict) -> SourceCfg:
    if not isinstance(raw, dict):
        raise RobotCfgError(f"sources.{name}: 매핑이어야 한다")
    role, side = split_role(name)
    if role not in TYPE_BY_ROLE:
        raise RobotCfgError(f"sources.{name}: 모르는 소스 역할 (허용 {sorted(TYPE_BY_ROLE)} [+ _left/_right])")
    typ = raw.get("type")
    if typ not in SOURCE_TYPES:
        raise RobotCfgError(f"sources.{name}.type={typ!r}: 리더 타입은 {SOURCE_TYPES} 뿐이다")
    if typ != TYPE_BY_ROLE[role]:
        raise RobotCfgError(f"sources.{name}: 역할 {role} 은 type {TYPE_BY_ROLE[role]} 이어야 한다")
    if not raw.get("topic"):
        raise RobotCfgError(f"sources.{name}.topic 이 없다")
    cfg = SourceCfg(
        name=name, type=typ, role=role, side=side,
        topic=str(raw["topic"]), stale_sec=float(raw.get("stale_sec", 0.5)),
        required=bool(raw.get("required", True)), joints=tuple(raw.get("joints", ())),
        naming=str(raw.get("naming", "source")), mirror=dict(raw.get("mirror", {}) or {}),
        velocity=str(raw.get("velocity", "measured")), effort_unit=raw.get("effort_unit"),
        tips=tuple(raw.get("tips", ())), axes=int(raw.get("axes", 3)),
        frame=str(raw.get("frame", "")), mode=str(raw.get("mode", "")),
    )
    _check_source(cfg)
    return cfg


def split_role(name: str) -> tuple[str, str]:
    """'arm_left' → ('arm', 'left'); 'object' → ('object', ''). 물체 소스는 팔에 속하지 않는다."""
    for side in SIDE_SUFFIXES:
        if name.endswith(f"_{side}"):
            role = name[: -len(side) - 1]
            if role == "object":
                raise RobotCfgError(f"sources.{name}: object 소스에는 팔 접미사를 붙이지 않는다")
            return role, side
    return name, ""


def _check_source(cfg: SourceCfg) -> None:
    n = cfg.name
    if cfg.stale_sec <= 0:
        raise RobotCfgError(f"sources.{n}.stale_sec 는 양수여야 한다")
    if cfg.type == "joint_state":
        if not cfg.joints:
            raise RobotCfgError(f"sources.{n}.joints 가 비었다")
        if cfg.naming not in NAMINGS or cfg.velocity not in VELOCITY_MODES:
            raise RobotCfgError(f"sources.{n}: naming {cfg.naming!r} / velocity {cfg.velocity!r} 허용 밖")
        if set(cfg.mirror.values()) - set(cfg.joints):
            raise RobotCfgError(f"sources.{n}.mirror 의 원본이 joints 에 없다")
    elif cfg.type == "float_array":
        if not cfg.tips or cfg.axes <= 0:
            raise RobotCfgError(f"sources.{n}: float_array 는 tips 와 axes 가 필요하다")
    elif cfg.role == "object" and cfg.mode not in OBJECT_MODES:
        raise RobotCfgError(f"sources.object.mode={cfg.mode!r}: 허용 {OBJECT_MODES}")


def _table_cfg(table: dict) -> TableCfg:
    known = {"top", "clearance_min", "center_xy", "size_xy"}
    unknown = set(table) - known
    if unknown:
        raise RobotCfgError(f"table: unknown keys {sorted(unknown)}")
    def pair(key):
        v = table.get(key)
        if v is None:
            return None
        if len(v) != 2:
            raise RobotCfgError(f"table.{key} must be [x, y]")
        return (float(v[0]), float(v[1]))
    if (table.get("center_xy") is None) != (table.get("size_xy") is None):
        raise RobotCfgError("table.center_xy and table.size_xy must be given together")
    return TableCfg(top=float(table["top"]), clearance_min=float(table["clearance_min"]),
                    center_xy=pair("center_xy"), size_xy=pair("size_xy"))


def load_robot_cfg(path: Path) -> RobotCfg:
    """yaml → RobotCfg. 스키마 위반은 전부 RobotCfgError 로 즉시 죽는다."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RobotCfgError(f"{path}: 읽기 실패 — {exc}") from exc
    if not isinstance(raw, dict):
        raise RobotCfgError(f"{path}: 최상위는 매핑이어야 한다")
    for key in ("robot", "sources", "table"):
        if key not in raw:
            raise RobotCfgError(f"{path}: '{key}' 가 없다")
    profiles = _profile_paths(path, raw)
    sources = {name: _source_cfg(name, s) for name, s in dict(raw["sources"]).items()}
    _check_required(path, sources)
    table = raw["table"]
    if not isinstance(table, dict) or "top" not in table or "clearance_min" not in table:
        raise RobotCfgError(f"{path}: table 은 top/clearance_min 을 가져야 한다")
    cfg = RobotCfg(robot=str(raw["robot"]), joint_profile=profiles[0], sources=sources,
                   groups=dict(raw.get("groups", {}) or {}),
                   table=_table_cfg(table), joint_profiles=tuple(profiles))
    _check_profile_names(cfg)
    return cfg


def _profile_paths(path: Path, raw: dict) -> list:
    """`joint_profile: <one>` 또는 `joint_profiles: [<main>, <supplement>…]` (둘 중 하나, 둘 다는 거부)."""
    if ("joint_profile" in raw) == ("joint_profiles" in raw):
        raise RobotCfgError(f"{path}: joint_profile 과 joint_profiles 중 정확히 하나만 적는다")
    items = [raw["joint_profile"]] if "joint_profile" in raw else list(raw["joint_profiles"] or [])
    if not items:
        raise RobotCfgError(f"{path}: joint_profiles 가 비었다")
    out = [_resolve(str(p)) for p in items]
    for p in out:
        if not p.exists():
            raise RobotCfgError(f"{path}: joint_profile 이 없다 — {p}")
    return out


def _check_required(path: Path, sources: dict) -> None:
    """한 팔 yaml 은 arm/ee/object, 양팔 yaml 은 팔마다 arm_<side>/ee_<side> + object 하나."""
    sided = sorted({s.side for s in sources.values() if s.side})
    if sided:
        missing = [f"{r}_{side}" for side in sided for r in ("arm", "ee") if f"{r}_{side}" not in sources]
        if "object" not in sources:
            missing.append("object")
        bare = [n for n, s in sources.items() if not s.side and s.role in ("arm", "ee", "tip_force", "decoder_target")]
        if bare:
            raise RobotCfgError(f"{path}: 양팔 yaml 에서는 {bare} 에도 팔 접미사를 붙인다")
    else:
        missing = [r for r in REQUIRED_SOURCES if r not in sources]
    if missing:
        raise RobotCfgError(f"{path}: 필수 소스가 없다 {missing}")


def _check_profile_names(cfg: RobotCfg) -> None:
    prof = load_profile(cfg.joint_profiles)
    for s in cfg.sources.values():
        if s.type == "joint_state" and s.naming == "source":
            unknown = [j for j in s.joints if j not in prof]
            if unknown:
                raise RobotCfgError(f"sources.{s.name}: 프로필에 없는 canonical 관절 {unknown}")


def with_object_mode(cfg: RobotCfg, mode: str) -> RobotCfg:
    """물체 모드만 바꾼 새 RobotCfg (테스트/CLI 오버라이드)."""
    if mode not in OBJECT_MODES:
        raise RobotCfgError(f"object mode {mode!r}: 허용 {OBJECT_MODES}")
    obj = dataclasses.replace(cfg.sources["object"], mode=mode)
    return dataclasses.replace(cfg, sources={**cfg.sources, "object": obj})


def _arm_side(cfg: RobotCfg) -> str:
    """한 팔 yaml 의 팔 — arm 소스의 canonical 접두사(l_/r_)로 읽는다."""
    arm = cfg.sources.get("arm")
    if arm is None or not arm.joints:
        raise RobotCfgError(f"{cfg.robot}: arm 소스가 없어 어느 팔인지 알 수 없다")
    try:
        return side_of_joint(str(arm.joints[0]))
    except ContractError as exc:
        raise RobotCfgError(f"{cfg.robot}: {exc}") from exc


def select_side(cfg: RobotCfg, side: str) -> RobotCfg:
    """한 팔의 소스만 남긴 새 RobotCfg. 양팔 yaml 은 `arm_<side>` → `arm` 으로 이름을 돌리고 다른 팔은
    버린다(object/head 같은 팔 무관 소스는 그대로). 한 팔 yaml 은 그 팔이 맞는지만 확인한다. 멱등."""
    if side not in SIDE_SUFFIXES:
        raise RobotCfgError(f"side {side!r}: 허용 {SIDE_SUFFIXES}")
    sided = [s for s in cfg.sources.values() if s.side]
    if not sided:
        have = _arm_side(cfg)
        if have != side:
            raise RobotCfgError(f"{cfg.robot}: 이 yaml 은 {have} 팔인데 {side} 팔을 요청했다")
        return cfg
    if side not in cfg.sides:
        raise RobotCfgError(f"{cfg.robot}: yaml 의 팔 {cfg.sides} 에 {side} 가 없다")
    out = {}
    for name, s in cfg.sources.items():
        if not s.side:
            out[name] = s
        elif s.side == side:
            out[s.role] = dataclasses.replace(s, name=s.role)
    return dataclasses.replace(cfg, sources=out)


# ------------------------------------------------------------------ state
@dataclass(frozen=True)
class RobotState:
    """한 tick 의 실기 스냅샷. 배열은 읽기 전용 새 복사본이다."""

    arm_q: np.ndarray | None          # (7,) yaml arm.joints 순
    arm_qd: np.ndarray | None
    ee_names: tuple                    # ee.joints + mirror 키
    ee_q: np.ndarray | None
    ee_qd: np.ndarray | None
    object_pos: np.ndarray | None      # (3,) base_link
    object_quat: np.ndarray | None     # (4,) wxyz
    tip_force: np.ndarray | None       # (n_tips, axes) 팁 로컬 [N]
    tip_names: tuple
    head: np.ndarray | None            # (n,) yaml head.joints 순
    decoder_target: np.ndarray | None  # (n,) yaml decoder_target.joints 순
    stamps: dict                       # source → 메시지 stamp
    stale: tuple                       # 필수 소스 중 stale_sec 초과
    missing: tuple                     # 필수 소스 중 한 번도 안 온 것


@dataclass(frozen=True)
class _Reading:
    value: object
    stamp: float
    t_recv: float


def _ro(a: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return None
    out = np.array(a, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


class SourceSet:
    """yaml 소스별 최신 표본을 들고 있다가 스냅샷을 만든다."""

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        self._profile = load_profile(cfg.joint_profiles)
        self._latest: dict[str, _Reading] = {}
        self._msg_names = {s.name: self._message_names(s) for s in cfg.sources.values() if s.type == "joint_state"}

    # ---------------------------------------------------------------- names / limits
    def _message_names(self, s: SourceCfg) -> tuple[list[str], np.ndarray]:
        if s.naming == "canonical":
            return list(s.joints), np.ones(len(s.joints))
        names = [self._profile[j]["source"] for j in s.joints]
        signs = np.array([self._profile[j]["sign"] for j in s.joints], dtype=np.float64)
        return names, signs

    def limits(self, canonical: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """프로필 관절 한계 (lower, upper)."""
        try:
            lo = np.array([self._profile[j]["lower"] for j in canonical], dtype=np.float64)
            hi = np.array([self._profile[j]["upper"] for j in canonical], dtype=np.float64)
        except KeyError as exc:
            raise RobotCfgError(f"프로필에 없는 관절 {exc}") from exc
        return lo, hi

    def _source(self, name: str, typ: str) -> SourceCfg:
        s = self.cfg.sources.get(name)
        if s is None:
            raise RobotCfgError(f"모르는 소스 {name!r} (yaml: {sorted(self.cfg.sources)})")
        if s.type != typ:
            raise RobotCfgError(f"소스 {name!r} 는 {s.type} 이지 {typ} 가 아니다")
        return s

    # ---------------------------------------------------------------- updates
    def update_from_joint_state(self, name: str, sample: JointSample, now: float) -> None:
        s = self._source(name, "joint_state")
        msg_names, signs = self._msg_names[name]
        pos, vel = select_joints(sample, msg_names)
        if s.velocity == "zero":
            vel = np.zeros(len(msg_names))
        elif vel is None and s.role in ("arm", "ee"):
            raise CodecError(f"소스 {name!r}: velocity 가 없다 (velocity: zero 로 선언하지 않았다)")
        pos = pos * signs
        vel = None if vel is None else vel * signs
        self._latest[name] = _Reading(value=(pos, vel), stamp=sample.stamp, t_recv=float(now))

    def update_from_float_array(self, name: str, sample: ArraySample, now: float) -> None:
        s = self._source(name, "float_array")
        want = len(s.tips) * s.axes
        if sample.data.size != want:
            raise CodecError(f"소스 {name!r}: {sample.data.size}개 — yaml 은 {len(s.tips)}×{s.axes}={want}")
        arr = sample.data.reshape(len(s.tips), s.axes)
        self._latest[name] = _Reading(value=arr, stamp=float(sample.seq), t_recv=float(now))

    def update_from_pose(self, name: str, sample: PoseSample, now: float) -> None:
        s = self._source(name, "pose")
        if s.frame and sample.frame and sample.frame != s.frame:
            raise CodecError(f"소스 {name!r}: frame {sample.frame!r} ≠ yaml {s.frame!r}")
        self._latest[name] = _Reading(value=(sample.pos, sample.quat), stamp=sample.stamp, t_recv=float(now))

    # ---------------------------------------------------------------- snapshot
    def _get(self, name: str):
        r = self._latest.get(name)
        return None if r is None else r.value

    def _ee(self) -> tuple[tuple, np.ndarray | None, np.ndarray | None]:
        s = self.cfg.sources.get("ee")
        if s is None:                      # 양팔 yaml 을 select_side 없이 스냅샷하면 ee 는 없다(결손으로 보고)
            return (), None, None
        names = tuple(s.joints) + tuple(s.mirror)
        v = self._get("ee")
        if v is None:
            return names, None, None
        pos, vel = v
        idx = [list(s.joints).index(s.mirror[m]) for m in s.mirror]
        return names, np.concatenate([pos, pos[idx]]), np.concatenate([vel, vel[idx]])

    def snapshot(self, now: float) -> RobotState:
        stale, missing, stamps = [], [], {}
        for s in self.cfg.sources.values():
            r = self._latest.get(s.name)
            if r is None:
                if s.required:
                    missing.append(s.name)
                continue
            stamps[s.name] = r.stamp
            if s.required and float(now) - r.t_recv > s.stale_sec:
                stale.append(s.name)
        arm = self._get("arm") or (None, None)
        ee_names, ee_q, ee_qd = self._ee()
        obj = self._get("object") or (None, None)
        head = self._get("head")
        dec = self._get("decoder_target")
        tf = self.cfg.sources.get("tip_force")
        return RobotState(
            arm_q=_ro(arm[0]), arm_qd=_ro(arm[1]), ee_names=ee_names, ee_q=_ro(ee_q), ee_qd=_ro(ee_qd),
            object_pos=_ro(obj[0]), object_quat=_ro(obj[1]),
            tip_force=_ro(self._get("tip_force")), tip_names=tuple(tf.tips) if tf else (),
            head=_ro(None if head is None else head[0]),
            decoder_target=_ro(None if dec is None else dec[0]),
            stamps=stamps, stale=tuple(stale), missing=tuple(missing),
        )
