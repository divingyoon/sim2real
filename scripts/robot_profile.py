#!/usr/bin/env python3
"""로봇 구성 프로필 로더 (yaml/numpy, ROS·Isaac 무의존).

배포 스택은 원래 Fabrics 자산·토픽·joint profile 경로를 **하드코딩**했다. 그래서 sim 이
로봇 자산을 바꿔도(08.05 `openarm_tesollo` → `openarm_tesollo_bi_s`) 배포는 구 자산에
머물렀고, palm 이 **6.5cm 어긋난 채**(실측 palm_link z 0.12863 vs 0.1935) 동작했다.
palm 은 obs 154D 중 36차원의 기준이자 Fabrics IK 목표이므로 관측과 지령이 동시에 틀린다.

이 모듈은 구성을 `config/robots/<name>.yaml` 로 외부화하고, **자산 매니페스트**
(`urdf/generated/rl/*_manifest.yaml` 의 `control_joint_order`)와 대조해 검증한다.
불일치는 조용히 넘어가지 않고 예외로 던진다 — 위 사고의 재발 방어선이다.

    prof = load_robot_profile("tesollo_bi_s__right")
    prof.arm_canonical   # ('r_aj_1', ..., 'r_aj_7')       — 매니페스트에서 유도
    prof.ee_canonical    # ('r_hj_thumb_1', ..., 20개)
    prof.arm_source      # ('openarm_right_joint1', ...)   — joint profile 에서 유도
    prof.fabrics         # FabricsAsset(robot_dir=..., world=...)

★좌우 미러 주의: palm delta 액션은 **미러되지 않는다**(좌우 모두 palm_delta_xyz[1]=+0.35).
  정책이 side 별로 학습되므로 배포에서 어떤 미러 부호도 적용하면 안 된다.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
SIM2REAL_ROOT = _SCRIPT_DIR.parent
WS_ROOT = SIM2REAL_ROOT.parent          # ~/rl_ws — 매니페스트 등 저장소 밖 자산 기준
ROBOT_CONFIG_DIR = SIM2REAL_ROOT / "config" / "robots"

_SIDE_PREFIX = {"right": "r", "left": "l"}


@dataclass(frozen=True)
class FabricsAsset:
    """Fabrics 기하 자산 — sim 학습과 반드시 동일해야 한다."""

    robot_dir: str
    robot_name: str
    class_name: str
    world: str


@dataclass(frozen=True)
class PolicyContract:
    """정책 계약. hdgp 상수와 대조해 검증한다."""

    hdgp_package: str
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class RobotProfile:
    name: str
    acting_side: str
    ee_type: str
    ee_dof: int
    fabrics: FabricsAsset
    contract: PolicyContract
    topics: dict[str, str]
    manifest_path: Path
    arm_canonical: tuple[str, ...]
    ee_canonical: tuple[str, ...]
    arm_source: tuple[str, ...]
    ee_source: tuple[str, ...]
    joint_limits: dict[str, dict]       # canonical → {source, sign, lower, upper, unit}
    tip_force_sign: float

    @property
    def side_prefix(self) -> str:
        return _SIDE_PREFIX[self.acting_side]


def _resolve(raw: str) -> Path:
    """'~' 확장 + 상대경로는 sim2real 루트 기준, 저장소 밖은 워크스페이스 기준."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    for base in (SIM2REAL_ROOT, WS_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return SIM2REAL_ROOT / p            # 존재하지 않으면 호출자가 시끄럽게 실패


def load_joint_profiles(paths) -> dict[str, dict]:
    """여러 joint profile yaml 병합.

    같은 canonical 이 두 번 나오면 **값이 완전히 같을 때만** 허용한다. 다르면 ValueError —
    조용한 덮어쓰기는 좌/우 매핑이 뒤섞이는 사고로 이어진다.
    """
    from jtc_bridge_core import load_profile_joints   # 기존 로더 재사용

    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict[str, dict] = {}
    origin: dict[str, str] = {}
    for raw in paths:
        path = _resolve(str(raw))
        for canonical, spec in load_profile_joints(path).items():
            prev = merged.get(canonical)
            if prev is not None and prev != spec:
                raise ValueError(
                    f"joint profile 충돌: {canonical!r} 이 {origin[canonical]} 와 "
                    f"{path} 에서 다르게 정의됨\n  {prev}\n  {spec}"
                )
            merged[canonical] = spec
            origin[canonical] = str(path)
    if not merged:
        raise ValueError(f"joint profile 이 비어 있음: {paths}")
    return merged


def load_manifest_joint_order(manifest_path: Path) -> list[str]:
    """자산 매니페스트의 control_joint_order (canonical 관절 순서 진실원천)."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"자산 매니페스트 없음: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text())
    order = data.get("control_joint_order")
    if not order:
        raise ValueError(f"{manifest_path}: control_joint_order 섹션 없음")
    return list(order)


def _side_joints(order: list[str], side: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """매니페스트 순서에서 해당 side 의 (팔 7, EE) canonical 을 순서 그대로 뽑는다."""
    p = _SIDE_PREFIX[side]
    arm = tuple(j for j in order if j.startswith(f"{p}_aj_"))
    ee = tuple(j for j in order if j.startswith(f"{p}_hj_"))
    if len(arm) != 7:
        raise ValueError(f"매니페스트 {side} 팔 관절이 7개가 아님: {len(arm)} ({arm})")
    if not ee:
        raise ValueError(f"매니페스트에 {side} EE 관절({p}_hj_*)이 없음")
    return arm, ee


HDGP_OPENARM_SRC = WS_ROOT / "hdgp" / "source" / "openarm"


def _check_contract(contract: PolicyContract) -> None:
    """hdgp 상수와 대조.

    hdgp 소스가 **아예 없는** 배포 전용 머신에서만 검증을 건너뛴다. 소스가 있는데
    import 가 실패하면 예외를 던진다 — 조용한 통과는 계약이 어긋난 채 정책을 로드하게
    만들고, 그 실패는 shape mismatch 나 (더 나쁘게) 잘못된 obs 로 나타난다.
    """
    import importlib
    import sys

    if not HDGP_OPENARM_SRC.exists():
        return                            # hdgp 부재 — parity 테스트가 별도로 잡는다
    if str(HDGP_OPENARM_SRC) not in sys.path:
        sys.path.insert(0, str(HDGP_OPENARM_SRC))

    side = contract.hdgp_package.rsplit(".", 2)[-2]
    module = f"{contract.hdgp_package}.grasp_{side}_constants"
    try:
        C = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"hdgp 소스는 있는데 계약 모듈을 import 하지 못했다: {module}\n"
            f"  경로 {HDGP_OPENARM_SRC}\n  원인 {exc}"
        ) from exc
    if C.NUM_OBSERVATIONS != contract.obs_dim:
        raise ValueError(
            f"obs 계약 불일치: 프로필 {contract.obs_dim} != hdgp {C.NUM_OBSERVATIONS} ({module})"
        )
    if C.NUM_ACTIONS != contract.action_dim:
        raise ValueError(
            f"action 계약 불일치: 프로필 {contract.action_dim} != hdgp {C.NUM_ACTIONS} ({module})"
        )


def load_robot_profile(name_or_path: str | Path) -> RobotProfile:
    """구성 프로필 로드 + 매니페스트/계약 대조 검증.

    name_or_path: `config/robots/` 안의 이름(확장자 생략 가능) 또는 yaml 경로.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = ROBOT_CONFIG_DIR / f"{path.name}.yaml"
    elif not path.is_absolute():
        path = _resolve(str(path))
    if not path.exists():
        available = sorted(p.stem for p in ROBOT_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"구성 프로필 없음: {path}\n  사용 가능: {available}")

    d = yaml.safe_load(path.read_text())
    side = d["acting_side"]
    if side not in _SIDE_PREFIX:
        raise ValueError(f"{path}: acting_side 는 right|left — 받은 값 {side!r}")

    manifest_path = _resolve(d["asset"]["manifest"])
    order = load_manifest_joint_order(manifest_path)
    arm_canonical, ee_canonical = _side_joints(order, side)

    ee = d["end_effector"]
    if len(ee_canonical) != int(ee["dof"]):
        raise ValueError(
            f"{path}: end_effector.dof={ee['dof']} 인데 매니페스트 {side} EE 관절은 "
            f"{len(ee_canonical)}개 ({manifest_path.name})"
        )

    limits = load_joint_profiles(d["joint_profiles"])
    missing = [j for j in (*arm_canonical, *ee_canonical) if j not in limits]
    if missing:
        raise ValueError(
            f"{path}: joint profile 에 없는 관절 {len(missing)}개 — {missing[:6]}"
            f"\n  (좌 Tesollo 라면 config/openarm_tesollo_left_hand.yaml 보충이 필요하다)"
        )

    fab = d["asset"]["fabrics"]
    contract = PolicyContract(
        hdgp_package=d["contract"]["hdgp_package"],
        obs_dim=int(d["contract"]["obs_dim"]),
        action_dim=int(d["contract"]["action_dim"]),
    )
    _check_contract(contract)

    return RobotProfile(
        name=d["name"],
        acting_side=side,
        ee_type=ee["type"],
        ee_dof=int(ee["dof"]),
        fabrics=FabricsAsset(fab["robot_dir"], fab["robot_name"], fab["class"], fab["world"]),
        contract=contract,
        topics=dict(d["topics"]),
        manifest_path=manifest_path,
        arm_canonical=arm_canonical,
        ee_canonical=ee_canonical,
        arm_source=tuple(limits[j]["source"] for j in arm_canonical),
        ee_source=tuple(limits[j]["source"] for j in ee_canonical),
        joint_limits=limits,
        tip_force_sign=float(d.get("tip_sensor", {}).get("force_sign", 1.0)),
    )


def load_hdgp_module(profile: RobotProfile, suffix: str):
    """구성에 대응하는 hdgp 모듈을 import 한다 (`preset` / `constants` / `utils`).

    하드코딩된 자세·상수를 배포에 복제하지 않기 위한 통로다. 복제하면 sim 이 값을
    바꿨을 때 조용히 어긋난다 — 좌우 미러 자세가 특히 그렇다.

        preset = load_hdgp_module(profile, "preset")
        preset.HAND_APPROACH_POSE      # side 에 맞는 20D 자세
    """
    import importlib
    import sys

    if not HDGP_OPENARM_SRC.exists():
        raise FileNotFoundError(
            f"hdgp 소스가 없어 {suffix} 를 불러올 수 없다: {HDGP_OPENARM_SRC}"
        )
    if str(HDGP_OPENARM_SRC) not in sys.path:
        sys.path.insert(0, str(HDGP_OPENARM_SRC))
    side = profile.acting_side
    return importlib.import_module(f"{profile.contract.hdgp_package}.grasp_{side}_{suffix}")


def ee_limit_arrays(profile: RobotProfile):
    """EE canonical 순서의 (lower, upper) numpy 배열 — 손가락 지령 clamp 주입용."""
    import numpy as np

    lo = np.array([profile.joint_limits[j]["lower"] for j in profile.ee_canonical], dtype=np.float64)
    hi = np.array([profile.joint_limits[j]["upper"] for j in profile.ee_canonical], dtype=np.float64)
    return lo, hi


# ---------------------------------------------------------------------------
# hdgp env_cfg 리터럴 읽기
# ---------------------------------------------------------------------------
# `grasp_{side}_env_cfg.py` 는 isaaclab(→pxr) 의존이라 import 할 수 없다. 값을 배포에
# 복제하면 sim 이 바꿨을 때 조용히 어긋나므로, **소스를 AST 로 읽어** 리터럴을 가져온다.
_CFG_NUMERIC_NODES = (ast.BinOp, ast.UnaryOp, ast.Constant, ast.Tuple, ast.List)

# env 가 cfg 가 아니라 코드에 직접 박아 둔 값 — 출처를 남긴다.
RESET_FABRICS_DAMPING_GAIN = 10.0   # grasp_{side}_env.py `self._reset_damping = 10.0 * ...`


def _eval_cfg_node(node):
    """리터럴 또는 **순수 산술식**(1.0/60.0 같은)만 평가한다. 이름 참조는 거부."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) or isinstance(sub, ast.Call) or isinstance(sub, ast.Attribute):
            raise ValueError("이름/호출이 포함된 표현식은 읽지 않는다")
        if not isinstance(sub, (*_CFG_NUMERIC_NODES, ast.operator, ast.unaryop, ast.Load)):
            raise ValueError(f"허용되지 않은 노드 {type(sub).__name__}")
    return eval(compile(ast.Expression(node), "<cfg>", "eval"), {"__builtins__": {}}, {})


def load_env_cfg_literals(path: Path) -> dict[str, object]:
    """env_cfg 소스의 `name: T = <literal>` 을 모두 수집한다.

    같은 이름이 여러 클래스에서 **다른 값**으로 나오면 예외 — 어느 쪽이 진짜인지
    추측하지 않는다.
    """
    tree = ast.parse(Path(path).read_text())
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.value is None:
            continue
        try:
            value = _eval_cfg_node(node.value)
        except (ValueError, SyntaxError, TypeError):
            continue
        name = node.target.id
        if name in out and out[name] != value:
            raise ValueError(f"{path}: {name!r} 이 서로 다른 값으로 두 번 정의됨 ({out[name]} vs {value})")
        out[name] = value
    return out


CFG_KEYS_REQUIRED = (
    "max_pose_angle", "palm_delta_xyz", "palm_delta_rot_deg", "reset_home_palm_pose",
    "fabrics_dt", "fabric_decimation", "fabrics_damping_gain", "fabrics_max_objects_per_env",
    "finger_close_speed", "couple_four_fingers", "retighten_after_latch",
    "lift_wait_joint7_delta", "warm_j7_min", "warm_j7_max",
    "lift_start_min_grip_fingers", "grasp_ready_hold_steps",
)


def load_profile_env_cfg(profile: RobotProfile) -> dict[str, object]:
    """구성에 대응하는 env_cfg 리터럴 + 코드 상수. 누락 키가 있으면 예외."""
    side = profile.acting_side
    pkg_rel = Path(profile.contract.hdgp_package.replace(".", "/"))
    path = HDGP_OPENARM_SRC / pkg_rel / f"grasp_{side}_env_cfg.py"
    if not path.exists():
        raise FileNotFoundError(f"env_cfg 소스 없음: {path}")
    cfg = load_env_cfg_literals(path)
    missing = [k for k in CFG_KEYS_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: 필요한 cfg 키 누락 {missing}")
    picked = {k: cfg[k] for k in CFG_KEYS_REQUIRED}
    picked["reset_fabrics_damping_gain"] = RESET_FABRICS_DAMPING_GAIN
    return picked
