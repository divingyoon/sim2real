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
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
SIM2REAL_ROOT = _SCRIPT_DIR.parent
WS_ROOT = SIM2REAL_ROOT.parent          # ~/rl_ws — 매니페스트 등 저장소 밖 자산 기준
ROBOT_CONFIG_DIR = SIM2REAL_ROOT / "config" / "robots"

_SIDE_PREFIX = {"right": "r", "left": "l"}

#: 팔 액션 규약. 기본값을 두지 않는다 — 짐작이 곧 §8 의 사고다.
DELTA_ANCHOR = "delta_anchor"
ABSOLUTE_PALM = "absolute_palm"
ACTION_CONVENTIONS = frozenset({DELTA_ANCHOR, ABSOLUTE_PALM})


def profiles_with_convention(convention: str) -> list[str]:
    """그 규약을 선언한 구성 이름들. 계약 테스트가 대상을 고를 때 쓴다.

    이름을 테스트에 나열하지 않는 이유: 새 구성이 추가되면 자동으로 대상이 되어야 하고,
    빠지더라도 **조용히** 빠지면 안 된다. 규약 필드가 그 판정을 대신한다.
    """
    out = []
    for name in available_profiles():
        if load_robot_profile(name).action_convention == convention:
            out.append(name)
    return out


@dataclass(frozen=True)
class FabricsAsset:
    """Fabrics 기하 자산 — sim 학습과 반드시 동일해야 한다."""

    robot_dir: str
    robot_name: str
    class_name: str
    world: str


@dataclass(frozen=True)
class PolicyContract:
    """정책 계약과 **그것을 무엇에 대고 검증하는지**.

    DirectRL 태스크는 `grasp_{side}_constants.py` 에 `NUM_OBSERVATIONS` 를 적어 두므로
    소스가 진실원천이다. manager-based 태스크(`gripper/left/grasp_sensor`)는 그런 파일이
    없다 — obs 는 `ObservationManager` 가 런타임에 조립하고, 항의 차원은 대부분 상위
    스코프라 정적으로 셀 수 없다. 그 경우 진실원천은 **학습 산출물**이다: actor 첫 층이
    obs, mu 헤드가 action. 어느 쪽이든 검증을 건너뛰지는 않는다.
    """

    hdgp_package: str
    obs_dim: int
    action_dim: int
    source: str = "constants"           # "constants" | "checkpoint"
    checkpoint: Path | None = None


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
    # 유휴(반대편) 팔 — 명령하지는 않지만 **자세를 확인**해야 한다.
    # sim 은 유휴 팔을 rest 로 고정한 채 학습하고 그 팔은 물리 충돌체로 존재한다.
    # 실기 유휴 팔이 다른 곳에 있으면 장면이 달라져 학습된 궤적이 안전하지 않다.
    idle_arm_canonical: tuple[str, ...]
    idle_arm_source: tuple[str, ...]
    joint_limits: dict[str, dict]       # canonical → {source, sign, lower, upper, unit}
    tip_force_sign: float
    # 팔 액션이 무엇을 뜻하는가 — "delta_anchor" | "absolute_palm".
    # §8 의 교훈을 계약으로 올린 것이다: 스칼라만 맞춰서는 부족하고 **무엇을 기준으로 한
    # 델타인지**까지 선언해야 한다. 두 규약은 a=0 의 의미가 정반대라서, 하나를 위해 쓴
    # 계약 테스트를 다른 쪽에 적용하면 사실이 아닌 것을 단언하게 된다.
    action_convention: str

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
    """매니페스트 순서에서 해당 side 의 (팔 7, EE) canonical 을 순서 그대로 뽑는다.

    EE 는 구성마다 다르다(Tesollo 20 / 2지 그리퍼 1~2) — 개수를 강요하지 않는다.
    """
    p = _SIDE_PREFIX[side]
    arm = tuple(j for j in order if j.startswith(f"{p}_aj_"))
    ee = tuple(j for j in order if j.startswith(f"{p}_hj_"))
    if len(arm) != 7:
        raise ValueError(f"매니페스트 {side} 팔 관절이 7개가 아님: {len(arm)} ({arm})")
    if not ee:
        raise ValueError(f"매니페스트에 {side} EE 관절({p}_hj_*)이 없음")
    return arm, ee


def idle_arm_rest_pose(profile: "RobotProfile") -> list[float]:
    """유휴 팔의 rest 자세(7,) — preset 의 `{반대편}_ARM_REST_JOINT_POS`.

    sim 은 이 자세로 유휴 팔을 고정한 채 학습한다. 실기도 같아야 장면이 일치한다.
    """
    preset = load_hdgp_module(profile, "preset")
    other = "LEFT" if profile.acting_side == "right" else "RIGHT"
    rest = getattr(preset, f"{other}_ARM_REST_JOINT_POS", None)
    if rest is None:
        raise AttributeError(f"preset 에 {other}_ARM_REST_JOINT_POS 가 없다")
    return [float(rest[j]) for j in profile.idle_arm_canonical]


HDGP_OPENARM_SRC = WS_ROOT / "hdgp" / "source" / "openarm"


_ACTOR_FIRST_LAYER = "a2c_network.actor_mlp.0.weight"
_MU_HEAD = "a2c_network.mu.weight"
#: obs 정규화 통계. 길이 = **원 관측 차원**. rl_games 는 정규화를 raw obs 에 걸므로
#: 망 구조(RNN 유무·concat_input)와 무관하게 이것이 가장 곧은 진실원천이다.
_OBS_NORM = "running_mean_std.running_mean"
_RNN_INPUT = "a2c_network.rnn.rnn.weight_ih_l0"
_RNN_HIDDEN = "a2c_network.rnn.rnn.weight_hh_l0"


def checkpoint_is_recurrent(state: dict) -> bool:
    return any(".rnn." in k for k in state)


def _obs_dim_from_state(state: dict, where: str) -> int:
    """관측 차원. **`actor_mlp` 첫 층을 그냥 믿으면 안 된다.**

    rl_games 의 RNN 망은 `concat_input: true` 일 때 actor_mlp 입력이
    `rnn_hidden + obs` 다. fab_test42 실측: actor_mlp 입력 1096 = LSTM 1024 + obs **72**.
    1096 을 obs 로 읽으면 계약 검증이 통과할 리 없는 값으로 조용히 어긋난다.
    """
    if _OBS_NORM in state:
        return int(state[_OBS_NORM].reshape(-1).shape[0])
    if _RNN_INPUT in state:
        # RNN 이 mlp 앞에 있으면 rnn 입력이 곧 obs 다.
        return int(state[_RNN_INPUT].shape[1])
    if _ACTOR_FIRST_LAYER in state:
        if checkpoint_is_recurrent(state):
            raise KeyError(
                f"{where}: RNN 체크포인트인데 obs 차원을 곧게 읽을 텐서가 없다"
                f"({_OBS_NORM} · {_RNN_INPUT} 둘 다 없음). {_ACTOR_FIRST_LAYER} 는 "
                "concat_input 때문에 obs 가 아닐 수 있어 쓰지 않는다."
            )
        return int(state[_ACTOR_FIRST_LAYER].shape[1])
    raise KeyError(f"{where}: obs 차원을 읽을 텐서가 없다")


def checkpoint_contract(checkpoint: Path) -> tuple[int, int]:
    """rl_games 체크포인트가 **실제로** 학습된 (obs, action) 차원.

    짐작한 차원으로 정책을 로드하면 shape mismatch 로 죽거나(운이 좋은 경우)
    잘못된 obs 로 조용히 돈다.
    """
    import torch

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    if _MU_HEAD not in state:
        raise KeyError(
            f"{checkpoint}: 액션 차원을 읽을 {_MU_HEAD} 가 없다\n"
            f"  (rl_games actor 체크포인트가 맞는지 확인할 것)"
        )
    return _obs_dim_from_state(state, str(checkpoint)), int(state[_MU_HEAD].shape[0])


def _check_contract_against_checkpoint(contract: PolicyContract) -> None:
    if contract.checkpoint is None:
        raise ValueError(
            f"{contract.hdgp_package}: source=checkpoint 인데 contract.checkpoint 가 없다"
        )
    if not contract.checkpoint.is_file():
        # 학습 머신에만 있는 산출물이라 배포 머신에서는 없을 수 있다. 그래도 조용히
        # 넘기지 않고 무엇을 못 봤는지 말한다 — 테스트가 이 경로를 따로 덮는다.
        print(
            f"[robot_profile] 경고: 체크포인트가 없어 계약을 검증하지 못했다: "
            f"{contract.checkpoint}"
        )
        return
    obs_dim, action_dim = checkpoint_contract(contract.checkpoint)
    if (obs_dim, action_dim) != (contract.obs_dim, contract.action_dim):
        raise ValueError(
            f"계약 불일치: 프로필 obs={contract.obs_dim} act={contract.action_dim} != "
            f"체크포인트 obs={obs_dim} act={action_dim} ({contract.checkpoint})"
        )


def _check_contract(contract: PolicyContract) -> None:
    """hdgp 상수와 대조.

    hdgp 소스가 **아예 없는** 배포 전용 머신에서만 검증을 건너뛴다. 소스가 있는데
    import 가 실패하면 예외를 던진다 — 조용한 통과는 계약이 어긋난 채 정책을 로드하게
    만들고, 그 실패는 shape mismatch 나 (더 나쁘게) 잘못된 obs 로 나타난다.
    """
    import importlib
    import sys

    if contract.source == "checkpoint":
        _check_contract_against_checkpoint(contract)
        return
    if contract.source != "constants":
        raise ValueError(f"contract.source 는 constants|checkpoint — 받은 값 {contract.source!r}")

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

    idle_side = "left" if side == "right" else "right"
    idle_arm, _ = _side_joints(order, idle_side)

    limits = load_joint_profiles(d["joint_profiles"])
    missing = [j for j in (*arm_canonical, *ee_canonical, *idle_arm) if j not in limits]
    if missing:
        raise ValueError(
            f"{path}: joint profile 에 없는 관절 {len(missing)}개 — {missing[:6]}"
            f"\n  (좌 Tesollo 라면 config/openarm_tesollo_left_hand.yaml 보충이 필요하다)"
        )

    convention = d.get("action", {}).get("convention")
    if convention not in ACTION_CONVENTIONS:
        raise ValueError(
            f"{path}: action.convention 이 {sorted(ACTION_CONVENTIONS)} 중 하나여야 한다 "
            f"— 받은 값 {convention!r}. 선언하지 않으면 배포가 a=0 의 뜻을 짐작하게 된다."
        )

    fab = d["asset"]["fabrics"]
    raw_contract = d["contract"]
    raw_checkpoint = raw_contract.get("checkpoint")
    contract = PolicyContract(
        hdgp_package=raw_contract["hdgp_package"],
        obs_dim=int(raw_contract["obs_dim"]),
        action_dim=int(raw_contract["action_dim"]),
        source=raw_contract.get("source", "constants"),
        checkpoint=_resolve(str(raw_checkpoint)) if raw_checkpoint else None,
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
        idle_arm_canonical=idle_arm,
        idle_arm_source=tuple(limits[j]["source"] for j in idle_arm),
        joint_limits=limits,
        tip_force_sign=float(d.get("tip_sensor", {}).get("force_sign", 1.0)),
        action_convention=convention,
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


def load_env_cfg_literals(path: Path, class_name: str | None = None) -> dict[str, object]:
    """env_cfg 소스의 `name: T = <literal>` 을 수집한다.

    같은 이름이 여러 클래스에서 **다른 값**으로 나오면 예외 — 어느 쪽이 진짜인지
    추측하지 않는다. 한 파일에 좌·우 cfg 가 같이 사는 경우가 있으므로(grasp_s2r 은
    `GraspS2RTesolloRightEnvCfg` 와 `GraspS2RGripperLeftEnvCfg` 가 한 파일이다)
    `class_name` 으로 범위를 좁힐 수 있다.
    """
    tree = ast.parse(Path(path).read_text())
    scope: ast.AST = tree
    if class_name is not None:
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == class_name]
        if not found:
            raise KeyError(f"{path}: 클래스 {class_name!r} 이 없다")
        scope = found[0]
    out: dict[str, object] = {}
    for node in ast.walk(scope):
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


# 있으면 함께 싣는 키(구성에 따라 없을 수 있어 필수는 아니다)
CFG_KEYS_OPTIONAL = ("object_spawn_x_center", "object_spawn_y_center",
                     "object_spawn_x_range", "object_spawn_y_range")

CFG_KEYS_REQUIRED = (
    "max_pose_angle", "palm_delta_xyz", "palm_delta_rot_deg", "reset_home_palm_pose",
    "fabrics_dt", "fabric_decimation", "fabrics_damping_gain", "fabrics_max_objects_per_env",
    "finger_close_speed", "couple_four_fingers", "retighten_after_latch",
    "lift_wait_joint7_delta", "warm_j7_min", "warm_j7_max",
    "lift_start_min_grip_fingers", "grasp_ready_hold_steps",
    # 컵 정준 액션 기준점(anchor="cup")의 유일한 출처. home anchor 구성에서도
    # 존재하므로 필수로 둔다 — 없으면 조용히 0 을 쓰는 대신 예외가 난다.
    "pregrasp_offset_x", "pregrasp_offset_y", "pregrasp_offset_z",
)


def load_profile_env_cfg(profile: RobotProfile) -> dict[str, object]:
    """구성에 대응하는 env_cfg 리터럴 + 코드 상수. 누락 키가 있으면 예외."""
    side = profile.acting_side
    path = hdgp_task_dir(profile) / f"grasp_{side}_env_cfg.py"
    if not path.exists():
        raise FileNotFoundError(f"env_cfg 소스 없음: {path}")
    cfg = load_env_cfg_literals(path)
    missing = [k for k in CFG_KEYS_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: 필요한 cfg 키 누락 {missing}")
    picked = {k: cfg[k] for k in CFG_KEYS_REQUIRED}
    picked.update({k: cfg[k] for k in CFG_KEYS_OPTIONAL if k in cfg})
    picked["reset_fabrics_damping_gain"] = RESET_FABRICS_DAMPING_GAIN
    return picked


def load_action_anchor(profile: RobotProfile) -> str:
    """액션 델타의 **기준점**을 sim env 소스에서 유도한다 → "home" | "cup".

    왜 유도하는가: 프로필에 손으로 적어두면 sim 이 바뀔 때 조용히 어긋난다. 실제로
    `grasp_sensor` 는 hdgp c99b37d(2026-08-19)에서 기준점을 홈 → 컵 정준 pregrasp 로
    옮겼고, 배포는 그대로 홈이라 팔이 홈 근처에서만 움직이는 상태였다.

    판별 지점: 고정 홈 분기(`if self.q_home_arm is not None:`) 안에서
    `pregrasp_palm_pose` 를 `self.home_palm_pose` 로 **덮어쓰면 home**, 그대로 두면
    (컵 + pregrasp_offset 이 살아남으므로) **cup**.

    분기 자체를 못 찾으면 추측하지 않고 예외를 던진다.
    """
    side = profile.acting_side
    path = hdgp_task_dir(profile) / f"grasp_{side}_env.py"
    if not path.exists():
        raise FileNotFoundError(f"env 소스 없음: {path}")
    tree = ast.parse(path.read_text())

    def _is_home_branch(test) -> bool:
        # self.q_home_arm is not None
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "q_home_arm"
            and any(isinstance(op, ast.IsNot) for op in test.ops)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_home_branch(node.test):
            continue
        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not isinstance(sub, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "pregrasp_palm_pose" for t in sub.targets
            ):
                continue
            if any(
                isinstance(n, ast.Attribute) and n.attr == "home_palm_pose"
                for n in ast.walk(sub.value)
            ):
                return "home"
        return "cup"

    raise ValueError(
        f"{path}: 고정 홈 분기(`self.q_home_arm is not None`)를 찾지 못해 액션 기준점을 "
        "판별할 수 없다 — sim 구조가 바뀌었으면 이 판별식을 갱신하라"
    )


def load_arm_mirror_sign(profile: RobotProfile) -> list[float]:
    """env 소스에서 `_ARM_MIRROR_SIGN` 을 읽는다.

    preset 이 아니라 `grasp_{side}_env.py` 에 있고, env 는 isaaclab 의존이라 import 할 수
    없다 → 소스를 AST 로 읽는다. `getattr(preset, ...)` 로 찾으면 **항상 None 이라
    검증이 조용히 통과**한다(실제로 그 상태였다).
    """
    side = profile.acting_side
    path = hdgp_task_dir(profile) / f"grasp_{side}_env.py"
    if not path.exists():
        raise FileNotFoundError(f"env 소스 없음: {path}")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_ARM_MIRROR_SIGN" for t in node.targets
        ):
            return [float(v) for v in ast.literal_eval(node.value)]
    raise ValueError(f"{path}: _ARM_MIRROR_SIGN 을 찾지 못했다")


def expected_q_home_arm(profile: RobotProfile) -> "list[float]":
    """sim preset 이 강제하는 정답 q_home 을 **유도**한다.

    sim `_build_home_pose` 는 `반대편_ARM_REST == _ARM_MIRROR_SIGN × q_home`(오차 ≤0.05)
    를 시작 시 강제한다. 따라서 반대편 rest 값에서 q_home 을 되돌릴 수 있고, 이것이
    배포 홈 IK 의 회귀 기준값이 된다(자산이 어긋나면 여기서 갈린다).
    """
    preset = load_hdgp_module(profile, "preset")
    other = "LEFT" if profile.acting_side == "right" else "RIGHT"
    rest = getattr(preset, f"{other}_ARM_REST_JOINT_POS", None)
    if rest is None:
        raise AttributeError(f"preset 에 {other}_ARM_REST_JOINT_POS 가 없다")
    # rest 는 {관절명: 값} dict(27개) — 반대편 팔 7관절만 순서대로 뽑는다.
    other_prefix = "l" if profile.acting_side == "right" else "r"
    keys = [f"{other_prefix}_aj_{i}" for i in range(1, 8)]
    missing = [k for k in keys if k not in rest]
    if missing:
        raise KeyError(f"{other}_ARM_REST_JOINT_POS 에 없는 관절: {missing}")
    sign = load_arm_mirror_sign(profile)
    return [float(sg) * float(rest[k]) for sg, k in zip(sign[:7], keys)]


def _task_sources(profile: RobotProfile) -> list[Path]:
    """자산 이름이 적힐 수 있는 그 태스크의 파일들.

    DirectRL 태스크는 `grasp_{side}_env.py` 안에서 fabric 을 직접 만든다. manager-based
    태스크는 env 파일 자체가 없고 `*_fabric_action.py` 가 만들며 이름은 preset 상수로
    간접 참조한다. 한 형태만 보면 다른 형태에서는 **검사가 조용히 사라진다** — 그리고
    사라진 검사는 통과처럼 보인다.
    """
    task = hdgp_task_dir(profile)
    if not task.is_dir():
        return []
    named = [task / f"grasp_{profile.acting_side}_env.py"]
    named += sorted(task.glob("*_fabric_action.py"))
    named += sorted(task.glob("grasp_*_preset.py"))
    return [path for path in named if path.is_file()]


def _literal_and_constant_names(profile: RobotProfile, keys: tuple[str, ...]) -> set[str]:
    """소스에서 `key="값"` 리터럴을 모으고, `P.CONST` 참조는 preset 값으로 푼다."""
    found: set[str] = set()
    sources = _task_sources(profile)
    text = "\n".join(path.read_text() for path in sources)
    for key in keys:
        found |= set(re.findall(rf'{key}\s*=\s*"([^"]+)"', text))
        for const in re.findall(rf"{key}\s*=\s*(?:P|preset)\.([A-Z_][A-Z0-9_]*)", text):
            value = _preset_constant(profile, const)
            if value is not None:
                found.add(value)
    return found


def _preset_constant(profile: RobotProfile, name: str):
    for path in hdgp_task_dir(profile).glob("grasp_*_preset.py"):
        match = re.search(rf'^{name}\s*=\s*"([^"]+)"', path.read_text(), re.M)
        if match:
            return match.group(1)
    return None


def sim_fabrics_assets(profile: RobotProfile) -> set[str]:
    """sim 이 이 태스크에서 쓰는 fabric 로봇 디렉토리 이름들."""
    return _literal_and_constant_names(profile, ("robot_dir_name", "robot_name"))


def sim_fabrics_worlds(profile: RobotProfile) -> set[str]:
    """sim 이 이 태스크에서 쓰는 fabric world 이름들."""
    return _literal_and_constant_names(profile, ("world_filename",))



def available_profiles() -> list[str]:
    """`config/robots/` 의 구성 프로필 이름 목록(정렬).

    테스트가 이걸로 파라미터화하면 **새 구성을 추가하는 순간 자동으로 검증 대상**이 된다.
    """
    return sorted(p.stem for p in ROBOT_CONFIG_DIR.glob("*.yaml"))


def hdgp_task_dir(profile: RobotProfile) -> Path:
    """구성에 대응하는 hdgp 태스크 소스 디렉토리."""
    return HDGP_OPENARM_SRC / Path(profile.contract.hdgp_package.replace(".", "/"))
