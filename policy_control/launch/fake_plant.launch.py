"""무하드웨어 플랜트 — 컨트롤러·손·물체·손끝 센서 자리를 fake 노드로 채운다.

    ros2 launch policy_control fake_plant.launch.py side:=left                       # 좌 그리퍼(레거시 프로필)
    ros2 launch policy_control fake_plant.launch.py side:=right                      # 우 DG-5F(레거시 프로필, 손 echo)
    ros2 launch policy_control fake_plant.launch.py side:=left robot:=dg5f_m_left_fake \
        contract:=logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json          # 새 자산: 계약 + robot yaml 로 배선
    ros2 launch policy_control fake_plant.launch.py side:=both robot:=dg5f_m_bi_fake contract:=…   # 양팔

robot:= 이 있으면 **계약 모드**: fake_arm_bridge --contract/--robot-yaml/--sides (팔마다 MockArm, controller_manager
스텁 하나가 양팔 JTC+forward 를 안다, 중력은 pd yaml 의 모델과 같은 식), 팔마다 fake_hand_state_pub(namespace dg5f_<side>,
/policy_control/joint_target 을 그 팔 관절로 걸러 반사) + fake_tip_contact_pub(namespace). 없으면 레거시 프로필 모드.

★안전: ROS_DOMAIN_ID 가 비어 있거나 0(실기 기본 도메인)이면 기동을 거부한다 — 같은 호스트의
  실팔 DDS 그래프에 fake 메시지가 섞이는 사고의 유일한 소프트웨어 방어선이다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction

PKG_ROOT = Path(__file__).resolve().parents[1]
SIM2REAL = PKG_ROOT.parent
RL_WS = SIM2REAL.parent
FAKES = SIM2REAL / "scripts" / "fakes"
PY = str(SIM2REAL / ".venv" / "bin" / "python")

ROBOT_BY_SIDE = {"left": "gripper_left", "right": "tesollo_sensor__right"}   # 레거시 프로필(scripts/robot_profile)
SIDES_BY_ARG = {"left": ("left",), "right": ("right",), "both": ("right", "left")}
OBJECT_TOPIC = "/objects/cup_big_s100/pose"
DEFAULT_PD_CONFIG = PKG_ROOT / "config" / "pd_dg5f_m_fake.yaml"
HAND_HZ = "60"


def _refuse_real_domain() -> None:
    domain = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if domain in ("", "0"):
        raise RuntimeError(
            "fake_plant refuses to start on ROS_DOMAIN_ID 0/unset (the real robot's domain). "
            "export ROS_DOMAIN_ID=99 (or any non-zero test domain) first.")


def _script(name: str, *args: str) -> ExecuteProcess:
    """venv python 으로 scripts/fakes/<name>.py 실행(launch_ros.Node 는 패키지 이름이 필수)."""
    return ExecuteProcess(cmd=[PY, str(FAKES / f"{name}.py"), *args], name=name, output="screen")


def _resolve(value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    for base in (Path.cwd(), SIM2REAL, RL_WS):
        if (base / p).exists():
            return base / p
    return RL_WS / p


def _require(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{what} not found: {path}")
    return path


def resolve_robot(value: str) -> Path:
    if "/" not in value and not value.endswith(".yaml"):
        return PKG_ROOT / "config" / "robots" / f"{value}.yaml"
    return _resolve(value)


def _plant_args(cfg: dict) -> list[str]:
    model = cfg.get("plant_model", "pd")
    if model not in ("pd", "rate"):
        raise RuntimeError(f"plant_model must be pd|rate, got {model!r}")
    plant = ["--model", model, "--forward", "--rate-hz", cfg["plant_hz"]]
    if model == "pd":
        plant += ["--friction-scale", cfg.get("plant_friction", "1.0")]
    return plant


def legacy_nodes(cfg: dict, side: str) -> list:
    """레거시 프로필 모드(gripper_left / tesollo_sensor__right)."""
    if side not in ROBOT_BY_SIDE:
        raise RuntimeError(f"side must be left|right without robot:=, got {side!r}")
    robot = ROBOT_BY_SIDE[side]
    plant = ["--robot", robot, *_plant_args(cfg)]
    if cfg.get("plant_model", "pd") == "pd":
        plant.append("--gravity")
    if cfg.get("contract"):
        # 유효관성은 계약 홈 자세에서 — 차렷(펴진 팔)의 관성으로는 플랜트가 sim 보다 몇 배 굼뜨다
        home = json.loads(Path(cfg["contract"]).read_text())["pd"]["home_arm"]
        plant += ["--inertia-q=" + ",".join(f"{v:.6f}" for v in home)]   # 음수로 시작해 --inertia-q= 형태
    nodes = [_script("fake_arm_bridge", *plant), _cup(cfg)]
    if side == "right":
        nodes.append(_script("fake_hand_state_pub", "--robot", robot, "--rate", HAND_HZ,
                             "--echo-topic", "/policy_control/joint_target", "--controller-node"))
        nodes.append(_script("fake_tip_contact_pub", "--robot", robot, "--rate", HAND_HZ))
    return nodes


def contract_nodes(cfg: dict, sides: tuple) -> list:
    """계약 모드: 팔마다 MockArm(하나의 브리지) + 팔마다 손/손끝 fake(namespace dg5f_<side>)."""
    contract = _require(_resolve(cfg["contract"]), "contract")
    robot = _require(resolve_robot(cfg["robot"]), "robot yaml")
    pd_config = _require(_resolve(cfg.get("pd_config") or str(DEFAULT_PD_CONFIG)), "pd_config")
    common = ["--contract", str(contract), "--robot-yaml", str(robot)]
    plant = [*common, "--sides", ",".join(sides), "--pd-config", str(pd_config), *_plant_args(cfg)]
    if cfg.get("inertia_q"):
        plant.append("--inertia-q=" + cfg["inertia_q"])
    nodes = [_script("fake_arm_bridge", *plant), _cup(cfg)]
    for side in sides:
        nodes.append(_script("fake_hand_state_pub", *common, "--side", side, "--rate", HAND_HZ,
                             "--echo-topic", "/policy_control/joint_target", "--controller-node"))
        nodes.append(_script("fake_tip_contact_pub", "--namespace", f"dg5f_{side}", "--rate", HAND_HZ))
    return nodes


def _cup(cfg: dict) -> ExecuteProcess:
    return _script("fake_cup_pose_pub", "--topic", OBJECT_TOPIC, "--x", cfg["cup_x"], "--y", cfg["cup_y"],
                   "--z", cfg["cup_z"])


def plant_nodes(cfg: dict) -> list:
    """launch_configurations(dict) → 프로세스 목록. 검증은 여기서(실패 = 기동 거부)."""
    _refuse_real_domain()
    side = str(cfg.get("side", "left"))
    if side not in SIDES_BY_ARG:
        raise RuntimeError(f"side must be left|right|both, got {side!r}")
    if cfg.get("robot"):
        if not cfg.get("contract"):
            raise RuntimeError("robot:= (contract mode) needs contract:=")
        return contract_nodes(cfg, SIDES_BY_ARG[side])
    if side == "both":
        raise RuntimeError("side:=both needs robot:= (contract mode, e.g. robot:=dg5f_m_bi_fake)")
    return legacy_nodes(cfg, side)


def _nodes(context):
    return plant_nodes(dict(context.launch_configurations))


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("side", default_value="left", description="left | right | both"),
        DeclareLaunchArgument("robot", default_value="",
                              description="policy_control robot yaml name (dg5f_m_left_fake …) → contract mode"),
        DeclareLaunchArgument("contract", default_value="", description="deploy_contract.json (required with robot:=)"),
        DeclareLaunchArgument("pd_config", default_value="",
                              description="contract mode: pd yaml whose gravity model the plant reuses (default pd_dg5f_m_fake)"),
        DeclareLaunchArgument("inertia_q", default_value="",
                              description="contract mode: 7 CSV joint pose for the effective inertia (default = contract home)"),
        DeclareLaunchArgument("plant_hz", default_value="100.0"),
        DeclareLaunchArgument("plant_friction", default_value="1.0",
                              description="pd 모델 쿨롱 마찰 배율 (0 = sim 처럼 마찰 없음)"),
        DeclareLaunchArgument("plant_model", default_value="pd",
                              description="pd = 실측 게인 PD+마찰+중력 모델 | rate = 속도제한만(이상 추종, 배선 검증용)"),
        # 좌 v2B25 학습 스폰 중심(x 0.38, y 0.19) · 컵 원점 z = **학습 sim 테이블 0.200** + 0.09209 = 0.29209
        # (정책이 본 유일한 z — left_inference_node TRAIN_CUP_Z). 실기 datum 0.205 는 실기 FP++ 가 준다.
        DeclareLaunchArgument("cup_x", default_value="0.38"),
        DeclareLaunchArgument("cup_y", default_value="0.19"),
        DeclareLaunchArgument("cup_z", default_value="0.29209"),
        OpaqueFunction(function=_nodes),
    ])
