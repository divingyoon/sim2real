"""pd 노드 단독 launch — 상류(obs/policy/fabric)가 죽어도 pd 가 같이 내려가지 않도록 분리한다.

    ros2 launch policy_control pd_controller.launch.py contract:=logs/policy/left_v2B25/deploy_contract.json \
        robot:=left_gripper_fake pd_config:=policy_control/config/pd_left.yaml fake:=true execute:=false
    ros2 launch policy_control pd_controller.launch.py contract:=logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json \
        robot:=dg5f_m_bi_fake pd_config:=dg5f_m_fake sides:=right,left fake:=true execute:=true use_source:=true

인자
  contract     deploy_contract.json (필수)
  robot        config/robots/<name>.yaml 이름 또는 경로
  pd_config    pd_*.yaml 경로, 또는 이름(`dg5f_m` → config/pd_dg5f_m.yaml, `dg5f_m_fake` → config/pd_dg5f_m_fake.yaml).
               기본 config/pd_left.yaml
  sides        쉼표 목록(left,right). 기본 '' = robot yaml 과 계약 양쪽에 있는 팔 전부(우 → 좌)
  execute      기본 false — false 면 pd 노드는 컨트롤러 토픽·controller_manager 를 건드리지 않는다
  fake         true 면 ROS_DOMAIN_ID 0/unset 거부
  use_source   true 면 venv python 으로 policy_control/policy_control/pd_node.py 실행
  params_file  선택 — 덧붙일 ROS 파라미터 yaml
pd 노드는 ROS 파라미터 contract / robot / pd_config / execute / stage / sides 를 읽는다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def _load_chain_helpers():
    """형제 launch(policy_chain.launch.py)의 헬퍼 재사용 — 파일명에 점이 있어 import 로는 못 부른다."""
    path = Path(__file__).resolve().parent / "policy_chain.launch.py"
    spec = importlib.util.spec_from_file_location("policy_chain_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chain = _load_chain_helpers()
PKG_ROOT, check_domain, is_true = _chain.PKG_ROOT, _chain.check_domain, _chain.is_true
make_node, require_file, resolve_path, resolve_robot = (
    _chain.make_node, _chain.require_file, _chain.resolve_path, _chain.resolve_robot)

PD_NODE = "pd_node"
DEFAULT_PD_CONFIG = PKG_ROOT / "config" / "pd_left.yaml"
STAGES = ("reduced", "full")
SIDES = ("left", "right")


def resolve_pd_config(value: str) -> Path:
    """이름(dg5f_m, dg5f_m_fake, right …) → config/pd_<name>.yaml, 아니면 경로."""
    value = str(value).strip()
    if not value:
        return DEFAULT_PD_CONFIG
    if "/" not in value and not value.endswith(".yaml"):
        name = value[3:] if value.startswith("pd_") else value
        return PKG_ROOT / "config" / f"pd_{name}.yaml"
    return resolve_path(value)


def parse_sides(value: str) -> str:
    """'right,left' / 'both' / '' → 정규화된 쉼표 목록(빈 문자열 = 노드 자동 선택)."""
    text = str(value).strip().lower()
    if text in ("", "both", "all"):
        return "" if text == "" else "right,left"
    sides = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in sides if s not in SIDES]
    if bad or len(set(sides)) != len(sides):
        raise RuntimeError(f"sides must be a comma list of {SIDES} (or 'both'), got {value!r}")
    return ",".join(sides)


def pd_nodes(cfg: dict) -> list:
    check_domain(is_true(cfg.get("fake", "false")))
    contract = require_file(resolve_path(cfg["contract"]), "contract")
    robot = require_file(resolve_robot(cfg["robot"]), "robot yaml")
    pd_config = require_file(resolve_pd_config(cfg.get("pd_config", "")), "pd_config")
    execute = is_true(cfg.get("execute", "false"))
    stage = str(cfg.get("stage", "reduced"))
    if stage not in STAGES:
        raise RuntimeError(f"stage must be reduced|full, got {stage!r}")
    params: list = [{"contract": str(contract), "robot": str(robot), "pd_config": str(pd_config),
                     "execute": execute, "stage": stage, "sides": parse_sides(cfg.get("sides", ""))}]
    if cfg.get("params_file", ""):
        params.append(str(require_file(resolve_path(cfg["params_file"]), "params_file")))
    return [make_node(PD_NODE, params, use_source=is_true(cfg.get("use_source", "false")))]


def _opaque(context):
    return pd_nodes(dict(context.launch_configurations))


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("contract", description="deploy_contract.json"),
        DeclareLaunchArgument("robot", description="config/robots/<name>.yaml name or path"),
        DeclareLaunchArgument("pd_config", default_value=str(DEFAULT_PD_CONFIG),
                              description="pd yaml path or name (dg5f_m | dg5f_m_fake | left | right …)"),
        DeclareLaunchArgument("sides", default_value="", description="comma list left,right | both | '' = auto"),
        DeclareLaunchArgument("execute", default_value="false", description="★true only after approval"),
        DeclareLaunchArgument("stage", default_value="reduced", description="pd max_vel stage: reduced | full"),
        DeclareLaunchArgument("fake", default_value="false", description="true → refuse ROS_DOMAIN_ID 0"),
        DeclareLaunchArgument("use_source", default_value="false"),
        DeclareLaunchArgument("params_file", default_value=""),
        OpaqueFunction(function=_opaque),
    ])
