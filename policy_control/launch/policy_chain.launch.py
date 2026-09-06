"""obs → policy → fabric 노드 (pd 노드는 pd_controller.launch.py — 별도 launch 가 설계다).

    ros2 launch policy_control policy_chain.launch.py contract:=logs/policy/left_v2B25/deploy_contract.json \
        robot:=left_gripper_fake fake:=true use_source:=true
    ros2 launch policy_control policy_chain.launch.py contract:=logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json \
        robot:=dg5f_m_bi_fake side:=both fake:=true use_source:=true       # 제어 전용: fabric_node_{right,left} (direct 모드) 만

인자
  contract     deploy_contract.json 경로 (필수; rl_ws 기준 상대 또는 절대)
  robot        config/robots/<name>.yaml 의 이름 또는 yaml 경로
  side         left | right | both | '' (기본 = 계약 primary_side). obs/fabric 노드의 ROS 파라미터 `side` 로 넘긴다.
               both 는 fabric 노드를 팔마다 하나씩(`fabric_node_right`, `fabric_node_left` — fabrics_sim 은 프로세스당 1개) 띄운다.
  device       정책·fabric 디바이스 (기본 cuda:0)
  fake         true 면 ROS_DOMAIN_ID 0/unset 을 거부한다 (fake_plant 와 같은 방어선). 실기는 기본 도메인.
  use_source   true 면 colcon 설치 없이 venv python 으로 policy_control/policy_control/<node>.py 를 띄운다
  params_file  선택 — 모든 노드에 덧붙일 ROS 파라미터 yaml
계약이 control_only(정책 없음)면 obs/policy 노드 대신 episode_master(에피소드 서비스/이벤트)를 띄운다 — fabric 노드가 direct 모드
(/policy_control/palm_cmd · hand_cmd 구독)로 돈다. 각 노드는 ROS 파라미터 contract / robot / device (+ side) 를 읽는다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node

PKG_ROOT = Path(__file__).resolve().parents[1]           # sim2real/policy_control
SIM2REAL = PKG_ROOT.parent
RL_WS = SIM2REAL.parent
PY = str(SIM2REAL / ".venv" / "bin" / "python")
PACKAGE = "policy_control"
CHAIN_NODES = ("obs_node", "policy_node", "fabric_node")
SIDES = ("left", "right")
SIDE_ORDER = ("right", "left")                             # 양팔 규약: 우 먼저
TRUE = ("true", "1", "yes")


def is_true(value: str) -> bool:
    return str(value).strip().lower() in TRUE


def check_domain(fake: bool) -> None:
    """fake 모드는 실기 기본 도메인(0/unset)을 거부한다."""
    domain = os.environ.get("ROS_DOMAIN_ID", "").strip()
    if fake and domain in ("", "0"):
        raise RuntimeError(
            "fake:=true refuses ROS_DOMAIN_ID 0/unset (the real robot's domain). "
            "export ROS_DOMAIN_ID=99 (or any non-zero test domain) first.")


def resolve_path(value: str, base: Path | None = None) -> Path:
    """절대경로 그대로; 상대경로는 cwd → sim2real → rl_ws 순으로 **존재하는 첫 것**(노드와 같은 규칙)."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    bases = [base] if base is not None else [Path.cwd(), SIM2REAL, RL_WS]
    for b in bases:
        if (b / p).exists():
            return b / p
    return bases[-1] / p


def resolve_robot(value: str) -> Path:
    """이름(left_gripper_fake) → config/robots/<name>.yaml, 아니면 경로."""
    if "/" not in value and not value.endswith(".yaml"):
        return PKG_ROOT / "config" / "robots" / f"{value}.yaml"
    return resolve_path(value)


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{what} not found: {path}")
    return path


def make_node(name: str, params: list, *, use_source: bool, extra_args: list[str] | None = None,
              node_name: str | None = None) -> Node:
    """console_script(ros2 run 식) 또는 소스 파일(venv python) — 파라미터는 같은 방식으로 전달.

    ``name`` 은 실행 파일(console_script / <name>.py), ``node_name`` 은 ROS 노드 이름(기본 = name)."""
    node_name = node_name or name
    if use_source:
        # launch_ros.Node 는 패키지 이름이 필수라 venv python 스크립트는 ExecuteProcess 로 띄운다.
        script = require_file(PKG_ROOT / PACKAGE / f"{name}.py", f"{name} source")
        cmd = [PY, str(script), *(extra_args or []), "--ros-args", "-r", f"__node:={node_name}"]
        for p in params:
            if isinstance(p, dict):
                for k, v in p.items():
                    cmd += ["-p", f"{k}:={str(v).lower() if isinstance(v, bool) else v}"]
            else:
                cmd += ["--params-file", str(p)]
        return ExecuteProcess(cmd=cmd, name=node_name, output="screen")
    return Node(package=PACKAGE, executable=name, name=node_name, arguments=list(extra_args or []),
                parameters=params, output="screen")


def parse_side(value: str, contract_sides: list) -> list:
    """side 인자 → 팔 목록. '' = [''](계약 primary 에 맡긴다), both = 계약의 팔 전부(우 먼저)."""
    text = str(value).strip().lower()
    if text == "":
        return [""]
    if text in ("both", "all"):
        sides = [s for s in SIDE_ORDER if s in contract_sides]
        if len(sides) < 2:
            raise RuntimeError(f"side:=both needs a bimanual contract (contract sides {contract_sides})")
        return sides
    if text not in SIDES:
        raise RuntimeError(f"side must be left|right|both, got {value!r}")
    if contract_sides and text not in contract_sides:
        raise RuntimeError(f"side {text!r} is not in the contract (sides {contract_sides})")
    return [text]


def contract_facts(contract: Path) -> tuple[bool, list]:
    """(control_only, sides) — launch 는 계약 JSON 의 두 필드만 본다(검증은 노드가 한다)."""
    try:
        raw = json.loads(contract.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"contract unreadable: {contract}: {exc}") from exc
    return bool(raw.get("control_only", False)), list(raw.get("sides", {}) or {})


def chain_nodes(cfg: dict) -> list[Node]:
    """launch_configurations(dict) → Node 액션들. 검증은 여기서 (실패 = 기동 거부)."""
    check_domain(is_true(cfg.get("fake", "false")))
    contract = require_file(resolve_path(cfg["contract"]), "contract")
    robot = require_file(resolve_robot(cfg["robot"]), "robot yaml")
    control_only, contract_sides = contract_facts(contract)
    sides = parse_side(cfg.get("side", ""), contract_sides)
    base = {"contract": str(contract), "robot": str(robot), "device": str(cfg.get("device", "cuda:0"))}
    extra = [str(require_file(resolve_path(cfg["params_file"]), "params_file"))] if cfg.get("params_file", "") else []
    use_source = is_true(cfg.get("use_source", "false"))
    nodes: list = []
    if not control_only:
        nodes.append(make_node("obs_node", [{**base, "side": sides[0]}, *extra], use_source=use_source))
        nodes.append(make_node("policy_node", [base, *extra], use_source=use_source))
    else:   # 정책이 없으면 obs 노드(에피소드 마스터)도 없다 → 같은 서비스/이벤트를 내는 작은 마스터
        nodes.append(make_node("episode_master", [{"contract": str(contract)}], use_source=use_source))
    for side in sides:
        name = "fabric_node" if len(sides) == 1 else f"fabric_node_{side}"
        nodes.append(make_node("fabric_node", [{**base, "side": side}, *extra], use_source=use_source, node_name=name))
    return nodes


def _opaque(context):
    return chain_nodes(dict(context.launch_configurations))


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("contract", description="deploy_contract.json"),
        DeclareLaunchArgument("robot", description="config/robots/<name>.yaml name or path"),
        DeclareLaunchArgument("side", default_value="", description="left | right | both | '' = contract primary_side"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("fake", default_value="false", description="true → refuse ROS_DOMAIN_ID 0"),
        DeclareLaunchArgument("use_source", default_value="false", description="run sources with the venv python"),
        DeclareLaunchArgument("params_file", default_value=""),
        OpaqueFunction(function=_opaque),
    ])
