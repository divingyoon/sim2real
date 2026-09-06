"""launch 2종(policy_chain / pd_controller) + tools/episode_ctl 의 순수 부분 — ROS 그래프 없음(unit)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
LAUNCH = SIM2REAL / "policy_control" / "launch"
TOOLS = SIM2REAL / "policy_control" / "tools"
CONTRACT = SIM2REAL / "logs" / "policy" / "left_v2B25" / "deploy_contract.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                  # dataclass 문자열 애너테이션 해석용
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def chain():
    pytest.importorskip("launch_ros")
    return _load(LAUNCH / "policy_chain.launch.py", "policy_chain_launch_t")


@pytest.fixture(scope="module")
def pd_launch():
    pytest.importorskip("launch_ros")
    return _load(LAUNCH / "pd_controller.launch.py", "pd_controller_launch_t")


@pytest.fixture
def cfg():
    return {"contract": str(CONTRACT), "robot": "left_gripper_fake", "device": "cuda:0",
            "fake": "false", "use_source": "false", "params_file": ""}


def test_generate_launch_description_types(chain, pd_launch):
    from launch import LaunchDescription

    assert isinstance(chain.generate_launch_description(), LaunchDescription)
    assert isinstance(pd_launch.generate_launch_description(), LaunchDescription)


def test_fake_mode_refuses_domain_zero(chain, pd_launch, cfg, monkeypatch):
    for env in ("", "0"):
        monkeypatch.setenv("ROS_DOMAIN_ID", env)
        with pytest.raises(RuntimeError, match="ROS_DOMAIN_ID"):
            chain.chain_nodes({**cfg, "fake": "true"})
        with pytest.raises(RuntimeError, match="ROS_DOMAIN_ID"):
            pd_launch.pd_nodes({**cfg, "fake": "true"})
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    assert len(chain.chain_nodes({**cfg, "fake": "true"})) == 3


def test_real_mode_allows_default_domain(chain, cfg, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "")
    assert len(chain.chain_nodes({**cfg, "fake": "false"})) == 3


def test_chain_and_pd_nodes_carry_params(chain, pd_launch, cfg, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    nodes = chain.chain_nodes(cfg)
    assert [n._Node__node_name for n in nodes] == list(chain.CHAIN_NODES)
    for n in nodes:                                          # console_script 모드 = ros2 run 식
        assert n._Node__package == "policy_control" and n._Node__node_executable == n._Node__node_name
    (pd,) = pd_launch.pd_nodes({**cfg, "execute": "true"})
    assert pd._Node__package == "policy_control" and pd._Node__node_executable == "pd_node"


def test_use_source_runs_venv_python_or_refuses_missing_source(chain, pd_launch, cfg, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    src = chain.PKG_ROOT / "policy_control" / "pd_node.py"
    if not src.is_file():                                    # M7 전: 소스가 없으면 기동 거부가 맞다
        with pytest.raises(RuntimeError, match="source"):
            pd_launch.pd_nodes({**cfg, "use_source": "true"})
        return
    (pd,) = pd_launch.pd_nodes({**cfg, "use_source": "true"})
    # launch_ros.Node 는 패키지 이름이 필수라 소스 모드는 ExecuteProcess(venv python) 로 띄운다
    from launch.actions import ExecuteProcess
    assert isinstance(pd, ExecuteProcess)
    cmd = [c if isinstance(c, str) else "".join(getattr(x, "text", "") for x in c) for c in pd.cmd]
    assert cmd[0] == chain.PY and cmd[1].endswith("pd_node.py") and "--ros-args" in cmd


def test_missing_contract_or_robot_is_refused(chain, cfg, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    with pytest.raises(RuntimeError, match="contract"):
        chain.chain_nodes({**cfg, "contract": "logs/policy/nope/deploy_contract.json"})
    with pytest.raises(RuntimeError, match="robot"):
        chain.chain_nodes({**cfg, "robot": "no_such_robot"})


def test_resolve_robot_name_vs_path(chain):
    assert chain.resolve_robot("left_gripper_fake") == chain.PKG_ROOT / "config" / "robots" / "left_gripper_fake.yaml"
    assert chain.resolve_robot("/abs/x.yaml") == Path("/abs/x.yaml")
    assert chain.resolve_robot("rel/x.yaml") == chain.RL_WS / "rel" / "x.yaml"


# ------------------------------------------------------------------ episode_ctl (pure parts)
@pytest.fixture(scope="module")
def ctl():
    return _load(TOOLS / "episode_ctl.py", "episode_ctl_t")


def test_episode_ctl_plan_only_without_execute(ctl, capsys):
    before = set(sys.modules)
    assert ctl.main(["--steps", "10"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "pd_engage" in out and "pd_release" in out
    assert "rclpy" not in (set(sys.modules) - before)        # 드라이런은 ROS 를 import 조차 안 한다


def test_episode_ctl_execute_requires_all_real_approvals(ctl, capsys):
    rc = ctl.main(["--steps", "10", "--execute", "--approve", "pd_engage"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "pd_goto_home" in err and "ep_start" in err
    assert ctl.main(["--steps", "10", "--approve", "bogus"]) == 2
    assert ctl.main(["--steps", "0"]) == 2


def test_episode_ctl_stage_order_and_helpers(ctl):
    ids = [s.id for s in ctl.STAGES]
    assert ids == ["pd_engage", "pd_goto_home", "ep_reset", "ep_start", "run", "ep_stop", "pd_release"]
    assert ctl.missing_approvals(frozenset()) == ["pd_engage", "pd_goto_home", "ep_start"]
    assert ctl.missing_approvals(frozenset(ids)) == []
    assert ctl.parse_trigger(True, '{"ok": true, "reasons": []}') == (True, [])
    assert ctl.parse_trigger(True, '{"ok": false, "reasons": ["estop"]}') == (False, ["estop"])
    assert ctl.parse_trigger(False, '{"ok": true, "reasons": []}')[0] is False
    assert ctl.parse_trigger(True, "not json")[0] is False
    engage = ctl.stage_by_id("pd_engage")
    assert ctl.phase_ok({"phase": "RAMPING"}, engage) and not ctl.phase_ok({"phase": "IDLE"}, engage)
    assert ctl.phase_ok(None, ctl.stage_by_id("run"))
    with pytest.raises(KeyError):
        ctl.stage_by_id("nope")


# ------------------------------------------------------------------ 09.06 양팔 DG-5F-M: fake_plant side/robot/contract + pd sides/pd_config
ASSET_CONTRACT = SIM2REAL / "logs" / "policy" / "asset_openarm_dg5f-m_bi_rl" / "deploy_contract.json"
needs_asset = pytest.mark.skipif(not ASSET_CONTRACT.exists(), reason="asset contract 없음")


@pytest.fixture(scope="module")
def plant():
    pytest.importorskip("launch")
    return _load(LAUNCH / "fake_plant.launch.py", "fake_plant_launch_t")


def _plant_cfg(**over) -> dict:
    base = {"side": "left", "robot": "", "contract": "", "pd_config": "", "inertia_q": "", "plant_hz": "100.0",
            "plant_friction": "1.0", "plant_model": "pd", "cup_x": "0.38", "cup_y": "0.19", "cup_z": "0.29209"}
    base.update(over)
    return base


def _cmd(proc) -> list[str]:
    return [c if isinstance(c, str) else "".join(getattr(x, "text", "") for x in c) for c in proc.cmd]


def _script(proc) -> str:
    return Path(_cmd(proc)[1]).stem


def _text(v) -> str:
    """launch 치환 튜플(yaml 덤프 문자열) / bool / str → 평문."""
    import yaml

    if isinstance(v, (bool, int, float)):
        return str(v).lower()
    text = v if isinstance(v, str) else "".join(getattr(x, "text", "") for x in v)
    loaded = yaml.safe_load(text) if text.strip() else ""
    return "" if loaded is None else str(loaded).lower() if isinstance(loaded, bool) else str(loaded)


def _params(node) -> dict:
    """launch_ros.Node 가 정규화한 첫 파라미터 dict → {str: str}."""
    return {_text(k): _text(v) for k, v in node._Node__parameters[0].items()}


def test_fake_plant_refuses_domain_zero_and_both_without_robot(plant, monkeypatch):
    for env in ("", "0"):
        monkeypatch.setenv("ROS_DOMAIN_ID", env)
        with pytest.raises(RuntimeError, match="ROS_DOMAIN_ID"):
            plant.plant_nodes(_plant_cfg())
    monkeypatch.setenv("ROS_DOMAIN_ID", "97")
    with pytest.raises(RuntimeError, match="robot"):
        plant.plant_nodes(_plant_cfg(side="both"))
    with pytest.raises(RuntimeError, match="contract"):
        plant.plant_nodes(_plant_cfg(side="left", robot="dg5f_m_left_fake"))


def test_fake_plant_legacy_sides_unchanged(plant, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "97")
    left = plant.plant_nodes(_plant_cfg(side="left"))
    right = plant.plant_nodes(_plant_cfg(side="right"))
    assert [_script(p) for p in left] == ["fake_arm_bridge", "fake_cup_pose_pub"]
    assert [_script(p) for p in right] == ["fake_arm_bridge", "fake_cup_pose_pub", "fake_hand_state_pub", "fake_tip_contact_pub"]
    bridge = _cmd(left[0])
    assert "--robot" in bridge and "gripper_left" in bridge and "--forward" in bridge and "--gravity" in bridge
    assert "--controller-node" in _cmd(right[2])


@needs_asset
def test_fake_plant_contract_mode_wires_each_side(plant, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "97")
    both = plant.plant_nodes(_plant_cfg(side="both", robot="dg5f_m_bi_fake", contract=str(ASSET_CONTRACT)))
    assert [_script(p) for p in both] == ["fake_arm_bridge", "fake_cup_pose_pub", "fake_hand_state_pub", "fake_tip_contact_pub",
                                          "fake_hand_state_pub", "fake_tip_contact_pub"]
    bridge = _cmd(both[0])
    assert bridge[bridge.index("--sides") + 1] == "right,left" and "--pd-config" in bridge and "--robot" not in bridge
    assert bridge[bridge.index("--robot-yaml") + 1].endswith("robots/dg5f_m_bi_fake.yaml")
    assert bridge[bridge.index("--pd-config") + 1].endswith("pd_dg5f_m_fake.yaml")
    hands = [_cmd(p) for p in both if _script(p) == "fake_hand_state_pub"]
    assert [h[h.index("--side") + 1] for h in hands] == ["right", "left"] and all("--controller-node" in h for h in hands)
    tips = [_cmd(p) for p in both if _script(p) == "fake_tip_contact_pub"]
    assert [t[t.index("--namespace") + 1] for t in tips] == ["dg5f_right", "dg5f_left"]
    left = plant.plant_nodes(_plant_cfg(side="left", robot="dg5f_m_left_fake", contract=str(ASSET_CONTRACT)))
    assert len(left) == 4 and _cmd(left[0])[_cmd(left[0]).index("--sides") + 1] == "left"


@needs_asset
def test_pd_launch_sides_and_pd_config_shorthand(pd_launch, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "97")
    cfg = {"contract": str(ASSET_CONTRACT), "robot": "dg5f_m_bi_fake", "fake": "true", "use_source": "false",
           "params_file": "", "pd_config": "dg5f_m_fake", "sides": "both", "execute": "true"}
    (pd,) = pd_launch.pd_nodes(cfg)
    params = _params(pd)
    assert params["sides"] == "right,left" and params["execute"].lower() == "true"
    assert params["pd_config"].endswith("config/pd_dg5f_m_fake.yaml")
    (pd,) = pd_launch.pd_nodes({**cfg, "sides": "left", "pd_config": "pd_dg5f_m"})
    params = _params(pd)
    assert params["sides"] == "left" and params["pd_config"].endswith("config/pd_dg5f_m.yaml")
    assert _params(pd_launch.pd_nodes({**cfg, "sides": ""})[0])["sides"] == ""
    with pytest.raises(RuntimeError, match="sides"):
        pd_launch.pd_nodes({**cfg, "sides": "up"})
    with pytest.raises(RuntimeError, match="pd_config"):
        pd_launch.pd_nodes({**cfg, "pd_config": "nope"})


# ------------------------------------------------------------------ 09.06 chain launch: side 인자 · 양팔 fabric 2개 · 제어 전용 계약
def test_chain_side_param_and_single_fabric_by_default(chain, cfg, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    nodes = chain.chain_nodes({**cfg, "side": "left"})
    assert [n._Node__node_name for n in nodes] == list(chain.CHAIN_NODES)
    obs, policy, fabric = nodes
    assert _params(obs)["side"] == "left" and _params(fabric)["side"] == "left"
    assert "side" not in _params(policy)
    with pytest.raises(RuntimeError, match="side"):
        chain.chain_nodes({**cfg, "side": "right"})                # 좌 계약에 없는 팔
    with pytest.raises(RuntimeError, match="both"):
        chain.chain_nodes({**cfg, "side": "both"})                 # 한 팔 계약


@needs_asset
def test_chain_control_only_contract_runs_fabric_only_and_both_sides(chain, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    base = {"contract": str(ASSET_CONTRACT), "robot": "dg5f_m_bi_fake", "device": "cuda:0", "fake": "true",
            "use_source": "false", "params_file": ""}
    master, fabric = chain.chain_nodes({**base, "side": "left"})
    # 정책이 없으면 obs 노드(에피소드 마스터)도 없다 → episode_master 가 같은 서비스/이벤트를 낸다
    assert master._Node__node_executable == "episode_master" and _params(master)["contract"] == str(ASSET_CONTRACT)
    assert fabric._Node__node_name == "fabric_node" and _params(fabric)["side"] == "left"
    both = chain.chain_nodes({**base, "side": "both"})
    assert [n._Node__node_name for n in both] == ["episode_master", "fabric_node_right", "fabric_node_left"]
    assert all(n._Node__node_executable == "fabric_node" for n in both[1:])
    assert [_params(n)["side"] for n in both[1:]] == ["right", "left"]
    src_master, src = chain.chain_nodes({**base, "side": "right", "use_source": "true"})
    cmd = [c if isinstance(c, str) else "".join(getattr(x, "text", "") for x in c) for c in src.cmd]
    assert cmd[1].endswith("fabric_node.py") and "__node:=fabric_node" in cmd and "side:=right" in cmd
    mcmd = [c if isinstance(c, str) else "".join(getattr(x, "text", "") for x in c) for c in src_master.cmd]
    assert mcmd[1].endswith("episode_master.py")
    two = chain.chain_nodes({**base, "side": "both", "use_source": "true"})
    names = ["".join(getattr(x, "text", "") for x in c) if not isinstance(c, str) else c for c in two[1].cmd]
    assert "__node:=fabric_node_right" in names
