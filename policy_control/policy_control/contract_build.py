"""Build a deploy contract from a training run dump — no numbers typed by hand.

Two task families are known: the manager-based left gripper track
(``open-grip_l_grasp_sensor*``) and the DirectRL ``grasp_s2r`` track (right DG-5F).
Every value is read through the readers the live nodes already trust
(``segments_from_run``, ``cfg_from_run``…) or from the training source constants;
the tests in ``tests/policy_control/test_pc_contract.py`` compare the two.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from . import _paths  # noqa: F401  (scripts/, hdgp openarm on sys.path)
from .contract import (SCHEMA, ActionCfg, ActionGroup, DeployContract, FabricCfg, Gains,
                       GravityCfg, HandCfg, ObsCfg, ObsSegment, PalmCfg, PdCfg, PolicyCfg,
                       RateCfg, RunInfo, file_md5, file_sha1, validate)

SIM2REAL = _paths.SIM2REAL
#: the asset both deployed checkpoints were trained on (sim joint limits = soft limits, factor 1.0)
ASSET_URDF = _paths.RL_WS / "urdf/generated/rl/openarm_tesollo_sensor_rl.urdf"
LEFT_ARM = [f"l_aj_{i}" for i in range(1, 8)]
RIGHT_ARM = [f"r_aj_{i}" for i in range(1, 8)]


# ------------------------------------------------------------------ small dump readers
def _text(path: Path) -> str:
    return Path(path).read_text()


def _scalar(text: str, key: str, default=None):
    m = re.search(rf"^\s*{key}:\s*(-?[0-9.eE+]+|null|true|false|[A-Za-z_][\w./-]*)\s*$", text, re.M)
    if m is None or m.group(1) == "null":
        return default
    v = m.group(1)
    if v in ("true", "false"):
        return v == "true"
    try:
        return float(v)
    except ValueError:
        return v


def _triple(text: str, key: str, default=None):
    m = re.search(rf"^\s*{key}:.*?\n((?:\s*- -?[\d.eE+-]+\n){{3}})", text, re.M)
    if m is None:
        return default
    return [float(v) for v in re.findall(r"-?[\d.eE+-]+", m.group(1).replace("- ", " "))]


def _block(text: str, header_re: str) -> str:
    """The YAML block whose header line matches header_re (indent taken from the match).

    The block runs until the next non-blank line indented at or above the header.
    """
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(header_re, ln)), None)
    if start is None:
        return ""
    indent = len(lines[start]) - len(lines[start].lstrip(" "))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            continue
        ind = len(ln) - len(ln.lstrip(" "))
        # YAML puts list items of a key at the key's own indent — they belong to the block.
        if ind < indent or (ind == indent and not ln.lstrip().startswith("- ")):
            end = j
            break
    return "\n".join(lines[start:end])


def _robot_block(env_text: str) -> str:
    """The robot articulation block: manager-based `scene.robot` or DirectRL `robot_cfg`."""
    for header in (r"^  robot:\s*$", r"^robot_cfg:\s*$"):
        blk = _block(env_text, header)
        if blk:
            return blk
    raise SystemExit("[contract] neither scene.robot nor robot_cfg found in env.yaml")


def _home_values(env_text: str, names: list[str]) -> list[float]:
    """init_state.joint_pos of the robot block (scoped so actuator gain lines cannot match)."""
    return _joint_values(_block(_robot_block(env_text), r"^\s*joint_pos:\s*$"), names)


def _joint_values(text: str, names: list[str]) -> list[float]:
    out = []
    for n in names:
        m = re.search(rf"^\s*{n}:\s*(-?[0-9.eE+]+)\s*$", text, re.M)
        if m is None:
            raise SystemExit(f"[contract] dump has no value for joint {n}")
        out.append(float(m.group(1)))
    return out


def _actuator_gains(env_text: str, joints: list[str]) -> Gains:
    """kp/kd per arm joint from the robot's actuators (per-joint blocks or regex groups)."""
    actuators = _block(_robot_block(env_text), r"^\s*actuators:\s*$")
    if not actuators:
        raise SystemExit("[contract] robot actuators block not found")
    head = actuators.split("\n", 1)[0]
    child = len(head) - len(head.lstrip(" ")) + 2
    groups = [_block(actuators, rf"^ {{{child}}}{name}:\s*$")
              for name in re.findall(rf"^ {{{child}}}([A-Za-z_][\w]*):\s*$", actuators, re.M)]
    kp, kd = [], []
    for j in joints:
        found = None
        for blk in groups:
            exprs = re.findall(r"^\s*- ([^\n]+)$", _block(blk, r"^\s*joint_names_expr:"), re.M)
            if any(re.fullmatch(e.strip(), j) for e in exprs):
                found = blk
                break
        if found is None:
            raise SystemExit(f"[contract] no actuator block matches {j}")
        kp.append(_gain_for(found, "stiffness", j))
        kd.append(_gain_for(found, "damping", j))
    return Gains(joints=list(joints), kp=kp, kd=kd)


def _gain_for(block: str, key: str, joint: str) -> float:
    scalar = re.search(rf"^\s*{key}:\s*(-?[0-9.eE+]+)\s*$", block, re.M)
    if scalar:
        return float(scalar.group(1))
    sub = _block(block, rf"^\s*{key}:\s*$")
    for m in re.finditer(r"^\s*([\w\[\]\-]+):\s*(-?[0-9.eE+]+)\s*$", sub, re.M):
        if re.fullmatch(m.group(1), joint):
            return float(m.group(2))
    raise SystemExit(f"[contract] {key} for {joint} not found in actuator block")


def _urdf_joint_limits(urdf: Path, joints: list[str]) -> list[list[float]]:
    """[lower, upper] per joint from the asset URDF (sim soft limits, soft_joint_pos_limit_factor 1.0)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(urdf)).getroot()
    limits = {j.get("name"): j.find("limit") for j in root.iter("joint")}
    out = []
    for name in joints:
        lim = limits.get(name)
        if lim is None or lim.get("lower") is None:
            raise SystemExit(f"[contract] {urdf.name} has no limit for joint {name}")
        out.append([float(lim.get("lower")), float(lim.get("upper"))])
    return out


def _robot_gravity_disabled(env_text: str) -> bool:
    robot = _robot_block(env_text)
    m = re.search(r"^\s*disable_gravity:\s*(true|false)", robot, re.M)
    if m is None:
        raise SystemExit("[contract] scene.robot rigid_props.disable_gravity not found")
    return m.group(1) == "true"


def _gravity_mode(env_text: str) -> str:
    """**학습 팔이 중력을 느꼈는가**로 실기 모드를 정한다. 규칙은 하나다.

    정책이 배운 조건을 실기가 재현해야 한다. 팔에 실린 중력이 sim 에서 어떻게 처리됐는지
    두 축(로봇 중력 스위치 · 중력보상 배율)을 함께 읽는다.

    | sim 중력 | sim 보상 | 실기 모드        | 왜 |
    |---------|---------|-----------------|---|
    | OFF     | (없음)   | `model_tau_ff`  | 정책이 무중력 팔에서 배웠다 — 실기가 상쇄해야 재현 |
    | ON      | > 0     | `model_tau_ff`  | sim 도 τ_ff 를 얹었다 — 같은 자리에 같은 항 |
    | ON      | 0       | `off`           | 정책이 처짐을 그대로 맞으며 배웠다 — 넣으면 두 번 지운다 |

    ★2026-09-06 확정은 세 번째 줄이 아니라 **두 번째 줄**이다. sim 중력을 켜되 팔에는
      보상을 얹는다. 보상이 없으면 홈 자세를 유지하는 것만으로 손이 218mm 낙하해
      테이블 아래 54.5mm 로 들어간다(실측). 실기도 같은 12.76° 를 처지므로 이건
      sim 만의 문제가 아니라 하드웨어 문제다.
    """
    if _robot_gravity_disabled(env_text):
        return "model_tau_ff"
    comp = _scalar(env_text, "gravity_compensation", 0.0)
    return "model_tau_ff" if float(comp) > 0.0 else "off"


def _agent(agent_yaml: Path) -> dict:
    text = _text(agent_yaml)
    try:
        params = yaml.safe_load(text)["params"]
    except Exception as exc:  # noqa: BLE001 — dump may carry python tags
        raise SystemExit(f"[contract] cannot parse {agent_yaml}: {exc}") from exc
    net, cfg = params["network"], params["config"]
    rnn = None
    if "rnn" in net:
        r = net["rnn"]
        rnn = {"type": r.get("name", "lstm"), "units": int(r["units"]), "layers": int(r.get("layers", 1)),
               "before_mlp": bool(r.get("before_mlp", False)), "layer_norm": bool(r.get("layer_norm", False)),
               "concat_input": bool(r.get("concat_input", False)),
               "concat_output": bool(r.get("concat_output", False))}
    env = params.get("env", {}) or {}
    clip_obs = env.get("clip_observations")
    return {
        "task": str(cfg["name"]),
        "experiment": str(cfg.get("full_experiment_name", cfg["name"])),
        "rnn": rnn,
        "mlp_units": [int(u) for u in net["mlp"]["units"]],
        "normalize_input": bool(cfg.get("normalize_input", True)),
        "action_clip": None if env.get("clip_actions") is None else float(env["clip_actions"]),
        "obs_clip": None if clip_obs is None else float(clip_obs),
    }


# ------------------------------------------------------------------ family detection
def detect_family(env_yaml: Path) -> str:
    text = _text(env_yaml)
    if "FabricPalmAction" in text and "joint_pos_rel" in text:
        return "gripper_left"
    if re.search(r"^fabrics_dt:", text, re.M) and re.search(r"^palm_anchor_mode:", text, re.M):
        return "grasp_s2r"
    raise SystemExit(f"[contract] cannot tell the task family of {env_yaml}")


# ------------------------------------------------------------------ public entry
def build_contract(run_dir: Path, checkpoint: Path | None = None, grasp_band: str | None = None,
                   asset: str | None = None) -> DeployContract:
    """grasp_band (gripper_left only): 'v1' | 'v2' | 'lo,hi' (table-height m). Required for that family —
    the run dump does not serialise the band and the task name alone picked the wrong one for v2B25
    (golden stream proves the gate opened at 62 mm = v1 band).

    asset: re-base the contract onto an ``hdgp/assets/robot`` asset (``contract_assets.bind_asset``);
    default keeps the run's own (training-time) fabric URDF."""
    run = Path(run_dir)
    env_yaml, agent_yaml = run / "params/env.yaml", run / "params/agent.yaml"
    family = detect_family(env_yaml)
    if family == "gripper_left":
        body = _build_left(_text(env_yaml), agent_yaml, grasp_band)
    else:
        body = _build_right(_text(env_yaml), agent_yaml)
    ckpt = _pick_checkpoint(run, checkpoint)
    _check_checkpoint_dims(ckpt, body["policy"].obs_dim, body["policy"].action_dim)
    try:
        rel_dir = str(run.resolve().relative_to(SIM2REAL))
    except ValueError:
        rel_dir = str(run.resolve())
    info = RunInfo(dir=rel_dir, task=body["task"], experiment=body["experiment"],
                   checkpoint=str(ckpt.relative_to(run)) if ckpt else "",
                   checkpoint_md5=file_md5(ckpt) if ckpt else "",
                   env_yaml_sha1=file_sha1(env_yaml), agent_yaml_sha1=file_sha1(agent_yaml))
    c = DeployContract(schema=SCHEMA, run=info, rate=body["rate"], policy=body["policy"],
                       obs=body["obs"], action=body["action"], fabric=body["fabric"], pd=body["pd"])
    # normalise through the JSON representation so a loaded file compares equal (derives `sides`)
    from .contract import from_dict, to_dict
    c = validate(from_dict(to_dict(c)))
    if asset is None:
        return c
    from .contract_assets import bind_asset
    return bind_asset(c, asset)


def _pick_checkpoint(run: Path, checkpoint: Path | None) -> Path | None:
    if checkpoint is not None:
        return Path(checkpoint)
    nn = run / "nn"
    if not nn.exists():
        return None
    files = sorted(nn.glob("*.pth"))
    if len(files) != 1:
        raise SystemExit(f"[contract] {nn} must hold exactly one .pth (found {len(files)}); pass --checkpoint")
    return files[0]


def _check_checkpoint_dims(ckpt: Path | None, obs_dim: int, action_dim: int) -> None:
    if ckpt is None:
        return
    from robot_profile import checkpoint_contract

    got_obs, got_act = checkpoint_contract(ckpt)
    if (got_obs, got_act) != (obs_dim, action_dim):
        raise SystemExit(f"[contract] checkpoint {ckpt.name} was trained with obs {got_obs}/act {got_act}, "
                         f"the run dump says {obs_dim}/{action_dim}")


# ------------------------------------------------------------------ rates shared
def _rate(env_text: str) -> RateCfg:
    dec = int(_scalar(env_text, "decimation"))
    dt = float(re.search(r"^  dt: ([\d.eE+-]+)", env_text, re.M).group(1))
    step_dt = dec * dt
    episode_s = float(_scalar(env_text, "episode_length_s"))
    return RateCfg(policy_hz=1.0 / step_dt, step_dt=step_dt, episode_steps=int(round(episode_s / step_dt)))


# ------------------------------------------------------------------ left gripper (manager-based)
def _grasp_band_axis(spec: str | None) -> tuple[list[float], str]:
    """Gate band in cup-origin axis coordinates + where it came from."""
    from left_grasp_gate import CUP_BOTTOM_TO_ORIGIN
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as P1

    if spec is None:
        raise SystemExit("[contract] gripper_left runs need --grasp-band v1|v2|lo,hi — the dump does not "
                         "serialise it and the task name picked the wrong band for v2B25")
    if spec == "v1":
        band, source = P1.GRASP_HEIGHT_BAND, "grasp_left_preset.GRASP_HEIGHT_BAND (v1)"
    elif spec == "v2":
        from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P2
        band, source = P2.GRASP_HEIGHT_BAND, "v2_preset.GRASP_HEIGHT_BAND (v2, 09.03+)"
    else:
        parts = [float(v) for v in spec.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"[contract] --grasp-band must be v1|v2|lo,hi, got {spec!r}")
        band, source = tuple(parts), f"explicit {spec}"
    return [float(band[0] - CUP_BOTTOM_TO_ORIGIN), float(band[1] - CUP_BOTTOM_TO_ORIGIN)], source


def _build_left(env_text: str, agent_yaml: Path, grasp_band: str | None) -> dict:
    from gripper_left_palm_command import cfg_from_run as palm_cfg_from_run
    from left_obs_builder import segments_from_run
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

    env_yaml = agent_yaml.parent / "env.yaml"
    segments = segments_from_run(env_yaml)              # SystemExit on an unknown term
    agent = _agent(agent_yaml)
    home = _home_values(env_text, LEFT_ARM)
    goal3 = _goal_center(env_text)
    palm = palm_cfg_from_run(env_yaml)
    grip = _block(env_text, r"^  gripper_action:")
    open_pos = float(_scalar(grip, "l_hj_gripper_1"))
    close_pos = float(re.findall(r"^\s*l_hj_gripper_1:\s*(-?[0-9.eE+]+)", grip, re.M)[1])
    palm_box = [list(P.PALM_BOX_X), list(P.PALM_BOX_Y), list(P.PALM_BOX_Z)]
    band_axis, band_source = _grasp_band_axis(grasp_band)
    gate = {"band_axis": band_axis, "band_source": band_source,
            "pad_offset": float(_scalar(grip, "pad_offset")),
            "lateral_ok": float(_scalar(grip, "lateral_ok")),
            "along_ok": float(_scalar(grip, "along_ok")),
            "release_lateral": float(_scalar(grip, "release_lateral"))}
    seg_params = {
        "joint_pos": ("joint_pos_rel", {"default": home + [open_pos, open_pos]}),
        "joint_vel": ("joint_vel_rel", {"default": [0.0] * 9}),
        "object_position": ("object_pos_root", {}),
        "target_object_position": ("goal_pose", {"goal": goal3 + [1.0, 0.0, 0.0, 0.0]}),
        "actions": ("last_action", {}),
        "gripper_gate": ("gripper_gate", gate),
        "tcp_pos": ("tcp_pos_normalized", {"palm_box": palm_box}),
        "palm_rot": ("rot6d_rows", {"body": P.GRIPPER_BASE_BODY}),
        "goal_minus_cup": ("goal_minus_object", {}),
        "cup_upright": ("object_upright", {}),
    }
    segs = tuple(ObsSegment(name=n, dim=d, builder=seg_params[n][0], params=seg_params[n][1])
                 for n, d in segments)
    obs = ObsCfg(joint_orders={"arm": LEFT_ARM, "ee": list(P.GRIPPER_JOINT_NAMES)},
                 fk={"kind": "left_gripper", "urdf": "urdf/generated/rl/openarm_tesollo_sensor_rl.urdf"},
                 segments=segs)
    action = ActionCfg(
        groups=(ActionGroup("palm", [0, 6]), ActionGroup("gripper", [6, 7])),
        palm=PalmCfg(convention="absolute_palm", box_lo=list(palm.box_lo), box_hi=list(palm.box_hi),
                     pos_rate_limit=palm.pos_rate_limit, euler_center=list(palm.euler_center),
                     max_pose_angle=float(palm.max_pose_angle), rot_rate_limit=palm.rot_rate_limit),
        hand=HandCfg(decoder="binary_gripper", joints=[P.GRIPPER_DRIVE_JOINT],
                     params={"open": open_pos, "close": close_pos, "close_when": "a<0",
                             "force_open_when_gate_closed": True}))
    rate = _rate(env_text)
    # ★fabric 의 default_config/리셋 q 는 로봇 리셋 자세(dump init_state = v2 LEFT_ARM_HOME_LOW)가 아니라
    #   fabric 액션이 쓰는 `grasp_left_preset.LEFT_ARM_HOME_JOINT_POS`(J147) 다
    #   (grasp_left_fabric_action.py:113-119, :166, :404). 골든 스트림 arm_target[0] 이 그 증거.
    fabric_home = [float(P.LEFT_ARM_HOME_JOINT_POS[j]) for j in LEFT_ARM]
    fabric = FabricCfg(class_name="OpenArmGripperLeftPoseFabric", robot_dir=P.FABRIC_ROBOT_DIR,
                       params=P.FABRIC_PARAMS_FILENAME, world={"filename": P.FABRIC_WORLD_FILENAME},
                       dt=rate.step_dt, decimation=int(P.FABRIC_DECIMATION), damping=float(P.FABRIC_DAMPING_GAIN),
                       max_objects=8, joint_order=LEFT_ARM, home_q=fabric_home,
                       vel_ff_scale=float(P.FABRIC_VEL_FF_SCALE),
                       home_source="grasp_left_preset.LEFT_ARM_HOME_JOINT_POS (fabric default_config; robot reset = dump init_state)")
    limit = [_regex_lookup(P.ARM_IK_MAX_TRACKING_ERROR, j) for j in LEFT_ARM]
    pd = PdCfg(groups=["left_arm", "left_gripper"], home_arm=home, home_hand={P.GRIPPER_DRIVE_JOINT: open_pos},
               # ★좌 v2B25 는 `_gravity_mode` 규칙의 **명시 예외**다. sim 중력이 켜져 있으니
               #   규칙대로면 `off` 인데, 이 트랙은 "실기가 sim 보다 더 처진다"(홈에서 6.18°)를
               #   적분 droop 으로 닫도록 학습·배포됐다. 그 6.18° 는 09.02 유령질량 수정
               #   **이전** 측정이라 지금은 훨씬 작을 가능성이 크다 — 좌팔 재학습 때 다시 재고,
               #   0 에 가까우면 이 줄도 `_gravity_mode(env_text)` 로 넘긴다.
               gravity=GravityCfg(mode="integral_droop", sim_gravity_disabled=_robot_gravity_disabled(env_text),
                                  gain=float(P.GRAVITY_COMP_GAIN), limit=limit),
               sim_gains=_actuator_gains(env_text, LEFT_ARM))
    policy = PolicyCfg(obs_dim=sum(d for _, d in segments), action_dim=7, rnn=agent["rnn"],
                       mlp_units=agent["mlp_units"], normalize_input=agent["normalize_input"],
                       action_clip=agent["action_clip"], obs_clip=agent["obs_clip"])
    return {"task": agent["task"], "experiment": agent["experiment"], "rate": rate, "policy": policy,
            "obs": obs, "action": action, "fabric": fabric, "pd": pd}


def _regex_lookup(table: dict, joint: str) -> float:
    for pat, val in table.items():
        if re.fullmatch(pat, joint):
            return float(val)
    raise SystemExit(f"[contract] no entry matches {joint} in {list(table)}")


def _goal_center(env_text: str) -> list[float]:
    blk = _block(env_text, r"^  object_pose:")
    out = []
    for ax in ("pos_x", "pos_y", "pos_z"):
        m = re.search(rf"^\s*{ax}:.*?\n\s*- (-?[\d.eE+-]+)\n\s*- (-?[\d.eE+-]+)", blk, re.M)
        if m is None:
            raise SystemExit(f"[contract] commands.object_pose.ranges.{ax} not found")
        out.append(0.5 * (float(m.group(1)) + float(m.group(2))))
    return out


# ------------------------------------------------------------------ right DG-5F (grasp_s2r DirectRL)
def _build_right(env_text: str, agent_yaml: Path) -> dict:
    import grasp_s2r_fabric as F
    from grasp_s2r_core import _norm_from_run
    from grasp_s2r_obs_builder import SEGMENTS, hand_dof_order, tip_body_order
    from grasp_s2r_palm_command import (HOME_PALM, PALM_BOX_MAX, PALM_BOX_MIN, PALM_ROT_CENTER_DEG,
                                        PALM_ROT_HALF_DEG)
    from grasp_s2r_palm_command import cfg_from_run as palm_cfg_from_run
    from grasp_s2r_synergy import HAND_JOINT_NAMES, _pose_tables
    from grasp_s2r_synergy import cfg_from_run as syn_cfg_from_run

    env_yaml = agent_yaml.parent / "env.yaml"
    agent = _agent(agent_yaml)
    hand_profile = list(HAND_JOINT_NAMES)
    home_arm = _home_values(env_text, RIGHT_ARM)
    home_hand = dict(zip(hand_profile, _home_values(env_text, hand_profile)))
    norm = _norm_from_run(env_yaml)
    pc = palm_cfg_from_run(env_yaml)
    sc = syn_cfg_from_run(env_yaml)
    open_pose, grip_pose = _pose_tables(sc)
    goal_offset = _triple(env_text, "goal_offset_xyz", [0.0, 0.0, 0.12])
    seg_params = {
        "arm_q": ("joint_pos_abs", {"order": "arm"}),
        "arm_qd": ("joint_vel_abs", {"order": "arm"}),
        "hand_q": ("joint_pos_abs", {"order": "hand_obs"}),
        "hand_qd": ("joint_vel_abs", {"order": "hand_obs"}),
        "palm_pos": ("body_pos", {"body": "palm"}),
        "palm_ax": ("rot6d_columns", {"body": "palm"}),
        "tips_rel_palm": ("tips_rel_palm", {}),
        "palm_to_obj": ("palm_to_object", {}),
        "obj_to_tips": ("object_to_tips", {}),
        "tip_force": ("tip_force_local", {"contact_force_max": norm["contact_force_max"]}),
        "joint_err": ("joint_err_norm", {"joint_pos_err_max": norm["joint_pos_err_max"], "order": "hand_profile",
                                         "target_source": "decoder_target"}),
        "actions": ("last_action", {}),
        "goal_rel": ("goal_minus_object", {"goal_offset": goal_offset}),
    }
    segs = tuple(ObsSegment(name=n, dim=d, builder=seg_params[n][0], params=seg_params[n][1]) for n, d in SEGMENTS)
    obs = ObsCfg(joint_orders={"arm": RIGHT_ARM, "hand_obs": hand_dof_order("r"), "hand_profile": hand_profile,
                               "tips": tip_body_order("r")},
                 fk={"kind": "fabric"}, segments=segs)
    action = ActionCfg(
        groups=(ActionGroup("palm", [0, 6]), ActionGroup("hand", [6, 21])),
        palm=PalmCfg(convention="delta_anchor", box_lo=list(PALM_BOX_MIN), box_hi=list(PALM_BOX_MAX),
                     pos_rate_limit=float(pc.rate_limit_m), delta_xyz=list(pc.delta_xyz),
                     delta_rot_deg=float(pc.delta_rot_deg),
                     anchor={"mode": pc.anchor_mode, "offset_xyz": list(pc.anchor_offset_xyz),
                             "fab_to_env": [0.0, 0.0, 0.0]},
                     rot_center_deg=list(PALM_ROT_CENTER_DEG), rot_half_deg=float(PALM_ROT_HALF_DEG),
                     home_palm=list(HOME_PALM), rot_rate_limit_deg=float(pc.rate_limit_rot_deg)),
        hand=HandCfg(decoder="synergy", joints=hand_profile, params={
            "close_speed": sc.close_speed, "couple_four_fingers": sc.couple_four_fingers,
            "residual_scale": sc.residual_scale, "hand_layout": sc.hand_layout,
            "oppose_grip_delta_rad": sc.oppose_grip_delta_rad, "weak_finger": sc.weak_finger,
            "weak_finger_curl_scale": sc.weak_finger_curl_scale, "freeze_scope": sc.freeze_scope,
            "release_deadband": sc.release_deadband, "blocked_err_thr_rad": sc.blocked_err_thr_rad,
            "blocked_limit_eps_rad": sc.blocked_limit_eps_rad,
            "hold_mode": str(_scalar(env_text, "synergy_hold_mode", "contact")),
            "contact_freeze": bool(_scalar(env_text, "synergy_contact_freeze", True)),
            "open_pose": [float(v) for v in open_pose], "grip_pose": [float(v) for v in grip_pose],
            "soft_limits": _urdf_joint_limits(ASSET_URDF, hand_profile),
            "close_gate": {"enabled": bool(norm["close_gate_enabled"]), "ramp": norm["close_gate_ramp"],
                           "z_deadband": norm["grasp_z_deadband"]}}))
    rate = _rate(env_text)
    fabric = FabricCfg(
        class_name=F.FABRIC_CLASS, robot_dir=F.FABRIC_ROBOT_DIR, params=F.FABRIC_PARAMS,
        world={"table_obstacle": bool(_scalar(env_text, "fabric_table_obstacle", False)),
               "margin_xy": float(_scalar(env_text, "fabric_table_margin_xy", 0.0)),
               "thickness": float(_scalar(env_text, "fabric_table_thickness", 0.0))},
        dt=float(_scalar(env_text, "fabrics_dt")), decimation=int(_scalar(env_text, "fabric_decimation")),
        damping=float(_scalar(env_text, "fabrics_damping_gain")),
        max_objects=int(_scalar(env_text, "fabrics_max_objects_per_env", 8)),
        joint_order=RIGHT_ARM + hand_profile, home_q=home_arm + [home_hand[j] for j in hand_profile],
        vel_ff_scale=float(_scalar(env_text, "fabric_velocity_ff_scale", 1.0)),
        hand_vel_ff_scale=float(_scalar(env_text, "hand_velocity_ff_scale", 1.0)), hand_sync="syn_target",
        table_z=float(_scalar(env_text, "table_surface_z")),
        use_hand_repulsion=bool(_scalar(env_text, "use_hand_repulsion", False)),
        use_body_repulsion_pairs=bool(_scalar(env_text, "use_body_repulsion_pairs", False)))
    pd = PdCfg(groups=["right_arm", "right_hand"], home_arm=home_arm, home_hand=home_hand,
               gravity=GravityCfg(mode=_gravity_mode(env_text),
                                  sim_gravity_disabled=_robot_gravity_disabled(env_text)),
               sim_gains=_actuator_gains(env_text, RIGHT_ARM))
    policy = PolicyCfg(obs_dim=sum(d for _, d in SEGMENTS), action_dim=21, rnn=agent["rnn"],
                       mlp_units=agent["mlp_units"], normalize_input=agent["normalize_input"],
                       action_clip=agent["action_clip"], obs_clip=agent["obs_clip"])
    return {"task": agent["task"], "experiment": agent["experiment"], "rate": rate, "policy": policy,
            "obs": obs, "action": action, "fabric": fabric, "pd": pd}
