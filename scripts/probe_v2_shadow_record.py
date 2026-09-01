"""[v2 이식판 · sim2real 소유] 좌 그리퍼 Fabrics 정책(grasp_sensor_v2)을 sim 에서 굴리며 **실기로 내보낼 것과 비교할 것**을 전부 남긴다.

이 파일이 하는 일은 기록뿐이다. env 도 액션 항도 건드리지 않고, 액션 항이 이미 갖고 있는
텐서를 읽기만 한다. 정책을 바꾸면 그림자 비교의 기준이 사라지기 때문이다.

무엇을 남기고 왜 남기는가 — "Fabrics IK 가 도는가"는 세 층으로 갈린다:

    L1  FK(fabric_q) vs 지령 palm pose   attractor 가 목표에 수렴하나   (여기서 나온다)
    L2  sim 물리 TCP  vs FK(fabric_q)     sim PD 가 fabric 해를 따라가나 (여기서 나온다)
    L3  실기 measured vs arm_target       실팔이 그 관절 목표를 따라가나 (재생 뒤 나온다)

★`fabric_q` 와 **중력 처짐 보상분(droop)** 을 따로 남긴다. 액션 항은 관절공간에서
  `target = fabric_q + droop` 을 지령하는데, droop 의 상한이 `effort/강성` 이고 그 강성이
  **sim 의 400** 이다. 실기 펌웨어는 70/60/10 이라 같은 보정량이 전혀 다른 뜻이 된다.
  합쳐서 남기면 실기 쪽에서 그 둘을 다시 가를 방법이 없다.

★리셋 오염 차단: `episode_length_s` 를 크게 잡는다. 이 태스크는 그 함정에 세 번 당했다
  (`probe_fab_action_mapping.py` docstring).

실행 (학습 시점 FABRICS 를 PYTHONPATH 로 지정하는 것이 중요하다 — 08.25 실측으로 트리가
다르면 IK 해가 최대 0.32 rad 갈리는 것을 확인했다):

    PYTHONPATH=<학습시점>/source/FABRICS/src \\
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_fab_shadow_record.py \\
        --checkpoint log/rl_games/open-grip/left/grasp-sensor-fab/fab_test16/nn/open-grip_l_grasp_sensor_fab.pth \\
        --steps 1500 --out logs/shadow/sim_fab_test16.npz
"""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="open-grip_l_grasp_sensor_v2-play")
parser.add_argument("--checkpoint", required=True, type=Path, help="rl_games .pth")
parser.add_argument("--steps", type=int, default=1500, help="기록할 env 스텝 수")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--gravity_comp", choices=["on", "off"], default="on",
                    help="액션 항의 처짐 보상. off 는 HDGP_GRAVITY_COMP=0 과 같다.")
parser.add_argument("--cup_pose", type=Path, default=None,
                    help="perception 이 준 컵 pose(json). 주면 리셋 직후 컵을 그 자리로 "
                         "옮긴다. env 는 건드리지 않는다 — 씬 객체에 직접 쓴다. "
                         "`sim2real/scripts/cup_pose_capture.py` 가 만든다.")
parser.add_argument("--keep_drop_termination", action="store_true",
                    help="컵 전도 종료를 살려 둔다. 기본은 끈다 — 리셋마다 팔이 홈으로 "
                         "텔레포트해 **연속 궤적이 아니게** 되고, 그림자 재생에서 그건 "
                         "이동이 아니라 도약이다. 실기에는 컵이 없으므로 컵 상태는 "
                         "이 측정의 대상이 아니다.")
parser.add_argument("--fabrics_src", type=Path, default=None,
                    help="쓸 fabrics_sim 소스 트리(.../source/FABRICS/src). "
                         "★PYTHONPATH 로는 안 된다 — `openarm.tasks` 가 저장소 사본을 "
                         "sys.path[0] 에 꽂아 덮어쓴다. 이 인자는 그보다 먼저 import 한다.")
parser.add_argument("--stream_udp", type=int, default=None,
                    help="포트를 주면 매 정책 스텝의 arm_target(7)+grip(1)을 UDP 로 "
                         "127.0.0.1:<포트> 에 쏜다(라이브 그림자용). Isaac 파이썬엔 "
                         "rclpy 가 없어 ROS 발행은 어댑터(udp_cmd_to_ros.py)가 맡는다. "
                         "★이 옵션은 루프를 **벽시계 50 Hz** 로 조절한다 — 없으면 sim 이 "
                         "실시간보다 빠르거나 느리게 달려 실기가 따라올 기준이 없다.")
parser.add_argument("--stream_rate_scale", type=float, default=0.25,
                    help="라이브 재생 속도 배율. 0.25 = sim 의 1/4 속도로 실시간 전개 — "
                         "관절 요구속도도 1/4 이 된다(v2H_wide peak 3.73→0.93 rad/s). "
                         "★sim 그대로(1.0)는 peak 가 실기 한계 2.0 을 넘는다(사용자 경고: "
                         "빠르면 망가질 수 있음). 경로는 불변, 시간만 늘어난다.")
parser.add_argument("--stream_meas", action="store_true",
                    help="지령(arm_target) 대신 **sim 팔 실측(arm_meas)** 을 스트림한다. "
                         "★08.31 실측 근거: 정책이 테이블을 피한 몸은 지령이 아니라 "
                         "지령−sim처짐(손목 j5 +5.7°/j7 −5.0°)이었다. 지령을 충실히 따른 "
                         "실팔이 오히려 긁었다 — 그림자의 목표는 sim 의 '화면 속 자세'다.")
parser.add_argument("--hold_open", action="store_true",
                    help="기록이 끝나도 창을 닫지 않는다(부팅 1분+ 절약, RViz 처럼 상주). "
                         "스트림은 멈추므로 실팔은 마지막 자세를 유지한다. 창을 직접 닫으면 종료.")
parser.add_argument("--from_rest", action="store_true",
                    help="정책 시작 전에 차렷(전관절 0)→preset 리셋 궤적(reset_both npz)을 "
                         "sim 에서 먼저 재생한다 — 실기 시나리오 전체를 눈으로 보는 용도. "
                         "★fabric/액션 항은 건드리지 않는다(env.step 없이 직접 구동). "
                         "끝 자세 == env 홈이라 인계 시 도약이 없다.")
parser.add_argument("--gui", action="store_true",
                    help="창을 띄워 **눈으로 보면서** 기록한다. 한 번 돌려 그림과 데이터를 같이 얻는다. "
                         "기본은 headless(빠름).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# ★기록 프로브는 원래 headless 전용이었다. 보면서 확인하려면 창이 필요한데, 그렇다고
#   별도 스크립트를 만들면 **보는 것과 재는 것이 다른 실행**이 되어 비교가 성립하지 않는다.
#   같은 실행에서 둘 다 나오게 한다.
args.headless = not args.gui
# ★두 방향 다 명시한다. `on` 일 때 아무것도 안 하면 preset 기본값
#   (`GRAVITY_COMP_ENABLED = False`)이 그대로 남는데, 기록에는 'on' 이라고 적히므로
#   **요청을 측정으로 오독**하게 된다. 실제로 그렇게 한 벌을 뽑았다.
os.environ["HDGP_GRAVITY_COMP"] = "0" if args.gravity_comp == "off" else "1"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym            # noqa: E402
import numpy as np                 # noqa: E402
import torch                       # noqa: E402

# ★08.31 v2 이식: t6x 스냅샷이 아니라 **live hdgp** 를 쓴다. v2H_wide 학습 소스
#   (vision-3090 dirty 포함)와 로컬 live 트리가 checksum 일치함을 실측했다
#   (grasp_sensor_v2 전체 + 상속 원본 grasp_sensor 전체, rsync -rcn).
#   옆 세션이 트리를 계속 고치므로, 기록 시각의 정합은 실행 중 params dump 로 재확인한다.
_HDGP = Path("/home/user/rl_ws/hdgp")
sys.path.insert(0, str(_HDGP / "source/openarm"))
sys.path.insert(0, str(_HDGP / "scripts/tools"))

# ★fabrics_sim 은 `openarm.tasks` **보다 먼저** 확정한다. 그 모듈이 저장소 사본을
#   sys.path[0] 에 꽂기 때문에, 나중에 import 하면 어떤 PYTHONPATH 를 줘도 저장소 것이
#   이긴다. 08.25 실측: 두 트리는 같은 목표에서 관절 해가 최대 0.32 rad 갈린다 —
#   조용히 다른 IK 로 기록하면 그림자 비교가 통째로 무의미해진다.
if args.fabrics_src is not None:
    resolved_src = args.fabrics_src.resolve()
    if not (resolved_src / "fabrics_sim").is_dir():
        raise SystemExit(f"--fabrics_src 아래에 fabrics_sim 이 없다: {resolved_src}")
    sys.path.insert(0, str(resolved_src))
import fabrics_sim                                               # noqa: E402
_FABRICS_FILE = Path(fabrics_sim.__file__).resolve()
if args.fabrics_src is not None and not str(_FABRICS_FILE).startswith(str(resolved_src)):
    raise SystemExit(
        f"fabrics_sim 이 요청한 트리에서 오지 않았다:\n  요청 {resolved_src}\n"
        f"  실제 {_FABRICS_FILE}"
    )

import openarm.tasks                                             # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                   # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz   # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P   # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available         # noqa: E402


def _dump_policy_input(player, obs) -> None:
    """정책에 들어가기 **직전** 값을 본다. obs 가 유한해도 정규화가 폭발할 수 있다.

    ★`running_var` 가 거의 0 인 채널(=학습 중 사실상 상수였던 obs)이 있으면
      `(x-mean)/sqrt(var+eps)` 가 그 채널을 수천 배로 키운다. 학습 분포 안에서는
      안 보이다가 재생에서 터진다.
    """
    model = player.model
    print(f"[probe] obs absmax={obs.abs().max():.4g} finite={bool(torch.isfinite(obs).all())}")
    try:
        z = model.norm_obs(obs)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[probe] norm_obs 실패: {type(exc).__name__}: {exc}")
        return
    print(f"[probe] 정규화 obs absmax={z.abs().max():.4g} "
          f"finite={bool(torch.isfinite(z).all())}")
    if not torch.isfinite(z).all():
        bad = torch.nonzero(~torch.isfinite(z).all(dim=0)).reshape(-1)
        print(f"[probe]   깨진 obs 인덱스: {bad.tolist()[:20]}")
    rms = getattr(model, "running_mean_std", None)
    if rms is not None and hasattr(rms, "running_var"):
        var = rms.running_var.reshape(-1)
        order = torch.argsort(var)[:6]
        print("[probe] 분산이 가장 작은 obs 채널 (상수에 가까움):")
        for i in order.tolist():
            print(f"[probe]   idx {i:3d}  var={float(var[i]):.3e}  "
                  f"증폭≈{1.0/float((var[i]+1e-5).sqrt()):.1f}배  "
                  f"정규화값 absmax={float(z[:, i].abs().max()):.4g}")


def nonfinite_obs_terms(env, group: str = "policy") -> list[str]:
    """유한하지 않은 obs 항의 이름·위치. **어디가 깨졌는지**를 말해야 고칠 수 있다.

    NaN 이 정책에 들어가면 mu/logstd 가 NaN 이 되고 `torch.normal` 이
    "std >= 0.0" 로 죽는다 — sigma 가 음수라는 뜻이 아니라 **NaN 이라는 뜻**이다
    (모델은 exp(logstd) 를 쓰므로 음수가 나올 수 없다). 실제로 이 메시지에 속았다.
    """
    buf = env.obs_buf[group] if isinstance(env.obs_buf, dict) else env.obs_buf
    mgr = env.observation_manager
    terms = list(mgr.active_terms[group])
    dims = [int(np.prod(d)) for d in mgr.group_obs_term_dim[group]]
    bad, start = [], 0
    for name, width in zip(terms, dims):
        chunk = buf[:, start:start + width]
        mask = ~torch.isfinite(chunk)
        if bool(mask.any()):
            envs = torch.nonzero(mask.any(dim=1)).reshape(-1).tolist()
            bad.append(f"{name}[{start}:{start+width}] env={envs[:6]}")
        start += width
    return bad


def obs_term_slice(env, name: str, group: str = "policy"):
    """이름으로 obs 항의 구간을 찾는다.

    ★위치로 찾으면 안 된다. `gripper_gate` 는 base cfg 가 붙이고 fab cfg 가 그 **뒤에**
      여러 항을 더한다(fabric_q/qd · palm_pose_target · palm_action_scale/anchor ...).
      `obs_buf["policy"][:, -1]` 이 게이트였던 것은 fab 항이 없던 시절 얘기다 —
      지금 그렇게 읽으면 조용히 다른 값을 게이트라고 기록한다.
    """
    mgr = env.observation_manager
    terms = list(mgr.active_terms[group])
    dims = [int(np.prod(d)) for d in mgr.group_obs_term_dim[group]]
    if name not in terms:
        return None
    i = terms.index(name)
    start = sum(dims[:i])
    return slice(start, start + dims[i])


def reset_rnn_states(player, done_mask) -> None:
    """순환 정책의 hidden state 를 끝난 env 에 대해 비운다.

    rl_games 의 `BasePlayer.run()` 은 이걸 하는데, 우리처럼 `get_action` 을 직접 부르는
    루프는 안 한다. 안 비우면 새 에피소드가 **이전 에피소드의 기억을 물고** 시작해서
    같은 상태에서 다른 액션이 나온다 — 재현도 비교도 안 된다.
    """
    if not getattr(player, "is_rnn", False):
        return
    states = getattr(player, "states", None)
    if not states:
        return
    idx = torch.nonzero(done_mask.reshape(-1), as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return
    for s in states:
        s[:, idx, :] = 0.0


def build_policy(checkpoint: Path, agent_cfg: dict, env):
    """rl_games player 를 만들고 체크포인트를 얹는다."""
    from rl_games.torch_runner import Runner
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    device = agent_cfg["params"]["config"]["device"]
    # ★★clip 값을 **런의 agent.yaml 에서** 읽는다. 하드코딩 금지.
    #   `clip_actions` 는 래퍼의 action_space 를 Box(-c, +c) 로 만들고, rl_games 의
    #   `get_action` 은 마지막에 `rescale_actions(actions_low, actions_high, ...)` 를 건다.
    #   c=inf 면 그 식이 `-inf + inf*x` 가 되어 **모든 액션이 NaN** 이 된다 — 그런데 그
    #   NaN 은 `torch.normal` 의 "std >= 0.0" 이나 "정책이 이상하다"로 나타나서 원인을
    #   완전히 다른 데서 찾게 만든다. 실제로 그 길로 갔다.
    #   fab_test42 는 clip_observations 5.0 / clip_actions 1.0 이다. play.py 와 같은 규약.
    env_cfg_rl = agent_cfg["params"].get("env", {}) or {}
    clip_obs = float(env_cfg_rl.get("clip_observations", math.inf))
    clip_actions = float(env_cfg_rl.get("clip_actions", math.inf))
    print(f"[probe] clip_observations={clip_obs} · clip_actions={clip_actions} (런 설정)")
    wrapped = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions, None, True)
    vecenv_name = "IsaacRlgWrapper"
    from rl_games.common import env_configurations, vecenv
    vecenv.register(vecenv_name, lambda name, num, **kw: RlGamesGpuEnv(name, num, **kw))
    env_configurations.register(vecenv_name, {
        "vecenv_type": vecenv_name, "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_name"] = vecenv_name
    agent_cfg["params"]["config"]["env_info"] = wrapped.get_env_info()

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    if hasattr(player, "has_batch_dimension"):
        player.has_batch_dimension = True
    # ★순환 정책은 hidden state 를 **env 수에 맞춰** 만들어야 한다.
    #   `BasePlayer.batch_size` 기본값이 1 이고 `init_rnn()` 이 그 값으로 states 를
    #   할당한다. rl_games 의 `run()` 은 시작할 때 이걸 채우지만 우리는 `get_action` 을
    #   직접 부르므로 아무도 안 채운다 → num_envs>1 에서
    #   "Expected hidden[0] size (1, N, H), got [1, 1, H]" 로 죽는다. 실제로 당했다.
    player.batch_size = env.num_envs
    player.reset()          # is_rnn 이면 여기서 init_rnn() 이 돈다

    # ★복원 자기점검. 체크포인트가 멀쩡해도 **모델에 올라가지 않았으면** 결과는 NaN 이고,
    #   그 NaN 은 정책 출력에서야 보인다. 거기서 보면 원인을 못 짚는다.
    bad = [n for n, t in player.model.state_dict().items()
           if t.is_floating_point() and not bool(torch.isfinite(t).all())]
    if bad:
        raise SystemExit(f"[probe] 복원 뒤 모델 텐서가 유한하지 않다: {bad[:8]}")
    print(f"[probe] 복원 OK — 파라미터 {len(player.model.state_dict())} 개 전부 유한, "
          f"is_rnn={getattr(player, 'is_rnn', False)}, "
          f"normalize_input={getattr(player, 'normalize_input', '?')}, "
          f"batch_size={player.batch_size}")
    st = getattr(player, "states", None)
    print(f"[probe] rnn states = "
          + (", ".join(str(tuple(x.shape)) for x in st) if st else "None"))
    return player, wrapped


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    import yaml
    run_dir = args.checkpoint.parent.parent
    agent_cfg = yaml.safe_load((run_dir / "params/agent.yaml").read_text())
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(args.checkpoint),
        workspace_root=str(_HDGP.parent),
    )
    agent_cfg["params"]["config"]["device"] = args.device
    agent_cfg["params"]["config"]["device_name"] = args.device
    # ★★복원 **뒤에** 다시 강제한다. `params/env.yaml` 은 학습의 num_envs(1024)를 담고
    #   있어서 `parse_env_cfg(num_envs=1)` 을 조용히 되돌린다. 순서를 바꾸면 1024 env 를
    #   돌리고도 눈치채지 못한다(367 MB npz 로 알아챘다).
    env_cfg.scene.num_envs = args.num_envs
    # ★리셋 오염 차단. 에피소드가 중간에 끊기면 지령·처짐 적분이 초기화돼 시계열이 갈린다.
    env_cfg.episode_length_s = 1e6
    if not args.keep_drop_termination and hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
        print("[REC] 컵 전도 종료를 껐다 — 팔 궤적을 끊기지 않게 한다", flush=True)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    player, wrapped = build_policy(args.checkpoint, agent_cfg, env)

    cup_pose = None
    if args.cup_pose is not None:
        sys.path.insert(0, str(_HDGP.parent / "sim2real/scripts"))
        from cup_pose_capture import load_capture, spawn_box_from_preset, verdict

        cup_pose = load_capture(args.cup_pose, expect_frame="base_link")
        report = verdict(cup_pose, spawn_box_from_preset(P))
        print("[CUP] " + report.describe().replace("\n", "\n[CUP] "), flush=True)

    robot = env.scene["robot"]
    # ★fabric 의 `palm_link` 은 그리퍼 **TCP** 인데(fabric URDF 주석) USD 에서는 그 프레임이
    #   강체로 병합돼 사라진다. base 바디를 그대로 쓰면 두 점이 구조적으로 80 mm 떨어져
    #   있어 L2 가 "추종오차 99 mm" 로 읽힌다 — 실제로는 오프셋이다. 회전시켜 더한다.
    tcp_offset = torch.tensor([0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z], device=env.device)
    term = env.action_manager.get_term("arm_action")
    arm_ids = term._arm_joint_ids
    grip_ids = [robot.joint_names.index(n) for n in P.GRIPPER_JOINT_NAMES]
    base_body = robot.body_names.index(P.GRIPPER_BASE_BODY)
    fabric = term._fabric

    # ★액션 항의 palm 지령 표현은 트랙이 살아 있는 동안 바뀐다(08.25 실측: 7D xyz+quat(xyzw)
    #   → 6D xyz+euler_zyx). 어느 쪽이든 **위치 + wxyz 쿼터니언**으로 정규화해 남긴다.
    #   둘 다 없으면 조용히 넘어가지 않고 무엇이 있는지 적어 죽는다 — 기록이 반쯤 비면
    #   그림자 비교는 실패가 아니라 **틀린 결론**으로 나타난다.
    def palm_command():
        if hasattr(term, "_palm_target_xyz_q"):          # 구 규약: xyz + xyzw
            raw = term._palm_target_xyz_q
            xyzw = raw[:, 3:7]
            return raw[:, :3], torch.cat([xyzw[:, 3:4], xyzw[:, 0:3]], dim=-1)
        if hasattr(term, "_palm_pose_target"):           # 신 규약: xyz + euler_zyx
            raw = term._palm_pose_target
            return raw[:, :3], quat_from_euler_xyz(raw[:, 5], raw[:, 4], raw[:, 3])
        raise AttributeError(
            "액션 항에서 palm 지령을 못 찾았다. 가진 것: "
            + ", ".join(sorted(a for a in vars(term) if "palm" in a or "target" in a))
        )

    palm_command()   # 기록 전에 한 번 불러 API 를 확정한다
    print(f"[REC] task={args.task} ckpt={args.checkpoint.name}", flush=True)
    print(f"[REC] gravity_comp={args.gravity_comp} (P.GRAVITY_COMP_ENABLED={P.GRAVITY_COMP_ENABLED})",
          flush=True)
    print(f"[REC] fabrics_sim -> {_FABRICS_FILE}", flush=True)
    print(f"[REC] 팔 관절 {[robot.joint_names[i] for i in arm_ids]}", flush=True)
    print(f"[REC] 그리퍼 {[robot.joint_names[i] for i in grip_ids]}", flush=True)

    rec: dict[str, list] = {k: [] for k in (
        "action", "palm_cmd_pos", "palm_cmd_quat_wxyz", "fabric_q", "droop", "arm_target",
        "arm_meas", "arm_vel", "palm_fk_pos", "palm_fk_quat_wxyz", "tcp_pos", "tcp_quat_wxyz",
        "grip_cmd", "grip_meas", "gripper_gate", "cup_pos", "cmd_step_norm", "reward", "done")}

    def policy_obs(raw):
        """rl_games 래퍼는 obs 를 dict 로 준다 — actor 가 먹는 텐서만 꺼낸다."""
        return raw["obs"] if isinstance(raw, dict) else raw

    def place_cup() -> None:
        """리셋 뒤 컵을 인지가 준 자리로 옮긴다.

        env 를 고치지 않는 이유 두 가지: ①이 태스크 파일은 자매 세션이 지금 고치고 있다
        ②스폰은 이벤트가 무작위로 하는데, 그걸 바꾸면 학습 경로를 건드리게 된다.
        씬 객체에 직접 쓰면 둘 다 피하면서 같은 결과를 얻는다.
        """
        if cup_pose is None:
            return
        cup = env.scene["object"]
        pose = torch.tensor(
            [*cup_pose.position, *cup_pose.orientation_wxyz],
            device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        pose[:, :3] += env.scene.env_origins          # 씬 좌표는 world 다
        cup.write_root_pose_to_sim(pose)
        cup.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))

    gate_slice = obs_term_slice(env, "gripper_gate")
    if gate_slice is None:
        print("[probe] ⚠ obs 에 gripper_gate 가 없다 — 게이트 채널은 비운다.")
    else:
        print(f"[probe] gripper_gate obs 구간 = {gate_slice.start}:{gate_slice.stop} "
              f"(전체 {sum(int(np.prod(d)) for d in env.observation_manager.group_obs_term_dim['policy'])})")
    is_rnn = bool(getattr(player, "is_rnn", False))
    print(f"[probe] 정책 순환 여부 = {is_rnn}"
          + ("  (에피소드 종료마다 hidden state 를 비운다)" if is_rnn else ""))

    obs = policy_obs(wrapped.reset())
    place_cup()

    _sock = _addr = None
    if args.stream_udp is not None:
        import socket as _socket
        import struct as _struct
        import time as _time
        _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        _addr = ("127.0.0.1", int(args.stream_udp))
        _scale = float(args.stream_rate_scale)
        if not (0.0 < _scale <= 1.0):
            raise SystemExit(f"--stream_rate_scale 은 (0,1] — 받은 값 {_scale}")
        _step_wall = float(env.step_dt) / _scale
        print(f"[STREAM] UDP → {_addr} · rate_scale {_scale} → 발행 "
              f"{1.0/_step_wall:.1f} Hz (sim {float(env.step_dt)*1000:.0f} ms/스텝을 "
              f"벽시계 {_step_wall*1000:.0f} ms 로) · 요구속도 = sim 의 {_scale}배", flush=True)

    if args.from_rest:
        # ── 차렷 → preset 리셋 궤적 재생(양팔) ─────────────────────────────────
        #   env.step 을 부르지 않고 sim 을 직접 굴린다 — 정책·fabric 상태는 홈 기준
        #   그대로이고, 궤적 끝 == 홈이라 인계 순간 상태가 일치한다.
        _rd = Path("/home/user/rl_ws/sim2real/logs/shadow/reset_both")
        _rl = np.load(_rd / "reset_left.npz", allow_pickle=True)
        _rr = np.load(_rd / "reset_right.npz", allow_pickle=True)
        _jn = robot.joint_names
        def _ids(names): return [_jn.index(str(n)) for n in names]
        _sets = [
            (_ids(_rl["meta_joint_names"]), _rl["arm_target"][:, 0, :]),
            (_ids(_rl["meta_grip_names"]),  _rl["grip_cmd"][:, 0, :]),
            (_ids(_rr["meta_joint_names"]), _rr["arm_target"][:, 0, :]),
            (_ids(_rr["meta_grip_names"]),  _rr["grip_cmd"][:, 0, :]),
        ]
        _T = _sets[0][1].shape[0]
        _spf = max(1, round(float(_rl["meta_step_dt"]) / env.physics_dt))
        print(f"[REST] 차렷→preset 재생 {_T} 프레임 × {_spf} 물리스텝 "
              f"({_T*float(_rl['meta_step_dt']):.1f}s)", flush=True)
        # 차렷: 전 관절 0 (실기 시작 규약)
        _zero = torch.zeros_like(robot.data.joint_pos)
        robot.write_joint_state_to_sim(_zero, torch.zeros_like(_zero))
        robot.set_joint_position_target(_zero)
        robot.write_data_to_sim()
        for _ in range(40 * _spf):                      # 출발 정착
            env.sim.step(render=args.gui)
        for _t in range(_T):
            for _idx, _traj in _sets:
                robot.set_joint_position_target(
                    torch.tensor(_traj[_t], device=env.device,
                                 dtype=torch.float32).unsqueeze(0), joint_ids=_idx)
            robot.write_data_to_sim()
            for _ in range(_spf):
                env.sim.step(render=args.gui)
        for _ in range(50 * _spf):                      # 도착 정착
            env.sim.step(render=args.gui)
        env.scene.update(env.sim.get_physics_dt())
        _all_arm = _sets[0][0]
        _err = (robot.data.joint_pos[0, _all_arm]
                - torch.tensor(_sets[0][1][-1], device=env.device)).abs().max()
        print(f"[REST] 재생 종료 — 좌팔 홈 잔차 {float(_err)*57.3:.2f}°", flush=True)
    bad0 = nonfinite_obs_terms(env)
    if bad0:
        raise SystemExit(f"[probe] 리셋 직후부터 obs 가 깨져 있다: {bad0}")

    with torch.inference_mode():
        for step in range(args.steps):
            if not bool(torch.isfinite(obs).all()):
                raise SystemExit(
                    f"[probe] step {step}: obs 에 유한하지 않은 값 — "
                    f"{nonfinite_obs_terms(env)}\n"
                    "  fabric 이 발산했을 가능성이 크다(metric 역행렬). "
                    "정책 NaN 은 결과이지 원인이 아니다."
                )
            if step == 0:
                _dump_policy_input(player, obs)
            action = player.get_action(obs, is_deterministic=True)
            if not bool(torch.isfinite(action).all()):
                raise SystemExit(f"[probe] step {step}: 정책이 유한하지 않은 액션을 냈다")
            raw_obs, reward, dones, _ = wrapped.step(action)
            reset_rnn_states(player, dones)
            if cup_pose is not None and bool(dones.any()):
                # 리셋이 나면 이벤트가 컵을 다시 무작위로 놓는다 — 되돌린다.
                place_cup()
            obs = policy_obs(raw_obs)

            palm = fabric.get_palm_pose(term._fabric_q, "quaternion")     # xyz + xyzw
            body_quat = robot.data.body_quat_w[:, base_body]              # wxyz
            body_pos = (robot.data.body_pos_w[:, base_body] - env.scene.env_origins
                        + quat_apply(body_quat, tcp_offset.expand(env.num_envs, 3)))
            gate = (env.obs_buf["policy"][:, gate_slice].reshape(env.num_envs)
                    if gate_slice is not None and isinstance(env.obs_buf, dict) else None)

            rec["action"].append(action.detach().cpu().numpy().copy())
            cmd_pos, cmd_quat = palm_command()
            rec["palm_cmd_pos"].append(cmd_pos.detach().cpu().numpy().copy())
            rec["palm_cmd_quat_wxyz"].append(cmd_quat.detach().cpu().numpy().copy())
            rec["fabric_q"].append(term._fabric_q.detach().cpu().numpy().copy())
            rec["droop"].append(term._droop.detach().cpu().numpy().copy())
            rec["arm_target"].append(
                (term._fabric_q + term._droop).detach().cpu().numpy().copy()
                if P.GRAVITY_COMP_ENABLED else term._fabric_q.detach().cpu().numpy().copy())
            rec["arm_meas"].append(robot.data.joint_pos[:, arm_ids].detach().cpu().numpy().copy())
            rec["arm_vel"].append(robot.data.joint_vel[:, arm_ids].detach().cpu().numpy().copy())
            fk_xyzw = palm[:, 3:7]
            rec["palm_fk_pos"].append(palm[:, :3].detach().cpu().numpy().copy())
            rec["palm_fk_quat_wxyz"].append(
                torch.cat([fk_xyzw[:, 3:4], fk_xyzw[:, 0:3]], dim=-1).detach().cpu().numpy().copy())
            rec["tcp_pos"].append(body_pos.detach().cpu().numpy().copy())
            rec["tcp_quat_wxyz"].append(body_quat.detach().cpu().numpy().copy())
            rec["grip_cmd"].append(
                robot.data.joint_pos_target[:, grip_ids].detach().cpu().numpy().copy())
            rec["grip_meas"].append(robot.data.joint_pos[:, grip_ids].detach().cpu().numpy().copy())
            rec["gripper_gate"].append(
                gate.detach().cpu().numpy().copy() if gate is not None
                else np.full(env.num_envs, np.nan))
            rec["cup_pos"].append(
                (env.scene["object"].data.root_pos_w - env.scene.env_origins)
                .detach().cpu().numpy().copy())
            rec["cmd_step_norm"].append(term.cmd_step_norm.detach().cpu().numpy().copy())
            rec["reward"].append(reward.detach().cpu().numpy().copy())

            if _sock is not None:
                import struct as _struct
                import time as _time
                # ★v2 패킷: 실행지령(8) + action(7) + sim지령(7) + sim실측(7).
                #   어댑터가 /shadow/* 로 발행해 real /joint_states 와 **한 bag** 에
                #   시간동기 기록된다(ACTION/SIM/REAL 최적화 프레임워크 — r2s 와 동형).
                if args.stream_meas:
                    _ex = rec["arm_meas"][-1][0]
                    _gx = float(rec["grip_meas"][-1][0][0])
                else:
                    _ex = rec["arm_target"][-1][0]
                    _gx = float(rec["grip_cmd"][-1][0][0])
                _payload = ([float(v) for v in _ex] + [_gx]
                            + [float(v) for v in rec["action"][-1][0]]
                            + [float(v) for v in rec["arm_target"][-1][0]]
                            + [float(v) for v in rec["arm_meas"][-1][0]])
                _sock.sendto(_struct.pack("<Id29f", 0x5A2B02, _time.time(), *_payload), _addr)
                # 벽시계 페이싱 — 다음 스텝 예정 시각까지 잔여만 잔다
                if step == 0:
                    _t_next = _time.monotonic() + _step_wall
                else:
                    _rem = _t_next - _time.monotonic()
                    if _rem > 0:
                        _time.sleep(_rem)
                    _t_next += _step_wall
            rec["done"].append(dones.detach().cpu().numpy().copy())

            if step % 200 == 0:
                err = float(np.linalg.norm(rec["palm_fk_pos"][-1][0] - rec["palm_cmd_pos"][-1][0]))
                print(f"[REC] {step:5d}/{args.steps}  L1 {err*1000:6.1f} mm  "
                      f"done {int(rec['done'][-1][0])}", flush=True)

    arrays = {k: np.stack(v) for k, v in rec.items()}
    # 팔 게인은 "액션이 얼마나 따라와지는가"의 절반이다 — 기록이 스스로 말해야 한다.
    _left_arm_group = next(
        (g for g, a in env.scene["robot"].actuators.items()
         if any(n.startswith("l_aj_") for n in a.joint_names)), None)
    arrays["meta_joint_names"] = np.array([robot.joint_names[i] for i in arm_ids])
    arrays["meta_grip_names"] = np.array([robot.joint_names[i] for i in grip_ids])
    arrays["meta_step_dt"] = np.array([env.step_dt])
    # 요청이 아니라 **실제로 켜졌는지**를 적는다.
    arrays["meta_gravity_comp"] = np.array(["on" if P.GRAVITY_COMP_ENABLED else "off"])
    arrays["meta_gravity_comp_requested"] = np.array([args.gravity_comp])
    if _left_arm_group is not None:
        _act = env.scene["robot"].actuators[_left_arm_group]
        arrays["meta_arm_group"] = np.array([_left_arm_group])
        arrays["meta_arm_stiffness"] = np.array([float(_act.stiffness.reshape(-1)[0])])
        arrays["meta_arm_damping"] = np.array([float(_act.damping.reshape(-1)[0])])
    else:
        print("[REC] ⚠ 좌팔 액추에이터 그룹을 못 찾았다 — 게인을 기록하지 못한다.")
    arrays["meta_checkpoint"] = np.array([str(args.checkpoint)])
    arrays["meta_fabrics"] = np.array([str(_FABRICS_FILE)])
    # ★소스 신원. 이 트랙은 살아 있는 동안 태스크 파일이 바뀐다(08.25 실측: 기록 도중
    #   다른 세션이 preset·액션 항을 갈아 recorder 가 죽었다). 어떤 코드로 잰 기록인지
    #   파일 자신이 답할 수 있어야 사후에 해석이 가능하다.
    import hashlib
    task_dir = _HDGP / "source/openarm/openarm/gripper/left/grasp_sensor"
    digests = [f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()[:12]}"
               for f in sorted(task_dir.glob("*.py"))]
    arrays["meta_task_sha256"] = np.array(digests)
    arrays["meta_cup_pose_source"] = np.array(
        [cup_pose.source if cup_pose is not None else "env 무작위 스폰"])
    # ★fabric 을 **다른 인터프리터에서 다시 풀어 보려면** 이 다섯이 있어야 한다. 지금은
    #   전부 preset 상수라 소스만 바뀌면 조용히 달라진다(08.25 에 damping 20→10,
    #   fabric_dt step_dt/2→step_dt 로 바뀌었다). 기록이 스스로 답하게 한다.
    arrays["meta_fabric_dt"] = np.array([float(term._fabric_dt)])
    arrays["meta_fabric_decimation"] = np.array([int(P.FABRIC_DECIMATION)])
    arrays["meta_fabric_damping"] = np.array([float(term._damping[0, 0].item())])
    arrays["meta_fabric_vel_ff"] = np.array(
        [float(getattr(term, "_vel_ff_scale", float("nan")))])
    arrays["meta_home_q"] = np.array(term._q_home.detach().cpu().numpy())
    # 계약을 기록에 박아 둔다 — 배포 쪽이 npz 만 보고도 무엇을 재생하는지 알아야 한다.
    arrays["meta_obs_dim"] = np.array([int(env.observation_manager.group_obs_dim["policy"][0])])
    arrays["meta_is_rnn"] = np.array(["yes" if getattr(player, "is_rnn", False) else "no"])
    arrays["meta_fabric_robot_dir"] = np.array([str(P.FABRIC_ROBOT_DIR)])
    arrays["meta_fabric_world"] = np.array([str(P.FABRIC_WORLD_FILENAME)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    l1 = np.linalg.norm(arrays["palm_fk_pos"] - arrays["palm_cmd_pos"], axis=-1)[:, 0] * 1000
    l2 = np.linalg.norm(arrays["tcp_pos"] - arrays["palm_fk_pos"], axis=-1)[:, 0] * 1000
    trk = np.abs(arrays["arm_meas"] - arrays["arm_target"])[:, 0].max(axis=-1)
    print(f"\n[REC] -> {args.out}  ({arrays['action'].shape[0]} 스텝)")
    print(f"[REC] L1 FK vs 지령      mean {l1.mean():7.2f}  p95 {np.percentile(l1,95):7.2f}  max {l1.max():7.2f}  mm")
    print(f"[REC] L2 물리TCP vs FK   mean {l2.mean():7.2f}  p95 {np.percentile(l2,95):7.2f}  max {l2.max():7.2f}  mm")
    print(f"[REC] sim 관절 추종오차  mean {trk.mean():7.4f}  max {trk.max():7.4f}  rad")
    print(f"[REC] 에피소드 종료 {int(arrays['done'].sum())}회")

    if args.hold_open:
        print("[REC] --hold_open: 창 유지(스트림 정지·창을 닫으면 종료)", flush=True)
        with torch.inference_mode():
            while simulation_app.is_running():
                env.sim.step(render=args.gui)
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
