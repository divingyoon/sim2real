#!/usr/bin/env python3
"""좌 v2 정책 **드라이런** — 실기 센서로 정책·fabric 을 돌리되 **발행하지 않는다**.

"정책이 안전하게 원하는 대로 출력하는가"를 로봇을 움직이지 않고 답하기 위한 도구다.
`left_policy_core` 배선을 그대로 쓰되 출력은 화면·csv 로만 나간다 —
**퍼블리셔가 아예 없다**(있으면 실수로 발행할 여지가 남는다).

    실기 /joint_states  ─┐
    /shaker_pose        ─┴─▶ core(obs 49D → 정책 → palm → fabric) ─▶ 화면·csv

검사하는 것 (각 스텝)
  · palm 목표가 PALM_BOX 안인가, 스텝 이동이 리미터 안인가
  · fabric 관절목표가 **관절 한계 안**이고 속도가 프로필 한계 안인가
  · 그 목표가 지금 실측에서 얼마나 떨어져 있는가(= 실기가 따라가야 할 거리)
  · 그리퍼 게이트가 언제 열리는가
  · TCP 가 테이블 발자국 위에서 얼마나 뜨는가 ← 안전의 핵심

실행 (로봇 무동작):
    python3 left_inference_dryrun.py --run logs/policy/left_v2E29 --steps 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[1]
ROBOT_CONTROL_SRC = "/home/user/rl_ws/robot_control/src"
HDGP = Path("/home/user/rl_ws/hdgp")
URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")

# fabric world 의 테이블 장애물(러너 로그 실측) — 발자국 안에서만 여유를 잰다.
TABLE = dict(cx=0.375, cy=-0.165, sx=0.550, sy=0.970, top=0.205)
JOINT_VEL_LIMIT = 2.0     # 프로필 velocity


def over_table(p) -> bool:
    return (abs(p[0] - TABLE["cx"]) <= TABLE["sx"] / 2
            and abs(p[1] - TABLE["cy"]) <= TABLE["sy"] / 2)


def joint_limits() -> tuple:
    sys.path.insert(0, ROBOT_CONTROL_SRC)
    from robot_control.profile import load_builtin_profile
    prof = load_builtin_profile("openarm_tesollo")
    lim = {j.canonical: (j.lower, j.upper) for j in prof.joints}
    lo = np.array([lim[f"l_aj_{i}"][0] for i in range(1, 8)])
    hi = np.array([lim[f"l_aj_{i}"][1] for i in range(1, 8)])
    return lo, hi


def make_fabric(home, device: str, env_step_dt: float = 0.02):
    """좌 fabric — L1 게이트(probe_gripper_left_fabrics)와 같은 셋업.

    `env_step_dt` 는 학습 env 의 step_dt(= sim dt × env decimation). 런 dump 에서
    읽어 넘긴다 — 리터럴을 박으면 다른 런을 배포할 때 조용히 어긋난다.
    """
    sys.path.insert(0, str(HDGP / "source/FABRICS/src"))
    sys.path.insert(0, str(HDGP / "source/openarm"))
    import torch
    from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmGripperLeftPoseFabric
    from fabrics_sim.integrator.integrators import DisplacementIntegrator
    from fabrics_sim.utils.utils import initialize_warp
    from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

    initialize_warp(str(device)[-1])
    world = WorldMeshesModel(batch_size=1, max_objects_per_env=8, device=device,
                             world_filename=P.FABRIC_WORLD_FILENAME)
    obj_ids, obj_ind = world.get_object_ids()
    # ★★fabric 적분 dt 는 **env.step_dt** 다(`grasp_left_fabric_action.py:112`
    #   `self._fabric_dt = float(env.step_dt)`), 즉 sim dt × env decimation = 0.02 s.
    #   09.03: 여기에 1/60/decimation = 0.00833 을 넣어 **2.4배 작게** 적분했다.
    #   그러면 정책 스텝당 fabric 이 학습의 42% 만 전진하는데, 팜 지령은 리미터
    #   속도로 계속 앞서가므로 관절목표가 목표점으로 곧게 질러간다 — 실기에서
    #   컵을 수평으로 쓸고 지나간(수평 153 mm vs 수직 17 mm) 궤적의 원인이다.
    dt = env_step_dt
    fab = OpenArmGripperLeftPoseFabric(
        1, device, dt, graph_capturable=False,
        robot_dir_name=P.FABRIC_ROBOT_DIR, robot_name=P.FABRIC_ROBOT_DIR,
        default_config_override=list(home))
    integ = DisplacementIntegrator(fab)
    state = dict(
        q=torch.tensor(home, device=device, dtype=torch.float32).unsqueeze(0).contiguous(),
        qd=torch.zeros(1, 7, device=device), qdd=torch.zeros(1, 7, device=device))
    pca = torch.zeros(1, 5, device=device)
    damp = P.FABRIC_DAMPING_GAIN * torch.ones(1, 1, device=device)
    dec = int(P.FABRIC_DECIMATION)

    def step(palm6, n: int = 0) -> np.ndarray:
        """한 정책 지령에 대한 fabric 블록. (n, 7) — 서브스텝마다 한 줄.

        ★★env 와 **정확히 같은 순서**여야 한다(`grasp_left_fabric_action.py:342`):
          `set_features` 를 블록당 **한 번** 부르고, 그 특징을 고정한 채
          `FABRIC_DECIMATION` 번 적분한다. 적분 스텝마다 다시 부르면 가속장이
          매번 재계산되어 학습과 다른 궤적이 나온다.

          서브스텝을 다 돌려주는 이유: 저Hz 정책 사이의 빈 공간을 메우는 것이
          fabric 의 일이라, 노드가 이 줄들을 차례로 발행해야 그 뒤의 500~700 Hz
          PD 가 이어받을 수 있다. 마지막 줄이 env 가 다음 스텝에 쓰는 값이다.
        """
        feat = torch.tensor(np.asarray(palm6, dtype=np.float32),
                            device=device, dtype=torch.float32).unsqueeze(0)
        fab.set_features(pca, feat, "euler_zyx", state["q"].detach(),
                         state["qd"].detach(), obj_ids, obj_ind, damp)
        out = []
        for _ in range(n if n > 0 else dec):
            state["q"], state["qd"], state["qdd"] = integ.step(
                state["q"].detach(), state["qd"].detach(), state["qdd"].detach(), dt)
            out.append(state["q"][0].detach().cpu().numpy().astype(np.float64))
        return np.asarray(out)

    step.fabric_dt = dt
    step.decimation = dec
    return step

def goal_center_from_run(env_yaml_path):
    """런 dump 의 `commands.object_pose.ranges` 중심을 목표로 쓴다.

    ★목표 상자는 트랙마다 다르다(fab79 x[0.36,0.46] vs v2E29 x[0.325,0.425]).
      다른 런의 goal 을 그대로 쓰면 정책이 학습 분포 밖 목표를 받는다.
    """
    import re as _re
    lines = Path(env_yaml_path).read_text().split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if _re.match(r"^  object_pose:", ln)), None)
    if start is None:
        raise SystemExit(f"[goal] {env_yaml_path} 에 commands.object_pose 가 없다")
    end = next((i for i in range(start + 1, len(lines))
                if _re.match(r"^  [a-z_]+:", lines[i])), len(lines))
    blk = lines[start:end]
    out = []
    for ax in ("pos_x", "pos_y", "pos_z"):
        i = next((k for k, ln in enumerate(blk)
                  if _re.match(rf"^\s*{ax}:", ln)), None)
        vals = []
        if i is not None:
            for ln in blk[i + 1:i + 3]:
                m = _re.match(r"^\s*- (-?[\d.eE+-]+)\s*$", ln)
                if m:
                    vals.append(float(m.group(1)))
        if len(vals) != 2:
            raise SystemExit(f"[goal] {ax} 범위를 못 읽었다 ({env_yaml_path})")
        out.append(0.5 * (vals[0] + vals[1]))
    return out


def step_dt_from_run(env_yaml_path) -> float:
    """런 dump 에서 학습 env 의 step_dt(= sim dt × env decimation)를 읽는다.

    ★fabric 적분 dt 가 이 값이다(`grasp_left_fabric_action.py:112`). 리터럴로 박으면
      다른 런(dt·decimation 이 다른)을 배포할 때 조용히 어긋난다 — 09.03 에 2.4배
      작은 dt 로 돌려 궤적이 통째로 달라졌다.
    """
    import re as _re
    txt = Path(env_yaml_path).read_text()
    dec = _re.search(r"^decimation: (\d+)", txt, _re.M)
    sdt = _re.search(r"^  dt: ([\d.eE+-]+)", txt, _re.M)
    if not (dec and sdt):
        raise SystemExit(f"[fabric] {env_yaml_path} 에서 decimation/dt 를 못 읽었다")
    return int(dec.group(1)) * float(sdt.group(1))



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=SIM2REAL / "logs/policy/left_v2E29")
    ap.add_argument("--goal", default=None, help="쉼표 7값 (기본: 스트림 npz 에서)")
    ap.add_argument("--stream", type=Path,
                    default=SIM2REAL / "logs/shadow/pour_entry/stream_left_v2b25.npz")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--rate", type=float, default=10.0, help="tick Hz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--cup-udp", type=int, default=46011,
                    help="물체 pose 를 받을 UDP 포트(pose_udp_tx). 0 이면 ROS 토픽")
    ap.add_argument("--cup-topic", default="/shaker_pose",
                    help="--cup-udp 0 일 때 쓸 ROS 토픽")
    ap.add_argument("--cup-side", type=int, default=1, help="UDP side 코드 (1=좌 대상)")
    args = ap.parse_args()

    sys.path.insert(0, str(SIM2REAL / "scripts"))
    from left_policy_core import LeftPolicyCore, LeftSensors
    from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy

    env_yaml = args.run / "params/env.yaml"
    agent_yaml = args.run / "params/agent.yaml"
    ckpt = next((args.run / "nn").glob("*.pth"))
    if args.goal:
        goal7 = np.array([float(v) for v in args.goal.split(",")])
    else:
        goal7 = np.load(args.stream, allow_pickle=True)["goal"][0].astype(np.float64)
    print(f"[dry] 체크포인트 {ckpt.name} · goal {np.round(goal7, 3).tolist()}", flush=True)

    import torch
    import yaml as _yaml
    # ★RNN 여부는 agent.yaml 이 진실원천이다 — v2E29 는 rnn 블록이 없는 **MLP** 다.
    #   LSTM 로더를 쓰면 get_default_rnn_state() 가 None 이라 초기화에서 죽는다.
    _net = _yaml.safe_load(agent_yaml.read_text())["params"]["network"]
    is_rnn = "rnn" in _net
    Loader = RLGamesLstmActorPolicy if is_rnn else RLGamesActorPolicy
    print(f"[dry] 네트워크 {'LSTM' if is_rnn else 'MLP'} (agent.yaml 기준)", flush=True)
    pol = Loader(str(agent_yaml), str(ckpt), obs_dim=49, action_dim=7, device=args.device)
    if is_rnn:
        pol.reset_states()

    def policy(obs):
        with torch.no_grad():
            a = pol.get_action(torch.as_tensor(obs, dtype=torch.float32,
                                               device=args.device).unsqueeze(0))
        return a[0].detach().cpu().numpy()

    core = LeftPolicyCore(policy=policy, fabric=None, run_env_yaml=env_yaml,
                          goal7=goal7, urdf_path=URDF)
    core.fabric = make_fabric(core.home, args.device)
    lo, hi = joint_limits()

    # ── ROS 구독 (읽기 전용 — 퍼블리셔 없음) ────────────────────────────
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from robot_profile import load_robot_profile

    prof = load_robot_profile("gripper_left")
    src = list(prof.arm_source)
    rclpy.init()
    node = Node("left_inference_dryrun")
    box: dict = {}

    def on_js(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in src):
            return
        box["q"] = np.array([m.position[idx[s]] for s in src])
        box["qd"] = (np.array([m.velocity[idx[s]] for s in src])
                     if len(m.velocity) >= len(m.name) else np.zeros(7))
        g = idx.get("openarm_left_finger_joint1")
        box["g"] = float(m.position[g]) if g is not None else 0.044

    node.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)

    # ★물체 pose 는 UDP 로 받는다. 인식은 vision-3090 의 ROS_DOMAIN_ID=126 에 있고
    #   실기 /joint_states 는 이쪽 도메인이라 한 노드에서 둘 다 구독할 수 없다 —
    #   그래서 `pose_udp_tx` 다리가 있다. 전용 FP++ 노드가 서면 그때 정리한다.
    # ⚠ 패킷 <Bfff> 는 **위치만** 담는다. 자세는 단위 쿼터니언(직립)으로 둔다 —
    #   셰이커가 테이블에 똑바로 서 있다는 전제다. 기울면 게이트 판정이 달라진다.
    cup_sock = None
    if args.cup_udp:
        import socket
        import struct
        cup_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cup_sock.bind(("0.0.0.0", args.cup_udp))
        cup_sock.setblocking(False)
        print(f"[dry] 물체 pose ← UDP :{args.cup_udp} (side {args.cup_side})", flush=True)

        def drain_cup():
            try:
                while True:
                    pkt, _ = cup_sock.recvfrom(64)
                    if len(pkt) == 13:
                        sd, x, y, z = struct.unpack("<Bfff", pkt)
                        if sd == args.cup_side:
                            box["cup_xyz"] = np.array([x, y, z], dtype=np.float64)
            except BlockingIOError:
                pass
    else:
        node.create_subscription(PoseStamped, args.cup_topic,
                                 lambda m: box.__setitem__(
                                     "cup_xyz", np.array([m.pose.position.x,
                                                          m.pose.position.y,
                                                          m.pose.position.z])), 10)

        def drain_cup():
            return None

    t0 = time.time()
    while time.time() - t0 < 10 and not ("q" in box and "cup_xyz" in box):
        rclpy.spin_once(node, timeout_sec=0.2)
        drain_cup()
    for key, what in (("q", "/joint_states"),
                      ("cup_xyz", f"물체 pose ({'UDP' if args.cup_udp else args.cup_topic})")):
        if key not in box:
            print(f"[dry] {what} 수신 없음 — 중단", flush=True)
            return 1
    print(f"[dry] 물체 {np.round(box['cup_xyz'], 3).tolist()}", flush=True)

    rows = []
    core.reset()
    if is_rnn:
        pol.reset_states()
    print(f"{'k':>3} {'gate':>5} {'palmXYZ(m)':>26} {'Δq목표−실측(deg)':>17} "
          f"{'q̇(rad/s)':>10} {'한계밖':>6} {'TCP여유(mm)':>11}", flush=True)
    period = 1.0 / args.rate
    prev_t = None
    for k in range(args.steps):
        tick_start = time.time()
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.01)
        drain_cup()
        s = LeftSensors(arm_q=box["q"], arm_qd=box["qd"], grip_q=box["g"], grip_qd=0.0,
                        cup_pos=box["cup_xyz"],
                        cup_quat=np.array([1.0, 0.0, 0.0, 0.0]))
        out = core.step(s)
        tgt = out.arm_q_target
        dq = np.degrees(tgt - box["q"])
        now = time.time()
        qd = (np.abs(tgt - rows[-1][1]) / max(now - prev_t, 1e-6)).max() if rows else 0.0
        outside = int(np.sum((tgt < lo) | (tgt > hi)))
        poses = core.fk.poses(tgt, box["g"], box["g"])
        clear = ((poses.tcp_pos[2] - TABLE["top"]) * 1000
                 if over_table(poses.tcp_pos) else float("nan"))
        print(f"{k:3d} {str(out.gate_open):>5} "
              f"{np.round(out.palm_target[:3], 3).tolist()!s:>26} "
              f"{np.abs(dq).max():17.1f} {qd:10.2f} {outside:6d} {clear:11.0f}", flush=True)
        rows.append((now, tgt.copy(), out.palm_target.copy(), out.action.copy(),
                     out.gate_open, clear, outside))
        prev_t = now
        time.sleep(max(0.0, period - (time.time() - tick_start)))

    ts = np.array([r[0] for r in rows])
    tg = np.array([r[1] for r in rows])
    pl = np.array([r[2] for r in rows])
    ac = np.array([r[3] for r in rows])
    cl = np.array([r[5] for r in rows])
    ob = np.array([r[6] for r in rows])
    vel = np.abs(np.diff(tg, axis=0)) / np.maximum(np.diff(ts), 1e-6)[:, None]
    print("\n=== 요약 ===", flush=True)
    print(f"  관절목표 한계 초과 스텝 {int((ob > 0).sum())}/{len(rows)}", flush=True)
    print(f"  관절목표 속도 최대 {vel.max():.2f} rad/s (프로필 한계 {JOINT_VEL_LIMIT})",
          flush=True)
    fin = cl[~np.isnan(cl)]
    print(f"  테이블 위 구간 {len(fin)}스텝 · TCP 최소여유 "
          f"{fin.min() if len(fin) else float('nan'):.0f} mm", flush=True)
    print(f"  게이트 열린 스텝 {int(sum(1 for r in rows if r[4]))}/{len(rows)}", flush=True)
    print(f"  palm 이동/스텝 최대 {np.abs(np.diff(pl[:, :3], axis=0)).max()*1000:.1f} mm",
          flush=True)
    print(f"  액션 |a| 최대 {np.abs(ac).max():.2f} (박스 ±1)", flush=True)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.csv, np.hstack([ts[:, None], tg, pl, ac]), delimiter=",",
                   header="t," + ",".join([f"q{i}" for i in range(1, 8)]
                                          + [f"palm{i}" for i in range(6)]
                                          + [f"a{i}" for i in range(7)]))
        print(f"  csv {args.csv}", flush=True)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
