#!/usr/bin/env python3
"""우팔 `grasp_s2r` 라이브 추론 노드 — 정책 → fabric → JTC(팔) · 손 컨트롤러.

로직은 `grasp_s2r_core.GraspS2RCore` 에 있고 이 노드는 **ROS 배선과 안전 가드**만
담당한다(좌팔 `left_inference_node` 와 같은 구조).

09.03 좌팔에서 실기로 검증된 요소를 그대로 옮긴다:

  · **계약을 런 dump 에서 읽는다** — obs 차원·액션 clip·`step_dt`·목표 상자.
    손으로 옮긴 상수 네 곳이 하루를 태웠다. 리터럴은 여기 두지 않는다.
  · **정착** — 정책 시작 전에 실기를 **실제로 홈에 앉힌다**. sim 은 리셋에서
    텔레포트하지만 실기는 중력으로 처져 낮게 출발한다. 좌팔에서 TCP 가 52 mm
    낮게 시작해 같은 지령으로 판에 닿았다.
  · **가드는 실측 기준** — 관절한계 · 손 최저높이 · 추종오차 · 센서 두절.
  · **가드에 걸리면 즉시 홈 복귀** — 가드는 발행만 멈추므로, 두면 팔이 판 코앞에
    선 채로 남는다(좌팔 실측 1~6 mm).

★손 순서가 셋이다 — 드라이버 canonical(프로필 순) · obs(DOF 순) · fabric(자기 순).
  이 노드가 세 순열을 명시적으로 오간다. 섞으면 정책이 죽지 않고 조용히 이상해진다.

실행(무발행 연습이 기본 — `--execute` 를 줘야 발행한다):
    python3 right_inference_node.py --run logs/policy/right_g1 --steps 60
"""

from __future__ import annotations

import argparse
import math
import re
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[1]
URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")

#: 테이블 — 실기 0.205(09.05 줄자·CAD 확정, sim env_v1 동일) + 10 mm 여유.
#  카메라 사슬 +21 mm 편향(head 마운트 datum, B4 대기)은 여기서 보정하지 않는다.
TABLE = dict(cx=0.375, cy=-0.165, sx=0.550, sy=0.970, top=0.215)

# ⚠우팔 `grasp_s2r` 트랙에는 **적분형 처짐 보상이 없다**(있는 건 좌팔
#   `gripper/left/grasp_sensor` 쪽이다). 실기 팔이 sim 보다 더 처지는 것(홈에서 6.18°)은
#   별개의 sim-실기 질량/마찰 격차이고, 그건 **외부 중력보상 노드**가 담당한다.
#   그래서 기본값은 **끔** — 켜려면 `--droop-comp` 로 명시한다(sim 에서 멀어지는 쪽이라
#   실험으로만 쓴다). 좌팔 트랙과 같은 이득·클램프를 쓴다.
DROOP_GAIN = 0.05
DROOP_LIMIT = np.array([0.1000, 0.1000, 0.0675, 0.0675, 0.0175, 0.0175, 0.0175])

# ★★★sim PD 에는 **속도 목표**가 들어가는데 실기 JTC 에는 그 입구가 없다.
#     env: `set_joint_velocity_target(fabric_qd)` →  τ = kp(q*−q) + kd(qd*−qd)
#     실기: JTC command_interfaces 가 position 뿐 →  τ = kp(q_cmd−q) − kd·qd
#   두 식을 같게 만드는 q_cmd 가 **정확히** 존재한다:
#       kp(q*−q) + kd(qd*−qd) = kp[(q* + (kd/kp)·qd*) − q] − kd·qd
#   즉 위치 지령에 `(kd/kp)·qd*` 를 더하면 속도 목표를 준 것과 대수적으로 동일하다.
#   근사가 아니라 항등식이다.
#
#   이게 없으면 오차가 **속도에 비례**해 생긴다((kd/kp)·v). 그래서
#     · 궤적을 그대로 재생해도(정책 없이) sim 동작이 안 나오고
#     · 정책 주기를 60 Hz 로 올리면 속도가 3배 → 뒤처짐도 3배로 **더 나빠진다**
#   09.03 실측: sim 추종오차 평균 1.34° vs 실기 3.9°(같은 게인·같은 궤적).
#
#   ⚠게인은 실기 드라이버가 쓰는 값이어야 한다 — 여기 적어두지 않고 그 파일에서 읽는다.
ARM_GAINS_YAML = Path("/home/user/rl_ws/robot_control/ros_ws/src/openarm_description"
                      "/config/arm/v10/control_gains.yaml")
VEL_FF_LIMIT = 0.20      # rad. 정상 최대는 0.07·2.14 ≈ 0.15 — 여기 걸리면 이상신호다
# 정책 주기(50 ms) 중 콜백 배수에 쓰는 시간. 남는 시간은 어차피 sleep 으로 버린다.
SPIN_DRAIN_SEC = 0.015


def vel_ff_ratio(path: Path = ARM_GAINS_YAML) -> np.ndarray:
    """관절별 `kd/kp`. 게인 파일이 없으면 **죽는다** — 조용히 0 을 쓰면 그만큼
    뒤처지고, 그 뒤처짐은 "정책이 이상하다"로 위장된다."""
    import yaml
    if not path.exists():
        raise SystemExit(f"[right] 실기 게인 파일이 없다: {path}")
    g = yaml.safe_load(path.read_text())
    out = []
    for i in range(1, 8):
        k = g.get(f"joint{i}")
        if not k or "kp" not in k or "kd" not in k:
            raise SystemExit(f"[right] {path} 에 joint{i} 의 kp/kd 가 없다")
        out.append(float(k["kd"]) / float(k["kp"]))
    return np.array(out)


def over_table(p) -> bool:
    return (abs(p[0] - TABLE["cx"]) <= TABLE["sx"] / 2
            and abs(p[1] - TABLE["cy"]) <= TABLE["sy"] / 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=SIM2REAL / "logs/policy/right_g1")
    ap.add_argument("--steps", type=int, default=60, help="정책 스텝 수")
    ap.add_argument("--policy-hz", type=float, default=None,
                    help="정책·발행 주기. 기본은 런의 1/step_dt (시간축 1배)")
    ap.add_argument("--max-vel", type=float, default=2.0,
                    help="관절목표 속도 상한 rad/s — 넘으면 정지한다")
    ap.add_argument("--min-clearance", type=float, default=0.015,
                    help="손 최저점의 판 위 여유 하한 m")
    ap.add_argument("--abort-tracking", type=float, default=0.8,
                    help="팔 목표−실측 상한 rad")
    ap.add_argument("--settle-tol", type=float, default=0.01)
    ap.add_argument("--settle-timeout", type=float, default=8.0)
    ap.add_argument("--settle-clamp", type=float, default=0.12)
    ap.add_argument("--max-settle-gap", type=float, default=math.radians(10.0),
                    help="정착 램프를 허용할 최대 홈 간극 rad — 넘으면 거부한다"
                         "(직선 램프가 판을 관통할 수 있다)")
    ap.add_argument("--return-home", action="store_true", default=True)
    ap.add_argument("--no-return-home", dest="return_home", action="store_false")
    ap.add_argument("--return-rate", type=float, default=0.12)
    ap.add_argument("--return-stride", type=int, default=1,
                    help="역순 복귀에서 몇 스텝씩 건너뛸지 — 크면 빠르지만 거칠다")
    ap.add_argument("--return-speed", type=float, default=1.0,
                    help="역순 복귀 속도 배수(정책 주기 대비)")
    ap.add_argument("--object-udp", type=int, default=46012,
                    help="물체 pose UDP 포트. 0 이면 `--object` 고정값을 쓴다")
    ap.add_argument("--object", default=None, help="물체 x,y,z 고정값")
    ap.add_argument("--object-side", type=int, default=2)
    ap.add_argument("--no-stall-freeze", action="store_true",
                    help="★실기용 **실속 동결**을 끈다. sim 은 마디별 접촉으로 동결하지만 "
                         "실기 손은 손끝 렌치만 줘서 `|목표−실측|>0.3 rad` 로 대체했다. "
                         "그런데 테솔로는 게인이 낮아(p=4.5) 닫는 동안 그만큼 뒤처지므로 "
                         "**막히지 않았는데 동결**된다 — 09.03 실측: 정책은 폐쇄 0.366 을 "
                         "지시했는데 시너지가 0.201 에서 멈췄다(차 +0.165)")
    ap.add_argument("--close-speed", type=float, default=None,
                    help="시너지 폐쇄 rate 상한을 런 dump 값 대신 이걸로 쓴다. "
                         "★학습값(0.005)을 바꾸는 배포 override 다 — sim 과 달라진다. "
                         "근거: 정책은 스텝 ~80 에 리프트로 넘어가는데 그때 폐쇄가 0.31 "
                         "밖에 안 돼(rate 상한이 물림) 반경 40 mm 컵을 스치기만 한다. "
                         "sim 은 리프트 시점(~100)에 ~0.5 다.")
    ap.add_argument("--droop-comp", action="store_true",
                    help="적분형 처짐 보상을 위치 목표에 더한다 — ★sim 우팔 트랙에는 "
                         "없는 항이다(좌팔 트랙 방식). 실험용")
    ap.add_argument("--no-vel-ff", action="store_true",
                    help="속도 전향보상 (kd/kp)·fabric_qd 를 끈다(sim 은 켜져 있다)")
    ap.add_argument("--tip-force-zero", action="store_true",
                    help="★진단 전용 — 손끝 힘을 0 으로 강제한다. 바이어스는 홈 자세에서 "
                         "뜨는데 팔이 움직이면 센서에 걸리는 중력 성분이 달라져 잔차가 "
                         "남는다. `contact_force_max=10 N` 이라 2 N 만 남아도 정규화 0.2 로, "
                         "sim 에서 접촉 전까지 0 인 자리에 신호가 생긴다. 정책이 '이미 "
                         "닿았다'로 읽고 손을 여는지 이 옵션으로 가른다(09.03 step 70 반전)")
    ap.add_argument("--tip-bias-sec", type=float, default=2.0,
                    help="기동 시 손끝 F/T 바이어스를 이만큼 떠서 뺀다(무접촉 전제). "
                         "0 이면 빼지 않는다")
    ap.add_argument("--goal", default=None, help="목표 x,y,z (기본: 런 상자 중심)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--execute", action="store_true", help="★없으면 발행하지 않는다")
    args = ap.parse_args()

    sys.path.insert(0, str(SIM2REAL / "scripts"))
    import yaml as _yaml

    from grasp_s2r_core import DOF_TO_PROFILE, GraspS2RCore, S2RSensors
    from grasp_s2r_fabric import make_right_fabric, permutation
    from grasp_s2r_obs_builder import hand_dof_order
    from grasp_s2r_synergy import HAND_JOINT_NAMES, HAND_OPEN_POSE
    from jtc_bridge_core import JointRemap
    from left_inference_dryrun import step_dt_from_run
    from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy
    from robot_profile import load_robot_profile

    env_yaml = args.run / "params/env.yaml"
    agent_yaml = args.run / "params/agent.yaml"
    ckpt = next((args.run / "nn").glob("*.pth"))

    acfg = _yaml.safe_load(agent_yaml.read_text())["params"]
    is_rnn = "rnn" in acfg["network"]
    act_clip = float(acfg.get("env", {}).get("clip_actions", 1.0))
    step_dt = step_dt_from_run(env_yaml)
    hz = args.policy_hz or (1.0 / step_dt)
    # ★★우팔의 목표는 좌팔처럼 **고정 상자**가 아니다. env 는
    #   `goal_pos = settled + goal_offset_xyz` 로 정한다 — "물체가 실제로 놓인 자리에서
    #   이만큼 들어올려라". 그래서 물체를 래치한 뒤 그 값으로 만든다(아래 reset 부근).
    goal_off = _triple(env_yaml, "goal_offset_xyz", (0.0, 0.0, 0.12))
    goal3 = ([float(v) for v in args.goal.split(",")] if args.goal
             else [0.0, 0.0, 0.0])          # 물체 래치 뒤에 채운다

    import torch
    Loader = RLGamesLstmActorPolicy if is_rnn else RLGamesActorPolicy
    pol = Loader(str(agent_yaml), str(ckpt), obs_dim=155, action_dim=21,
                 device=args.device, action_clip=act_clip)
    if is_rnn:
        pol.reset_states()

    def policy(obs):
        with torch.no_grad():
            a = pol.get_action(torch.as_tensor(obs, dtype=torch.float32,
                                               device=args.device).unsqueeze(0))
        return a[0].detach().cpu().numpy()

    # ── ROS ────────────────────────────────────────────────────────────
    import rclpy
    from geometry_msgs.msg import WrenchStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    prof = load_robot_profile("tesollo_bi_s__right")
    arm_remap = JointRemap(list(prof.arm_canonical), list(prof.arm_source),
                           prof.joint_limits)
    hand_src = list(prof.ee_source)
    hand_canon = list(prof.ee_canonical)     # = 프로필 순
    if hand_canon != list(HAND_JOINT_NAMES):
        raise SystemExit(
            "[right] 드라이버 canonical 순서가 프로필 순과 다르다 — 순열을 다시 만들 것\n"
            f"  canonical {hand_canon[:4]} … vs 프로필 {list(HAND_JOINT_NAMES)[:4]} …")
    dof_names = list(hand_dof_order("r"))
    PROFILE_TO_DOF_IDX = permutation(hand_canon, dof_names)

    rclpy.init()
    node = Node("right_inference")
    box: dict = {}
    tips_f = np.zeros((5, 3))

    def on_arm(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in prof.arm_source):
            return
        box["aq"] = np.array([m.position[idx[s]] for s in prof.arm_source])
        box["aqd"] = (np.array([m.velocity[idx[s]] for s in prof.arm_source])
                      if len(m.velocity) >= len(m.name) else np.zeros(7))
        box["at"] = time.time()

    def on_hand(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in hand_src):
            return
        # 드라이버는 canonical(프로필) 순 — obs 는 DOF 순이라 여기서 바꾼다.
        hq = np.array([m.position[idx[s]] for s in hand_src])
        hqd = (np.array([m.velocity[idx[s]] for s in hand_src])
               if len(m.velocity) >= len(m.name) else np.zeros(20))
        box["hq_prof"], box["hqd_prof"] = hq, hqd
        box["hq"] = hq[PROFILE_TO_DOF_IDX]
        box["hqd"] = hqd[PROFILE_TO_DOF_IDX]
        box["ht"] = time.time()

    node.create_subscription(JointState, prof.topics["arm_state"], on_arm,
                             qos_profile_sensor_data)
    node.create_subscription(JointState, prof.topics["ee_state"], on_hand,
                             qos_profile_sensor_data)

    def _mk_tip_cb(k):
        def cb(m):
            tips_f[k] = (m.wrench.force.x, m.wrench.force.y, m.wrench.force.z)
        return cb

    fmt = prof.topics.get("tip_wrench_fmt")
    if fmt:
        for k in range(5):
            node.create_subscription(WrenchStamped, fmt.format(i=k + 1),
                                     _mk_tip_cb(k), qos_profile_sensor_data)

    arm_pub = node.create_publisher(JointTrajectory, prof.topics["arm_traj"], 10)
    hand_pub = node.create_publisher(JointTrajectory, prof.topics["ee_traj"], 10)

    # ── 물체 pose ───────────────────────────────────────────────────────
    sock = None
    if args.object:
        box["obj"] = np.array([float(v) for v in args.object.split(",")])
    elif args.object_udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", args.object_udp))
        sock.setblocking(False)

    def drain_obj(latch: bool = False) -> None:
        """★물체는 **정책 시작 시점의 한 값으로 고정**한다(사용자 규약 09.03).

        앵커가 `spawn` 모드라 더욱 그렇다 — 실시간 물체를 쓰면 액션 원점이 따라
        움직이는 되먹임이 된다. 큐는 계속 비워야 버퍼가 안 쌓인다.
        """
        if sock is None:
            return
        try:
            while True:
                pkt, _ = sock.recvfrom(64)
                if latch or len(pkt) != 13:
                    continue
                sd, x, y, z = struct.unpack("<Bfff", pkt)
                if sd == args.object_side:
                    box["obj"] = np.array([x, y, z], dtype=np.float64)
        except BlockingIOError:
            pass

    # ── ★손끝 F/T 바이어스 ─────────────────────────────────────────────
    #   09.03 실측: 무접촉인데 |F| 가 8.4~18.4 N 나온다(표준편차 0.05~0.3 N — 잡음이
    #   아니라 오프셋이다). `contact_force_max=10.0` 이라 빼지 않으면 obs 의
    #   `tip_force` 15칸이 **시작부터 포화**해, 학습에서 0 이던 자리가 ±1 이 된다.
    #   정책은 "항상 세게 쥐고 있다"고 읽는다.
    #   ⚠자세 의존성은 아직 안 쟀다 — 기동 자세에서 뜬 값이다. 팔이 크게 움직이면
    #     중력 성분이 달라질 수 있으니, 실기 라운드에서 무접촉 구간의 잔차를 볼 것.
    tip_bias = np.zeros((5, 3))
    t0 = time.time()
    while time.time() - t0 < 10 and not all(k in box for k in ("aq", "hq", "obj")):
        rclpy.spin_once(node, timeout_sec=0.2)
        drain_obj()
    for k, what in (("aq", prof.topics["arm_state"]), ("hq", prof.topics["ee_state"]),
                    ("obj", "물체 pose")):
        if k not in box:
            print(f"[right] {what} 수신 없음 — 중단", flush=True)
            return 1

    # ── fabric ─────────────────────────────────────────────────────────
    fab_dt = float(_scalar(env_yaml, "fabrics_dt", 0.008333333333333333))
    damping = float(_scalar(env_yaml, "fabrics_damping_gain", 10.0))
    # ★★fabric 의 초기 상태이자 cspace rest 는 **홈**이어야 한다. 0 으로 두면 내부
    #   상태가 차렷에 앉아 첫 출력이 실측(g1 홈)에서 55° 떨어지고 가드가 즉시 막는다
    #   (09.03 실기: 추종오차 0.97 rad · 0스텝). env 도 `default_joint_pos` 로 세운다.
    #   손 구간은 s2r 의 open 자세(정책이 리셋에서 보는 자세)로 채운다.
    _home_arm7 = np.array(_home_arm(env_yaml))
    _fab_home = np.zeros(27)
    _fab_home[:7] = _home_arm7
    _hand_open_prof = np.asarray(HAND_OPEN_POSE, dtype=float)
    fabric = make_right_fabric(home_q27=_fab_home, device=args.device,
                               dt=fab_dt, damping=damping)
    # ★★fabric 은 **URDF source 이름**을 쓴다(`rj_dg_1_1` …), canonical(`r_hj_thumb_1`)
    #   이 아니다. canonical 로 매칭하면 손 관절이 **0개**로 잡혀 순열이 조용히 비어
    #   버린다(09.03 실측). 그래서 프로필의 `source` 를 거쳐 만든다.
    fab_names = fabric.joint_names()
    dof_src = [prof.joint_limits[n]["source"] for n in dof_names]
    fab_hand = [n for n in fab_names if n in set(hand_src)]
    if len(fab_hand) != 20:
        raise SystemExit(
            f"[right] fabric 손 관절 {len(fab_hand)}개 — 20개여야 한다. "
            f"fabric 이름 표본 {fab_names[:10]}")
    PROFILE_TO_FAB = permutation(hand_src, fab_hand)
    DOF_TO_FAB = permutation(dof_src, fab_hand)
    # 손 구간을 순열이 생긴 뒤에 채운다 — fabric 순서를 알아야 넣을 수 있다.
    fabric.sync_hand(_hand_open_prof[PROFILE_TO_FAB])

    core = GraspS2RCore(
        policy=policy, fabric_palm_pose=fabric.palm_pose, fabric_tips=fabric.tips,
        fabric_step=fabric.step, run_dir=args.run, goal3=goal3,
        # ★프로필의 `joint_limits` 는 dict 다(source/sign/lower/upper/…) — 튜플이 아니다.
        soft_limits=np.array([[prof.joint_limits[n]["lower"],
                               prof.joint_limits[n]["upper"]] for n in hand_canon]),
        hand_dof_to_fabric=DOF_TO_FAB)
    if args.close_speed is not None:
        import dataclasses
        core.syn_cfg = dataclasses.replace(core.syn_cfg,
                                           close_speed=float(args.close_speed))
        core.hand.cfg = core.syn_cfg
        print(f"[right] ★폐쇄 rate override {args.close_speed:g} "
              f"(학습값 0.005) — 스텝 80 예상 폐쇄 "
              f"{min(1.0, args.close_speed*80):.2f}", flush=True)

    lo = np.array([prof.joint_limits[n]["lower"] for n in prof.arm_canonical])
    hi = np.array([prof.joint_limits[n]["upper"] for n in prof.arm_canonical])

    print(f"[right] 계약: obs 155 · act 21{' · LSTM' if is_rnn else ''} · "
          f"act_clip {act_clip:g} · step_dt {step_dt:.4f} s · goal "
          f"{np.round(goal3, 3).tolist()}", flush=True)
    print(f"[right] {'★발행' if args.execute else '무발행(연습)'} · "
          f"체크포인트 {ckpt.name} · 물체 {np.round(box['obj'], 3).tolist()}", flush=True)
    print(f"[right] 주기 {hz:.1f} Hz (학습 {1/step_dt:.1f} Hz · 시간축 "
          f"{step_dt * hz:.2f}배) · 상한: 속도 {args.max_vel} rad/s · 손 여유 "
          f"≥{args.min_clearance*1000:.0f} mm · 추종오차 ≤{args.abort_tracking} rad",
          flush=True)

    dt_pub = 1.0 / hz
    max_step = args.max_vel * dt_pub

    def publish_arm(q_can) -> None:
        if not args.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = list(arm_remap.output_source)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in arm_remap.apply(list(q_can))]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        arm_pub.publish(msg)

    def publish_hand(q_prof) -> None:
        if not args.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = hand_src
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in np.asarray(q_prof, dtype=float)]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        hand_pub.publish(msg)

    def hand_min_z() -> float:
        """손끝 5개의 최저 높이 — 판 긁힘 판정의 기준."""
        q27 = _q27(box, DOF_TO_FAB)
        return float(fabric.tips(q27)[:, 2].min())

    # ── 정착 ───────────────────────────────────────────────────────────
    home_arm = np.array(_home_arm(env_yaml))
    bias = np.zeros(7)
    err_home = float(np.abs(home_arm - box["aq"]).max())
    # ★★케이지(중심·반경)는 **홈 자세**에서 재야 한다. env 도 부팅 시 홈에서 한 번
    #   재고 고정한다(`_report_home_cage`). 차렷의 편 손에서 재면 반경이 절반 이하로
    #   나온다(09.03 실측: 차렷 0.0491 m vs sim 홈 0.1202 m) — 닫기 게이트의 문턱이
    #   통째로 틀어진다. 그래서 **정착이 끝난 뒤** `core.reset` 을 부른다.
    if args.settle_timeout > 0:
        # ★★홈까지 **램프**로 간다. 계단 지령은 위험하다 — g1 홈은 차렷에서 55°
        #   떨어져 있고, kp 70 에 그만큼을 한 번에 주면 팔이 튄다.
        #   램프 중에도 매 프레임 손 최저높이를 확인한다(판 긁힘 차단).
        q_start = np.asarray(box["aq"], dtype=float).copy()
        gap = float(np.abs(home_arm - q_start).max())
        # ★★관절공간 직선 램프는 **간극이 작을 때만** 안전하다. 차렷(간극 55°)에서
        #   g1 홈까지 직선으로 가면 경로가 판 상면보다 **246 mm 아래**를 지난다
        #   (09.03 FK 검산). 검증된 전이 궤적이 따로 있다:
        #       logs/shadow/reset_both/reset_right_v2.npz  (차렷 → g1 홈, 손끝 +16 mm)
        #   그러니 큰 간극이면 여기서 멈추고 사람을 부른다 — 조용히 질러가지 않는다.
        if gap > args.max_settle_gap:
            print(f"[right] ★정착 거부: 홈과 {math.degrees(gap):.1f}° 떨어져 있다"
                  f"(상한 {math.degrees(args.max_settle_gap):.1f}°).\n"
                  "  직선 램프는 판을 관통한다 — 먼저 전이 궤적을 재생할 것:\n"
                  "  python3 scripts/nodes/shadow_replay.py --sim "
                  "logs/shadow/reset_both/reset_right_v2.npz --robot tesollo_bi_s__right "
                  "--arm-only --execute", flush=True)
            rclpy.shutdown()
            return 1
        n_ramp = max(int(np.ceil(gap / args.return_rate * hz)), 1)
        print(f"[right] 정착 램프 {n_ramp} 프레임 ({n_ramp/hz:.1f} s · 간극 "
              f"{math.degrees(gap):.1f}°)", flush=True)
        for i in range(1, n_ramp + 1):
            a = i / n_ramp
            publish_arm(q_start * (1.0 - a) + home_arm * a)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt_pub)
        # 램프가 끝난 뒤에만 적분 보상으로 잔여 처짐을 지운다.
        ts = time.time()
        while time.time() - ts < args.settle_timeout:
            rclpy.spin_once(node, timeout_sec=0.002)
            err_home = float(np.abs(home_arm - box["aq"]).max())
            if err_home < args.settle_tol:
                break
            bias = np.clip(bias + 0.05 * (home_arm - box["aq"]),
                           -args.settle_clamp, args.settle_clamp)
            publish_arm(home_arm + bias)
            time.sleep(dt_pub)
        print(f"[right] 정착: 실기−홈 {math.degrees(err_home):.2f}° · 선보상 "
              f"|max| {math.degrees(float(np.abs(bias).max())):.2f}°", flush=True)

    if args.tip_bias_sec > 0:
        acc, t0 = [], time.time()
        while time.time() - t0 < args.tip_bias_sec:
            rclpy.spin_once(node, timeout_sec=0.01)
            acc.append(tips_f.copy())
        tip_bias = np.mean(acc, axis=0)
        print(f"[right] 손끝 바이어스 |F| "
              f"{np.round(np.linalg.norm(tip_bias, axis=1), 2).tolist()} N "
              f"(무접촉 {args.tip_bias_sec:.1f}s · 빼고 쓴다)", flush=True)

    fabric.sync_hand(box["hq_prof"][PROFILE_TO_FAB])
    obj_latched = np.array(box["obj"], dtype=float)
    if not args.goal:
        core.goal3 = obj_latched + np.asarray(goal_off, dtype=float)
        print(f"[right] 목표 = 물체 + {list(goal_off)} = "
              f"{np.round(core.goal3, 3).tolist()}", flush=True)
    core.reset(arm_q=box["aq"], hand_q=box["hq"], object_pos=box["obj"])
    # ★케이지 반경은 홈 자세의 손끝 배치가 정한다. sim 홈에서 잰 값이 0.1202 m 인데
    #   크게 벗어나면 정착을 건너뛰었거나(차렷에서 재면 0.049 m) 손 자세가 다르다는
    #   뜻이다 — 게이트 문턱이 통째로 틀어지므로 조용히 넘어가지 않는다.
    _R_CAGE_SIM = 0.1202
    _rel = abs(core._r_cage - _R_CAGE_SIM) / _R_CAGE_SIM
    print(f"[right] 물체 고정 {np.round(obj_latched, 3).tolist()} · r_cage "
          f"{core._r_cage:.4f} m (sim 홈 {_R_CAGE_SIM:.4f})"
          f"{'  ★sim 과 %.0f%% 차이 — 정착을 건너뛰었나?' % (_rel * 100) if _rel > 0.25 else ''}",
          flush=True)

    rows, stop, prev = [], None, None
    # ★스케일은 런 dump 가 진실원천 — sim 이 속도 목표에 곱하는 그 값이다.
    #   ⚠env.yaml 은 `python/tuple` 태그를 담고 있어 `yaml.safe_load` 로 안 열린다.
    #   같은 파일의 `step_dt_from_run` 과 같은 방식(텍스트 파싱)을 쓴다.
    _m = re.search(r"^fabric_velocity_ff_scale:\s*([\d.eE+-]+)",
                   (args.run / "params/env.yaml").read_text(), re.M)
    if _m is None:
        raise SystemExit("[right] 런 dump 에 fabric_velocity_ff_scale 이 없다")
    _ff_scale = float(_m.group(1))
    droop = np.zeros(7)
    ff_ratio = np.zeros(7) if args.no_vel_ff else vel_ff_ratio() * _ff_scale
    if args.droop_comp:
        print(f"[right] ★적분형 처짐 보상 켬 (이득 {DROOP_GAIN} · 상한 "
              f"{np.round(np.degrees(DROOP_LIMIT), 1).tolist()}°) — sim 에는 없는 항",
              flush=True)
    ff_seen = 0.0
    if not args.no_vel_ff:
        print(f"[right] 속도 전향보상 (kd/kp)×{_ff_scale:g} = "
              f"{np.round(ff_ratio, 4).tolist()}", flush=True)
    trail: list = []           # (arm7, hand20 프로필순) — 역순 복귀용 발행 이력
    for step in range(args.steps):
        loop = time.time()
        # ★★큐를 **비운다**. spin_once 몇 번으로는 못 따라간다 — 손끝 wrench 5개
        #   (각 100 Hz)까지 구독하면 초당 1400여 건이 들어오는데 정책은 20 Hz 다.
        #   덜 비우면 콜백이 밀려 `box["at"]/["ht"]` 타임스탬프가 낡고, 실제로는
        #   멀쩡한 센서가 "두절"로 잡힌다(09.03 실측: 42스텝에서 오진 정지).
        _drain = time.time() + SPIN_DRAIN_SEC
        while time.time() < _drain:
            rclpy.spin_once(node, timeout_sec=0.0)
        drain_obj(latch=True)
        if time.time() - box.get("at", 0) > 0.5 or time.time() - box.get("ht", 0) > 0.5:
            stop = "센서 두절"
            break

        # ★fabric 의 손 상태만 실측으로 맞춘다(팔은 영속 상태) — env 와 동일.
        fabric.sync_hand(box["hq_prof"][PROFILE_TO_FAB])
        core.hand_stall_freeze = not args.no_stall_freeze
        out = core.step(S2RSensors(
            arm_q=box["aq"], arm_qd=box["aqd"], hand_q=box["hq"], hand_qd=box["hqd"],
            object_pos=obj_latched,
            tip_force_world=(np.zeros((5, 3)) if args.tip_force_zero
                             else tips_f - tip_bias),
            tip_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))))

        tgt = out.arm_q_target
        if np.any(tgt < lo) or np.any(tgt > hi):
            stop = f"관절한계 밖 {np.round(np.degrees(tgt), 1).tolist()}"
            break
        # ★속도 전향보상 — sim 의 `set_joint_velocity_target` 과 대수적으로 등가.
        #   ⚠가드는 **실제로 발행하는 값**을 봐야 한다. 보상 전 목표만 검사하면
        #   보상이 얹은 몫은 아무도 안 본 채 모터로 나간다.
        lead = np.clip(ff_ratio * fabric.qd(), -VEL_FF_LIMIT, VEL_FF_LIMIT)
        ff_seen = max(ff_seen, float(np.abs(lead).max()))
        if args.droop_comp:
            droop = np.clip(droop + DROOP_GAIN * (tgt - box["aq"]),
                            -DROOP_LIMIT, DROOP_LIMIT)
        cmd = tgt + lead + droop
        if np.any(cmd < lo) or np.any(cmd > hi):
            stop = f"보상 후 관절한계 밖 {np.round(np.degrees(cmd), 1).tolist()}"
            break
        if prev is not None and float(np.abs(cmd - prev).max()) > max_step:
            stop = (f"관절지령 속도 {float(np.abs(cmd-prev).max())/dt_pub:.2f} rad/s "
                    f"> 상한 {args.max_vel}")
            break
        zmin = hand_min_z()
        if over_table(fabric.palm_pose(_q27(box, DOF_TO_FAB))[:3]) and \
                zmin - TABLE["top"] < args.min_clearance:
            stop = f"손 최저 {((zmin - TABLE['top']) * 1000):.0f} mm < 하한"
            break
        err = float(np.abs(box["aq"] - tgt).max())
        if err > args.abort_tracking:
            stop = f"추종오차 {err:.2f} rad"
            break

        publish_arm(cmd)
        hand_prof = out.hand_q_target[DOF_TO_PROFILE]
        publish_hand(hand_prof)
        prev = cmd.copy()
        # ★★지나온 **경로**를 쌓아 둔다 — 끝나면 역순으로 되짚어 되돌아간다.
        #   폐루프로 간 길이라 되돌아가는 길도 그 길이다(직선 램프는 판을 관통한다 —
        #   09.03 실충돌). ★속도 전향보상(lead)은 빼고 `tgt` 를 쌓는다: lead 는 그
        #   순간의 속도에 대한 보정이라 다른 속도로 되짚을 때는 틀린 값이 된다.
        trail.append((tgt.copy(), hand_prof.copy()))
        rows.append((time.time(), tgt.copy(), out.close_gate, err, zmin,
                     out.action.copy(), out.diag["syn_close_mean"],
                     out.obs[111:131].copy()))          # ★joint_err 20칸 — sim 대조용
        if step % 5 == 0:
            print(f"  [{step:3d}] gate {out.close_gate:.2f} · 폐쇄 "
                  f"{out.diag['syn_close_mean']:.3f} · 추종오차 "
                  f"{math.degrees(err):5.1f}° · 손 판위 "
                  f"{(zmin - 0.200) * 1000:5.0f} mm", flush=True)
        time.sleep(max(0.0, dt_pub - (time.time() - loop)))

    print(f"\n[right] {'정지: ' + stop if stop else '완료'} · {len(rows)}스텝",
          flush=True)
    if rows:
        ts = np.array([r[0] for r in rows])
        tg = np.array([r[1] for r in rows])
        errs = np.array([r[3] for r in rows])
        zs = np.array([r[4] for r in rows])
        print(f"  추종오차 최대 {math.degrees(errs.max()):.1f}° · 평균 "
              f"{math.degrees(errs.mean()):.1f}° · 손 최저 판위 "
              f"{(zs.min() - 0.200) * 1000:.0f} mm", flush=True)
        # ★sim 기준선: 추종오차 평균 1.34° · 최대 10.22°(g1_y00, 같은 게인·같은 정책).
        #   실기가 이보다 한참 크면 보상이 아니라 배선을 의심할 것.
        print(f"  속도전향 최대 {math.degrees(ff_seen):.2f}° · 처짐보상 "
              f"{math.degrees(float(np.abs(droop).max())):.2f}°"
              f"{'  ★상한에 걸렸다' if ff_seen >= VEL_FF_LIMIT - 1e-9 else ''}"
              f"   (sim 추종오차 평균 1.34° 최대 10.22°)", flush=True)
        if len(ts) > 1:
            v = np.abs(np.diff(tg, axis=0)).max(axis=1) / np.maximum(np.diff(ts), 1e-6)
            print(f"  발행 주기 중앙 {np.median(np.diff(ts))*1000:.0f} ms · "
                  f"관절목표 속도 최대 {v.max():.2f} rad/s", flush=True)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            acts = np.array([r[5] for r in rows])
            sc = np.array([[r[6]] for r in rows])
            je = np.array([r[7] for r in rows])
            np.savetxt(args.csv, np.hstack([ts[:, None], tg, acts, sc, je]),
                       delimiter=",",
                       header="t," + ",".join(f"q{i}" for i in range(1, 8)) + ","
                       + ",".join(f"a{i}" for i in range(21)) + ",syn_close,"
                       + ",".join(f"je{i}" for i in range(20)))
            print(f"  csv {args.csv}", flush=True)

    # ── ★★역순 복귀 — 간 길을 그대로 되짚는다 ────────────────────────
    #   정책이 간 경로는 **가 봤으니 통과 가능**하다. 직선 램프는 그 보장이 없다
    #   (09.03: 차렷→홈 직선이 판 상면보다 246 mm 아래를 지났다).
    #   성공·가드정지 **둘 다**에서 돈다 — 성공해도 팔은 컵 앞에 남아 있다.
    #   ⚠손도 함께 되짚는다. 컵을 쥔 채로 팔만 물러나면 컵을 끌고 온다 —
    #     역순 첫 구간이 손을 펴는 구간이라 자연히 놓게 된다.
    if args.execute and args.return_home and trail:
        print(f"[right] 역순 복귀 — 발행 이력 {len(trail)} 스텝을 되짚는다", flush=True)
        for k in range(len(trail) - 1, -1, -args.return_stride):
            q_a, q_h = trail[k]
            publish_arm(q_a)
            publish_hand(q_h)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt_pub / max(args.return_speed, 1e-6))
        # 이력의 시작점 = 정책 첫 스텝 목표. 거기서 홈까지 짧게 램프해 마무리한다.
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.05)
        q0 = np.asarray(box["aq"], dtype=float)
        n = max(int(np.ceil(np.abs(home_arm - q0).max() / args.return_rate * hz)), 1)
        for i in range(1, n + 1):
            a = i / n
            publish_arm(q0 * (1.0 - a) + home_arm * a)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt_pub)
        time.sleep(1.0)
        for _ in range(6):
            rclpy.spin_once(node, timeout_sec=0.05)
        print(f"[right] 복귀 완료 · 실기−홈 "
              f"{math.degrees(float(np.abs(home_arm - box['aq']).max())):.2f}°",
              flush=True)

    rclpy.shutdown()
    return 1 if stop else 0


# ---------------------------------------------------------------------------
def _q27(box, dof_to_fab) -> np.ndarray:
    """실측 → fabric 순서 27값."""
    return np.concatenate([np.asarray(box["aq"], dtype=float),
                           np.asarray(box["hq"], dtype=float)[dof_to_fab]])


def _triple(env_yaml_path, key, default):
    """dump 의 3값 튜플. ★`strip("- ")` 은 음수 부호를 지운다 — 정규식으로 잡는다."""
    import re
    txt = Path(env_yaml_path).read_text()
    m = re.search(rf"^\s*{key}:.*?\n((?:\s*- -?[\d.eE+-]+\n){{3}})", txt, re.M)
    if m is None:
        return default
    vals = re.findall(r"-?[\d.eE+-]+", m.group(1).replace("- ", " "))
    return tuple(float(v) for v in vals) if len(vals) == 3 else default


def _scalar(env_yaml_path, key, default):
    import re
    m = re.search(rf"^\s*{key}:\s*([\d.eE+-]+)\s*$",
                  Path(env_yaml_path).read_text(), re.M)
    return float(m.group(1)) if m else default


def _home_arm(env_yaml_path) -> list[float]:
    """런 dump 의 팔 홈 7값. ★홈의 진실원천은 소스 상수가 아니라 dump 다."""
    import re
    lines = Path(env_yaml_path).read_text().split("\n")
    for i, ln in enumerate(lines):
        if re.match(r"^\s+joint_pos:", ln):
            d = {}
            for k in range(i + 1, min(i + 60, len(lines))):
                m = re.match(r"^\s+(r_aj_\d): (-?[\d.eE+-]+)\s*$", lines[k])
                if m:
                    d[m.group(1)] = float(m.group(2))
                elif re.match(r"^\s{0,6}[a-z_]+:", lines[k]) and d:
                    break
            if len(d) == 7:
                return [d[f"r_aj_{j}"] for j in range(1, 8)]
    raise SystemExit(f"[right] {env_yaml_path} 에서 팔 홈을 못 읽었다")


if __name__ == "__main__":
    raise SystemExit(main())
