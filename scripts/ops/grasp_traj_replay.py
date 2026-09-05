#!/usr/bin/env python3
"""sim 성공 궤적(`hdgp/log/grasp_traj/g1/*.hdf5`)을 실기 우팔로 **위치 재생**한다.

정책 추론 없이 "sim 이 성공한 그 동작"을 그대로 내는 경로다. 소스와 재현 조건은
`hdgp/log/grasp_traj/g1/README.md` 에 있다.

## ★재생에 반드시 들어가야 하는 것 셋 (전부 09.03 실측에서 나왔다)

**① 속도 전향보상.** 실기 JTC 는 `command_interfaces: [position]` 뿐이라 하드웨어
   `vel_cmd` 가 **영원히 0** 이다(`openarm_simple_hardware.cpp:291` 은 `{kp, kd,
   pos, vel, tau}` 를 한 프레임에 싣는다). 그러면 감쇠항이 `kd·(0 − v)` 가 되어
   움직임을 상시 반대로 밀고, 오차가 **속도에 비례**한다: `err ≈ (kd/kp)·v`.
   sim 은 `set_joint_velocity_target(fabric_qd)` 로 이걸 없앤다. 위치 지령에
   `(kd/kp)·q̇*` 를 더하면 **대수적으로 동일**하다:
       kp(q*−q) + kd(q̇*−q̇) = kp[(q* + (kd/kp)q̇*) − q] − kd·q̇
   ★이게 빠져서 "궤적을 그대로 재생해도 sim 동작이 안 나온다"가 됐었다.

**② 손 지령 과주행 제한.** sim 의 `hand_q_cmd` 는 "도달한 자세"가 아니라 "밀고 있는
   목표"다 — 실제 `hand_q` 와 p95 **1.45 rad** 차이 난다(손 kp 5.0 에서 포화).
   그대로 실기(p=4.5)에 넣으면 손가락을 그 격차만큼 쥐어짠다. 그렇다고 `hand_q`(측정)
   만 주면 그 자세가 sim 에서 이미 평형이라 **쥐는 힘이 0** 이 된다.
   → `hand_q + clip(hand_q_cmd − hand_q, ±overtravel)` 로 **제한된 과주행**을 준다.

**③ 실제 테이블 기준 바닥 가드.** 실기 테이블은 **0.205**(09.05 확정) 이고 sim env_v1 도 0.205 —
   30 mm 높다. sim 궤적은 sim 판 높이로 내려오므로 실기에서는 그만큼 **판을 파고든다**.
   손끝 z 를 FK 로 매 스텝 재고 하한을 지킨다. `--dry` 로 **미리** 전 구간을 검사한다.

사용 (★실기 동작은 사용자 승인 후):
    # 먼저 반드시 오프라인 검사
    python3 grasp_traj_replay.py --traj ../../hdgp/log/grasp_traj/g1/g1_y00.hdf5 --dry
    # 전이(reset_right_v2_fast)로 홈에 간 뒤
    python3 grasp_traj_replay.py --traj .../g1_y00.hdf5 --time-scale 0.5 --execute
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


SIM2REAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM2REAL / "scripts"))

# 실기 테이블 상면(m). ★09.05 정정: 줄자 0.205 + Fusion CAD(받침대 195 mm) 로 확정,
# sim env_v1 도 0.205. 09.03 의 0.230 은 카메라·짚기 사슬의 datum 오차(+21~24 mm)였다.
REAL_TABLE_TOP = 0.205
DEFAULT_CLEARANCE = 0.010          # 손끝이 판 위로 최소 이만큼
DEFAULT_OVERTRAVEL = 0.25          # rad. 손 과주행 상한(sim 격차 1.45 의 1/6)
VEL_FF_LIMIT = 0.20                # rad. 속도 전향보상 상한


def load_traj(path: Path) -> dict:
    """HDF5 에서 재생에 필요한 채널만 꺼낸다. 없으면 죽는다."""
    import h5py

    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        eps = sorted(f["episodes"].keys())
        if not eps:
            raise SystemExit(f"[traj] {path} 에 에피소드가 없다")
        ep = f[f"episodes/{eps[0]}"]
        need = ("arm_q_cmd", "hand_q", "hand_q_cmd", "object_pose", "palm_cmd")
        missing = [k for k in need if k not in ep]
        if missing:
            raise SystemExit(f"[traj] 채널 없음 {missing} — 스키마 확인")
        out = {k: ep[k][:].astype(float) for k in need}
    out["dt"] = float(attrs["dt"])
    out["hand_names"] = [str(x) for x in attrs["hand_joint_names"]]
    out["arm_names"] = [str(x) for x in attrs["arm_joint_names"]]
    out["spawn_xy"] = np.asarray(attrs["spawn_center_xy"], dtype=float)
    out["species"] = str(attrs["object_species"])
    return out


def permutation(src_names, dst_names) -> np.ndarray:
    """`dst[i] = src[perm[i]]`. 이름이 안 맞으면 즉시 죽는다 —
    손 관절 순서 스크램블은 조용히 통과하면 절대 안 된다."""
    src = list(src_names)
    missing = [n for n in dst_names if n not in src]
    if missing:
        raise SystemExit(f"[traj] 순열 실패 — 없는 관절 {missing}")
    return np.array([src.index(n) for n in dst_names], dtype=int)


def build_commands(traj: dict, prof, overtravel: float) -> dict:
    """재생용 지령 스트림을 만든다(발행 전 전부 계산해 둔다)."""
    arm_cmd = traj["arm_q_cmd"]
    perm = permutation(traj["hand_names"], list(prof.ee_canonical))
    hand_meas = traj["hand_q"][:, perm]
    hand_push = traj["hand_q_cmd"][:, perm]
    # ★② 제한된 과주행 — 도달 자세 + 밀던 방향으로 최대 `overtravel`
    hand_cmd = hand_meas + np.clip(hand_push - hand_meas, -overtravel, overtravel)

    # ★① 지령 자체의 속도 = fabric_qd. 차분으로 얻는다(끝점은 0 으로 둔다).
    qd = np.zeros_like(arm_cmd)
    qd[1:] = np.diff(arm_cmd, axis=0) / traj["dt"]
    return {"arm": arm_cmd, "arm_qd": qd, "hand": hand_cmd,
            "hand_meas": hand_meas, "hand_push": hand_push}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--robot", default="tesollo_bi_s__right")
    ap.add_argument("--run", type=Path,
                    default=SIM2REAL / "logs/policy/right_g1",
                    help="fabric dt·damping·decimation 을 읽을 런 dump")
    ap.add_argument("--time-scale", type=float, default=0.5,
                    help="(0,1]. 낮출수록 느리게 — sim 최대 2.14 rad/s 가 프로필 "
                         "한계 2.0 을 넘으므로 기본 0.5")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="팜 목표를 z 로 이만큼 올려 **fabric 에 다시 통과**시켜 관절 "
                         "궤적을 재생성한다[m]. 실기 판이 sim 보다 30 mm 높아 원본을 "
                         "그대로 재생하면 판을 파고든다. 0 이면 원본 arm_q_cmd 사용")
    ap.add_argument("--reverse", action="store_true",
                    help="★왔던 길을 그대로 되짚어 홈으로 돌아온다. 재생이 끝나면 팔이 "
                         "궤적 끝(홈에서 35°)에 있어 직선 램프로는 못 돌아온다 — "
                         "브리지가 판을 지날 수 있기 때문이다(09.03 실충돌)")
    ap.add_argument("--z-ramp", type=float, default=1.5,
                    help="z 오프셋을 넣는 데 걸리는 시간[s] — 0 이면 즉시(급가속)")
    ap.add_argument("--overtravel", type=float, default=DEFAULT_OVERTRAVEL,
                    help="손 과주행 상한[rad]. 0 이면 sim 도달자세만(쥐는 힘 없음)")
    ap.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE,
                    help="손끝이 **실제** 판 위로 지킬 최소 높이[m]")
    ap.add_argument("--table-top", type=float, default=REAL_TABLE_TOP)
    ap.add_argument("--max-vel", type=float, default=1.5, help="지령 속도 상한[rad/s]")
    ap.add_argument("--abort-tracking", type=float, default=0.30,
                    help="추종오차 중단 문턱[rad]")
    ap.add_argument("--no-vel-ff", action="store_true")
    ap.add_argument("--verify-fabric", action="store_true",
                    help="z 오프셋 0 으로도 fabric 재생성을 돌려 원본과 대조한다")
    ap.add_argument("--frames", type=int, default=0, help=">0 이면 앞에서 그만큼만")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--dry", action="store_true",
                    help="발행하지 않고 오프라인 검사만 — ★실행 전 필수")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if not 0.0 < args.time_scale <= 1.0:
        raise SystemExit("--time-scale 은 (0,1] 이다")


    from grasp_s2r_fabric import make_right_fabric, permutation as fab_perm
    from right_inference_node import vel_ff_ratio
    from robot_profile import load_robot_profile

    from right_inference_node import _scalar          # 런 dump 스칼라 파서

    prof = load_robot_profile(args.robot)
    traj = load_traj(args.traj)
    # ★fabric 파라미터는 런 dump 가 진실원천 — 리터럴로 박으면 다른 런에서 어긋난다.
    _env = args.run / "params/env.yaml"
    traj["fabrics_dt"] = float(_scalar(_env, "fabrics_dt", 0.008333333333333333))
    traj["damping"] = float(_scalar(_env, "fabrics_damping_gain", 10.0))
    traj["fabric_decimation"] = int(_scalar(_env, "fabric_decimation", 2))
    cmds = build_commands(traj, prof, args.overtravel)
    n = len(cmds["arm"]) if args.frames <= 0 else min(args.frames, len(cmds["arm"]))

    lo = np.array([prof.joint_limits[j]["lower"] for j in prof.arm_canonical])
    hi = np.array([prof.joint_limits[j]["upper"] for j in prof.arm_canonical])
    ff = np.zeros(7) if args.no_vel_ff else vel_ff_ratio()

    dt_pub = traj["dt"] / args.time_scale
    peak = float(np.abs(cmds["arm_qd"][:n]).max()) * args.time_scale
    print(f"[traj] {args.traj.name} · {n} 프레임 · 소스 {1/traj['dt']:.0f} Hz")
    print(f"  물체 {traj['species']} 소환 {traj['spawn_xy'].tolist()} · "
          f"정착 z {traj['object_pose'][0, 2]:.4f}")
    print(f"  time-scale {args.time_scale:g} → 발행 {1/dt_pub:.1f} Hz · "
          f"총 {n*dt_pub:.1f} s · 최대 관절속도 {peak:.3f} rad/s")
    if peak > args.max_vel:
        print(f"  ★요구 {peak:.3f} > 상한 {args.max_vel:g} — --time-scale 을 "
              f"{args.time_scale*args.max_vel/peak:.2f} 이하로 낮출 것")

    # ── 손 지령 요약 ────────────────────────────────────────────────────
    push = np.abs(cmds["hand_push"][:n] - cmds["hand_meas"][:n])
    used = np.abs(cmds["hand"][:n] - cmds["hand_meas"][:n])
    print(f"  손: sim 이 밀던 격차 p95 {np.percentile(push,95):.3f} rad → "
          f"실제 과주행 p95 {np.percentile(used,95):.3f} (상한 {args.overtravel:g})")

    # ── ③ 바닥 여유 — FK 로 전 구간 미리 검사 ────────────────────────────
    # ★fabric 은 여기선 **FK 전용**이다 — `tips()` 만 쓰고 적분은 돌리지 않는다.
    # ★★fabric 은 **URDF source 이름**(`rj_dg_1_1` …)을 쓴다. canonical 로 매칭하면
    #   손 관절이 0개로 잡혀 순열이 조용히 빈다 — 프로필의 `source` 를 거쳐 만든다.
    home27 = np.zeros(27)
    home27[:7] = cmds["arm"][0]
    fabric = make_right_fabric(home_q27=home27, device=args.device,
                               dt=traj["dt"], damping=0.0)
    fab_names = fabric.joint_names()
    ee_src = [prof.joint_limits[n]["source"] for n in prof.ee_canonical]
    fab_hand = [n for n in fab_names if n in set(ee_src)]
    if len(fab_hand) != len(prof.ee_canonical):
        raise SystemExit(
            f"[traj] fabric 손 관절 {len(fab_hand)}개 — 이름 매칭 실패. "
            f"표본 {fab_names[:10]}")
    hand_to_fab = fab_perm(ee_src, fab_hand)

    def tips_z(k: int) -> float:
        q27 = np.concatenate([cmds["arm"][k], cmds["hand"][k][hand_to_fab]])
        return float(fabric.tips(q27)[:, 2].min())

    # ★z 오프셋 — 팜 목표를 올려 fabric 을 다시 굴린다. env 와 **같은 순서**로:
    #   `set_features` 를 블록당 한 번, 그 뒤 fabric_decimation 번 적분.
    if args.z_offset != 0.0 or args.verify_fabric:
        # ★목표는 `palm_cmd`(fabric 이 느슨하게만 추종하는 액션 목표)가 아니라
        #   **기록된 관절이 실제로 만든 팜 자세**를 FK 로 되읽은 값이다. 그래야
        #   오프셋 0 에서 원본을 재현하고, 오프셋을 줘도 그만큼만 움직인다.
        fab2 = make_right_fabric(home_q27=home27, device=args.device,
                                 dt=float(traj["fabrics_dt"]),
                                 damping=float(traj["damping"]))
        dec = int(traj["fabric_decimation"])
        regen = np.zeros_like(cmds["arm"])
        # ★오프셋을 프레임 0 부터 통째로 주면 fabric 이 시작하자마자 30 mm 를 메우려
        #   급가속한다(09.03 실기: 20프레임 만에 추종오차 18.7° 로 가드 정지).
        #   `--z-ramp` 초에 걸쳐 서서히 넣어 홈에서 매끄럽게 떠나게 한다.
        ramp_n = max(1, int(args.z_ramp / traj["dt"]))
        for k in range(len(regen)):
            q27 = np.concatenate([cmds["arm"][k], cmds["hand"][k][hand_to_fab]])
            palm6 = np.asarray(fabric.palm_pose(q27), dtype=float).copy()
            palm6[2] += args.z_offset * min(1.0, (k + 1) / ramp_n)
            regen[k] = fab2.step(palm6, dec)
        d = np.degrees(np.abs(regen - cmds["arm"]))
        print(f"  fabric 재생성 vs 원본 arm_q_cmd: 평균 {d.mean():.2f}° · "
              f"최대 {d.max():.2f}°  (z 오프셋 {args.z_offset*1000:+.0f} mm)")
        if args.z_offset == 0.0 and d.max() > 5.0:
            print("  ★오프셋 0 인데 원본과 크게 다르다 — fabric 배선이 sim 과 어긋났다")
        if args.z_offset != 0.0:
            cmds["arm"] = regen
            cmds["arm_qd"][1:] = np.diff(regen, axis=0) / traj["dt"]


    if args.reverse:
        for k in ("arm", "hand", "hand_meas", "hand_push"):
            cmds[k] = cmds[k][::-1].copy()
        cmds["arm_qd"] = np.zeros_like(cmds["arm"])
        cmds["arm_qd"][1:] = np.diff(cmds["arm"], axis=0) / traj["dt"]
        print("  ★역방향 — 왔던 길을 그대로 되짚는다")

    step = max(1, n // 300)
    zs = np.array([tips_z(k) for k in range(0, n, step)])
    zmin = float(zs.min())
    floor = args.table_top + args.clearance
    print(f"  손끝 최저 z {zmin:.4f} · 실제 판 {args.table_top:.3f} → "
          f"판 위 {(zmin-args.table_top)*1000:+.0f} mm (하한 {args.clearance*1000:.0f} mm)")
    if zmin < floor:
        print(f"  ★★이 궤적은 실제 판을 {int((floor-zmin)*1000)} mm 파고든다.\n"
              f"    원인: sim 판 0.200 vs 실기 {args.table_top:.3f} — 30 mm 격차.\n"
              f"    ⚠그대로 재생하면 손이 테이블을 친다. --clearance 를 낮추는 것은\n"
              f"      해결이 아니다(가드만 눈감기는 것). 근본은 table_surface_z 재학습.")
    if args.dry or not args.execute:
        print("\nDRY RUN: 아무것도 발행하지 않았다.")
        return 0 if zmin >= floor else 2
    if zmin < floor:
        raise SystemExit("[traj] 바닥 가드 위반 — 발행하지 않는다")

    return run(args, prof, cmds, n, dt_pub, ff, lo, hi, tips_z, floor)


def run(args, prof, cmds, n, dt_pub, ff, lo, hi, tips_z, floor) -> int:
    """실제 발행 루프. 매 스텝 가드를 통과한 것만 내보낸다."""
    import rclpy
    from jtc_bridge_core import JointRemap
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    arm_remap = JointRemap(list(prof.arm_canonical), list(prof.arm_source),
                           prof.joint_limits)
    hand_remap = JointRemap(list(prof.ee_canonical), list(prof.ee_source),
                            prof.joint_limits)
    rclpy.init()
    node = Node("grasp_traj_replay")
    box: dict = {}

    def on_js(m):
        idx = {k: i for i, k in enumerate(m.name)}
        if all(s in idx for s in prof.arm_source):
            box["aq"] = np.array([m.position[idx[s]] for s in prof.arm_source])
            box["at"] = time.time()
        if all(s in idx for s in prof.ee_source):
            box["hq"] = np.array([m.position[idx[s]] for s in prof.ee_source])
            box["ht"] = time.time()

    node.create_subscription(JointState, prof.topics["arm_state"], on_js,
                             qos_profile_sensor_data)
    node.create_subscription(JointState, prof.topics["ee_state"], on_js,
                             qos_profile_sensor_data)
    apub = node.create_publisher(JointTrajectory, prof.topics["arm_traj"], 10)
    hpub = node.create_publisher(JointTrajectory, prof.topics["ee_traj"], 10)

    t0 = time.time()
    while time.time() - t0 < 10 and not all(k in box for k in ("aq", "hq")):
        rclpy.spin_once(node, timeout_sec=0.2)
    for k, what in (("aq", prof.topics["arm_state"]), ("hq", prof.topics["ee_state"])):
        if k not in box:
            raise SystemExit(f"[traj] {what} 수신 없음 — bringup 확인")

    # ★시작 자세 정합. 궤적 첫 프레임에서 멀면 **거부한다** — 직선으로 데려오면
    #   그 경로가 테이블을 지날 수 있다(09.03 실충돌).
    gap = float(np.abs(box["aq"] - cmds["arm"][0]).max())
    if gap > 0.15:
        raise SystemExit(
            f"[traj] 시작 자세가 궤적 첫 프레임에서 {math.degrees(gap):.1f}° 멀다.\n"
            "  전이 bag(reset_right_v2_fast)으로 홈에 먼저 갈 것.")
    print(f"[traj] 시작 정합 {math.degrees(gap):.2f}° · 재생 시작", flush=True)

    def publish(pub, remap, q):
        msg = JointTrajectory()
        msg.joint_names = list(remap.output_source)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in remap.apply(list(q))]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        pub.publish(msg)

    rows, stop, prev = [], None, None
    max_step = args.max_vel * dt_pub
    ff_seen = 0.0
    for k in range(n):
        loop = time.time()
        for _ in range(2):
            rclpy.spin_once(node, timeout_sec=0.001)
        if time.time() - box.get("at", 0) > 0.5 or time.time() - box.get("ht", 0) > 0.5:
            stop = "센서 두절"
            break

        # ★① 속도 전향보상 — 실제 재생 속도(= 소스 × time_scale)로 환산한다.
        lead = np.clip(ff * cmds["arm_qd"][k] * args.time_scale,
                       -VEL_FF_LIMIT, VEL_FF_LIMIT)
        ff_seen = max(ff_seen, float(np.abs(lead).max()))
        cmd = cmds["arm"][k] + lead
        if np.any(cmd < lo) or np.any(cmd > hi):
            stop = f"관절한계 밖 {np.round(np.degrees(cmd),1).tolist()}"
            break
        if prev is not None and float(np.abs(cmd - prev).max()) > max_step:
            stop = (f"지령 속도 {float(np.abs(cmd-prev).max())/dt_pub:.2f} rad/s "
                    f"> 상한 {args.max_vel}")
            break
        z = tips_z(k)
        if z < floor:
            stop = f"손끝 {(z-args.table_top)*1000:.0f} mm — 판 하한 위반"
            break
        # ★오차는 **실제로 보낸 값** 기준이다. 보상 전 목표로 재면 전향보상이 얹은
        #   몫이 그대로 "오차"로 잡혀 가드가 헛돈다.
        dev = box["aq"] - cmd
        err = float(np.abs(dev).max())
        if err > args.abort_tracking:
            _w = int(np.abs(dev).argmax())
            stop = (f"추종오차 {math.degrees(err):.1f}° @{prof.arm_canonical[_w]} "
                    f"(전관절 {np.round(np.degrees(dev),1).tolist()})")
            break

        publish(apub, arm_remap, cmd)
        publish(hpub, hand_remap, cmds["hand"][k])
        prev = cmd.copy()
        rows.append((time.time(), cmds["arm"][k].copy(), box["aq"].copy(), err, z))
        if k % 60 == 0:
            print(f"  [{k:4d}/{n}] 추종오차 {math.degrees(err):5.1f}° · "
                  f"손끝 판위 {(z-args.table_top)*1000:5.0f} mm", flush=True)
        time.sleep(max(0.0, dt_pub - (time.time() - loop)))

    print(f"\n[traj] {'정지: ' + stop if stop else '완료'} · {len(rows)}/{n} 프레임",
          flush=True)
    if rows:
        e = np.array([r[3] for r in rows])
        zz = np.array([r[4] for r in rows])
        print(f"  추종오차 최대 {math.degrees(e.max()):.1f}° · 평균 "
              f"{math.degrees(e.mean()):.1f}°  (sim 내부 평균 1.34°)")
        print(f"  손끝 최저 판위 {(zz.min()-args.table_top)*1000:.0f} mm · "
              f"속도전향 최대 {math.degrees(ff_seen):.2f}°")
    if args.log and rows:
        import csv
        with open(args.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + [f"cmd_{j}" for j in prof.arm_canonical]
                       + [f"meas_{j}" for j in prof.arm_canonical] + ["err", "tip_z"])
            for t, c, m, er, z in rows:
                w.writerow([t] + list(c) + list(m) + [er, z])
        print(f"  → {args.log}")
    rclpy.shutdown()
    return 0 if stop is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
