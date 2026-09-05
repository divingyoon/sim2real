#!/usr/bin/env python3
"""배포 인터프리터의 Fabrics 가 Isaac 의 Fabrics 와 **같은 관절 해**를 내는가.

이것이 "fabric IK 연동"의 진짜 관문이다. 실기에서 Fabrics 를 돌리려면 다음 셋이 성립해야
하는데, 앞의 둘은 이미 확인됐고 **셋째가 여기서 정해진다**:

  ① 배포 파이썬(ROS py3.10 + warp)에서 fabrics_sim 이 GPU 로 돈다      ✅ (§10, P1)
  ② 자산(URDF·world·params)이 sim 학습과 동일하다                      ✅ (프로필 테스트가 고정)
  ③ **같은 지령을 주면 같은 관절 목표가 나온다**                        ← 여기

왜 ①②만으로는 부족한가. 같은 라이브러리·같은 자산이라도 인터프리터·부동소수·버전이
다르면 적분 궤적이 갈릴 수 있고, 갈리면 실기는 sim 이 계획한 자세가 아닌 곳으로 간다.
그리고 그 어긋남은 오류가 아니라 **조용한 오프셋**으로만 나타난다.

방법: Isaac 기록(npz)에서 palm 지령 시계열과 fabric 파라미터를 읽어, 여기서 같은 fabric 을
같은 초기 상태로 세우고 **같은 순서로** 적분한 뒤 `fabric_q` 를 스텝별로 대조한다.
파라미터는 전부 npz 에서 온다 — preset 에서 다시 읽으면 그 사이 소스가 바뀐 만큼 갈린다.

실행:
    source /opt/ros/humble/setup.bash && . .venv/bin/activate
    python3 scripts/probes/probe_fabric_deploy_parity.py --sim logs/shadow/sim_fab_test16_gcON.npz \\
        --fabrics-src ~/rl_ws/hdgp/source/FABRICS/src
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REQUIRED_META = (
    "meta_fabric_dt", "meta_fabric_decimation", "meta_fabric_damping",
    "meta_home_q", "meta_fabric_robot_dir", "meta_fabric_world",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="probe_fab_shadow_record 의 npz")
    parser.add_argument("--fabrics-src", type=Path,
                        default=Path.home() / "rl_ws/hdgp/source/FABRICS/src")
    parser.add_argument("--steps", type=int, default=0, help=">0 이면 앞에서 그만큼만")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance-mm", type=float, default=1.0,
                        help="TCP 공간 허용 편차[mm]. 기본 1.0 — Fabrics 자신의 수렴 바닥"
                             "(정상구간 L1 3.6 mm)보다 넉넉히 아래이고, 실기 중력 처짐"
                             "(예측 53 mm)에 비하면 무시할 크기다. rad 로 문턱을 잡으면"
                             "그 숫자가 무엇을 뜻하는지 아무도 모른다.")
    # ↓ npz 가 fabric 파라미터를 담기 전에 뜬 기록을 위한 수동 지정. 새 기록에는 메타가
    #   들어 있으므로 주지 않아도 된다. **성공은 자기검증적이다** — dt·damping 이 틀리면
    #   궤적이 크게 갈리므로, 1e-4 이내로 맞았다는 것은 파라미터도 맞았다는 뜻이다.
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--home", type=str, default=None, help="쉼표로 구분한 7개 rad")
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--world", default=None)
    args = parser.parse_args()

    sim = dict(np.load(args.sim, allow_pickle=False))
    overrides = {"meta_fabric_dt": args.dt, "meta_fabric_decimation": args.decimation,
                 "meta_fabric_damping": args.damping, "meta_home_q": args.home,
                 "meta_fabric_robot_dir": args.robot_dir, "meta_fabric_world": args.world}
    missing = [k for k in REQUIRED_META if k not in sim and overrides.get(k) is None]
    if missing:
        raise SystemExit(
            f"{args.sim} 에 fabric 파라미터가 없다: {missing}\n"
            "  파라미터를 메타에 담는 버전의 probe_fab_shadow_record 로 다시 기록하거나,\n"
            "  --dt/--decimation/--damping/--home 으로 명시할 것. preset 에서 조용히\n"
            "  다시 읽지는 않는다 — 그 사이 소스가 바뀐 만큼 어긋난다."
        )
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "meta_home_q":
            sim[key] = np.array([float(x) for x in value.split(",")], dtype=np.float32)
        elif isinstance(value, str):
            sim[key] = np.array([value])
        else:
            sim[key] = np.array([value])
        print(f"[수동 지정] {key} = {sim[key]}")

    sys.path.insert(0, str(args.fabrics_src.expanduser().resolve()))
    import torch
    from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import (  # noqa: E402
        OpenArmGripperLeftPoseFabric,
    )
    from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
    from fabrics_sim.utils.utils import initialize_warp  # noqa: E402
    from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel  # noqa: E402
    import fabrics_sim

    device = args.device
    initialize_warp(str(device)[-1])

    dt = float(sim["meta_fabric_dt"][0])
    decimation = int(sim["meta_fabric_decimation"][0])
    damping_value = float(sim["meta_fabric_damping"][0])
    home = np.asarray(sim["meta_home_q"], dtype=np.float32).reshape(-1)
    robot_dir = str(sim["meta_fabric_robot_dir"][0])
    world_name = str(sim["meta_fabric_world"][0])

    print(f"기록  {args.sim.name}")
    print(f"  fabric  dt {dt} · decimation {decimation} · damping {damping_value}")
    print(f"  자산    {robot_dir} · world {world_name}")
    print(f"  Isaac   {str(sim['meta_fabrics'][0])}")
    print(f"  여기    {Path(fabrics_sim.__file__).resolve()}")

    world = WorldMeshesModel(batch_size=1, max_objects_per_env=8, device=device,
                             world_filename=world_name)
    object_ids, object_indicator = world.get_object_ids()
    fabric = OpenArmGripperLeftPoseFabric(
        1, device, dt, graph_capturable=False,
        robot_dir_name=robot_dir, robot_name=robot_dir,
        default_config_override=home.tolist(),
    )
    integrator = DisplacementIntegrator(fabric)

    cmd_pos = sim["palm_cmd_pos"][:, 0]
    cmd_quat_wxyz = sim["palm_cmd_quat_wxyz"][:, 0]
    isaac_q = sim["fabric_q"][:, 0]
    n = len(cmd_pos) if args.steps <= 0 else min(args.steps, len(cmd_pos))

    q = torch.tensor(home, device=device).unsqueeze(0).contiguous()
    qd = torch.zeros(1, 7, device=device)
    qdd = torch.zeros(1, 7, device=device)
    damping = damping_value * torch.ones(1, 1, device=device)
    pca_zeros = torch.zeros(1, 5, device=device)
    features = torch.zeros(1, 7, device=device)

    def palm_of(joints) -> np.ndarray:
        """fabric 자신의 FK 로 TCP 위치[m]. 별도 FK 를 세우면 그게 새 오차원이 된다."""
        tensor = torch.tensor(np.asarray(joints, dtype=np.float32).reshape(1, 7), device=device)
        return fabric.get_palm_pose(tensor, "quaternion").detach().cpu().numpy()[0, :3]

    deviation = np.zeros(n)
    tcp_deviation_mm = np.zeros(n)
    for step in range(n):
        wxyz = cmd_quat_wxyz[step]
        # set_features 의 쿼터니언 규약은 xyzw — 기록은 wxyz 다.
        features[0] = torch.tensor(
            np.concatenate([cmd_pos[step], wxyz[1:4], wxyz[:1]]),
            device=device, dtype=torch.float32)
        fabric.set_features(pca_zeros, features, "quaternion",
                            q.detach(), qd.detach(), object_ids, object_indicator, damping)
        for _ in range(decimation):
            q, qd, qdd = integrator.step(q.detach(), qd.detach(), qdd.detach(), dt)
        here = q[0].detach().cpu().numpy()
        deviation[step] = float(np.max(np.abs(here - isaac_q[step])))
        tcp_deviation_mm[step] = float(
            np.linalg.norm(palm_of(here) - palm_of(isaac_q[step])) * 1000.0)

    worst = int(np.argmax(tcp_deviation_mm))
    print(f"\n스텝 {n} · 배포 인터프리터 vs Isaac")
    print(f"  관절 최대편차   mean {deviation.mean():.3e}  p95 "
          f"{np.percentile(deviation, 95):.3e}  max {deviation.max():.3e}  rad")
    print(f"  TCP 편차        mean {tcp_deviation_mm.mean():.3f}  p95 "
          f"{np.percentile(tcp_deviation_mm, 95):.3f}  max {tcp_deviation_mm.max():.3f}  mm")
    print(f"  최악 스텝 {worst}")

    # 누적 발산인가, 유계인가 — 뒤쪽 1/4 이 앞쪽 1/4 보다 크게 나쁘면 발산이다.
    quarter = max(n // 4, 1)
    head, tail = tcp_deviation_mm[:quarter].mean(), tcp_deviation_mm[-quarter:].mean()
    trend = "유계(누적 발산 아님)" if tail <= head * 2.0 + 0.05 else "★누적 발산"
    print(f"  추세            앞 1/4 {head:.3f} mm → 뒤 1/4 {tail:.3f} mm — {trend}")

    if tcp_deviation_mm.max() <= args.tolerance_mm:
        print(f"\n✅ 배포 인터프리터의 Fabrics 가 Isaac 과 같은 해를 낸다 "
              f"(TCP 최대 {tcp_deviation_mm.max():.3f} mm ≤ 허용 {args.tolerance_mm:g} mm).")
        print("   실기에서 Fabrics 를 직접 굴려도 sim 이 계획한 자세로 간다.")
        return 0
    print(f"\n❌ 해가 갈린다 — TCP 최대 {tcp_deviation_mm.max():.3f} mm "
          f"(허용 {args.tolerance_mm:g}). 실기에서 Fabrics 를 돌리면 sim 이 계획한 자세와")
    print("   다른 곳으로 간다. 오류가 아니라 조용한 오프셋으로만 나타난다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
