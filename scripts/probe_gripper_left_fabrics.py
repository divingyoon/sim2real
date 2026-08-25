#!/usr/bin/env python3
"""좌 그리퍼 Fabrics 를 **Isaac 앱 없이** 단독으로 굴려 IK 자체를 검증한다.

왜 이게 첫 게이트인가. "Fabrics IK 가 제대로 도는가"는 세 층으로 갈린다:

    L1  FK(fabric_q) vs 지령 palm pose     ← attractor 가 목표에 수렴하나  (여기)
    L2  sim 물리 TCP vs FK(fabric_q)        ← sim PD 가 fabric 해를 따라가나
    L3  실기 measured vs fabric_q           ← 실팔이 그 관절 목표를 따라가나

L1 은 로봇도 Isaac 도 필요 없다 — `fabrics_sim` 은 warp 만으로 돈다. 그러니 여기서
먼저 답을 내고, L1 이 이미 크면 그 위층 측정은 해석이 불가능하다는 것을 알고 시작한다.

검사하는 것
  ①  cspace 가 7 DOF 인가 (손 20관절이 fixed 로 동결됐는가)
  ②  cspace rest 가 **태스크 홈**인가 — 내장 기본값은 폐기된 ABORTED 트랙 홈이고
      j7 이 +1.3563 대 −0.3306 으로 전혀 다르다. 넘기지 않으면 조용히 그쪽으로 당긴다.
  ③  a=0 수렴 — 절대 규약이므로 팔은 홈에서 박스 중심으로 스스로 가야 한다. 남는 잔차가
      L1 의 바닥이다.
  ④  PALM_BOX 꼭짓점 도달성 — 정책이 지시할 수 있는 극단에 못 가는 곳이 있는지 먼저 안다.
      여기서 크게 벗어나는 꼭짓점은 제어 실패가 아니라 워크스페이스 사실이다.

두 소스 트리를 대조하려면 `--fabrics-src` 를 바꿔 두 번 돌리고 json 을 비교한다.
학습이 돈 트리와 지금 트리가 다른데도 같은 IK 인지는 **추론이 아니라 이걸로** 답한다.

실행:
    source /opt/ros/humble/setup.bash && . ~/rl_ws/sim2real/.venv/bin/activate
    python3 scripts/probe_gripper_left_fabrics.py --out /tmp/l1_head.json
    python3 scripts/probe_gripper_left_fabrics.py \
        --fabrics-src <다른 트리>/source/FABRICS/src --out /tmp/l1_train.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

DEFAULT_HDGP = Path.home() / "rl_ws/hdgp"


def load_preset(hdgp_root: Path):
    path = hdgp_root / "source/openarm/openarm/gripper/left/grasp_sensor/grasp_left_preset.py"
    if not path.is_file():
        raise SystemExit(f"preset 을 못 찾았다: {path}")
    spec = importlib.util.spec_from_file_location("_grasp_left_preset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """두 wxyz 쿼터니언 사이 각[rad]. 부호 모호성은 abs 로 접는다."""
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdgp", type=Path, default=DEFAULT_HDGP,
                        help="preset·자산을 읽을 hdgp 체크아웃")
    parser.add_argument("--fabrics-src", type=Path, default=None,
                        help="fabrics_sim 소스 트리 (기본: --hdgp 아래 source/FABRICS/src)")
    parser.add_argument("--settle", type=int, default=400,
                        help="목표 하나당 수렴 대기 스텝 (env step 기준)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None, help="결과 json 경로")
    args = parser.parse_args()

    fabrics_src = args.fabrics_src or (args.hdgp / "source/FABRICS/src")
    sys.path.insert(0, str(fabrics_src))

    preset = load_preset(args.hdgp)

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

    home = [preset.LEFT_ARM_HOME_JOINT_POS[f"l_aj_{i}"] for i in range(1, 8)]
    fabric_dt = 1.0 / 60.0 / float(preset.FABRIC_DECIMATION)

    world = WorldMeshesModel(batch_size=1, max_objects_per_env=8, device=device,
                             world_filename=preset.FABRIC_WORLD_FILENAME)
    object_ids, object_indicator = world.get_object_ids()

    fabric = OpenArmGripperLeftPoseFabric(
        1, device, fabric_dt, graph_capturable=False,
        robot_dir_name=preset.FABRIC_ROBOT_DIR, robot_name=preset.FABRIC_ROBOT_DIR,
        default_config_override=home,
    )
    integrator = DisplacementIntegrator(fabric)

    report: dict = {
        "fabrics_src": str(fabrics_src),
        "fabrics_module": str(Path(fabrics_sim.__file__).resolve()),
        "hdgp": str(args.hdgp),
        "robot_dir": preset.FABRIC_ROBOT_DIR,
        "world": preset.FABRIC_WORLD_FILENAME,
        "palm_box": {"x": list(preset.PALM_BOX_X), "y": list(preset.PALM_BOX_Y),
                     "z": list(preset.PALM_BOX_Z)},
        "home": home,
        "num_joints": int(fabric.num_joints),
        "settle_steps": args.settle,
    }

    # ① cspace 차원
    report["cspace_is_arm_only"] = int(fabric.num_joints) == 7

    # ② rest 가 태스크 홈인가 — 내장 기본값과 다른지 직접 확인한다
    from fabrics_sim.fabrics import openarm_tesollo_pose_fabric as _mod
    builtin = list(getattr(_mod, "_GRIPPER_LEFT_DEFAULT_CONFIG", []))
    report["builtin_default_config"] = builtin
    report["home_differs_from_builtin_max_rad"] = (
        float(np.max(np.abs(np.array(home) - np.array(builtin)))) if builtin else None
    )

    damping = preset.FABRIC_DAMPING_GAIN * torch.ones(1, 1, device=device)
    pca_zeros = torch.zeros(1, 5, device=device)

    box_low = np.array([preset.PALM_BOX_X[0], preset.PALM_BOX_Y[0], preset.PALM_BOX_Z[0]])
    box_high = np.array([preset.PALM_BOX_X[1], preset.PALM_BOX_Y[1], preset.PALM_BOX_Z[1]])
    box_center, box_half = 0.5 * (box_low + box_high), 0.5 * (box_high - box_low)
    ref_quat_wxyz = np.array(preset.PALM_REF_QUAT_WXYZ)

    def run_to(target_action6, steps):
        """홈에서 출발해 목표를 `steps` 만큼 유지하고 (오차, 최종 관절)을 돌려준다.

        ★여기서 지령하는 것은 **정지 목표**뿐이다(박스 중심과 꼭짓점). 그래서 액션 항의
          변화율 상한·회전 규약을 흉내 낼 필요가 없다 — 그걸 복제하면 sim 이 규약을 바꿀
          때마다 이 프로브가 조용히 옛 계약을 재게 된다(08.25 에 실제로 그렇게 갈렸다).
        """
        q = torch.tensor(home, device=device, dtype=torch.float32).unsqueeze(0).contiguous()
        qd = torch.zeros(1, 7, device=device)
        qdd = torch.zeros(1, 7, device=device)
        features = torch.zeros(1, 7, device=device)
        action = np.asarray(target_action6, dtype=float)
        pos = box_center + np.clip(action[:3], -1.0, 1.0) * box_half
        quat = ref_quat_wxyz                      # 회전은 기준 자세 고정
        features[0] = torch.tensor(
            np.concatenate([pos, quat[1:4], quat[:1]]),   # set_features 규약 = xyzw
            device=device, dtype=torch.float32)
        for _ in range(steps):
            fabric.set_features(pca_zeros, features, "quaternion",
                                q.detach(), qd.detach(), object_ids, object_indicator, damping)
            for _ in range(int(preset.FABRIC_DECIMATION)):
                q, qd, qdd = integrator.step(q.detach(), qd.detach(), qdd.detach(), fabric_dt)
        palm_pos, palm_quat = palm_pose(q)
        return {
            "cmd_pos": [round(v, 6) for v in pos],
            "fk_pos": [round(v, 6) for v in palm_pos],
            "pos_err_mm": round(float(np.linalg.norm(palm_pos - pos)) * 1000.0, 3),
            "rot_err_deg": round(np.degrees(quat_angle_between(palm_quat, quat)), 3),
            "q": [round(v, 6) for v in q[0].detach().cpu().numpy().tolist()],
        }

    def palm_pose(q):
        """fabric 자신의 palm FK. 별도 FK 를 세우면 그게 새 오차원이 된다.

        `get_palm_pose` 는 attractor 가 쓰는 바로 그 taskmap 을 통과하므로, 여기서
        나오는 오차는 순수하게 "attractor 가 목표에 얼마나 갔나"다. 반환 쿼터니언은
        **xyzw** 이므로 wxyz 로 되돌려 비교한다.
        """
        pose = fabric.get_palm_pose(q, "quaternion").detach().cpu().numpy().reshape(-1)
        pos = pose[:3].astype(float)
        xyzw = pose[3:7].astype(float)
        return pos, np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    # ③ a = 0
    report["zero_action"] = run_to(np.zeros(6), args.settle)

    # ④ PALM_BOX 꼭짓점 (회전은 기준 자세 유지)
    corners = {}
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                key = f"x{'+' if sx > 0 else '-'}y{'+' if sy > 0 else '-'}z{'+' if sz > 0 else '-'}"
                corners[key] = run_to([sx, sy, sz, 0.0, 0.0, 0.0], args.settle)
    report["corners"] = corners
    errs = [c["pos_err_mm"] for c in corners.values()]
    report["corner_pos_err_mm"] = {
        "min": round(min(errs), 3), "max": round(max(errs), 3),
        "mean": round(float(np.mean(errs)), 3),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\n-> {args.out}", file=sys.stderr)

    if not report["cspace_is_arm_only"]:
        print("BLOCK: cspace 가 7 DOF 가 아니다", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
