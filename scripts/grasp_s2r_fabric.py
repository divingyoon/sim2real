#!/usr/bin/env python3
"""우팔 `grasp_s2r` 용 Fabrics 인스턴스 — FK(palm·손끝) + 팔 IK 적분.

`grasp_s2r_control._setup_fabrics` 를 배포 쪽에서 재현한다. 코어는 이 모듈이 만드는
콜러블 세 개(`palm_pose` · `tips` · `step`)만 받아 쓰므로 Isaac 무의존을 유지한다.

★★**관절 순서가 셋이다.** 09.03 sim 대조에서 여기 걸렸다:
    · 실기 드라이버 canonical  — 프로필 순(`r_hj_thumb_1..4, index_1..4, …`)
    · obs                       — Isaac DOF 순(`_1` 전 손가락 → `_2` 전 손가락 …)
    · **fabric**                — 또 다른 자기 순서(URDF 순)
  fabric 에 DOF 순을 그대로 넘기면 손끝이 thumb 148.1 · middle 145.7 · ring 31.8 mm
  어긋난다 — 그런데 index·pinky 는 **0.0 mm 로 정확히 일치**해서 일부만 보면
  "대체로 맞는다"고 넘어간다. 순서를 맞추면 5개 전부 0.0 mm 가 된다.
  그래서 이 모듈은 fabric 의 관절 이름을 읽어 **이름으로** 순열을 만든다.

★`fabric_q` 는 **영속 궤적생성기 상태**다. 매 tick 실측으로 재동기화하면 느린 실팔
  위치로 명령이 붕괴해 전진하지 못한다(08.03 실기 RUNNING 동결의 근본원인).
  단 **손 구간은 동기화한다** — env 도 그렇게 한다(`fabric_q[:, arm:] = syn_to_fab`).
  안 그러면 fabric 이 실재하지 않는 손으로 충돌구 FK 를 계산해, 없는 자기충돌을
  피하려 팔을 민다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HDGP = Path("/home/user/rl_ws/hdgp")

#: 프로필(`grasp_s2r/robot_profiles.TESOLLO_RIGHT`)에서 온 fabric 설정.
FABRIC_CLASS = "OpenArmTeoslloPoseFabric"
FABRIC_ROBOT_DIR = "openarm_tesollo_sensor_right"
FABRIC_PARAMS = "openarm_tesollo_sensor_pose_params.yaml"
NUM_ARM = 7
NUM_HAND = 20


def _fabric_class(name: str):
    import importlib
    mod = importlib.import_module("fabrics_sim.fabrics.openarm_tesollo_pose_fabric")
    return getattr(mod, name)


def make_right_fabric(*, home_q27, device: str = "cuda:0", dt: float,
                      damping: float, world_dict=None,
                      max_objects_per_env: int = 8):
    """FK/IK 콜러블 묶음을 만든다.

    Args:
        home_q27: fabric 순서의 홈 자세 27값 — cspace rest(널스페이스)와 초기 상태.
        dt: fabric 적분 dt. ★런의 `fabrics_dt` 를 쓸 것(리터럴 금지).
        damping: `fabrics_damping_gain`.

    Returns:
        `Fabric` 인스턴스(속성으로 `palm_pose`/`tips`/`step`/`hand_slice`/`q` 제공).
    """
    sys.path.insert(0, str(HDGP / "source/FABRICS/src"))
    import torch
    from fabrics_sim.integrator.integrators import DisplacementIntegrator
    from fabrics_sim.utils.utils import initialize_warp
    from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

    initialize_warp(str(device)[-1])
    world = WorldMeshesModel(batch_size=1, device=device,
                             max_objects_per_env=max_objects_per_env,
                             **({"world_dict": world_dict} if world_dict else {}))
    obj_ids, obj_ind = world.get_object_ids()

    fab = _fabric_class(FABRIC_CLASS)(
        batch_size=1, device=device, timestep=float(dt), graph_capturable=False,
        use_hand_fabric=False, tip_per_finger=False, hand_mode="pca",
        robot_dir_name=FABRIC_ROBOT_DIR, robot_name=FABRIC_ROBOT_DIR,
        fabric_params_filename=FABRIC_PARAMS,
    )
    if fab.num_joints != NUM_ARM + NUM_HAND:
        raise SystemExit(
            f"[s2r fabric] num_joints={fab.num_joints} != {NUM_ARM + NUM_HAND} — "
            "fabric URDF 와 자산이 어긋났다")
    integ = DisplacementIntegrator(fab)

    q0 = torch.tensor(np.asarray(home_q27, dtype=np.float32).reshape(1, -1),
                      device=device, dtype=torch.float32).contiguous()
    state = {"q": q0.clone(), "qd": torch.zeros(1, fab.num_joints, device=device),
             "qdd": torch.zeros(1, fab.num_joints, device=device)}
    fab.default_config.copy_(state["q"])
    pca = torch.zeros(1, 5, device=device)
    damp = float(damping) * torch.ones(1, 1, device=device)

    class Fabric:
        num_joints = fab.num_joints
        hand_slice = slice(NUM_ARM, NUM_ARM + NUM_HAND)

        @staticmethod
        def joint_names() -> list[str]:
            """fabric 자기 관절 순서 — 순열을 **이름으로** 만들기 위한 진실원천."""
            for attr in ("joint_names", "cspace_joint_names", "dof_names"):
                v = getattr(fab, attr, None)
                if v:
                    return [str(x) for x in (v() if callable(v) else v)]
            raise SystemExit("[s2r fabric] fabric 관절 이름을 못 읽었다 — 순열 불가")

        @staticmethod
        def palm_pose(q27) -> np.ndarray:
            t = torch.tensor(np.asarray(q27, dtype=np.float32).reshape(1, -1),
                             device=device, dtype=torch.float32)
            with torch.inference_mode():
                return fab.get_palm_pose(t, "euler_zyx")[0].cpu().numpy().astype(float)

        @staticmethod
        def tips(q27) -> np.ndarray:
            t = torch.tensor(np.asarray(q27, dtype=np.float32).reshape(1, -1),
                             device=device, dtype=torch.float32)
            with torch.inference_mode():
                return fab.get_fingertip_positions(t)[0].cpu().numpy().astype(float)

        @staticmethod
        def sync_hand(hand_fab_order) -> None:
            """fabric 의 **손 구간만** 실제 자세로 맞춘다(팔은 영속 상태 유지)."""
            h = torch.tensor(np.asarray(hand_fab_order, dtype=np.float32).reshape(1, -1),
                             device=device, dtype=torch.float32)
            state["q"][:, Fabric.hand_slice] = h

        @staticmethod
        def step(palm6, n: int = 1) -> np.ndarray:
            """★env 와 같은 순서: `set_features` 를 블록당 **한 번**, 그 뒤 n 번 적분."""
            feat = torch.tensor(np.asarray(palm6, dtype=np.float32).reshape(1, 6),
                                device=device, dtype=torch.float32)
            fab.set_features(pca, feat, "euler_zyx", state["q"].detach(),
                             state["qd"].detach(), obj_ids, obj_ind, damp)
            for _ in range(max(int(n), 1)):
                state["q"], state["qd"], state["qdd"] = integ.step(
                    state["q"].detach(), state["qd"].detach(),
                    state["qdd"].detach(), float(dt))
            return state["q"][0, :NUM_ARM].detach().cpu().numpy().astype(float)

        @staticmethod
        def q() -> np.ndarray:
            return state["q"][0].detach().cpu().numpy().astype(float)

        @staticmethod
        def qd() -> np.ndarray:
            """★팔 구간의 **목표 속도**. env 는 이걸 `set_joint_velocity_target` 으로
            sim PD 에 넘긴다 — 실기 JTC 에는 그 입구가 없어서 우리가 위치로 환산한다.
            (`right_inference_node.VEL_FF_RATIO` 주석에 그 대수가 있다.)"""
            return state["qd"][0, :NUM_ARM].detach().cpu().numpy().astype(float)

    return Fabric()


def permutation(src_names, dst_names) -> np.ndarray:
    """`dst[i] = src[perm[i]]` 가 되는 인덱스. 이름이 안 맞으면 즉시 죽는다."""
    src = list(src_names)
    missing = [n for n in dst_names if n not in src]
    if missing:
        raise SystemExit(f"[s2r fabric] 순열 실패 — 없는 관절 {missing}")
    return np.array([src.index(n) for n in dst_names], dtype=int)
