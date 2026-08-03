#!/usr/bin/env python3
"""grasp-v1 라이브 루프 오프라인 재현기 — RUNNING 팔 후퇴 원인 분리 실험 (ROS/실기 불필요).

grasp_inference.py 의 RUNNING 루프(관절→FK→obs114→LSTM→Δpalm→Fabrics IK→명령)를
그대로 복제하되, 실기 대신 **kinematic mock 로봇**으로 폐루프를 돌린다:

  팔:   arm_pos ← arm_pos + clip(cmd - arm_pos, ±max_vel/CONTROL_HZ)
        (= 브리지 velocity limiter + JTC 추종의 1차 근사. max_vel 을 크게 주면 sim 처럼 즉시 추종)
  손:   --hand-mode static  = APPROACH 동결 (실기 fake손 재현)
        --hand-mode zero    = 전관절 0 동결 (실손 하드웨어 두절 재현 — thumb _2 가 1.57rad 오차)
        --hand-mode echo    = hand_cmd 즉시 반영 (sim 처럼 진화하는 손 obs)
  접촉: 항상 0 (Stage A 동일 — lift 래치 미발동, 접근 거동만 판정)
  지연: --obs-delay K = 팔 관측이 K 제어틱(1/60s) 늦게 도착 (DDS/파이프라인 지연 재현)

2×2 (hand-mode × max-vel) 로 돌리면 어느 인자가 palm→cup 후퇴를 일으키는지 분리된다:

    python3 grasp_loop_sim.py --agent ... --ckpt ... --max-vel 0.1 --hand-mode static
    python3 grasp_loop_sim.py --agent ... --ckpt ... --max-vel 99  --hand-mode echo

★ 루프 로직은 grasp_inference.py `_policy_loop` 와 1:1 유지할 것 (drift 시 실험 무효).
실행 환경: torch+warp+rl_games (로컬은 IsaacLab 번들 python: isaaclab.sh -p 로 실행).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo" / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break
_OPENARM_SRC = _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "openarm"
sys.path.insert(0, str(_OPENARM_SRC))
sys.path.insert(0, str(_SCRIPT_DIR))

import torch

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from policy_loader import RLGamesLstmActorPolicy
from grasp_obs_builder import ACTOR_OBS_DIM, assemble_actor_obs, make_object_onehot
from grasp_action_decoder import (
    DEFAULT_FINGER_CLOSE_SPEED,
    DEFAULT_GRASP_READY_HOLD_STEPS,
    DEFAULT_LIFT_MIN_GRIP_FINGERS,
    GraspFingerController,
    LiftLatch,
    scale_palm_delta,
)

import importlib as _il

_preset = _il.import_module("openarm.tesollo.right.grasp_v1.grasp_right_preset")
_consts = _il.import_module("openarm.tesollo.right.grasp_v1.grasp_right_constants")

HAND_APPROACH_POSE = _preset.HAND_APPROACH_POSE
HAND_FULL_GRIP_POSE = _preset.HAND_FULL_GRIP_POSE
RIGHT_ARM_START_POSE = _preset.RIGHT_ARM_START_POSE
PREGRASP_OFFSET = _preset.PREGRASP_OFFSET
palm_pose_mins = _preset.palm_pose_mins
palm_pose_maxs = _preset.palm_pose_maxs

NUM_ARM_DOF = _consts.NUM_ARM_DOF
PREGRASP_FABRICS_STEPS = _consts.PREGRASP_FABRICS_STEPS
EPISODE_STEPS = _consts.EPISODE_STEPS

# grasp_inference.py 상수와 동일 (출처: grasp_v1 env_cfg)
PALM_DELTA_XYZ = 0.15
PALM_DELTA_ROT_DEG = 20.0
MAX_POSE_ANGLE = 45.0
FABRIC_DECIMATION = 2
FABRICS_DT = 1.0 / 60.0
FABRICS_DAMPING = 20.0
CONTROL_HZ = 60.0
PREGRASP_ORI = [math.radians(90.0), math.radians(0.0), math.radians(90.0)]


def _t(vals, device: str) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32, device=device)


def run(args: argparse.Namespace) -> None:
    device = args.device
    cup_pos = np.array([args.cup_x, args.cup_y, args.cup_z])

    policy = RLGamesLstmActorPolicy(
        agent_yaml_path=args.agent, checkpoint_path=args.ckpt,
        obs_dim=ACTOR_OBS_DIM, action_dim=11, device=device,
    )
    object_onehot = make_object_onehot(args.object)

    initialize_warp("0")
    world_model = WorldMeshesModel(
        batch_size=1, max_objects_per_env=8, device=device,
        world_filename="open_tesollo_boxes_no_table",
    )
    object_ids, object_indicator = world_model.get_object_ids()
    fabric = OpenArmTeoslloPoseFabric(
        batch_size=1, device=device, timestep=FABRICS_DT,
        graph_capturable=False, use_hand_fabric=False,
    )
    integrator = DisplacementIntegrator(fabric)

    cspace_def = fabric.default_config.clone()
    cspace_def[0, NUM_ARM_DOF:] = _t(HAND_FULL_GRIP_POSE, device)
    fabric.default_config.copy_(cspace_def)

    _dr = math.radians(PALM_DELTA_ROT_DEG)
    delta_mins = np.array([-PALM_DELTA_XYZ] * 3 + [-_dr] * 3)
    delta_maxs = np.array([PALM_DELTA_XYZ] * 3 + [_dr] * 3)
    palm_mins = np.array(palm_pose_mins(MAX_POSE_ANGLE), dtype=np.float64)
    palm_maxs = np.array(palm_pose_maxs(MAX_POSE_ANGLE), dtype=np.float64)
    damping_gain = FABRICS_DAMPING * torch.ones(1, 1, device=device)

    finger_ctrl = GraspFingerController(
        hand_open=np.array(HAND_APPROACH_POSE, dtype=np.float64),
        hand_full_grip=np.array(HAND_FULL_GRIP_POSE, dtype=np.float64),
        close_speed=DEFAULT_FINGER_CLOSE_SPEED,
    )
    lift_latch = LiftLatch(
        min_contacts=DEFAULT_LIFT_MIN_GRIP_FINGERS,
        hold_steps=DEFAULT_GRASP_READY_HOLD_STEPS,
    )

    # ── Pregrasp IK rollout (grasp_inference._compute_pregrasp 동일) ─────────
    pregrasp_pos = cup_pos + np.array(PREGRASP_OFFSET)
    pose6 = np.concatenate([pregrasp_pos, np.array(PREGRASP_ORI)])
    pregrasp_palm_pose = np.clip(pose6, palm_mins, palm_maxs)

    fabric_q = torch.cat([
        _t(RIGHT_ARM_START_POSE, device), _t(HAND_APPROACH_POSE, device)
    ]).unsqueeze(0)
    fabric_qd = torch.zeros(1, 27, device=device)
    fabric_qdd = torch.zeros(1, 27, device=device)
    palm_tgt = _t(pregrasp_palm_pose, device).unsqueeze(0)
    pca_zero = torch.zeros(1, 5, device=device)
    for _ in range(PREGRASP_FABRICS_STEPS):
        fabric.set_features(
            pca_zero, palm_tgt, "euler_zyx", fabric_q.detach(), fabric_qd.detach(),
            object_ids, object_indicator, damping_gain,
        )
        for _ in range(FABRIC_DECIMATION):
            fabric_q, fabric_qd, fabric_qdd = integrator.step(
                fabric_q.detach(), fabric_qd.detach(), fabric_qdd.detach(), FABRICS_DT
            )
    pregrasp_arm_pos = fabric_q[0, :NUM_ARM_DOF].cpu().numpy()

    cspace_def = fabric.default_config.clone()
    cspace_def[0, :NUM_ARM_DOF] = _t(pregrasp_arm_pos, device)
    fabric.default_config.copy_(cspace_def)

    # ── Mock 로봇 상태: settle 완료 가정(팔=pregrasp, 손=APPROACH, 정지) ────
    arm_pos = pregrasp_arm_pos.copy()
    arm_vel = np.zeros(NUM_ARM_DOF)
    hand_pos = (
        np.zeros(20) if args.hand_mode == "zero"
        else np.array(HAND_APPROACH_POSE, dtype=np.float64)
    )
    hand_vel = np.zeros(20)
    from collections import deque
    obs_delay_buf: deque = deque(maxlen=max(1, args.obs_delay + 1))
    tip_contact = np.zeros(5)
    last_actions = np.zeros(11)
    policy.reset_states()
    finger_ctrl.reset()
    lift_latch.reset()

    fabric_q[0, :NUM_ARM_DOF] = _t(arm_pos, device)
    fabric_q[0, NUM_ARM_DOF:] = _t(hand_pos, device)
    fabric_qd.zero_()
    fabric_qdd.zero_()

    max_step = args.max_vel / CONTROL_HZ   # rad per control tick
    dist_log: list[float] = []

    print(f"[loop_sim] cup={cup_pos.tolist()} max_vel={args.max_vel} hand={args.hand_mode}")
    for step in range(args.steps):
        # obs 지연: 관측되는 팔 상태 = obs_delay 틱 전 값 (명령은 현재 팔에 적용)
        obs_delay_buf.append((arm_pos.copy(), arm_vel.copy()))
        obs_arm_pos, obs_arm_vel = obs_delay_buf[0]

        # 1. fabric_q 실제 관절 동기화 (grasp_inference._policy_loop §1)
        fabric_q[0, :NUM_ARM_DOF] = _t(obs_arm_pos, device)
        fabric_q[0, NUM_ARM_DOF:] = _t(hand_pos, device)
        fabric_qd[0, :NUM_ARM_DOF] = _t(obs_arm_vel, device)
        fabric_qd[0, NUM_ARM_DOF:] = _t(hand_vel, device)

        # 2. FK (§2)
        with torch.inference_mode():
            palm_pose_6d = fabric.get_palm_pose(fabric_q, "euler_zyx")
            fingertip_pos = fabric.get_fingertip_positions(fabric_q)
        palm_center = palm_pose_6d[0, :3].cpu().numpy()
        tips = fingertip_pos[0].cpu().numpy()

        dist = float(np.linalg.norm(palm_center - cup_pos))
        dist_log.append(dist)
        if step % 30 == 0:
            print(f"[loop_sim] step={step:4d} dist={dist:.3f} palm={palm_center.round(3).tolist()}")

        # 3~4. obs → policy (§3~4)
        obs_np = assemble_actor_obs(
            arm_joint_pos=obs_arm_pos, arm_joint_vel=obs_arm_vel,
            finger_joint_pos=hand_pos, finger_joint_vel=hand_vel,
            palm_center=palm_center, fingertip_pos=tips, cup_pos=cup_pos,
            binary_contact=tip_contact, last_actions=last_actions,
            object_onehot=object_onehot,
        )
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy.get_action(obs)[0].cpu().numpy()
        last_actions = action.copy()
        palm_action = action[:6]
        finger_action = action[6:11]

        # 5~6. lift 래치 + 손가락 (§5~6; 접촉 0이라 래치 미발동)
        lift_latch.update(int(tip_contact.sum()))
        hand_cmd = finger_ctrl.step(finger_action, tip_contact)

        # 7. 팔 grasp phase (§7; lift 미진입 전제)
        delta = scale_palm_delta(palm_action, delta_mins, delta_maxs)
        palm_pose = pregrasp_palm_pose + delta
        palm_pose = np.clip(
            palm_pose,
            np.minimum(palm_mins, pregrasp_palm_pose),
            np.maximum(palm_maxs, pregrasp_palm_pose),
        )
        fabric.set_features(
            torch.zeros(1, 5, device=device), _t(palm_pose, device).unsqueeze(0),
            "euler_zyx", fabric_q.detach(), fabric_qd.detach(),
            object_ids, object_indicator, damping_gain,
        )
        for _ in range(FABRIC_DECIMATION):
            fabric_q, fabric_qd, fabric_qdd = integrator.step(
                fabric_q.detach(), fabric_qd.detach(), fabric_qdd.detach(), FABRICS_DT
            )
        arm_cmd = fabric_q[0, :NUM_ARM_DOF].cpu().numpy()

        # 8. mock 로봇 갱신 (브리지 velocity limiter 1차 근사)
        step_arm = np.clip(arm_cmd - arm_pos, -max_step, max_step)
        arm_vel = step_arm * CONTROL_HZ
        arm_pos = arm_pos + step_arm
        if args.hand_mode == "echo":
            hand_vel = (hand_cmd - hand_pos) * CONTROL_HZ
            hand_pos = hand_cmd.copy()
        # static: hand_pos/vel 불변 (APPROACH 동결)

    d0, dmin, dend = dist_log[0], min(dist_log), dist_log[-1]
    verdict = "후퇴(발산)" if dend > d0 + 0.03 else ("접근" if dend < d0 - 0.03 else "정체")
    print(
        f"[loop_sim] === 결과: hand={args.hand_mode} max_vel={args.max_vel} "
        f"obs_delay={args.obs_delay} cup={cup_pos.tolist()} "
        f"dist {d0:.3f} → {dend:.3f} (min {dmin:.3f}) → {verdict}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--object", default="cup_big_s100")
    ap.add_argument("--max-vel", type=float, default=0.1,
                    help="브리지 관절 속도제한 [rad/s] (실기 Stage A=0.1, 99=즉시추종)")
    ap.add_argument("--hand-mode", choices=["static", "zero", "echo"], default="static",
                    help="static=APPROACH 동결(fake손), zero=전관절 0(실손 두절), echo=hand_cmd 반영(sim)")
    ap.add_argument("--obs-delay", type=int, default=0,
                    help="팔 관측 지연 [제어틱=1/60s] (DDS/파이프라인 지연 재현, 6≈100ms)")
    ap.add_argument("--steps", type=int, default=EPISODE_STEPS)
    ap.add_argument("--cup-x", type=float, default=0.40)
    ap.add_argument("--cup-y", type=float, default=-0.15)
    ap.add_argument("--cup-z", type=float, default=0.38)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
