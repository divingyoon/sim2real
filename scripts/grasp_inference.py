#!/usr/bin/env python3
"""grasp-v1 라이브 sim2real inference 노드 (tesollo/right grasp-v1 LSTM, obs 114D).

기준 체크포인트:
    hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test3/nn/
        last_open-tesol_r_grasp_v1-lstm_ep_20000_rew_9920.256.pth
    (+ 같은 폴더 params/agent.yaml, LSTM 1024)

실행:
    python3 grasp_inference.py \
        --agent /path/to/agent.yaml --ckpt /path/to/...ep_20000....pth \
        [--device cuda:0] [--settle_time 4.0] [--object cup_big_s100]

    ros2 service call /grasp/start std_srvs/srv/Trigger

구독:
    /joint_states               arm 7D  (canonical r_aj_1..7)
    /dg5f_right/joint_states    hand 20D (canonical r_hj_*)
    /cup_pose                   PoseStamped — 컵 위치 (robot base)
    /dg5f_right/contact_forces  Float64MultiArray — fingertip(tip) 5D [N]

발행: /isaacsim/right_arm_cmd (7D rad), /isaacsim/right_hand_cmd (20D rad)
      → 브리지(isaacsim_cmd_to_jtc)가 robot_control JTC 로 변환 (Phase 3)

env `grasp_right_env.py` 재현:
    - obs 114D  = grasp_obs_builder (base106 + object onehot8)
    - action 11D = grasp_action_decoder:
        · palm[:6] → Δpalm → Fabrics IK arm
        · finger[6:11] → 접촉-게이트 stateful 폐쇄 (GraspFingerController)
    - lift 진입 = 접촉 래치(LiftLatch, ≥3손가락 grip 8스텝 hold), joint7-only lift-wait
    - 정책 60Hz, fabric_decimation=2

★ tip-only 제어: env 의 손가락 폐쇄 게이트·lift 래치는 (tip|mid|distal) 을 쓰지만,
  middle/distal 은 **critic 전용(privileged)** 으로 학습돼 실기서 감지 불가하다. 따라서
  라이브 제어는 **tip 접촉만** 사용한다(디코더 distal 입력 미배선). rigid 컵에선 tip 미접촉
  구간에 _3/_4 가 FULL_GRIP 까지 감겨 더 단단히 잡힌다(허용). distal 은 sim parity 검증용.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

# ── Fabrics 경로 ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo" / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

# ── Task 경로 (grasp_v1 preset/constants; Isaac 무의존) ───────────────────────
_OPENARM_SRC = _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "openarm"
sys.path.insert(0, str(_OPENARM_SRC))

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

sys.path.insert(0, str(_SCRIPT_DIR))
from fabrics_ros_interface import create_publisher
from policy_loader import RLGamesActorPolicy
from grasp_obs_builder import (
    ACTOR_OBS_DIM,
    REAL_CUP_INDEX,
    assemble_actor_obs,
    make_object_onehot,
)
from grasp_action_decoder import (
    DEFAULT_FINGER_CLOSE_SPEED,
    DEFAULT_GRASP_READY_HOLD_STEPS,
    DEFAULT_LIFT_MIN_GRIP_FINGERS,
    DEFAULT_LIFT_WAIT_JOINT7_DELTA,
    DEFAULT_WARM_J7_MAX,
    DEFAULT_WARM_J7_MIN,
    GraspFingerController,
    LiftLatch,
    joint7_lift_wait_target,
    lift_arm_interp,
    scale_palm_delta,
)

import importlib as _il

_preset = _il.import_module("openarm.tesollo.right.grasp_v1.grasp_right_preset")
RIGHT_ARM_JOINT_NAMES = _preset.RIGHT_ARM_JOINT_NAMES
RIGHT_HAND_JOINT_NAMES = _preset.RIGHT_HAND_JOINT_NAMES
HAND_APPROACH_POSE = _preset.HAND_APPROACH_POSE
HAND_FULL_GRIP_POSE = _preset.HAND_FULL_GRIP_POSE
RIGHT_ARM_START_POSE = _preset.RIGHT_ARM_START_POSE
PREGRASP_OFFSET = _preset.PREGRASP_OFFSET
palm_pose_mins = _preset.palm_pose_mins
palm_pose_maxs = _preset.palm_pose_maxs

_consts = _il.import_module("openarm.tesollo.right.grasp_v1.grasp_right_constants")
NUM_ARM_DOF = _consts.NUM_ARM_DOF
NUM_HAND_DOF = _consts.NUM_HAND_DOF
LIFT_PHASE_STEPS = _consts.LIFT_PHASE_STEPS
PREGRASP_FABRICS_STEPS = _consts.PREGRASP_FABRICS_STEPS
EPISODE_STEPS = _consts.EPISODE_STEPS
CONTACT_FORCE_THRESHOLD = _consts.CONTACT_FORCE_THRESHOLD

# ---------------------------------------------------------------------------
# 상수 (grasp_v1 env_cfg 기본값 — Isaac 무의존이라 여기 반영, 출처 주석)
# ---------------------------------------------------------------------------
PALM_DELTA_XYZ = 0.15          # env_cfg palm_delta_xyz
PALM_DELTA_ROT_DEG = 20.0      # env_cfg palm_delta_rot_deg
MAX_POSE_ANGLE = 45.0          # env_cfg max_pose_angle
FABRIC_DECIMATION = 2          # env_cfg fabric_decimation
FABRICS_DT = 1.0 / 60.0
FABRICS_DAMPING = 20.0
CONTROL_HZ = 60.0

# pregrasp 방향: ez=90°, ey=0°, ex=90°
PREGRASP_ORI = [math.radians(90.0), math.radians(0.0), math.radians(90.0)]
APPROACH_CMD_HZ = 10.0


class State(Enum):
    IDLE = auto()
    APPROACHING = auto()
    RUNNING = auto()
    DONE = auto()


def _t(vals, device: str) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32, device=device)


class GraspInferenceNode(Node):

    def __init__(
        self,
        agent_yaml: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        settle_time: float = 4.0,
        object_name: str | int = REAL_CUP_INDEX,
    ) -> None:
        super().__init__("grasp_inference")
        self.device = device
        self.settle_time = settle_time

        # ── Policy ───────────────────────────────────────────────────────────
        self.get_logger().info("Policy 로드 중...")
        self.policy = RLGamesActorPolicy(
            agent_yaml_path=agent_yaml,
            checkpoint_path=checkpoint_path,
            obs_dim=ACTOR_OBS_DIM,   # 114
            action_dim=11,
            device=device,
        )

        # 잡는 물체 onehot (라이브 고정). 기본 cup_big_s100(index 1).
        self.object_onehot = make_object_onehot(object_name)
        self.get_logger().info(f"물체 onehot 고정: {object_name} → {self.object_onehot.tolist()}")

        # ── Fabrics ──────────────────────────────────────────────────────────
        self.get_logger().info("Fabrics 초기화 중...")
        initialize_warp("0")
        self.world_model = WorldMeshesModel(
            batch_size=1,
            max_objects_per_env=6,
            device=device,
            world_filename="open_tesollo_boxes_no_table",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()
        self.fabric = OpenArmTeoslloPoseFabric(
            batch_size=1,
            device=device,
            timestep=FABRICS_DT,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        self.integrator = DisplacementIntegrator(self.fabric)

        # cspace default hand = FULL_GRIP 방향 (arm IK null-space)
        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, NUM_ARM_DOF:] = _t(HAND_FULL_GRIP_POSE, device)
        self.fabric.default_config.copy_(cspace_def)

        # ── 파라미터 텐서 (팔 delta / palm workspace) ─────────────────────────
        _dr = math.radians(PALM_DELTA_ROT_DEG)
        self.delta_mins = np.array([-PALM_DELTA_XYZ] * 3 + [-_dr] * 3)
        self.delta_maxs = np.array([PALM_DELTA_XYZ] * 3 + [_dr] * 3)
        self.palm_mins = np.array(palm_pose_mins(MAX_POSE_ANGLE), dtype=np.float64)
        self.palm_maxs = np.array(palm_pose_maxs(MAX_POSE_ANGLE), dtype=np.float64)
        self.damping_gain = FABRICS_DAMPING * torch.ones(1, 1, device=device)

        # ── grasp action 디코더 (stateful) ────────────────────────────────────
        self.finger_ctrl = GraspFingerController(
            hand_open=np.array(HAND_APPROACH_POSE, dtype=np.float64),
            hand_full_grip=np.array(HAND_FULL_GRIP_POSE, dtype=np.float64),
            close_speed=DEFAULT_FINGER_CLOSE_SPEED,
        )
        self.lift_latch = LiftLatch(
            min_contacts=DEFAULT_LIFT_MIN_GRIP_FINGERS,
            hold_steps=DEFAULT_GRASP_READY_HOLD_STEPS,
        )

        # fabric 상태
        q0 = torch.cat([_t(RIGHT_ARM_START_POSE, device),
                        _t(HAND_APPROACH_POSE, device)]).unsqueeze(0)
        self.fabric_q = q0.clone()
        self.fabric_qd = torch.zeros(1, 27, device=device)
        self.fabric_qdd = torch.zeros(1, 27, device=device)

        # ── 센서 버퍼 ────────────────────────────────────────────────────────
        self.arm_pos = np.zeros(7)
        self.arm_vel = np.zeros(7)
        self.hand_pos = np.zeros(20)
        self.hand_vel = np.zeros(20)
        self.cup_pos = np.zeros(3)
        self.tip_contact = np.zeros(5)   # fingertip F/T 이진 (제어에 쓰는 유일한 접촉)

        self._arm_ready = False
        self._hand_ready = False
        self._cup_ready = False

        self._arm_idx = {n: i for i, n in enumerate(RIGHT_ARM_JOINT_NAMES)}
        self._hand_idx = {n: i for i, n in enumerate(RIGHT_HAND_JOINT_NAMES)}

        # ── 에피소드 상태 ────────────────────────────────────────────────────
        self.state = State.IDLE
        self.step_count = 0
        self.last_actions = np.zeros(11)

        self.pregrasp_palm_pose = np.zeros(6)
        self.pregrasp_arm_pos = np.zeros(7)

        # lift 캡처 버퍼
        self.lift_arm_start = None
        self.prelift_target = None
        self.lift_start_step = None

        self._approach_start_time = 0.0

        # ── ROS2 ─────────────────────────────────────────────────────────────
        self.create_subscription(JointState, "/joint_states", self._arm_cb, 10)
        self.create_subscription(JointState, "/dg5f_right/joint_states", self._hand_cb, 10)
        self.create_subscription(PoseStamped, "/cup_pose", self._cup_cb, 10)
        self.create_subscription(Float64MultiArray, "/dg5f_right/contact_forces", self._tip_cb, 10)

        self.cmd_pub = create_publisher()

        self.create_service(Trigger, "/grasp/start", self._start_cb)
        self.create_service(Trigger, "/grasp/stop", self._stop_cb)
        self.create_service(Trigger, "/grasp/reset", self._reset_cb)

        self.create_timer(1.0 / APPROACH_CMD_HZ, self._approach_loop)
        self.create_timer(1.0 / CONTROL_HZ, self._policy_loop)

        self.get_logger().info("준비 완료. '/grasp/start' 서비스 호출 시 에피소드 시작.")

    # ------------------------------------------------------------------
    # 센서 Callbacks
    # ------------------------------------------------------------------
    def _arm_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in self._arm_idx:
                idx = self._arm_idx[name]
                self.arm_pos[idx] = msg.position[i]
                if msg.velocity:
                    self.arm_vel[idx] = msg.velocity[i]
        self._arm_ready = True

    def _hand_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in self._hand_idx:
                idx = self._hand_idx[name]
                self.hand_pos[idx] = msg.position[i]
                if msg.velocity:
                    self.hand_vel[idx] = msg.velocity[i]
        self._hand_ready = True

    def _cup_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.cup_pos[:] = [p.x, p.y, p.z]
        self._cup_ready = True

    def _binary(self, data) -> np.ndarray:
        f = np.asarray(list(data[:5]), dtype=np.float64)
        if f.shape[0] < 5:
            return np.zeros(5)
        return (f > CONTACT_FORCE_THRESHOLD).astype(np.float64)

    def _tip_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 5:
            self.tip_contact = self._binary(msg.data)

    # ------------------------------------------------------------------
    # 서비스 Callbacks
    # ------------------------------------------------------------------
    def _start_cb(self, request, response):
        if self.state not in (State.IDLE, State.DONE):
            response.success = False
            response.message = f"ERROR: 현재 상태={self.state.name}, IDLE/DONE 에서만 start"
            return response
        if not (self._arm_ready and self._hand_ready and self._cup_ready):
            missing = []
            if not self._arm_ready:  missing.append("/joint_states")
            if not self._hand_ready: missing.append("/dg5f_right/joint_states")
            if not self._cup_ready:  missing.append("/cup_pose")
            response.success = False
            response.message = f"ERROR: 미수신 토픽: {missing}"
            self.get_logger().error(response.message)
            return response

        self._compute_pregrasp()
        self._reset_episode_state()
        self.state = State.APPROACHING
        self._approach_start_time = time.monotonic()
        response.success = True
        response.message = (
            f"APPROACHING 시작 (settle={self.settle_time}s). "
            f"pregrasp_arm={[f'{v:.3f}' for v in self.pregrasp_arm_pos.tolist()]}"
        )
        self.get_logger().info(response.message)
        return response

    def _stop_cb(self, request, response):
        self.state = State.IDLE
        response.success = True
        response.message = "중단 → IDLE"
        return response

    def _reset_cb(self, request, response):
        self.state = State.IDLE
        self._reset_episode_state()
        response.success = True
        response.message = "리셋 → IDLE"
        return response

    def _reset_episode_state(self) -> None:
        self.step_count = 0
        self.last_actions = np.zeros(11)
        self.finger_ctrl.reset()
        self.lift_latch.reset()
        self.lift_arm_start = None
        self.prelift_target = None
        self.lift_start_step = None

    # ------------------------------------------------------------------
    # Pregrasp IK (env reset 재현)
    # ------------------------------------------------------------------
    def _compute_pregrasp(self) -> None:
        pregrasp_pos = self.cup_pos + np.array(PREGRASP_OFFSET)
        pose6 = np.concatenate([pregrasp_pos, np.array(PREGRASP_ORI)])
        self.pregrasp_palm_pose = np.clip(pose6, self.palm_mins, self.palm_maxs)

        q_start = torch.cat([
            _t(RIGHT_ARM_START_POSE, self.device),
            _t(HAND_APPROACH_POSE, self.device),
        ]).unsqueeze(0)
        self.fabric_q = q_start.clone()
        self.fabric_qd = torch.zeros(1, 27, device=self.device)
        self.fabric_qdd = torch.zeros(1, 27, device=self.device)

        palm_tgt = _t(self.pregrasp_palm_pose, self.device).unsqueeze(0)
        pca_zero = torch.zeros(1, 5, device=self.device)
        self.get_logger().info(
            f"Pregrasp IK rollout ({PREGRASP_FABRICS_STEPS}스텝)... cup={self.cup_pos.tolist()}"
        )
        for _ in range(PREGRASP_FABRICS_STEPS):
            self.fabric.set_features(
                pca_zero, palm_tgt, "euler_zyx",
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.object_ids, self.object_indicator, self.damping_gain,
            )
            for _ in range(FABRIC_DECIMATION):
                self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                    self.fabric_q.detach(), self.fabric_qd.detach(),
                    self.fabric_qdd.detach(), FABRICS_DT,
                )
        self.pregrasp_arm_pos = self.fabric_q[0, :NUM_ARM_DOF].cpu().numpy()

        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, :NUM_ARM_DOF] = _t(self.pregrasp_arm_pos, self.device)
        self.fabric.default_config.copy_(cspace_def)

        self.fabric_q[0, :NUM_ARM_DOF] = _t(self.arm_pos, self.device)
        self.fabric_q[0, NUM_ARM_DOF:] = _t(self.hand_pos, self.device)
        self.fabric_qd.zero_()
        self.fabric_qdd.zero_()

    # ------------------------------------------------------------------
    # APPROACHING (10Hz)
    # ------------------------------------------------------------------
    def _approach_loop(self) -> None:
        if self.state != State.APPROACHING:
            return
        self.cmd_pub.send_right_full(self.pregrasp_arm_pos.tolist(), list(HAND_APPROACH_POSE))
        if time.monotonic() - self._approach_start_time >= self.settle_time:
            self.fabric_q[0, :NUM_ARM_DOF] = _t(self.arm_pos, self.device)
            self.fabric_q[0, NUM_ARM_DOF:] = _t(self.hand_pos, self.device)
            self.fabric_qd.zero_()
            self.fabric_qdd.zero_()
            self.state = State.RUNNING
            self.get_logger().info("settle 완료 → RUNNING")

    # ------------------------------------------------------------------
    # RUNNING (60Hz)
    # ------------------------------------------------------------------
    def _policy_loop(self) -> None:
        if self.state != State.RUNNING:
            return

        # 1. fabric_q 실제 관절 동기화
        self.fabric_q[0, :NUM_ARM_DOF] = _t(self.arm_pos, self.device)
        self.fabric_q[0, NUM_ARM_DOF:] = _t(self.hand_pos, self.device)
        self.fabric_qd[0, :NUM_ARM_DOF] = _t(self.arm_vel, self.device)
        self.fabric_qd[0, NUM_ARM_DOF:] = _t(self.hand_vel, self.device)

        # 2. FK palm_center, fingertip_pos (Fabrics)
        with torch.inference_mode():
            palm_pose_6d = self.fabric.get_palm_pose(self.fabric_q, "euler_zyx")
            fingertip_pos = self.fabric.get_fingertip_positions(self.fabric_q)
        palm_center = palm_pose_6d[0, :3].cpu().numpy()
        tips = fingertip_pos[0].cpu().numpy()   # (5,3)

        # 3. obs 114D
        obs_np = assemble_actor_obs(
            arm_joint_pos=self.arm_pos,
            arm_joint_vel=self.arm_vel,
            finger_joint_pos=self.hand_pos,
            finger_joint_vel=self.hand_vel,
            palm_center=palm_center,
            fingertip_pos=tips,
            cup_pos=self.cup_pos,
            binary_contact=self.tip_contact,
            last_actions=self.last_actions,
            object_onehot=self.object_onehot,
        )
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 4. Policy → action 11D
        action = self.policy.get_action(obs)[0].cpu().numpy()
        self.last_actions = action.copy()
        palm_action = action[:6]
        finger_action = action[6:11]

        # 5. lift 접촉 래치 판정 (tip-only: middle/distal 은 critic 전용·제어 미사용)
        grip = int(self.tip_contact.sum())
        was_latched = self.lift_latch.latched
        is_lift = self.lift_latch.update(grip)
        just_entering = is_lift and not was_latched

        # 6. 손가락: 항상 tip-only 접촉-게이트 폐쇄 (env: 두 phase 모두 policy-controlled)
        hand_cmd = self.finger_ctrl.step(finger_action, self.tip_contact)   # distal 미배선 → tip-only

        # 7. 팔
        if not is_lift:
            # Grasp phase: Δpalm → Fabrics IK
            delta = scale_palm_delta(palm_action, self.delta_mins, self.delta_maxs)
            palm_pose = self.pregrasp_palm_pose + delta
            palm_mins_eff = np.minimum(self.palm_mins, self.pregrasp_palm_pose)
            palm_maxs_eff = np.maximum(self.palm_maxs, self.pregrasp_palm_pose)
            palm_pose = np.clip(palm_pose, palm_mins_eff, palm_maxs_eff)

            self.fabric.set_features(
                torch.zeros(1, 5, device=self.device),
                _t(palm_pose, self.device).unsqueeze(0),
                "euler_zyx",
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.object_ids, self.object_indicator, self.damping_gain,
            )
            for _ in range(FABRIC_DECIMATION):
                self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                    self.fabric_q.detach(), self.fabric_qd.detach(),
                    self.fabric_qdd.detach(), FABRICS_DT,
                )
            arm_cmd = self.fabric_q[0, :NUM_ARM_DOF].cpu().numpy()
        else:
            # Lift phase: 진입 시 캡처 → joint7-only lift-wait 선형보간
            if just_entering:
                self.lift_arm_start = self.arm_pos.copy()
                self.prelift_target = joint7_lift_wait_target(
                    self.arm_pos,
                    joint7_delta=DEFAULT_LIFT_WAIT_JOINT7_DELTA,
                    joint7_min=DEFAULT_WARM_J7_MIN,
                    joint7_max=DEFAULT_WARM_J7_MAX,
                )
                self.lift_start_step = self.step_count
                self.get_logger().info(
                    f"[Lift latch] step={self.step_count}, "
                    f"j7: {self.lift_arm_start[6]:.3f} → {self.prelift_target[6]:.3f}"
                )
            progress = min(
                float(self.step_count - self.lift_start_step) / max(1, LIFT_PHASE_STEPS - 1), 1.0
            )
            arm_cmd = lift_arm_interp(self.lift_arm_start, self.prelift_target, progress)

        # 8. 명령 전송
        self.cmd_pub.send_right_full(arm_cmd.tolist(), hand_cmd.tolist())

        # 9. 스텝
        self.step_count += 1
        if self.step_count >= EPISODE_STEPS:
            self.state = State.DONE
            self.get_logger().info(
                f"에피소드 완료 ({EPISODE_STEPS}스텝 / {EPISODE_STEPS / CONTROL_HZ:.1f}s) → DONE"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settle_time", type=float, default=4.0)
    parser.add_argument("--object", default="cup_big_s100",
                        help="잡는 물체 onehot id 또는 인덱스 (기본 cup_big_s100)")
    args = parser.parse_args()

    obj: str | int
    try:
        obj = int(args.object)
    except ValueError:
        obj = args.object

    rclpy.init()
    node = GraspInferenceNode(
        agent_yaml=args.agent,
        checkpoint_path=args.ckpt,
        device=args.device,
        settle_time=args.settle_time,
        object_name=obj,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
