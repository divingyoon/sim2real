#!/usr/bin/env python3
"""
⚠️ **DEPRECATED — 구 v7 계약(obs 106D / action 11D)**
   현행 grasp_v1 계약은 obs 154D / action 21D 이고, 진입점은
   `grasp_inference.py --robot <구성>` 이다. 이 파일은 이력 보존용으로만 남긴다.
   계약 요약: docs/CONTRACT_grasp_v1_{right,left}.md
sim2real dry-run 시각화 노드 (5g_grasp_right_v7).

실제 하드웨어 없이 정책 동작을 RViz로 시각화합니다.

특징:
  - 실제 센서 구독 없음
  - Fabrics IK 출력 → arm_pos 직접 반영 (완벽 추종 가정)
  - per-finger lerp 출력 → hand_pos 직접 반영
  - /joint_states publish → robot_state_publisher → RViz

실행 순서:
  # 1. fake hardware + RViz (robot_state_publisher 포함)
  ros2 launch openarm_control openarm_left_gripper_bimanual_real.launch.py use_fake_hardware:=true

  # 2. isaacsim_bridge (fake hardware JointTrajectoryController 연결)
  ros2 launch isaacsim_bridge isaacsim_bridge.launch.py

  # 3. dry-run 노드
  python3 sim2real_dryrun.py \\
      --agent .../params/agent.yaml \\
      --ckpt  .../nn/5g_grasp_right-v7.pth \\
      --cup_x 0.40 --cup_y -0.15 --cup_z 0.38

  # 4. 에피소드 시작
  ros2 service call /sim2real/start std_srvs/srv/Trigger

구독 토픽:
  /cup_pose  (geometry_msgs/PoseStamped)  cup 위치 실시간 업데이트 (선택)

발행 토픽:
  /joint_states              (sensor_msgs/JointState)      arm+hand → RViz
  /isaacsim/right_arm_cmd   (std_msgs/Float64MultiArray)  arm → isaacsim_bridge → 컨트롤러
  /isaacsim/right_hand_cmd  (std_msgs/Float64MultiArray)  hand → isaacsim_bridge → 컨트롤러

서비스:
  /sim2real/start  /sim2real/stop  /sim2real/reset
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

# ── Fabrics 경로 ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo"  / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

# ── Task 경로 ─────────────────────────────────────────────────────────────────
_OPENARM_SRC = _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "openarm"
sys.path.insert(0, str(_OPENARM_SRC))

import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

sys.path.insert(0, str(_SCRIPT_DIR))
from policy_loader import RLGamesActorPolicy

import importlib as _il

_preset = _il.import_module(
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right"
    ".5g_grasp_right_v7.grasp_right_preset"
)
RIGHT_ARM_JOINT_NAMES = _preset.RIGHT_ARM_JOINT_NAMES
RIGHT_HAND_JOINT_NAMES = _preset.RIGHT_HAND_JOINT_NAMES
HAND_APPROACH_POSE = _preset.HAND_APPROACH_POSE
HAND_GRASP_POSE = _preset.HAND_GRASP_POSE
RIGHT_ARM_START_POSE = _preset.RIGHT_ARM_START_POSE
palm_pose_mins = _preset.palm_pose_mins
palm_pose_maxs = _preset.palm_pose_maxs
PREGRASP_OFFSET = _preset.PREGRASP_OFFSET

_consts = _il.import_module(
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right"
    ".5g_grasp_right_v7.grasp_right_constants"
)
NUM_ARM_DOF = _consts.NUM_ARM_DOF
NUM_HAND_DOF = _consts.NUM_HAND_DOF
LIFT_START_STEP = _consts.LIFT_START_STEP
EPISODE_STEPS = _consts.EPISODE_STEPS
LIFT_PHASE_STEPS = _consts.LIFT_PHASE_STEPS
PREGRASP_FABRICS_STEPS = _consts.PREGRASP_FABRICS_STEPS

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
PALM_DELTA_XYZ     = 0.15
PALM_DELTA_ROT_DEG = 20.0
MAX_POSE_ANGLE     = 45.0
FABRICS_DAMPING    = 20.0
FABRIC_DECIMATION  = 2
FABRICS_DT         = 1.0 / 60.0
CONTROL_HZ         = 60.0

PREGRASP_ORI = [math.radians(90.0), math.radians(0.0), math.radians(90.0)]


# ---------------------------------------------------------------------------
# 상태 머신
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE        = auto()
    APPROACHING = auto()   # ARM_START_POSE → pregrasp 보간 시각화
    RUNNING     = auto()   # 정책 60Hz 실행
    DONE        = auto()


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def _scale(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (hi - lo) * (x + 1.0) / 2.0


def _t(vals, device: str) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------
class Sim2RealDryRunNode(Node):

    def __init__(
        self,
        agent_yaml:      str,
        checkpoint_path: str,
        cup_pos:         list,
        device:          str   = "cuda:0",
        settle_time:     float = 3.0,
    ) -> None:
        super().__init__("sim2real_dryrun")
        self.device      = device
        self.settle_time = settle_time

        # ── Policy ────────────────────────────────────────────────────────
        self.get_logger().info("Policy 로드 중...")
        self.policy = RLGamesActorPolicy(
            agent_yaml_path=agent_yaml,
            checkpoint_path=checkpoint_path,
            obs_dim=106,
            action_dim=11,
            device=device,
        )

        # ── Fabrics ───────────────────────────────────────────────────────
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

        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, NUM_ARM_DOF:] = _t(HAND_GRASP_POSE, device)
        self.fabric.default_config.copy_(cspace_def)

        # ── 파라미터 텐서 ─────────────────────────────────────────────────
        _dr = math.radians(PALM_DELTA_ROT_DEG)
        self.delta_mins = _t([-PALM_DELTA_XYZ]*3 + [-_dr]*3, device)
        self.delta_maxs = _t([ PALM_DELTA_XYZ]*3 + [ _dr]*3, device)
        self.palm_mins  = _t(palm_pose_mins(MAX_POSE_ANGLE), device)
        self.palm_maxs  = _t(palm_pose_maxs(MAX_POSE_ANGLE), device)
        self.hand_open  = _t(HAND_APPROACH_POSE, device)
        self.hand_grasp = _t(HAND_GRASP_POSE,    device)
        self.damping_gain = FABRICS_DAMPING * torch.ones(1, 1, device=device)

        # fabric 상태
        q0 = torch.cat([_t(RIGHT_ARM_START_POSE, device),
                        _t(HAND_APPROACH_POSE,   device)]).unsqueeze(0)
        self.fabric_q   = q0.clone()
        self.fabric_qd  = torch.zeros(1, 27, device=device)
        self.fabric_qdd = torch.zeros(1, 27, device=device)

        # ── 내부 로봇 상태 (완벽 추종 가정) ──────────────────────────────
        self.arm_pos  = _t(RIGHT_ARM_START_POSE, device)
        self.arm_vel  = torch.zeros(7,  device=device)
        self.hand_pos = _t(HAND_APPROACH_POSE,   device)
        self.hand_vel = torch.zeros(20, device=device)
        self.cup_pos  = _t(cup_pos, device)
        self.contact  = torch.zeros(5,  device=device)

        # ── 에피소드 상태 ──────────────────────────────────────────────────
        self.state        = State.IDLE
        self.step_count   = 0
        self.last_actions = torch.zeros(11, device=device)

        self.pregrasp_palm_pose  = torch.zeros(6, device=device)
        self.pregrasp_arm_pos    = torch.zeros(7, device=device)
        self._approach_start_pos = torch.zeros(7, device=device)
        self._approach_start_time = 0.0

        self.lift_arm_start   = None
        self.prelift_target   = None
        self.lift_hand_frozen = None

        # ── ROS2 퍼블리셔 ──────────────────────────────────────────────────
        # /joint_states → robot_state_publisher → RViz
        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)
        # isaacsim_bridge 경유 컨트롤러 전송 (선택적)
        self._arm_cmd_pub  = self.create_publisher(Float64MultiArray, "/isaacsim/right_arm_cmd", 10)
        self._hand_cmd_pub = self.create_publisher(Float64MultiArray, "/isaacsim/right_hand_cmd", 10)

        # /cup_pose 구독 (실시간 컵 위치 업데이트)
        self.create_subscription(PoseStamped, "/cup_pose", self._cup_cb, 10)

        # 서비스
        self.create_service(Trigger, "/sim2real/start", self._start_cb)
        self.create_service(Trigger, "/sim2real/stop",  self._stop_cb)
        self.create_service(Trigger, "/sim2real/reset", self._reset_cb)

        # 60Hz 통합 루프
        self.create_timer(1.0 / CONTROL_HZ, self._loop)

        self.get_logger().info(
            f"Dry-run 준비 완료. cup_pos={cup_pos}\n"
            "'ros2 service call /sim2real/start std_srvs/srv/Trigger' 로 시작"
        )

    # ------------------------------------------------------------------
    # 센서 Callback
    # ------------------------------------------------------------------

    def _cup_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.cup_pos[0] = p.x
        self.cup_pos[1] = p.y
        self.cup_pos[2] = p.z

    # ------------------------------------------------------------------
    # 서비스
    # ------------------------------------------------------------------

    def _start_cb(self, request, response):
        if self.state not in (State.IDLE, State.DONE):
            response.success = False
            response.message = f"ERROR: 현재 상태={self.state.name}"
            return response

        self._compute_pregrasp()

        self._approach_start_pos  = self.arm_pos.clone()
        self._approach_start_time = time.monotonic()
        self.state        = State.APPROACHING
        self.step_count   = 0
        self.last_actions.zero_()
        self.lift_arm_start   = None
        self.prelift_target   = None
        self.lift_hand_frozen = None

        response.success = True
        response.message = (
            f"APPROACHING 시작 (보간 {self.settle_time}s). "
            f"pregrasp_arm={[f'{v:.3f}' for v in self.pregrasp_arm_pos.tolist()]}"
        )
        self.get_logger().info(response.message)
        return response

    def _stop_cb(self, request, response):
        self.state = State.IDLE
        response.success = True
        response.message = "중단 → IDLE"
        self.get_logger().info(response.message)
        return response

    def _reset_cb(self, request, response):
        self.state      = State.IDLE
        self.step_count = 0
        self.last_actions.zero_()
        self.arm_pos    = _t(RIGHT_ARM_START_POSE, self.device)
        self.hand_pos   = _t(HAND_APPROACH_POSE,   self.device)
        self.lift_arm_start   = None
        self.prelift_target   = None
        self.lift_hand_frozen = None
        response.success = True
        response.message = "리셋 → IDLE"
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Pregrasp IK
    # ------------------------------------------------------------------

    def _compute_pregrasp(self) -> None:
        pregrasp_pos = self.cup_pos + _t(PREGRASP_OFFSET, self.device)
        self.pregrasp_palm_pose = torch.cat([
            pregrasp_pos, _t(PREGRASP_ORI, self.device)
        ]).clamp(min=self.palm_mins, max=self.palm_maxs)

        q_start = torch.cat([
            _t(RIGHT_ARM_START_POSE, self.device),
            _t(HAND_APPROACH_POSE,   self.device),
        ]).unsqueeze(0)
        self.fabric_q   = q_start.clone()
        self.fabric_qd  = torch.zeros(1, 27, device=self.device)
        self.fabric_qdd = torch.zeros(1, 27, device=self.device)

        palm_tgt = self.pregrasp_palm_pose.unsqueeze(0)
        pca_zero = torch.zeros(1, 5, device=self.device)

        self.get_logger().info(
            f"Pregrasp IK rollout ({PREGRASP_FABRICS_STEPS}스텝)... "
            f"cup={self.cup_pos.tolist()}"
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

        self.pregrasp_arm_pos = self.fabric_q[0, :NUM_ARM_DOF].clone()
        self.get_logger().info(
            f"Pregrasp arm: {[f'{v:.3f}' for v in self.pregrasp_arm_pos.tolist()]}"
        )

        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, :NUM_ARM_DOF] = self.pregrasp_arm_pos
        self.fabric.default_config.copy_(cspace_def)

    # ------------------------------------------------------------------
    # 통합 루프 (60Hz)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        if self.state == State.IDLE:
            self._publish_joint_states()
            return

        if self.state == State.APPROACHING:
            self._approaching_step()
            return

        if self.state == State.RUNNING:
            self._policy_step()
            return

        if self.state == State.DONE:
            self._publish_joint_states()

    # ------------------------------------------------------------------
    # APPROACHING: ARM_START_POSE → pregrasp 선형 보간 시각화
    # ------------------------------------------------------------------

    def _approaching_step(self) -> None:
        elapsed  = time.monotonic() - self._approach_start_time
        progress = min(elapsed / self.settle_time, 1.0)

        # arm 선형 보간
        self.arm_pos = (
            self._approach_start_pos * (1.0 - progress)
            + self.pregrasp_arm_pos  * progress
        )
        self._publish_joint_states()
        self._publish_cmd(self.arm_pos, self.hand_pos)

        if progress >= 1.0:
            # pregrasp 도달 → fabric_q 동기화 → RUNNING
            self.arm_pos = self.pregrasp_arm_pos.clone()
            self.fabric_q[0, :NUM_ARM_DOF] = self.arm_pos
            self.fabric_q[0, NUM_ARM_DOF:] = self.hand_pos
            self.fabric_qd.zero_()
            self.fabric_qdd.zero_()
            self.state = State.RUNNING
            self.get_logger().info("Approach 완료 → RUNNING 시작")

    # ------------------------------------------------------------------
    # RUNNING: 정책 60Hz (완벽 추종 가정)
    # ------------------------------------------------------------------

    def _policy_step(self) -> None:
        # ── 1. fabric_q 동기화 ──────────────────────────────────────────
        self.fabric_q[0, :NUM_ARM_DOF] = self.arm_pos
        self.fabric_q[0, NUM_ARM_DOF:] = self.hand_pos

        # ── 2. FK ───────────────────────────────────────────────────────
        with torch.inference_mode():
            palm_pose_6d  = self.fabric.get_palm_pose(self.fabric_q, "euler_zyx")
            fingertip_pos = self.fabric.get_fingertip_positions(self.fabric_q)

        palm_center = palm_pose_6d[0, :3]
        tips        = fingertip_pos[0]

        # ── 3. 106D obs ─────────────────────────────────────────────────
        obs = torch.cat([
            self.arm_pos,
            self.arm_vel,
            self.hand_pos,
            self.hand_vel,
            palm_center,
            (tips - palm_center).flatten(),
            self.cup_pos - palm_center,
            (tips - self.cup_pos).flatten(),
            self.contact,
            self.last_actions,
        ]).unsqueeze(0)

        # ── 4. Policy ───────────────────────────────────────────────────
        action = self.policy.get_action(obs)[0]
        self.last_actions = action.clone()

        # ── 5. Phase 판정 ───────────────────────────────────────────────
        is_lift = (self.step_count >= LIFT_START_STEP)

        # ── 6a. Grasp phase ─────────────────────────────────────────────
        if not is_lift:
            palm_action   = action[:6]
            finger_action = action[6:11]

            delta      = _scale(palm_action, self.delta_mins, self.delta_maxs)
            palm_target = (self.pregrasp_palm_pose + delta).clamp(
                min=self.palm_mins, max=self.palm_maxs
            )

            self.fabric.set_features(
                torch.zeros(1, 5, device=self.device),
                palm_target.unsqueeze(0),
                "euler_zyx",
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.object_ids,
                self.object_indicator,
                self.damping_gain,
            )
            for _ in range(FABRIC_DECIMATION):
                self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                    self.fabric_q.detach(), self.fabric_qd.detach(),
                    self.fabric_qdd.detach(), FABRICS_DT,
                )

            arm_cmd = self.fabric_q[0, :NUM_ARM_DOF].clone()

            t        = (finger_action + 1.0) / 2.0
            t_exp    = t.repeat_interleave(4)
            hand_cmd = self.hand_open + t_exp * (self.hand_grasp - self.hand_open)

        # ── 6b. Lift phase ──────────────────────────────────────────────
        else:
            if self.lift_arm_start is None:
                self.lift_arm_start = self.arm_pos.clone()
                prelift = self.arm_pos.clone()
                prelift[3] = (prelift[3] + 0.31).clamp(max=3.14)
                self.prelift_target   = prelift
                self.lift_hand_frozen = self.hand_pos.clone()
                self.get_logger().info(
                    f"[Lift] step={self.step_count}, "
                    f"j4: {self.lift_arm_start[3]:.3f} → {self.prelift_target[3]:.3f}"
                )

            progress = min(
                float(self.step_count - LIFT_START_STEP) / LIFT_PHASE_STEPS, 1.0
            )
            arm_cmd  = (
                self.lift_arm_start * (1.0 - progress)
                + self.prelift_target * progress
            )
            hand_cmd = self.lift_hand_frozen

        # ── 7. 완벽 추종: command → state 직접 반영 ──────────────────────
        self.arm_pos  = arm_cmd.clone()
        self.hand_pos = hand_cmd.clone()

        # ── 8. 발행 ─────────────────────────────────────────────────────
        self._publish_joint_states()
        self._publish_cmd(arm_cmd, hand_cmd)

        # ── 9. 스텝 카운트 ───────────────────────────────────────────────
        self.step_count += 1
        if self.step_count % 60 == 0:
            phase = "Lift" if is_lift else "Grasp"
            self.get_logger().info(
                f"[{phase}] step={self.step_count}/{EPISODE_STEPS}"
            )
        if self.step_count >= EPISODE_STEPS:
            self.state = State.DONE
            self.get_logger().info(f"에피소드 완료 ({EPISODE_STEPS}스텝) → DONE")

    # ------------------------------------------------------------------
    # 퍼블리시 헬퍼
    # ------------------------------------------------------------------

    def _publish_joint_states(self) -> None:
        """arm + hand joint 상태를 /joint_states로 publish → RViz 시각화."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES
        msg.position = self.arm_pos.cpu().tolist() + self.hand_pos.cpu().tolist()
        msg.velocity = self.arm_vel.cpu().tolist() + self.hand_vel.cpu().tolist()
        self._js_pub.publish(msg)

    def _publish_cmd(self, arm_cmd: torch.Tensor, hand_cmd: torch.Tensor) -> None:
        """isaacsim_bridge 경유 컨트롤러 전송 (fake_hardware 연동 시 사용)."""
        arm_msg = Float64MultiArray()
        arm_msg.data = arm_cmd.cpu().tolist()
        self._arm_cmd_pub.publish(arm_msg)

        hand_msg = Float64MultiArray()
        hand_msg.data = hand_cmd.cpu().tolist()
        self._hand_cmd_pub.publish(hand_msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",       required=True)
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--cup_x",       type=float, default=0.40)
    parser.add_argument("--cup_y",       type=float, default=-0.15)
    parser.add_argument("--cup_z",       type=float, default=0.38)
    parser.add_argument("--settle_time", type=float, default=3.0,
                        help="APPROACHING 보간 시간 [s]")
    args = parser.parse_args()

    rclpy.init()
    node = Sim2RealDryRunNode(
        agent_yaml=args.agent,
        checkpoint_path=args.ckpt,
        cup_pos=[args.cup_x, args.cup_y, args.cup_z],
        device=args.device,
        settle_time=args.settle_time,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
