#!/usr/bin/env python3
"""
⚠️ **DEPRECATED — 구 v7 계약(obs 106D / action 11D)**
   현행 grasp_v1 계약은 obs 154D / action 21D 이고, 진입점은
   `grasp_inference.py --robot <구성>` 이다. 이 파일은 이력 보존용으로만 남긴다.
   계약 요약: docs/CONTRACT_grasp_v1_{right,left}.md
sim2real inference 노드 (5g_grasp_right_v7).

실행 방법:
    python3 sim2real_inference.py \\
        --agent  /path/to/agent.yaml \\
        --ckpt   /path/to/5g_grasp_right-v7.pth \\
        [--device cuda:0] \\
        [--settle_time 4.0]

    ros2 service call /sim2real/start std_srvs/srv/Trigger

구독 토픽:
    /joint_states               (sensor_msgs/JointState)   arm 7D
    /dg5f_right/joint_states    (sensor_msgs/JointState)   hand 20D
    /cup_pose                   (geometry_msgs/PoseStamped) 컵 위치 (robot base frame)
    /dg5f_right/contact_forces  (std_msgs/Float64MultiArray) FT force 5D [N]

발행 토픽 (via Sim2RealCommandPublisher):
    /isaacsim/right_arm_cmd   (Float64MultiArray)  arm 7D  [rad]
    /isaacsim/right_hand_cmd  (Float64MultiArray)  hand 20D [rad]

서비스:
    /sim2real/start  → 에피소드 시작 (APPROACHING → RUNNING)
    /sim2real/stop   → 에피소드 중단
    /sim2real/reset  → IDLE 복귀

에피소드 상태 머신:
    IDLE  →(start)→  APPROACHING  →(settle_time 경과)→  RUNNING  →(600스텝)→  DONE→IDLE
                      (pregrasp 위치 반복 전송)           (policy 60Hz)

v7 reset 재현 항목:
    1. Fabrics rollout → pregrasp arm IK 계산
    2. pregrasp arm + APPROACH hand → 실기 전송 (settle_time 동안)
    3. fabric.default_config arm 부분 = pregrasp arm (null-space 안정화)
    4. fabric_q를 실제 관절 위치로 재동기화
    5. 정책 루프 시작 (step 0)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from enum import Enum, auto
from pathlib import Path

# ── Fabrics 경로 ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo"  / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

# ── Task 경로 ─────────────────────────────────────────────────────────────────
_OPENARM_SRC = (
    _SCRIPT_DIR.parent.parent
    / "hdgp" / "source" / "openarm"
)
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
GRASP_PHASE_STEPS = _consts.GRASP_PHASE_STEPS
LIFT_START_STEP = _consts.LIFT_START_STEP
EPISODE_STEPS = _consts.EPISODE_STEPS
LIFT_PHASE_STEPS = _consts.LIFT_PHASE_STEPS
PREGRASP_FABRICS_STEPS = _consts.PREGRASP_FABRICS_STEPS
CONTACT_FORCE_THRESHOLD = _consts.CONTACT_FORCE_THRESHOLD

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

# pregrasp 방향: ez=90°, ey=0°, ex=90°
PREGRASP_ORI = [math.radians(90.0), math.radians(0.0), math.radians(90.0)]

# APPROACHING 중 pregrasp 명령 전송 주파수 (Hz)
APPROACH_CMD_HZ = 10.0


# ---------------------------------------------------------------------------
# 상태 머신
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE       = auto()
    APPROACHING = auto()   # pregrasp 위치로 이동 중 (settle_time 대기)
    RUNNING    = auto()    # 정책 실행 중
    DONE       = auto()    # 에피소드 완료 (자동으로 IDLE 복귀)


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
class Sim2RealInferenceNode(Node):

    def __init__(
        self,
        agent_yaml:      str,
        checkpoint_path: str,
        device:          str  = "cuda:0",
        settle_time:     float = 4.0,
    ) -> None:
        super().__init__("sim2real_inference")
        self.device      = device
        self.settle_time = settle_time   # pregrasp 이동 후 안정화 대기 시간 [s]

        # ── Policy ───────────────────────────────────────────────────────────
        self.get_logger().info("Policy 로드 중...")
        self.policy = RLGamesActorPolicy(
            agent_yaml_path=agent_yaml,
            checkpoint_path=checkpoint_path,
            obs_dim=106,
            action_dim=11,
            device=device,
        )

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

        # cspace default: hand = GRASP_POSE 방향 (v7와 동일)
        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, NUM_ARM_DOF:] = _t(HAND_GRASP_POSE, device)
        self.fabric.default_config.copy_(cspace_def)

        # ── 파라미터 텐서 ────────────────────────────────────────────────────
        _dr = math.radians(PALM_DELTA_ROT_DEG)
        self.delta_mins = _t([-PALM_DELTA_XYZ]*3 + [-_dr]*3, device)
        self.delta_maxs = _t([ PALM_DELTA_XYZ]*3 + [ _dr]*3, device)
        self.palm_mins  = _t(palm_pose_mins(MAX_POSE_ANGLE), device)
        self.palm_maxs  = _t(palm_pose_maxs(MAX_POSE_ANGLE), device)
        self.hand_open  = _t(HAND_APPROACH_POSE, device)
        self.hand_grasp = _t(HAND_GRASP_POSE,    device)
        self.damping_gain = FABRICS_DAMPING * torch.ones(1, 1, device=device)

        # fabric 상태 (에피소드마다 초기화)
        q0 = torch.cat([_t(RIGHT_ARM_START_POSE, device),
                        _t(HAND_APPROACH_POSE,   device)]).unsqueeze(0)
        self.fabric_q   = q0.clone()
        self.fabric_qd  = torch.zeros(1, 27, device=device)
        self.fabric_qdd = torch.zeros(1, 27, device=device)

        # ── 센서 버퍼 ────────────────────────────────────────────────────────
        self.arm_pos  = torch.zeros(7,  device=device)
        self.arm_vel  = torch.zeros(7,  device=device)
        self.hand_pos = torch.zeros(20, device=device)
        self.hand_vel = torch.zeros(20, device=device)
        self.cup_pos  = torch.zeros(3,  device=device)
        self.contact  = torch.zeros(5,  device=device)

        self._arm_ready  = False
        self._hand_ready = False
        self._cup_ready  = False

        self._arm_idx  = {n: i for i, n in enumerate(RIGHT_ARM_JOINT_NAMES)}
        self._hand_idx = {n: i for i, n in enumerate(RIGHT_HAND_JOINT_NAMES)}

        # ── 에피소드 상태 ────────────────────────────────────────────────────
        self.state       = State.IDLE
        self.step_count  = 0
        self.last_actions = torch.zeros(11, device=device)

        # pregrasp 계산 결과 (APPROACHING 진입 시 설정)
        self.pregrasp_palm_pose = torch.zeros(6, device=device)
        self.pregrasp_arm_pos   = torch.zeros(7, device=device)

        # lift 버퍼
        self.lift_arm_start   = None
        self.prelift_target   = None
        self.lift_hand_frozen = None

        # APPROACHING 타이머 관련
        self._approach_start_time = 0.0

        # ── ROS2 ─────────────────────────────────────────────────────────────
        self.create_subscription(JointState,         "/joint_states",              self._arm_cb,     10)
        self.create_subscription(JointState,         "/dg5f_right/joint_states",   self._hand_cb,    10)
        self.create_subscription(PoseStamped,        "/cup_pose",                  self._cup_cb,     10)
        self.create_subscription(Float64MultiArray,  "/dg5f_right/contact_forces", self._contact_cb, 10)

        self.cmd_pub = create_publisher()

        self.create_service(Trigger, "/sim2real/start", self._start_cb)
        self.create_service(Trigger, "/sim2real/stop",  self._stop_cb)
        self.create_service(Trigger, "/sim2real/reset", self._reset_cb)

        # APPROACHING: 10Hz 명령 전송 (물리 이동)
        self.create_timer(1.0 / APPROACH_CMD_HZ, self._approach_loop)
        # RUNNING: 60Hz 정책 루프
        self.create_timer(1.0 / CONTROL_HZ,      self._policy_loop)

        self.get_logger().info(
            "준비 완료. '/sim2real/start' 서비스를 호출하면 에피소드가 시작됩니다."
        )

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
        self.cup_pos[0] = p.x
        self.cup_pos[1] = p.y
        self.cup_pos[2] = p.z
        self._cup_ready = True

    def _contact_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 5:
            return
        f = torch.tensor(list(msg.data[:5]), dtype=torch.float32, device=self.device)
        self.contact = (f > CONTACT_FORCE_THRESHOLD).float()

    # ------------------------------------------------------------------
    # 서비스 Callbacks
    # ------------------------------------------------------------------

    def _start_cb(self, request, response):
        if self.state not in (State.IDLE, State.DONE):
            response.success = False
            response.message = f"ERROR: 현재 상태={self.state.name}, IDLE 또는 DONE일 때만 start 가능"
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

        # ── pregrasp 계산 ──────────────────────────────────────────────────
        self._compute_pregrasp()

        # ── APPROACHING 진입 ───────────────────────────────────────────────
        self.state = State.APPROACHING
        self._approach_start_time = time.monotonic()
        self.step_count    = 0
        self.last_actions.zero_()
        self.lift_arm_start   = None
        self.prelift_target   = None
        self.lift_hand_frozen = None

        response.success = True
        response.message = (
            f"APPROACHING 시작 (settle_time={self.settle_time}s). "
            f"pregrasp_arm={[f'{v:.3f}' for v in self.pregrasp_arm_pos.tolist()]}"
        )
        self.get_logger().info(response.message)
        return response

    def _stop_cb(self, request, response):
        self.state = State.IDLE
        response.success = True
        response.message = "에피소드 중단 → IDLE"
        self.get_logger().info(response.message)
        return response

    def _reset_cb(self, request, response):
        self.state = State.IDLE
        self.step_count = 0
        self.last_actions.zero_()
        self.lift_arm_start   = None
        self.prelift_target   = None
        self.lift_hand_frozen = None
        response.success = True
        response.message = "리셋 완료 → IDLE"
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Pregrasp IK 계산 (v7 _reset_idx 재현)
    # ------------------------------------------------------------------

    def _compute_pregrasp(self) -> None:
        """컵 위치 기반 pregrasp palm pose 계산 + Fabrics IK rollout.

        v7 reset 재현:
          1. pregrasp_palm_pose = cup_pos + PREGRASP_OFFSET, orientation=[90°,0°,90°]
          2. fabric_q를 ARM_START_POSE + APPROACH_POSE로 초기화
          3. Fabrics rollout (PREGRASP_FABRICS_STEPS) → pregrasp_arm_pos
          4. fabric.default_config[:NUM_ARM_DOF] = pregrasp_arm_pos  ← null-space 안정화
          5. fabric_q를 실제 관절 위치로 재동기화 (다음 _policy_loop 에서 사용)
        """
        # 1. pregrasp palm pose
        pregrasp_pos = self.cup_pos + _t(PREGRASP_OFFSET, self.device)
        self.pregrasp_palm_pose = torch.cat([
            pregrasp_pos, _t(PREGRASP_ORI, self.device)
        ]).clamp(min=self.palm_mins, max=self.palm_maxs)

        # 2. fabric_q 초기화: ARM_START_POSE + APPROACH_POSE (v7 reset step 1~2와 동일)
        q_start = torch.cat([
            _t(RIGHT_ARM_START_POSE, self.device),
            _t(HAND_APPROACH_POSE,   self.device),
        ]).unsqueeze(0)
        self.fabric_q   = q_start.clone()
        self.fabric_qd  = torch.zeros(1, 27, device=self.device)
        self.fabric_qdd = torch.zeros(1, 27, device=self.device)

        # 3. Fabrics IK rollout → pregrasp arm pos
        palm_tgt = self.pregrasp_palm_pose.unsqueeze(0)
        pca_zero = torch.zeros(1, 5, device=self.device)

        self.get_logger().info(
            f"Pregrasp IK rollout ({PREGRASP_FABRICS_STEPS}스텝)... "
            f"cup={self.cup_pos.tolist()}, palm_tgt={self.pregrasp_palm_pose.tolist()}"
        )
        for _ in range(PREGRASP_FABRICS_STEPS):
            self.fabric.set_features(
                pca_zero,
                palm_tgt,
                "euler_zyx",
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.object_ids,
                self.object_indicator,
                self.damping_gain,
            )
            for _ in range(FABRIC_DECIMATION):
                self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                    self.fabric_q.detach(),
                    self.fabric_qd.detach(),
                    self.fabric_qdd.detach(),
                    FABRICS_DT,
                )

        self.pregrasp_arm_pos = self.fabric_q[0, :NUM_ARM_DOF].clone()
        self.get_logger().info(
            f"Pregrasp arm: {[f'{v:.3f}' for v in self.pregrasp_arm_pos.tolist()]}"
        )

        # 4. cspace default_config arm 부분 = pregrasp arm pos  ← v7 reset 재현
        #    null-space attractor가 ARM_START_POSE로 당기지 않도록
        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, :NUM_ARM_DOF] = self.pregrasp_arm_pos
        self.fabric.default_config.copy_(cspace_def)

        # 5. fabric_q를 실제 관절 위치로 재동기화 (정책 루프 시작 시 실제값 기반)
        self.fabric_q[0, :NUM_ARM_DOF] = self.arm_pos
        self.fabric_q[0, NUM_ARM_DOF:] = self.hand_pos
        self.fabric_qd.zero_()
        self.fabric_qdd.zero_()

    # ------------------------------------------------------------------
    # APPROACHING 루프 (10Hz): pregrasp 위치 반복 전송 → 실기 물리 이동
    # ------------------------------------------------------------------

    def _approach_loop(self) -> None:
        if self.state != State.APPROACHING:
            return

        # pregrasp arm + APPROACH hand 전송
        self.cmd_pub.send_right_full(
            self.pregrasp_arm_pos.cpu().tolist(),
            HAND_APPROACH_POSE,
        )

        # settle_time 경과 → RUNNING 진입
        elapsed = time.monotonic() - self._approach_start_time
        if elapsed >= self.settle_time:
            # 실제 관절 위치로 fabric_q 재동기화 (로봇이 실제로 이동했으므로)
            self.fabric_q[0, :NUM_ARM_DOF] = self.arm_pos
            self.fabric_q[0, NUM_ARM_DOF:] = self.hand_pos
            self.fabric_qd.zero_()
            self.fabric_qdd.zero_()

            self.state = State.RUNNING
            self.get_logger().info(
                f"settle 완료 ({elapsed:.1f}s) → RUNNING 시작. "
                f"실제 arm={[f'{v:.3f}' for v in self.arm_pos.tolist()]}"
            )

    # ------------------------------------------------------------------
    # RUNNING 루프 (60Hz): 정책 실행
    # ------------------------------------------------------------------

    def _policy_loop(self) -> None:
        if self.state != State.RUNNING:
            return

        # ── 1. fabric_q를 실제 관절 상태로 동기화 ──────────────────────────
        self.fabric_q[0, :NUM_ARM_DOF] = self.arm_pos
        self.fabric_q[0, NUM_ARM_DOF:] = self.hand_pos
        self.fabric_qd[0, :NUM_ARM_DOF] = self.arm_vel
        self.fabric_qd[0, NUM_ARM_DOF:] = self.hand_vel

        # ── 2. FK: palm_center, fingertip_pos ──────────────────────────────
        with torch.inference_mode():
            palm_pose_6d  = self.fabric.get_palm_pose(self.fabric_q, "euler_zyx")
            fingertip_pos = self.fabric.get_fingertip_positions(self.fabric_q)

        palm_center = palm_pose_6d[0, :3]      # (3,)
        tips        = fingertip_pos[0]          # (5, 3)

        # ── 3. 106D obs ────────────────────────────────────────────────────
        obs = torch.cat([
            self.arm_pos,                               # 7
            self.arm_vel,                               # 7
            self.hand_pos,                              # 20
            self.hand_vel,                              # 20
            palm_center,                                # 3
            (tips - palm_center).flatten(),             # 15
            self.cup_pos - palm_center,                 # 3
            (tips - self.cup_pos).flatten(),            # 15
            self.contact,                               # 5
            self.last_actions,                          # 11
        ]).unsqueeze(0)  # (1, 106)

        # ── 4. Policy → 11D action ─────────────────────────────────────────
        action = self.policy.get_action(obs)[0]   # (11,)
        self.last_actions = action.clone()

        # ── 5. Phase 판정 ──────────────────────────────────────────────────
        is_lift = (self.step_count >= LIFT_START_STEP)

        # ── 6a. Grasp phase: Fabrics arm + per-finger lerp ────────────────
        if not is_lift:
            palm_action   = action[:6]
            finger_action = action[6:11]

            # delta + pregrasp 기준점 → palm target (v7 _pre_physics_step 재현)
            delta      = _scale(palm_action, self.delta_mins, self.delta_maxs)
            palm_target = (self.pregrasp_palm_pose + delta).clamp(
                min=self.palm_mins, max=self.palm_maxs
            )

            # Fabrics IK: palm target → fabric_q[:NUM_ARM_DOF]
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
                    self.fabric_q.detach(),
                    self.fabric_qd.detach(),
                    self.fabric_qdd.detach(),
                    FABRICS_DT,
                )

            arm_cmd = self.fabric_q[0, :NUM_ARM_DOF].clone()

            # per-finger lerp: action[-1]=APPROACH_POSE, action[+1]=GRASP_POSE
            t        = (finger_action + 1.0) / 2.0       # (5,) ∈ [0,1]
            t_exp    = t.repeat_interleave(4)             # (20,)
            hand_cmd = self.hand_open + t_exp * (self.hand_grasp - self.hand_open)

        # ── 6b. Lift phase: scripted arm 보간 + 손가락 고정 ───────────────
        else:
            # 진입 시점 1회 캡처
            if self.lift_arm_start is None:
                self.lift_arm_start = self.arm_pos.clone()
                prelift = self.arm_pos.clone()
                prelift[3] = (prelift[3] + 0.31).clamp(max=3.14)
                self.prelift_target   = prelift
                self.lift_hand_frozen = self.hand_pos.clone()
                self.get_logger().info(
                    f"[Lift phase] step={self.step_count}, "
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

        # ── 7. 명령 전송 ───────────────────────────────────────────────────
        self.cmd_pub.send_right_full(
            arm_cmd.cpu().tolist(),
            hand_cmd.cpu().tolist(),
        )

        # ── 8. 스텝 카운트 ─────────────────────────────────────────────────
        self.step_count += 1
        if self.step_count >= EPISODE_STEPS:
            self.state = State.DONE
            self.get_logger().info(
                f"에피소드 완료 ({EPISODE_STEPS}스텝 / {EPISODE_STEPS/CONTROL_HZ:.1f}s) → DONE"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",       required=True)
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--settle_time", type=float, default=4.0,
                        help="pregrasp 이동 후 안정화 대기 시간 [s]")
    args = parser.parse_args()

    rclpy.init()
    node = Sim2RealInferenceNode(
        agent_yaml=args.agent,
        checkpoint_path=args.ckpt,
        device=args.device,
        settle_time=args.settle_time,
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
