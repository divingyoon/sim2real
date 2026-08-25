#!/usr/bin/env python3
"""pour_v1 라이브 sim2real inference 노드 (tesollo/right/pour-v1 LSTM 체크포인트).

grasp 성공(컵을 쥔 상태)에서 시작해 pour를 수행한다. (6.3) 하이브리드 전략:
grasp 단계는 별도(sim2real_inference.py + FoundationPose /cup_pose)로 수행하고,
이 노드는 **잡은 이후**를 담당한다 — 시작 시 /cup_pose 스냅샷으로 grasp offset을
캘리브하고, 이후 pour 동안 컵 pose를 palm FK ∘ offset으로 추정한다(비전 무의존).

실행:
    python3 pour_inference.py \
        --agent /path/to/agent.yaml --ckpt /path/to/pour-v1-lstm.pth \
        [--device cuda:0] [--target-cup 0.268 0.100 0.291]

    ros2 service call /pour/capture_grasp_offset std_srvs/srv/Trigger
    ros2 service call /pour/start std_srvs/srv/Trigger

구독:
    /joint_states               arm 7D (openarm_right_joint1..7)
    /dg5f_right/joint_states    hand 20D (rj_dg_*)
    /cup_pose                   PoseStamped — grasp offset 캘리브 시 1회만 사용

발행: /isaacsim/right_arm_cmd (7D rad), /isaacsim/right_hand_cmd (20D rad)

상태머신: IDLE →(capture_grasp_offset)→ ARMED →(start)→ RUNNING →(stop/steps)→ IDLE

sim 대응 재현:
    정책 60Hz(decimation=2·물리120Hz), fabric_decimation=2, LSTM hidden 스텝 유지,
    action 디코드는 pour_action_decoder(=env _pre_physics_step 기본 경로 포팅),
    obs 55D는 pour_obs_builder(=env _get_observations actor 경로 포팅).
"""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo" / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from fabrics_ros_interface import create_publisher
from policy_loader import RLGamesLstmActorPolicy
from palm_fk import extract_joints, grasp_offset_from_snapshot, palm_pose
from pour_action_decoder import PourDecoderState, decode
from pour_obs_builder import (
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    NUM_PALM_ACTION,
    assemble_actor_obs,
    compose_pose,
)

# ── sim 정합 상수 (pour_v1 env_cfg) ─────────────────────────────────────────
CONTROL_HZ = 60.0          # 정책 60Hz (물리 120Hz, decimation 2)
FABRICS_DT = 1.0 / 60.0
FABRIC_DECIMATION = 2
FABRICS_DAMPING = 20.0
NUM_ARM_DOF = 7
NUM_HAND_DOF = 20
NUM_ACTIONS = 12
OBS_DIM = 55
EPISODE_STEPS = int(20.0 * CONTROL_HZ)   # episode_length_s=20

RIGHT_ARM_JOINT_NAMES = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
RIGHT_HAND_JOINT_NAMES = tuple(
    f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)
)


class State(Enum):
    IDLE = auto()      # 대기 (grasp offset 미캘리브)
    ARMED = auto()     # offset 캘리브 완료, start 대기
    RUNNING = auto()   # 정책 실행 중


def _t(vals, device):
    return torch.tensor(vals, dtype=torch.float32, device=device)


class PourInferenceNode(Node):
    # 서브클래스가 action 차원을 바꿀 수 있게 클래스 속성으로 노출
    # (both/pour_sensor 양팔 배포는 15D — pour_sensor_inference.py 참조).
    ACTION_DIM = NUM_ACTIONS

    def __init__(
        self,
        agent_yaml: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        target_cup_pos: tuple[float, float, float] = (0.268, 0.100, 0.291),
    ) -> None:
        super().__init__("pour_inference")
        self.device = device

        # 타깃 컵: 실물 세팅에서 측정한 고정 pose (base 프레임). 직립 가정.
        # sim 기본값은 좌팔 FK 유래 — 실물에서는 반드시 실측으로 교체할 것.
        self.target_cup_pos = np.array(target_cup_pos, dtype=np.float64)
        self.target_cup_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.get_logger().info("LSTM policy 로드 중...")
        self.policy = RLGamesLstmActorPolicy(
            agent_yaml_path=agent_yaml,
            checkpoint_path=checkpoint_path,
            obs_dim=OBS_DIM,
            action_dim=self.ACTION_DIM,
            device=device,
        )

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
        self.damping_gain = FABRICS_DAMPING * torch.ones(1, 1, device=device)

        # cspace default: hand는 grasp 유지 (pour 동안 컵을 쥔 상태)
        cspace_def = self.fabric.default_config.clone()
        cspace_def[0, NUM_ARM_DOF:] = _t(HAND_GRASP_POSE, device)
        self.fabric.default_config.copy_(cspace_def)

        self.fabric_q = torch.zeros(1, 27, device=device)
        self.fabric_qd = torch.zeros(1, 27, device=device)
        self.fabric_qdd = torch.zeros(1, 27, device=device)

        # ── 센서 버퍼 (numpy — obs builder가 numpy) ──────────────────────────
        self.arm_pos = np.zeros(7)
        self.arm_vel = np.zeros(7)
        self.hand_pos = np.zeros(20)
        self.cup_pose_msg: tuple[np.ndarray, np.ndarray] | None = None
        self._arm_ready = False
        self._hand_ready = False

        # ── 에피소드 상태 ────────────────────────────────────────────────────
        self.state = State.IDLE
        self.step_count = 0
        self.decoder_state = PourDecoderState()
        self.grasp_offset: tuple[np.ndarray, np.ndarray] | None = None
        self.last_actions = np.zeros(self.ACTION_DIM)

        # ── ROS2 ────────────────────────────────────────────────────────────
        self.create_subscription(JointState, "/joint_states", self._arm_cb, 10)
        self.create_subscription(JointState, "/dg5f_right/joint_states", self._hand_cb, 10)
        self.create_subscription(PoseStamped, "/cup_pose", self._cup_cb, 10)
        self.cmd_pub = create_publisher()

        self.create_service(Trigger, "/pour/capture_grasp_offset", self._capture_cb)
        self.create_service(Trigger, "/pour/start", self._start_cb)
        self.create_service(Trigger, "/pour/stop", self._stop_cb)
        self.create_timer(1.0 / CONTROL_HZ, self._policy_loop)

        self.get_logger().info(
            "준비 완료. 컵을 쥔 상태에서 /pour/capture_grasp_offset → /pour/start."
        )

    # ── 콜백 ────────────────────────────────────────────────────────────────
    def _arm_cb(self, msg: JointState) -> None:
        try:
            self.arm_pos = extract_joints(msg.name, msg.position, RIGHT_ARM_JOINT_NAMES)
            if msg.velocity:
                self.arm_vel = extract_joints(msg.name, msg.velocity, RIGHT_ARM_JOINT_NAMES)
            self._arm_ready = True
        except KeyError:
            pass  # 병합 전 부분 메시지 무시

    def _hand_cb(self, msg: JointState) -> None:
        try:
            self.hand_pos = extract_joints(msg.name, msg.position, RIGHT_HAND_JOINT_NAMES)
            self._hand_ready = True
        except KeyError:
            pass

    def _cup_cb(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self.cup_pose_msg = (
            np.array([p.x, p.y, p.z]),
            np.array([o.w, o.x, o.y, o.z]),  # wxyz
        )

    # ── 서비스 ──────────────────────────────────────────────────────────────
    def _capture_cb(self, request, response):
        """컵을 쥔 지금, FoundationPose 컵 pose + palm FK로 grasp offset 캘리브."""
        if not self._arm_ready:
            response.success = False
            response.message = "arm joint_states not received yet"
            return response
        if self.cup_pose_msg is None:
            response.success = False
            response.message = "/cup_pose not received yet (FoundationPose 노드 확인)"
            return response
        palm_p, palm_q = palm_pose(self.arm_pos)
        cup_p, cup_q = self.cup_pose_msg
        self.grasp_offset = grasp_offset_from_snapshot(palm_p, palm_q, cup_p, cup_q)
        self.state = State.ARMED
        off_p = self.grasp_offset[0]
        response.success = True
        response.message = f"grasp offset captured: pos={off_p.round(4).tolist()}"
        self.get_logger().info(response.message)
        return response

    def _start_cb(self, request, response):
        if self.state != State.ARMED or self.grasp_offset is None:
            response.success = False
            response.message = "call /pour/capture_grasp_offset first"
            return response
        if not (self._arm_ready and self._hand_ready):
            response.success = False
            response.message = "joint states not ready"
            return response
        # sim 에피소드 시작 재현: LSTM hidden·디코더 상태·fabric q 초기화
        self.policy.reset_states()
        self.decoder_state = PourDecoderState()
        self.last_actions = np.zeros(self.ACTION_DIM)
        self.step_count = 0
        q0 = torch.cat(
            [_t(self.arm_pos, self.device), _t(self.hand_pos, self.device)]
        ).unsqueeze(0)
        self.fabric_q.copy_(q0)
        self.fabric_qd.zero_()
        self.fabric_qdd.zero_()
        self.state = State.RUNNING
        response.success = True
        response.message = "pour episode started"
        self.get_logger().info(response.message)
        return response

    def _stop_cb(self, request, response):
        self.state = State.IDLE if self.grasp_offset is None else State.ARMED
        response.success = True
        response.message = "stopped"
        return response

    # ── 정책 루프 (60Hz) ─────────────────────────────────────────────────────
    def _policy_loop(self) -> None:
        if self.state != State.RUNNING:
            return

        # 1) FK: palm pose → 컵 pose (grasp offset 합성, 비전 무의존)
        palm_p, palm_q = palm_pose(self.arm_pos)
        cup_p, cup_q = compose_pose(palm_p, palm_q, *self.grasp_offset)

        # 2) obs 55D (한팔: 좌팔 항목 0)
        obs_np = assemble_actor_obs(
            arm_joint_pos=self.arm_pos,
            arm_joint_vel=self.arm_vel,
            finger_joint_pos=self.hand_pos,
            left_arm_joint_pos=np.zeros(9),
            left_arm_joint_vel=np.zeros(9),
            source_cup_pos=cup_p,
            source_cup_quat=cup_q,
            target_cup_pos=self.target_cup_pos,
            target_cup_quat=self.target_cup_quat,
            last_palm_actions=self.last_actions[:NUM_PALM_ACTION],
        )
        obs = _t(obs_np, self.device).unsqueeze(0)

        # 3) LSTM policy → action 12D
        action = self.policy.get_action(obs).squeeze(0).cpu().numpy().astype(np.float64)
        self.last_actions = action

        # 4) action 디코드 → palm_link pose target (env과 동일 경로)
        target = decode(
            action, self.decoder_state,
            cup_p, cup_q, self.target_cup_pos, self.target_cup_quat,
            palm_center_pos=palm_p, palm_quat_wxyz=palm_q,
        )

        # 5) Fabrics IK: palm pose target 추종 (quaternion 형식 = env과 동일)
        pose7 = _t(
            np.concatenate([target.pos, target.quat_xyzw]), self.device
        ).unsqueeze(0)
        hand_pca = torch.zeros(1, 5, device=self.device)
        self.fabric.set_features(
            hand_pca,
            pose7,
            "quaternion",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.object_ids,
            self.object_indicator,
            self.damping_gain,
        )
        for _ in range(FABRIC_DECIMATION):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach()
            )

        # 6) hand: per-finger lerp (action[7:12] ∈[-1,1], -1=approach/+1=grasp)
        lerp = (target.hand_action * 0.5 + 0.5)[:, None]  # (5,1)
        approach = np.array(HAND_APPROACH_POSE).reshape(5, 4)
        grasp = np.array(HAND_GRASP_POSE).reshape(5, 4)
        hand_cmd = (approach + lerp * (grasp - approach)).reshape(20)

        # 7) 발행
        arm_cmd = self.fabric_q[0, :NUM_ARM_DOF].cpu().numpy()
        self.cmd_pub.send_right_arm(arm_cmd.tolist())
        self.cmd_pub.send_right_hand(hand_cmd.tolist())

        self.step_count += 1
        if self.step_count % int(CONTROL_HZ) == 0:
            self.get_logger().info(
                f"step={self.step_count} ready={target.ready} "
                f"cup=[{cup_p[0]:.3f},{cup_p[1]:.3f},{cup_p[2]:.3f}]"
            )
        if self.step_count >= EPISODE_STEPS:
            self.get_logger().info("episode steps exhausted → ARMED")
            self.state = State.ARMED


def main() -> None:
    parser = argparse.ArgumentParser(description="pour_v1 live sim2real inference")
    parser.add_argument("--agent", required=True, help="agent.yaml 경로")
    parser.add_argument("--ckpt", required=True, help="pour-v1 LSTM 체크포인트 .pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target-cup", nargs=3, type=float, default=[0.268, 0.100, 0.291],
        metavar=("X", "Y", "Z"),
        help="타깃 컵 중심 위치 (base 프레임, 실측 필수)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = PourInferenceNode(
        agent_yaml=args.agent,
        checkpoint_path=args.ckpt,
        device=args.device,
        target_cup_pos=tuple(args.target_cup),
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
