#!/usr/bin/env python3
"""both/pour_sensor(양팔·15D) 라이브 배포 노드 — RA-L 정책용.

`pour_inference.PourInferenceNode`(pour_v1 한팔·12D)를 상속해 **차이분만** 덮어쓴다.
두 env의 오른팔 경로가 동일함은 `test_pour_sensor_bimanual.py`의 drift-guard가 감시한다.

차이분 3가지:
  1. policy action 12D → **15D** ([12:15] = 왼팔 receiver TCP 증분)
  2. actor obs의 좌팔 9+9 슬롯을 **실제 왼팔 엔코더**로 채움 (한팔 배포는 0이었음)
  3. 왼팔 명령 발행 — 기본은 **M0(frozen)**: rest 자세로 receiver를 든 채 고정

M0 축소 배포 근거: 논문이 nominal C0에서 M4(95.1%) ≈ M0(94.0%)이고 EXP-2 freeze에서도
성능 하락이 없음을 보고했다. 따라서 frozen receiver 배포는 임의 축소가 아니라 논문
자체 분석이 정당화하는 범위다. `--receiver learned`로 능동 모드도 켤 수 있으나,
그 경우 왼팔 DiffIK 경로 검증이 선행돼야 한다(미검증).

실행:
    python3 pour_sensor_inference.py \
        --agent /path/agent.yaml --ckpt /path/pour_sensor-lstm.pth \
        [--receiver frozen] [--target-cup 0.268 0.100 0.291]

    ros2 service call /pour/capture_grasp_offset std_srvs/srv/Trigger
    ros2 service call /pour/start std_srvs/srv/Trigger

구독(추가): /joint_states 에서 왼팔 l_aj_1..7 + l_hj_gripper_1..2 (9D)
발행(추가): /isaacsim/left_arm_cmd (7D rad)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import rclpy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from pour_action_decoder import decode  # noqa: E402
from pour_inference import (  # noqa: E402
    CONTROL_HZ,
    FABRIC_DECIMATION,
    NUM_ARM_DOF,
    PourInferenceNode,
    State,
    _t,
)
from pour_obs_builder import (  # noqa: E402
    HAND_APPROACH_POSE,
    HAND_GRASP_POSE,
    NUM_PALM_ACTION,
    assemble_actor_obs,
    compose_pose,
)
from pour_sensor_bimanual import (  # noqa: E402
    ACTION_DIM as ACTION_DIM_15,
    LeftTcpController,
    split_bimanual_action,
)
import torch  # noqa: E402
from palm_fk import palm_pose  # noqa: E402

# 왼팔 관절 이름 — hdgp pour_right_preset.LEFT_ARM_AND_GRIPPER_JOINT_NAMES 순서와 동일해야 한다.
LEFT_ARM_JOINT_NAMES = tuple(f"l_aj_{i}" for i in range(1, 8))
LEFT_GRIPPER_JOINT_NAMES = ("l_hj_gripper_1", "l_hj_gripper_2")
LEFT_ALL_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES
NUM_LEFT_DOF = len(LEFT_ALL_JOINT_NAMES)   # 9

# hdgp preset LEFT_ARM_REST_JOINT_POS (drift는 배포 전 대조할 것)
LEFT_ARM_REST = (-0.315, -0.079, 0.217, 0.513, 0.666, -0.729, -0.957)
LEFT_GRIPPER_REST = (0.044, 0.044)


class PourSensorInferenceNode(PourInferenceNode):
    """양팔 pour_sensor 배포 노드 (기본 M0 = frozen receiver)."""

    ACTION_DIM = ACTION_DIM_15   # 부모가 이 값으로 policy를 만든다 (12 → 15)

    def __init__(
        self,
        agent_yaml: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        target_cup_pos: tuple[float, float, float] = (0.268, 0.100, 0.291),
        receiver_mode: str = "frozen",
    ) -> None:
        super().__init__(agent_yaml, checkpoint_path, device, target_cup_pos)

        self.receiver_mode = receiver_mode
        # M0에서는 왼팔이 rest에 고정되므로 TCP 목표도 rest 상수면 충분하다.
        # (learned 모드로 확장할 때는 왼손 FK로 rest TCP를 계산해 넣어야 한다.)
        self.left_tcp = LeftTcpController(
            rest_pos_b=np.zeros(3), mode=receiver_mode, hold_steps=0
        )
        self.left_pos = np.array(LEFT_ARM_REST + LEFT_GRIPPER_REST, dtype=np.float64)
        self.left_vel = np.zeros(NUM_LEFT_DOF)
        self._left_seen = False

        self.create_subscription(JointState, "/joint_states", self._left_cb, 10)
        self.get_logger().info(
            f"[pour_sensor] 양팔 배포 노드 준비 — action {self.ACTION_DIM}D, "
            f"receiver={receiver_mode}"
        )
        if receiver_mode == "frozen":
            self.get_logger().info(
                "  M0 축소 배포: 왼팔은 rest로 receiver를 든 채 고정 "
                "(논문 C0에서 M4 95.1% ≈ M0 94.0% 근거)"
            )

    # -- callbacks ---------------------------------------------------------
    def _left_cb(self, msg: JointState) -> None:
        """/joint_states에서 왼팔 9관절을 이름으로 추출 (순서 무관, 누락은 무시)."""
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        pos = np.array(self.left_pos, dtype=np.float64)
        vel = np.zeros(NUM_LEFT_DOF)
        found = 0
        for k, jn in enumerate(LEFT_ALL_JOINT_NAMES):
            i = name_to_i.get(jn)
            if i is None:
                continue
            found += 1
            if i < len(msg.position):
                pos[k] = msg.position[i]
            if i < len(msg.velocity):
                vel[k] = msg.velocity[i]
        if found == 0:
            return
        if found < NUM_LEFT_DOF and not self._left_seen:
            self.get_logger().warn(
                f"왼팔 관절 {found}/{NUM_LEFT_DOF}개만 수신 — 나머지는 rest 값으로 채운다"
            )
        self.left_pos, self.left_vel = pos, vel
        self._left_seen = True

    # -- main loop ---------------------------------------------------------
    def _policy_loop(self) -> None:
        if self.state != State.RUNNING:
            return
        if not self._left_seen and self.step_count == 0:
            self.get_logger().warn(
                "왼팔 /joint_states 미수신 — obs 좌팔 항목이 rest 상수로 채워진다"
            )

        # 1) 소스 컵: palm FK ∘ grasp offset (비전 무의존, pour_v1과 동일)
        palm_p, palm_q = palm_pose(self.arm_pos)
        cup_p, cup_q = compose_pose(palm_p, palm_q, *self.grasp_offset)

        # 2) obs 55D — ★ 좌팔 슬롯을 실제 엔코더로 채운다 (한팔 배포와의 유일한 obs 차이)
        obs_np = assemble_actor_obs(
            arm_joint_pos=self.arm_pos,
            arm_joint_vel=self.arm_vel,
            finger_joint_pos=self.hand_pos,
            left_arm_joint_pos=self.left_pos,
            left_arm_joint_vel=self.left_vel,
            source_cup_pos=cup_p,
            source_cup_quat=cup_q,
            target_cup_pos=self.target_cup_pos,
            target_cup_quat=self.target_cup_quat,
            last_palm_actions=self.last_actions[:NUM_PALM_ACTION],
        )
        obs = _t(obs_np, self.device).unsqueeze(0)

        # 3) LSTM policy → action 15D → (오른팔 12D, 왼팔 TCP 3D)
        action = self.policy.get_action(obs).squeeze(0).cpu().numpy().astype(np.float64)
        self.last_actions = action
        right_action, left_action = split_bimanual_action(action)
        self.left_tcp.step(left_action)   # frozen이면 rest 유지 (상태만 진행)

        # 4~5) 오른팔: 기존 디코더 + Fabrics (완전 동일 경로)
        target = decode(
            right_action, self.decoder_state,
            cup_p, cup_q, self.target_cup_pos, self.target_cup_quat,
            palm_center_pos=palm_p, palm_quat_wxyz=palm_q,
        )
        pose7 = _t(np.concatenate([target.pos, target.quat_xyzw]), self.device).unsqueeze(0)
        hand_pca = torch.zeros(1, 5, device=self.device)
        self.fabric.set_features(
            hand_pca, pose7, "quaternion",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self.object_ids, self.object_indicator, self.damping_gain,
        )
        for _ in range(FABRIC_DECIMATION):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach()
            )

        # 6) 손 lerp (pour_v1과 동일)
        lerp = (target.hand_action * 0.5 + 0.5)[:, None]
        approach = np.array(HAND_APPROACH_POSE).reshape(5, 4)
        grasp = np.array(HAND_GRASP_POSE).reshape(5, 4)
        hand_cmd = (approach + lerp * (grasp - approach)).reshape(20)

        # 7) 발행 — 오른팔/손 + 왼팔(frozen이면 rest 유지 명령)
        arm_cmd = self.fabric_q[0, :NUM_ARM_DOF].cpu().numpy()
        self.cmd_pub.send_right_arm(arm_cmd.tolist())
        self.cmd_pub.send_right_hand(hand_cmd.tolist())
        if self.receiver_mode == "frozen":
            self.cmd_pub.send_left_arm(list(LEFT_ARM_REST))

        self.step_count += 1
        if self.step_count % int(CONTROL_HZ) == 0:
            self.get_logger().info(
                f"[{self.step_count / CONTROL_HZ:.0f}s] "
                f"spout→opening |d|={np.linalg.norm(obs_np[37:40]):.3f} m"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-cup", nargs=3, type=float, default=[0.268, 0.100, 0.291],
                    help="receiver 컵 pose(base 프레임) — 실물 실측값으로 교체할 것")
    ap.add_argument("--receiver", choices=("frozen", "learned"), default="frozen",
                    help="frozen=M0 축소 배포(권장·검증됨) / learned=M4(왼팔 DiffIK 미검증)")
    args = ap.parse_args()

    rclpy.init()
    node = PourSensorInferenceNode(
        agent_yaml=args.agent, checkpoint_path=args.ckpt, device=args.device,
        target_cup_pos=tuple(args.target_cup), receiver_mode=args.receiver,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
