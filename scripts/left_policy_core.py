#!/usr/bin/env python3
"""좌팔 v2 정책 tick **코어** — ROS 무의존. 노드는 배선만 한다.

    센서 ──▶ 게이트 ──▶ obs 49D ──▶ 정책 ──▶ action 7D ─┬─ [0:6] palm 절대 ──▶ fabric ──▶ 관절목표 7
                                                        └─ [6]   이진 그리퍼 ──▶ 개폐 지령

이 파일은 **한 스텝의 규약**만 담는다. 정책과 fabric 은 주입받는다(테스트가 가짜를 넣을
수 있게, 그리고 torch/warp 를 import 하지 않고도 규약을 검사할 수 있게).

**학습 env 와 맞춰야 하는 것들** — 어긋나면 정책은 죽지 않고 조용히 이상하게 돈다:

  · obs 49D 레이아웃 → `left_obs_builder`(env 에서 뽑은 계약, E29=B25 확인)
  · 액션 [0:6] 절대 palm → `gripper_left_palm_command`
  · 액션 [6] 이진 그리퍼 → IsaacLab `BinaryJointAction`: **a<0 이면 닫기**
  · 게이트가 닫혀 있으면 그리퍼는 **강제 개방**(0.044) — 정책 지령을 덮는다
  · 게이트 술어·래치 → `left_grasp_gate`
  · 바디 자세(FK) → `left_gripper_fk`

★게이트/obs 순서. env 는 액션 적용 시점(물리 전)에 게이트를 갱신하고, 그 값이 그 스텝의
 obs 에 실린다. 배포에서는 **현재 센서로 게이트를 갱신한 뒤 obs 를 만든다** — 실기 센서는
 sim 처럼 반스텝 어긋나 있지 않으므로 이쪽이 더 일관된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gripper_left_palm_command import PalmCommand, PalmCommandCfg, cfg_from_run
from left_grasp_gate import CUP_GRASP_BAND_AXIS, GateCfg, GraspGate, band_axis_from_run
from left_gripper_fk import LeftGripperFK
from left_obs_builder import assemble_actor_obs, segments_from_run

NUM_ACTIONS = 7
GRIPPER_OPEN = 0.044      # dump: open_command_expr
GRIPPER_CLOSE = 0.0       # dump: close_command_expr
RELEASE_LATERAL = 0.06    # dump: release_lateral
PALM_BOX = ((0.22, 0.60), (0.10, 0.43), (0.16, 0.60))


@dataclass(frozen=True)
class LeftSensors:
    """한 tick 의 실기 입력. 전부 robot base(=world) 프레임."""

    arm_q: np.ndarray          # 7
    arm_qd: np.ndarray         # 7
    grip_q: float              # m — 실기는 mimic 이라 한 값
    grip_qd: float
    cup_pos: np.ndarray        # 3
    cup_quat: np.ndarray       # 4 (w, x, y, z)


@dataclass(frozen=True)
class LeftTick:
    """한 tick 의 산출물. 노드는 arm_q_target 과 gripper_cmd 를 발행한다."""

    arm_q_target: np.ndarray   # 7 — fabric 이 낸 관절 목표
    gripper_cmd: float         # m
    action: np.ndarray         # 7 — 정책 원본
    palm_target: np.ndarray    # 6 — pos3 + euler_zyx3
    obs: np.ndarray            # 49
    gate_open: bool
    diag: dict = field(default_factory=dict)


def home_from_run(env_yaml_path) -> np.ndarray:
    """런 dump 에서 좌팔 홈 7값. ★홈의 진실원천은 소스 상수가 아니라 dump 다."""
    import re
    text = Path(env_yaml_path).read_text()
    out = []
    for i in range(1, 8):
        m = re.search(rf"^\s*l_aj_{i}:\s*(-?[0-9.eE+]+)\s*$", text, re.M)
        if m is None:
            raise SystemExit(f"dump 에 l_aj_{i} 가 없다: {env_yaml_path}")
        out.append(float(m.group(1)))
    return np.array(out)


def gripper_command(action_gripper: float, gate_open: bool) -> float:
    """액션 [6] → 그리퍼 지령. IsaacLab BinaryJointAction 규약 + 게이트 강제 개방.

    ★`a < 0` 이 **닫기**다(binary_mask = actions < 0 → close_command). 부호를 뒤집으면
     정책이 잡으려 할 때마다 손이 열린다.
    ★게이트가 닫혀 있으면 정책 지령과 무관하게 열어 둔다 — 학습에서 접근 성공 전에는
     그리퍼를 못 닫게 막았고, 그 제약 위에서 정책이 학습됐다.
    """
    if not gate_open:
        return GRIPPER_OPEN
    return GRIPPER_CLOSE if float(action_gripper) < 0.0 else GRIPPER_OPEN


class LeftPolicyCore:
    """좌 v2 한 tick. 정책·fabric 은 주입받는다."""

    def __init__(
        self,
        *,
        policy,                      # callable: obs(49,) -> action(7,)
        fabric=None,                 # callable: palm(6,) -> arm_q_target(7,)
        run_env_yaml,
        goal7,
        run_agent_yaml=None,         # ★파지 대역은 태스크 이름으로만 갈린다
        urdf_path=None,
        fk: LeftGripperFK | None = None,
        palm_cfg: PalmCommandCfg | None = None,
        gate_cfg: GateCfg | None = None,
    ) -> None:
        self.policy = policy
        self.fabric = fabric
        self.home = home_from_run(run_env_yaml)
        self.goal7 = np.asarray(goal7, dtype=np.float64).reshape(7)
        self.fk = fk or (LeftGripperFK(urdf_path) if urdf_path else LeftGripperFK())
        self.palm = PalmCommand(palm_cfg or cfg_from_run(run_env_yaml))
        if gate_cfg is None:
            band = (band_axis_from_run(run_agent_yaml) if run_agent_yaml is not None
                    else CUP_GRASP_BAND_AXIS)
            gate_cfg = GateCfg(release_lateral=RELEASE_LATERAL, band_axis=band)
        self.gate = GraspGate(gate_cfg)
        # ★관측 레이아웃도 런이 정한다 — 트랙마다 항이 다르다(v2 49D · fab 45D).
        self.segments = segments_from_run(run_env_yaml)
        self.obs_dim = sum(d for _, d in self.segments)
        # 관측은 팔7 + 그리퍼2 = 9칸을 **기본자세 대비 상대**로 쓴다.
        self._q_default = np.concatenate([self.home, [GRIPPER_OPEN, GRIPPER_OPEN]])
        self._qd_default = np.zeros(9)
        self._last_action = np.zeros(NUM_ACTIONS)
        self.step_count = 0

    def reset(self) -> None:
        self.palm.reset()
        self.gate.reset()
        self._last_action = np.zeros(NUM_ACTIONS)
        self.step_count = 0

    def step(self, s: LeftSensors) -> LeftTick:
        poses = self.fk.poses(s.arm_q, s.grip_q, s.grip_q)

        gate_open = self.gate.update(
            finger_l_pos=poses.finger_l_pos, finger_r_pos=poses.finger_r_pos,
            gripper_base_quat=poses.base_quat,
            cup_pos=s.cup_pos, cup_quat=s.cup_quat,
        )

        q9 = np.concatenate([np.asarray(s.arm_q).reshape(7), [s.grip_q, s.grip_q]])
        qd9 = np.concatenate([np.asarray(s.arm_qd).reshape(7), [s.grip_qd, s.grip_qd]])
        obs = assemble_actor_obs(
            joint_pos=q9, joint_vel=qd9,
            joint_pos_default=self._q_default, joint_vel_default=self._qd_default,
            root_pos=np.zeros(3), root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            cup_pos=s.cup_pos, cup_quat=s.cup_quat,
            goal_pos=self.goal7[:3], goal_quat=self.goal7[3:],
            tcp_pos=poses.tcp_pos,
            gripper_base_pos=poses.base_pos, gripper_base_quat=poses.base_quat,
            last_action=self._last_action, gripper_gate=self.gate.obs_value,
            palm_box=PALM_BOX,
            segments=self.segments,
        )

        action = np.asarray(self.policy(obs), dtype=np.float64).reshape(-1)
        if action.size != NUM_ACTIONS:
            raise ValueError(f"액션은 {NUM_ACTIONS}D — 받은 {action.size}")

        palm_target = self.palm.step(action[:6])
        grip = gripper_command(action[6], gate_open)
        arm_q_target = (np.asarray(self.fabric(palm_target), dtype=np.float64).reshape(7)
                        if self.fabric is not None else np.full(7, np.nan))

        self._last_action = action.copy()
        self.step_count += 1
        jf = self.gate.last
        return LeftTick(
            arm_q_target=arm_q_target, gripper_cmd=grip, action=action,
            palm_target=palm_target, obs=obs, gate_open=gate_open,
            diag={"lateral": None if jf is None else jf.lateral,
                  "along": None if jf is None else jf.along,
                  "axis_t": None if jf is None else jf.axis_t_raw,
                  "step": self.step_count},
        )
