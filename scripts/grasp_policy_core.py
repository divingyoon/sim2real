#!/usr/bin/env python3
"""grasp_v1 정책 tick 코어 — ROS 무의존 (numpy/torch/fabrics).

`grasp_inference`(라이브 노드)와 `grasp_loop_sim`(오프라인 재현기)이 **같은 코어**를
쓰게 해 드리프트를 막는다. 과거엔 "1:1 유지할 것"을 주석으로만 강제했고 실제로 갈라졌다
(loop_sim 은 매 tick fabric_q 를 실측 재동기화, 노드는 안 함).

tick 흐름 (sim `_pre_physics_step` + `_apply_action` 1:1):

    실측 관절 → Fabrics FK(palm·fingertip) → 접촉 거리게이트 → obs 154D
      → policy → action 21D → lift 래치 → 손가락 20D → 팔 7D(Fabrics IK 또는 lift 보간)

★자산은 구성 프로필에서 주입한다. 기본값에 의존하면 sim 이 자산을 옮겼을 때 조용히
  어긋난다 — 실제로 palm 이 6.5cm 어긋난 채 동작한 이력이 있다.

★fabric_q 는 매 tick 실측으로 재동기화하지 않는다. sim 과 동일하게 fabric_q 는 영속
  궤적생성기 상태(로봇이 추종할 명령)다. 실측 동기화하면 느린 실팔 위치로 명령이 붕괴해
  전진 불가(08.03 실기 RUNNING 동결 근본원인).

무거운 의존(torch/fabrics)은 `GraspPolicyCore` 안에서 **지연 import** 한다 —
아래 순수 함수들은 numpy 만으로 테스트 가능해야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NUM_ARM_DOF = 7
NUM_HAND_DOF = 20
NUM_FINGERTIPS = 5


# ---------------------------------------------------------------------------
# 순수 함수 (numpy only — fabrics 없이 테스트 가능)
# ---------------------------------------------------------------------------
def home_pose_to_radians(reset_home_palm_pose) -> np.ndarray:
    """cfg (x, y, z, ez°, ey°, ex°) → (6,) 라디안 pose."""
    hp = np.asarray(reset_home_palm_pose, dtype=np.float64).reshape(-1)
    if hp.shape != (6,):
        raise ValueError(f"reset_home_palm_pose expected 6 values, got {hp.shape}")
    return np.concatenate([hp[:3], np.radians(hp[3:6])])


def validate_home_in_workspace(home_pose, palm_mins, palm_maxs) -> None:
    """홈이 palm workspace 밖이면 **예외**. 조용히 클램프하지 않는다(sim 과 동일).

    조용히 클램프하면 정책이 학습한 기준점과 다른 곳에서 delta 를 더하게 되어
    액션 의미가 통째로 어긋난다.
    """
    h = np.asarray(home_pose, dtype=np.float64).reshape(-1)
    lo = np.asarray(palm_mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(palm_maxs, dtype=np.float64).reshape(-1)
    if np.any(h < lo) or np.any(h > hi):
        raise ValueError(
            f"홈 palm pose 가 workspace 밖이다: {h.round(4).tolist()}\n"
            f"  mins={lo.round(4).tolist()}\n  maxs={hi.round(4).tolist()}"
        )


#: 컵 정준 pregrasp 의 palm 자세(도) — sim 리터럴 `math.radians(90/0/90)` 과 1:1.
PREGRASP_PALM_ROT_DEG = (90.0, 0.0, 90.0)


def palm_target_from_delta(anchor_pose, delta, palm_mins, palm_maxs) -> np.ndarray:
    """**액션 기준점** + Δ → workspace clamp 된 palm 목표 (sim `_pre_physics_step` 1:1).

    clamp 범위를 기준점으로 **완화**한다(min(palm_mins, anchor) / max(palm_maxs, anchor)):
    기준점 자체는 언제나 도달 가능해야 action=0 이 "기준점 유지"를 뜻한다.

    ★기준점은 구성마다 다르다 — `grasp_v1` 은 홈, `grasp_sensor` 는 컵 정준 pregrasp.
    `action_anchor_pose()` 로 고른다. 홈을 넣으면 컵 기준 구성에서 팔이 홈 근처에서만
    움직인다(2026-08-20 발견).
    """
    base = np.asarray(anchor_pose, dtype=np.float64).reshape(-1)
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    lo = np.minimum(np.asarray(palm_mins, dtype=np.float64).reshape(-1), base)
    hi = np.maximum(np.asarray(palm_maxs, dtype=np.float64).reshape(-1), base)
    return np.clip(base + d, lo, hi)


def pregrasp_anchor_pose(cup_pos, cfg, palm_mins, palm_maxs) -> np.ndarray:
    """컵 pose → 컵 정준 pregrasp palm 6D (sim `_reset_idx` 1:1).

        pregrasp = clamp(cup_pos + (offset_x, offset_y, offset_z), palm_mins, palm_maxs)
        rot      = (90°, 0°, 90°)

    sim 은 여기에 DR 노이즈(`pregrasp_noise_*`)를 더하지만 **배포는 더하지 않는다** —
    그 노이즈는 기준점 오차에 정책을 강인하게 만들려는 학습 장치이지 배포 시 재현할
    대상이 아니다.
    """
    cup = np.asarray(cup_pos, dtype=np.float64).reshape(-1)
    if cup.shape[0] != 3:
        raise ValueError(f"컵 위치는 3D 여야 한다: {cup.shape}")
    offset = np.array(
        [cfg["pregrasp_offset_x"], cfg["pregrasp_offset_y"], cfg["pregrasp_offset_z"]],
        dtype=np.float64,
    )
    pose = np.empty(6, dtype=np.float64)
    pose[:3] = cup + offset
    pose[3:] = np.radians(PREGRASP_PALM_ROT_DEG)
    lo = np.asarray(palm_mins, dtype=np.float64).reshape(-1)
    hi = np.asarray(palm_maxs, dtype=np.float64).reshape(-1)
    return np.clip(pose, lo, hi)


def action_anchor_pose(anchor: str, home_pose, cup_pos, cfg, palm_mins, palm_maxs) -> np.ndarray:
    """구성의 액션 기준점을 고른다. `anchor` 는 `robot_profile.load_action_anchor()` 산출."""
    if anchor == "home":
        return np.asarray(home_pose, dtype=np.float64).reshape(-1).copy()
    if anchor == "cup":
        return pregrasp_anchor_pose(cup_pos, cfg, palm_mins, palm_maxs)
    raise ValueError(f"알 수 없는 anchor: {anchor!r} (home | cup)")


def require_anchor_established(anchor: str, established: bool) -> None:
    """컵 기준 구성에서 기준점이 아직 안 잡혔으면 **예외**.

    조용히 홈으로 되돌아가면 팔이 홈 근처에서만 움직이고(증상 B) 로그에는 아무것도 안
    남는다. 명령을 내는 지점에서 시끄럽게 실패시킨다.
    """
    if anchor == "cup" and not established:
        raise RuntimeError(
            "액션 기준점이 아직 확립되지 않았다 — anchor='cup' 구성은 "
            "reset_episode(..., cup_pos=...) 로 컵 pose 를 한 번 잡아야 한다"
        )


def gate_tip_contact(
    tip_force_local, palm_cup_dist: float, gate_dist: float, threshold: float,
    gate_force_obs: bool = True,
):
    """거리 게이트를 적용한 (tip_force(5,3), binary(5,)) 반환.

    실물 tip F/T 는 **테이블 접촉도** 잡는다(sim 접촉센서는 컵-필터 쌍이라 테이블 무감지).
    palm 이 컵에서 멀면 접촉이 컵 유래일 수 없으므로 0 처리한다 — 시작 직후 거짓 lift
    래치 차단(08.03 step7, dist 0.16m 에서 발동한 사고).

    gate_force_obs=True 면 obs 로 나가는 **힘 자체도** 0 으로 만든다(sim 정합).
    False 면 이진 접촉만 게이트하고 힘은 그대로 흘린다.
    """
    f = np.asarray(tip_force_local, dtype=np.float64)
    if f.shape != (NUM_FINGERTIPS, 3):
        raise ValueError(f"tip_force_local expected (5,3), got {f.shape}")
    near = float(palm_cup_dist) < float(gate_dist)
    if not near:
        binary = np.zeros(NUM_FINGERTIPS, dtype=np.float64)
        return (np.zeros_like(f) if gate_force_obs else f), binary
    binary = (np.linalg.norm(f, axis=1) > float(threshold)).astype(np.float64)
    return f, binary


# ---------------------------------------------------------------------------
# tick 입출력
# ---------------------------------------------------------------------------
@dataclass
class TickSensors:
    arm_pos: np.ndarray          # (7,)
    arm_vel: np.ndarray          # (7,)
    hand_pos: np.ndarray         # (20,)
    hand_vel: np.ndarray         # (20,)
    cup_pos: np.ndarray          # (3,)
    tip_force_local: np.ndarray  # (5,3) [N] tip-local 원시값


@dataclass
class TickOutput:
    arm_cmd: np.ndarray          # (7,)
    hand_cmd: np.ndarray         # (20,)
    action: np.ndarray           # (21,) 정책 원출력
    obs: np.ndarray              # (154,)
    palm_center: np.ndarray      # (3,)
    palm_cup_dist: float
    tip_contact: np.ndarray      # (5,) 게이트 후 이진
    tip_force_gated: np.ndarray  # (5,3) 게이트 후 힘
    is_lift: bool
    just_entering_lift: bool
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 정책 tick 코어 (torch/fabrics 지연 import)
# ---------------------------------------------------------------------------
class GraspPolicyCore:
    """구성 프로필 + 정책 → tick 단위 관절 명령 생성기.

    profile:  robot_profile.RobotProfile (자산·토픽·관절·계약)
    policy:   `.get_action(obs_tensor) -> (1, 21)` / `.reset_states()` 를 가진 객체
    """

    def __init__(
        self,
        profile,
        policy,
        device: str = "cuda:0",
        contact_threshold: float | None = None,
        contact_gate_dist: float = 0.10,
        gate_force_obs: bool = True,
        object_onehot: np.ndarray | None = None,
    ) -> None:
        import torch                                    # 지연 import
        from robot_profile import load_hdgp_module, load_profile_env_cfg

        self.profile = profile
        self.policy = policy
        self.device = device
        self._torch = torch
        self.contact_gate_dist = float(contact_gate_dist)
        self.gate_force_obs = bool(gate_force_obs)

        preset = load_hdgp_module(profile, "preset")
        consts = load_hdgp_module(profile, "constants")
        side_upper = profile.acting_side.upper()

        self.hand_approach = np.asarray(preset.HAND_APPROACH_POSE, dtype=np.float64)
        self.hand_full_grip = np.asarray(preset.HAND_FULL_GRIP_POSE, dtype=np.float64)
        self.hand_grasp = np.asarray(preset.HAND_GRASP_POSE, dtype=np.float64)
        self.arm_start = np.asarray(
            getattr(preset, f"{side_upper}_ARM_START_POSE"), dtype=np.float64
        )
        self.contact_threshold = float(
            consts.CONTACT_FORCE_THRESHOLD if contact_threshold is None else contact_threshold
        )
        self.lift_phase_steps = int(consts.LIFT_PHASE_STEPS)
        self.pregrasp_fabric_steps = int(consts.PREGRASP_FABRICS_STEPS)
        self.episode_steps = int(consts.EPISODE_STEPS)
        self.contact_force_max = float(consts.CONTACT_FORCE_MAX)
        self.joint_pos_err_max = float(consts.JOINT_POS_ERR_MAX)

        cfg = load_profile_env_cfg(profile)
        self.cfg = cfg
        self.palm_mins = np.asarray(preset.palm_pose_mins(cfg["max_pose_angle"]), dtype=np.float64)
        self.palm_maxs = np.asarray(preset.palm_pose_maxs(cfg["max_pose_angle"]), dtype=np.float64)

        dx, dy, dz = cfg["palm_delta_xyz"]
        dr = math.radians(cfg["palm_delta_rot_deg"])
        self.delta_mins = np.array([-dx, -dy, -dz, -dr, -dr, -dr], dtype=np.float64)
        self.delta_maxs = -self.delta_mins

        self.home_palm_pose = home_pose_to_radians(cfg["reset_home_palm_pose"])
        validate_home_in_workspace(self.home_palm_pose, self.palm_mins, self.palm_maxs)

        # 액션 델타의 기준점. **sim 소스에서 유도**한다 — 프로필에 적어두면 sim 이 바뀔 때
        # 조용히 어긋난다(2026-08-20: grasp_sensor 가 홈→컵으로 옮겼는데 배포는 홈이었다).
        from robot_profile import load_action_anchor

        self.action_anchor = load_action_anchor(profile)
        # reset_episode 전까지는 홈. cup anchor 구성은 reset 에서 컵 pose 로 덮어쓴다.
        self.anchor_palm_pose = self.home_palm_pose.copy()
        #: cup anchor 에서 "컵 pose 로 기준점을 잡았는가". 생성자의 초기 리셋은 fabric
        #: 버퍼 초기화가 목적이라 이 플래그를 세우지 않는다 — 명령 시점에서 막는다.
        self._anchor_established = False

        from grasp_action_decoder import GraspFingerController, LiftLatch
        from grasp_obs_builder import make_object_onehot, REAL_CUP_INDEX
        from robot_profile import ee_limit_arrays

        lo, hi = ee_limit_arrays(profile)
        self.finger_ctrl = GraspFingerController(
            self.hand_approach, self.hand_full_grip,
            close_speed=cfg["finger_close_speed"],
            lower_limits=lo, upper_limits=hi,               # ★D3 관절 한계 주입
            couple_four=cfg["couple_four_fingers"],
            retighten_after_latch=cfg["retighten_after_latch"],
        )
        self.lift_latch = LiftLatch(
            min_contacts=cfg["lift_start_min_grip_fingers"],
            hold_steps=cfg["grasp_ready_hold_steps"],
        )
        self.object_onehot = (
            make_object_onehot(REAL_CUP_INDEX) if object_onehot is None
            else np.asarray(object_onehot, dtype=np.float64)
        )

        self._setup_fabrics()
        self.q_home_arm = self.build_home()
        # 초기화용 리셋 — 기준점은 아직 확립하지 않는다(cup anchor 는 컵 pose 가 필요).
        self.reset_episode(self.q_home_arm, self.hand_approach, _establish_anchor=False)

    # -- Fabrics -----------------------------------------------------------
    def _setup_fabrics(self) -> None:
        """구성 프로필이 지정한 자산으로 메인/리셋 fabric 을 만든다."""
        import importlib

        import torch
        from fabrics_sim.integrator.integrators import DisplacementIntegrator
        from fabrics_sim.utils.utils import initialize_warp
        from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

        fab = self.profile.fabrics
        initialize_warp(self.device[-1])
        mod = importlib.import_module("fabrics_sim.fabrics.openarm_tesollo_pose_fabric")
        FabricCls = getattr(mod, fab.class_name)

        self.world_model = WorldMeshesModel(
            batch_size=1, max_objects_per_env=self.cfg["fabrics_max_objects_per_env"],
            device=self.device, world_filename=fab.world,
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        common = dict(
            graph_capturable=False, use_hand_fabric=False,
            robot_dir_name=fab.robot_dir, robot_name=fab.robot_name,
        )
        self.fabric = FabricCls(1, self.device, self.cfg["fabrics_dt"], **common)
        self.integrator = DisplacementIntegrator(self.fabric)
        # ★D1: cspace attractor 의 손은 GRASP_POSE (sim `:413-415`). FULL_GRIP 이 아니다.
        cs = self.fabric.default_config.clone()
        cs[:, NUM_ARM_DOF:] = self._t(self.hand_grasp)
        self.fabric.default_config.copy_(cs)

        # ★D2: 홈 IK 전용 fabric — damping 10.0, cspace 손 GRASP_POSE (sim `:583-585`)
        self._reset_fabric = FabricCls(1, self.device, self.cfg["fabrics_dt"], **common)
        self._reset_integrator = DisplacementIntegrator(self._reset_fabric)
        rcs = self._reset_fabric.default_config.clone()
        rcs[:, NUM_ARM_DOF:] = self._t(self.hand_grasp)
        self._reset_fabric.default_config.copy_(rcs)
        self._reset_damping = self.cfg["reset_fabrics_damping_gain"] * torch.ones(
            1, 1, device=self.device
        )
        self.damping_gain = self.cfg["fabrics_damping_gain"] * torch.ones(
            1, 1, device=self.device
        )
        self._pca_zero = torch.zeros(1, 5, device=self.device)
        self.num_joints = self.fabric.num_joints

    def _t(self, arr):
        return self._torch.as_tensor(
            np.asarray(arr, dtype=np.float64), dtype=self._torch.float32, device=self.device
        )

    # -- 홈 IK -------------------------------------------------------------
    def build_home(self) -> np.ndarray:
        """고정 홈 palm 자세를 Fabrics IK 로 풀어 q_home(7,) 반환 (sim `_build_home_pose`).

        sim 과 동일하게 **단일 set_features + N step**(수렴 판정 없음), damping 10.
        결과가 sim preset 에서 유도한 기준값과 어긋나면 예외 — 자산/월드 불일치의 신호다.
        """
        torch = self._torch
        q_init = np.concatenate([self.arm_start, self.hand_approach])
        fq = self._t(q_init).unsqueeze(0).contiguous()
        fqd = torch.zeros(1, self.num_joints, device=self.device)
        fqdd = torch.zeros(1, self.num_joints, device=self.device)
        palm = self._t(self.home_palm_pose).unsqueeze(0)

        self._reset_fabric.set_features(
            self._pca_zero, palm, "euler_zyx",
            fq.detach(), fqd.detach(),
            self.object_ids, self.object_indicator, self._reset_damping,
        )
        for _ in range(self.pregrasp_fabric_steps):
            fq, fqd, fqdd = self._reset_integrator.step(
                fq.detach(), fqd.detach(), fqdd.detach(), self.cfg["fabrics_dt"]
            )
        q_home = fq[0, :NUM_ARM_DOF].detach().cpu().numpy().astype(np.float64)
        self._check_home_against_sim(q_home)
        return q_home

    def _check_home_against_sim(self, q_home: np.ndarray, tol: float = 0.05) -> None:
        """sim preset 이 강제하는 기준값과 대조 — ★자산 불일치 재발 방어선.

        기준값은 `robot_profile.expected_q_home_arm()` 이 반대편 preset rest 와
        `_ARM_MIRROR_SIGN`(env 소스)에서 유도한다. 예전엔 preset 에서 부호를 찾으려다
        항상 None 이 되어 **검증이 조용히 통과**했다 — 그 구멍을 막았다.
        """
        from robot_profile import expected_q_home_arm

        want = np.asarray(expected_q_home_arm(self.profile), dtype=np.float64)
        err = float(np.abs(q_home - want).max())
        if err > tol:
            raise RuntimeError(
                f"홈 IK 결과가 sim 기준값과 어긋난다 (최대 {err:.4f} rad > {tol}).\n"
                f"  배포 q_home = {q_home.round(4).tolist()}\n"
                f"  sim  기준값 = {want.round(4).tolist()}\n"
                f"  Fabrics 자산({self.profile.fabrics.robot_dir}/{self.profile.fabrics.world})이"
                " sim 학습과 다른지 먼저 의심하라 — 구 자산은 palm 이 6.5cm 짧다."
            )

    # -- 에피소드 -----------------------------------------------------------
    def reset_episode(self, arm_pos, hand_pos, cup_pos=None, *, _establish_anchor=True) -> None:
        """에피소드 시작 상태로 되돌린다.

        hand_cmd_prev 초기값은 **APPROACH**다 — zeros 금지. sim 리셋이
        `hand_joint_targets = approach_hand` 로 두는 값과 같아야 첫 tick 의
        joint_pos_err 이 sim 과 정합한다.

        ★`cup_pos` 는 anchor="cup" 구성에서 **필수**다. sim 은 리셋 시점의 컵 위치로
        기준점을 한 번 정하고 에피소드 내내 고정하므로(`pregrasp_palm_pose_buf`),
        배포도 여기서 한 번 잡고 이후 갱신하지 않는다 — 매 tick 컵을 추종하면 컵이
        밀릴 때 기준점이 따라가 학습과 달라진다.
        cup_pos 없이 cup anchor 를 리셋하면 **예외**다. 홈으로 조용히 되돌아가면
        팔이 홈 근처에서만 움직이는 바로 그 증상이 된다.
        """
        torch = self._torch
        if _establish_anchor:
            if self.action_anchor == "cup" and cup_pos is None:
                raise ValueError(
                    "anchor='cup' 구성은 리셋에 컵 pose 가 필요하다 "
                    "(홈으로 대체하면 팔이 홈 근처에서만 움직인다)"
                )
            self.anchor_palm_pose = action_anchor_pose(
                self.action_anchor, self.home_palm_pose,
                cup_pos if cup_pos is not None else np.zeros(3),
                self.cfg, self.palm_mins, self.palm_maxs,
            )
            self._anchor_established = True
        q0 = np.concatenate([
            np.asarray(arm_pos, dtype=np.float64).reshape(-1)[:NUM_ARM_DOF],
            np.asarray(hand_pos, dtype=np.float64).reshape(-1)[:NUM_HAND_DOF],
        ])
        self.fabric_q = self._t(q0).unsqueeze(0).contiguous()
        self.fabric_qd = torch.zeros(1, self.num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(1, self.num_joints, device=self.device)
        # null-space attractor 를 홈으로 — 에피소드 시작 시 항이 ≈0 이라 흔들림이 없다
        self.fabric.default_config[:, :NUM_ARM_DOF] = self._t(self.q_home_arm)

        self.finger_ctrl.reset()
        self.lift_latch.reset()
        self.hand_cmd_prev = self.hand_approach.copy()      # ★zeros 금지
        self.last_actions = np.zeros(21, dtype=np.float64)
        self.lift_arm_start = None
        self.prelift_target = None
        self.lift_start_step = 0
        if hasattr(self.policy, "reset_states"):
            self.policy.reset_states()

    # -- FK ----------------------------------------------------------------
    def forward_kinematics(self, arm_pos, hand_pos):
        """실측 관절 → (palm_center(3,), fingertip_pos(5,3)). sim obs 와 같은 기준."""
        torch = self._torch
        q = self._t(np.concatenate([
            np.asarray(arm_pos, dtype=np.float64).reshape(-1),
            np.asarray(hand_pos, dtype=np.float64).reshape(-1),
        ])).unsqueeze(0)
        with torch.inference_mode():
            palm6 = self.fabric.get_palm_pose(q, "euler_zyx")
            tips = self.fabric.get_fingertip_positions(q)
        return (
            palm6[0, :3].cpu().numpy().astype(np.float64),
            tips[0].cpu().numpy().astype(np.float64),
        )

    # -- tick --------------------------------------------------------------
    def step(self, s: TickSensors, step_count: int) -> TickOutput:
        """정책 1 tick — sim `_pre_physics_step` + `_apply_action` 1:1."""
        from grasp_obs_builder import assemble_actor_obs

        palm_center, tips = self.forward_kinematics(s.arm_pos, s.hand_pos)
        dist = float(np.linalg.norm(palm_center - np.asarray(s.cup_pos, dtype=np.float64)))
        tip_force, tip_contact = gate_tip_contact(
            s.tip_force_local, dist, self.contact_gate_dist,
            self.contact_threshold, self.gate_force_obs,
        )

        obs_np = assemble_actor_obs(
            arm_joint_pos=s.arm_pos, arm_joint_vel=s.arm_vel,
            finger_joint_pos=s.hand_pos, finger_joint_vel=s.hand_vel,
            palm_center=palm_center, fingertip_pos=tips, cup_pos=s.cup_pos,
            tip_force_local=tip_force,
            hand_cmd_prev=self.hand_cmd_prev,
            last_actions=self.last_actions,
            object_onehot=self.object_onehot,
            contact_force_max=self.contact_force_max,
            joint_pos_err_max=self.joint_pos_err_max,
        )
        obs = self._torch.as_tensor(
            obs_np, dtype=self._torch.float32, device=self.device
        ).unsqueeze(0)
        action = self.policy.get_action(obs)[0].cpu().numpy().astype(np.float64)
        self.last_actions = action.copy()          # ★coupling 이전 원출력

        was_latched = self.lift_latch.latched
        is_lift = self.lift_latch.update(int(tip_contact.sum()))
        just_entering = bool(is_lift and not was_latched)

        hand_cmd = self.finger_ctrl.step(
            action[6:21], tip_contact, latched=is_lift
        )   # distal 미배선 → tip-only (라이브 의도)

        if not is_lift:
            arm_cmd = self._grasp_arm_cmd(action[:6], hand_cmd)
        else:
            arm_cmd = self._lift_arm_cmd(s.arm_pos, step_count, just_entering)

        self.hand_cmd_prev = np.asarray(hand_cmd, dtype=np.float64).copy()
        return TickOutput(
            arm_cmd=arm_cmd, hand_cmd=np.asarray(hand_cmd, dtype=np.float64),
            action=action, obs=obs_np, palm_center=palm_center, palm_cup_dist=dist,
            tip_contact=tip_contact, tip_force_gated=tip_force,
            is_lift=bool(is_lift), just_entering_lift=just_entering,
        )

    def _grasp_arm_cmd(self, palm_action, hand_cmd) -> np.ndarray:
        """Δpalm → Fabrics IK. fabric_q 는 영속 상태로 적분(실측 재동기화 금지)."""
        from grasp_action_decoder import scale_palm_delta

        require_anchor_established(self.action_anchor, self._anchor_established)
        delta = scale_palm_delta(palm_action, self.delta_mins, self.delta_maxs)
        palm_pose = palm_target_from_delta(
            self.anchor_palm_pose, delta, self.palm_mins, self.palm_maxs
        )
        # sim parity: fabric_q 의 손 부분은 손 명령으로 동기화(FK·null-space 용)
        self.fabric_q[0, NUM_ARM_DOF:] = self._t(hand_cmd)
        self.fabric_qd[0, NUM_ARM_DOF:] = 0.0
        self.fabric.set_features(
            self._pca_zero, self._t(palm_pose).unsqueeze(0), "euler_zyx",
            self.fabric_q.detach(), self.fabric_qd.detach(),
            self.object_ids, self.object_indicator, self.damping_gain,
        )
        for _ in range(self.cfg["fabric_decimation"]):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(),
                self.fabric_qdd.detach(), self.cfg["fabrics_dt"],
            )
        return self.fabric_q[0, :NUM_ARM_DOF].detach().cpu().numpy().astype(np.float64)

    def _lift_arm_cmd(self, arm_pos, step_count: int, just_entering: bool) -> np.ndarray:
        """joint7-only lift-wait 선형보간. lift 중 fabric arm 은 실측으로 동결(발산 방지)."""
        from grasp_action_decoder import joint7_lift_wait_target, lift_arm_interp

        arm = np.asarray(arm_pos, dtype=np.float64).reshape(-1)
        self.fabric_q[0, :NUM_ARM_DOF] = self._t(arm)
        self.fabric_qd[0, :NUM_ARM_DOF] = 0.0
        self.fabric_qdd[0, :NUM_ARM_DOF] = 0.0
        if just_entering:
            self.lift_arm_start = arm.copy()
            self.prelift_target = joint7_lift_wait_target(
                arm,
                joint7_delta=self.cfg["lift_wait_joint7_delta"],
                joint7_min=self.cfg["warm_j7_min"],
                joint7_max=self.cfg["warm_j7_max"],
            )
            self.lift_start_step = step_count
        progress = min(
            float(step_count - self.lift_start_step) / max(1, self.lift_phase_steps - 1), 1.0
        )
        return lift_arm_interp(self.lift_arm_start, self.prelift_target, progress)
