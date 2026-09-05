#!/usr/bin/env python3
"""우팔 `grasp_s2r` 정책 tick 코어 — ROS/Isaac 무의존 (numpy + 주입된 FK/IK).

한 tick 은 sim `_get_observations` + `_pre_physics_step` 을 1:1 로 따른다:

    실측(팔7·손20·물체·손끝힘)
      → FK(palm pose · 손끝 5)          # 주입된 fabric FK
      → obs 155D                        # grasp_s2r_obs_builder
      → policy(LSTM)                    # action 21D, **±1 클램프**(런 계약)
      → palm 목표 6D                    # 앵커+델타+박스+리미터
      → 닫기 게이트 → 시너지 손 목표 20D
      → fabric 적분 → 팔 관절목표 7D

★★**순서가 두 종류다.** obs 의 손 40칸과 `joint_err` 20칸은 **Isaac DOF 순**
  (`_1` 전 손가락 → `_2` 전 손가락 …)이고, 시너지 자세표는 **프로필 순**
  (`thumb_1..4, index_1..4, …`)이다. 09.01 표본 대조에서 순서를 섞으면 오차가
  0.024 → 1.572 로 뛰었다. 정책은 죽지 않고 **조용히 이상하게 돈다**.
  이 코어는 두 순서를 명시적 순열로 오간다 — 밖에서는 DOF 순만 쓴다.

★액션 클램프는 **런 계약**이다(g1: `clip_actions=1.0`, env 도 `clamp(-1,1)` 한 값을
  obs 에 넣는다). 좌팔 fab79 는 100 이라 자르지 않았다 — 09.03 에 이걸 ±1 로 잘라
  넣어 step 1 부터 궤적이 갈렸다. 그래서 로더에서 런 값을 읽어 넘긴다.

★`fabric_q` 는 **영속 궤적생성기 상태**다. 매 tick 실측으로 재동기화하면 느린 실팔
  위치로 명령이 붕괴해 전진하지 못한다(08.03 실기 RUNNING 동결의 근본원인).

★앵커가 `spawn` 이면 `reset(object_pos=...)` 로 **에피소드 시작 시 한 번** 고정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grasp_s2r_obs_builder import (
    NUM_HAND_DOF,
    assemble_actor_obs,
    hand_dof_order,
    normalized_joint_err,
)
from grasp_s2r_palm_command import PalmCommand, cfg_from_run as palm_cfg_from_run
from grasp_s2r_synergy import (
    HAND_JOINT_NAMES,
    SynergyHand,
    cfg_from_run as syn_cfg_from_run,
)

#: DOF 순 ↔ 프로필 순 순열. 이름으로 만든다 — 슬라이스 금지.
_DOF_NAMES = tuple(hand_dof_order("r"))
DOF_TO_PROFILE = np.array([_DOF_NAMES.index(nm) for nm in HAND_JOINT_NAMES], dtype=int)
PROFILE_TO_DOF = np.array([HAND_JOINT_NAMES.index(nm) for nm in _DOF_NAMES], dtype=int)


@dataclass
class S2RSensors:
    """한 tick 의 실측 입력. 손 관절은 **DOF 순** 20개다."""

    arm_q: np.ndarray            # 7
    arm_qd: np.ndarray           # 7
    hand_q: np.ndarray           # 20, DOF 순
    hand_qd: np.ndarray          # 20, DOF 순
    object_pos: np.ndarray       # 3, 로봇 root 기준
    #: 손끝 힘 (5, 3) **월드 프레임** [N] — 빌더가 팁 자세로 로컬 변환한다.
    #: ⚠실기 `tip_forces_xyz` 는 이미 **팁 로컬**이다. 그 경우 `tip_quat` 에 단위
    #:   쿼터니언을 주면 변환이 항등이 되어 그대로 통과한다.
    tip_force_world: np.ndarray  # (5, 3)
    tip_quat: np.ndarray         # (5, 4) wxyz — 팁 body 자세


@dataclass
class S2RTick:
    """한 tick 의 산출물. 노드는 `arm_q_target` 과 `hand_q_target` 을 발행한다."""

    arm_q_target: np.ndarray     # 7
    hand_q_target: np.ndarray    # 20, **DOF 순**(발행 순서와 맞춘다)
    palm_target: np.ndarray      # 6 = pos3 + euler_zyx3
    action: np.ndarray           # 21, 정책 원출력을 런 계약대로 자른 값
    obs: np.ndarray              # 155
    close_gate: float
    diag: dict = field(default_factory=dict)


def _banded_dist(delta: np.ndarray, deadband: float) -> float:
    """z 데드밴드를 넣은 거리. 파지 높이는 원래 여유가 있는 축이라 z 를 밴드 안에서 0 으로 본다."""
    dz = max(abs(float(delta[2])) - deadband, 0.0)
    return float(np.sqrt(float(delta[0]) ** 2 + float(delta[1]) ** 2 + dz ** 2))


class GraspS2RCore:
    """우팔 s2r 정책 코어.

    Args:
        policy: obs(155,) -> action(21,) — 클램프는 **호출자**(로더)가 런 계약대로 한다.
        fabric_palm_pose: q27 -> palm 6D (pos3 + euler_zyx3)
        fabric_tips: q27 -> (5, 3) 손끝 위치
        fabric_step: (palm6, n) -> 팔 관절목표 7 — 영속 상태로 적분한다
        run_dir: 런 dump 디렉터리(`params/env.yaml`, `params/agent.yaml`)
        goal3: 목표 위치 3 (root 기준)
        soft_limits: 손 20관절 soft limit (20, 2) — **프로필 순**
        hand_dof_to_fabric: DOF 순 → **fabric 관절 순** 인덱스 (20,). 없으면 변환 없음.
    """

    def __init__(self, *, policy, fabric_palm_pose, fabric_tips, fabric_step,
                 run_dir, goal3, soft_limits=None, cage_offset_palm=None,
                 r_cage: float | None = None, hand_dof_to_fabric=None) -> None:
        run = Path(run_dir)
        env_yaml = run / "params/env.yaml"
        self.policy = policy
        self._palm_pose = fabric_palm_pose
        self._tips = fabric_tips
        self._fab_step = fabric_step
        self.goal3 = np.asarray(goal3, dtype=float).reshape(3)

        self.palm = PalmCommand(palm_cfg_from_run(env_yaml))
        self.syn_cfg = syn_cfg_from_run(env_yaml)
        self.hand = SynergyHand(self.syn_cfg, soft_limits=soft_limits)
        self.norm = _norm_from_run(env_yaml)

        # 케이지 — 홈 자세 손끝 FK 에서 재는 값이다. 주지 않으면 첫 reset 에서 만든다.
        self._cage_off = (None if cage_offset_palm is None
                          else np.asarray(cage_offset_palm, dtype=float).reshape(3))
        self._r_cage = r_cage
        self._dof_to_fab = (None if hand_dof_to_fabric is None
                            else np.asarray(hand_dof_to_fabric, dtype=int).reshape(-1))
        self._last_action = np.zeros(21)
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, *, arm_q, hand_q, object_pos) -> None:
        """에피소드 시작. ★`spawn` 앵커는 여기서 물체를 한 번 스냅샷한다."""
        self.palm.reset(object_spawn_pos=object_pos)
        self.hand.reset(hand_q=np.asarray(hand_q, dtype=float)[DOF_TO_PROFILE])
        self._last_action = np.zeros(21)
        self.step_count = 0
        if self._cage_off is None or self._r_cage is None:
            self._calibrate_cage(arm_q, hand_q)

    def _calibrate_cage(self, arm_q, hand_q) -> None:
        """홈 자세에서 케이지 중심·반경을 잰다(`_report_home_cage` 이식).

        ★중심을 **palm 에 강체로 붙인다.** 실시간 손끝 평균으로 두면 팔을 안 움직이고
          손만 오므려도 중심이 컵 쪽으로 당겨져 게이트가 저절로 열린다(08.27 실측
          corr −0.974) — "정렬되면 닫아라"가 "닫으면 닫아도 된다"가 되어 무의미해진다.
        """
        palm6, R, tips = self._fk(arm_q, hand_q)
        others = tips[1:].mean(axis=0)          # tip 순서는 canonical(thumb 먼저)
        cage = 0.5 * (tips[0] + others)
        self._r_cage = 0.5 * float(np.linalg.norm(tips[0] - others))
        self._cage_off = R.T @ (cage - palm6[:3])

    # ------------------------------------------------------------------
    def _fk(self, arm_q, hand_q):
        """실측 → (palm6, R(3,3), tips(5,3)). 손 관절은 **DOF 순**으로 받는다.

        ★★Fabrics 는 **자기 관절 순서**를 쓴다 — DOF 순을 그대로 넘기면 손끝이 최대
          **148 mm** 어긋난다(09.03 sim 대조 실측: thumb 148.1 · middle 145.7 ·
          ring 31.8 mm, index·pinky 는 우연히 0.0). env 도 `_syn_to_fab_idx` 로
          변환한다. 순서를 바로잡으면 손끝 5개가 전부 0.0 mm 로 일치한다.
          `hand_dof_to_fabric` 을 주지 않으면 변환 없이 넘긴다 —
          그 경우 호출자가 이미 fabric 순으로 준다는 뜻이다.
        """
        hand = np.asarray(hand_q, dtype=float).reshape(NUM_HAND_DOF)
        if self._dof_to_fab is not None:
            hand = hand[self._dof_to_fab]
        q27 = np.concatenate([np.asarray(arm_q, dtype=float).reshape(7), hand])
        palm6 = np.asarray(self._palm_pose(q27), dtype=float).reshape(6)
        tips = np.asarray(self._tips(q27), dtype=float).reshape(5, 3)
        return palm6, _rot_euler_zyx(palm6[3:]), tips

    def close_gate(self, palm_pos, R, obj_pos) -> float:
        """닫기 게이트 [0,1]. 케이지가 컵에 정렬돼야 오므릴 수 있다."""
        if not self.syn_cfg_close_gate_enabled:
            return 1.0
        if self._r_cage is None or self._cage_off is None:
            raise SystemExit("[s2r] 케이지가 아직 없다 — reset() 을 먼저 부를 것")
        cage = palm_pos + R @ self._cage_off
        d = _banded_dist(cage - obj_pos, self.norm["grasp_z_deadband"])
        r = float(self._r_cage)
        ramp = max(self.norm["close_gate_ramp"] * r, 1e-6)
        return float(np.clip((r - d) / ramp, 0.0, 1.0))

    @property
    def syn_cfg_close_gate_enabled(self) -> bool:
        return bool(self.norm["close_gate_enabled"])

    # ------------------------------------------------------------------
    def step(self, s: S2RSensors) -> S2RTick:
        palm6, R, tips = self._fk(s.arm_q, s.hand_q)
        palm_quat = _quat_from_matrix(R)
        obj = np.asarray(s.object_pos, dtype=float).reshape(3)

        hand_q_prof = np.asarray(s.hand_q, dtype=float)[DOF_TO_PROFILE]
        # ★★`joint_err` 은 **프로필 순**이다 — obs 의 `hand_q`(DOF 순)와 순서가 다르다.
        #   env `_joint_pos_err()` 가 `_syn_ids`(프로필 순)로 재기 때문이다.
        #   09.03 sim 대조에서 DOF 순으로 만들었다가 오차가 스텝마다 커졌다
        #   (0.0022 → 0.0087) — 관절값이 0 근처라 스크램블이 반올림처럼 보였다.
        joint_err = normalized_joint_err(hand_q_prof, self.hand.target,
                                         self.norm["joint_pos_err_max"])

        obs = assemble_actor_obs(
            arm_q=s.arm_q, arm_qd=s.arm_qd,
            hand_q=s.hand_q, hand_qd=s.hand_qd, joint_err_profile_order=joint_err,
            palm_pos=palm6[:3], palm_quat=palm_quat,
            tip_pos=tips, cup_pos=obj, goal_pos=self.goal3,
            tip_force_world=s.tip_force_world, tip_quat=s.tip_quat,
            last_action=self._last_action,
            contact_force_max=self.norm["contact_force_max"],
            joint_pos_err_max=self.norm["joint_pos_err_max"],
        )

        action = np.asarray(self.policy(obs), dtype=float).reshape(21)
        self._last_action = action.copy()

        palm_target = self.palm.step(action[:6])
        gate = self.close_gate(palm6[:3], R, obj)
        # ★실속 동결은 실기 대체 신호다 — 오탐이 잦으면 끌 수 있어야 한다.
        #   끄면 `hand_q` 를 넘기지 않아 `_freeze_mask` 가 None 을 돌려준다.
        hand_prof = self.hand.step(
            action[6:], close_gate=gate,
            hand_q=hand_q_prof if getattr(self, "hand_stall_freeze", True) else None)
        arm_target = np.asarray(self._fab_step(palm_target), dtype=float).reshape(7)

        self.step_count += 1
        return S2RTick(
            arm_q_target=arm_target,
            hand_q_target=hand_prof[PROFILE_TO_DOF],
            palm_target=palm_target, action=action, obs=obs, close_gate=gate,
            diag={"box_sat": self.palm.state.box_sat.copy(),
                  "cmd_step_raw": self.palm.state.step_raw,
                  "syn_close_mean": float(self.hand.close[self.hand.movable].mean()),
                  "r_cage": float(self._r_cage) if self._r_cage else 0.0},
        )


# ---------------------------------------------------------------------------
def _rot_euler_zyx(e) -> np.ndarray:
    """euler_zyx (ez, ey, ex) → 회전행렬. fabric palm pose 규약과 같다."""
    ez, ey, ex = (float(v) for v in np.asarray(e, dtype=float).reshape(3))
    cz, sz = np.cos(ez), np.sin(ez)
    cy, sy = np.cos(ey), np.sin(ey)
    cx, sx = np.cos(ex), np.sin(ex)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """회전행렬 → 쿼터니언 wxyz. 빌더가 `palm_quat` 로 rot6d 를 만든다."""
    m = np.asarray(R, dtype=float).reshape(3, 3)
    t = float(np.trace(m))
    if t > 0.0:
        sq = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * sq, (m[2, 1] - m[1, 2]) / sq,
                         (m[0, 2] - m[2, 0]) / sq, (m[1, 0] - m[0, 1]) / sq])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    sq = np.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2.0
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / sq
    q[i + 1] = 0.25 * sq
    q[j + 1] = (m[j, i] + m[i, j]) / sq
    q[k + 1] = (m[k, i] + m[i, k]) / sq
    return q


def _norm_from_run(env_yaml_path) -> dict:
    """정규화·게이트 상수를 런에서 읽는다 — 손으로 옮기지 않는다."""
    import re

    text = Path(env_yaml_path).read_text()

    def val(key, default):
        m = re.search(rf"^\s*{key}:\s*(.+?)\s*$", text, re.M)
        if m is None:
            return default
        v = m.group(1).strip()
        if v in ("true", "false"):
            return v == "true"
        try:
            return float(v)
        except ValueError:
            return default

    return {
        "contact_force_max": float(val("contact_force_max", 10.0)),
        "joint_pos_err_max": float(val("joint_pos_err_max", 1.2)),
        "grasp_z_deadband": float(val("grasp_z_deadband", 0.03)),
        "close_gate_enabled": bool(val("close_gate_enabled", True)),
        "close_gate_ramp": float(val("close_gate_ramp", 0.5)),
    }
