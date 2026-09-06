#!/usr/bin/env python3
"""fake 플랜트의 **한 팔** — MockArm + 지령 버퍼 + /joint_states 행 (fake_arm_bridge 가 팔마다 하나씩 든다).

이름·부호·홈은 두 갈래로 온다: 레거시 프로필(scripts/robot_profile) 또는 **계약 + policy_control robot yaml**
(`side_spec_from_contract`). 중력·관성은 자산 URDF 에서 — 중력은 pd yaml 의 모델(`pd_gravity.make_gravity`)을
그대로 빌려 pd 노드의 τ_ff 와 같은 식이 되게 한다(플랜트는 제어 경로 검증용이지 모델 정합 실험이 아니다).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_SIM2REAL = Path(__file__).resolve().parents[2]
for _p in (_SIM2REAL / "scripts", _SIM2REAL / "policy_control"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arm_pd_model import MockArm  # noqa: E402

NUM_ARM = 7


@dataclass(frozen=True)
class SideSpec:
    """한 팔의 이름 계약: canonical ↔ source(컨트롤러 joints 순) + 부호 + 관성 계산 자세."""

    side: str
    canonical: tuple            # 7, canonical (l_aj_1 …)
    source: tuple               # 7, source (openarm_left_joint1 …)
    sign: np.ndarray            # 7
    home: np.ndarray            # 7, canonical 순 — 유효관성 기본 자세
    jtc_topic: str              # /<side>_joint_trajectory_controller/joint_trajectory


def side_spec_from_profile(profile, side: str, arm_canonical, home) -> SideSpec:
    """scripts/robot_profile 프로필(레거시) → SideSpec."""
    lim = profile.joint_limits
    return SideSpec(side=side, canonical=tuple(arm_canonical),
                    source=tuple(lim[c]["source"] for c in arm_canonical),
                    sign=np.array([float(lim[c]["sign"]) for c in arm_canonical]),
                    home=np.asarray(home, dtype=float).copy(), jtc_topic=jtc_topic(side))


def side_spec_from_contract(contract, profile: dict, side: str) -> SideSpec:
    """deploy contract sides[side] + 합친 프로필(policy_control.sources.load_profile) → SideSpec."""
    s = contract.side(side)
    can = tuple(s.arm_joints)
    return SideSpec(side=side, canonical=can, source=tuple(profile[c]["source"] for c in can),
                    sign=np.array([float(profile[c]["sign"]) for c in can]),
                    home=np.asarray(s.home_arm, dtype=float).copy(), jtc_topic=jtc_topic(side))


def jtc_topic(side: str) -> str:
    return f"/{side}_joint_trajectory_controller/joint_trajectory"


# ------------------------------------------------------------------ physics inputs (asset URDF)
def inertia_at(urdf: Path, spec: SideSpec, q: np.ndarray | None = None) -> np.ndarray:
    from arm_inertia import effective_inertia

    pose = spec.home if q is None else np.asarray(q, dtype=float)
    if pose.shape != (NUM_ARM,):
        raise ValueError(f"inertia pose needs {NUM_ARM} values, got {pose.shape}")
    return np.asarray(effective_inertia(str(urdf), list(spec.canonical), dict(zip(spec.canonical, pose.tolist()))),
                      dtype=float)


def gravity_from_urdf(urdf: Path, spec: SideSpec, tip_link: str) -> Callable[[np.ndarray], np.ndarray]:
    """자산 URDF 팔 체인의 g(q) (robot_control.kinematics). 손 페이로드는 안 넣는다(레거시 --gravity)."""
    sys.path.insert(0, str(_SIM2REAL.parent / "robot_control" / "src"))
    from robot_control.kinematics import chain_from_urdf

    chain = chain_from_urdf(Path(urdf).read_text(), list(spec.canonical), tip_link)
    return lambda q: chain.gravity_torque(np.asarray(q, dtype=float))


def gravity_from_pd_config(pd_config: Path, contract, side: str) -> Callable[[np.ndarray], np.ndarray]:
    """pd yaml 의 gravity 블록 그대로(팔별 tip_link/payload 포함) → pd 노드와 **같은** τ_ff 모델."""
    from policy_control.pd_gravity import make_gravity
    from policy_control.pd_law import load_pd_config

    cfg = load_pd_config(Path(pd_config))
    return make_gravity(cfg.gravity, contract, side=side)


def driver_gains(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """control_gains.yaml → (kp[7], kd[7]) — pd 노드가 대조하는 바로 그 파일."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    kp = np.array([float(data[f"joint{i}"]["kp"]) for i in range(1, NUM_ARM + 1)])
    kd = np.array([float(data[f"joint{i}"]["kd"]) for i in range(1, NUM_ARM + 1)])
    return kp, kd


# ------------------------------------------------------------------ one arm
class SideArm:
    """MockArm 한 팔 + 지령 버퍼. 지령은 canonical 순(부호 적용 뒤)으로 든다."""

    def __init__(self, spec: SideSpec, *, model: str, max_vel: float, dt: float, kp, kd, fc, inertia,
                 gravity=None) -> None:
        self.spec = spec
        self.q0 = np.zeros(NUM_ARM)
        self.arm = MockArm(q0=self.q0.copy(), model=model, max_vel=max_vel, dt=dt, kp=kp, kd=kd, fc=fc,
                           inertia=inertia, gravity=gravity)
        self.cmd = self.q0.copy()
        self.qd_cmd = np.zeros(NUM_ARM)
        self.tau_ff = np.zeros(NUM_ARM)
        self.received = 0

    # ---------------------------------------------------------------- commands
    def set_jtc(self, joint_names, positions) -> None:
        """JointTrajectory 첫 점(source 이름) → canonical 순 지령. 빠진 관절은 그대로 둔다."""
        idx = {n: i for i, n in enumerate(joint_names)}
        cmd = self.cmd.copy()
        for k, src in enumerate(self.spec.source):
            i = idx.get(src)
            if i is not None and i < len(positions):
                cmd[k] = float(positions[i]) * self.spec.sign[k]
        self.cmd = cmd
        self.received += 1

    def forward_vector(self, data) -> np.ndarray | None:
        """forward 명령(source 순) → canonical 순. 길이가 틀리면 None."""
        if len(data) != NUM_ARM:
            return None
        return np.asarray(data, dtype=float) * self.spec.sign

    def set_forward(self, kind: str, values: np.ndarray) -> None:
        if kind == "position":
            self.cmd = values.copy()
            self.received += 1
        elif kind == "velocity":
            self.qd_cmd = values.copy()
        elif kind == "effort":
            self.tau_ff = values.copy()
        else:
            raise ValueError(f"unknown forward kind {kind!r}")

    # ---------------------------------------------------------------- dynamics / output
    def step(self, mit: bool) -> None:
        """mit=True 면 MIT 3중(q*, q̇*, τ_ff), 아니면 옛 JTC 경로(q* 만)."""
        if mit:
            self.arm.step(self.cmd, qd_cmd=self.qd_cmd, tau_ff=self.tau_ff)
        else:
            self.arm.step(self.cmd)

    @property
    def q(self) -> np.ndarray:
        return self.arm.q

    def rows(self, with_effort: bool) -> tuple[list, list, list, list]:
        """/joint_states 행(source 이름·부호): (names, position, velocity, effort)."""
        q, qd = self.arm.q, self.arm.qd
        tau = self.arm.tau if with_effort else np.zeros(NUM_ARM)
        s = self.spec.sign
        return (list(self.spec.source), (q * s).tolist(), (qd * s).tolist(), (tau * s).tolist())


def static_rows(spec: SideSpec, q: np.ndarray) -> tuple[list, list, list, list]:
    """명령하지 않는 팔의 정적 행(실기 /joint_states 는 양팔을 한 메시지에 싣는다)."""
    pos = (np.asarray(q, dtype=float) * spec.sign).tolist()
    return list(spec.source), pos, [0.0] * NUM_ARM, [0.0] * NUM_ARM
