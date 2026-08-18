#!/usr/bin/env python3
"""실기 팔의 추종 능력 모델 (numpy only, 정책·ROS 무의존).

sim 은 팔을 `set_joint_position_target` + stiffness 400 / damping 80 으로 굴린다. 실기는
JTC(position) → CAN MIT 펌웨어 PD 이고 게인이 kp 70·60·10 / kd 2.75~0.5 수준이다.
즉 **sim 팔이 실기보다 훨씬 잘 추종한다** — 정책이 droop 과 지연을 겪어본 적이 없다.

이 모듈은 그 격차를 오프라인에서 재현·정량화한다:

    tau  = kp(sp − q) − kd·qd − Fc·sgn(qd)
    qdd  = tau / I
  sp = 브리지 rate-limit 을 거친 세트포인트, I = 자산 URDF 에서 계산한 유효 관성.

★게인도 관성도 **지어내지 않는다**. 게인은 실측 캘리브(r2s autotune), 관성은 URDF.
  캘리브 파일이 없으면 예외를 던진다 — 추정값으로 돌린 실험은 결론이 무의미하다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_ARM_DOF = 7
CONTROL_HZ = 60.0

_WS_ROOT = Path(__file__).resolve().parent.parent.parent
CALIB_JSON = _WS_ROOT / "hdgp/log/logs/r2s_autotune/results/right_arm_best_calibration.json"

# 관절군 → 팔 관절 인덱스 (r2s 캘리브의 그룹 정의)
ARM_GROUPS = (("arm_proximal", (0, 1, 2)), ("arm_elbow", (3,)), ("arm_wrist", (4, 5, 6)))

# sim 팔 액추에이터 (grasp_{side}_env_cfg.py ImplicitActuatorCfg)
SIM_STIFFNESS = 400.0
SIM_DAMPING = 80.0


def load_arm_pd(path: Path = CALIB_JSON):
    """실측 캘리브 → (kp, kd, Fc, dataset). 파일/키가 없으면 예외."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"팔 캘리브레이션이 없다: {path}\n"
            "  PD 모델은 실측 게인을 요구한다 — 추정값으로 돌리면 실험이 무의미하다."
        )
    data = json.loads(Path(path).read_text())
    groups = data.get("groups", data)
    kp = np.zeros(NUM_ARM_DOF)
    kd = np.zeros(NUM_ARM_DOF)
    fc = np.zeros(NUM_ARM_DOF)
    for name, idxs in ARM_GROUPS:
        g = groups.get(name)
        if g is None:
            raise KeyError(f"{path}: 관절군 {name!r} 없음 (있는 키: {sorted(groups)})")
        for i in idxs:
            kp[i] = float(g["stiffness"])
            kd[i] = float(g["damping"])
            fc[i] = float(g.get("joint_friction", 0.0))
    return kp, kd, fc, data.get("source_dataset", "?")


def second_order_characteristics(kp, kd, inertia):
    """(ω_n [rad/s], ζ). 추종 대역폭 비교의 공통 척도."""
    kp = np.asarray(kp, dtype=np.float64)
    kd = np.asarray(kd, dtype=np.float64)
    I = np.asarray(inertia, dtype=np.float64)
    if np.any(I <= 0):
        raise ValueError(f"관성은 양수여야 한다: {I}")
    wn = np.sqrt(kp / I)
    zeta = kd / (2.0 * np.sqrt(kp * I))
    return wn, zeta


def bandwidth_gap(inertia, kp_real=None, kd_real=None):
    """실기 vs sim 팔의 (ω_n, ζ, 대역폭비) 표를 만든다."""
    if kp_real is None or kd_real is None:
        kp_real, kd_real, _, _ = load_arm_pd()
    wn_r, z_r = second_order_characteristics(kp_real, kd_real, inertia)
    n = len(np.asarray(inertia))
    wn_s, z_s = second_order_characteristics(
        np.full(n, SIM_STIFFNESS), np.full(n, SIM_DAMPING), inertia
    )
    return dict(wn_real=wn_r, zeta_real=z_r, wn_sim=wn_s, zeta_sim=z_s, ratio=wn_s / wn_r)


class MockArm:
    """브리지+컨트롤러+팔의 1차 근사.

    rate : arm ← arm + clip(cmd − arm, ±max_vel/CONTROL_HZ)   (속도제한만)
    pd   : 위 세트포인트를 실측 게인 PD 가 2차로 추종           (지연·오버슛 재현)
    """

    def __init__(self, q0, model: str, max_vel: float, dt: float,
                 kp=None, kd=None, fc=None, inertia=None, substeps: int = 32) -> None:
        if model not in ("rate", "pd"):
            raise ValueError(f"arm model 은 rate|pd — 받은 값 {model!r}")
        if model == "pd" and any(v is None for v in (kp, kd, fc, inertia)):
            raise ValueError("pd 모델은 kp/kd/fc/inertia 가 모두 필요하다")
        if substeps < 1:
            raise ValueError("substeps 는 1 이상")
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.qd = np.zeros_like(self.q)
        self.model = model
        self.max_step = float(max_vel) * float(dt)
        self.dt = float(dt)
        # ★물리 적분은 제어주기보다 잘게 쪼갠다. 60Hz 로 적분하면 저관성 관절
        #   (ω_n ≈ 27 rad/s)이 제대로 움직이지 않아 "추종 실패" 를 모델이 만들어낸다.
        #   실기 펌웨어 MIT 루프도 제어 주기보다 훨씬 빠르게 돈다.
        self.substeps = int(substeps)
        self.sub_dt = self.dt / self.substeps
        self.kp, self.kd, self.fc = kp, kd, fc
        self.I = inertia

    def step(self, cmd) -> None:
        """제어 1 tick 진행. 세트포인트는 브리지 rate-limit 을 거친다."""
        cmd = np.asarray(cmd, dtype=np.float64)
        sp = self.q + np.clip(cmd - self.q, -self.max_step, self.max_step)
        if self.model == "rate":
            self.qd = (sp - self.q) / self.dt
            self.q = sp
            return
        for _ in range(self.substeps):
            self._pd_substep(sp)

    def _pd_substep(self, sp) -> None:
        # ★감쇠는 **암시적**으로 푼다. 명시적으로 적분하면 kd/I·dt > 2 인 저관성 관절
        #   (손목·전완 roll)에서 발산한다 — 실측 게인·관성으로 실제 NaN 이 났다.
        #     qd⁺ = (qd + (kp(sp−q)/I)·dt) / (1 + (kd/I)·dt)   ← 무조건 안정
        dt = self.sub_dt
        accel_spring = self.kp * (sp - self.q) / self.I
        qd_free = (self.qd + accel_spring * dt) / (1.0 + (self.kd / self.I) * dt)
        # 쿨롱 마찰은 운동을 **막을 뿐 역전시키지 못한다**. `-Fc·sign(qd)` 를 그대로 적분하면
        # 속도가 0 을 지날 때 부호가 튀며 에너지를 주입한다 → 속도 변화량을 |qd_free| 로 상한.
        dv_fric = np.minimum(self.fc / self.I * dt, np.abs(qd_free))
        self.qd = qd_free - np.sign(qd_free) * dv_fric
        self.q = self.q + self.qd * dt
