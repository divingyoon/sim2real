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
                 kp=None, kd=None, fc=None, inertia=None, substeps: int = 32,
                 gravity=None) -> None:
        if model not in ("rate", "pd"):
            raise ValueError(f"arm model 은 rate|pd — 받은 값 {model!r}")
        if model == "pd" and any(v is None for v in (kp, kd, fc, inertia)):
            raise ValueError("pd 모델은 kp/kd/fc/inertia 가 모두 필요하다")
        if substeps < 1:
            raise ValueError("substeps 는 1 이상")
        if model == "pd":
            # ★★세트포인트는 실제 위치에서 `max_vel·dt` 만큼만 앞선다. 쿨롱 마찰은
            #   `kp·err > fc` 여야 풀린다. 그러니 선행량이 `fc/kp` 를 못 넘으면 스프링이
            #   마찰을 이길 수 없고 **그 관절은 영원히 정지한다** — cmd 가 아무리 멀어져도
            #   그렇다. 그리고 그 결과는 오류가 아니라 "추종 실패"라는 그럴듯한 숫자로
            #   나온다(08.25 에 200 Hz 리그에서 손목 3관절이 얼어붙은 것을 실기 능력
            #   부족으로 읽을 뻔했다). 세 값의 결합이라 어느 하나만 봐서는 안 보인다.
            #   조건 유도(정지 상태 qd=0 에서 한 substep):
            #       qd_free = (kp·err/I)·dt_s / (1 + (kd/I)·dt_s)
            #       마찰이 제거하는 양 = (fc/I)·dt_s
            #     움직이려면 qd_free > (fc/I)·dt_s
            #       ⟺ err > (fc/kp)·(1 + (kd/I)·dt_s),   dt_s = dt/substeps
            #   ★맨 뒤 괄호가 핵심이다. 순진한 `fc/kp` 만 보면 통과하는데 실제로는
            #     정지하는 구간이 있다(60 Hz·substeps 4: 요구 0.107 vs 선행량 0.033).
            #     암시적 감쇠가 한 substep 의 속도를 그만큼 눌러서 마찰이 이긴다.
            #     그래서 substeps 를 줄이면 조용히 얼어붙는다 — 7개 조건 실측으로 확인.
            leash = float(max_vel) * float(dt)
            sub_dt = float(dt) / int(substeps)
            kp_a = np.asarray(kp, dtype=np.float64)
            kd_a = np.asarray(kd, dtype=np.float64)
            fc_a = np.asarray(fc, dtype=np.float64)
            inertia_a = np.asarray(inertia, dtype=np.float64)
            required = (fc_a / kp_a) * (1.0 + (kd_a / inertia_a) * sub_dt)
            stuck = np.flatnonzero(leash <= required)
            if stuck.size:
                raise ValueError(
                    f"세트포인트 선행량 {leash:.5f} rad 이 마찰을 못 이긴다 — "
                    f"관절 {stuck.tolist()} 은 구조적으로 움직이지 못한다.\n"
                    f"  요구 (fc/kp)·(1+(kd/I)·dt/substeps) = "
                    f"{np.round(required[stuck], 5).tolist()}\n"
                    f"  max_vel 을 올리거나 substeps 를 늘릴 것(dt 를 줄이면 오히려 나빠진다).\n"
                    f"  이 상태로 돌리면 모델이 만들어낸 정지를 실기 추종 실패로 읽게 된다."
                )
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
        #: g(q) → 관절 중력토크(N·m). None 이면 무중력(기존 동작). policy_control 의
        #: fake 플랜트가 robot_control.kinematics 로 넣는다.
        self.gravity = gravity
        #: 마지막 tick 의 모터 토크(스프링+감쇠+τ_ff, 마찰 제외). /joint_states effort 로 나간다.
        self.tau = np.zeros_like(self.q)

    def step(self, cmd, qd_cmd=None, tau_ff=None) -> None:
        """제어 1 tick 진행. 세트포인트는 브리지 rate-limit 을 거친다.

        MIT 3중 지령: τ = kp(sp−q) + kd(q̇*−q̇) + τ_ff − g(q) − Fc·sgn(q̇).
        `qd_cmd`/`tau_ff` 를 생략하면 옛 호출(q̇*=0, τ_ff=0)과 같다. `rate` 모델은 둘을 무시한다.
        """
        cmd = np.asarray(cmd, dtype=np.float64)
        qd_star = self._vector(qd_cmd, "qd_cmd")
        tau_ff_v = self._vector(tau_ff, "tau_ff")
        sp = self.q + np.clip(cmd - self.q, -self.max_step, self.max_step)
        if self.model == "rate":
            self.qd = (sp - self.q) / self.dt
            self.q = sp
            return
        for _ in range(self.substeps):
            self._pd_substep(sp, qd_star, tau_ff_v)

    def _vector(self, value, name: str) -> np.ndarray:
        if value is None:
            return np.zeros_like(self.q)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != self.q.shape:
            raise ValueError(f"{name} 길이 {arr.shape[0]} != 관절 수 {self.q.shape[0]}")
        return arr

    def _pd_substep(self, sp, qd_star, tau_ff) -> None:
        # ★감쇠는 **암시적**으로 푼다. 명시적으로 적분하면 kd/I·dt > 2 인 저관성 관절
        #   (손목·전완 roll)에서 발산한다 — 실측 게인·관성으로 실제 NaN 이 났다.
        #     qd⁺ = (qd + (kp(sp−q) + kd·q̇* + τ_ff − g(q))/I·dt) / (1 + (kd/I)·dt)   ← 무조건 안정
        dt = self.sub_dt
        g = np.zeros_like(self.q) if self.gravity is None else np.asarray(self.gravity(self.q), dtype=np.float64)
        drive = self.kp * (sp - self.q) + self.kd * qd_star + tau_ff - g
        qd_free = (self.qd + drive / self.I * dt) / (1.0 + (self.kd / self.I) * dt)
        # 쿨롱 마찰은 운동을 **막을 뿐 역전시키지 못한다**. `-Fc·sign(qd)` 를 그대로 적분하면
        # 속도가 0 을 지날 때 부호가 튀며 에너지를 주입한다 → 속도 변화량을 |qd_free| 로 상한.
        dv_fric = np.minimum(self.fc / self.I * dt, np.abs(qd_free))
        self.qd = qd_free - np.sign(qd_free) * dv_fric
        self.q = self.q + self.qd * dt
        self.tau = self.kp * (sp - self.q) + self.kd * (qd_star - self.qd) + tau_ff
