#!/usr/bin/env python3
"""좌팔(7 DOF + 2지 그리퍼)이 중력에 대해 무엇을 할 수 있는가 — F1 / F2 / 대역폭.

우팔 + DG-5F 로 낸 §6·§6-1 수치를 좌팔에 그대로 인용하면 안 된다. 끝단이 20관절 손이
아니라 2지 그리퍼라 중력토크가 전혀 다르다. 그래서 같은 계산을 좌팔 체인으로 다시 한다.

    F1  한 액션 스텝의 홀딩 토크 ÷ 중력 토크. 1.0 미만 = 그 관절은 자기 무게를 못 든다.
        스텝 크기는 정책이 실제로 낼 수 있는 palm 지령 변화율 상한(preset)에서 온다 —
        δ 를 크게 잡으면 비도 같이 커지므로 **다른 δ 로 낸 F1 끼리는 비교할 수 없다.**
    F2  홈 자세 정적 평형 kp(q_cmd − q*) = τ_g(q*) 의 q*, 그리고 TCP 가 어디로 가는가.
    대역폭  2차계 f_n·ζ. 홈 자세 대각 근사이므로 **추종 비교용**이지 절대 토크 예측용이 아니다.

값의 출처(전부 자산·설정 — 지어낸 것 없음)
  · 기하·질량 : assets/robot/openarm_tesollo_sensor_rl.urdf (실기 URDF 와 같은 자산)
  · 홈        : hdgp `grasp_left_preset.LEFT_ARM_HOME_JOINT_POS`
  · 펌웨어 게인 : robot_control `openarm_description/config/arm/v10/control_gains.yaml`
  · 식별 게인  : hdgp `r2s_autotune/results/right_arm_best_calibration.json`
    ★**우팔 값이다.** 같은 팔 하드웨어라 참고로 싣지만 좌팔 실측이 아니다. 그 파일의
      `openarm_left_arm` 항목은 400/80 인데 그건 측정치가 아니라 sim 기본값이 남은 것이다.
  · sim 게인   : preset `ARM_IK_STIFFNESS / ARM_IK_DAMPING`

실행: source /opt/ros/humble/setup.bash && . .venv/bin/activate
      python3 scripts/analysis/report_left_arm_gravity.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/rl_ws/sim2real/scripts")
sys.path.insert(0, "/home/user/rl_ws/robot_control/src")
import importlib.util
from robot_control.kinematics import chain_from_urdf
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_inertia import effective_inertia

HDGP = Path.home() / "rl_ws/hdgp"
URDF = HDGP / "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"
spec = importlib.util.spec_from_file_location(
    "_p", HDGP / "source/openarm/openarm/gripper/left/grasp_sensor/grasp_left_preset.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

JOINTS = [f"l_aj_{i}" for i in range(1, 8)]
HOME = np.array([P.LEFT_ARM_HOME_JOINT_POS[j] for j in JOINTS])
TIP = P.GRIPPER_BASE_BODY

# 실측 게인. 우팔 캘리브(r2s)와 펌웨어 control_gains.yaml — 팔은 좌우 같은 하드웨어다.
# control_gains.yaml v10 (하드웨어 플러그인이 CAN MIT 패킷에 싣는 값) — 좌우 동일.
KP_FIRMWARE = np.array([70., 70., 70., 60., 10., 10., 10.])
KD_FIRMWARE = np.array([2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5])
# r2s 식별 게인은 **우팔만** 존재한다(right_arm_best_calibration.json).
# 같은 하드웨어라 참고로 쓰되, 좌팔 실측이 아니라는 것을 표에 적어 둔다.
# 그 파일의 `openarm_left_arm` 항목은 400/80 — 측정치가 아니라 sim 기본값이 그대로 남은 것이다.
KP_IDENT = np.array([67.59, 67.59, 67.59, 66.98, 12.02, 12.02, 12.02])
KD_IDENT = np.array([6.376, 6.376, 6.376, 5.635, 2.154, 2.154, 2.154])
KP_SIM, KD_SIM = P.ARM_IK_STIFFNESS, P.ARM_IK_DAMPING

chain = chain_from_urdf(URDF.read_text(), JOINTS, TIP)
tau_g = np.asarray(chain.gravity_torque(HOME), dtype=float)
inertia = np.asarray(effective_inertia(str(URDF), JOINTS, dict(zip(JOINTS, HOME.tolist()))), dtype=float)

print(f"자산   : {URDF.name}")
print(f"체인   : {JOINTS} -> {TIP}")
print(f"홈     : {np.round(HOME, 4).tolist()}\n")

# --- F1 ------------------------------------------------------------------
# |Δq| = |J† δ| — 한 액션 스텝이 낼 수 있는 관절 변위. δ = palm 지령 변화율 상한.
J = np.asarray(chain.jacobian(HOME), dtype=float)
step_m = float(P.PALM_CMD_RATE_LIMIT)
step_rad = float(P.PALM_ROT_RATE_LIMIT)
twist = np.concatenate([np.full(3, step_m / np.sqrt(3)), np.full(3, step_rad / np.sqrt(3))])
dq = np.asarray(chain.delta_q(HOME, twist, 0.01), dtype=float)

print(f"F1 — 한 스텝({step_m*1000:.0f} mm / {np.degrees(step_rad):.1f}°) 홀딩토크 ÷ 중력토크")
print(f"{'관절':8s} {'τ_g[N·m]':>10s} {'|Δq|[rad]':>10s} {'@펌웨어':>9s} {'@sim400':>9s} {'I[kg·m²]':>10s}")
for i, j in enumerate(JOINTS):
    g = abs(tau_g[i])
    fw = KP_FIRMWARE[i] * abs(dq[i]) / g if g > 1e-9 else float("inf")
    sm = KP_SIM * abs(dq[i]) / g if g > 1e-9 else float("inf")
    flag = "  <-- 1.0 미만" if fw < 1.0 else ""
    print(f"{j:8s} {tau_g[i]:+10.3f} {abs(dq[i]):10.4f} {fw:9.2f} {sm:9.2f} {inertia[i]:10.4f}{flag}")

# --- F2 ------------------------------------------------------------------
# kp(q_cmd − q) = τ_g(q) 를 고정점 반복으로 푼다.
def settle(kp, q_cmd, iters=400):
    q = q_cmd.copy()
    for _ in range(iters):
        q_new = q_cmd - np.asarray(chain.gravity_torque(q), dtype=float) / kp
        if np.max(np.abs(q_new - q)) < 1e-12:
            return q_new
        q = 0.5 * q + 0.5 * q_new
    return q

home_pose = np.asarray(chain.pose(HOME), dtype=float)
print(f"\nF2 — 홈 자세 정적 정착 (명령 홈 TCP base = {np.round(home_pose[:3, 3], 4).tolist()})")
print(f"{'게인':16s} {'Δq 최대[mrad]':>14s} {'최악 관절':>10s} {'TCP 이동[mm]':>13s}")
for label, kp in (("sim 400/80", np.full(7, KP_SIM)),
                  ("펌웨어(좌우 공통)", KP_FIRMWARE),
                  ("r2s 식별(우팔값)", KP_IDENT)):
    q = settle(kp, HOME)
    dq_settle = q - HOME
    worst = int(np.argmax(np.abs(dq_settle)))
    move = np.linalg.norm(np.asarray(chain.pose(q), dtype=float)[:3, 3] - home_pose[:3, 3])
    print(f"{label:16s} {dq_settle[worst]*1000:+14.1f} {JOINTS[worst]:>10s} {move*1000:13.1f}")

# --- 대역폭 -----------------------------------------------------------------
print(f"\n대역폭 — 2차계 f_n[Hz] / ζ  (관성은 홈 자세 대각 근사, 결합·코리올리 무시)")
print(f"{'관절':8s} {'펌웨어 f_n':>11s} {'펌웨어 ζ':>10s} {'식별 f_n':>10s} {'식별 ζ':>9s} {'sim f_n':>9s} {'sim ζ':>8s} {'비(펌)':>8s}")
for i, j in enumerate(JOINTS):
    I = inertia[i]
    fn_f = np.sqrt(KP_FIRMWARE[i] / I) / (2 * np.pi); z_f = KD_FIRMWARE[i] / (2 * np.sqrt(KP_FIRMWARE[i] * I))
    fn_i = np.sqrt(KP_IDENT[i] / I) / (2 * np.pi);    z_i = KD_IDENT[i] / (2 * np.sqrt(KP_IDENT[i] * I))
    fn_s = np.sqrt(KP_SIM / I) / (2 * np.pi);         z_s = KD_SIM / (2 * np.sqrt(KP_SIM * I))
    mark = " *부족감쇠" if z_f < 1.0 else ""
    print(f"{j:8s} {fn_f:11.2f} {z_f:10.2f} {fn_i:10.2f} {z_i:9.2f} {fn_s:9.2f} {z_s:8.2f} {fn_s/fn_f:8.2f}{mark}")
