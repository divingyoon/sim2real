#!/usr/bin/env python3
"""restore_j7.py — 오른팔 j7(손목) 영점만 URDF 홈으로 복원 (Tesollo 구성용).

배경: `set_zero --id 7` 로 j7 공장영점이 덮어써짐(다른 관절은 온전). j7 기계 스톱은
URDF ±90° 대칭(MECH_LIM_V1[J7]=[-90°,+90°])이라 **두 스톱의 중점 = URDF 홈(j7=0)**.
이 스크립트는 j7만 양쪽 스톱까지 구동 → 중점(홈)으로 이동시킨다.

영점 기록은 python API에 per-joint set_zero 가 없어(=set_zero_all뿐, 전관절이라 못 씀)
**스크립트 종료 후 CLI로** 한다:
    openarm-can-cli -i can0 set_zero --id 7

openarm-can-zero-position-calibration 의 bump_to_limit/interpolate/_hit_thresholds/init
로직을 그대로 이식하되, 그리퍼는 건드리지 않고(Tesollo=그리퍼 없음, 원본이 여기서 멈춤),
스톱 미검출 안전상한 + 스팬 검증을 추가.

⚠️ 실제 모터 구동. 반드시:
  - 테솔로 손 제거 상태(j7 가벼움)
  - bringup(ros2_control) 내린 상태 (CAN 충돌 방지)
  - 문제 시 Ctrl-C → 안전 disable
"""

import time

import numpy as np
import openarm_can as oa

CAN = "can0"
J7 = 6                       # arm motor index (r_aj_7), CAN ID 0x07

STEP_DEG = 0.2               # bump 스텝(원본과 동일)
BUMP_KP, BUMP_KD = 45.0, 1.2
# j7 스톱 감지 임계(_hit_thresholds: non-gripper, idx!=0) : |vel|<0.1 & |torque|>2.0
DQ_TH, TAU_TH = 0.1, 2.0
# ★ 최소 이동 가드: 이만큼 움직이기 전엔 스톱 판정 안 함(직전 bump 잔류토크 false-trigger 방지)
MIN_TRAVEL_DEG = 15.0
# bump 사이 후퇴각(스톱에서 물러나 토크 해소)
BACKOFF_DEG = 20.0
# 안전상한: j7 전범위 ~180°. 240° 넘게 스텝해도 스톱 미검출이면 이상 → 중단
MAX_BUMP_STEPS = int(240.0 / STEP_DEG)
SPAN_MIN_DEG, SPAN_MAX_DEG = 150.0, 210.0   # 두 스톱 간격 검증(기대 ~180°)

ARM_MOTOR_TYPES = [oa.MotorType.DM8009, oa.MotorType.DM8009,
                   oa.MotorType.DM4340, oa.MotorType.DM4340,
                   oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310]
ARM_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
ARM_MASTER_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
HOLD_KP = [300, 300, 150, 150, 40, 40, 30]
HOLD_KD = [2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8]


def bump_to_limit(openarm, arm, idx, step_deg):
    """j7 을 step 방향 스톱까지 구동. 스톱 위치(rad) 반환. (원본 bump_to_limit + 최소이동 가드)

    ★ MIN_TRAVEL_DEG 이상 움직이기 전엔 스톱 판정 안 함 — 직전 bump 의 잔류 토크로
      시작하자마자 false-trigger 하는 것을 막는다(span 0° 사고 방지).
    """
    step_rad = np.deg2rad(step_deg)
    q_start = arm.get_motors()[idx].get_position()
    q_target = q_start
    min_travel = np.deg2rad(MIN_TRAVEL_DEG)
    for _ in range(MAX_BUMP_STEPS):
        q_target += step_rad
        arm.mit_control_one(idx, oa.MITParam(BUMP_KP, BUMP_KD, q_target, 0.0, 0.0))
        openarm.recv_all()
        time.sleep(0.005)
        m = arm.get_motors()[idx]
        moved = abs(m.get_position() - q_start)
        if moved > min_travel and abs(m.get_velocity()) < DQ_TH and abs(m.get_torque()) > TAU_TH:
            return float(m.get_position())
    raise RuntimeError(
        f"j7 스톱 미검출({MAX_BUMP_STEPS} 스텝, {step_deg:+.1f}°/step) — 방향/임계 확인 필요, 중단")


def interpolate(openarm, arm, idx, target_abs, interp_time=3.0, kp=52.0, kd=1.5):
    """j7 을 현재→target_abs 로 선형 이동. (원본 interpolate 이식)"""
    q0 = arm.get_motors()[idx].get_position()
    n = 500
    dt = interp_time / n
    for i in range(n + 1):
        q = q0 + (target_abs - q0) * (i / n)
        arm.mit_control_one(idx, oa.MITParam(kp, kd, q, 0.0, 0.0))
        openarm.recv_all()
        time.sleep(dt)


def main():
    openarm = oa.OpenArm(CAN, True)
    openarm.init_arm_motors(ARM_MOTOR_TYPES, ARM_IDS, ARM_MASTER_IDS)
    # ★ 그리퍼 init 안 함 (Tesollo 구성 → ID 0x08 없음 → 원본은 여기서 hang)
    openarm.set_callback_mode_all(oa.CallbackMode.STATE)

    print("Enabling arm...")
    openarm.enable_all()
    time.sleep(0.1)
    openarm.recv_all()

    arm = openarm.get_arm()
    am = arm.get_motors()
    initial_q = [m.get_position() for m in am]
    print("initial q:", [f"{q:+.3f}" for q in initial_q])

    # 전 관절 light PD 홀딩(j7 구동 중 다른 관절 유지)
    arm.mit_control_all([oa.MITParam(HOLD_KP[i], HOLD_KD[i], initial_q[i], 0.0, 0.0)
                         for i in range(7)])
    openarm.recv_all()

    try:
        print("\n[1/3] j7 → +stop bump ...")
        q_plus = bump_to_limit(openarm, arm, J7, +STEP_DEG)
        print(f"    +stop = {q_plus:+.4f} rad ({np.rad2deg(q_plus):+.2f}°)")

        # ★ +스톱에서 물러나 잔류 토크 해소 후 -bump (false-trigger 방지)
        print(f"    ...{BACKOFF_DEG:.0f}° 후퇴 + 정착")
        interpolate(openarm, arm, J7, q_plus - np.deg2rad(BACKOFF_DEG), interp_time=1.5)
        time.sleep(0.7)

        print("[2/3] j7 → -stop bump ...")
        q_minus = bump_to_limit(openarm, arm, J7, -STEP_DEG)
        print(f"    -stop = {q_minus:+.4f} rad ({np.rad2deg(q_minus):+.2f}°)")

        span_deg = abs(np.rad2deg(q_plus - q_minus))
        home = 0.5 * (q_plus + q_minus)
        print(f"    stop span = {span_deg:.2f}° (기대 ~180°), home(midpoint) = {home:+.4f} rad")
        if not (SPAN_MIN_DEG < span_deg < SPAN_MAX_DEG):
            raise RuntimeError(
                f"스톱 간격 {span_deg:.1f}° 가 ~180°(±30) 밖 — 잘못된 스톱/false trigger 의심, 중단")

        print("[3/3] j7 → home(midpoint) 이동 ...")
        interpolate(openarm, arm, J7, home, interp_time=3.0)
        # home 잠시 홀딩(안정화)
        for _ in range(200):
            arm.mit_control_one(J7, oa.MITParam(52.0, 1.5, home, 0.0, 0.0))
            openarm.recv_all()
            time.sleep(0.005)

        now = arm.get_motors()[J7].get_position()
        print(f"\n✅ j7 이 URDF 홈(중점)에 위치. 현재 pos = {now:+.4f} rad ({np.rad2deg(now):+.2f}°)")
        print("=" * 64)
        print(">>> 이 스크립트 종료 후(모터 disable됨) 즉시 CLI로 영점 기록:")
        print(">>>     openarm-can-cli -i can0 set_zero --id 7")
        print(">>> 그다음 검증:  openarm-can-cli -i can0 monitor --id 7")
        print(">>>   (홈에서 pos≈0, 양쪽 스톱에서 ±1.5708 이면 복원 성공)")
        print("=" * 64)

    except KeyboardInterrupt:
        print("\n[중단] Ctrl-C — 안전 disable")
    finally:
        openarm.disable_all()
        print("[모터 disable, 종료]")


if __name__ == "__main__":
    main()
