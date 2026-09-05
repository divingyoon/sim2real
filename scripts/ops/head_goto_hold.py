#!/usr/bin/env python3
"""head(목) 2-DOF Dynamixel(XC330 x2)을 특정 pan/tilt 로 이동 후 홀드.

캘리브용: pan=0, tilt≈30° 로 보내고 고정한 뒤 ChArUco 캘리브.
각도 기본 단위 = **도(deg)**. 라디안으로 줄 땐 --rad.
DynamixelSDK 독립 스크립트(ros2_control 불필요). SDK/하드웨어 없이도 import 가능
하도록 dynamixel_sdk 는 지연 import(순수 변환·검증은 유닛테스트 가능).

XC330 컨트롤 테이블 근거: repo/dynamixel_hardware_interface/param/dxl_model/xc330_m288.model
  Goal/Present Position: 0.0015339807878856412 rad/tick, offset -3.14159265359
  Operating Mode@11 · Torque Enable@64 · Goal Position@116(4B) · Present Position@132(4B)
관절 한계(±90°)는 URDF head_j_pan/head_j_tilt(±1.5708 rad) 근거 → 범위 밖이면 거부.

실기 정보(미확정): 포트 /dev/ttyACM0 추정, ID/baud 미상 → --scan 으로 발견.
"""
import argparse
import math
import sys
import time

# --- XC330 상수 ---
RES_RAD_PER_TICK = 0.0015339807878856412
OFFSET_RAD = -3.14159265359
TICK_MAX = 4095
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
OP_MODE_POSITION = 3
TORQUE_ON, TORQUE_OFF = 1, 0
PROTOCOL = 2.0

# 관절 한계 (URDF head_j_pan/head_j_tilt lower/upper) — 범위 밖 명령 거부
PAN_LIMIT_RAD = 1.5708
TILT_LIMIT_RAD = 1.5708

SCAN_BAUDS = [57600, 1000000, 2000000, 115200, 3000000]


# ---------- 순수 변환·검증 (하드웨어 무관, 테스트 대상) ----------
def clamp_tick(tick: int) -> int:
    return max(0, min(TICK_MAX, int(tick)))


def rad_to_tick(rad: float) -> int:
    return clamp_tick(round((rad - OFFSET_RAD) / RES_RAD_PER_TICK))


def tick_to_rad(tick: int) -> float:
    return tick * RES_RAD_PER_TICK + OFFSET_RAD


def to_rad(value: float, use_deg: bool) -> float:
    return math.radians(value) if use_deg else value


def validate_within_limits(pan_rad: float, tilt_rad: float) -> None:
    """URDF 관절 한계(±90°) 밖이면 ValueError. 조용한 clamp 대신 명시적 거부."""
    if abs(pan_rad) > PAN_LIMIT_RAD:
        raise ValueError(f"pan {math.degrees(pan_rad):.1f}° 가 한계 ±90° 밖 "
                         f"(단위 확인: 도는 기본, 라디안은 --rad)")
    if abs(tilt_rad) > TILT_LIMIT_RAD:
        raise ValueError(f"tilt {math.degrees(tilt_rad):.1f}° 가 한계 ±90° 밖 "
                         f"(단위 확인: 도는 기본, 라디안은 --rad)")


def parse_ids(text: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"--ids 는 'pan,tilt' 두 개 필요: {text!r}")
    pan_id, tilt_id = (int(p) for p in parts)
    if pan_id == tilt_id:
        raise ValueError(f"pan/tilt ID 가 같음: {text!r}")
    return pan_id, tilt_id


# ---------- 하드웨어 (지연 import) ----------
class HeadBus:
    def __init__(self, port: str, baud: int):
        from dynamixel_sdk import PortHandler, PacketHandler
        self.port = PortHandler(port)
        self.packet = PacketHandler(PROTOCOL)
        if not self.port.openPort():
            raise RuntimeError(f"포트 열기 실패: {port}")
        try:
            if not self.port.setBaudRate(baud):
                raise RuntimeError(f"baud 설정 실패: {baud}")
        except Exception:
            self.port.closePort()   # 열린 핸들 누수 방지
            raise

    def _check(self, comm, err, ctx):
        from dynamixel_sdk import COMM_SUCCESS
        if comm != COMM_SUCCESS:
            raise RuntimeError(f"{ctx}: {self.packet.getTxRxResult(comm)}")
        if err:
            raise RuntimeError(f"{ctx}: {self.packet.getRxPacketError(err)}")

    def ping(self, dxl_id: int):
        from dynamixel_sdk import COMM_SUCCESS
        model, comm, err = self.packet.ping(self.port, dxl_id)
        return model if (comm == COMM_SUCCESS and not err) else None

    def torque(self, dxl_id: int, on: bool):
        comm, err = self.packet.write1ByteTxRx(
            self.port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON if on else TORQUE_OFF)
        self._check(comm, err, f"torque({dxl_id})")

    def set_position_mode(self, dxl_id: int):
        self.torque(dxl_id, False)                       # 모드 변경은 토크 off 필요
        comm, err = self.packet.write1ByteTxRx(
            self.port, dxl_id, ADDR_OPERATING_MODE, OP_MODE_POSITION)
        self._check(comm, err, f"op_mode({dxl_id})")

    def write_goal(self, dxl_id: int, tick: int):
        comm, err = self.packet.write4ByteTxRx(
            self.port, dxl_id, ADDR_GOAL_POSITION, clamp_tick(tick))
        self._check(comm, err, f"goal({dxl_id})")

    def read_present(self, dxl_id: int) -> int:
        val, comm, err = self.packet.read4ByteTxRx(
            self.port, dxl_id, ADDR_PRESENT_POSITION)
        self._check(comm, err, f"present({dxl_id})")
        return val

    def close(self):
        self.port.closePort()


def scan(port: str) -> None:
    """baud/ID 스윕(핑만, 모터 안 움직임)."""
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
    ph, pk = PortHandler(port), PacketHandler(PROTOCOL)
    if not ph.openPort():
        sys.exit(f"포트 열기 실패: {port}")
    found = []
    try:
        for baud in SCAN_BAUDS:
            ph.setBaudRate(baud)
            for dxl_id in range(0, 21):
                model, comm, err = pk.ping(ph, dxl_id)
                if comm == COMM_SUCCESS and not err:
                    found.append((baud, dxl_id, model))
                    print(f"  발견: baud={baud} id={dxl_id} model={model}")
    finally:
        ph.closePort()
    if not found:
        print("모터 없음 — 포트/전원/케이블 확인")
    else:
        print(f"총 {len(found)}개. --baud/--ids 로 지정해 실행하세요.")


def goto_hold(port, baud, pan_id, tilt_id, pan_rad, tilt_rad, hold,
              tolerance_tick, timeout_s) -> bool:
    """이동+홀드. 도달 여부(bool) 반환. 범위 검증은 호출 전에 끝났다고 가정."""
    targets = {pan_id: rad_to_tick(pan_rad), tilt_id: rad_to_tick(tilt_rad)}
    bus = HeadBus(port, baud)
    try:
        # 1) 둘 다 핑 성공 확인 후에만 토크 건드림(부분실패 방지)
        for dxl_id in (pan_id, tilt_id):
            if bus.ping(dxl_id) is None:
                raise RuntimeError(f"id={dxl_id} 핑 실패 → --scan 으로 ID/baud 확인")
        # 2) position mode(토크 off 상태) → 현재값을 goal 로 seed(토크 on 시 점프 방지) → 토크 on
        for dxl_id in (pan_id, tilt_id):
            bus.set_position_mode(dxl_id)
            bus.write_goal(dxl_id, bus.read_present(dxl_id))
            bus.torque(dxl_id, True)
        # 3) 목표 write
        for dxl_id, tick in targets.items():
            bus.write_goal(dxl_id, tick)
            print(f"  id={dxl_id} → tick={tick} ({math.degrees(tick_to_rad(tick)):.1f}°)")

        t0, reached = time.time(), False
        while time.time() - t0 < timeout_s:
            if all(abs(bus.read_present(i) - t) <= tolerance_tick for i, t in targets.items()):
                reached = True
                break
            time.sleep(0.05)
        for dxl_id in (pan_id, tilt_id):
            print(f"  present id={dxl_id}: {math.degrees(tick_to_rad(bus.read_present(dxl_id))):.1f}°")
        print("도달·홀드 완료." if reached else "⚠ 타임아웃(미도달) — 토크/부하/범위 확인")

        if not hold:
            for dxl_id in (pan_id, tilt_id):
                bus.torque(dxl_id, False)
            print("토크 해제(홀드 안 함).")
        return reached
    finally:
        bus.close()


def main():
    ap = argparse.ArgumentParser(description="head 2-DOF Dynamixel 이동+홀드 (캘리브용). 각도 기본=도")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--ids", default="1,2", help="pan_id,tilt_id (서로 달라야 함)")
    ap.add_argument("--pan", type=float, default=0.0, help="기본 도(deg)")
    ap.add_argument("--tilt", type=float, default=30.0, help="기본 도(deg), 아래로 +")
    ap.add_argument("--rad", action="store_true", help="pan/tilt 를 라디안으로 해석")
    ap.add_argument("--scan", action="store_true", help="baud/ID 스윕만(모터 안 움직임)")
    ap.add_argument("--no-hold", action="store_true", help="도달 후 토크 해제")
    ap.add_argument("--tolerance-deg", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args()

    if a.scan:
        scan(a.port)
        return
    try:
        pan_id, tilt_id = parse_ids(a.ids)
        pan = to_rad(a.pan, not a.rad)
        tilt = to_rad(a.tilt, not a.rad)
        validate_within_limits(pan, tilt)     # 범위 밖이면 하드웨어 접촉 전에 거부
    except ValueError as e:
        sys.exit(f"입력 오류: {e}")

    tol = max(1, rad_to_tick(math.radians(a.tolerance_deg)) - rad_to_tick(0.0))
    reached = goto_hold(a.port, a.baud, pan_id, tilt_id, pan, tilt,
                        hold=not a.no_hold, tolerance_tick=tol, timeout_s=a.timeout)
    sys.exit(0 if reached else 1)             # 미도달을 실패로 전파(파이프라인 안전)


if __name__ == "__main__":
    main()
