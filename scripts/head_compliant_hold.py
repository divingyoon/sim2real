#!/usr/bin/env python3
"""head(목) 다이나믹셀을 **손으로 돌려 맞출 수 있는** 순응 홀드로 둔다.

RealSense 화면을 보면서 카메라 자세를 손으로 맞추기 위한 도구다. 모터는

  · 스스로 움직이지 않고,
  · 손으로 밀면 밀리고,
  · 손을 떼면 **그 자리에 머문다**.

**Current-based Position Control Mode(모드 5)** 를 쓴다. `Goal Current` 가
하드웨어 레벨 토크 상한이라 소프트웨어가 죽어도 과토크가 물리적으로 불가능하다.
모드 5 는 Extended Position 과 같은 멀티턴 좌표계를 쓰므로 `head_position_hold_node`
의 deg↔tick 언랩 수학을 그대로 재사용한다.

동작은 세 가지 뿐이다:

  HOLD    손이 없다 — 목표를 고정한다
  FOLLOW  손이 밀고 있다 — 목표가 현재 위치를 따라간다
  LATCH   방금 멈췄다 — 목표를 현재 위치로 확정하고 HOLD 로 돌아간다

★중력 처짐 대책은 추종 로직이 아니라 **시작 전 측정**이다. Goal Current 가
중력토크보다 작으면 어떤 추종 로직도 흘러내림을 막지 못한다. 그래서 시작 시
목표를 고정한 채 처짐을 재고, 한계를 넘으면 **추종에 들어가지 않고 종료**한다.

제어 테이블 근거: repo/dynamixel_hardware_interface/param/dxl_model/xc330_m288.model

    # 하드웨어를 열지 않고 쓸 값만 출력
    python head_compliant_hold.py --dry-run
    # 실제 순응 홀드 (Ctrl-C 로 종료하면 토크를 반드시 끈다)
    python head_compliant_hold.py --goal-current 20
    # 맞춘 각도를 저장
    python head_compliant_hold.py --goal-current 20 --save ../config/head_hand_set.yaml
"""

from __future__ import annotations

import argparse
import glob
import math
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from head_position_hold_node import (
    ADDR_GOAL_POSITION,
    ADDR_OPERATING_MODE,
    ADDR_TORQUE_ENABLE,
    TICK_MAX,
    TORQUE_OFF,
    TORQUE_ON,
    DynamixelPositionController,
    tick_to_deg,
)

# ---------- 제어 테이블 (XC330-M288) ----------
ADDR_CURRENT_LIMIT = 38          # 2B · EEPROM
ADDR_POSITION_D_GAIN = 80        # 2B · RAM
ADDR_POSITION_I_GAIN = 82        # 2B · RAM · 중력 정상상태 오차를 지운다
ADDR_POSITION_P_GAIN = 84        # 2B · RAM
ADDR_GOAL_CURRENT = 102          # 2B · RAM
ADDR_PRESENT_CURRENT = 126       # 2B · signed
ADDR_PRESENT_VELOCITY = 128      # 4B · signed
ADDR_PRESENT_TEMPERATURE = 146   # 1B
ADDR_HARDWARE_ERROR_STATUS = 70  # 1B · Overload 등이 래치된다

OP_MODE_CURRENT_POSITION = 5     # Current-based Position Control

# ---------- 단위 (model 파일의 [unit info]) ----------
CURRENT_UNIT_MA = 1.0                       # XC330 은 1 raw = 1 mA
VELOCITY_UNIT_RAD_S = 0.0239691227          # 1 raw = 0.02397 rad/s
VELOCITY_UNIT_DEG_S = math.degrees(VELOCITY_UNIT_RAD_S)
TICK_PER_DEG = TICK_MAX / 360.0

# ---------- 정책 상한 ----------
#: XC330-M288-T 정격 stall 0.93 N·m @ 1.8 A → 1 mA 당 0.000517 N·m.
#: 상한 400 mA = 0.207 N·m = 5 cm 지렛대에서 4.1 N(≈420 gf) — 손으로 충분히 이긴다.
#: 하드웨어 Current Limit(2352 raw)의 1/6 이다. 이 값을 더 올리면 "손으로 조정 가능"
#: 이라는 안전 성질 자체가 사라진다.
STALL_TORQUE_NM = 0.93
STALL_CURRENT_MA = 1800.0
GOAL_CURRENT_HARD_CAP_MA = 400.0
HAND_LEVER_M = 0.05              # 카메라 마운트 어림 지렛대. 표시용일 뿐 제어에 안 쓴다
MAX_TEMPERATURE_C = 55

# ---------- 기본값 ----------
DEFAULT_GOAL_CURRENT_MA = 20.0
DEFAULT_RATE_HZ = 50.0
DEFAULT_DEADBAND_DEG = 1.0
DEFAULT_VEL_THRESHOLD_DEG_S = 3.0
DEFAULT_STILL_SECONDS = 0.5
DEFAULT_MAX_FOLLOW_DEG_S = 180.0
DEFAULT_MAX_TRAVEL_DEG = 120.0
DEFAULT_SAG_LIMIT_DEG = 3.0
DEFAULT_SAG_SECONDS = 2.0
DEFAULT_BAUD = 1_000_000         # 2026-09-01 실측. 버스가 바뀌면 ./dxl.sh scan 으로 먼저 확인할 것
PORT_GLOBS = ("/dev/ttyUSB*", "/dev/ttyACM*")

ACTION_HOLD = "hold"
ACTION_FOLLOW = "follow"
ACTION_LATCH = "latch"


# ==================== 순수 변환·검증 ====================

def to_signed(value: int, bits: int) -> int:
    """다이나믹셀이 돌려주는 무부호 정수를 부호 있는 값으로 읽는다."""
    half = 1 << (bits - 1)
    return int(value) - (1 << bits) if int(value) >= half else int(value)


def ma_to_raw(ma: float) -> int:
    return round(float(ma) / CURRENT_UNIT_MA)


def raw_to_ma(raw: int) -> float:
    return int(raw) * CURRENT_UNIT_MA


def ma_to_torque_nm(ma: float) -> float:
    """전류를 토크로. mA 는 사람이 세기를 가늠할 수 없는 숫자다."""
    return float(ma) * STALL_TORQUE_NM / STALL_CURRENT_MA


def torque_to_force_n(torque_nm: float, lever_m: float = HAND_LEVER_M) -> float:
    """지렛대 끝에서 손이 느끼는 힘. 이길 수 있는지는 이 값으로 판단한다."""
    return float(torque_nm) / float(lever_m)


def deg_per_s_to_velocity_raw(deg_per_s: float) -> float:
    return float(deg_per_s) / VELOCITY_UNIT_DEG_S


def deg_to_tick_span(deg: float) -> int:
    """각도 **차이**를 틱 차이로. 절대 위치 변환(deg_to_tick)과 다르다."""
    return round(float(deg) * TICK_PER_DEG)


def validate_goal_current(ma: float) -> float:
    """정책 상한을 넘으면 clamp 하지 않고 거부한다 — 조용히 낮추면 사용자가 모른다."""
    value = float(ma)
    if value <= 0.0:
        raise ValueError(f"goal current 는 0 보다 커야 한다: {value}")
    if value > GOAL_CURRENT_HARD_CAP_MA:
        raise ValueError(
            f"goal current {value} mA 가 정책 상한 {GOAL_CURRENT_HARD_CAP_MA} mA 를 넘는다. "
            "이 상한은 '손으로 이길 수 있음'을 보장한다"
        )
    return value


def parse_motor_ids(text: str) -> tuple[int, ...]:
    """쉼표 구분 id 목록.

    `head_position_hold_node.parse_ids` 는 정확히 2개를 강제한다. 실기 head 버스에
    모터가 하나만 잡히는 경우가 있어 여기서는 1개 이상을 허용한다.
    """
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("id 를 하나 이상 지정할 것")

    ids: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise ValueError(f"id 가 정수가 아니다: {part!r}") from None
        if not 0 <= value <= 252:
            raise ValueError(f"id 는 0~252 여야 한다: {value}")
        ids.append(value)

    if len(set(ids)) != len(ids):
        raise ValueError(f"id 가 중복이다: {text!r}")
    return tuple(ids)


def autodetect_port(candidates: list[str]) -> str:
    """직렬 포트가 정확히 하나일 때만 자동으로 고른다."""
    if not candidates:
        raise RuntimeError(f"직렬 포트가 없다 — {' '.join(PORT_GLOBS)} 확인")
    if len(candidates) > 1:
        raise RuntimeError(f"포트가 여러 개다 ({', '.join(candidates)}) — --port 로 지정")
    return candidates[0]


def discover_ports() -> list[str]:
    found: list[str] = []
    for pattern in PORT_GLOBS:
        found.extend(sorted(glob.glob(pattern)))
    return found


# ==================== 추종 상태기계 ====================

@dataclass(frozen=True)
class FollowConfig:
    deadband_tick: int
    vel_threshold_raw: float
    still_cycles_needed: int
    max_step_tick: int


@dataclass(frozen=True)
class FollowState:
    goal_tick: int
    still_cycles: int
    action: str


def clamp_follow_step(goal_tick: int, present_tick: int, max_step_tick: int) -> int:
    """목표가 한 주기에 움직일 수 있는 양을 제한한다.

    통신 글리치나 급격한 당김이 목표를 순간이동시키지 못하게 한다.
    """
    delta = int(present_tick) - int(goal_tick)
    if abs(delta) <= max_step_tick:
        return int(present_tick)
    return int(goal_tick) + (max_step_tick if delta > 0 else -max_step_tick)


def follow_decision(
    state: FollowState, present_tick: int, velocity_raw: float, config: FollowConfig
) -> FollowState:
    """새 상태를 **반환한다** — 입력 상태는 바꾸지 않는다."""
    displaced = abs(int(present_tick) - state.goal_tick) > config.deadband_tick
    moving = abs(float(velocity_raw)) > config.vel_threshold_raw

    if not displaced:
        return replace(state, still_cycles=0, action=ACTION_HOLD)

    if moving:
        stepped = clamp_follow_step(state.goal_tick, present_tick, config.max_step_tick)
        return FollowState(goal_tick=stepped, still_cycles=0, action=ACTION_FOLLOW)

    settled = state.still_cycles + 1
    if settled < config.still_cycles_needed:
        return replace(state, still_cycles=settled, action=ACTION_HOLD)

    stepped = clamp_follow_step(state.goal_tick, present_tick, config.max_step_tick)
    return FollowState(goal_tick=stepped, still_cycles=0, action=ACTION_LATCH)


def remaining_sleep(period: float, elapsed: float) -> float:
    """주기에서 실제 작업 시간을 뺀다. 안 빼면 설정 Hz 보다 느려져 정지 판정이 늦는다."""
    return max(0.0, float(period) - float(elapsed))


def travel_exceeded(present_tick: int, origin_tick: int, max_travel_tick: int) -> bool:
    return abs(int(present_tick) - int(origin_tick)) > max_travel_tick


def select_min_holding(
    results: list[tuple[float, int]], limit_tick: int
) -> float | None:
    """처짐이 한계 안으로 들어오는 **가장 낮은** 전류.

    ★단조성을 요구한다 — 낮은 값이 우연히 통과해도 그 위에 실패가 있으면 버린다.
    기구 마찰이 특정 자세에서만 붙잡아 주는 경우가 있어(09.01 tilt 10 mA 가 그랬다)
    한 번의 통과를 근거로 삼으면 안 된다.
    """
    ordered = sorted(results)
    answer: float | None = None
    for ma, sag in reversed(ordered):                 # 높은 값부터 내려온다
        if abs(sag) > limit_tick:
            break                                     # 여기서 실패 — 아래는 볼 것 없다
        answer = ma
    return answer


def sags_under_gravity(displacement_tick: int, limit_tick: int) -> bool:
    """목표를 고정했는데도 이만큼 밀렸다면 전류 상한이 중력토크보다 작다."""
    return abs(int(displacement_tick)) > limit_tick


# ==================== SDK 어댑터 ====================

class CompliantController(DynamixelPositionController):
    """위치 홀드 어댑터에 전류 제어에 필요한 접근만 더한다."""

    def write2(self, dxl_id: int, address: int, value: int, label: str) -> None:
        comm, err = self.packet.write2ByteTxRx(self.port, dxl_id, address, int(value))
        self._check(comm, err, f"{label} id={dxl_id}")

    def read2_signed(self, dxl_id: int, address: int, label: str) -> int:
        value, comm, err = self.packet.read2ByteTxRx(self.port, dxl_id, address)
        self._check(comm, err, f"{label} id={dxl_id}")
        return to_signed(value, 16)

    def read4_signed(self, dxl_id: int, address: int, label: str) -> int:
        value, comm, err = self.packet.read4ByteTxRx(self.port, dxl_id, address)
        self._check(comm, err, f"{label} id={dxl_id}")
        return to_signed(value, 32)

    def read1(self, dxl_id: int, address: int, label: str) -> int:
        value, comm, err = self.packet.read1ByteTxRx(self.port, dxl_id, address)
        self._check(comm, err, f"{label} id={dxl_id}")
        return int(value)

    def read_present_current_ma(self, dxl_id: int) -> float:
        return raw_to_ma(self.read2_signed(dxl_id, ADDR_PRESENT_CURRENT, "present current"))

    def read_present_velocity_raw(self, dxl_id: int) -> int:
        return self.read4_signed(dxl_id, ADDR_PRESENT_VELOCITY, "present velocity")

    def read_temperature_c(self, dxl_id: int) -> int:
        return self.read1(dxl_id, ADDR_PRESENT_TEMPERATURE, "present temperature")

    def configure_compliant_mode(self, dxl_id: int, goal_current_ma: float) -> int:
        """모드 5 로 바꾸고 현재 위치를 목표로 심은 뒤 토크를 켠다.

        Operating Mode 는 EEPROM 이라 토크가 켜져 있으면 쓰기가 거부된다.
        """
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
        self.write1(dxl_id, ADDR_OPERATING_MODE, OP_MODE_CURRENT_POSITION, "operating mode 5")
        limit_raw = self.read2_signed(dxl_id, ADDR_CURRENT_LIMIT, "current limit")
        goal_raw = ma_to_raw(goal_current_ma)
        if goal_raw > limit_raw:
            raise ValueError(
                f"id={dxl_id}: goal current {goal_raw} raw 가 EEPROM Current Limit "
                f"{limit_raw} raw 를 넘는다 — 모터가 조용히 잘라 먹는다"
            )
        self.write2(dxl_id, ADDR_GOAL_CURRENT, goal_raw, "goal current")
        seed = self.read_present_tick(dxl_id)
        self.write4(dxl_id, ADDR_GOAL_POSITION, seed, "seed goal position")
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")
        return seed


# ==================== 세션 ====================

@dataclass(frozen=True)
class SessionConfig:
    ids: tuple[int, ...]
    names: tuple[str, ...]
    goal_current_ma: float
    rate_hz: float
    follow: FollowConfig
    max_travel_tick: int
    sag_limit_tick: int
    sag_seconds: float


def build_follow_config(
    deadband_deg: float, vel_threshold_deg_s: float, still_seconds: float,
    max_follow_deg_s: float, rate_hz: float,
) -> FollowConfig:
    """사람이 읽는 물리 단위를 모터 단위로 옮긴다."""
    period = 1.0 / float(rate_hz)
    return FollowConfig(
        deadband_tick=max(1, deg_to_tick_span(deadband_deg)),
        vel_threshold_raw=deg_per_s_to_velocity_raw(vel_threshold_deg_s),
        still_cycles_needed=max(1, round(float(still_seconds) * float(rate_hz))),
        max_step_tick=max(1, deg_to_tick_span(float(max_follow_deg_s) * period)),
    )


def check_gravity_sag(
    controller: CompliantController, config: SessionConfig, seeds: dict[int, int]
) -> dict[int, int]:
    """목표를 고정한 채 처짐을 잰다. 반환값은 관절별 변위(틱)."""
    period = 1.0 / config.rate_hz
    deadline = time.monotonic() + config.sag_seconds
    while time.monotonic() < deadline:
        time.sleep(min(period, deadline - time.monotonic()))
    return {
        dxl_id: controller.read_present_tick(dxl_id) - seeds[dxl_id]
        for dxl_id in config.ids
    }


def _format_row(name: str, tick: int, current_ma: float, action: str) -> str:
    return f"{name} {tick_to_deg(tick):+8.2f}°(cur {current_ma:+6.1f}mA) [{action.upper()}]"


def run_session(
    controller: CompliantController, config: SessionConfig, seeds: dict[int, int],
    should_stop,
) -> dict[int, int]:
    """순응 홀드 루프. 반환값은 관절별 최종 틱."""
    period = 1.0 / config.rate_hz
    states = {i: FollowState(seeds[i], 0, ACTION_HOLD) for i in config.ids}
    frozen: set[int] = set()
    cycle = 0

    while not should_stop():
        started = time.monotonic()
        rows = []
        for dxl_id, name in zip(config.ids, config.names):
            present = controller.read_present_tick(dxl_id)
            velocity = controller.read_present_velocity_raw(dxl_id)
            state = states[dxl_id]

            if dxl_id not in frozen and travel_exceeded(
                present, seeds[dxl_id], config.max_travel_tick
            ):
                frozen.add(dxl_id)
                print(
                    f"\n⚠ {name}: 시작 위치에서 "
                    f"{abs(tick_to_deg(present) - tick_to_deg(seeds[dxl_id])):.1f}° 벗어났다 — "
                    "이 관절의 추종을 멈춘다(--max-travel-deg)",
                    flush=True,
                )

            if dxl_id not in frozen:
                state = follow_decision(state, present, velocity, config.follow)
                if state.action in (ACTION_FOLLOW, ACTION_LATCH):
                    controller.write4(dxl_id, ADDR_GOAL_POSITION, state.goal_tick, "goal")
                states[dxl_id] = state

            action = "frozen" if dxl_id in frozen else state.action
            rows.append(_format_row(name, present, controller.read_present_current_ma(dxl_id), action))

        cycle += 1
        if cycle % round(config.rate_hz) == 0:
            _warn_if_hot(controller, config)
        print("\r" + " · ".join(rows) + "   ", end="", flush=True)
        time.sleep(remaining_sleep(period, time.monotonic() - started))

    print()
    return {i: controller.read_present_tick(i) for i in config.ids}


def passes_sag_gate(
    controller: CompliantController, config: SessionConfig, seeds: dict[int, int]
) -> bool:
    """중력 처짐을 재고 결과를 보고한다. 통과하지 못하면 추종에 들어가면 안 된다."""
    print(f"\n처짐 확인 {config.sag_seconds}s — 손대지 마세요")
    print("  ★이 검사는 **지금 자세 한 곳**만 본다. 중력 토크는 자세에 따라 변하므로,"
          "\n   중력이 가장 크게 걸리는 자세(팔이 수평)에서 시작해야 의미가 있다.")
    sag = check_gravity_sag(controller, config, seeds)
    for name, dxl_id in zip(config.names, config.ids):
        print(f"  {name}: {sag[dxl_id] / TICK_PER_DEG:+.2f}°")

    sagging = [n for n, i in zip(config.names, config.ids)
               if sags_under_gravity(sag[i], config.sag_limit_tick)]
    if not sagging:
        return True
    print(f"\n❌ {', '.join(sagging)} 가 중력을 못 버틴다. --goal-current 를 "
          f"{config.goal_current_ma} 보다 **조금씩** 올려 다시 시도할 것", file=sys.stderr)
    return False


def _warn_if_hot(controller: CompliantController, config: SessionConfig) -> None:
    for dxl_id, name in zip(config.ids, config.names):
        temp = controller.read_temperature_c(dxl_id)
        if temp >= MAX_TEMPERATURE_C:
            print(f"\n🔥 {name}: {temp}°C — 상한 {MAX_TEMPERATURE_C}°C. 중단하고 식힐 것", flush=True)


def render_angles_yaml(names: tuple[str, ...], ids: tuple[int, ...], ticks: dict[int, int]) -> str:
    lines = ["# head_compliant_hold.py 로 손으로 맞춘 자세", "motors:"]
    for name, dxl_id in zip(names, ids):
        lines.append(f"  {name}:")
        lines.append(f"    id: {dxl_id}")
        lines.append(f"    tick: {ticks[dxl_id]}")
        lines.append(f"    deg: {tick_to_deg(ticks[dxl_id]):.3f}")
    return "\n".join(lines) + "\n"


# ==================== CLI ====================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", default=None, help="기본: 직렬 포트가 하나면 자동 탐지")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ids", default="1,2")
    p.add_argument("--names", default=None, help="쉼표 구분. 기본 id<n>")
    p.add_argument("--goal-current", type=float, default=DEFAULT_GOAL_CURRENT_MA,
                   help=f"mA. 정책 상한 {GOAL_CURRENT_HARD_CAP_MA}")
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    p.add_argument("--deadband-deg", type=float, default=DEFAULT_DEADBAND_DEG)
    p.add_argument("--vel-threshold-deg-s", type=float, default=DEFAULT_VEL_THRESHOLD_DEG_S)
    p.add_argument("--still-seconds", type=float, default=DEFAULT_STILL_SECONDS)
    p.add_argument("--max-follow-deg-s", type=float, default=DEFAULT_MAX_FOLLOW_DEG_S)
    p.add_argument("--max-travel-deg", type=float, default=DEFAULT_MAX_TRAVEL_DEG)
    p.add_argument("--sag-limit-deg", type=float, default=DEFAULT_SAG_LIMIT_DEG)
    p.add_argument("--sag-seconds", type=float, default=DEFAULT_SAG_SECONDS)
    p.add_argument("--save", default=None, help="종료 시 최종 각도를 이 yaml 로 저장")
    p.add_argument("--dry-run", action="store_true", help="포트를 열지 않고 쓸 값만 출력")
    return p


def build_session_config(args) -> SessionConfig:
    ids = parse_motor_ids(args.ids)
    names = tuple(n.strip() for n in args.names.split(",")) if args.names \
        else tuple(f"id{i}" for i in ids)
    if len(names) != len(ids):
        raise ValueError(f"--names {len(names)}개 ≠ --ids {len(ids)}개")
    return SessionConfig(
        ids=ids,
        names=names,
        goal_current_ma=validate_goal_current(args.goal_current),
        rate_hz=float(args.rate_hz),
        follow=build_follow_config(
            args.deadband_deg, args.vel_threshold_deg_s, args.still_seconds,
            args.max_follow_deg_s, args.rate_hz,
        ),
        max_travel_tick=deg_to_tick_span(args.max_travel_deg),
        sag_limit_tick=deg_to_tick_span(args.sag_limit_deg),
        sag_seconds=float(args.sag_seconds),
    )


def print_plan(config: SessionConfig, port: str, baud: int) -> None:
    print(f"포트 {port} @ {baud}")
    print(f"관절 {', '.join(f'{n}(id={i})' for n, i in zip(config.names, config.ids))}")
    print(f"모드 {OP_MODE_CURRENT_POSITION}(Current-based Position) @주소 {ADDR_OPERATING_MODE}")
    torque = ma_to_torque_nm(config.goal_current_ma)
    print(f"Goal Current {config.goal_current_ma} mA = raw {ma_to_raw(config.goal_current_ma)} "
          f"@주소 {ADDR_GOAL_CURRENT}")
    print(f"  ≈ {torque:.4f} N·m · 손이 느끼는 힘 {torque_to_force_n(torque):.2f} N "
          f"({torque_to_force_n(torque) * 102:.0f} gf @ {HAND_LEVER_M * 100:.0f}cm)")
    print(f"주기 {config.rate_hz} Hz")
    print(f"데드밴드 {config.follow.deadband_tick} tick "
          f"({config.follow.deadband_tick / TICK_PER_DEG:.2f}°)")
    print(f"정지 판정 {config.follow.still_cycles_needed} 주기 · "
          f"속도 임계 {config.follow.vel_threshold_raw:.1f} raw")
    print(f"1주기 최대 추종 {config.follow.max_step_tick} tick · "
          f"이동 한계 ±{config.max_travel_tick} tick")
    print(f"처짐 판정 {config.sag_seconds}s 동안 ±{config.sag_limit_tick} tick")


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = build_session_config(args)
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    print_plan(config, port, args.baud)
    if args.dry_run:
        print("\n--dry-run — 포트를 열지 않았다")
        return 0

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    try:
        controller = CompliantController(port, args.baud)
    except RuntimeError as exc:
        print(f"❌ {exc}\n   포트 권한을 확인할 것: `sudo usermod -aG dialout $USER` 후 재로그인",
              file=sys.stderr)
        return 2

    try:
        for dxl_id in config.ids:
            controller.ping(dxl_id)
        seeds = {i: controller.configure_compliant_mode(i, config.goal_current_ma)
                 for i in config.ids}

        if not passes_sag_gate(controller, config, seeds):
            return 1

        print("\n손으로 맞추세요. 끝나면 Ctrl-C\n")
        final = run_session(controller, config, seeds, lambda: stop["now"])
        if args.save:
            Path(args.save).write_text(
                render_angles_yaml(config.names, config.ids, final), encoding="utf-8"
            )
            print(f"저장: {args.save}")
        return 0
    finally:
        for dxl_id in config.ids:
            try:
                controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
            except Exception as exc:                      # 종료 경로는 절대 막지 않는다
                print(f"⚠ id={dxl_id} 토크 해제 실패: {exc}", file=sys.stderr)
        controller.close()
        print("토크 해제 · 포트 닫음")


if __name__ == "__main__":
    raise SystemExit(main())
