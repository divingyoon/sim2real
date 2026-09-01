#!/usr/bin/env python3
"""손으로 밀 때 모터가 실제로 어떻게 반응하는지 관찰한다. **목표는 고정한다.**

"조금만 움직여도 힘이 풀린다"의 원인을 가른다:

  ⓐ 설계대로다 — `head_compliant_hold.py` 는 밀면 목표가 **따라가므로**(FOLLOW)
     저항이 0 으로 떨어진다. 그게 정상이다. 이 스크립트는 목표를 **고정**하므로
     저항이 유지되어야 한다. 여기서도 저항이 사라지면 ⓐ가 아니다.
  ⓑ Overload 셧다운 — XC330 의 Shutdown 레지스터 0x35 에 Overload 비트가 있다.
     걸리면 **모터가 스스로 토크를 끈다.** torque_enable 이 0 으로 바뀌면 이것이다.
  ⓒ 전류 상한이 낮아 애초에 저항이 없다 — present current 가 상한에 붙는데도
     밀린다면 이것이다. 전류를 올려야 한다.

    python probe_head_push_response.py --ids 2 --names tilt --goal-current 150 --seconds 20
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from head_compliant_hold import (
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_TORQUE_ENABLE,
    DEFAULT_BAUD,
    TICK_PER_DEG,
    TORQUE_OFF,
    CompliantController,
    autodetect_port,
    discover_ports,
    ma_to_torque_nm,
    parse_motor_ids,
    remaining_sleep,
    torque_to_force_n,
    validate_goal_current,
)

DEFAULT_SECONDS = 20.0
RATE_HZ = 20.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ids", default="2")
    p.add_argument("--names", default=None)
    p.add_argument("--goal-current", type=float, default=150.0)
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    return p


def observe(
    controller: CompliantController, ids: tuple[int, ...], names: tuple[str, ...],
    goal_current_ma: float, seconds: float, should_stop,
) -> dict[int, dict]:
    """목표 고정 상태로 관찰. 반환값은 관절별 요약."""
    seeds = {i: controller.configure_compliant_mode(i, goal_current_ma) for i in ids}
    summary = {i: {"max_err_deg": 0.0, "max_cur_ma": 0.0,
                   "torque_dropped": False, "error_latched": 0} for i in ids}
    period = 1.0 / RATE_HZ
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline and not should_stop():
        started = time.monotonic()
        rows = []
        for dxl_id, name in zip(ids, names):
            err_deg = (controller.read_present_tick(dxl_id) - seeds[dxl_id]) / TICK_PER_DEG
            cur = controller.read_present_current_ma(dxl_id)
            torque_on = bool(controller.read1(dxl_id, ADDR_TORQUE_ENABLE, "t"))
            hw = controller.read1(dxl_id, ADDR_HARDWARE_ERROR_STATUS, "hw")

            s = summary[dxl_id]
            s["max_err_deg"] = max(s["max_err_deg"], abs(err_deg))
            s["max_cur_ma"] = max(s["max_cur_ma"], abs(cur))
            if not torque_on:
                s["torque_dropped"] = True
            s["error_latched"] |= hw

            rows.append(f"{name} err{err_deg:+7.2f}° cur{cur:+7.1f}mA "
                        f"{'TQ' if torque_on else '⚠OFF'}{'' if hw == 0 else f' ⚠hw0x{hw:02X}'}")
        print("\r" + " · ".join(rows) + "   ", end="", flush=True)
        time.sleep(remaining_sleep(period, time.monotonic() - started))

    print()
    return summary


def report(summary: dict[int, dict], ids, names, goal_current_ma: float) -> None:
    print(f"\n{'='*60}")
    for dxl_id, name in zip(ids, names):
        s = summary[dxl_id]
        print(f"{name}: 최대 편차 {s['max_err_deg']:.2f}° · "
              f"최대 전류 {s['max_cur_ma']:.1f} mA / 상한 {goal_current_ma:g} mA")
        if s["torque_dropped"]:
            print("  ⚠ **토크가 도중에 꺼졌다** → Overload 셧다운(ⓑ). "
                  "전류 상한을 낮추거나 부하를 줄일 것")
        if s["error_latched"]:
            print(f"  ⚠ 하드웨어 에러 래치 0x{s['error_latched']:02X}")
        if not s["torque_dropped"] and s["max_cur_ma"] >= goal_current_ma * 0.9:
            print("  → 전류가 상한에 붙었다. 그런데도 밀렸다면 상한이 낮은 것(ⓒ) — "
                  "--goal-current 를 올릴 것")
        if not s["torque_dropped"] and s["max_cur_ma"] < goal_current_ma * 0.5:
            print("  → 전류가 상한 근처에도 못 갔다. 목표 고정인데 저항이 없다면 "
                  "기구가 헐겁거나(백래시) 밀린 양이 데드밴드 안이다")


def main() -> int:
    args = build_parser().parse_args()
    try:
        ids = parse_motor_ids(args.ids)
        names = tuple(n.strip() for n in args.names.split(",")) if args.names \
            else tuple(f"id{i}" for i in ids)
        goal = validate_goal_current(args.goal_current)
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    torque = ma_to_torque_nm(goal)
    print(f"포트 {port} @ {args.baud} · 목표 **고정**(추종 안 함)")
    print(f"Goal Current {goal:g} mA ≈ {torque:.4f} N·m "
          f"({torque_to_force_n(torque):.2f} N @ 5cm)")
    print(f"{args.seconds:g}초 동안 손으로 밀어 보세요 — 저항이 사라지는지 봅니다\n")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    controller = CompliantController(port, args.baud)
    try:
        for dxl_id in ids:
            controller.ping(dxl_id)
        summary = observe(controller, ids, names, goal, args.seconds, lambda: stop["now"])
        report(summary, ids, names, goal)
        return 0
    finally:
        for dxl_id in ids:
            try:
                controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
            except Exception as exc:
                print(f"⚠ id={dxl_id} 토크 해제 실패: {exc}", file=sys.stderr)
        controller.close()
        print("토크 해제 · 포트 닫음")


if __name__ == "__main__":
    raise SystemExit(main())
