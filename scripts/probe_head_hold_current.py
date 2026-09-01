#!/usr/bin/env python3
"""전류를 낮은 값부터 올려가며 **중력을 버티는 최소 Goal Current** 를 찾는다.

`head_compliant_hold.py` 의 처짐 게이트는 **시작 자세 한 곳**만 본다. 중력 토크는
자세에 따라 변하므로, 기구 마찰이 붙잡아 주는 자세에서 재면 통과해 놓고 조금만
움직이면 힘이 풀린다 — 2026-09-01 tilt 가 10 mA 에서 정확히 그랬다.

그래서 이 도구를 쓴다:

  1. 토크가 꺼진 상태에서 **중력이 가장 크게 걸리는 자세**(팔이 수평)로 손으로 옮긴다
  2. 이 스크립트를 돌린다 — 전류를 올려가며 단계마다 처짐을 잰다
  3. `select_min_holding` 이 **단조롭게** 통과하는 최소값을 고른다
     (낮은 값의 우연한 통과는 그 위에 실패가 있으면 버린다)

각 단계는 토크를 켰다가 **반드시 다시 끈다**. Ctrl-C 로 끊어도 마찬가지다.

    python probe_head_hold_current.py --ids 2 --names tilt
    python probe_head_hold_current.py --ids 2 --currents 10,20,40,80,150,250
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from head_compliant_hold import (
    ADDR_TORQUE_ENABLE,
    DEFAULT_BAUD,
    DEFAULT_SAG_LIMIT_DEG,
    HAND_LEVER_M,
    TICK_PER_DEG,
    TORQUE_OFF,
    CompliantController,
    autodetect_port,
    deg_to_tick_span,
    discover_ports,
    ma_to_torque_nm,
    parse_motor_ids,
    select_min_holding,
    tick_to_deg,
    torque_to_force_n,
    validate_goal_current,
)

DEFAULT_CURRENTS = "10,20,40,80,150,250"
DEFAULT_HOLD_SECONDS = 3.0


def parse_currents(text: str) -> tuple[float, ...]:
    """오름차순 전류 목록. 반드시 낮은 값부터 올라간다 — 위에서 내려오면 힘겨루기 구간을 지난다."""
    values = tuple(validate_goal_current(float(p)) for p in text.split(",") if p.strip())
    if not values:
        raise ValueError("전류를 하나 이상 지정할 것")
    if list(values) != sorted(values):
        raise ValueError(f"전류는 오름차순이어야 한다: {text!r}")
    return values


def measure_sag(
    controller: CompliantController, dxl_id: int, goal_current_ma: float, hold_seconds: float
) -> int:
    """한 전류에서 처짐(틱)을 잰다. 끝나면 토크를 끈다."""
    seed = controller.configure_compliant_mode(dxl_id, goal_current_ma)
    try:
        time.sleep(hold_seconds)
        return controller.read_present_tick(dxl_id) - seed
    finally:
        controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")


def _describe(ma: float) -> str:
    torque = ma_to_torque_nm(ma)
    return (f"{ma:6.1f} mA ≈ {torque:.4f} N·m "
            f"({torque_to_force_n(torque):.2f} N @ {HAND_LEVER_M * 100:.0f}cm)")


def sweep(
    controller: CompliantController, dxl_id: int, name: str,
    currents: tuple[float, ...], limit_tick: int, hold_seconds: float, should_stop,
) -> list[tuple[float, int]]:
    results: list[tuple[float, int]] = []
    print(f"\n── {name} (id={dxl_id}) · 시작 {tick_to_deg(controller.read_present_tick(dxl_id)):+.2f}°")
    for ma in currents:
        if should_stop():
            print("  중단됨"); break
        sag = measure_sag(controller, dxl_id, ma, hold_seconds)
        verdict = "버팀 ✓" if abs(sag) <= limit_tick else "처짐 ✗"
        print(f"  {_describe(ma)} → {sag / TICK_PER_DEG:+7.2f}°  {verdict}", flush=True)
        results.append((ma, sag))
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ids", default="2", help="중력 부하가 있는 축만 재면 된다")
    p.add_argument("--names", default=None)
    p.add_argument("--currents", default=DEFAULT_CURRENTS, help="오름차순 mA")
    p.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    p.add_argument("--sag-limit-deg", type=float, default=DEFAULT_SAG_LIMIT_DEG)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        ids = parse_motor_ids(args.ids)
        names = tuple(n.strip() for n in args.names.split(",")) if args.names \
            else tuple(f"id{i}" for i in ids)
        if len(names) != len(ids):
            raise ValueError(f"--names {len(names)}개 ≠ --ids {len(ids)}개")
        currents = parse_currents(args.currents)
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    limit_tick = deg_to_tick_span(args.sag_limit_deg)
    print(f"포트 {port} @ {args.baud} · 처짐 한계 ±{args.sag_limit_deg}° ({limit_tick} tick)")
    print(f"단계마다 {args.hold_seconds}s 유지 후 측정 · 매 단계 끝에 토크 해제")
    print("★중력이 가장 크게 걸리는 자세에서 시작해야 의미가 있다 — 지금 자세로 잰다\n")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    try:
        controller = CompliantController(port, args.baud)
    except RuntimeError as exc:
        print(f"❌ {exc}\n   `sudo usermod -aG dialout $USER` 후 재로그인", file=sys.stderr)
        return 2

    try:
        for dxl_id, name in zip(ids, names):
            controller.ping(dxl_id)
            results = sweep(controller, dxl_id, name, currents, limit_tick,
                            args.hold_seconds, lambda: stop["now"])
            best = select_min_holding(results, limit_tick)
            if best is None:
                print(f"  ⇒ {name}: 어떤 값도 못 버틴다. --currents 를 더 높여볼 것 "
                      f"(정책 상한까지)", file=sys.stderr)
            else:
                print(f"  ⇒ {name} 최소 유지 전류 = **{best:g} mA** — "
                      f"head_compliant_hold.py --goal-current {best:g}")
        return 0
    finally:
        for dxl_id in ids:
            try:
                controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
            except Exception as exc:
                print(f"⚠ id={dxl_id} 토크 해제 실패: {exc}", file=sys.stderr)
        controller.close()
        print("\n토크 해제 · 포트 닫음")


if __name__ == "__main__":
    raise SystemExit(main())
