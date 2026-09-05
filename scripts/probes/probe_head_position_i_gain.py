#!/usr/bin/env python3
"""중력 처짐을 지우는 **Position I Gain** 을 찾는다.

XC330 은 Position I Gain 이 **0 으로 출하**된다. P 만으로는 중력처럼 일정한 부하를
이기지 못해 정상상태 오차가 남는다 — 2026-09-01 head tilt 가 −20° 지령에서
**+1.49° 처졌다**(0.56 m 거리에서 약 15 mm). pan 은 수직축이라 0.13° 로 멀쩡했다.

적분항을 올리면 오차가 사라지지만 **와인드업으로 진동**할 수 있으므로, 정착 후
표준편차를 같이 재서 진동이 시작되는 지점 앞에서 멈춘다.

    python probe_head_position_i_gain.py --id 2 --name tilt --target-deg -20
"""

from __future__ import annotations

import argparse
import signal
import statistics
import sys
import time

from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_compliant_hold import (
    ADDR_GOAL_POSITION,
    ADDR_POSITION_I_GAIN,
    ADDR_POSITION_P_GAIN,
    ADDR_TORQUE_ENABLE,
    DEFAULT_BAUD,
    CompliantController,
    autodetect_port,
    discover_ports,
    tick_to_deg,
)
from head_position_hold_node import deg_to_tick

DEFAULT_GAINS = "0,50,100,200,400,800"
SETTLE_SECONDS = 3.0
SAMPLE_SECONDS = 2.0
SAMPLE_HZ = 20.0
#: 정착 후 표준편차가 이보다 크면 진동으로 본다. 기저 잡음(≈0.05°) 위로 잡았다.
VIBRATION_SIGMA_DEG = 0.15
GAIN_MAX = 16383          # 2바이트 레지스터의 유효 상한


def parse_gains(text: str) -> tuple[int, ...]:
    values = tuple(int(p) for p in text.split(",") if p.strip())
    if not values:
        raise ValueError("게인을 하나 이상 지정할 것")
    for v in values:
        if not 0 <= v <= GAIN_MAX:
            raise ValueError(f"게인은 0~{GAIN_MAX}: {v}")
    if list(values) != sorted(values):
        raise ValueError("게인은 오름차순이어야 한다 — 진동은 위쪽에서 시작한다")
    return values


def measure(
    controller: CompliantController, dxl_id: int, target_tick: int, gain: int
) -> tuple[float, float]:
    """(평균 오차 deg, 표준편차 deg). 지령을 다시 넣어 적분항을 새로 쌓게 한다."""
    controller.write2(dxl_id, ADDR_POSITION_I_GAIN, gain, "position i gain")
    controller.write4(dxl_id, ADDR_GOAL_POSITION, target_tick, "goal")
    time.sleep(SETTLE_SECONDS)

    samples: list[float] = []
    period = 1.0 / SAMPLE_HZ
    deadline = time.monotonic() + SAMPLE_SECONDS
    while time.monotonic() < deadline:
        samples.append(tick_to_deg(controller.read_present_tick(dxl_id))
                       - tick_to_deg(target_tick))
        time.sleep(period)
    return statistics.fmean(samples), statistics.pstdev(samples)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--id", type=int, default=2)
    p.add_argument("--name", default="tilt")
    p.add_argument("--target-deg", type=float, required=True)
    p.add_argument("--gains", default=DEFAULT_GAINS)
    p.add_argument("--tolerance-deg", type=float, default=0.2)
    p.add_argument("--restore", action="store_true", help="끝나고 I 게인을 0 으로 되돌린다")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        gains = parse_gains(args.gains)
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    target_tick = deg_to_tick(args.target_deg)
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    controller = CompliantController(port, args.baud)

    try:
        controller.ping(args.id)
        p_gain = controller.read2_signed(args.id, ADDR_POSITION_P_GAIN, "p")
        original_i = controller.read2_signed(args.id, ADDR_POSITION_I_GAIN, "i")
        controller.write1(args.id, ADDR_TORQUE_ENABLE, 1, "torque on")
        print(f"{args.name}(id={args.id}) · 목표 {args.target_deg:+.2f}° "
              f"(tick {target_tick}) · Position P={p_gain} · 시작 I={original_i}")
        print(f"단계마다 {SETTLE_SECONDS:g}s 정착 후 {SAMPLE_SECONDS:g}s 측정\n")

        best: int | None = None
        for gain in gains:
            if stop["now"]:
                print("중단됨"); break
            mean, sigma = measure(controller, args.id, target_tick, gain)
            shaky = sigma > VIBRATION_SIGMA_DEG
            close = abs(mean) <= args.tolerance_deg
            mark = "진동 ⚠" if shaky else ("✓" if close else "처짐")
            print(f"  I={gain:>5} → 오차 {mean:+6.2f}° · σ {sigma:.3f}°  {mark}", flush=True)
            if shaky:
                print("   ↑ 여기서 진동이 시작된다. 더 올리지 않는다")
                break
            if close and best is None:
                best = gain

        if best is None:
            print("\n⇒ 허용오차 안으로 들어온 값이 없다. --gains 를 더 높이거나 "
                  "--tolerance-deg 를 재고할 것", file=sys.stderr)
        else:
            print(f"\n⇒ {args.name} 권장 Position I Gain = **{best}** "
                  f"(진동 없이 오차 ≤{args.tolerance_deg}°)")
        if args.restore:
            controller.write2(args.id, ADDR_POSITION_I_GAIN, original_i, "restore i")
            print(f"I 게인을 {original_i} 로 되돌림")
        return 0
    finally:
        controller.close()
        print("포트 닫음 (토크는 유지 — 자세를 놓지 않는다)")


if __name__ == "__main__":
    raise SystemExit(main())
