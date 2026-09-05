#!/usr/bin/env python3
"""head 모터의 **Position I Gain** 을 적용한다 (전원을 켤 때마다).

XC330 은 Position I Gain 이 **0 으로 출하**되고, 이 레지스터는 **RAM** 이라
전원을 끄면 0 으로 돌아간다. I=0 이면 P 만으로 중력을 이겨야 해서 정상상태 오차가
남는다 — 2026-09-01 tilt 가 −20° 지령에서 **+1.49° 처졌다**(0.56 m 에서 약 15 mm).

2026-09-01 스윕(`probe_head_position_i_gain.py`) 결과:

    I=0 → +0.44° · I=200 → +0.09° · **I=400 → +0.00°** · I=800 → +0.00°
    진동(σ)은 800 까지도 0.000° — 400 은 오차 0 이면서 여유가 크다.

pan 은 수직축이라 중력 부하가 없어 I 가 없어도 0.13° 다. 그래도 같이 넣어 둔다.

    python apply_head_gains.py            # 현재 값 확인만
    python apply_head_gains.py --execute  # I=400 적용
    python apply_head_gains.py --restore --execute   # 출하값 0 으로
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_compliant_hold import (
    ADDR_POSITION_D_GAIN,
    ADDR_POSITION_I_GAIN,
    ADDR_POSITION_P_GAIN,
    DEFAULT_BAUD,
    CompliantController,
    autodetect_port,
    discover_ports,
    parse_motor_ids,
)

TUNED_I = 400
VENDOR_I = 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ids", default="1,2")
    p.add_argument("--names", default="pan,tilt")
    p.add_argument("--i-gain", type=int, default=None, help=f"기본 {TUNED_I}")
    p.add_argument("--restore", action="store_true", help=f"출하값 {VENDOR_I}")
    p.add_argument("--execute", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        ids = parse_motor_ids(args.ids)
        names = tuple(n.strip() for n in args.names.split(","))
        if len(names) != len(ids):
            raise ValueError(f"--names {len(names)}개 ≠ --ids {len(ids)}개")
        port = args.port or autodetect_port(discover_ports())
    except (ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    target = VENDOR_I if args.restore else (TUNED_I if args.i_gain is None else args.i_gain)

    controller = CompliantController(port, args.baud)
    try:
        for dxl_id, name in zip(ids, names):
            controller.ping(dxl_id)
            before = controller.read2_signed(dxl_id, ADDR_POSITION_I_GAIN, "i")
            p_gain = controller.read2_signed(dxl_id, ADDR_POSITION_P_GAIN, "p")
            d_gain = controller.read2_signed(dxl_id, ADDR_POSITION_D_GAIN, "d")
            if not args.execute:
                print(f"{name}(id={dxl_id}): P={p_gain} I={before} D={d_gain} "
                      f"→ I={target} (--execute 로 적용)")
                continue
            controller.write2(dxl_id, ADDR_POSITION_I_GAIN, target, "position i gain")
            after = controller.read2_signed(dxl_id, ADDR_POSITION_I_GAIN, "i")
            ok = "✓" if after == target else "❌ 검증 실패"
            print(f"{name}(id={dxl_id}): P={p_gain} I={before}→{after} D={d_gain} {ok}")
        if not args.execute:
            print("\n(확인만 함 — 아무것도 쓰지 않았다)")
        else:
            print("\n★이 값은 RAM 이다 — 전원을 끄면 0 으로 돌아간다. bringup 마다 다시 실행할 것")
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
