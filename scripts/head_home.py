#!/usr/bin/env python3
"""head 를 기준 자세로 보내고 게인까지 **올바른 순서로** 적용한다. local5090 기본값.

값은 `config/head_home.yaml` 하나에 모여 있다 — 자세·baud·게인의 단일 진실원천이다.

★★**순서가 핵심이다.** DYNAMIXEL 은 Operating Mode 를 쓰면 제어 게인을 그 모드의
기본값으로 **리셋한다.** 게인을 모드보다 먼저 넣으면 조용히 지워지고 처짐만 남는다
(2026-09-01 에 겪었다 — `apply_head_gains.py` 로 I=400 을 넣은 뒤 `head_goto_hold.py`
를 돌렸더니 I 가 0 으로 돌아가 tilt 가 +1.56° 처졌다).

    토크 off → Operating Mode → **게인** → 목표 위치 → 토크 on

Position I Gain 은 RAM 이라 **전원을 끄면 사라진다.** bringup 마다 이 스크립트를 돌린다.

    python head_home.py                 # 계획만 출력
    python head_home.py --execute       # 적용 + 검증
    python head_home.py --execute --pan 5 --tilt -25   # 일회성 다른 자세
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from head_compliant_hold import (
    ADDR_GOAL_POSITION,
    ADDR_OPERATING_MODE,
    ADDR_POSITION_I_GAIN,
    ADDR_TORQUE_ENABLE,
    TORQUE_OFF,
    TORQUE_ON,
    CompliantController,
    tick_to_deg,
)
from head_position_hold_node import ADDR_PROFILE_ACCELERATION, ADDR_PROFILE_VELOCITY, deg_to_tick

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "head_home.yaml"
GAIN_MAX = 16383
SETTLE_SECONDS = 4.0
VERIFY_SAMPLES = 30
VERIFY_TOLERANCE_DEG = 0.3

_REQUIRED = ("port", "baud", "motors", "position_i_gain", "operating_mode",
             "profile_acceleration", "profile_velocity")


@dataclass(frozen=True)
class HeadHome:
    port: str
    baud: int
    targets_deg: dict[int, float]
    names: dict[int, str]
    position_i_gain: int
    operating_mode: int
    profile_acceleration: int
    profile_velocity: int


def load_head_home(path: Path) -> HeadHome:
    """설정을 읽고 **경계에서 검증한다** — 잘못된 값이 모터까지 가면 안 된다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"설정에 빠진 항목: {', '.join(missing)}")

    gain = int(raw["position_i_gain"])
    if not 0 <= gain <= GAIN_MAX:
        raise ValueError(f"position_i_gain 게인은 0~{GAIN_MAX}: {gain}")

    targets: dict[int, float] = {}
    names: dict[int, str] = {}
    for name, motor in dict(raw["motors"]).items():
        dxl_id = int(motor["id"])
        deg = float(motor["deg"])
        if dxl_id in targets:
            raise ValueError(f"id 가 중복이다: {dxl_id}")
        if not -180.0 <= deg <= 180.0:
            raise ValueError(f"{name}: 각도는 -180~180: {deg}")
        targets[dxl_id] = deg
        names[dxl_id] = str(name)

    return HeadHome(
        port=str(raw["port"]), baud=int(raw["baud"]), targets_deg=targets, names=names,
        position_i_gain=gain, operating_mode=int(raw["operating_mode"]),
        profile_acceleration=int(raw["profile_acceleration"]),
        profile_velocity=int(raw["profile_velocity"]),
    )


def apply_one(controller: CompliantController, config: HeadHome, dxl_id: int) -> None:
    """★순서가 전부다. 두 가지 펌웨어 동작에 각각 물린 적이 있다(2026-09-01).

    1. **Operating Mode 를 쓰면 제어 게인이 그 모드 기본값으로 리셋된다**
       → 게인은 모드 **뒤에**.
    2. **Torque Enable 이 0→1 이 되면 Goal Position 이 Present Position 으로 덮어써진다**
       (급격한 점프 방지) → 목표는 토크 **뒤에**. 앞에 쓰면 토크가 꺼진 동안 중력에
       떨어진 자리가 목표가 되어 조용히 어긋난다.

        토크off → 모드 → 게인 → 프로파일 → 토크on → 목표
    """
    controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
    controller.write1(dxl_id, ADDR_OPERATING_MODE, config.operating_mode, "operating mode")
    controller.write2(dxl_id, ADDR_POSITION_I_GAIN, config.position_i_gain, "position i gain")
    controller.write4(dxl_id, ADDR_PROFILE_ACCELERATION, config.profile_acceleration, "accel")
    controller.write4(dxl_id, ADDR_PROFILE_VELOCITY, config.profile_velocity, "vel")
    controller.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")
    controller.write4(dxl_id, ADDR_GOAL_POSITION,
                      deg_to_tick(config.targets_deg[dxl_id]), "goal")


def verify(controller: CompliantController, config: HeadHome) -> bool:
    time.sleep(SETTLE_SECONDS)
    ok = True
    for dxl_id, target in config.targets_deg.items():
        samples = [tick_to_deg(controller.read_present_tick(dxl_id))
                   for _ in range(VERIFY_SAMPLES)]
        mean = statistics.fmean(samples)
        error = mean - target
        gain = controller.read2_signed(dxl_id, ADDR_POSITION_I_GAIN, "i")
        good = abs(error) <= VERIFY_TOLERANCE_DEG and gain == config.position_i_gain
        ok &= good
        print(f"  {config.names[dxl_id]}: 목표 {target:+.2f}° · 실제 {mean:+.3f}° · "
              f"오차 {error:+.3f}° · σ {statistics.pstdev(samples):.3f}° · "
              f"I={gain} · {'✓' if good else '❌'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pan", type=float, default=None, help="일회성 덮어쓰기")
    parser.add_argument("--tilt", type=float, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    try:
        config = load_head_home(Path(args.config))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"❌ 설정을 읽을 수 없다: {exc}", file=sys.stderr)
        return 2

    overrides = {"pan": args.pan, "tilt": args.tilt}
    targets = dict(config.targets_deg)
    for dxl_id, name in config.names.items():
        if overrides.get(name) is not None:
            targets[dxl_id] = float(overrides[name])
    config = HeadHome(**{**config.__dict__, "targets_deg": targets})

    print(f"설정 {args.config}")
    print(f"포트 {config.port} @ {config.baud} · 모드 {config.operating_mode} · "
          f"Position I Gain {config.position_i_gain}")
    for dxl_id, target in config.targets_deg.items():
        print(f"  {config.names[dxl_id]}(id={dxl_id}) → {target:+.2f}° "
              f"(tick {deg_to_tick(target)})")
    print("순서: 토크off → 모드 → 게인 → 목표 → 토크on  "
          "(모드가 게인을 리셋하므로 뒤바꾸면 안 된다)")

    if not args.execute:
        print("\n--execute 없이 실행함 — 아무것도 쓰지 않았다")
        return 0

    controller = CompliantController(config.port, config.baud)
    try:
        for dxl_id in config.targets_deg:
            controller.ping(dxl_id)
            apply_one(controller, config, dxl_id)
        print(f"\n{SETTLE_SECONDS:g}s 정착 후 검증")
        ok = verify(controller, config)
        print("\n★Position I Gain 은 RAM 이다 — 전원을 끄면 사라진다. bringup 마다 실행할 것")
        return 0 if ok else 1
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
