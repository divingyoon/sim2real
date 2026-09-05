#!/usr/bin/env python3
"""pan 고정·tilt 스윕으로 ChArUco 보드가 보이는 최적 tilt를 찾아 고정.

vision-3090 전용. perception_plus_plus/.venv(cv2 5.x + dynamixel_sdk +
pyrealsense2)에서 실행한다. 각도 프레임은 head_position_hold_node 와 동일한
unwrapped signed 인코더 deg(-180 = tick 0)이고, 설정(port/baud/id/범위)은
config/head_v1.yaml 에서 읽는다.

Phase A 도구: 각 tilt 후보에서 capture_frame.py 로 한 프레임을 캡처하고
charuco_calib.detect_charuco 로 (코너수, 재투영px)을 측정한 뒤, 최적 tilt에서
torque hold 로 고정한다. 이후 그 자세 그대로 Phase B(카메라 extrinsics 교정)를
진행한다.

  # 후보각만 출력(하드웨어 미동작)
  python find_head_tilt.py --config config/head_v1.yaml --dry-run
  # 실제 스윕 후 최적 tilt로 고정
  python find_head_tilt.py --config config/head_v1.yaml --tilt-step 10 --min-corners 8 --hold

cv2/charuco 는 detect 단계에서만 lazy import 하므로 순수 함수(midpoint,
generate_tilt_candidates, select_best)는 하드웨어/cv2 없이 테스트된다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from head_position_hold_node import (
    DynamixelPositionController,
    deg_to_tick,
    load_calibration_yaml,
    tick_to_deg,
    unwrap_calibrated_range,
    unwrap_target_for_current,
)

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = _HERE.parent / "config" / "head_v1.yaml"
DEFAULT_CAPTURE_SCRIPT = (
    Path.home() / "rl_ws" / "perception_plus_plus" / "scripts" / "capture_frame.py"
)


# --------------------------------------------------------------------------
# 순수 함수 (하드웨어/cv2 무관 — 단위 테스트 대상)
# --------------------------------------------------------------------------
def midpoint(min_deg: float, max_deg: float) -> float:
    """캘리브 범위의 방향성 중앙(기본 자세). pan -180..0 -> -90, tilt 135..325 -> 230."""
    start, end = unwrap_calibrated_range(min_deg, max_deg)
    return round((start + end) / 2.0, 3)


def generate_tilt_candidates(start: float, end: float, step: float) -> list[float]:
    """[min(start,end)..max] 를 step 간격으로, 양 끝을 포함해 오름차순 생성.

    floor 로 스텝 수를 세어 max 를 절대 넘지 않는다(넘으면 캘리브 범위 밖이라
    move_and_settle 이 이동 직전 거부하지만, 여기서부터 초과를 막아 둔다).
    """
    if step <= 0:
        raise ValueError(f"step must be > 0: {step}")
    lo, hi = (start, end) if start <= end else (end, start)
    count = int((hi - lo + 1e-9) // step)
    candidates = [round(lo + i * step, 3) for i in range(count + 1)]
    if candidates[-1] < hi - 1e-9:
        candidates.append(round(hi, 3))
    return candidates


@dataclass(frozen=True)
class SweepRow:
    tilt_deg: float
    corners: int
    reproj_px: float | None


def select_best(rows: list[SweepRow], min_corners: int) -> SweepRow | None:
    """코너수 최대(동률이면 재투영 최소). min_corners 미만/검출실패 행은 제외."""
    ok = [r for r in rows if r.corners >= min_corners and r.reproj_px is not None]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r.corners, r.reproj_px))[0]


def format_table(rows: list[SweepRow]) -> str:
    lines = [f"{'tilt_deg':>10} {'corners':>8} {'reproj_px':>10}"]
    for r in rows:
        reproj = "-" if r.reproj_px is None else f"{r.reproj_px:.3f}"
        lines.append(f"{r.tilt_deg:>10.1f} {r.corners:>8d} {reproj:>10}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 하드웨어/캡처 (lazy — 실행 시에만 의존성 필요)
# --------------------------------------------------------------------------
def capture_frame_npz(
    capture_script: Path,
    out_path: Path,
    warmup: int,
    width: int | None,
    height: int | None,
    timeout_s: float,
) -> None:
    """같은 venv 의 capture_frame.py 를 호출해 rgb/depth/K NPZ 를 기록."""
    cmd = [sys.executable, str(capture_script), "--out", str(out_path), "--warmup", str(warmup)]
    if width:
        cmd += ["--width", str(width)]
    if height:
        cmd += ["--height", str(height)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)


def detect_frame(npz_path: Path) -> tuple[int, float | None]:
    """NPZ 한 장에서 (charuco 코너수, 재투영px). 검출 실패 시 (0, None)."""
    import numpy as np
    import cv2
    from charuco_calib import detect_charuco

    data = np.load(npz_path)
    gray = cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2GRAY)
    result = detect_charuco(gray, data["K"].astype(np.float64))
    if result is None:
        return 0, None
    _T, objp, _imgp, reproj = result
    return len(objp), float(reproj)


def _tolerance_ticks(tolerance_deg: float) -> int:
    return max(1, deg_to_tick(tolerance_deg) - deg_to_tick(0.0))


def move_and_settle(
    controller: DynamixelPositionController,
    dxl_id: int,
    target_deg: float,
    min_deg: float,
    max_deg: float,
    tolerance_deg: float,
    timeout_s: float,
) -> None:
    """현재 위치 기준으로 target 을 캘리브 범위로 unwrap 해 이동, 도달까지 대기."""
    current = tick_to_deg(controller.read_present_tick(dxl_id))
    target = unwrap_target_for_current(current, target_deg, min_deg, max_deg)
    tick = deg_to_tick(target)
    controller.write_goal_tick(dxl_id, tick)
    tol = _tolerance_ticks(tolerance_deg)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if abs(controller.read_present_tick(dxl_id) - tick) <= tol:
            return
        time.sleep(0.05)


def _torque_off_all(controller: DynamixelPositionController, ids: tuple[int, ...]) -> None:
    for dxl_id in ids:
        controller.torque_off(dxl_id)


def run_sweep(args: argparse.Namespace) -> int:
    calibration = load_calibration_yaml(Path(args.config))
    pan = calibration.motors["pan"]
    tilt = calibration.motors["tilt"]

    pan_target = args.pan if args.pan is not None else midpoint(pan.min_deg, pan.max_deg)
    tilt_start = args.tilt_start if args.tilt_start is not None else tilt.min_deg
    tilt_end = args.tilt_end if args.tilt_end is not None else tilt.max_deg
    candidates = generate_tilt_candidates(tilt_start, tilt_end, args.tilt_step)

    print(f"port={calibration.port} baud={calibration.baud}")
    print(
        f"pan id={pan.dxl_id} 고정={pan_target:.1f}deg  "
        f"tilt id={tilt.dxl_id} 스윕={tilt_start:.1f}..{tilt_end:.1f} "
        f"step={args.tilt_step} ({len(candidates)}점)"
    )
    print("tilt 후보:", ", ".join(f"{c:.1f}" for c in candidates))
    if args.dry_run:
        print("[dry-run] 하드웨어 미동작.")
        return 0

    capture_script = Path(args.capture_script).expanduser()
    if not capture_script.exists():
        print(f"캡처 스크립트 없음: {capture_script}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">>> 모터 이동 시작 (MOVING MOTORS) <<<")
    controller = DynamixelPositionController(calibration.port, calibration.baud)
    rows: list[SweepRow] = []
    ids = (pan.dxl_id, tilt.dxl_id)
    holding = False  # True 로 끝날 때만 torque 를 켜둔 채 종료(그 외 모든 경로는 torque off)
    try:
        for dxl_id in ids:
            controller.configure_position_mode(
                dxl_id, args.profile_acceleration, args.profile_velocity
            )
        move_and_settle(
            controller, pan.dxl_id, pan_target, pan.min_deg, pan.max_deg,
            args.tolerance_deg, args.settle_timeout,
        )
        for index, tilt_deg in enumerate(candidates):
            move_and_settle(
                controller, tilt.dxl_id, tilt_deg, tilt.min_deg, tilt.max_deg,
                args.tolerance_deg, args.settle_timeout,
            )
            npz_path = out_dir / f"tilt_{index:02d}_{tilt_deg:.0f}.npz"
            try:
                capture_frame_npz(
                    capture_script, npz_path, args.warmup,
                    args.width, args.height, args.capture_timeout,
                )
                corners, reproj = detect_frame(npz_path)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                print(f"  캡처 실패 tilt={tilt_deg:.1f}: {detail[:160]}", file=sys.stderr)
                corners, reproj = 0, None
            rows.append(SweepRow(tilt_deg, corners, reproj))
            reproj_txt = "-" if reproj is None else f"{reproj:.3f}"
            print(f"  tilt={tilt_deg:>7.1f}  corners={corners:>3}  reproj={reproj_txt}")

        print("\n" + format_table(rows))
        best = select_best(rows, args.min_corners)
        if best is None:
            print(
                f"\n최소 코너({args.min_corners}) 충족 각도 없음 "
                "→ 조명/보드 배치/step 확인 후 재스윕.",
                file=sys.stderr,
            )
            return 1

        print(
            f"\n최적 tilt={best.tilt_deg:.1f}deg "
            f"(corners={best.corners}, reproj={best.reproj_px:.3f})"
        )
        if args.hold:
            move_and_settle(
                controller, tilt.dxl_id, best.tilt_deg, tilt.min_deg, tilt.max_deg,
                args.tolerance_deg, args.settle_timeout,
            )
            holding = True
            print(f"고정 유지: pan={pan_target:.1f} / tilt={best.tilt_deg:.1f} (torque ON)")
            print("→ 이 자세로 Phase B(capture_frame → charuco_calib → recompute_from_cad) 진행.")
        else:
            print("torque OFF (고정하려면 --hold).")
        return 0
    finally:
        # 정상 non-hold, no-board, 예외(comm/캡처/ctrl-c) 모두 여기서 torque off.
        # close() 는 시리얼만 닫고 torque 를 끄지 않으므로 반드시 먼저 해제한다.
        if not holding:
            try:
                _torque_off_all(controller, ids)
            except Exception as exc:  # noqa: BLE001 - 원 예외를 가리지 않도록 삼킴
                print(f"경고: torque off 실패: {exc}", file=sys.stderr)
        controller.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pan 고정·tilt 스윕 ChArUco 가시 최적각 탐색")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="head 캘리브 yaml")
    parser.add_argument("--pan", type=float, default=None, help="pan 고정각(deg, 기본=범위 중앙)")
    parser.add_argument("--tilt-start", type=float, default=None, help="기본=tilt min")
    parser.add_argument("--tilt-end", type=float, default=None, help="기본=tilt max")
    parser.add_argument("--tilt-step", type=float, default=10.0)
    parser.add_argument("--min-corners", type=int, default=8, help="가시 판정 최소 charuco 코너수")
    parser.add_argument("--hold", action="store_true", help="최적 tilt로 torque 고정 유지")
    parser.add_argument("--dry-run", action="store_true", help="후보각만 출력, 하드웨어 미동작")
    parser.add_argument("--capture-script", default=str(DEFAULT_CAPTURE_SCRIPT))
    parser.add_argument("--warmup", type=int, default=8, help="AE 정착 프레임 수")
    parser.add_argument("--capture-timeout", type=float, default=30.0, help="캡처 서브프로세스 워치독(s)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--out-dir", default="/tmp/tiltsweep")
    parser.add_argument("--settle-timeout", type=float, default=3.0)
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--profile-acceleration", type=int, default=20)
    parser.add_argument("--profile-velocity", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
