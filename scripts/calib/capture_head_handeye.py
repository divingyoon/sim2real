#!/usr/bin/env python3
"""hand-eye 캘리브용 샘플 수집 — 목을 여러 자세로 옮기며 ChArUco 를 찍는다.

**보드 위치를 몰라도 된다.** `solve_head_handeye.py` 가 `T_neck_cam`(카메라 장착)과
`T_base_board`(보드가 테이블 어디 있는지)를 **동시에** 풀어낸다. 그래서 테이블
세팅이 예전과 달라도 상관없다.

기계가 둘로 나뉘어 있어 이 스크립트가 양쪽을 번갈아 부린다:

    local5090   목 모터 (U2D2)          ← 자세 지령 · 실제 각도 읽기
    vision-3090 RealSense + capture     ← 프레임 캡처 (ssh)
    local5090   charuco_calib           ← 검출 (cv2 4.14 로컬)

★MJPEG 스트리머가 카메라를 붙잡고 있으면 캡처가 실패한다 — 시작할 때 멈추고
끝나면 되살린다.

★자세는 **지령이 아니라 실측 인코더 각**을 기록한다. FK 에 지령을 넣으면 처짐만큼
틀어진다 (I 게인을 넣어도 0.13° 는 남는다).

    python capture_head_handeye.py --out ../logs/handeye/run1 --dry-run
    python capture_head_handeye.py --out ../logs/handeye/run1
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from head_compliant_hold import CompliantController, tick_to_deg
from head_home import DEFAULT_CONFIG, HeadHome, apply_one, load_head_home

VISION_HOST = "vision-3090"
REMOTE_PY = "~/rl_ws/perception_plus_plus/.venv/bin/python"
REMOTE_CAPTURE = "~/rl_ws/perception_plus_plus/scripts/capture_frame.py"
REMOTE_NPZ = "/tmp/handeye_frame.npz"
STREAM_CTL = "/tmp/head_stream_ctl.sh"

SETTLE_SECONDS = 2.5
ANGLE_SAMPLES = 15
MIN_CORNERS = 12          # 24 중 절반 미만이면 자세가 나쁘다고 본다
MAX_REPROJ_PX = 1.0


def make_poses(pan_lo: float, pan_hi: float, tilt_lo: float, tilt_hi: float,
               steps: int) -> tuple[tuple[float, float], ...]:
    """격자 자세. **모서리를 먼저** 돌아 회전 폭을 일찍 확보한다.

    ★hand-eye 의 병진 정확도는 **회전 폭**에 비례한다. 폭이 좁으면 잔차가 mm 단위라도
    병진이 병렬해라 값이 통째로 틀린다(2026-09-01: 폭 24.7° 에서 잔차 5 mm 인데
    보드 높이가 z=1.48 m 로 나왔다 — 실제로는 테이블 위다).

    범위를 넓히면 보드가 시야에서 나간다. **보드를 더 멀리 두면** 화각에서 차지하는
    각도가 줄어 그만큼 회전 여유가 생긴다 — 가장 확실한 개선책이다.
    """
    pans = [pan_lo + (pan_hi - pan_lo) * i / (steps - 1) for i in range(steps)]
    tilts = [tilt_lo + (tilt_hi - tilt_lo) * i / (steps - 1) for i in range(steps)]
    corners = [(p, t) for p in (pan_lo, pan_hi) for t in (tilt_lo, tilt_hi)]
    grid = [(p, t) for p in pans for t in tilts]
    return tuple(dict.fromkeys(corners + grid))


def default_poses() -> tuple[tuple[float, float], ...]:
    """기준 자세 주변. 보드가 시야에서 나가지 않는 범위로 잡았다.

    hand-eye 는 **회전 다양성**을 먹고 산다 — 두 축을 함께 흔든 조합이 핵심이다.
    보드(0.21x0.15 m)가 0.64 m 에서 약 19°x13° 를 차지하고 FOV 는 55°x43° 이므로
    여유가 크지 않다. 검출이 실패한 자세는 건너뛴다.
    """
    pans = (-10.0, -5.0, 0.0, 5.0, 10.0)
    tilts = (-27.0, -23.5, -20.0, -16.5, -13.0)
    grid = [(p, t) for p in pans for t in tilts]
    corners = [(p, t) for p in (-10.0, 10.0) for t in (-27.0, -13.0)]
    return tuple(dict.fromkeys(corners + grid))       # 모서리 먼저, 중복 제거


@dataclass(frozen=True)
class Sample:
    pan_cmd: float
    tilt_cmd: float
    pan_meas: float
    tilt_meas: float
    corners: int
    reproj_px: float
    t_cam_board: list[list[float]]


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def stream_control(action: str) -> None:
    """원격 MJPEG 스트리머를 멈추거나 되살린다. 실패해도 캘리브를 막지 않는다."""
    result = _run(["ssh", "-o", "BatchMode=yes", VISION_HOST,
                   f"{STREAM_CTL} {action}"], timeout=60)
    print(f"  스트림 {action}: {result.stdout.strip() or result.stderr.strip() or 'ok'}")


def measure_angles(controller: CompliantController, ids: tuple[int, int]) -> tuple[float, float]:
    """실측 인코더 각(deg). 지령이 아니라 이 값을 FK 에 넣는다."""
    return tuple(
        statistics.fmean([tick_to_deg(controller.read_present_tick(i))
                          for _ in range(ANGLE_SAMPLES)])
        for i in ids
    )


def capture_and_detect(work_dir: Path, index: int) -> dict | None:
    """vision-3090 에서 한 프레임 → 로컬로 → ChArUco 검출. 실패하면 None."""
    remote = _run(["ssh", "-o", "BatchMode=yes", VISION_HOST,
                   f"{REMOTE_PY} {REMOTE_CAPTURE} --out {REMOTE_NPZ} "
                   f"--preview /tmp/handeye_frame.png"], timeout=120)
    if remote.returncode != 0:
        print(f"  ❌ 원격 캡처 실패: {remote.stderr.strip()[:200]}")
        return None

    npz = work_dir / f"frame_{index:02d}.npz"
    if _run(["scp", "-q", "-o", "BatchMode=yes",
             f"{VISION_HOST}:{REMOTE_NPZ}", str(npz)], timeout=60).returncode != 0:
        print("  ❌ 프레임 복사 실패")
        return None

    here = Path(__file__).resolve().parent
    detect = _run([sys.executable, str(here / "charuco_calib.py"), str(npz)], timeout=120)
    if detect.returncode != 0:
        print(f"  ❌ 검출 실패: {detect.stderr.strip()[:200]}")
        return None
    try:
        return json.loads(detect.stdout)
    except json.JSONDecodeError:
        print(f"  ❌ 검출 출력이 JSON 이 아니다: {detect.stdout[:150]}")
        return None


def goto(controller: CompliantController, config: HeadHome,
         ids: tuple[int, int], pan: float, tilt: float) -> None:
    moved = HeadHome(**{**config.__dict__,
                        "targets_deg": {ids[0]: pan, ids[1]: tilt}})
    for dxl_id in ids:
        apply_one(controller, moved, dxl_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--out", required=True, help="샘플을 모을 디렉터리")
    p.add_argument("--max-poses", type=int, default=0, help="0 이면 전부")
    p.add_argument("--pan-range", default=None, help="lo,hi (deg). 넓을수록 병진이 잘 풀린다")
    p.add_argument("--tilt-range", default=None, help="lo,hi (deg)")
    p.add_argument("--steps", type=int, default=5, help="축당 격자 수")
    p.add_argument("--dry-run", action="store_true", help="자세 목록만 출력")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_head_home(Path(args.config))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"❌ 설정: {exc}", file=sys.stderr)
        return 2

    if args.pan_range or args.tilt_range:
        try:
            pan_lo, pan_hi = (float(v) for v in (args.pan_range or "-10,10").split(","))
            tilt_lo, tilt_hi = (float(v) for v in (args.tilt_range or "-27,-13").split(","))
        except ValueError:
            print("❌ --pan-range / --tilt-range 는 'lo,hi' 형식", file=sys.stderr)
            return 2
        poses = make_poses(pan_lo, pan_hi, tilt_lo, tilt_hi, max(2, args.steps))
    else:
        poses = default_poses()
    if args.max_poses:
        poses = poses[:args.max_poses]

    ids = tuple(config.targets_deg)
    if len(ids) != 2:
        print(f"❌ 모터가 2개여야 한다: {ids}", file=sys.stderr)
        return 2

    print(f"자세 {len(poses)}개 · 포트 {config.port} @ {config.baud} · I={config.position_i_gain}")
    print(f"  pan {min(p for p,_ in poses):+.1f}~{max(p for p,_ in poses):+.1f}° · "
          f"tilt {min(t for _,t in poses):+.1f}~{max(t for _,t in poses):+.1f}°")
    print(f"  샘플당 {SETTLE_SECONDS}s 정착 + 캡처 + 검출")
    if args.dry_run:
        for i, (p, t) in enumerate(poses):
            print(f"  {i:2d}: pan {p:+6.1f}° tilt {t:+6.1f}°")
        print("\n--dry-run — 하드웨어를 건드리지 않았다")
        return 0

    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    stream_control("stop")

    samples: list[Sample] = []
    controller = CompliantController(config.port, config.baud)
    try:
        for index, (pan, tilt) in enumerate(poses):
            print(f"[{index + 1}/{len(poses)}] pan {pan:+6.1f}° tilt {tilt:+6.1f}°", flush=True)
            goto(controller, config, ids, pan, tilt)
            time.sleep(SETTLE_SECONDS)
            pan_meas, tilt_meas = measure_angles(controller, ids)

            detected = capture_and_detect(work, index)
            if detected is None:
                continue
            corners = int(detected.get("charuco_corners", 0))
            reproj = float(detected.get("reproj_px", 9e9))
            if corners < MIN_CORNERS or reproj > MAX_REPROJ_PX:
                print(f"  건너뜀 — 코너 {corners} · 재투영 {reproj:.3f} px")
                continue
            samples.append(Sample(pan, tilt, pan_meas, tilt_meas, corners, reproj,
                                  detected["T_cam_board"]))
            print(f"  ✓ 실측 pan {pan_meas:+.3f}° tilt {tilt_meas:+.3f}° · "
                  f"코너 {corners} · 재투영 {reproj:.3f} px")
        # ★끝나면 기준 자세로 되돌린다. 안 그러면 마지막 캡처 자세에 그대로 서서
        #   다음 사람이 "왜 pan 이 돌아가 있지?" 로 헤맨다(2026-09-01 에 실제로 그랬다).
        print("\n기준 자세로 복귀")
        goto(controller, config, ids, *[config.targets_deg[i] for i in ids])
        time.sleep(SETTLE_SECONDS)
        for dxl_id in ids:
            print(f"  {config.names[dxl_id]}: "
                  f"{tick_to_deg(controller.read_present_tick(dxl_id)):+.2f}°")
    finally:
        controller.close()
        stream_control("start")

    out = work / "samples.json"
    out.write_text(json.dumps([s.__dict__ for s in samples], indent=1), encoding="utf-8")
    print(f"\n{len(samples)}/{len(poses)} 자세 성공 → {out}")
    if len(samples) < 3:
        print("❌ hand-eye 에는 최소 3자세가 필요하다", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
