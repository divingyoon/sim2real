#!/usr/bin/env python3
"""여진(`robotctl r2s collect`)이 손끝을 얼마나 더 내리는지 실기 없이 미리 잰다.

왜 필요한가(09.01). r2s collect 는 팔이 **지금 있는 자세 주변**을 흔든다
(`cli.py:_collect_track_run` "Around where the arm is, not the middle of its range").
그 자세를 정책 홈으로 잡으면 정책이 실제로 사는 영역의 동특성을 재게 되지만,
정책 홈은 새끼손끝이 테이블을 스칠락 말락 하는 자세다(08.31 실기 관찰).
`authorize_trajectory` 는 이걸 못 잡는다 — 관절 한계와 속도만 보고, 테이블은
로봇의 관절 공간에 없다.

★검사 공간이 1차원인 이유. 여진의 네 위상이 모두 **관절마다 같은 스칼라**를 쓴다:

    step      : neutral + amplitude                    (α = +1)
    ramp      : neutral + linspace(+1,−1) · amplitude  (α: +1 → −1)
    multisine : neutral + wave · amplitude             (α = wave, 전 관절 공통)

(`identification.py:761-772`) 방문하는 자세는 `q(α) = neutral + α·amp` 라는 **선분**
이지 2^7 개의 상자 꼭짓점이 아니다. 다만 위상을 잇는 `bridge` 는 관절마다 다른
속도로 건너므로 상자 안을 지난다 — 그래서 선분과 꼭짓점을 둘 다 본다(꼭짓점은 공짜).

★왜 절대 높이를 말하지 않는가. 이 저장소에는 FK 가 둘 있고 **서로 17 cm 어긋난다**:
`hdgp/scripts/tools/openarm_fk.py` 는 fabrics 용 `openarm_tesollo.urdf` 에 손으로 맞춘
캘리브 오프셋을 얹은 것이고, 여기서 쓰는 것은 자산 원본 `openarm_tesollo_sensor_rl.urdf`
다. 어느 쪽이 sim 월드의 진실인지는 이 도구가 답할 문제가 아니다. 그래서 절대 z 를
주장하지 않고 **차분만** 낸다 — 차분은 두 FK 가 일치한다(오프셋은 상쇄된다).

  · `기준최저` : 그 자세에서 가장 낮은 링크의 z. **자세끼리 비교할 때만** 의미가 있다.
  · `Δz`      : 여진이 그 최저점을 추가로 내리는 양. 실측 여유에서 이만큼 빼면 된다.

검산. 이 FK 는 정책 홈의 최저점을 `pinky:tip` 으로 짚는다 — 08.31 에 사용자가
"마지막 preset 자세는 새끼 손가락이 테이블에 닿고 있어"라고 관찰한 바로 그 손가락이다.

    python3 probe_excite_clearance.py                     # 전 자세 × 전 scale
    python3 probe_excite_clearance.py --clearance 102     # 실측 여유를 주면 판정까지
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/home/user/rl_ws/robot_control/src")

PROFILE = Path("/home/user/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml")
URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
FIST = Path("/home/user/rl_ws/sim2real/config/right_hand_fist.yaml")
PRESET_NPZ = Path("/home/user/rl_ws/sim2real/logs/shadow/reset_both/reset_right_v2.npz")

ARM = [f"r_aj_{i}" for i in range(1, 8)]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SCALES = (0.10, 0.20, 0.30, 0.50, 0.65, 1.00)
#: 여진 자세로 검토하는 후보들. preset 궤적의 경유점에서 온다.
POSES = {
    "R2 팔접음":     [0.0, 0.9, 0.0, 2.0, 0.0, 0.0, 0.0],
    "R3 손목맞춤":   [0.038, 0.9, 0.6015, 2.0, 0.0294, 0.706, 0.4213],
    "preset 정책홈": [0.038, 0.4012, 0.6015, 0.9643, 0.0294, 0.706, 0.4213],
    "safe j2+0.40":  [0.038, 0.8012, 0.6015, 0.9643, 0.0294, 0.706, 0.4213],
}
#: 08.31 실측 앵커: safe 홈에서 보상 scale 0.9 일 때 palm 이 판 위 250 mm.
#: 최저점은 palm 이 아니라 새끼손끝이므로, 그 차이는 FK 로 환산한다.
SAFE_PALM_ABOVE_TABLE_MM = 250.0


def _load():
    body = yaml.safe_load(PROFILE.read_text())
    joints = {j["canonical"]: j for j in body["joints"]}
    by_source = {j["source"]: (c, j.get("sign", 1)) for c, j in joints.items()}
    raw = yaml.safe_load(FIST.read_text())["joints"]
    missing = [s for s in raw if s not in by_source]
    if missing:
        raise SystemExit(f"주먹 자세의 {missing[0]!r} 가 프로필에 없다")
    fist = {by_source[s][0]: float(v) * by_source[s][1] for s, v in raw.items()}
    return joints, fist


def _chains(fist: dict[str, float]):
    from robot_control.kinematics import chain_from_urdf

    urdf = URDF.read_text()
    built = []
    for finger in FINGERS:
        names = ARM + [f"r_hj_{finger}_{i}" for i in range(1, 5)]
        chain = chain_from_urdf(urdf, names, f"r_hl_{finger}_tip")
        built.append((finger, chain, np.array([fist[n] for n in names[7:]])))
    return built, chain_from_urdf(urdf, ARM, "r_hl_palm_ee")


def _lowest(built, arm_q: np.ndarray) -> tuple[float, str]:
    """이 팔 자세에서 가장 낮은 링크 원점의 z 와 그 이름.

    메시가 아니라 원점만 본다 — 손가락 살집만큼 낙관적이다. 판정에 여유를 둘 것.
    """
    best, where = math.inf, "?"
    for finger, chain, hand_q in built:
        frames = chain.frames(np.concatenate([arm_q, hand_q]))
        for index, frame in enumerate(frames):
            if frame[2, 3] < best:
                best, where = float(frame[2, 3]), f"{finger}:{chain.joints[index].name}"
        z = float((frames[-1] @ chain.tip)[2, 3])
        if z < best:
            best, where = z, f"{finger}:tip"
    return best, where


def _amplitude(joints: dict, scale: float) -> np.ndarray:
    return np.array([(joints[n]["upper"] - joints[n]["lower"]) * 0.05 * scale
                     for n in ARM])


def _drop(built, neutral: np.ndarray, amp: np.ndarray, samples: int) -> float:
    """여진이 최저점을 추가로 내리는 양(양수 mm). 선분 + 상자 꼭짓점."""
    base, _ = _lowest(built, neutral)
    low = base
    for alpha in np.linspace(-1.0, 1.0, samples):
        low = min(low, _lowest(built, neutral + alpha * amp)[0])
    for corner in range(1 << len(neutral)):
        sign = np.array([1.0 if corner >> k & 1 else -1.0 for k in range(len(neutral))])
        low = min(low, _lowest(built, neutral + sign * amp)[0])
    return (base - low) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--clearance", type=float, default=None,
                        help="정책 홈에서 실측한 손끝-테이블 여유[mm]. 주면 판정까지 한다.")
    parser.add_argument("--margin", type=float, default=30.0,
                        help="링크 원점 근사를 메우는 여유[mm]")
    args = parser.parse_args()

    joints, fist = _load()
    built, palm = _chains(fist)

    home = np.array(POSES["preset 정책홈"])
    safe = np.array(POSES["safe j2+0.40"])
    palm_z = float(palm.pose(safe)[2, 3])
    tip_z, tip_name = _lowest(built, safe)
    below = (palm_z - tip_z) * 1000.0
    print(f"검산 · 최저점 = {_lowest(built, home)[1]}  "
          "(08.31 관찰 \"새끼 손가락이 테이블에 닿고 있어\"와 일치)")
    print(f"      safe 홈에서 최저점은 palm_ee 보다 {below:.0f} mm 아래 ({tip_name})")
    print(f"      ⇒ 실측 palm {SAFE_PALM_ABOVE_TABLE_MM:.0f} mm 는 손끝으로 "
          f"약 {SAFE_PALM_ABOVE_TABLE_MM - below:.0f} mm\n")

    ref, _ = _lowest(built, home)
    print(f"{'자세':16s} {'홈대비':>8s} | " + " ".join(f"{s:>7.2f}" for s in SCALES))
    print(f"{'':16s} {'높이차':>8s} | " + " ".join(f"{'Δz mm':>7s}" for _ in SCALES))
    rows = {}
    for name, q in POSES.items():
        q = np.array(q)
        base, where = _lowest(built, q)
        drops = [_drop(built, q, _amplitude(joints, s), args.samples) for s in SCALES]
        rows[name] = ((base - ref) * 1000.0, drops)
        print(f"{name:16s} {(base-ref)*1000:+7.0f} | "
              + " ".join(f"{d:7.1f}" for d in drops) + f"   ({where})")

    print("\nΔz = 여진이 최저점을 추가로 내리는 양. FK 차분이라 절대 오프셋과 무관하다.")
    print("홈대비 = 정책 홈의 최저점을 0 으로 둔 상대 높이. 클수록 테이블에서 멀다.")

    if args.clearance is None:
        print("\n실측 여유를 알면 `--clearance <mm>` 로 판정까지 한다 "
              "(정책 홈에서 보상 ON 으로 정착시킨 뒤 손끝-테이블 간격).")
        return 0

    print(f"\n═══ 판정 (정책 홈 실측 여유 {args.clearance:.0f} mm · "
          f"안전여유 {args.margin:.0f} mm) ═══")
    for name, (offset, drops) in rows.items():
        available = args.clearance + offset - args.margin
        best = None
        for scale, drop in zip(SCALES, drops):
            if drop <= available:
                best = scale
        marks = " ".join(("✅" if d <= available else "❌") + f"{s:.2f}"
                         for s, d in zip(SCALES, drops))
        verdict = f"최대 scale {best:.2f}" if best else "여진 불가"
        print(f"  {name:16s} 쓸 수 있는 여유 {available:6.0f} mm → {verdict:16s} {marks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
