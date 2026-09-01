#!/usr/bin/env python3
"""pour 궤적(sim 추출) → `shadow_replay` 재생 npz. 순수 변환 + 실기 가능성 판정.

**왜 필요한가.** 오늘의 목표는 정책 추론이 아니라 **행동 모사**다 — 학습한 pour 정책을
sim 에서 한 번 굴려 궤적을 뽑고, 실기는 그 관절 궤적을 그대로 재생한다. 정책을 실기에
올리는 것보다 훨씬 적은 것이 걸려 있고, 되는 것 하나를 먼저 확보할 수 있다.

**두 가지가 이 변환의 전부다.**

  ① **열 고르기.** sim 은 38관절(머리·양팔·손)을 한 배열에 담는데 드라이버 계약은
     팔 7 + 손 20 이고 **순서가 다르다**(sim 은 좌우가 번갈아 섞여 있다). 이름으로
     골라야지 위치로 자르면 조용히 어긋난다.
  ② **속도 판정.** sim 은 실기보다 빠르게 움직일 수 있다. 어느 관절이 얼마나 넘치는지
     세고, 궤적 모양을 지키려면 **전체를 얼마로 늦춰야 하는지**를 낸다.

★**앞 몇 프레임은 버린다.** pour 는 warm 상태로 **텔레포트**해 시작하므로 첫 프레임의
  `q_target` 이 텔레포트 이전 값이다 — 09.01 d3 실측에서 프레임 0 이 94.2 rad/s 로
  나왔다(실제 궤적은 2.6). 버린 프레임 수는 항상 보고한다(조용히 자르지 않는다).

★**기본 소스는 `meas` 다.** 좌팔 라이브 실측에서 sim **지령**을 충실히 따르면 오히려
  테이블을 긁었고, sim **실측**을 목표로 주었을 때 맞았다
  ([[live-shadow-left-first-real]]). 같은 이유로 여기서도 실측이 기본이다.
  `--source target` 로 바꿀 수 있다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

#: 재생 목표로 쓸 수 있는 두 시계열.
REPLAY_SOURCES = ("target", "meas")

#: sim 에서 뽑은 궤적의 어느 열이 어느 시계열인가.
_SOURCE_KEY = {"target": "q_target", "meas": "q_meas"}

#: 텔레포트 잔재를 버릴 기본 프레임 수. 실측 근거는 모듈 docstring 참조.
DEFAULT_SKIP_FRAMES = 3


def column_index(
    sim_names: Sequence[str],
    wanted: Sequence[str],
    *,
    what: str,
    alias: Mapping[str, str] | None = None,
) -> list[int]:
    """`wanted` 순서 그대로 sim 배열의 열 번호를 낸다.

    `alias` 는 sim 이름 → 계약 이름. 자산에 따라 손 관절이 `rj_dg_*` 로 나오는 판이
    있어서 통로를 열어 둔다(09.01 d3 자산은 이름이 그대로라 필요 없었다).
    """
    if len(set(wanted)) != len(wanted):
        dup = sorted({n for n in wanted if list(wanted).count(n) > 1})
        raise ValueError(f"{what} 관절 요청에 중복이 있다: {dup}")
    canonical_of = dict(alias or {})
    lookup: dict[str, int] = {}
    for i, name in enumerate(sim_names):
        lookup.setdefault(canonical_of.get(name, name), i)
    missing = [n for n in wanted if n not in lookup]
    if missing:
        raise KeyError(f"{what} 관절이 기록에 없다: {missing}")
    return [lookup[n] for n in wanted]


def peak_velocity(target: np.ndarray, *, dt: float) -> np.ndarray:
    """관절별 |Δq|/dt 의 최대. 프레임이 하나뿐이면 0."""
    arr = np.asarray(target, dtype=float)
    if arr.shape[0] < 2:
        return np.zeros(arr.shape[1])
    return (np.abs(np.diff(arr, axis=0)) / dt).max(axis=0)


def required_rate_scale(peaks: Sequence[float], limits: Sequence[float | None]) -> float:
    """궤적 모양을 지키면서 전 관절을 한계 안에 넣는 배속.

    한 관절이라도 넘으면 **전체**를 늦춘다 — 관절마다 다르게 늦추면 궤적이 뒤틀린다.
    """
    ratios = [p / lim for p, lim in zip(peaks, limits) if lim]
    worst = max(ratios, default=0.0)
    return 1.0 if worst <= 1.0 else 1.0 / worst


def describe_feasibility(
    names: Sequence[str],
    peaks: Sequence[float],
    limits: Sequence[float | None],
    *,
    dt: float,
) -> str:
    lines = [f"관절별 peak |Δq|/dt  (기록 {dt*1000:.2f} ms 주기)"]
    over = []
    for name, peak, lim in zip(names, peaks, limits):
        ratio = f"{peak / lim:.2f}배" if lim else "한계 미상"
        mark = "  ★초과" if lim and peak > lim else ""
        lines.append(f"  {name:10} {peak:7.3f} rad/s   한계 {lim}   {ratio}{mark}")
        if lim and peak > lim:
            over.append(name)
    scale = required_rate_scale(peaks, limits)
    if over:
        lines.append(f"\n초과 {len(over)}개: {', '.join(over)}")
        lines.append(f"→ rate_scale {scale:.3f} 이하로 재생하면 전부 한계 안에 든다")
    else:
        lines.append("\n전 관절 한계 안 — 등속(rate_scale 1.0) 재생 가능")
    return "\n".join(lines)


def to_replay(
    traj: Mapping[str, np.ndarray],
    arm_names: Sequence[str],
    hand_names: Sequence[str],
    *,
    alias: Mapping[str, str] | None = None,
    source: str = "meas",
    skip_frames: int = 0,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """`shadow_replay.build_plan` 이 그대로 읽는 dict 를 만든다."""
    if source not in REPLAY_SOURCES:
        raise ValueError(f"모르는 재생 소스: {source} (가능한 값 {REPLAY_SOURCES})")
    q = np.asarray(traj[_SOURCE_KEY[source]], dtype=np.float32)[skip_frames:]
    q_meas = np.asarray(traj["q_meas"], dtype=np.float32)[skip_frames:]
    sim_names = [str(x) for x in traj["meta_joint_names"]]
    arm_i = column_index(sim_names, arm_names, what="팔", alias=alias)
    hand_i = column_index(sim_names, hand_names, what="손", alias=alias)

    out: dict[str, np.ndarray] = {
        "arm_target": q[:, arm_i][:, None, :],
        "grip_cmd": q[:, hand_i][:, None, :],
        "q_meas": q_meas[:, arm_i],
        "meta_joint_names": np.array(list(arm_names)),
        "meta_grip_names": np.array(list(hand_names)),
        "meta_step_dt": np.array([float(traj["meta_step_dt"])], dtype=np.float32),
        "meta_replay_source": np.array(source),
        "meta_skipped_frames": np.array([skip_frames], dtype=np.int32),
    }
    for key, value in (provenance or {}).items():
        out[key] = np.array(value)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────
def _main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from robot_profile import load_robot_profile

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--traj", type=Path, required=True, help="추출기가 낸 pour_traj_*.npz")
    p.add_argument("--profile", default="tesollo_sensor__right")
    p.add_argument("--source", choices=REPLAY_SOURCES, default="meas")
    p.add_argument("--skip-frames", type=int, default=DEFAULT_SKIP_FRAMES,
                   help="앞에서 버릴 프레임 수 (텔레포트 잔재)")
    p.add_argument("--out", type=Path, help="쓰지 않으면 판정만 하고 끝낸다")
    args = p.parse_args(argv)

    traj = dict(np.load(args.traj, allow_pickle=False))
    profile = load_robot_profile(args.profile)
    arm, hand = list(profile.arm_canonical), list(profile.ee_canonical)
    dt = float(traj["meta_step_dt"])

    out = to_replay(traj, arm, hand, source=args.source, skip_frames=args.skip_frames,
                    provenance={k: str(traj[k]) for k in traj if k.startswith("meta_")
                                and k not in ("meta_joint_names", "meta_step_dt")})
    n = out["arm_target"].shape[0]
    peaks = peak_velocity(out["arm_target"][:, 0], dt=dt)
    limits = [profile.joint_limits[j].get("velocity") for j in arm]
    scale = required_rate_scale(peaks, limits)

    print(f"\n입력 {args.traj.name} · 소스 {args.source} · 앞 {args.skip_frames}프레임 버림")
    print(f"프레임 {n} · {n*dt:.2f} s @ {1/dt:.0f} Hz\n")
    print(describe_feasibility(arm, peaks, limits, dt=dt))
    if scale < 1.0:
        print(f"\n등속 {n*dt:.2f} s → rate_scale {scale:.3f} 에서 {n*dt/scale:.2f} s")
        print("★붓기는 동역학 과제다 — 늦추면 결과가 달라질 수 있다. sim 에서 먼저 볼 것.")

    if args.out is None:
        print("\n(--out 없음 — 판정만 했다. 아무것도 쓰지 않았다.)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\n저장 {args.out}")
    print(f"재생:  python3 scripts/shadow_replay.py --sim {args.out} "
          f"--robot {args.profile} --rate-scale {min(scale, 1.0):.3f} --with-hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
