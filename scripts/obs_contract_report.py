#!/usr/bin/env python3
"""hdgp 태스크들의 관측 계약을 자동 추출해 표로 낸다.

hdgp 에는 관측을 정의하는 태스크가 20개 넘게 있다. 어느 것을 배포 대상으로 삼든,
그 태스크의 obs 계약을 손으로 옮겨 적는 순간 드리프트가 시작된다. 이 도구는 소스에서
직접 뽑아 **베낄 필요를 없앤다**.

사용:
    python3 obs_contract_report.py                      # 전 태스크 요약
    python3 obs_contract_report.py --task tesollo/right/grasp_sensor   # 상세
    python3 obs_contract_report.py --diff tesollo_sensor__right        # 배포와 대조

한계는 obs_contract 모듈 docstring 참조 — 차원과 프레임/단위 의미는 정적으로 안 나온다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from obs_contract import ObsContractError, diff_segments, extract_obs_contract  # noqa: E402
from robot_profile import HDGP_OPENARM_SRC  # noqa: E402


def _task_envs(root: Path):
    for f in sorted(root.rglob("*_env.py")):
        if "/tests/" in str(f):
            continue
        if "_get_observations" in f.read_text():
            yield f


def _rel(f: Path, root: Path) -> str:
    return str(f.relative_to(root).parent)


def cmd_summary(root: Path) -> int:
    print(f"{'태스크':52}{'세그':>5}{'노이즈':>7}{'표현식':>7}  정규화 상수")
    print("-" * 104)
    ok = fail = 0
    for f in _task_envs(root):
        try:
            c = extract_obs_contract(f)
        except ObsContractError as e:
            print(f"{_rel(f, root):52}{'—':>5}{'—':>7}{'—':>7}  추출 실패: {e}")
            fail += 1
            continue
        ok += 1
        consts = sorted({k for s in c.segments for k in s.constants})
        print(
            f"{_rel(f, root):52}{len(c.segments):5}"
            f"{sum(s.dr_noise for s in c.segments):7}"
            f"{sum(not s.is_named for s in c.segments):7}  "
            + (", ".join(consts[:4]) + (" …" if len(consts) > 4 else ""))
        )
    print(f"\n추출 성공 {ok} / 실패 {fail}")
    return 0 if fail == 0 else 1


def cmd_detail(root: Path, task: str) -> int:
    cands = [f for f in _task_envs(root) if task in str(f)]
    if not cands:
        print(f"태스크를 찾지 못했다: {task}")
        return 1
    for f in cands:
        c = extract_obs_contract(f)
        print(f"\n=== {_rel(f, root)}   (obs 변수 `{c.var}`, {len(c.segments)} 세그먼트)")
        print(f"{'#':>3} {'세그먼트':26}{'노이즈':>7}  정의식")
        print("-" * 110)
        for s in c.segments:
            print(f"{s.index:3} {s.name:26}{'DR' if s.dr_noise else '':>7}  {s.expr[:64]}")
            if s.constants:
                print(f"{'':37}상수: {', '.join(s.constants)}")
    return 0


def cmd_diff(root: Path, profile_name: str) -> int:
    from grasp_obs_builder import OBS_SEGMENTS
    from robot_profile import hdgp_task_dir, load_robot_profile

    prof = load_robot_profile(profile_name)
    env = hdgp_task_dir(prof) / f"grasp_{prof.acting_side}_env.py"
    c = extract_obs_contract(env)
    dep = [n for n, _ in OBS_SEGMENTS]
    print(f"sim  ({_rel(env, root)}): {len(c.segments)} 세그먼트")
    print(f"배포 (grasp_obs_builder):   {len(dep)} 세그먼트")
    diffs = diff_segments(c.names, dep)
    if not diffs:
        print("\n세그먼트 이름·순서 일치 ✅")
        noised = [s.name for s in c.segments if s.dr_noise]
        if noised:
            print("  ※ sim 에서 DR 노이즈가 더해지는 세그먼트: " + ", ".join(noised))
            print("     배포는 노이즈를 더하지 않는다 — 실센서가 곧 노이즈다.")
        return 0
    print("\n불일치:")
    for d in diffs:
        print("  ⚠ " + d)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", help="부분 경로로 태스크 지정 (상세 출력)")
    ap.add_argument("--diff", help="구성 프로필 이름과 배포 obs 빌더를 대조")
    args = ap.parse_args()
    root = HDGP_OPENARM_SRC / "openarm"
    if not root.exists():
        print(f"hdgp 소스 없음: {root}")
        return 1
    if args.diff:
        return cmd_diff(root, args.diff)
    if args.task:
        return cmd_detail(root, args.task)
    return cmd_summary(root)


if __name__ == "__main__":
    raise SystemExit(main())
