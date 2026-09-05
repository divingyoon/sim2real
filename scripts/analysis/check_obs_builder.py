#!/usr/bin/env python3
"""배포용 obs 빌더를 **학습 env 의 표본과 대조**한다 (Isaac 불필요).

**왜.** 배포 빌더가 학습 env 와 한 칸이라도 어긋나면 정책은 죽지 않고 **조용히
이상하게 돈다.** 그래서 `probe_obs_layout.py` 가 표본 obs 와 **그 표본을 만든 로봇·
물체 상태**를 함께 남기고, 여기서 같은 상태를 빌더에 넣어 세그먼트별로 비교한다.

09.01 에 이 대조가 실제로 두 가지를 잡았다:
  · 손 20관절이 canonical 이 아니라 **sim DOF 순**  (오차 1.572 → 0.024)
  · 손끝 body 가 알파벳순이 아니라 **손가락 canonical 순** (오차 0.218 → 0.011)
둘 다 정책을 돌려서는 못 잡는다 — 돌기는 도니까.

**허용 오차는 관측 노이즈다.** 학습 env 는 obs 에 DR 노이즈를 얹으므로 완전 일치는
불가능하다. 세그먼트마다 그 노이즈의 크기가 다르니 항목별로 임계를 준다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: 세그먼트별 허용 오차 [단위는 그 항의 단위]. env 의 DR 노이즈 상한에서 온다.
#: 이 값을 넘으면 노이즈가 아니라 **계약 불일치**다.
DEFAULT_TOLERANCE = 0.05
TOLERANCES: dict[str, float] = {
    "arm_qd": 0.15,       # obs_noise_qvel
    "hand_qd": 0.15,
    "joint_err": 0.15,    # 실측 노이즈가 오차에 그대로 실린다
}


@dataclass(frozen=True)
class SegmentDiff:
    name: str
    offset: int
    dim: int
    max_abs: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.max_abs <= self.tolerance


def compare(built: np.ndarray, sample: np.ndarray, segments) -> list[SegmentDiff]:
    """세그먼트별 최대 절대오차. `segments` 는 (이름, 차원) 목록."""
    b = np.asarray(built, dtype=float).reshape(-1)
    s = np.asarray(sample, dtype=float).reshape(-1)
    if b.size != s.size:
        raise ValueError(f"길이가 다르다 — 빌더 {b.size} vs 표본 {s.size}")
    total = sum(d for _, d in segments)
    if total != s.size:
        raise ValueError(f"세그먼트 합 {total} 이 표본 길이 {s.size} 와 다르다")
    out, off = [], 0
    for name, dim in segments:
        out.append(SegmentDiff(
            name=name, offset=off, dim=dim,
            max_abs=float(np.abs(b[off:off + dim] - s[off:off + dim]).max()),
            tolerance=TOLERANCES.get(name, DEFAULT_TOLERANCE)))
        off += dim
    return out


def describe(diffs: list[SegmentDiff]) -> str:
    lines = [f"{'세그먼트':16}{'최대오차':>10}{'허용':>8}"]
    for d in diffs:
        lines.append(f"  {d.name:14}{d.max_abs:>10.4f}{d.tolerance:>8.2f}"
                     f"{'  ✅' if d.ok else '  ❌ 계약 불일치'}")
    bad = [d.name for d in diffs if not d.ok]
    lines.append("\n" + ("✅ 전 세그먼트 일치 (남은 차이는 관측 노이즈)" if not bad
                         else f"❌ {len(bad)}개 불일치: {', '.join(bad)}"))
    return "\n".join(lines)


def load_layout(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    for key in ("sample_obs", "state", "obs_dim"):
        if key not in d:
            raise KeyError(f"{path}: '{key}' 가 없다 — probe_obs_layout.py 로 다시 뽑을 것")
    return d
