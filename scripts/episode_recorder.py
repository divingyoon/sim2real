#!/usr/bin/env python3
"""에피소드 per-step CSV 레코더 (grasp_inference 전용).

한 에피소드 = 한 파일. 컬럼(고정 순서):
    t_sec, step, is_lift,
    action_0..10,                     # policy 원출력 11D
    arm_pos_0..6, arm_vel_0..6, arm_eff_0..6,      # canonical r_aj_*
    hand_pos_0..19, hand_eff_0..19,               # canonical r_hj_* (finger-major)
    tip_force_0..4, contact_0..4,                 # 원시 힘[N] / 이진(게이트 후)
    cup_x, cup_y, cup_z, palm_x, palm_y, palm_z, dist,
    arm_cmd_0..6, hand_cmd_0..19                  # 전송 명령
"""

from __future__ import annotations

import csv
import time
from pathlib import Path


def _cols(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]

HEADER = (
    ["t_sec", "step", "is_lift"]
    + _cols("action", 11)
    + _cols("arm_pos", 7) + _cols("arm_vel", 7) + _cols("arm_eff", 7)
    + _cols("hand_pos", 20) + _cols("hand_eff", 20)
    + _cols("tip_force", 5) + _cols("contact", 5)
    + ["cup_x", "cup_y", "cup_z", "palm_x", "palm_y", "palm_z", "dist"]
    + _cols("arm_cmd", 7) + _cols("hand_cmd", 20)
)

FLUSH_EVERY = 60   # 1초(60Hz)마다 디스크 반영


class EpisodeCsvRecorder:
    """에피소드 단위 CSV 기록. start() → record()×N → close()."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self._fh = None
        self._writer = None
        self._t0 = 0.0
        self._rows = 0
        self.path: Path | None = None

    def start(self) -> Path:
        self.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"grasp_ep_{stamp}.csv"
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(HEADER)
        self._t0 = time.monotonic()
        self._rows = 0
        return self.path

    def record(
        self,
        step: int,
        is_lift: bool,
        action,        # (11,)
        arm_pos, arm_vel, arm_eff,      # (7,)×3
        hand_pos, hand_eff,             # (20,)×2
        tip_force, contact,             # (5,)×2
        cup, palm,                      # (3,)×2
        dist: float,
        arm_cmd, hand_cmd,              # (7,), (20,)
    ) -> None:
        if self._writer is None or self._fh is None:
            return
        row = (
            [round(time.monotonic() - self._t0, 4), step, int(is_lift)]
            + [float(v) for v in action]
            + [float(v) for v in arm_pos] + [float(v) for v in arm_vel]
            + [float(v) for v in arm_eff]
            + [float(v) for v in hand_pos] + [float(v) for v in hand_eff]
            + [float(v) for v in tip_force] + [float(v) for v in contact]
            + [float(v) for v in cup] + [float(v) for v in palm] + [float(dist)]
            + [float(v) for v in arm_cmd] + [float(v) for v in hand_cmd]
        )
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % FLUSH_EVERY == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None
