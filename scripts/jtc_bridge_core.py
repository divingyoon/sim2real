#!/usr/bin/env python3
"""isaacsim cmd → JTC 브리지 순수 코어 (numpy/yaml, ROS 무의존).

정책 노드가 발행하는 canonical 관절 명령(`/isaacsim/right_{arm,hand}_cmd`, r_aj_*/r_hj_*
순서)을 robot_control 컨트롤러가 받는 **source 관절**(openarm_right_joint*/rj_dg_*) 순서·
부호·한계로 변환한다. 매핑 진실원천 = robot_control profile `openarm_tesollo.yaml`.

    source_pos[j] = clip( canonical_val[input_idx[j]] * sign[j], lower[j], upper[j] )

ROS 노드(`isaacsim_cmd_to_jtc.py`)는 이 코어로 위치를 만들고 단일포인트 JointTrajectory
(`time_from_start = k·dt > 0`, [[jtc-none-interpolation-silent-stall]])를 발행한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_profile_joints(path: str | Path) -> dict[str, dict]:
    """profile yaml → {canonical: {source, sign, lower, upper, unit}}."""
    data = yaml.safe_load(Path(path).read_text())
    joints = data.get("joints")
    if not joints:
        raise ValueError(f"{path}: 'joints' 섹션 없음")
    out: dict[str, dict] = {}
    for j in joints:
        out[j["canonical"]] = {
            "source": j["source"],
            "sign": float(j.get("sign", 1.0)),
            "lower": float(j["lower"]),
            "upper": float(j["upper"]),
            "unit": j.get("unit", "rad"),
        }
    return out


class JointRemap:
    """canonical 입력 배열 → source 순서 위치(부호·clamp 적용).

    input_canonical: 입력 배열의 canonical 관절명 순서 (정책 노드 발행 순서).
    output_source:   컨트롤러 joints 순서(source 관절명) = JointTrajectory joint_names.
    profile_joints:  load_profile_joints 결과.
    """

    def __init__(
        self,
        input_canonical: list[str],
        output_source: list[str],
        profile_joints: dict[str, dict],
    ) -> None:
        src_to_can = {v["source"]: c for c, v in profile_joints.items()}
        can_idx = {n: i for i, n in enumerate(input_canonical)}

        self.input_len = len(input_canonical)
        self.output_source = list(output_source)
        idx: list[int] = []
        sign: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for src in output_source:
            can = src_to_can.get(src)
            if can is None:
                raise KeyError(f"source 관절 {src!r} 이 profile joints 에 없음")
            if can not in can_idx:
                raise KeyError(f"canonical {can!r}(source {src!r}) 이 입력 순서에 없음")
            p = profile_joints[can]
            idx.append(can_idx[can])
            sign.append(p["sign"])
            lower.append(p["lower"])
            upper.append(p["upper"])
        self.input_idx = np.array(idx, dtype=int)
        self.sign = np.array(sign, dtype=np.float64)
        self.lower = np.array(lower, dtype=np.float64)
        self.upper = np.array(upper, dtype=np.float64)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """canonical 순서 입력 → source 순서 위치 (부호·clamp)."""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.input_len:
            raise ValueError(f"입력 길이 {v.shape[0]} != 기대 {self.input_len}")
        out = v[self.input_idx] * self.sign
        return np.clip(out, self.lower, self.upper)


def time_from_start_sec(control_dt: float, horizon_steps: float) -> float:
    """단일포인트 목표 도달 시각. >0 이어야 JTC 가 움직임(0이면 무동작).

    control_dt: 제어 주기[s]. horizon_steps: 몇 주기 뒤 목표로 둘지(>0).
    """
    if control_dt <= 0.0 or horizon_steps <= 0.0:
        raise ValueError("control_dt 와 horizon_steps 는 양수여야 함(0이면 JTC 무동작)")
    return float(control_dt) * float(horizon_steps)


def safe_time_from_start(
    current: np.ndarray,
    target: np.ndarray,
    max_vel: float,
    min_tfs: float,
) -> float:
    """속도 한계를 넘지 않는 목표 도달 시각.

    tfs = max(min_tfs, max_j |target_j - current_j| / max_vel).
    큰 점프(pregrasp 접근 등)는 tfs↑ 로 자동 감속(속도 ≤ max_vel), 작은 정책 델타는
    min_tfs 그대로 빠름. 어떤 명령이든 관절 속도가 max_vel 을 넘지 않아 모터를 못 퍼뜨림.
    """
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1)
    if cur.shape != tgt.shape:
        raise ValueError(f"current{cur.shape}/target{tgt.shape} 길이 불일치")
    if max_vel <= 0.0 or min_tfs <= 0.0:
        raise ValueError("max_vel, min_tfs 는 양수여야 함")
    max_disp = float(np.max(np.abs(tgt - cur))) if cur.size else 0.0
    return max(float(min_tfs), max_disp / float(max_vel))
