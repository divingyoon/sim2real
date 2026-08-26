#!/usr/bin/env python3
"""sim 롤아웃 기록(npz) → 로봇 드라이버 계약으로의 순수 변환 (ROS 무의존).

`probe_fab_shadow_record.py` 가 남긴 npz 는 **sim 의 말**로 적혀 있다 —
canonical 관절명(`l_aj_1..7`), sim 이 지령한 그리퍼 2채널, 정책 스텝 dt.
드라이버가 받는 것은 **source 관절명**(`openarm_left_joint1..`)과 프로필 한계다.
이 모듈이 그 사이를 옮기고, 옮기다 생긴 손실을 **전부 세어서 보고**한다.

세 가지를 조용히 넘기지 않는다:

1. **clamp** — 프로필 한계 밖 목표는 잘리는데, 잘렸다는 사실이 안 보이면
   "정책이 낸 궤적"과 "로봇이 받은 궤적"이 다른 채로 비교하게 된다. 센다.
2. **mimic 축약** — sim 은 `l_hj_gripper_{1,2}` 둘 다 지령한다(USD 가 mimic 을 잃어서).
   실기는 `l_hj_gripper_1` 하나면 된다. 버리는 채널이 남기는 채널과 **일치하는지
   확인하고** 버린다. 안 맞으면 mimic 가정이 깨진 것이므로 거부한다.
3. **속도 초과** — 기록된 요구속도가 프로필 상한을 넘으면 그대로 재생할 수 없다.
   `rate_scale` 로 **시간을 늘려** 해결한다(경로는 그대로). rate-limit 클램프와 달리
   궤적 모양이 안 변하는 쪽이다.

[[jtc-none-interpolation-silent-stall]] — 발행 메시지는 단일포인트 ·
`time_from_start=0` · `header.stamp=0`. 이 규약은 여기서 정하지 않고 그대로 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: npz 에 반드시 있어야 하는 채널. 없으면 이름을 대고 거부한다.
REQUIRED_CHANNELS = (
    "arm_target",
    "grip_cmd",
    "action",
    "palm_cmd_pos",
    "palm_cmd_quat_wxyz",
    "meta_joint_names",
    "meta_grip_names",
    "meta_step_dt",
)

#: mimic 형제 채널이 이만큼 넘게 어긋나면 mimic 가정이 깨진 것으로 본다 [m].
MIMIC_TOLERANCE_M = 1e-4

#: `max_safe_rate_scale` 이 남기는 여유. 상한에 **정확히** 착지시키면 부동소수점에서
#: 한 ULP 위로 올라가 "넘음"이 되고, 드라이버 쪽 판정이 이상/초과 중 무엇인지도 모른다.
#: 안전한 값을 달라는 함수가 경계값을 주면 안 된다.
RATE_SCALE_MARGIN = 0.99


@dataclass(frozen=True)
class GroupChannels:
    """한 컨트롤러 그룹으로 갈 채널 묶음 (source 순서)."""

    source_names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    positions: np.ndarray  # (N, len(source_names))
    clamped: tuple[int, ...]  # 관절별 clamp 발생 스텝 수
    dropped: tuple[str, ...]  # mimic 으로 버린 sim 채널명

    @property
    def clamp_total(self) -> int:
        return int(sum(self.clamped))


@dataclass(frozen=True)
class BagPlan:
    """재생 가능한 계획. 시간은 이미 `rate_scale` 이 반영된 값이다."""

    arm: GroupChannels
    grip: GroupChannels
    action: np.ndarray  # (N, A) 정책 원출력
    palm_pos: np.ndarray  # (N, 3)
    palm_quat_wxyz: np.ndarray  # (N, 4)
    t_ns: np.ndarray  # (N,) 정수 나노초, 0 부터
    publish_dt: float  # 실제 발행 간격 [s] = step_dt / rate_scale
    source_dt: float  # 기록 당시 정책 스텝 [s]
    rate_scale: float
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.arm.positions.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.n_frames * self.publish_dt

    def peak_speed(self) -> np.ndarray:
        """관절별 최대 요구속도 [rad/s] (발행 dt 기준)."""
        if self.n_frames < 2:
            return np.zeros(self.arm.positions.shape[1])
        return np.abs(np.diff(self.arm.positions, axis=0)).max(axis=0) / self.publish_dt


def load_npz(path: str | Path) -> dict:
    """npz 를 읽고 필수 채널이 다 있는지 확인. 없으면 이름을 대고 거부."""
    data = np.load(Path(path), allow_pickle=True)
    missing = [c for c in REQUIRED_CHANNELS if c not in data.files]
    if missing:
        raise KeyError(
            f"{path}: 필수 채널 없음 {missing}. 가진 것: {sorted(data.files)}"
        )
    return {k: data[k] for k in data.files}


def _squeeze_env(a: np.ndarray, env_index: int | None = None) -> np.ndarray:
    """(N, E, D) → (N, D). E>1 이면 **어느 env 인지 명시**해야 한다.

    ★`--num_envs 1` 을 요구하지 않는다. 좌 그리퍼 fabric 은 batch 1 에서 cspace metric 이
      특이해져 첫 스텝에 죽는다(실측: n=1 죽고 n=16 통과). 단일 env 기록이 애초에 불가능한
      구성이 있으므로, 여러 env 를 기록하고 **하나를 골라** 쓴다.
    ⚠ 고르는 것이지 **평균 내는 것이 아니다.** env 마다 컵 위치가 다르고 궤적도 다르다.
      평균 궤적은 어느 env 도 실제로 지나간 적 없는 경로이고, 그걸 로봇에 보내면
      "정책이 낸 궤적"이 아닌 것을 재생하게 된다.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim != 3:
        raise ValueError(f"기대한 차원이 아니다: shape={a.shape}")
    n_env = a.shape[1]
    if env_index is None:
        if n_env != 1:
            raise ValueError(
                f"env 가 {n_env} 개다 — 어느 env 를 재생할지 정해야 한다(env_index). "
                "평균을 내지 않는다: 평균 궤적은 어느 env 도 지나간 적 없는 경로다."
            )
        return a[:, 0, :]
    if not 0 <= env_index < n_env:
        raise IndexError(f"env_index {env_index} 가 기록의 env 수 {n_env} 밖이다")
    return a[:, env_index, :]


def _require_finite(a: np.ndarray, what: str) -> None:
    bad = ~np.isfinite(a)
    if bad.any():
        idx = np.argwhere(bad)[:5].tolist()
        raise ValueError(
            f"{what} 에 유한하지 않은 값 {int(bad.sum())} 개 (예: {idx}). "
            "보간해서 메우지 않는다 — 기록이 깨졌으면 다시 기록하라."
        )


def build_group(
    *,
    values: np.ndarray,
    sim_canonical: list[str],
    group_canonical: list[str],
    profile_joints: dict[str, dict],
    mimic_tol: float = MIMIC_TOLERANCE_M,
    env_index: int | None = None,
) -> GroupChannels:
    """sim 채널 → 한 컨트롤러 그룹의 source 위치 (부호·clamp 적용).

    `group_canonical` 에 없는 sim 채널은 mimic 형제로 보고 버리되, 버리기 전에
    그룹의 첫 관절과 `mimic_tol` 안에서 일치하는지 확인한다.
    """
    values = _squeeze_env(values, env_index)
    _require_finite(values, "관절 지령")
    if values.shape[1] != len(sim_canonical):
        raise ValueError(
            f"채널 수 {values.shape[1]} 와 이름 수 {len(sim_canonical)} 가 다르다"
        )
    idx_of = {n: i for i, n in enumerate(sim_canonical)}

    missing = [c for c in group_canonical if c not in idx_of]
    if missing:
        raise KeyError(
            f"컨트롤러 그룹이 요구하는 관절 {missing} 이 기록에 없다. "
            f"기록이 가진 것: {sim_canonical}"
        )

    keep_idx = [idx_of[c] for c in group_canonical]
    dropped = [n for n in sim_canonical if n not in group_canonical]
    if dropped:
        ref = values[:, keep_idx[0]]
        for name in dropped:
            delta = float(np.abs(values[:, idx_of[name]] - ref).max())
            if delta > mimic_tol:
                raise ValueError(
                    f"{name} 을 mimic 으로 버리려 했는데 {group_canonical[0]} 과 "
                    f"최대 {delta:.6f} 어긋난다(허용 {mimic_tol}). "
                    "실기 URDF 의 mimic 가정이 이 기록엔 성립하지 않는다."
                )

    out = np.empty((values.shape[0], len(group_canonical)), dtype=np.float64)
    sources: list[str] = []
    clamped: list[int] = []
    for col, can in enumerate(group_canonical):
        spec = profile_joints.get(can)
        if spec is None:
            raise KeyError(f"canonical {can!r} 이 robot_control 프로필에 없다")
        raw = values[:, idx_of[can]] * float(spec["sign"])
        lo, hi = float(spec["lower"]), float(spec["upper"])
        out[:, col] = np.clip(raw, lo, hi)
        clamped.append(int(np.count_nonzero((raw < lo) | (raw > hi))))
        sources.append(spec["source"])

    return GroupChannels(
        source_names=tuple(sources),
        canonical_names=tuple(group_canonical),
        positions=out,
        clamped=tuple(clamped),
        dropped=tuple(dropped),
    )


def build_plan(
    npz: dict,
    *,
    profile_joints: dict[str, dict],
    arm_group: list[str],
    grip_group: list[str],
    rate_scale: float = 1.0,
    env_index: int | None = None,
) -> BagPlan:
    """기록 + 프로필 → 재생 계획. `rate_scale` 은 (0, 1] 로 **시간을 늘린다**."""
    if not 0.0 < rate_scale <= 1.0:
        raise ValueError(f"rate_scale 은 (0, 1] — 받은 값 {rate_scale}")

    arm = build_group(
        values=npz["arm_target"],
        sim_canonical=[str(x) for x in npz["meta_joint_names"]],
        group_canonical=arm_group,
        profile_joints=profile_joints,
        env_index=env_index,
    )
    grip = build_group(
        values=npz["grip_cmd"],
        sim_canonical=[str(x) for x in npz["meta_grip_names"]],
        group_canonical=grip_group,
        profile_joints=profile_joints,
        env_index=env_index,
    )
    if arm.positions.shape[0] != grip.positions.shape[0]:
        raise ValueError("팔과 그리퍼 프레임 수가 다르다")

    action = _squeeze_env(npz["action"], env_index)
    palm_pos = _squeeze_env(npz["palm_cmd_pos"], env_index)
    palm_quat = _squeeze_env(npz["palm_cmd_quat_wxyz"], env_index)
    for a, what in ((action, "action"), (palm_pos, "palm_cmd_pos"),
                    (palm_quat, "palm_cmd_quat_wxyz")):
        _require_finite(a, what)

    source_dt = float(np.asarray(npz["meta_step_dt"]).reshape(-1)[0])
    if not source_dt > 0:
        raise ValueError(f"기록의 step_dt 가 {source_dt} 다 — 시간축을 못 만든다")
    publish_dt = source_dt / rate_scale
    n = arm.positions.shape[0]
    t_ns = (np.arange(n, dtype=np.int64) * round(publish_dt * 1e9)).astype(np.int64)

    meta = {k: np.asarray(v).reshape(-1).tolist()
            for k, v in npz.items() if k.startswith("meta_")}
    meta["env_index"] = env_index
    return BagPlan(
        arm=arm, grip=grip, action=action, palm_pos=palm_pos,
        palm_quat_wxyz=palm_quat, t_ns=t_ns, publish_dt=publish_dt,
        source_dt=source_dt, rate_scale=rate_scale, meta=meta,
    )


def velocity_verdict(plan: BagPlan, profile_joints: dict[str, dict]) -> list[dict]:
    """관절별 요구속도 vs 프로필 상한. 상한을 모르면 `limit=None` 으로 그렇게 말한다."""
    peaks = plan.peak_speed()
    rows: list[dict] = []
    for i, can in enumerate(plan.arm.canonical_names):
        limit = profile_joints[can].get("velocity")
        rows.append({
            "canonical": can,
            "source": plan.arm.source_names[i],
            "peak": float(peaks[i]),
            "limit": None if limit is None else float(limit),
            "over": None if limit is None else bool(peaks[i] > limit),
        })
    return rows


def max_safe_rate_scale(plan: BagPlan, profile_joints: dict[str, dict]) -> float | None:
    """현재 계획을 프로필 속도 상한 안으로 넣는 rate_scale 상한. 상한 미상이면 None."""
    worst = 0.0
    for row in velocity_verdict(plan, profile_joints):
        if row["limit"] is None:
            return None
        if row["limit"] > 0:
            worst = max(worst, row["peak"] / row["limit"])
    if worst <= 0:
        return 1.0
    return min(1.0, plan.rate_scale / worst * RATE_SCALE_MARGIN)
