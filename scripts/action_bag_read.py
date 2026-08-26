#!/usr/bin/env python3
"""rosbag2 → 기록 dict 되읽기 (`action_bag.py` 의 역방향).

백을 구울 때 canonical→source 로 옮겼으니, 되읽으면서 source→canonical 로 되돌린다.
그래야 `shadow_replay.py` 같은 canonical 소비자가 **npz 든 백이든 똑같이** 먹는다.
반환 키는 npz 와 같은 이름을 쓴다 — 소비자가 출처를 몰라도 되게.

되돌릴 수 있는 것과 없는 것을 구분한다:
  · 부호는 ±1 이라 되돌아온다.
  · **clamp 는 되돌아오지 않는다.** 잘린 값은 잘린 채로 온다. 그게 맞다 —
    백은 "정책이 원한 것"이 아니라 "로봇이 받는 것"의 기록이기 때문이다.
    얼마나 잘렸는지는 `/shadow/meta` 의 `clamped` 에 남아 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_bag(path: str | Path, *, profile) -> dict:
    """백 → npz 호환 dict. `profile` 은 source↔canonical 진실원천."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory

    path = Path(path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    arm_topic = profile.topics["arm_traj"]
    grip_topic = profile.topics["ee_traj"]
    available = {t.name for t in reader.get_all_topics_and_types()}
    missing = {arm_topic, grip_topic} - available
    if missing:
        raise KeyError(
            f"{path}: 드라이버 토픽 {sorted(missing)} 이 백에 없다. "
            f"가진 것: {sorted(available)}"
        )

    arm_rows: list[list[float]] = []
    grip_rows: list[list[float]] = []
    arm_names: list[str] = []
    grip_names: list[str] = []
    meta: dict = {}
    while reader.has_next():
        topic, raw, _t = reader.read_next()
        if topic == arm_topic:
            m = deserialize_message(raw, JointTrajectory)
            arm_names = list(m.joint_names)
            arm_rows.append(list(m.points[0].positions))
        elif topic == grip_topic:
            m = deserialize_message(raw, JointTrajectory)
            grip_names = list(m.joint_names)
            grip_rows.append(list(m.points[0].positions))
        elif topic == "/shadow/meta":
            meta = json.loads(deserialize_message(raw, String).data)

    if not arm_rows:
        raise ValueError(f"{path}: {arm_topic} 에 메시지가 없다")
    if len(arm_rows) != len(grip_rows):
        raise ValueError(
            f"{path}: 팔 {len(arm_rows)} · 그리퍼 {len(grip_rows)} 프레임 수가 다르다"
        )

    src_to_can = {v["source"]: c for c, v in profile.joint_limits.items()}

    def _to_canonical(rows: list[list[float]], sources: list[str]):
        arr = np.asarray(rows, dtype=np.float32)
        canon, out = [], np.empty_like(arr)
        for col, src in enumerate(sources):
            can = src_to_can.get(src)
            if can is None:
                raise KeyError(f"백의 source 관절 {src!r} 이 프로필에 없다")
            canon.append(can)
            out[:, col] = arr[:, col] / float(profile.joint_limits[can]["sign"])
        return canon, out

    arm_canon, arm = _to_canonical(arm_rows, arm_names)
    grip_canon, grip = _to_canonical(grip_rows, grip_names)

    dt = meta.get("publish_dt")
    if dt is None:
        raise KeyError(
            f"{path}: /shadow/meta 에 publish_dt 가 없다 — 시간축을 지어내지 않는다."
        )

    return {
        "arm_target": arm[:, None, :],
        "grip_cmd": grip[:, None, :],
        "meta_joint_names": np.array(arm_canon),
        "meta_grip_names": np.array(grip_canon),
        "meta_step_dt": np.array([float(dt)]),
        "meta_bag_meta": meta,
    }
