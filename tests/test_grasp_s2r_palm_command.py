#!/usr/bin/env python3
"""`grasp_s2r_palm_command` 계약 테스트.

env `_pre_physics_step` 의 팔 구간과 1:1 인지 본다. 특히 09.03 좌팔에서 사고가 났던
지점들을 여기서 못 박는다 — 앵커 규약·첫 지령 리미터 예외·노름 기준 스케일링.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grasp_s2r_palm_command import (  # noqa: E402
    HOME_PALM,
    PalmCmdCfg,
    PalmCommand,
    cfg_from_run,
)

RUN = Path(__file__).resolve().parents[1] / "logs/policy/right_g1/params/env.yaml"


def test_action_zero_maps_to_anchor_not_box_center():
    """★`a=0` 은 **앵커**다 — 좌팔(박스 중심)과 규약이 다르다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home"))
    out = pc.step(np.zeros(6))
    assert np.allclose(out, np.asarray(HOME_PALM), atol=1e-12)


def test_spawn_anchor_is_snapshot_not_live_object():
    """`spawn` 앵커는 리셋에서 한 번 찍고 그 뒤로 안 움직인다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="spawn",
                                anchor_offset_xyz=(-0.066, -0.022, 0.085)))
    pc.reset(object_spawn_pos=(0.40, -0.20, 0.30))
    first = pc.anchor()
    assert np.allclose(first[:3], [0.40 - 0.066, -0.20 - 0.022, 0.30 + 0.085])
    # 회전 성분은 홈 그대로여야 한다(위치만 재중심).
    assert np.allclose(first[3:], np.asarray(HOME_PALM)[3:])
    # 물체가 밀려도 앵커는 그대로 — 스냅샷이므로 여기서 갱신할 방법이 없다.
    assert np.allclose(pc.anchor(), first)


def test_spawn_unset_falls_back_to_home():
    """첫 리셋 전에는 스폰이 없다 — 조용한 0 앵커 대신 홈으로 대체한다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="spawn"))
    assert np.allclose(pc.anchor(), np.asarray(HOME_PALM))


def test_delta_box_maps_pm1_to_pm_delta():
    """a=±1 → 앵커 ± 델타 박스(위치 0.1 m · 회전 20°)."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home", rate_limit_m=0.0,
                                rate_limit_rot_deg=0.0))
    hi = pc.step(np.ones(6))
    pc.reset()
    lo = pc.step(-np.ones(6))
    home = np.asarray(HOME_PALM)
    # ★상한 쪽은 박스 안이지만, **하한 x 는 박스에 걸린다**(홈 0.28 − 0.1 = 0.18 <
    #   palm_box_min 0.20). 그게 정상 동작이므로 그대로 확인한다.
    assert np.allclose(hi[:3], home[:3] + 0.1)
    assert lo[0] == pytest.approx(0.20, abs=1e-12)          # 박스 하한에 클램프
    assert np.allclose(lo[1:3], home[1:3] - 0.1)
    assert np.allclose(hi[3:], home[3:] + math.radians(20.0))


def test_box_clamp_records_saturation():
    """박스 밖으로 나가려 하면 잘리고, 그 축이 포화로 기록된다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="spawn", rate_limit_m=0.0,
                                anchor_offset_xyz=(0.0, 0.0, 0.0)))
    pc.reset(object_spawn_pos=(0.54, 0.20, 0.68))   # 박스 상한 코너 근처
    out = pc.step(np.array([1.0, 1.0, 1.0, 0, 0, 0]))
    assert out[0] <= 0.55 + 1e-12 and out[1] <= 0.22 + 1e-12
    assert pc.state.box_sat[:2].sum() == 2.0


def test_first_command_is_not_rate_limited():
    """★첫 지령은 '변화'가 아니라 초기화다 — 리미터를 걸면 안 된다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home", rate_limit_m=0.001))
    out = pc.step(np.array([1.0, 0, 0, 0, 0, 0]))
    # 0.1 m 를 요구했고 상한은 1 mm 지만, 첫 지령이라 그대로 나가야 한다.
    assert out[0] == pytest.approx(HOME_PALM[0] + 0.1, abs=1e-12)


def test_rate_limit_scales_by_norm_and_preserves_direction():
    """★성분별 클램프가 아니라 노름 스케일링 — 방향이 보존돼야 한다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home", rate_limit_m=0.02,
                                rate_limit_rot_deg=0.0))
    pc.step(np.zeros(6))                      # primed
    out = pc.step(np.array([1.0, 1.0, 0.0, 0, 0, 0]))
    step = out[:3] - np.asarray(HOME_PALM)[:3]
    assert np.linalg.norm(step) == pytest.approx(0.02, abs=1e-9)
    # 방향: x 와 y 요구가 같았으니 결과도 같아야 한다(성분 클램프면 둘 다 0.02).
    assert step[0] == pytest.approx(step[1], abs=1e-12)


def test_rotation_rate_limit_is_separate():
    """회전 리미터는 위치와 **따로** 걸린다(2.9°)."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home", rate_limit_m=0.0,
                                rate_limit_rot_deg=2.9))
    pc.step(np.zeros(6))
    out = pc.step(np.array([0, 0, 0, 1.0, 0.0, 0.0]))
    d = out[3:] - np.asarray(HOME_PALM)[3:]
    assert np.linalg.norm(d) == pytest.approx(math.radians(2.9), abs=1e-9)


def test_action_is_clipped_to_pm1():
    """env 는 `actions.clamp(-1,1)` 를 먼저 한다 — 그 밖의 값은 포화로 취급."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home", rate_limit_m=0.0))
    a = pc.step(np.full(6, 5.0))
    pc.reset()
    b = pc.step(np.ones(6))
    assert np.allclose(a, b)


def test_reset_clears_primed_and_prev():
    """에피소드 사이에 상태가 새면 시작이 오염된다."""
    pc = PalmCommand(PalmCmdCfg(anchor_mode="home", rate_limit_m=0.001))
    pc.step(np.ones(6))
    pc.step(np.ones(6))
    pc.reset()
    assert pc.state.primed is False
    out = pc.step(-np.ones(6))
    # 홈 0.28 − 0.1 = 0.18 은 박스 하한 0.20 에 걸린다 — 리미터가 아니라 박스다.
    assert out[0] == pytest.approx(0.20, abs=1e-12)


@pytest.mark.skipif(not RUN.exists(), reason="g1 런 dump 없음")
def test_cfg_from_run_reads_g1_contract():
    """★상수는 런에서 온다 — 손으로 옮기면 09.03 좌팔 사고가 반복된다."""
    c = cfg_from_run(RUN)
    assert c.anchor_mode == "spawn"
    assert c.delta_xyz == (0.1, 0.1, 0.1)
    assert c.delta_rot_deg == pytest.approx(20.0)
    assert c.anchor_offset_xyz == pytest.approx((-0.066, -0.022, 0.085))
    assert c.rate_limit_m == pytest.approx(0.02)
    assert c.rate_limit_rot_deg == pytest.approx(2.9)
