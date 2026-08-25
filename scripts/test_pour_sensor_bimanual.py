#!/usr/bin/env python3
"""pour_sensor 양팔 어댑터 테스트 + hdgp cfg drift-guard.

두 종류를 검증한다:
  1. 어댑터 로직 (action 분해, TCP 누적/클램프/frozen, receiver 컵 FK)
  2. **drift-guard** — 상수와 "오른팔 경로 동일" 전제가 hdgp 소스와 계속 일치하는지.
     전제가 깨지면(예: pour_sensor가 palm_delta를 바꾸면) 배포 정책이 조용히
     어긋나므로, 여기서 실패시켜 드러낸다.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from pour_sensor_bimanual import (
    ACTION_DIM,
    LEFT_CUP_FOLLOW_LOCAL_Z,
    LEFT_TCP_ACTION_DELTA_M,
    LEFT_TCP_WORKSPACE_RANGE,
    LEFT_TCP_Z_DOWN_M,
    LeftTcpController,
    left_cup_follow_offset,
    receiver_cup_pose,
    split_bimanual_action,
)

_HDGP_CFG = Path.home() / (
    "rl_ws/hdgp/source/openarm/openarm/tesollo/both/pour_sensor/pour_right_env_cfg.py"
)
_POUR_V1_CFG = Path.home() / (
    "rl_ws/hdgp/source/openarm/openarm/tesollo/right/pour_v1/pour_right_env_cfg.py"
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _cfg_value(src: str, key: str) -> str:
    m = re.search(rf"^\s+{re.escape(key)}:\s*[^=]+=\s*([^#\n]+)", src, re.M)
    assert m, f"cfg에서 {key}를 찾지 못했다 — 이름이 바뀌었을 수 있다"
    return m.group(1).strip()


@pytest.fixture(scope="module")
def hdgp_src() -> str:
    if not _HDGP_CFG.exists():
        pytest.skip(f"hdgp cfg 없음: {_HDGP_CFG}")
    return _HDGP_CFG.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. 어댑터 로직
# --------------------------------------------------------------------------
def test_split_bimanual_action():
    a = np.arange(ACTION_DIM, dtype=float) / ACTION_DIM
    right, left = split_bimanual_action(a)
    assert right.shape == (12,)
    assert left.shape == (3,)
    np.testing.assert_allclose(right, a[:12])
    np.testing.assert_allclose(left, np.clip(a[12:], -1.0, 1.0))


def test_split_rejects_wrong_dim():
    with pytest.raises(ValueError, match="15D"):
        split_bimanual_action(np.zeros(12))


def test_left_action_is_clipped():
    a = np.zeros(ACTION_DIM)
    a[12:] = [5.0, -5.0, 0.5]
    _, left = split_bimanual_action(a)
    np.testing.assert_allclose(left, [1.0, -1.0, 0.5])


def test_tcp_accumulates_at_delta_rate():
    rest = np.array([0.1, 0.2, 0.3])
    c = LeftTcpController(rest_pos_b=rest)
    t = c.step(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(t, rest + [LEFT_TCP_ACTION_DELTA_M, 0, 0])
    t = c.step(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(t, rest + [2 * LEFT_TCP_ACTION_DELTA_M, 0, 0])


def test_tcp_clamped_to_workspace_box():
    rest = np.zeros(3)
    c = LeftTcpController(rest_pos_b=rest)
    for _ in range(200):                      # 200 × 1cm = 2m ≫ 8cm 박스
        c.step(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(c.target_pos_b, np.asarray(LEFT_TCP_WORKSPACE_RANGE))


def test_tcp_z_cannot_go_below_rest():
    """★ s2r 안전 계약 — receiver 컵이 테이블을 뚫지 못하게 z 하강을 rest에서 캡."""
    rest = np.array([0.0, 0.0, 0.32])
    c = LeftTcpController(rest_pos_b=rest)
    for _ in range(200):
        c.step(np.array([0.0, 0.0, -1.0]))
    assert c.target_pos_b[2] == pytest.approx(rest[2] - LEFT_TCP_Z_DOWN_M)
    assert c.target_pos_b[2] >= rest[2] - 1e-12


def test_frozen_mode_ignores_action():
    """M0 축소 배포 — 어떤 action이 와도 rest 유지."""
    rest = np.array([0.1, -0.2, 0.32])
    c = LeftTcpController(rest_pos_b=rest, mode="frozen")
    for _ in range(50):
        np.testing.assert_allclose(c.step(np.random.uniform(-1, 1, 3)), rest)


def test_hold_steps_pins_to_rest_then_moves():
    rest = np.zeros(3)
    c = LeftTcpController(rest_pos_b=rest, hold_steps=3)
    for _ in range(3):
        np.testing.assert_allclose(c.step(np.array([1.0, 0, 0])), rest)
    moved = c.step(np.array([1.0, 0, 0]))
    assert moved[0] == pytest.approx(LEFT_TCP_ACTION_DELTA_M)


def test_delay_shifts_action_by_n_steps():
    rest = np.zeros(3)
    c = LeftTcpController(rest_pos_b=rest, delay_steps=2)
    c.step(np.array([1.0, 0, 0]))            # 지연 버퍼로 들어감 → 적용은 0
    assert c.target_pos_b[0] == pytest.approx(0.0)
    c.step(np.zeros(3))
    assert c.target_pos_b[0] == pytest.approx(0.0)
    c.step(np.zeros(3))                      # 이제 첫 action이 나옴
    assert c.target_pos_b[0] == pytest.approx(LEFT_TCP_ACTION_DELTA_M)


def test_reset_restores_rest():
    rest = np.array([0.1, 0.1, 0.3])
    c = LeftTcpController(rest_pos_b=rest)
    c.step(np.array([1.0, 1.0, 1.0]))
    c.reset()
    np.testing.assert_allclose(c.target_pos_b, rest)


def test_receiver_cup_pose_identity_hand():
    """단위 자세의 왼손이면 컵은 손 위 local_z 만큼."""
    pos, quat = receiver_cup_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(pos, [0.0, 0.0, LEFT_CUP_FOLLOW_LOCAL_Z], atol=1e-12)
    fp, fq = left_cup_follow_offset()
    np.testing.assert_allclose(quat, fq, atol=1e-12)


def test_follow_quat_is_ry_90():
    _, q = left_cup_follow_offset()
    # R_y(+90°) → w=cos45°, y=sin45°
    assert q[0] == pytest.approx(math.cos(math.pi / 4))
    assert q[2] == pytest.approx(math.sin(math.pi / 4))
    assert q[1] == pytest.approx(0.0) and q[3] == pytest.approx(0.0)


def test_receiver_cup_moves_with_hand():
    p0, _ = receiver_cup_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    p1, _ = receiver_cup_pose(np.array([0.05, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(p1 - p0, [0.05, 0.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------
# 2. drift-guard — hdgp cfg와 상수 일치
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key, expected",
    [
        ("left_tcp_action_delta_m", LEFT_TCP_ACTION_DELTA_M),
        ("left_tcp_z_down_m", LEFT_TCP_Z_DOWN_M),
        ("left_cup_follow_local_z", LEFT_CUP_FOLLOW_LOCAL_Z),
    ],
)
def test_left_constants_match_hdgp(hdgp_src, key, expected):
    assert float(_cfg_value(hdgp_src, key)) == pytest.approx(expected)


def test_workspace_range_matches_hdgp(hdgp_src):
    raw = _cfg_value(hdgp_src, "left_tcp_workspace_range")
    vals = tuple(float(x) for x in re.findall(r"[-\d.]+", raw))
    assert vals == LEFT_TCP_WORKSPACE_RANGE


@pytest.mark.parametrize(
    "key",
    [
        "palm_delta_xyz", "palm_delta_rot_deg", "ema_action_alpha",
        "tilt_action_gate_xy_near", "tilt_action_gate_xy_far",
        "beta_target_tilt_amount", "beta_tilt_kp", "beta_tilt_max_step",
        "pour_z_margin", "target_inner_radius", "pour_corridor_xy_margin",
        "pour_corridor_z_min", "pour_corridor_z_max", "pour_corridor_scale",
        "ready_latch_threshold", "source_outer_radius",
        "pour_point_dyn_lo", "pour_point_dyn_hi",
    ],
)
def test_right_arm_path_identical_to_pour_v1(hdgp_src, key):
    """★ 이 어댑터의 전제 — 오른팔 디코딩 상수가 pour_v1과 같아야 기존 디코더를 재사용할 수 있다.

    하나라도 갈라지면 `pour_action_decoder`를 pour_sensor용으로 분기해야 한다.
    """
    if not _POUR_V1_CFG.exists():
        pytest.skip("pour_v1 cfg 없음")
    v1 = _POUR_V1_CFG.read_text(encoding="utf-8")
    assert _cfg_value(hdgp_src, key) == _cfg_value(v1, key), (
        f"{key}가 pour_v1과 갈라졌다 — 배포 디코더 재사용 전제가 깨진다"
    )


@pytest.mark.parametrize(
    "key, expected",
    [
        ("pour_action_mode", '"b_trajectory"'),
        ("pour_approach_pivot", '"palm"'),
        ("pour_spout_z_lock", "True"),
        ("pour_orient_release", "True"),
    ],
)
def test_decoder_mode_flags_unchanged(hdgp_src, key, expected):
    assert _cfg_value(hdgp_src, key) == expected


# --------------------------------------------------------------------------
# 3. 배포 노드 계약 (ROS 없이 소스 대조 — 실기 전 조용한 어긋남 방지)
# --------------------------------------------------------------------------
_NODE = Path(__file__).resolve().parent / "pour_sensor_inference.py"
_HDGP_PRESET = Path.home() / (
    "rl_ws/hdgp/source/openarm/openarm/tesollo/both/pour_sensor/pour_right_preset.py"
)


@pytest.fixture(scope="module")
def node_src() -> str:
    if not _NODE.exists():
        pytest.skip("배포 노드 없음")
    return _NODE.read_text(encoding="utf-8")


def test_node_compiles(node_src):
    compile(node_src, str(_NODE), "exec")


def test_node_uses_15d_action(node_src):
    assert "ACTION_DIM = ACTION_DIM_15" in node_src, (
        "노드가 부모의 12D를 15D로 덮어쓰지 않으면 policy 로드가 어긋난다"
    )


def test_node_feeds_real_left_encoders(node_src):
    """한팔 배포는 zeros(9)를 넣었다 — 양팔 노드는 실제 엔코더를 넣어야 한다."""
    loop = node_src.split("def _policy_loop")[1]
    assert "left_arm_joint_pos=self.left_pos" in loop
    assert "left_arm_joint_vel=self.left_vel" in loop
    assert "np.zeros(9)" not in loop, "좌팔 obs가 여전히 0으로 채워지고 있다"


def test_left_rest_matches_hdgp_preset(node_src):
    """왼팔 rest 자세가 sim preset과 다르면 receiver 위치가 어긋난다."""
    if not _HDGP_PRESET.exists():
        pytest.skip("hdgp preset 없음")
    preset = _HDGP_PRESET.read_text(encoding="utf-8")
    block = preset.split("LEFT_ARM_REST_JOINT_POS = {")[1].split("}")[0]
    sim = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r'"([a-z_0-9]+)":\s*(-?[\d.]+)', block)
    }
    m = re.search(r"^LEFT_ARM_REST = \(([^)]+)\)", node_src, re.M)
    assert m, "노드에서 LEFT_ARM_REST를 찾지 못했다"
    node_arm = [float(x) for x in m.group(1).split(",") if x.strip()]
    sim_arm = [sim[f"l_aj_{i}"] for i in range(1, 8)]
    assert node_arm == pytest.approx(sim_arm), (
        f"왼팔 rest 불일치 — node={node_arm} sim={sim_arm}"
    )

    g = re.search(r"^LEFT_GRIPPER_REST = \(([^)]+)\)", node_src, re.M)
    assert g
    node_grip = [float(x) for x in g.group(1).split(",") if x.strip()]
    sim_grip = [sim["l_hj_gripper_1"], sim["l_hj_gripper_2"]]
    assert node_grip == pytest.approx(sim_grip)


def test_left_joint_names_match_hdgp(node_src):
    if not _HDGP_PRESET.exists():
        pytest.skip("hdgp preset 없음")
    preset = _HDGP_PRESET.read_text(encoding="utf-8")
    # sim은 l_aj_1..7 + l_hj_gripper_1..2 (총 9)
    assert "l_aj_" in preset and "l_hj_gripper_1" in preset
    assert 'f"l_aj_{i}" for i in range(1, 8)' in node_src
    assert '"l_hj_gripper_1", "l_hj_gripper_2"' in node_src


def test_frozen_is_default_receiver_mode(node_src):
    """검증되지 않은 왼팔 DiffIK 경로가 기본값이 되면 안 된다."""
    assert 'receiver_mode: str = "frozen"' in node_src
    assert 'default="frozen"' in node_src


def test_action_dim_matches_hdgp():
    const = Path.home() / (
        "rl_ws/hdgp/source/openarm/openarm/tesollo/both/pour_sensor/pour_right_constants.py"
    )
    if not const.exists():
        pytest.skip("constants 없음")
    src = const.read_text(encoding="utf-8")
    # NUM_ACTIONS는 부분 상수들의 합으로 정의된다 — 각 항을 읽어 합산해 대조한다.
    parts = {}
    for name in ("NUM_PALM_ACTION", "NUM_NULLSPACE_ACTION", "NUM_HAND_ACTION",
                 "NUM_LEFT_TCP_ACTION"):
        m = re.search(rf"^{name}\s*=\s*(\d+)", src, re.M)
        assert m, f"{name}을 찾지 못했다"
        parts[name] = int(m.group(1))
    assert sum(parts.values()) == ACTION_DIM, f"action 차원 불일치: {parts}"
    assert parts["NUM_LEFT_TCP_ACTION"] == 3, "왼팔 TCP 채널이 3D가 아니다"
