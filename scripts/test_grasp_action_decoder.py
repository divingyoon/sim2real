import numpy as np
import pytest

from grasp_action_decoder import (
    FINGER_SLICE,
    NUM_ACTIONS,
    NUM_FINGER_ACTION,
    NUM_FINGER_CHANNELS,
    PALM_SLICE,
    GraspFingerController,
    LiftLatch,
    couple_four_fingers,
    expand_channels_to_joints,
    freeze_gate20,
    joint7_lift_wait_target,
    lift_arm_interp,
    num_grip_fingers,
    scale_palm_delta,
)

OPEN = np.zeros(20)
FULL = np.ones(20)  # 합성: 모든 관절 0→1 (실제는 APPROACH↔FULL_GRIP)


# ---------------------------------------------------------------------------
# palm delta scale
# ---------------------------------------------------------------------------
def test_scale_palm_delta_endpoints():
    lo = -np.ones(6) * 0.15
    hi = np.ones(6) * 0.15
    assert np.allclose(scale_palm_delta(np.zeros(6), lo, hi), 0.0)       # action 0 → 0
    assert np.allclose(scale_palm_delta(np.ones(6), lo, hi), 0.15)       # +1 → max
    assert np.allclose(scale_palm_delta(-np.ones(6), lo, hi), -0.15)     # -1 → min


def test_scale_palm_delta_wrong_dim():
    with pytest.raises(ValueError):
        scale_palm_delta(np.zeros(5), np.zeros(6), np.zeros(6))


# ---------------------------------------------------------------------------
# joint7 lift-wait
# ---------------------------------------------------------------------------
def test_joint7_only_moves_index6():
    arm = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    out = joint7_lift_wait_target(arm, joint7_delta=0.31, joint7_min=0.20, joint7_max=1.50)
    assert np.allclose(out[:6], arm[:6])           # 나머지 불변
    assert out[6] == pytest.approx(0.7 + 0.31)     # joint7만 +delta


def test_joint7_clamps_to_max():
    arm = np.zeros(7)
    arm[6] = 1.40
    out = joint7_lift_wait_target(arm, joint7_delta=0.31, joint7_min=0.20, joint7_max=1.50)
    assert out[6] == pytest.approx(1.50)


# ---------------------------------------------------------------------------
# lift arm interp
# ---------------------------------------------------------------------------
def test_lift_arm_interp_endpoints():
    a = np.zeros(7)
    b = np.ones(7)
    assert np.allclose(lift_arm_interp(a, b, 0.0), a)
    assert np.allclose(lift_arm_interp(a, b, 1.0), b)
    assert np.allclose(lift_arm_interp(a, b, 0.5), 0.5)
    assert np.allclose(lift_arm_interp(a, b, 2.0), b)  # clamp


# ---------------------------------------------------------------------------
# lift latch
# ---------------------------------------------------------------------------
def test_latch_requires_hold_steps():
    latch = LiftLatch(min_contacts=3, hold_steps=8)
    for _ in range(7):
        assert latch.update(3) is False   # 7스텝은 부족
    assert latch.update(3) is True        # 8번째 래치


def test_latch_resets_hold_on_contact_loss():
    latch = LiftLatch(min_contacts=3, hold_steps=8)
    for _ in range(5):
        latch.update(3)
    assert latch.update(2) is False       # 접촉 하락 → hold 리셋
    assert latch.hold_count == 0
    for _ in range(7):
        latch.update(3)
    assert latch.latched is False         # 아직 7
    assert latch.update(3) is True


def test_latch_stays_latched():
    latch = LiftLatch(min_contacts=3, hold_steps=2)
    latch.update(3)
    assert latch.update(3) is True
    assert latch.update(0) is True        # 이후 접촉 0이어도 유지


# ---------------------------------------------------------------------------
# num_grip_fingers
# ---------------------------------------------------------------------------
def test_num_grip_fingers_or_of_modes():
    tip = np.array([1, 0, 0, 0, 0])
    mid = np.array([0, 1, 0, 0, 0])
    dist = np.array([0, 0, 1, 0, 0])
    assert num_grip_fingers(tip, mid, dist) == 3   # 서로 다른 손가락
    # 같은 손가락 중복은 1로
    assert num_grip_fingers(tip, tip, tip) == 1


# ---------------------------------------------------------------------------
# finger controller (stateful, contact-gated) — ★08.16 계약: 15D(5×3 채널), 래칫 제거
# ---------------------------------------------------------------------------
def _fa(value=1.0):
    """모든 손가락·채널이 같은 값인 (15,) action."""
    return np.full(NUM_FINGER_ACTION, float(value))


def test_action_layout_constants():
    assert NUM_ACTIONS == 21
    assert NUM_FINGER_ACTION == 15 and NUM_FINGER_CHANNELS == 3
    assert PALM_SLICE == slice(0, 6) and FINGER_SLICE == slice(6, 21)


def test_finger_full_close_no_contact():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    no = np.zeros(5)
    for _ in range(10):
        hand = c.step(_fa(1.0), no, no)      # 목표 1.0, 접촉 없음
    assert np.allclose(c.close_buf, 1.0)
    assert np.allclose(hand, FULL)


def test_absolute_target_not_speed():
    """cmd 는 **절대 폐쇄도 목표**다 — 목표에 도달하면 더 진행하지 않는다."""
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    no = np.zeros(5)
    for _ in range(20):
        c.step(_fa(0.0), no, no)             # 목표 0.5
    assert np.allclose(c.close_buf, 0.5)     # 구 래칫이면 1.0 으로 포화했다


def test_ratchet_removed_close_buf_can_decrease():
    """★가장 중요: 목표를 낮추면 close_buf 가 **감소**해야 한다(구 구현은 불가)."""
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    no = np.zeros(5)
    for _ in range(15):
        c.step(_fa(1.0), no, no)
    assert np.allclose(c.close_buf, 1.0)
    for _ in range(15):
        c.step(_fa(-1.0), no, no)            # 목표 0.0
    assert np.allclose(c.close_buf, 0.0)


def test_rate_limit_per_step():
    c = GraspFingerController(OPEN, FULL, close_speed=0.05)
    no = np.zeros(5)
    prev = c.close_buf.copy()
    rng = np.random.RandomState(0)
    for _ in range(30):
        c.step(rng.uniform(-1.0, 1.0, NUM_FINGER_ACTION), no, no)
        assert np.all(np.abs(c.close_buf - prev) <= 0.05 + 1e-12)
        prev = c.close_buf.copy()


def test_couple_four_fingers_equalizes_index_to_pinky():
    """검지~소지는 채널별 평균으로 묶여 같은 폐쇄도를 갖는다(3지 국소최적 차단)."""
    c = GraspFingerController(OPEN, FULL, close_speed=0.5, couple_four=True)
    fa = np.zeros((5, 3))
    fa[1] = 1.0
    fa[2] = -1.0
    fa[3] = 0.5
    fa[4] = -0.5
    c.step(fa, np.zeros(5), np.zeros(5))
    buf = c.close_buf.reshape(5, 4)
    assert np.allclose(buf[1], buf[2]) and np.allclose(buf[2], buf[3]) and np.allclose(buf[3], buf[4])


def test_thumb_independent_of_four_finger_coupling():
    c = GraspFingerController(OPEN, FULL, close_speed=0.5, couple_four=True)
    fa = np.zeros((5, 3))
    fa[0] = 1.0        # 엄지만 최대
    fa[1:] = -1.0
    c.step(fa, np.zeros(5), np.zeros(5))
    buf = c.close_buf.reshape(5, 4)
    assert buf[0, 0] > buf[1, 0]


def test_coupling_happens_before_clamp():
    """★비대칭 입력으로 순서를 고정한다: 평균 후 clamp = 0.75, clamp 선행 = 0.25."""
    fa = np.zeros((5, 3))
    fa[1, 0] = 3.0                            # 한 손가락만 범위 밖
    coupled = couple_four_fingers(fa)
    assert coupled[1, 0] == pytest.approx(0.75)
    cmd = 0.5 * (np.clip(coupled, -1.0, 1.0) + 1.0)
    assert cmd[1, 0] == pytest.approx(0.875)


def test_channel_expansion_pip_dip_share():
    """_3(PIP)과 _4(DIP)는 채널 2를 공유 — 항상 같은 지령."""
    cmd_ch = np.arange(15, dtype=float).reshape(5, 3)
    out = expand_channels_to_joints(cmd_ch).reshape(5, 4)
    assert np.allclose(out[:, 2], out[:, 3])
    assert np.allclose(out[:, 0], cmd_ch[:, 0])
    assert np.allclose(out[:, 1], cmd_ch[:, 1])


def test_pip_dip_freeze_on_distal_contact():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1, couple_four=False)
    tip = np.zeros(5)
    distal = np.array([0, 1, 0, 0, 0])        # index 손가락 distal 접촉
    for _ in range(5):
        c.step(_fa(1.0), tip, distal)
    assert c.close_buf[4] == pytest.approx(0.5)   # index _1 진행
    assert c.close_buf[5] == pytest.approx(0.5)   # index _2 진행
    assert c.close_buf[6] == pytest.approx(0.0)   # index _3 동결
    assert c.close_buf[7] == pytest.approx(0.0)   # index _4 동결
    assert np.allclose(c.close_buf[0:4], 0.5)     # thumb 는 무접촉


def test_tip_contact_also_freezes_pip_dip():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1, couple_four=False)
    tip = np.array([1, 0, 0, 0, 0])
    for _ in range(3):
        c.step(_fa(1.0), tip, np.zeros(5))
    assert c.close_buf[2] == pytest.approx(0.0)
    assert c.close_buf[3] == pytest.approx(0.0)
    assert c.close_buf[0] == pytest.approx(0.3)


def test_tip_only_gate_when_distal_none():
    """라이브 배포: distal 미제공 → tip 접촉만으로 _3/_4 동결(의도된 sim 차이)."""
    c = GraspFingerController(OPEN, FULL, close_speed=0.1, couple_four=False)
    tip = np.array([0, 1, 0, 0, 0])
    for _ in range(5):
        c.step(_fa(1.0), tip)
    assert c.close_buf[6] == pytest.approx(0.0)
    assert c.close_buf[7] == pytest.approx(0.0)
    assert c.close_buf[4] == pytest.approx(0.5)


def test_freeze_blocks_both_directions():
    """동결은 증가·감소를 모두 막는다(그 자리에 멈춰 컵 형상에 드리워진다)."""
    c = GraspFingerController(OPEN, FULL, close_speed=0.2, couple_four=False)
    for _ in range(5):
        c.step(_fa(1.0), np.zeros(5))          # 무접촉으로 진행
    held = c.close_buf.reshape(5, 4)[0, 2]
    tip = np.array([1, 0, 0, 0, 0])
    for _ in range(5):
        c.step(_fa(-1.0), tip)                 # 목표를 0 으로 낮춰도 thumb _3 는 동결
    assert c.close_buf.reshape(5, 4)[0, 2] == pytest.approx(held)


def test_retighten_after_latch_releases_gate():
    """cfg 는 양측 False 지만, True 로 바뀌면 래치 후 동결이 풀려야 한다."""
    c = GraspFingerController(OPEN, FULL, close_speed=0.2, couple_four=False,
                              retighten_after_latch=True)
    tip = np.array([1, 0, 0, 0, 0])
    for _ in range(3):
        c.step(_fa(1.0), tip, latched=False)
    frozen = c.close_buf.reshape(5, 4)[0, 2]
    assert frozen == pytest.approx(0.0)
    for _ in range(3):
        c.step(_fa(1.0), tip, latched=True)
    assert c.close_buf.reshape(5, 4)[0, 2] > 0.0


def test_joint_limits_clamp_applied():
    """★D3: 한계를 주입하지 않으면 FULL_GRIP 이 실기 상한을 넘어 joint_pos_err 이 편향된다."""
    upper = np.full(20, 0.4)
    c = GraspFingerController(OPEN, FULL, close_speed=1.0,
                              lower_limits=np.full(20, -1.0), upper_limits=upper)
    hand = c.step(_fa(1.0), np.zeros(5))
    assert np.all(hand <= upper + 1e-12)


def test_reset():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    c.step(_fa(1.0), np.zeros(5), np.zeros(5))
    c.reset()
    assert np.allclose(c.close_buf, 0.0)


def test_finger_wrong_dims():
    c = GraspFingerController(OPEN, FULL)
    with pytest.raises(ValueError, match="21D action"):
        c.step(np.zeros(5), np.zeros(5))       # ★구 계약 5D 는 거부
    with pytest.raises(ValueError):
        c.step(np.zeros(14), np.zeros(5))
    with pytest.raises(ValueError):
        GraspFingerController(np.zeros(19), FULL)


def test_pure_fn_shape_validation():
    with pytest.raises(ValueError):
        couple_four_fingers(np.zeros(15))
    with pytest.raises(ValueError):
        expand_channels_to_joints(np.zeros((5, 4)))
    with pytest.raises(ValueError):
        freeze_gate20(np.zeros(4))
