import numpy as np
import pytest

from grasp_action_decoder import (
    GraspFingerController,
    LiftLatch,
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
# finger controller (stateful, contact-gated)
# ---------------------------------------------------------------------------
def test_finger_full_close_no_contact():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    no = np.zeros(5)
    for _ in range(10):
        hand = c.step(np.ones(5), no, no)   # cmd=1, 접촉 없음 → 게이트 열림
    assert np.allclose(c.close_buf, 1.0)    # 10*0.1 → 포화
    assert np.allclose(hand, FULL)


def test_finger_action_zero_half_speed():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    no = np.zeros(5)
    c.step(np.zeros(5), no, no)             # cmd=0.5 → advance 0.05
    assert np.allclose(c.close_buf, 0.05)


def test_pip_dip_freeze_on_distal_contact():
    # index(손가락1) distal 접촉 → _3(idx 6),_4(idx 7) 동결, _1(4)/_2(5)는 계속
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    tip = np.zeros(5)
    distal = np.array([0, 1, 0, 0, 0])      # index 손가락 distal 접촉
    for _ in range(5):
        c.step(np.ones(5), tip, distal)
    # index 손가락 관절 4개: [_1,_2,_3,_4] = buf idx 4,5,6,7
    assert c.close_buf[4] == pytest.approx(0.5)   # _1 계속 진행
    assert c.close_buf[5] == pytest.approx(0.5)   # _2 계속 진행
    assert c.close_buf[6] == pytest.approx(0.0)   # _3 동결
    assert c.close_buf[7] == pytest.approx(0.0)   # _4 동결
    # thumb(손가락0) 관절은 접촉 없어 전부 진행
    assert np.allclose(c.close_buf[0:4], 0.5)


def test_tip_contact_also_freezes_pip_dip():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    tip = np.array([1, 0, 0, 0, 0])         # thumb tip 접촉
    distal = np.zeros(5)
    for _ in range(3):
        c.step(np.ones(5), tip, distal)
    assert c.close_buf[2] == pytest.approx(0.0)   # thumb _3 동결(tip)
    assert c.close_buf[3] == pytest.approx(0.0)   # thumb _4 동결(tip)
    assert c.close_buf[0] == pytest.approx(0.3)   # thumb _1 진행


def test_tip_only_gate_when_distal_none():
    # 라이브 tip-only: distal 미제공 → tip 접촉만 _3/_4 동결
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    tip = np.array([0, 1, 0, 0, 0])   # index tip 접촉
    for _ in range(5):
        c.step(np.ones(5), tip)        # distal_contact 생략 → None
    assert c.close_buf[6] == pytest.approx(0.0)   # index _3 동결(tip)
    assert c.close_buf[7] == pytest.approx(0.0)   # index _4 동결(tip)
    assert c.close_buf[4] == pytest.approx(0.5)   # index _1 진행
    # 접촉 없는 thumb는 전부 진행 (distal 없어도 tip-only가 막지 않음)
    assert np.allclose(c.close_buf[0:4], 0.5)


def test_reset():
    c = GraspFingerController(OPEN, FULL, close_speed=0.1)
    c.step(np.ones(5), np.zeros(5), np.zeros(5))
    c.reset()
    assert np.allclose(c.close_buf, 0.0)


def test_finger_wrong_dims():
    c = GraspFingerController(OPEN, FULL)
    with pytest.raises(ValueError):
        c.step(np.zeros(4), np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError):
        GraspFingerController(np.zeros(19), FULL)
