"""head_home 의 설정 검증 로직 테스트."""

import pytest

from head_home import HeadHome, load_head_home


def _write(tmp_path, text):
    p = tmp_path / "head_home.yaml"
    p.write_text(text, encoding="utf-8")
    return p


GOOD = """
port: /dev/ttyUSB0
baud: 1000000
motors:
  pan: {id: 1, deg: 0.0}
  tilt: {id: 2, deg: -20.0}
position_i_gain: 400
operating_mode: 3
profile_acceleration: 20
profile_velocity: 50
"""


def test_loads_canonical_config(tmp_path):
    cfg = load_head_home(_write(tmp_path, GOOD))
    assert isinstance(cfg, HeadHome)
    assert cfg.baud == 1_000_000
    assert cfg.position_i_gain == 400
    assert cfg.targets_deg == {1: 0.0, 2: -20.0}
    assert cfg.names == {1: "pan", 2: "tilt"}


def test_rejects_missing_field(tmp_path):
    with pytest.raises(ValueError, match="position_i_gain"):
        load_head_home(_write(tmp_path, GOOD.replace("position_i_gain: 400", "")))


def test_rejects_duplicate_ids(tmp_path):
    bad = GOOD.replace("tilt: {id: 2,", "tilt: {id: 1,")
    with pytest.raises(ValueError, match="중복"):
        load_head_home(_write(tmp_path, bad))


def test_rejects_out_of_range_angle(tmp_path):
    bad = GOOD.replace("deg: -20.0", "deg: -400.0")
    with pytest.raises(ValueError, match="각도"):
        load_head_home(_write(tmp_path, bad))


def test_rejects_bad_i_gain(tmp_path):
    bad = GOOD.replace("position_i_gain: 400", "position_i_gain: 99999")
    with pytest.raises(ValueError, match="게인"):
        load_head_home(_write(tmp_path, bad))


# ---------- 쓰기 순서 (2026-09-01 에 두 번 물린 곳) ----------

class RecordingController:
    """쓰기 순서만 기록하는 가짜 컨트롤러."""

    def __init__(self):
        self.writes: list[tuple[str, int, int]] = []

    def write1(self, dxl_id, address, value, label):
        self.writes.append((label, address, value))

    write2 = write4 = write1


def _addresses(recorder):
    return [label for label, _a, _v in recorder.writes]


def _apply(tmp_path):
    from head_home import apply_one, load_head_home
    cfg = load_head_home(_write(tmp_path, GOOD))
    rec = RecordingController()
    apply_one(rec, cfg, 2)
    return rec


def test_operating_mode_written_before_gain(tmp_path):
    """모드를 쓰면 게인이 리셋된다 — 게인이 뒤여야 한다."""
    order = _addresses(_apply(tmp_path))
    assert order.index("operating mode") < order.index("position i gain")


def test_goal_written_after_torque_on(tmp_path):
    """토크 0→1 에서 펌웨어가 Goal Position 을 Present 로 덮어쓴다 — 목표가 뒤여야 한다."""
    order = _addresses(_apply(tmp_path))
    assert order.index("torque on") < order.index("goal")


def test_torque_off_comes_first(tmp_path):
    """Operating Mode 는 EEPROM 이라 토크가 켜져 있으면 쓰기가 거부된다."""
    order = _addresses(_apply(tmp_path))
    assert order[0] == "torque off"
