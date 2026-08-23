"""obs_contract_report 의 **탐색 범위** 테스트.

추출기 자체는 test_obs_contract.py 가 본다. 여기서 지키는 것은 하나다 —
**계약을 정의하는 태스크가 목록에서 조용히 사라지지 않는가.**

이전 구현은 `*_env.py` 중 `_get_observations` 를 가진 것만 훑었다. hdgp 에는
ObsTerm 으로 관측을 정의하는 manager-based 태스크가 12개 있는데, 그 전부가
실패로도 안 잡히고 사라져 "추출 성공 23 / 실패 0" 이라는 거짓 안심을 만들었다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import obs_contract_report as R  # noqa: E402
from robot_profile import HDGP_OPENARM_SRC  # noqa: E402


_GRIPPER_LEFT = "gripper/left/grasp_sensor/"   # ★끝의 / 필수 —
#   `grasp_sensor_fabrics_ABORTED/`(폐기된 direct env 판)가 부분일치로 딸려 온다.


def _sources():
    return list(R.task_sources(HDGP_OPENARM_SRC))


def test_manager_based_tasks_are_discovered_not_dropped():
    paths = {str(s.path) for s in _sources()}
    assert any(_GRIPPER_LEFT in p for p in paths), (
        "manager-based 태스크가 목록에 없다 — 사라지는 경로가 되살아났다"
    )


def test_manager_based_tasks_are_labelled_as_unsupported_rather_than_successful():
    managers = [s for s in _sources() if _GRIPPER_LEFT in str(s.path)]
    assert managers, "gripper/left/grasp_sensor 를 못 찾았다"
    for src in managers:
        assert src.kind == R.MANAGER_BASED


def test_direct_rl_tasks_keep_their_kind():
    direct = [s for s in _sources() if "agnostic/tasks/grasp_sensor" in str(s.path)]
    assert direct, "agnostic/tasks/grasp_sensor 를 못 찾았다"
    assert all(s.kind == R.DIRECT_RL for s in direct)


def test_every_obsterm_task_in_hdgp_is_accounted_for():
    """ObsTerm 을 쓰는 파일은 하나도 빠짐없이 목록에 있어야 한다."""
    on_disk = {
        f.resolve()
        for f in HDGP_OPENARM_SRC.rglob("*_env_cfg.py")
        if "/tests/" not in str(f) and "ObsTerm(" in f.read_text(encoding="utf-8")
    }
    listed = {s.path.resolve() for s in _sources()}
    assert on_disk <= listed, f"목록에서 누락: {sorted(on_disk - listed)}"


def test_summary_reports_manager_based_count_separately_from_failures():
    """미지원과 실패는 다르다 — 섞으면 종료코드가 늘 1 이 되어 신호가 죽는다."""
    counts = R.summarize(HDGP_OPENARM_SRC)
    assert counts.manager_based >= 12, counts
    assert counts.extracted >= 20, counts
    assert counts.failed == 0, f"직접 env 추출 실패: {counts.failed}"
