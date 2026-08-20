#!/usr/bin/env python3
"""obs_contract 테스트 — hdgp 태스크의 관측 계약을 소스에서 자동 추출한다."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from obs_contract import (  # noqa: E402
    ObsContractError,
    diff_segments,
    extract_obs_contract,
)


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "t_env.py"
    p.write_text(textwrap.dedent(body))
    return p


SIMPLE = '''
    import torch
    class E:
        def _get_observations(self):
            a = self.q
            b = self.qd
            actor_obs = torch.cat([a, b], dim=-1)
            return {"policy": actor_obs}
    '''


def test_extracts_segment_names_in_order(tmp_path):
    c = extract_obs_contract(_write(tmp_path, SIMPLE))
    assert [s.name for s in c.segments] == ["a", "b"]
    assert [s.index for s in c.segments] == [0, 1]


def test_records_defining_expression(tmp_path):
    c = extract_obs_contract(_write(tmp_path, SIMPLE))
    assert c.segments[0].expr == "self.q"


def test_keeps_expression_terms_instead_of_dropping_them(tmp_path):
    """★구 정규식 추출기는 표현식 항을 조용히 버렸다 — 세그먼트 수가 어긋난다."""
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                a = self.q
                actor_obs = torch.cat([a, (self.p - self.c).view(1, -1)], dim=-1)
                return {"policy": actor_obs}
        '''))
    assert len(c.segments) == 2
    assert c.segments[1].is_named is False
    assert "self.p - self.c" in c.segments[1].expr


def test_flags_domain_randomization_noise(tmp_path):
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                a = a_clean + torch.randn_like(a_clean) * s
                b = self.qd
                actor_obs = torch.cat([a, b], dim=-1)
                return {"policy": actor_obs}
        '''))
    assert c.segments[0].dr_noise is True
    assert c.segments[1].dr_noise is False


def test_collects_normalization_constants(tmp_path):
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                f = (raw / CONTACT_FORCE_MAX).clamp(-1.0, 1.0)
                actor_obs = torch.cat([f], dim=-1)
                return {"policy": actor_obs}
        '''))
    assert c.segments[0].constants == ("CONTACT_FORCE_MAX",)


def test_raises_when_no_obs_concat_found(tmp_path):
    with pytest.raises(ObsContractError, match="찾지 못"):
        extract_obs_contract(_write(tmp_path, '''
            class E:
                def _get_observations(self):
                    return {"policy": self.buf}
            '''))


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_obs_contract(tmp_path / "nope.py")


def test_accepts_alternative_variable_name(tmp_path):
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                obs = torch.cat([self.x], dim=-1)
                return {"policy": obs}
        '''))
    assert [s.name for s in c.segments] == ["self.x"]


def test_resolves_list_held_in_a_variable(tmp_path):
    """`parts = [...]; obs = torch.cat(parts)` — hdgp 4개 태스크가 쓰는 형태."""
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                a = self.q
                actor_obs_parts = [a, self.b]
                actor_obs = torch.cat(actor_obs_parts, dim=-1)
                return {"policy": actor_obs}
        '''))
    assert [s.name for s in c.segments] == ["a", "self.b"]


def test_sees_through_nan_to_num_reassignment(tmp_path):
    """`obs = cat(...)` 뒤에 `obs = nan_to_num(obs)` 로 덮어써도 cat 을 찾아야 한다."""
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                actor_obs = torch.cat([self.a, self.b], dim=-1)
                actor_obs = torch.nan_to_num(actor_obs, nan=0.0)
                return {"policy": actor_obs}
        '''))
    assert [s.name for s in c.segments] == ["self.a", "self.b"]


def test_resolves_list_built_by_append(tmp_path):
    c = extract_obs_contract(_write(tmp_path, '''
        import torch
        class E:
            def _get_observations(self):
                parts = []
                parts.append(self.a)
                parts.append(self.b)
                obs = torch.cat(parts, dim=1)
                return {"policy": obs}
        '''))
    assert [s.name for s in c.segments] == ["self.a", "self.b"]


# ------------------------------------------------------------------ diff
def test_diff_reports_nothing_when_equal():
    assert diff_segments(["a", "b"], ["a", "b"]) == []


def test_diff_reports_order_change():
    out = diff_segments(["a", "b"], ["b", "a"])
    assert any("순서" in d for d in out)


def test_diff_reports_missing_and_extra():
    out = diff_segments(["a", "b"], ["a", "c"])
    assert any("배포에 없음: b" in d for d in out)
    assert any("sim 에 없음: c" in d for d in out)


# ------------------------------------------------------------------ 실제 hdgp
_HDGP = Path.home() / "rl_ws/hdgp/source/openarm/openarm"
pytestmark_real = pytest.mark.skipif(not _HDGP.exists(), reason="hdgp 소스 없음")


@pytestmark_real
def test_real_grasp_sensor_contract_has_twelve_segments():
    c = extract_obs_contract(_HDGP / "tesollo/right/grasp_sensor/grasp_right_env.py")
    assert len(c.segments) == 12
    assert c.segments[0].name == "arm_joint_pos"
    assert c.segments[-1].name == "object_onehot"


@pytestmark_real
def test_real_grasp_sensor_marks_noised_segments():
    c = extract_obs_contract(_HDGP / "tesollo/right/grasp_sensor/grasp_right_env.py")
    noised = [s.name for s in c.segments if s.dr_noise]
    assert "arm_joint_pos" in noised
    assert "last_actions" not in noised
