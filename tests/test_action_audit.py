#!/usr/bin/env python3
"""action_audit 테스트 — 가이드 Stage 5(오프라인 추론) 판정 로직."""

from __future__ import annotations

import numpy as np
import pytest

from action_audit import ActionAudit, audit_report


def _feed(audit, arrs):
    for a in arrs:
        audit.add(np.asarray(a, dtype=np.float64))
    return audit


def test_counts_steps():
    a = _feed(ActionAudit(3), [np.zeros(3), np.ones(3)])
    assert a.steps == 2


def test_rejects_wrong_dim():
    with pytest.raises(ValueError, match="차원"):
        ActionAudit(3).add(np.zeros(4))


def test_detects_nan_and_inf():
    a = _feed(ActionAudit(2), [[0.0, np.nan], [np.inf, 0.0]])
    assert a.nan_steps == 1
    assert a.inf_steps == 1


def test_nan_does_not_poison_range():
    a = _feed(ActionAudit(2), [[0.5, np.nan], [-0.5, 0.25]])
    assert a.lo[0] == pytest.approx(-0.5)
    assert a.hi[1] == pytest.approx(0.25)


def test_saturation_fraction_per_dim():
    a = _feed(ActionAudit(2), [[1.0, 0.0], [0.99, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert a.saturated_frac()[0] == pytest.approx(0.5)
    assert a.saturated_frac()[1] == pytest.approx(0.0)


def test_out_of_range_is_flagged_separately_from_saturation():
    a = _feed(ActionAudit(1), [[1.4]])
    assert a.out_of_range_steps == 1


def test_max_jump_between_consecutive_steps():
    a = _feed(ActionAudit(2), [[0.0, 0.0], [0.1, 0.0], [0.1, 0.9]])
    assert a.max_jump == pytest.approx(0.9)


def test_constant_output_detected():
    a = _feed(ActionAudit(2), [[0.3, -0.2]] * 5)
    assert a.is_constant()


def test_varying_output_not_constant():
    a = _feed(ActionAudit(2), [[0.3, -0.2], [0.3, -0.19]])
    assert not a.is_constant()


def test_empty_audit_is_not_constant():
    assert not ActionAudit(2).is_constant()


def test_report_names_every_problem_found():
    a = _feed(ActionAudit(1), [[np.nan], [1.4], [0.0]])
    text = audit_report(a)
    assert "NaN" in text and "범위" in text


def test_report_is_clean_when_healthy():
    a = _feed(ActionAudit(2), [[0.1, 0.2], [0.15, 0.18], [0.2, 0.1]])
    text = audit_report(a)
    assert "이상 없음" in text
