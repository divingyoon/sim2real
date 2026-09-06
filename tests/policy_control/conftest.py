"""Fixtures for the policy_control package tests.

``sim2real/policy_control`` is put on ``sys.path`` so ``import policy_control``
works without a colcon install; ``policy_control._paths`` then exposes the
sibling trees. The ``ros`` fixture creates a **private rclpy Context on an
isolated domain** (``PC_TEST_DOMAIN``, default 99) so a test can never talk to a
real robot's DDS graph on the same host.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SIM2REAL = Path(__file__).resolve().parents[2]
PKG_DIR = SIM2REAL / "policy_control"
FIXTURES = SIM2REAL / "tests" / "fixtures" / "policy_control"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import policy_control._paths  # noqa: E402,F401  (side effect: sibling trees on sys.path)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def sim2real_dir() -> Path:
    return SIM2REAL


@pytest.fixture
def ros():
    """Private rclpy context on the test domain; torn down after the test."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context

    domain = int(os.environ.get("PC_TEST_DOMAIN", "99"))
    context = Context()
    rclpy.init(context=context, domain_id=domain)
    try:
        yield context
    finally:
        rclpy.shutdown(context=context)
