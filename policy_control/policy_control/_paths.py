"""The single place that puts the sibling code trees on ``sys.path``.

The pure logic this package reuses lives outside it: ``sim2real/scripts`` (obs
builders, decoders, jtc bridge core, policy loader), ``robot_control/src``
(kinematics, safety) and hdgp's Fabrics package. Every module that imports from
those trees does ``from . import _paths`` first, so there is exactly one
definition of where they are. Nothing here imports Isaac/omni.
"""
from __future__ import annotations

import sys
from pathlib import Path

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
SCRIPTS = SIM2REAL / "scripts"
ROBOT_CONTROL_SRC = RL_WS / "robot_control" / "src"
FABRICS_SRC = RL_WS / "hdgp" / "source" / "FABRICS" / "src"
HDGP_OPENARM = RL_WS / "hdgp" / "source" / "openarm"

_SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _script_dirs() -> list[Path]:
    subdirs = sorted(d for d in SCRIPTS.iterdir() if d.is_dir() and d.name not in _SKIP)
    return [SCRIPTS, *subdirs]


def install() -> None:
    """Idempotently prepend the sibling trees (scripts/ first, like tests/conftest.py)."""
    for path in [*_script_dirs(), ROBOT_CONTROL_SRC, FABRICS_SRC, HDGP_OPENARM]:
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


install()
