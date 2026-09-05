"""테스트가 `scripts/` 트리의 모듈을 이름으로 임포트할 수 있게 한다.

★모듈은 역할별 하위 디렉토리(`nodes/`·`ops/`·`calib/`·…)로 나뉘어 있고 라이브러리만
  `scripts/` 최상위에 있다. 예전에는 전부 한 디렉토리에 평평하게 있어 pytest 가
  알아서 경로를 잡아줬는데, 나눈 뒤로는 **이 파일 하나가** 그 역할을 대신한다.
  테스트마다 sys.path 를 손대지 말 것 — 여기 한 곳만 고치면 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache"}
for _d in [SCRIPTS, *(d for d in sorted(SCRIPTS.iterdir())
                      if d.is_dir() and d.name not in _SKIP)]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
