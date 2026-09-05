#!/usr/bin/env bash
# dynamixel-tools CLI 를 sim2real 의 venv(dynamixel_sdk 보유)로 실행하는 얇은 래퍼.
#
#   ./dxl.sh scan
#   ./dxl.sh health --baud 57600 --ids 1
#   ./dxl.sh set-id --baud 57600 --old-id 1 --new-id 2
#
# dialout 그룹 적용 전 셸이면 앞에 `sg dialout -c "..."` 를 붙일 것.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/../../../dynamixel-tools/src"
PY="$HERE/../../.venv/bin/python"

[ -d "$TOOLS" ] || { echo "dynamixel-tools 가 없다: $TOOLS" >&2; exit 2; }
[ -x "$PY" ] || { echo "venv 파이썬이 없다: $PY" >&2; exit 2; }

CMD="${1:-}"; shift || true
case "$CMD" in
  scan)     FN=scan_main ;;
  health)   FN=health_main ;;
  set-id)   FN=set_id_main ;;
  set-baud) FN=set_baud_main ;;
  *) echo "사용법: dxl.sh {scan|health|set-id|set-baud} [옵션...]" >&2; exit 2 ;;
esac

exec env PYTHONPATH="$TOOLS" "$PY" -c "
import sys
sys.argv = ['dxl-$CMD'] + sys.argv[1:]
from dynamixel_tools.cli import $FN
raise SystemExit($FN())
" "$@"
