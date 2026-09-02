#!/bin/bash
# usage: fpp_down.sh <container|all>
source "$(dirname "$0")/common.sh"
TARGET=${1:?container name or all}
if [ "$TARGET" = all ]; then
  docker ps -a --filter name='^fpp_' --format '{{.Names}}' | xargs -r docker rm -f >/dev/null
else
  docker rm -f "$TARGET" >/dev/null 2>&1 || true
fi
echo "down $TARGET"
