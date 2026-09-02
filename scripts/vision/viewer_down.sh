#!/bin/bash
source "$(dirname "$0")/common.sh"
pkill -f cup_view_stream.py || true
echo "viewer down"
