#!/bin/bash
# 마지막 줄이 JSON. 카메라 hz 는 로컬 런처가 DDS 로 직접 잰다(여기선 프로세스 유무만).
source "$(dirname "$0")/common.sh"
python3 - <<'EOF'
import json, subprocess
def up(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
out = subprocess.run(["docker", "ps", "-a", "--filter", "name=^fpp_", "--format", "{{.Names}}\t{{.Status}}"],
                     capture_output=True, text=True).stdout
containers = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)
print(json.dumps({"camera_up": up("realsense2_camera_node"), "containers": containers,
                  "viewer_up": up("cup_view_stream.py")}))
EOF
