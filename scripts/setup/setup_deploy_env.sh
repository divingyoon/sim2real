#!/usr/bin/env bash
# 배포 파이썬 환경 구축 — 라이브 정책 노드가 필요한 것 전부를 **한 인터프리터**에.
#
# 왜 필요한가 (2026-08-20 발견):
#   · ROS Humble 의 python3.10 : rclpy ✅ / warp·rl_games·fabrics_sim ❌
#     게다가 시스템 torch 2.2.1+cu121 은 sm_90 까지만 지원 → **RTX 5090(sm_120)에서
#     CUDA 커널이 없다**("no kernel image is available"). 정책을 GPU 로 못 돌린다.
#   · Isaac 번들 python3.11 : torch cu128·warp·rl_games ✅ / **rclpy ❌**
#     (ROS Humble 의 rclpy 는 python3.10 빌드라 3.11 에서 못 쓴다)
#   → 어느 쪽으로도 `grasp_inference.py` 를 실행할 수 없었다. 이 스크립트가 그 간극을 메운다.
#
# 방식: ROS 의 python3.10 위에 `--system-site-packages` venv 를 만들어 rclpy 를 상속받고,
#       torch(cu128)·warp·rl_games·fabrics_sim 만 venv 안에서 덮어쓴다. 시스템은 건드리지 않는다.
#
# 사용:
#   bash scripts/setup_deploy_env.sh
#   source /opt/ros/humble/setup.bash && sim2real/.venv/bin/python scripts/nodes/grasp_inference.py ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$HERE/.venv"
FABRICS_SRC="$(cd "$HERE/../hdgp/source/FABRICS" && pwd)"

# Isaac 학습 환경과 맞춘 핀 — 학습·배포가 다른 버전을 쓰면 체크포인트 거동이 갈릴 수 있다.
TORCH_SPEC="torch==2.7.1+cu128"
WARP_SPEC="warp-lang==1.8.1"      # fabrics_sim 요구: >=1.5.0,<1.8.2
RLGAMES_SPEC="rl-games==1.6.1"
NUMPY_SPEC="numpy==1.26.4"        # fabrics_sim 요구: <2.0.0 (torch/rl_games 가 2.x 를 끌어온다)
SCIPY_SPEC="scipy==1.13.1"        # 위 numpy 와 ABI 정합(시스템 scipy 1.8.0 은 불일치)

echo "== venv 생성 (rclpy 상속) =="
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip

echo "== torch (cu128, sm_120) =="
"$VENV/bin/python" -m pip install -q --index-url https://download.pytorch.org/whl/cu128 "$TORCH_SPEC"

echo "== warp / rl_games / numpy / scipy / urdfpy =="
"$VENV/bin/python" -m pip install -q "$WARP_SPEC" "$RLGAMES_SPEC"
"$VENV/bin/python" -m pip install -q "$NUMPY_SPEC" "$SCIPY_SPEC" "PyYAML>=6.0.2" "urdfpy==0.0.22"

echo "== fabrics_sim (소스 트리를 경로로 연결) =="
SP="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$FABRICS_SRC/src" > "$SP/fabrics_sim.pth"

echo "== urdfpy/networkx 패치 (venv 안에서만) =="
# urdfpy 0.0.22 가 끌어오는 networkx 2.2 는 py3.10 에서 `from collections import Mapping` 로 죽는다.
PATH="$VENV/bin:$PATH" bash "$FABRICS_SRC/urdfpy_patch.sh"

echo "== 검증 =="
source /opt/ros/humble/setup.bash
"$VENV/bin/python" - <<'PY'
import sys, numpy, scipy, torch, warp, rl_games, rclpy  # noqa: F401
import importlib.metadata as md
from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric  # noqa: F401
assert "sm_120" in torch.cuda.get_arch_list(), "torch 가 sm_120(RTX 5090)을 지원하지 않는다"
x = torch.zeros(4, device="cuda"); assert float((x + 1).sum()) == 4.0
print(f"  python {sys.version.split()[0]} | numpy {numpy.__version__} | scipy {scipy.__version__}")
print(f"  torch {torch.__version__} | warp {warp.config.version} | rl_games {md.version('rl-games')}")
print(f"  rclpy OK | fabrics_sim OK | CUDA on {torch.cuda.get_device_name(0)}")
PY
echo "완료. 실행 전 반드시: source /opt/ros/humble/setup.bash"
