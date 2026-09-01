#!/usr/bin/env bash
# 환경 진단 스크립트 — 어느 PC에서든 실행하면 뭐가 준비됐고 뭐가 빠졌는지 출력.
# 설치는 하지 않는다(읽기 전용). 각 항목의 설치 방법은 INSTALL.md 해당 Step 참조.
#
# 사용:
#   ./scripts/setup_check.sh            # 전체 점검
#   ./scripts/setup_check.sh control    # 로봇 제어 PC 항목만
#   ./scripts/setup_check.sh vision     # 비전(FoundationPose) PC 항목만
#   ./scripts/setup_check.sh policy     # 정책 추론 PC 항목만

set -u

ROLE="${1:-all}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0; MISS=0

ok()   { printf '  [ OK ] %s\n' "$1"; PASS=$((PASS+1)); }
miss() { printf '  [MISS] %s\n         → %s\n' "$1" "$2"; MISS=$((MISS+1)); }

section() { printf '\n== %s ==\n' "$1"; }

want() {  # want <role>: 현재 ROLE에서 이 섹션을 점검하나
  [[ "$ROLE" == "all" || "$ROLE" == "$1" ]]
}

# ── 공통 (모든 역할) ────────────────────────────────────────────────────────
section "공통 — OS / ROS2 / 레포"

if [[ "$(uname -m)" == "x86_64" ]]; then
  ok "아키텍처 x86_64"
else
  miss "아키텍처 $(uname -m)" "Isaac ROS는 x86_64 또는 Jetson만 지원 (INSTALL.md Step 1)"
fi

if grep -q 'VERSION_ID="22.04"' /etc/os-release 2>/dev/null; then
  ok "Ubuntu 22.04"
else
  miss "Ubuntu 22.04 아님 ($(grep VERSION_ID /etc/os-release 2>/dev/null || echo '?'))" \
       "ROS2 Humble + Isaac ROS 공식 지원은 22.04 (INSTALL.md Step 1)"
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  ok "ROS2 Humble (/opt/ros/humble)"
else
  miss "ROS2 Humble" "INSTALL.md Step 2"
fi

if command -v colcon >/dev/null 2>&1; then
  ok "colcon"
else
  miss "colcon" "sudo apt install python3-colcon-common-extensions (INSTALL.md Step 2)"
fi

if [[ -f "${REPO_DIR}/install/setup.bash" ]]; then
  ok "레포 colcon 빌드됨 (install/setup.bash)"
else
  miss "레포 미빌드" "./scripts/build_vendor_pkgs.sh (INSTALL.md Step 3)"
fi

if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
  ok "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
else
  miss "ROS_DOMAIN_ID 미설정" "export ROS_DOMAIN_ID=126 — 모든 PC 동일해야 통신됨 (INSTALL.md Step 2)"
fi

for pkg in numpy yaml pytest; do
  if python3 -c "import ${pkg}" >/dev/null 2>&1; then
    ok "python3-${pkg}"
  else
    miss "python3-${pkg}" "pip install ${pkg/yaml/pyyaml} (INSTALL.md Step 4)"
  fi
done

if python3 -c "import rclpy, vision_msgs" >/dev/null 2>&1; then
  ok "rclpy + vision_msgs"
else
  miss "rclpy/vision_msgs" "source /opt/ros/humble/setup.bash 후 재실행, 없으면 sudo apt install ros-humble-vision-msgs (INSTALL.md Step 2)"
fi

# ── 제어 PC (실물 로봇 드라이버) ────────────────────────────────────────────
if want control; then
  section "control — 실물 로봇 드라이버"

  for d in vendor/openarm vendor/inspire_ws; do
    if [[ -d "${REPO_DIR}/${d}" ]]; then
      ok "${d}/"
    else
      miss "${d}/ 없음" "레포 클론 불완전 — git status 확인 (INSTALL.md Step 3)"
    fi
  done

  # Tesollo 드라이버는 robot_control 소유. 여기서는 빌드된 install을 오버레이한다.
  ROBOT_CONTROL_INSTALL="${ROBOT_CONTROL_INSTALL:-${REPO_DIR}/../robot_control/ros_ws/install}"
  if [[ -f "${ROBOT_CONTROL_INSTALL}/dg5f_driver/share/dg5f_driver/package.xml" ]]; then
    ok "robot_control ros_ws/install (Tesollo 드라이버)"
  else
    miss "robot_control ROS 워크스페이스 미빌드" \
         "robot_control/ros_ws/build.sh 실행 (ROBOT_CONTROL_INSTALL 로 경로 지정 가능)"
  fi

  if [[ -d "${REPO_DIR}/vendor/inspire_ws/install" ]]; then
    ok "inspire_ws 빌드됨 (RH56F1)"
  else
    miss "inspire_ws 미빌드 (RH56F1 쓸 때만 필요)" "INSTALL.md Step 3-B"
  fi

  if ip link show 2>/dev/null | grep -q "can"; then
    ok "CAN 인터페이스 감지 (OpenArm)"
  else
    miss "CAN 인터페이스 없음 (OpenArm 연결 PC만 해당)" "USAGE_ISAACSIM_ROS2.md §1"
  fi
fi

# ── 비전 PC (FoundationPose) ────────────────────────────────────────────────
if want vision; then
  section "vision — Isaac ROS FoundationPose"

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  else
    miss "NVIDIA GPU/드라이버" "INSTALL.md Step 1"
  fi

  if command -v docker >/dev/null 2>&1; then
    ok "Docker: $(docker --version | cut -d, -f1)"
    if docker ps >/dev/null 2>&1; then
      ok "Docker 데몬 접근"
    else
      miss "Docker 데몬 접근 불가" "sudo usermod -aG docker \$USER 후 재로그인 (INSTALL.md Step 6)"
    fi
    if docker info 2>/dev/null | grep -q "nvidia"; then
      ok "nvidia container runtime"
    else
      miss "nvidia-container-toolkit" "INSTALL.md Step 6"
    fi
  else
    miss "Docker" "INSTALL.md Step 6"
  fi

  if [[ -d "${HOME}/workspaces/isaac_ros-dev/src/isaac_ros_common" ]]; then
    ok "isaac_ros_common (~/workspaces/isaac_ros-dev)"
  else
    miss "Isaac ROS dev 환경" "INSTALL.md Step 7"
  fi

  # ★`camera:` 블록의 position 만 본다. 파일 전체를 grep 하면 `cad_to_body` 의
  # position(회전만 하는 변환이라 0 벡터가 정상)을 플레이스홀더로 오독해
  # "교체 전 실기 구동 금지"라는 틀린 판정을 낸다 (2026-09-01 실측).
  CAM_POS="$(awk '/^camera:/{f=1;next} f&&/^[a-z_]/{f=0} f&&/position:/{print;exit}' \
      "${REPO_DIR}/config/global_camera_extrinsics.yaml" 2>/dev/null)"
  if [[ -z "$CAM_POS" || "$CAM_POS" == *"[0.0, 0.0, 0.0]"* ]]; then
    miss "카메라 extrinsics가 PLACEHOLDER" "실측 캘리브 후 config/global_camera_extrinsics.yaml 교체 (INSTALL.md Step 8) — 교체 전 실기 구동 금지"
  else
    ok "카메라 extrinsics 캘리브됨 (값 교체 확인됨 — 정확성은 별도 검증)"
  fi
fi

# ── 정책 PC (pour inference) ────────────────────────────────────────────────
if want policy; then
  section "policy — pour 정책 추론"

  if python3 -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    ok "torch + CUDA: $(python3 -c 'import torch; print(torch.__version__)')"
  else
    miss "torch(CUDA)" "INSTALL.md Step 4-B"
  fi

  if python3 -c "import rl_games" >/dev/null 2>&1; then
    ok "rl_games"
  else
    miss "rl_games" "pip install rl-games (INSTALL.md Step 4-B)"
  fi

  if python3 -c "
import sys
from pathlib import Path
for p in [Path('${REPO_DIR}').parent / 'hdgp/source/FABRICS/src', Path('${REPO_DIR}').parent / 'repo/FABRICS/src']:
    if p.exists():
        sys.path.insert(0, str(p))
import fabrics_sim" >/dev/null 2>&1; then
    ok "FABRICS (fabrics_sim)"
  else
    miss "FABRICS" "hdgp/source/FABRICS 또는 repo/FABRICS 필요 (INSTALL.md Step 4-B)"
  fi

  # ROS 소스된 셸에서 humble launch_testing 계열 플러그인이 최신 pytest와
  # 충돌하므로 외부 플러그인 자동로드 자체를 끈다 (순수 테스트라 무영향)
  if (cd "${REPO_DIR}/scripts" && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
      test_pour_obs_geometry.py test_pour_obs_builder.py test_palm_fk.py \
      test_pour_action_decoder.py test_cup_pose_relay.py >/dev/null 2>&1); then
    ok "pour s2r 회귀 테스트 (46 pass)"
  else
    miss "pour s2r 회귀 테스트 실패" "cd scripts && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test_pour_*.py test_palm_fk.py test_cup_pose_relay.py -q 로 원인 확인"
  fi
fi

# ── 결과 ────────────────────────────────────────────────────────────────────
printf '\n────────────────────────────────\n'
printf '통과 %d / 누락 %d  (role=%s)\n' "$PASS" "$MISS" "$ROLE"
if [[ $MISS -gt 0 ]]; then
  printf '누락 항목은 INSTALL.md의 표기된 Step을 따라 설치하세요.\n'
  exit 1
fi
printf '이 역할(%s)에 필요한 환경이 모두 준비됐습니다.\n' "$ROLE"
