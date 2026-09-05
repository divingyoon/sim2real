#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BUILD_BRIDGE_ONLY=0
if [[ "${1:-}" == "--bridge-only" ]]; then
  BUILD_BRIDGE_ONLY=1
fi

# The Tesollo drivers are owned by robot_control, which holds the canonical
# Humble snapshot. Its install is an overlay here rather than a copy: building
# both would put two identical package sets on the same search path.
ROBOT_CONTROL_INSTALL="${ROBOT_CONTROL_INSTALL:-${REPO_DIR}/../robot_control/ros_ws/install}"

set +u
source /opt/ros/humble/setup.bash
if [[ -f "${ROBOT_CONTROL_INSTALL}/setup.bash" ]]; then
  source "${ROBOT_CONTROL_INSTALL}/setup.bash"
else
  set -u
  echo "error: robot_control's ROS workspace is not built." >&2
  echo "  expected: ${ROBOT_CONTROL_INSTALL}/setup.bash" >&2
  echo "  build it with robot_control/ros_ws/build.sh, or point" >&2
  echo "  ROBOT_CONTROL_INSTALL at an existing install tree." >&2
  exit 2
fi
set -u

cd "${REPO_DIR}"

if [[ "${BUILD_BRIDGE_ONLY}" -eq 1 ]]; then
  colcon build --base-paths \
    "${REPO_DIR}/isaacsim_bridge"
else
  colcon build --base-paths \
    "${REPO_DIR}/isaacsim_bridge" \
    "${REPO_DIR}/vendor/openarm/openarm_description" \
    "${REPO_DIR}/vendor/openarm/openarm_can" \
    "${REPO_DIR}/vendor/openarm/openarm_hardware" \
    "${REPO_DIR}/vendor/openarm/openarm_bringup"
fi
