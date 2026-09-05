#!/usr/bin/env bash
# sim 성공 궤적 한 개를 **왕복**으로 재생한다: 홈 → 파지·리프트 → 왔던 길로 복귀.
#
# ★★왜 왕복이 한 묶음인가. 재생이 끝나면 팔이 궤적 끝(g1 홈에서 **35°**)에 선다.
#   거기서 전이 bag 을 걸면 브리지가 관절공간 **직선**으로 이어붙는데, 그 직선은
#   테이블을 모른다 — 09.03 에 그걸로 손이 테이블을 쳤다. 그래서 복귀는 반드시
#   **왔던 길**(--reverse)이어야 한다.
#
# ★z 오프셋 30 mm 는 선택이 아니다. 실기 판 0.230 vs sim 0.200 이라 원본 궤적은
#   손끝이 판을 16 mm 파고든다(재생기 --dry 가 막는다). 30 mm 올리면 +12 mm 여유.
#
# 사용 (★컵을 그 궤적의 소환 위치에 먼저 둘 것):
#   ./grasp_traj_round.sh ym05      # 컵 (0.362, −0.210)
#   ./grasp_traj_round.sh y00       # 컵 (0.362, −0.160)
#   ./grasp_traj_round.sh yp05      # 컵 (0.362, −0.110)
#   TIME_SCALE=0.25 ./grasp_traj_round.sh ym05     # 더 천천히
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "사용: $0 <ym05|y00|yp05>" >&2; exit 1
fi
TAG="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2R="$(cd "${HERE}/../.." && pwd)"
TRAJ="/home/user/rl_ws/hdgp/log/grasp_traj/g1/g1_${TAG}.hdf5"
[[ -f "${TRAJ}" ]] || { echo "궤적이 없다: ${TRAJ}"; exit 1; }

TIME_SCALE="${TIME_SCALE:-0.35}"      # 낮을수록 느리다(0.5 → 19.9 s, 0.35 → 28.4 s)
Z_OFFSET="${Z_OFFSET:-0.030}"
EXEC="--execute"
[[ "${DRY:-0}" == "1" ]] && EXEC="--dry"

set +u
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${S2R}/.venv/bin/activate"
set -u
cd "${S2R}"

# ── 전제 확인 — 브링업마다 사라지는 것들 ────────────────────────────────
if ! ros2 control list_controllers 2>/dev/null | grep -q "right_forward_effort_controller"; then
    echo "[round] effort 컨트롤러 로드"
    "${S2R}/../robot_control/ros_ws/load_effort_controllers.sh" right >/dev/null 2>&1 || true
fi
pgrep -f gravity_comp_node.py >/dev/null \
    || { echo "★중력보상이 안 돈다 — 먼저 켤 것:"; \
         echo "   python3 scripts/nodes/gravity_comp_node.py --scale 1.0 \\"; \
         echo "     --payload 0.9130,-0.00450,-0.01723,0.22147 --execute"; exit 1; }
python3 scripts/ops/apply_hand_gains.py --execute >/dev/null 2>&1 || \
    { echo "★손 게인 적용 실패 — 테솔로 드라이버 확인"; exit 1; }

echo "[round] ${TAG} · time-scale ${TIME_SCALE} · z +${Z_OFFSET}"
echo "[round] ① 정방향 — 파지·리프트"
python3 scripts/ops/grasp_traj_replay.py --traj "${TRAJ}" \
    --z-offset "${Z_OFFSET}" --time-scale "${TIME_SCALE}" \
    --log "logs/policy/traj_${TAG}_fwd.csv" ${EXEC}

echo "[round] ② 역방향 — 왔던 길로 복귀"
python3 scripts/ops/grasp_traj_replay.py --traj "${TRAJ}" \
    --z-offset "${Z_OFFSET}" --time-scale "${TIME_SCALE}" --reverse \
    --log "logs/policy/traj_${TAG}_rev.csv" ${EXEC}

echo "[round] ${TAG} 완료 — 팔은 g1 홈이다. 다음 궤적은 컵을 옮긴 뒤 실행할 것."
