#!/usr/bin/env bash
# 우팔 s2r 한 라운드: 인식 → 전이(≤20 s) → 정책(파지·리프트) → 역순 복귀 → 차렷
#
# ★★왜 스크립트로 묶는가. 09.03 실기에서 순서를 하나 빠뜨릴 때마다 사고가 났다:
#   · `--arm-only` 로 전이 → 손이 펴진 채 이동 → **테이블에 걸림**
#   · 중력보상 없이 전이 → 홈에 6.18° 미달
#   · 정책 정상종료 후 복귀 없음 → 다음 라운드가 정착 거부로 막힘
#   순서를 손으로 치면 또 빠뜨린다. 여기 한 곳에만 둔다.
#
# ★전이는 **20 s 이내**로 돈다(사용자 규약). 느릴수록 모터가 그 자세를 오래 버텨
#   발열이 커진다 — 특히 우 j7 은 홈에서 3.17 N·m 로 과열 이력이 있다.
#
# 사용:
#   ./right_grasp_round.sh                     # 인식값으로 한 라운드
#   ./right_grasp_round.sh 0.401,-0.183,0.271  # 좌표 직접 지정
#   DRY=1 ./right_grasp_round.sh               # 발행 없이 순서만 확인
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2R="$(cd "${HERE}/../.." && pwd)"
RUN="${RUN:-${S2R}/logs/policy/right_g1}"
BAG="${S2R}/logs/shadow/reset_both"
STEPS="${STEPS:-250}"
HZ="${HZ:-20}"
# ★`rate_scale` 은 (0,1] 감속 전용이라 1.0 이 최속(29.5 s)이고 20 s 를 못 맞춘다.
#   그래서 bag 을 2프레임마다 뽑은 `*_fast` 를 쓴다 — 737 프레임 · 50 Hz · **14.7 s**,
#   최대 관절속도 0.50 rad/s(프로필 한계 2.0 의 25%)로 여전히 완만하다.
RATE_SCALE="${RATE_SCALE:-1.0}"
# ★★z 보정 — 09.05 정정. 실기 테이블 = **0.205**(줄자·Fusion CAD, sim env_v1 도 0.205).
#   카메라 사슬(보드 평면 0.2301·hand-eye 0.2264·FP++ 컵 0.2269)은 테이블을 +21~25 mm
#   높게 본다 — FP++ 자체가 아니라 base→head 고정변환(마운트 0.750 의심, B4 실측 대기)
#   의 datum 오차이고 hand-eye 잔차에는 나타나지 않는다. 09.03 의 "0.2075 는 미검증
#   가정"은 틀렸다(사용자 09.02 줄자값). 로봇 짚기 0.231~0.245 는 TCP 값이고 손끝(+15.4 mm)
#   보정 시 0.229 — 팔 사슬도 +24 mm 로, 원인(기둥 높이/손가락 길이)은 실측 대기.
#   정책 obs 는 상대량이라 z 편향에 둔감하다. 편향이 head 로 확정되면 보정은 여기가
#   아니라 URDF HEAD_MOUNT_XYZ + camera 블록 재생성으로 한다. 그때까지 0.
CUP_Z_BIAS="${CUP_Z_BIAS:-0.0}"
EXEC="--execute"
[[ "${DRY:-0}" == "1" ]] && EXEC=""

# ★ROS·venv setup 은 미정의 변수를 쓴다 — 소싱 동안만 `-u` 를 끈다.
set +u
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${S2R}/.venv/bin/activate"
set -u
cd "${S2R}"

# ── 1. 컵 인식 ──────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    CUP="$1"
    echo "[round] 컵 좌표 지정: ${CUP}"
else
    echo "[round] 컵 인식 중 (perception, ROS_DOMAIN_ID=126)…"
    CUP="$(ROS_DOMAIN_ID=126 python3 "${HERE}/read_object_pose.py" \
           --topic /objects/cup_big_s100/pose --z-bias "${CUP_Z_BIAS}")"
    echo "[round] 컵 = ${CUP}  (z 보정 ${CUP_Z_BIAS} 반영)"
fi

# ── 2. 중력보상 (우팔은 이거 없이 홈에 못 간다) ────────────────────────
# ★기본 그룹이 우팔이라 `--group` 없이 뜰 수 있다 — 패턴을 좁히면 못 찾는다.
if pgrep -f "gravity_comp_node.py" >/dev/null 2>&1; then
    echo "[round] 중력보상 이미 실행 중 — 재사용"
else
    echo "[round] 우팔 중력보상 기동"
    # 이미 active 면 로더가 에러를 낸다 — 그건 실패가 아니다.
    "${S2R}/../robot_control/ros_ws/load_effort_controllers.sh" right >/dev/null 2>&1 || true
    python3 scripts/nodes/gravity_comp_node.py --scale 1.0 \
        --payload 0.9130,-0.00450,-0.01723,0.22147 --execute &
    GRAV_PID=$!
    trap 'kill -INT ${GRAV_PID} 2>/dev/null || true' EXIT
    sleep 3
fi

# ── 3. 전이: 차렷 → g1 홈 (★손도 함께 — 주먹 경유) ────────────────────
echo "[round] 전이 (fast bag 737프레임 ≈ 14.7 s · 손 포함)"
python3 scripts/nodes/shadow_replay.py --sim "${BAG}/reset_right_v2_fast.npz" \
    --robot tesollo_bi_s__right --with-hand --allow-idle-arm-mismatch \
    --rate-scale "${RATE_SCALE}" ${EXEC}

# ── 4. 정책 (역순 복귀 내장) ────────────────────────────────────────────
echo "[round] 정책 ${STEPS} 스텝 @ ${HZ} Hz"
python3 scripts/right_inference_node.py --run "${RUN}" \
    --steps "${STEPS}" --policy-hz "${HZ}" --object "${CUP}" \
    --csv "logs/policy/round_$(date +%H%M%S).csv" ${EXEC}

# ── 5. 차렷 복귀 (★역순 bag — 손 주먹 경유) ────────────────────────────
echo "[round] 차렷 복귀"
python3 scripts/nodes/shadow_replay.py --sim "${BAG}/reset_right_v2_reverse_fast.npz" \
    --robot tesollo_bi_s__right --with-hand --allow-idle-arm-mismatch \
    --rate-scale "${RATE_SCALE}" ${EXEC}

echo "[round] 완료 — 컵 위치를 바꾸고 다시 실행하면 다음 라운드"
