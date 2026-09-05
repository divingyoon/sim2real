# grasp-v1 sim2real 연계 (카메라 무관 플러밍/설정) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** vision-3090 D435i 설치 전에, perception(FP++ 베이스)과 sim2real grasp-v1을 잇는 카메라·로봇 무관 부분 — `/cup_pose` 계약 통일, T_cad_body 정합, DDS 도메인, dry-run obs 검증 — 을 완성해 카메라가 오면 바로 꽂히게 한다.

**Architecture:** `cup_pose_relay.py`는 이미 순수로직/ROS 노드가 분리돼 있고 출력(`/cup_pose` = PoseStamped/base)은 통일돼 있다. 통일이 필요한 건 **입력**: 현재 Isaac ROS FP의 `Detection3DArray` 전용 → FP++가 낼 camera-frame `PoseStamped`도 받도록 소스 어댑터를 추가한다. T_cad_body(cup.obj↔sim body)는 카메라 무관이라 지금 실측 도출한다. DDS는 이미 domain 126로 합의돼 있어 검증·문서화만. dry-run은 기존 `sim2real_dryrun.py`에 합성 `/cup_pose` 주입으로 grasp-v1 obs 조립을 검증한다.

**Tech Stack:** Python 3.10, numpy, PyYAML, pytest, ROS2 Humble(rclpy, geometry_msgs, vision_msgs), CycloneDDS.

## Global Constraints

- hdgp 는 **READ-ONLY** (읽기만, 수정 금지). 편집은 sim2real / perception 에서.
- 쿼터니언은 전부 **wxyz** 순서(코드 규약 — cup_pose_relay/pour_obs_builder 준수).
- `ROS_DOMAIN_ID=126` — 모든 PC 동일(INSTALL.md Step 2).
- `/cup_pose` = `geometry_msgs/PoseStamped`, frame_id = base_frame(기본 `base_link`). 다운스트림(sim2real_inference, pour_inference) 무수정.
- 순수 로직(numpy/yaml)과 ROS 노드는 파일 내 분리 유지 — 테스트는 ROS 없이 순수 로직 대상.
- 카메라/로봇 의존(extrinsics T_base_cam 캘리브, 라이브 grasp 동작)은 **범위 밖**(하드웨어 대기).

---

### Task 1: `/cup_pose` 입력 소스 어댑터 — camera-frame PoseStamped 수용

`cup_pose_relay`가 FP++(camera-frame `PoseStamped`)와 Isaac ROS FP(`Detection3DArray`)를 모두 입력으로 받게 한다. 변환 체인·출력은 그대로. 순수 함수 `pose_msg_to_candidate`를 추가하고 노드에 `--in-type` 스위치를 둔다.

**Files:**
- Modify: `sim2real/scripts/cup_pose_relay.py` (순수 로직 블록 + main 노드)
- Test: `sim2real/scripts/test_cup_pose_relay.py`

**Interfaces:**
- Consumes: 기존 `select_best_detection`, `cad_pose_to_base_body`, `Extrinsics`.
- Produces: `posestamped_to_candidate(px, py, pz, qw, qx, qy, qz, score=1.0) -> tuple[float, np.ndarray, np.ndarray]` — camera-frame pose를 relay 후보 튜플 `(score, pos, quat_wxyz)`로. Detection3DArray 경로와 동일 후보 포맷.

- [ ] **Step 1: 실패 테스트 작성** — `test_cup_pose_relay.py` 끝에 추가

```python
def test_posestamped_to_candidate_maps_fields():
    from cup_pose_relay import posestamped_to_candidate
    # camera-frame pose (pos, quat wxyz), score 기본 1.0
    score, pos, quat = posestamped_to_candidate(0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)
    assert score == 1.0
    assert np.allclose(pos, [0.1, 0.2, 0.3])
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_posestamped_candidate_feeds_same_transform_as_detection():
    # 같은 camera-frame pose면 Detection3DArray 경로와 동일한 base 결과여야 한다
    from cup_pose_relay import posestamped_to_candidate, cad_pose_to_base_body
    ext = _ident_ext(cam_pos=np.array([1.0, 0.0, 0.5]), cam_quat=ROT_Z_90)
    _, pos_c, quat_c = posestamped_to_candidate(0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    pos, quat = cad_pose_to_base_body(ext, pos_c, quat_c)
    assert np.allclose(pos, [1.0, 0.2, 0.5], atol=1e-12)
```

- [ ] **Step 2: 실패 확인**

Run: `cd sim2real/scripts && python -m pytest test_cup_pose_relay.py -k posestamped -v`
Expected: FAIL — `ImportError: cannot import name 'posestamped_to_candidate'`

- [ ] **Step 3: 최소 구현** — `cup_pose_relay.py` 순수 로직 블록(`select_best_detection` 아래)에 추가

```python
def posestamped_to_candidate(
    px: float, py: float, pz: float,
    qw: float, qx: float, qy: float, qz: float,
    score: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """camera-frame PoseStamped 필드 → relay 후보 (score, pos, quat_wxyz).

    FP++ 라이브 노드가 낼 camera optical 프레임 컵 pose 입력 경로.
    Detection3DArray 경로와 동일한 (score, pos, quat) 포맷을 반환한다.
    """
    return (
        float(score),
        np.array([px, py, pz], dtype=np.float64),
        np.array([qw, qx, qy, qz], dtype=np.float64),
    )
```

- [ ] **Step 4: 통과 확인**

Run: `cd sim2real/scripts && python -m pytest test_cup_pose_relay.py -k posestamped -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 노드에 `--in-type` 스위치 추가** — `main()`의 argparse + 구독 분기 수정

`main()` 안 argparse에 추가:
```python
    parser.add_argument("--in-type", choices=["detection3d", "posestamped"],
                        default="detection3d",
                        help="입력 메시지 타입: Isaac ROS FP=detection3d, FP++=posestamped")
```
`rclpy` import 아래에 `PoseStamped`는 이미 import됨. `Detection3DArray` import를 조건부로 두고, 구독 분기를 `CupPoseRelay.__init__` 안에서:
```python
            if args.in_type == "posestamped":
                self.create_subscription(
                    PoseStamped, args.in_topic, self._posestamped_cb, 10)
            else:
                from vision_msgs.msg import Detection3DArray
                self.create_subscription(
                    Detection3DArray, args.in_topic, self._detections_cb, 10)
```
그리고 콜백 추가(클래스 메서드):
```python
        def _posestamped_cb(self, msg) -> None:
            p, q = msg.pose.position, msg.pose.orientation  # ROS xyzw
            cand = posestamped_to_candidate(p.x, p.y, p.z, q.w, q.x, q.y, q.z)
            best = select_best_detection([cand], args.min_score)
            if best is None:
                return
            _, cad_pos, cad_quat = best
            pos, quat = cad_pose_to_base_body(ext, cad_pos, cad_quat)
            self._publish(msg.header.stamp, pos, quat)
```
`_detections_cb` 끝의 PoseStamped 조립부를 공용 `_publish(stamp, pos, quat)` 헬퍼로 추출(중복 제거, DRY):
```python
        def _publish(self, stamp, pos, quat) -> None:
            out = PoseStamped()
            out.header.stamp = stamp
            out.header.frame_id = ext.base_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = pos
            out.pose.orientation.w = quat[0]
            out.pose.orientation.x = quat[1]
            out.pose.orientation.y = quat[2]
            out.pose.orientation.z = quat[3]
            self._pub.publish(out)
            self._published += 1
            if self._published % 30 == 1:
                self.get_logger().info(
                    f"cup_pose #{self._published}: pos={np.round(pos, 3).tolist()}")
```
`_detections_cb`의 마지막 부분을 `self._publish(msg.header.stamp, pos, quat)`로 교체.

- [ ] **Step 6: 순수 로직 회귀 확인** (노드 편집이 순수 테스트를 깨지 않았는지)

Run: `cd sim2real/scripts && python -m pytest test_cup_pose_relay.py -v`
Expected: PASS (기존 + 신규 전부)

- [ ] **Step 7: 커밋**

```bash
cd sim2real && git add scripts/cup_pose_relay.py scripts/test_cup_pose_relay.py
git commit -m "feat(relay): /cup_pose 입력을 FP++ camera-frame PoseStamped 도 수용 (소스 무관)"
```

---

### Task 2: T_cad_body 도출 — cup.obj ↔ sim 컵 body 프레임 정합

extrinsics yaml의 `cad_to_body`(현재 PLACEHOLDER identity)를 실측으로 채운다. cup.obj는 **Y축이 높이(17.76cm)**, sim body는 **+z=위, 원점=바닥 중심**. 회전(Y-up→Z-up) + 원점 이동을 도출하는 순수 함수와, hdgp의 sim 컵 body 정의를 읽어 확정하는 검증 단계로 나눈다.

**Files:**
- Create: `sim2real/scripts/calib/cad_body_alignment.py` (순수 도출 로직)
- Create: `sim2real/scripts/test_cad_body_alignment.py`
- Modify: `sim2real/config/global_camera_extrinsics.yaml` (cad_to_body 값)
- Read-only 참조: `hdgp/assets/cup/*.stl`, hdgp 컵 body/USD 정의 (READ-ONLY)

**Interfaces:**
- Consumes: `pour_obs_geometry.quat_apply` (검증용), numpy.
- Produces:
  - `mesh_aabb(obj_path: str) -> tuple[np.ndarray, np.ndarray]` — (min_xyz, max_xyz).
  - `cad_to_body_yup_to_zup(aabb_min, aabb_max) -> tuple[np.ndarray, np.ndarray]` —
    (pos_xyz, quat_wxyz): mesh(Y-up, 원점 임의) → body(Z-up, 원점 바닥중심) 변환.

- [ ] **Step 1: 실패 테스트 작성** — `test_cad_body_alignment.py`

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cad_body_alignment import mesh_aabb, cad_to_body_yup_to_zup  # noqa: E402
from pour_obs_geometry import quat_apply  # noqa: E402

CUP_OBJ = str(Path(__file__).resolve().parents[2] / "perception/assets/Cup/cup.obj")

def test_mesh_aabb_matches_measured_cup():
    mn, mx = mesh_aabb(CUP_OBJ)
    assert np.allclose(mn, [-0.0463, -0.0773, -0.0440], atol=1e-3)
    assert np.allclose(mx, [ 0.0437,  0.1003,  0.0460], atol=1e-3)

def test_yup_to_zup_rotation_maps_mesh_Y_to_body_Z():
    _, quat = cad_to_body_yup_to_zup(
        np.array([-0.0463, -0.0773, -0.0440]),
        np.array([ 0.0437,  0.1003,  0.0460]))
    # mesh +Y(높이축)가 body +Z 로 가야 한다
    assert np.allclose(quat_apply(quat, [0, 1, 0]), [0, 0, 1], atol=1e-9)

def test_yup_to_zup_translation_puts_bottom_center_at_origin():
    pos, quat = cad_to_body_yup_to_zup(
        np.array([-0.0463, -0.0773, -0.0440]),
        np.array([ 0.0437,  0.1003,  0.0460]))
    # mesh 바닥면(y=min)의 x/z 중심점이 변환 후 body 원점(≈0,0,0) 근처
    bottom_center_mesh = np.array([(-0.0463 + 0.0437) / 2, -0.0773,
                                   (-0.0440 + 0.0460) / 2])
    mapped = np.array(quat_apply(quat, bottom_center_mesh)) + pos
    assert np.allclose(mapped, [0.0, 0.0, 0.0], atol=1e-6)
```

- [ ] **Step 2: 실패 확인**

Run: `cd sim2real/scripts && python -m pytest test_cad_body_alignment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cad_body_alignment'`

- [ ] **Step 3: 최소 구현** — `cad_body_alignment.py`

```python
"""cup.obj(mesh) ↔ sim 컵 body 프레임 정합. 순수 numpy — ROS 불필요.

mesh 규약: Y축이 높이(측정: x/z=9cm, y=17.76cm), 원점 임의(바닥이 y=min).
body 규약: +z=위, 원점=컵 바닥 중심 (extrinsics yaml 주석).
따라서 T_cad_body = (mesh를 Y-up→Z-up 회전) 후 (바닥중심을 원점으로 이동).
"""
from __future__ import annotations
import numpy as np


def mesh_aabb(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo.copy()
    with open(obj_path) as f:
        for ln in f:
            if ln.startswith("v "):
                xyz = np.array([float(v) for v in ln.split()[1:4]])
                lo = np.minimum(lo, xyz)
                hi = np.maximum(hi, xyz)
    return lo, hi


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def cad_to_body_yup_to_zup(
    aabb_min: np.ndarray, aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pos_xyz, quat_wxyz): mesh(Y-up) → body(Z-up, 원점=바닥중심)."""
    # Y-up→Z-up: x축 기준 +90° 회전 (Y→Z, Z→-Y)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    quat = np.array([c, s, 0.0, 0.0])  # wxyz, Rx(+90°)
    # mesh 바닥면 중심 (x/z 중앙, y=min)
    bottom_center = np.array([(aabb_min[0] + aabb_max[0]) / 2,
                              aabb_min[1],
                              (aabb_min[2] + aabb_max[2]) / 2])
    # 회전 후 그 점이 원점에 오도록 평행이동: pos = -R·bottom_center
    def rot(q, v):
        qv = np.array([0.0, *v])
        qc = np.array([q[0], -q[1], -q[2], -q[3]])
        return _quat_mul(_quat_mul(q, qv), qc)[1:]
    pos = -rot(quat, bottom_center)
    return pos, quat
```

- [ ] **Step 4: 통과 확인**

Run: `cd sim2real/scripts && python -m pytest test_cad_body_alignment.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: sim 컵 body 규약 대조 (hdgp READ-ONLY)** — 값 확정 전 sanity

Run:
```bash
grep -rn "rim\|height\|cup.*z\|0.10\|metersPerUnit" \
  hdgp/source/openarm/openarm/tesollo/right/grasp_v1/ 2>/dev/null | head
ls hdgp/assets/cup/
```
확인할 것: sim 컵 body 의 +z=위·원점=바닥중심 규약과 실제 높이. cup.obj(17.76cm)와
sim 컵 크기가 다르면(예: rim +z=0.100=10cm) **스케일 불일치** — 이때 body는 pose
프레임 정의(축·원점)만 맞추면 되고 물리 크기 차이는 grasp 정책이 접촉-게이트로 흡수
(메모리 grasp-v2-pos-only-obs 참조). 회전/원점 규약만 위 도출과 일치하는지 확인.
Expected: sim body +z=위, 원점=바닥중심 확인 → 도출된 quat=Rx(+90°) 유효.
(규약이 다르면 `cad_to_body_yup_to_zup`의 회전축을 그 규약에 맞춰 수정 후 Step 3~4 반복.)

- [ ] **Step 6: yaml 값 산출·기입** — 스크립트로 실수 방지

Run:
```bash
cd sim2real/scripts && python - <<'PY'
from cad_body_alignment import mesh_aabb, cad_to_body_yup_to_zup
import numpy as np
mn, mx = mesh_aabb("../../perception/assets/Cup/cup.obj")
pos, quat = cad_to_body_yup_to_zup(mn, mx)
print("cad_to_body.position:", [round(float(x), 6) for x in pos])
print("cad_to_body.orientation_wxyz:", [round(float(x), 6) for x in quat])
PY
```
출력 값을 `config/global_camera_extrinsics.yaml`의 `cad_to_body:` 블록에 기입
(position, orientation_wxyz). `camera:`(T_base_cam)는 PLACEHOLDER 유지 — 캘리브(범위 밖).
주석에 "cad_body_alignment.py 도출, cup.obj Y-up→body Z-up" 한 줄 추가.

- [ ] **Step 7: relay 회귀 확인** — 새 cad_to_body가 relay 로드/변환을 안 깨는지

Run: `cd sim2real/scripts && python -m pytest test_cup_pose_relay.py::test_default_yaml_loads_and_validates -v`
Expected: PASS

- [ ] **Step 8: 커밋**

```bash
cd sim2real && git add scripts/calib/cad_body_alignment.py scripts/test_cad_body_alignment.py config/global_camera_extrinsics.yaml
git commit -m "feat(extrinsics): T_cad_body 실측 도출 (cup.obj Y-up→sim body Z-up)"
```

---

### Task 3: DDS 도메인 — vision ↔ 로봇 제어 PC 연계 검증·문서화

domain 126은 이미 양측 합의. vision PC(arm3070/vision-3090)와 로봇 제어 PC가 `/cup_pose`를 주고받도록 도메인·전송 설정을 검증하는 스크립트와 런북 한 절을 만든다. (신규 브로커 없음 — 확인·문서화 태스크.)

**Files:**
- Create: `sim2real/scripts/check_cup_pose_link.sh` (양측 도메인/토픽 가시성 점검)
- Modify: `sim2real/docs/superpowers/specs/2026-07-21-grasp-v1-live-bringup-design.md` (연계 절 추가)

**Interfaces:**
- Consumes: 환경변수 `ROS_DOMAIN_ID`, `ros2 topic` CLI.
- Produces: `check_cup_pose_link.sh` — 종료코드 0=도메인 일치+`/cup_pose` 가시, 1=불일치.

- [ ] **Step 1: 점검 스크립트 작성** — `check_cup_pose_link.sh`

```bash
#!/usr/bin/env bash
# vision ↔ 로봇 제어 PC 간 /cup_pose 연계 점검. 각 PC 에서 실행.
#   기대: ROS_DOMAIN_ID=126 동일 + (vision 기동 상태면) /cup_pose 가 보인다.
set -uo pipefail
EXPECT_DOMAIN="${1:-126}"
fail=0
if [[ "${ROS_DOMAIN_ID:-unset}" != "$EXPECT_DOMAIN" ]]; then
  echo "[FAIL] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} (기대 $EXPECT_DOMAIN) — export ROS_DOMAIN_ID=$EXPECT_DOMAIN"
  fail=1
else
  echo "[OK] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
fi
echo "[info] RMW=${RMW_IMPLEMENTATION:-default} CYCLONEDDS_URI=${CYCLONEDDS_URI:-unset}"
if ros2 topic list 2>/dev/null | grep -qx /cup_pose; then
  echo "[OK] /cup_pose 가시 — 타입: $(ros2 topic type /cup_pose 2>/dev/null)"
else
  echo "[warn] /cup_pose 미가시 (vision 노드 미기동이면 정상)"
fi
exit $fail
```

- [ ] **Step 2: 실행 검증** — 로봇 제어 PC(또는 현재 셸)에서

Run:
```bash
cd sim2real/scripts && chmod +x check_cup_pose_link.sh
ROS_DOMAIN_ID=126 ./check_cup_pose_link.sh 126; echo "exit=$?"
```
Expected: `[OK] ROS_DOMAIN_ID=126` + exit=0 (그 외 환경은 `/cup_pose` warn 허용).

- [ ] **Step 3: 도메인 불일치 감지 확인**

Run: `cd sim2real/scripts && ROS_DOMAIN_ID=7 ./check_cup_pose_link.sh 126; echo "exit=$?"`
Expected: `[FAIL] ROS_DOMAIN_ID=7 ...` + exit=1

- [ ] **Step 4: 런북 절 추가** — grasp-v1 spec 하단에

`sim2real/docs/superpowers/specs/2026-07-21-grasp-v1-live-bringup-design.md`의 §7 아래
"## 8. 연계 토폴로지(카메라 무관)" 절 추가:
```markdown
## 8. 연계 토폴로지 (카메라 무관)
- vision PC(arm3070 현행 → vision-3090 향후)와 로봇 제어 PC는 **ROS_DOMAIN_ID=126** 공유.
  둘 다 `scripts/check_cup_pose_link.sh 126` 로 도메인·`/cup_pose` 가시성 점검.
- `/cup_pose` 계약: geometry_msgs/PoseStamped, base 프레임. 생산자 = `cup_pose_relay.py`
  (`--in-type detection3d`=Isaac ROS FP / `--in-type posestamped`=FP++). 소비자 무수정.
- Tailscale 은 SSH 용(멀티캐스트 불가) — DDS 통신은 동일 LAN/DDS 설정에 의존.
```

- [ ] **Step 5: 커밋**

```bash
cd sim2real && git add scripts/check_cup_pose_link.sh docs/superpowers/specs/2026-07-21-grasp-v1-live-bringup-design.md
git commit -m "feat(link): /cup_pose 도메인 연계 점검 스크립트 + 토폴로지 문서"
```

---

### Task 4: dry-run obs 검증 — 합성 `/cup_pose`로 grasp-v1 obs 조립 확인

`sim2real_dryrun.py`는 이미 `/cup_pose`를 선택 구독한다. 카메라·로봇 없이, 합성 `/cup_pose`를 발행하는 작은 노드로 grasp-v1 obs(cup-relative 항)가 컵 위치를 실제로 반영하는지 검증한다. (grasp-v1 spec Stage 0의 카메라 무관 부분.)

**Files:**
- Create: `sim2real/scripts/fakes/fake_cup_pose_pub.py` (합성 /cup_pose 발행 — 정지/원운동)
- Read-only 참조: `sim2real/scripts/deprecated/sim2real_dryrun.py`, `sim2real_inference.py` (obs 조립)

**Interfaces:**
- Consumes: rclpy, geometry_msgs/PoseStamped.
- Produces: `fake_cup_pose_pub.py` — `--x --y --z`(정지) 또는 `--orbit r,hz`(원운동)로
  `/cup_pose` 발행. 다운스트림(dryrun/inference)이 실제 카메라 없이 obs 검증 가능.

- [ ] **Step 1: 합성 발행 노드 작성** — `fake_cup_pose_pub.py`

```python
#!/usr/bin/env python3
"""합성 /cup_pose 발행 (카메라 무관 obs 검증용).

정지: --x 0.40 --y -0.15 --z 0.38
원운동(움직이는 컵 모사): --orbit 0.05,0.2  (반경 5cm, 0.2Hz, 중심=xyz)
"""
from __future__ import annotations
import argparse, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.40)
    ap.add_argument("--y", type=float, default=-0.15)
    ap.add_argument("--z", type=float, default=0.38)
    ap.add_argument("--frame", default="base_link")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--orbit", default="", help="r,hz (예: 0.05,0.2). 빈값=정지")
    args = ap.parse_args()
    r = hz = 0.0
    if args.orbit:
        r, hz = (float(v) for v in args.orbit.split(","))

    rclpy.init()
    node = Node("fake_cup_pose_pub")
    pub = node.create_publisher(PoseStamped, "/cup_pose", 10)
    t0 = node.get_clock().now()

    def tick():
        t = (node.get_clock().now() - t0).nanoseconds * 1e-9
        dx = r * math.cos(2 * math.pi * hz * t)
        dy = r * math.sin(2 * math.pi * hz * t)
        m = PoseStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = args.frame
        m.pose.position.x = args.x + dx
        m.pose.position.y = args.y + dy
        m.pose.position.z = args.z
        m.pose.orientation.w = 1.0
        pub.publish(m)

    node.create_timer(1.0 / args.rate, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 구문/기동 확인** (ROS 환경에서)

Run: `cd sim2real/scripts && python -c "import ast; ast.parse(open('fake_cup_pose_pub.py').read()); print('OK')"`
Expected: `OK` (구문). ROS 환경이면 추가로 `python fake_cup_pose_pub.py --x 0.4 --y -0.15 --z 0.38 &` 후 `ros2 topic echo /cup_pose --once`로 발행 확인.

- [ ] **Step 3: dry-run obs 반영 검증** (ROS 환경 필요 — 로봇 제어 PC)

Run (별도 셸들):
```bash
# 정지 컵
python fake_cup_pose_pub.py --x 0.40 --y -0.15 --z 0.38 &
# dry-run (RViz 없이 obs 로깅만 확인용으로 기동)
python sim2real_dryrun.py --agent <agent.yaml> --ckpt <ckpt.pth> \
    --cup_x 0.40 --cup_y -0.15 --cup_z 0.38
ros2 service call /sim2real/start std_srvs/srv/Trigger
```
확인: dry-run 로그의 cup-relative obs 항이 `/cup_pose`(0.40,-0.15,0.38)를 반영.
그다음 `--orbit 0.05,0.2`로 재발행 → obs의 cup 항이 원운동 추종하는지 확인.
Expected: 합성 /cup_pose 변화가 obs cup-relative 항에 실시간 반영.
(로봇/시뮬 없는 셸이면 이 스텝은 하드웨어 대기 — Step 2까지가 카메라 무관 게이트.)

- [ ] **Step 4: 커밋**

```bash
cd sim2real && git add scripts/fakes/fake_cup_pose_pub.py
git commit -m "feat(dryrun): 합성 /cup_pose 발행 노드 (카메라 무관 grasp-v1 obs 검증)"
```

---

### Task 5: 로봇 자료 매니페스트 + 디렉토리 연계 문서

로봇 PC·카메라 연결 시 "바로 사용"하도록, 필요한 **로봇용 자료**(정책 체크포인트, 드라이버/vendor, launch)와 **perception ↔ sim2real ↔ hdgp/log 디렉토리 연계**를 문서 하나로 정리한다. 사용자가 나중에 찾아 sim2real에 clone/배치할 목록이 된다. (문서 태스크 — 코드 없음. hdgp READ-ONLY 탐색.)

**Files:**
- Create: `sim2real/docs/ROBOT_MATERIALS.md`
- Read-only 참조: `hdgp/log`, `hdgp/outputs`, `hdgp/source/openarm/openarm/tesollo/right/grasp_v1/`, `sim2real/scripts/deprecated/sim2real_inference.py`, `sim2real_dryrun.py`, `policy_loader.py`

**Interfaces:**
- Consumes: 없음(문서). Produces: 사용자용 clone/배치 체크리스트.

- [ ] **Step 1: 정책 산출물 위치 조사 (hdgp READ-ONLY)**

Run:
```bash
find hdgp/log hdgp/outputs -name "*.pth" 2>/dev/null | grep -i "grasp.*right\|5g_grasp" | head
find hdgp -name "agent.yaml" 2>/dev/null | grep -i grasp | head
grep -n "ckpt\|agent\|\.pth\|load" sim2real/scripts/policy_loader.py | head
```
grasp-v1(=5g_grasp_right-v7) 체크포인트(.pth) + agent.yaml(params) 경로를 확정한다.

- [ ] **Step 2: 로봇 드라이버/vendor 목록 조사**

Run:
```bash
grep -rn "isaacsim_bridge\|openarm_control\|dg5f_right\|contact_forces\|/joint_states" \
  sim2real/scripts/deprecated/sim2real_inference.py sim2real/scripts/deprecated/sim2real_dryrun.py | head -20
```
sim2real가 의존하는 ROS2 패키지(openarm 팔 드라이버, Tesollo dg5f_right, isaacsim_bridge,
openarm_control launch)와 구독/발행 토픽을 목록화.

- [ ] **Step 3: `ROBOT_MATERIALS.md` 작성**

다음 4개 절로 구성:
```markdown
# 로봇용 자료 매니페스트 (연결 시 바로 사용)

## 1. 정책 산출물 (hdgp/log → sim2real)
- grasp-v1 체크포인트: <Step1에서 확정한 .pth 경로> → sim2real 실행 시 --ckpt 로 지정
- agent.yaml(params): <경로> → --agent
- **연계**: hdgp 학습 산출물. sim2real_inference/dryrun 이 로드. sim2real 로 복사 or 절대경로 참조.

## 2. 로봇 드라이버/vendor (로봇 제어 PC, clone 대상)
- OpenArm 팔 드라이버 + openarm_control (launch: openarm_..._real.launch.py)
- Tesollo dg5f_right 드라이버 (/dg5f_right/joint_states, /dg5f_right/contact_forces 발행)
- isaacsim_bridge (/isaacsim/right_arm_cmd, right_hand_cmd 소비 → 컨트롤러)
- **위치**: 로봇제어 레포(사용자 보유). sim2real 단일 디렉토리 원칙에서 vendor 는 예외.

## 3. 비전 (perception → sim2real)
- perception 이 /cup_pose 생산(FP++ 베이스 / Isaac ROS FP 현행) → cup_pose_relay → sim2real.
- ROS_DOMAIN_ID=126 공유. check_cup_pose_link.sh 로 점검.

## 4. 디렉토리 유기적 연계 (데이터 흐름)
    perception(/cup_pose) ─┐
    hdgp/log(정책 .pth) ───┼─▶ sim2real (relay·inference·dryrun) ─▶ 로봇 드라이버(vendor)
    hdgp(READ-ONLY 규약) ──┘
- sim2real = 런타임 글루. 코드는 sim2real 단일 디렉토리. 정책은 hdgp/log, 비전은 perception,
  드라이버는 vendor 레포에서 온다.

## 5. 연결 시 체크리스트
- [ ] 정책 .pth/agent.yaml 을 sim2real 에 배치(or 경로 지정)
- [ ] 로봇 드라이버 vendor clone + 기동
- [ ] perception /cup_pose 기동 + ROS_DOMAIN_ID=126 확인(check_cup_pose_link.sh)
- [ ] extrinsics 캘리브(T_base_cam) — 카메라 장착 후
```
각 <경로>는 Step 1~2 실측값으로 채운다(빈칸/TBD 금지).

- [ ] **Step 4: 커밋**

```bash
cd sim2real && git add docs/ROBOT_MATERIALS.md
git commit -m "docs: 로봇 자료 매니페스트 + perception/sim2real/hdgp-log 디렉토리 연계"
```

---

## 범위 밖 (하드웨어 대기)
- T_base_cam extrinsics **캘리브레이션** (perception `tools/calibrate_extrinsics.py`, ArUco) — 카메라 장착 후.
- FP++ **라이브 ROS 노드**(camera→/cup_pose) — vision-3090 D435i + py3.8↔Humble rclpy 브리지(별도 결정).
- 감독 하 **라이브 grasp 동작**(grasp-v1 spec Stage 2~4).

## Self-Review 체크
- 커버리지: (1)`/cup_pose` 계약 통일=Task 1, (2)T_cad_body=Task 2, (3)DDS=Task 3, (4)dry-run 검증=Task 4. ✓
- 타입 일관성: `posestamped_to_candidate`/`cad_to_body_yup_to_zup`/`mesh_aabb` 반환형이 소비처와 일치. quat 전부 wxyz. ✓
- 플레이스홀더: T_base_cam 은 의도적 범위 밖(카메라 대기)으로 명시, 그 외 실코드/실명령. ✓
