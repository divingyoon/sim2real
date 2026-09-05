# 테이블·원점 datum 감사 — 2026-09-05

> 결론: **실기 테이블 상면 = 플레이트 상면(body_root) + 0.205 m.** sim 도 0.205 로 맞췄다(`env_v1`).
> 09.03 의 "실기 테이블 0.230" 은 카메라·팔 두 측정 사슬의 **datum 오차**였다. FP++ 는 무죄.

## 1. 원점(datum) 정의

| 프레임 | 물리 위치 | 근거 |
|---|---|---|
| `body_root` (sim/URDF z=0) | **15 mm 마운트 플레이트 상면 = 로봇 60×60 기둥 밑면** | `urdf/tools/crop_body_plate.py` (벤더 8 mm 판을 잘라 원점을 판 상면으로), env_v1 platform z −0.015~0 |
| 테이블 상면 | body_root + **0.205** (top_plate 0.195~0.205, 받침대 195 mm) | `hdgp/assets/simulation_setting/env_v1` (Fusion, 09.05) · 사용자 줄자 0.205 |
| 팔 베이스 j1 축 (`r/l_al_0`) | body_root + 0.690 | 벤더 v10 0.698 − 8 mm |
| head 마운트 (`head_base` 원점) | body_root + **0.750** (URDF 상수, **미검증**) | `generate_rl_urdf.HEAD_MOUNT_XYZ` |
| 벤더 body 기둥 | 60×60 압출재 상단 0.750 · 캡 상면 0.765 · 브래킷 면 0.741 | `body_link0_visual_cut.stl` 수평면 분석 |

head_v1 CAD 의 베이스 플레이트는 원점 −20 mm 가 밑면이다. URDF 0.750 이면 플레이트 밑면이 0.730
(캡 0.765 보다 35 mm 아래 = 기둥 속) — 벤더 CAD 대로 캡 위에 얹으면 **0.785** 여야 한다.
그런데 카메라 사슬은 실기 head 가 0.750 보다 **21 mm 낮다**(≈0.729)고 말한다 → 실기 head 는
벤더 캡 위에 있지 않거나 기둥 길이가 다르다. **B4 실측이 결정한다.**

## 2. 각 숫자의 body_root 환산 (정정판)

| 값 | 출처 | 환산 | 상태 |
|---|---|---|---|
| 0.200 | 구 env.usd (08.20 CAD 공칭) | 0.200 | 폐기 — 받침대 190 mm 가 CAD 오류 |
| **0.205** | 줄자(09.02·09.05) + env_v1 CAD | **0.205** | ✅ 확정 |
| 0.2075 / 0.210 | 09.02 사용자 발언 범위의 중점 / 가드 상한 | 줄자 계열 | "미검증 가정"이라던 09.03 표기는 틀림 |
| 0.2264 | hand-eye T_base_board z (보드 = 테이블 위 종이) | 카메라 사슬 | **+21.4 mm** 편향 |
| 0.2301 | 보드 3곳 평면(정적 camera 블록) | 카메라 사슬 | +25.1 mm (촬영 시 목 각도 미기록, 손맞춤 자세면 +20.7) |
| 0.2269 / 0.2299 | FP++ 컵 0.3042−0.0773 / shaker 0.332−0.0921 | 카메라 사슬 | 컵 +22, shaker +25 mm. 09.03 의 "0.066" 오프셋은 코드에 없음 |
| 0.231~0.245 | 좌 그리퍼 짚기 3점 | 팔 사슬 | TCP(0.2445) vs 손끝 후보(0.2306)의 폭. 손끝은 TCP +15.4 mm(finger.stl 95.4 vs tcp 80 mm) → **0.229** = +24 mm |

## 3. 편향의 위치

- **카메라 사슬 +21 mm**: `T_base_cam = T_base_neck(pan,tilt) ∘ T_neck_cam`. hand-eye 는 base→목 고정변환의
  **z 병진 오차를 T_base_board 로 흡수**해 잔차에 안 나타난다(`refine_head_handeye.py` 서두, Δz −25 mm
  주입 실험: 잔차 불변·보드 z 0.2264→0.2003). head 내부 사슬(pan +0.0356 · tilt +0.030 · 렌즈 +0.036)은
  head_v1 CAD 와 1~3 mm 안에서 일치 → 남는 미지수는 **마운트 높이 0.750** 하나.
- **팔 사슬 +24 mm**: head 와 무관. 후보 ① 기둥(팔·head 공유)이 CAD 보다 ~22 mm 짧다 ② 좌 그리퍼 손가락이
  CAD 보다 길다(실측 스트로크 48.8 vs 모델 40 mm 로 이미 하드웨어가 다름).
- FP++ 자체(등록·메시·depth)는 30 mm 를 못 만든다 — 보드 PnP 평면과 ±10 mm 안에서 일치.

## 4. 실측 항목 (로봇 정지 · 강철자/캘리퍼 · 각 2회 + 사진)

기준면 = **플레이트 상면(기둥 밑면)**.

| # | 측정 | URDF/CAD | 판독 |
|---|---|---|---|
| B1 | 플레이트 상면 → 좌 j1 축 중심 | 0.690 | ≈0.666 이면 기둥이 짧다 → 팔·head z 함께 수정 |
| B3 | 플레이트 상면 → 기둥 캡 상면 | 0.765 | 기둥 길이 확인 |
| **B4** | 플레이트 상면 → head 베이스 플레이트 **밑면** / tilt 축 / RGB 렌즈 | 0.730 / 0.816 / 0.842 (URDF) · 카메라 역산 0.709 / 0.795 / 0.821 · 캡 위 0.765 / 0.851 / 0.877 | `HEAD_MOUNT_XYZ` = 밑면 + 0.020 |
| F | 손목 플랜지(`l_al_7` 끝면) → 닫힌 좌 손끝 | 0.1955 (0.1001 + 0.0954) | 팔 사슬 편향의 그리퍼 몫 |

## 5. 반영 상태 (09.05)

- sim: `table_surface_z 0.205` · `hand_floor_z 0.215` · env cfg 가 `simulation_setting/env_v1/usd/env_v1.usda`
  (루트 kinematic RigidBody 저작 — `env_rigid.usd` 사본 불필요) · 물체 자산은 `assets/multi_obj/` 로 이동.
- 배포: `TABLE.top 0.215`(가드) · `REAL_TABLE_TOP 0.205` · z 보정 0. 카메라 편향은 `HEAD_MOUNT_XYZ` 로 고칠 것
  (B4 뒤). `T_neck_cam` 은 head_v1 tilt 프레임(y +2.954 mm)으로 재표현해 `T_base_cam` 불변.
- head: `urdf/vendor/head_v1` (`tools/import_head_v1.py`) → RL URDF 4종 재생성. body_link↔head_mid 5.5 mm
  겹침은 마운트 0.750 의 산물이라 allowlist(accept_raw) + USD 충돌 필터로 처리, B4 뒤 제거.

## 6. 결론 (09.05 밤) — 카메라 사슬 편향의 정체와 해소

- **depth 평면 교차검증**(`scripts/calib/depth_table_plane.py`, run1 25 프레임): 옛 사슬로 테이블 0.225(배율 보정)·0.218(원 depth).
  D435i depth 는 charuco 거리보다 1.24 % 멀리 읽고, 평면이 1.75° 기울어 보인다(센서 왜곡 의심).
- **head_v1 CAD 사전 모델 hand-eye 재계산**(`scripts/calib/solve_head_cad_prior.py`): 옛 T_neck_cam 은 CAD 카메라 자세와
  tilt 축 기준 91° 차이 = tilt 인코더 영점 오프셋(**+90.51°**)이 통째로 흡수된 값. 카메라 장착을 CAD 로 고정하면
  RGB 렌즈가 `color_frame`(y −0.0115) 이 아니라 **y +0.0326 개구부(Fusion 라벨 ir_projector_frame)** 일 때만
  맞는다(RMS 1.86 px vs 5.94 px, 회전 잔차 불필요) — 그때 보드(테이블) z = **0.2037** = 줄자 0.205.
- **sim 렌더 vs 실사진**(`our_source/head_check_0905/`): 홈 자세 실사진의 상판 볼트 패턴(기둥 상단 x 0.135·y ±0.3/0)이
  CAD 사슬 sim 렌더와 높이·간격 일치, 좌우는 위 개구부 가정으로 3~8 px 안에 들어옴. pan 영점 ≈ 0.
- ⇒ **B4 불필요**: 마운트 0.750·테이블 0.205 유지. "+21 mm" 는 옛 hand-eye 가 RGB 개구부 오지정(≈44 mm 옆)을 카메라 높이로 흡수한 산물.
- 적용: `config/head_extrinsics.yaml` v2(옛 값 `head_extrinsics_pre0905_handeye_v1.yaml`), `global_camera_extrinsics.yaml` camera 블록,
  `head_camera_sim.json`, urdf `HEAD_RGB_LENS_FRAME`(head_cam_view = 그 개구부) → 자산 4종 재생성.
- 남은 팔 사슬 +24 mm(좌 그리퍼 짚기 0.229)는 별개 — B1/F 실측 또는 팔 인코더 영점 점검.
