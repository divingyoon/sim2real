# FP++ 인지 서브시스템 — 로컬 ROS2 노드 3종 설계 (2026-09-03)

## 목적

sim2real 에서 물체 pose 인지를 **로컬(pc5090)에서 명령 한 줄로** 켜고, 원하는 물체(1개 이상)를
지정하고, 결과 pose 를 정책 노드가 구독할 수 있는 ROS2 토픽으로 낸다. 정책 실행 노드는
별도 스펙(다음 단계)이며, 이 문서는 그 노드가 구독할 **출력 계약**만 고정한다.

## 확정된 사실 (설계 근거)

- pc5090(172.16.0.14) ↔ vision-3090(172.16.0.15) 은 같은 WiFi LAN 이라 **DDS 가 직접 통한다**
  (ROS_DOMAIN_ID 126, 로컬에서 vision 의 카메라 토픽 확인). Tailscale 만으로는 DDS 불가.
- tailscale ssh `vision-3090` 은 무비밀번호로 동작한다(계정 `usr`).
- FP++ 는 vision-3090 의 docker 이미지 `perception-plus-plus:humble-cup` 에서만 돈다. 물체마다
  컨테이너 하나(`parameters_file` 로 yaml 주입). 기동 후 pose 토픽까지 약 45 s.
- FP++ CAD 는 **sim 자산 메시와 같은 물체**여야 한다(09.03 확정: shaker 는
  `assets/meshes/shaker_sim.ply`, cad_to_body 항등; cup_big 은 `cup.obj`, cad_to_body 는 Y-up→Z-up 회전).
- 실물은 둘: 빨간 cup(sim `cup_big_s100`) · 파란 열린 shaker(sim `shaker_closed`).
- Isaac 러너는 더 이상 쓰지 않는다. 출력 소비자는 ROS 네이티브 정책 노드(미래).

## 접근 (승인: A)

로컬 관리 노드가 **ssh 로 vision-3090 을 조종**하고, 카메라 프레임 → base_link 변환(relay)은
**로컬**에서 한다. vision-3090 에는 카메라 노드·FP++ 컨테이너·뷰어 창만 남는다.
기존 vision 쪽 relay·UDP tx(`/tmp/run_relay_*.sh`, `run_tx.sh`)는 폐기한다.

## 구성 요소

### 1. `sim2real/config/objects.yaml` — 물체 레지스트리

```yaml
camera_extrinsics: config/global_camera_extrinsics.yaml   # camera: 블록만 공유
objects:
  shaker_closed:
    real: "파란 열린 shaker (무광)"
    fpp:
      mesh_path: assets/meshes/shaker_sim.ply
      mesh_scale_to_meters: 1.0
      cup_class_id: 41
      detection_pick: blue
      yolo_confidence: 0.35
    cad_to_body:
      position: [0.0, 0.0, 0.0]
      orientation_wxyz: [1.0, 0.0, 0.0, 0.0]
    sim:
      usd: hdgp/assets/cup/shaker_closed_rl.usd
      origin_above_bottom_m: 0.0921
  cup_big_s100:
    real: "빨간 컵"
    fpp:
      mesh_path: assets/meshes/cup.obj
      mesh_scale_to_meters: 1.0
      cup_class_id: 41
      detection_pick: red
      yolo_confidence: 0.15
    cad_to_body:
      position: [0.0, 0.0, 0.0]
      orientation_wxyz: [0.707107, 0.707107, 0.0, 0.0]
    sim:
      usd: hdgp/assets/cup/cup_big_rl.usd
      origin_above_bottom_m: 0.0773
aliases:
  cup_big_s080: cup_big_s100      # sim 스케일 변형 — 실물은 하나
  cup_big_s120: cup_big_s100
```

- 이름은 sim 물체 이름과 같다(`cup_pose_stub.json` 의 `물체` 필드).
- 토픽 네임스페이스는 이름에서 파생: 카메라 프레임 입력 `/perception_plus_plus/<name>/pose`,
  상태 `/perception_plus_plus/<name>/tracking_status`, 출력 `/objects/<name>/pose`.
- FP++ 노드용 yaml 은 레지스트리에서 **생성**한다(`render_fpp_yaml(name)`), 수기 yaml 은 폐기.
- 검증: 알 수 없는 이름·alias 순환·quat 노름·필수 키 누락은 즉시 `ValueError`.

### 2. `perception_launcher_node.py` — 노드 1 (켜기/끄기/뷰어)

- 구독 `/perception/cmd` (`std_msgs/String`, JSON):
  - `{"op":"start","objects":[...],"viewer":true}` — 카메라 up(멱등) → 물체별 yaml 생성·scp →
    `fpp_<name>` 컨테이너 up(이미 떠 있으면 유지) → 목록에 없는 `fpp_*` 는 down → 뷰어 up(요청 시).
  - `{"op":"stop"}` — 모든 `fpp_*`·뷰어 down. 카메라는 `"camera":true` 일 때만 내린다.
  - `{"op":"viewer","on":bool}`.
- 발행 `/perception/status` (`std_msgs/String`, JSON, 1 Hz):
  `{"camera_hz":29.9,"objects":{"shaker_closed":{"container":"Up 43s","pose_age_s":0.08}},
    "viewer":true,"error":null}`. pose_age 는 노드 3 이 발행하는 `/objects/<name>/pose` 로 잰다.
- 원격 실행: `ssh vision-3090 'bash ~/rl_ws/sim2real/scripts/vision/<script> ...'`. 뷰어는
  `DISPLAY=:0` 으로 vision-3090 모니터에 cv2 창을 띄우고 MJPEG 8080 도 함께 켠다.
- 실패 처리: ssh 비정상 종료·컨테이너 즉사(`docker ps` 에서 사라짐)는 status `error` 에 원인 문자열,
  로거 error. 조용히 재시도하지 않는다(사용자가 다시 start).
- 순수 로직(테스트 대상): cmd JSON 파싱·검증, 원하는 상태 vs 현재 상태 → 실행할 액션 목록,
  status 집계.

### 3. `perception_ctl.py` — 노드 2 의 사용자 면 (CLI)

- `perception_ctl.py start shaker_closed cup_big_s100 [--viewer]` / `stop [--camera]` /
  `viewer on|off` / `status` / `list`.
- 레지스트리로 이름을 검증·alias 해석한 뒤 `/perception/cmd` 에 1회 발행. `status` 는
  `/perception/status` 한 장을 받아 표로 출력.
- 노드 2 의 실체는 레지스트리 모듈(`object_registry.py`) + 이 CLI 다. 상주 노드는 아니다
  (상주가 필요한 상태가 없으므로 YAGNI).

### 4. `object_pose_node.py` — 노드 3 (pose 변환·발행)

- 레지스트리의 활성 물체마다 `/perception_plus_plus/<name>/pose` 구독(DDS 직접, 카메라 optical 프레임).
- 변환은 `cup_pose_relay.cad_pose_to_base_body` 재사용:
  `T_base_body = T_base_cam ∘ T_cam_cad ∘ T_cad_body`. camera 블록은 공유 extrinsics,
  `cad_to_body` 는 레지스트리 항목.
- 발행 `/objects/<name>/pose` (`geometry_msgs/PoseStamped`, `frame_id=base_link`, stamp 는 입력 그대로).
- 물체 집합은 파라미터 `objects`(기본: 레지스트리 전체). 구독은 전부 열어 두고 안 오는 토픽은
  그냥 조용하다(컨테이너가 없으면 발행이 없을 뿐).
- 목 각도 추종(`--head-joint-topic`)은 기존 relay 기능 그대로 옵션으로 둔다.

### 5. `sim2real/scripts/vision/` — vision-3090 쪽 스크립트(repo 화)

- `camera_up.sh` / `camera_down.sh` (RealSense, align_depth, domain 126)
- `fpp_up.sh <name> <yaml>` / `fpp_down.sh [name|all]` — 오늘 검증한 `/tmp/run_fpp_shaker.sh` 의 인자화
- `viewer_up.sh` / `viewer_down.sh` — `cup_view_stream.py`(현재 `/tmp` 에만 존재) 를 repo 로 옮겨 실행
- `status.sh` — `docker ps` + 카메라 hz 를 JSON 한 줄로
- 배포: vision-3090 의 `~/rl_ws/sim2real` 은 같은 git 저장소이므로 `git pull` 로 동기화.
  yaml 은 런처가 매번 scp 한다(레지스트리가 진실원천).

## 데이터 흐름

```
perception_ctl start A B --viewer
  → /perception/cmd ─▶ launcher ── ssh ──▶ vision-3090: camera, fpp_A, fpp_B, viewer(창)
                                             │ DDS (LAN)
/perception_plus_plus/A/pose ◀───────────────┘
  → object_pose_node ─▶ /objects/A/pose (base_link)  ─▶ 정책 노드(미래)
launcher ─▶ /perception/status (1 Hz)
```

## 테스트

- 순수 로직 pytest(`scripts/test_object_registry.py`, `test_perception_launcher.py`):
  레지스트리 로드·alias·yaml 렌더·검증 실패, cmd 파싱, 원하는 상태→액션 계획, status 집계.
- pose 변환은 기존 `test_cup_pose_relay.py` 가 덮는다. `object_pose_node` 의 순수 부분(레지스트리
  항목 → Extrinsics 조립)만 추가 테스트.
- 통합(수동, 카메라 필요): `perception_ctl.py start shaker_closed --viewer` → status 에 컨테이너 Up ·
  vision 모니터에 창 → `/objects/shaker_closed/pose` z ≈ 0.322(09.03 기준선) · `stop` 후 컨테이너 0.

## 범위 밖

- ROS 정책 실행 노드(다음 스펙). 이 문서의 `/objects/<name>/pose` 가 그 입력 계약이다.
- 목 자세 캘리브 갱신, FP++ 파인튜닝, 새 물체 메시 제작(레지스트리에 항목만 추가하면 됨).
