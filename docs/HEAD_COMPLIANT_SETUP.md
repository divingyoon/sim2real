# head 순응 홀드 — 손으로 카메라 자세 맞추기

RealSense 화면을 보면서 head 의 다이나믹셀을 **손으로 돌려** 자세를 맞추기 위한 절차.
도구는 `scripts/head_compliant_hold.py`.

> **실기 검증 완료 (2026-09-01).** 두 모터에서 모드 전환 → 처짐 게이트 → 30 초 홀드 →
> 토크 해제까지 끝까지 돌았다. 로직은 단위테스트 28개.

---

## 0. 무엇이 달라지나

|  | 기존 | 이 도구 |
|---|---|---|
| 자세를 정하는 방법 | 절대 엔코더 값을 **입력** | RealSense 를 보며 **손으로** 돌림 |
| 모터 상태 | Extended Position(모드 4), 지령대로 감 | Current-based Position(모드 5), 전류 상한 안에서만 |
| 결과 | 입력값이 참값 | **손으로 맞춘 자세를 읽어 저장한 것**이 참값 |

`config/head_v1.yaml` · `head_v2.yaml` 은 **쓰지 않는다.** 두 파일의 tilt 범위가
135~325° ↔ 69.5~290° 로 어긋나 있어 어느 쪽도 신뢰할 수 없다. 이 도구는
캘리브 범위 파일을 읽지 않고, `--save` 가 새 참값을 만든다.

---

## 1. 전제조건

```bash
# 1) 직렬 포트 권한 — 현재 사용자는 dialout 그룹에 없다
sudo usermod -aG dialout $USER      # 실행 후 재로그인 필요
id | grep dialout                   # 확인

# 2) 포트 확인 (local5090 기준 /dev/ttyUSB0 · FTDI)
ls -l /dev/ttyUSB* /dev/ttyACM*

# 3) 파이썬 — sim2real/.venv 에 dynamixel_sdk 가 이미 있다
cd ~/rl_ws/sim2real/scripts
../.venv/bin/python -c "import dynamixel_sdk; print('OK')"
```

RealSense 는 **local5090 에 없다**(Intel USB 장치 0개). 화면은 vision-3090 에서 보고,
모터 조작은 local5090 에서 한다. 손과 화면이 같은 자리에 있으면 문제없다.

---

## 1.5 버스 상태 — 2026-09-01 실측

```
$ cd ~/rl_ws/sim2real/scripts && ./dxl.sh scan --port /dev/ttyUSB0
baud=1000000 id=1 model=1240 firmware=52
baud=1000000 id=2 model=1240 firmware=52
```

| | |
|---|---|
| 어댑터 | U2D2 (FT232H `0403:6014`) → `/dev/ttyUSB0` |
| **baud** | **1 Mbps** |
| 모터 | id 1 = **pan** · id 2 = **tilt** · 둘 다 XC330-M288(model 1240) fw52 |
| 상태 | 5.1/5.2 V · 21~23 °C · hardware_status 0x00 · errors none |
| Current Limit | 2352 raw (EEPROM) — 정책 상한 150 mA 는 한참 아래다 |

> ★**baud 를 가정하지 말 것.** 이 버스는 하루 안에 57600 → 1 Mbps 로 바뀌었다(모터 교체).
> 실행 전에 항상 `./dxl.sh scan` 을 먼저 돌린다.

## 1.6 이력 — 불량 모터 판별법 (2026-09-01, 교체로 해결됨)

당시 pan 자리의 모터가 **전원을 받아 정상 부팅(LED 점멸 1회)하면서도 통신에 전혀
응답하지 않았다.** 교체로 해결됐다. 같은 증상이 다시 나오면 아래 순서로 좁힌다.

### 배제 과정

| 검사 | 결과 | 배제된 원인 |
|---|---|---|
| 부팅 LED | 점멸 1회 (정상) | 전원 없음 · MCU 사망 |
| 같은 U2D2·포트로 tilt 발견 | id 1 · 5.1 V · 에러 없음 | 어댑터 · 포트 · 권한 · 전원공급 |
| 케이블 A·B 모두 정상 | ↓ 아래 논증 | 케이블 단선 |
| 전 조합 스캔 | **0건** | baud · ID · 프로토콜 설정 오류 |

**케이블이 둘 다 정상인 근거.** 원래 체인은 `U2D2 —A— pan —B— tilt` 였고 tilt 가
응답했다. 즉 데이터가 A 를 지나 pan 의 **수동 통과배선**을 거쳐 B 로 갔다. 데이지체인은
모터 기판을 경유하지 않고 두 커넥터가 PCB 에서 병렬로 이어져 있으므로, 이 사실만으로
A·B 가 모두 정상임이 증명된다. (같은 이유로 "pan 이 죽었는데 pan→tilt 선이 살아 있다"는
모순이 아니다.)

이후 B 케이블로 pan 을 U2D2 에 직결해도 결과는 같았다.

### 스캔 커버리지 (전부 0건)

| | |
|---|---|
| baud | 9600 · 57600 · 115200 · 1M · 2M · 3M · **4.5M** 포함 전 8종 |
| 프로토콜 | 2.0 · 1.0 |
| ID | 0~252 전수 + broadcast |

> ★**4.5 Mbps 는 SDK 가 거부한다.** `PortHandler.setBaudRate` 가 표준 목록에 없는 값을
> False 로 반환하고 조용히 건너뛴다. pyserial 은 이 호스트에서 4.5 M 을 정상 설정하므로
> **소프트웨어 제약일 뿐 하드웨어 한계가 아니다.** `port.ser.baudrate` 를 직접 써서
> 우회한다 — 안 그러면 "전 baud 스캔했다"가 거짓말이 된다.

### 남은 원인과 조치

전원은 받는데 응답을 못 보낸다 → 커넥터 DATA 핀 손상 또는 내부 반이중 통신 IC 불량.
둘 다 **수리/교체 대상**이다. 교체품: **XC330-M288-T** (5 V 급 — XC330-T288-T 와 혼동 금지).

### 그동안 어떻게 할 것인가

**tilt 만으로 진행할 수 있다.** tilt 는 중력 부하를 지는 축이라 순응 홀드가 실제로 필요한
쪽이고, 지금 정상이다. pan 은 **수직축이라 중력 부하가 없어** 토크가 없어도 손으로 돌린
자리에 대체로 머문다 — 다만 **각도를 읽을 수 없어** `--save` 에 기록되지 않는다.

```bash
./dxl.sh scan --port /dev/ttyUSB0 --bauds 57600      # tilt 확인
python head_compliant_hold.py --ids 1 --names tilt --goal-current 10
```

### 교체 후 할 일 — ID 재할당

새 pan 을 달면 공장 기본값(id 1)이라 tilt 와 **충돌**한다. 반드시 **한 개만 연결한
상태**에서 ID 를 바꾼다:

```bash
# 전원 OFF → 새 pan 만 U2D2 에 연결 → 전원 ON
./dxl.sh scan --port /dev/ttyUSB0
./dxl.sh set-id --port /dev/ttyUSB0 --baud 57600 --old-id 1 --new-id 2
# 전원 OFF → 데이지체인 복구 → 전원 ON
./dxl.sh scan --port /dev/ttyUSB0 --bauds 57600      # 1,2 둘 다 보여야 한다
```

`set-id` 는 토크를 끄고 → 쓰고 → **읽기로 검증**하며, 새 ID 가 쓰이는 중이면 거부한다.
다만 그 검사는 **같은 버스의 모터만** 본다 — 그래서 한 개만 연결한 상태가 필수다.

## 2. 왜 Current-based Position Control(모드 5)인가

`Goal Current` 가 **하드웨어 레벨 토크 상한**이다. 소프트웨어가 죽거나 루프가 멈춰도
모터가 그 이상 밀지 못한다 — "과한 토크를 막는다"가 소프트웨어 약속이 아니라
물리적 성질이 된다.

모드 5 는 Extended Position 과 같은 멀티턴 좌표계를 쓰므로 기존 deg↔tick 언랩
수학이 그대로 통한다.

동작은 셋뿐이다:

| | 언제 | 목표 위치 |
|---|---|---|
| **HOLD** | 손이 없다 | 고정 |
| **FOLLOW** | 손이 밀고 있다 | 현재 위치를 따라감 |
| **LATCH** | 방금 멈췄다 | 현재 위치로 확정 → HOLD |

> ★**미는 동안 저항이 사라지는 것은 고장이 아니다.** FOLLOW 에서 목표가 손을 따라가
> 위치 오차가 0 이 되므로 모터가 밀어낼 이유가 없어진다. 손을 떼면 LATCH 로 그 자리를
> 확정해 머문다. 2026-09-01 실기에서 이 동작을 확인했다 —
> "조금만 움직여도 힘이 풀린다"는 관찰은 이 정상 동작을 본 것이었다.
>
> 반대로 **손을 뗐는데 떨어진다면** 그건 진짜 문제다. 그때는 전류가 중력에 모자란 것이니
> `probe_head_hold_current.py` 를 그 자세에서 돌려 최소 유지 전류를 다시 찾는다.

---

## 3. ★Goal Current 찾기 — 반드시 아래에서 위로

적정 전류는 tilt 축 중력토크에 달렸고, 그 값을 아직 재지 않았다. 그래서 **낮은 값에서
시작해 올린다.**

> **위에서 내려오지 말 것.** 높은 값에서 시작하면 사람이 모터와 힘겨루기를 하는
> 구간을 반드시 지나간다. 손도 모터도 그때 상한다.

도구가 시작할 때마다 **처짐을 먼저 잰다**. 목표를 고정한 채 2초 기다려
변위가 한계를 넘으면 추종에 들어가지 않고 종료한다:

```
처짐 확인 2.0s — 손대지 마세요
  pan:   +0.05°
  tilt:  -7.31°

❌ tilt 가 중력을 못 버틴다. --goal-current 를 10.0 보다 **조금씩** 올려 다시 시도할 것
```

절차:

```bash
cd ~/rl_ws/sim2real/scripts
# 계획만 확인 (포트를 열지 않는다)
../.venv/bin/python head_compliant_hold.py --dry-run --names pan,tilt

# 10 mA 부터. 처짐 판정에서 떨어지면 5 mA 씩 올린다
../.venv/bin/python head_compliant_hold.py --names pan,tilt --goal-current 10
../.venv/bin/python head_compliant_hold.py --names pan,tilt --goal-current 15
../.venv/bin/python head_compliant_hold.py --names pan,tilt --goal-current 20
```

**처짐 판정을 통과하는 가장 낮은 값**이 답이다. 그보다 높이면 손으로 돌리기가
그만큼 힘들어질 뿐 이득이 없다.

> **2026-09-01 실측: 10 mA 로 충분하다.** `probe_head_hold_current.py` 로 −41.27°
> 자세에서 10·20·40·80·150·250 mA 를 훑었고 **6단계 전부 처짐 ≤ 1.14°** 로 통과했다.
> 홀드 중 present current 도 0.0 mA — 이 헤드는 기구 마찰이 대부분을 붙잡는다.
>
> ★그래도 게이트는 **시작 자세 한 곳**만 본다. 중력 토크는 자세 의존이므로
> 중력이 가장 크게 걸리는 자세에서 시작해야 의미가 있다. 자세를 크게 바꿨다면
> `probe_head_hold_current.py` 를 그 자세에서 다시 돌린다.

정책 상한은 **150 mA**(`GOAL_CURRENT_HARD_CAP_MA`)이고 넘으면 거부한다. 정격
Current Limit(910 mA)보다 훨씬 낮게 묶은 이유는 "손으로 이길 수 있음"이
이 도구의 안전 성질 전부이기 때문이다. 상한을 올리면 그 성질이 사라진다.

pan 은 수직축이라 중력 부하가 없다 — tilt 보다 훨씬 낮은 값으로 충분하다.
두 축의 적정값이 크게 다르면 따로 돌려도 된다(`--ids 2 --names tilt`).

---

## 3.5 화면 — RealSense 는 다른 기계에 있다

모터는 local5090, **RealSense D435 는 vision-3090** 이다(local5090 에 Intel USB 장치가
없다). `head_compliant_hold.py` 는 `/dev/ttyUSB0` 만 열고 카메라는 건드리지 않으므로
**화면은 따로 띄워야 한다.**

```bash
cd ~/rl_ws/sim2real/scripts
./head_view_up.sh          # 배포 → 원격 기동 → SSH 터널
#   ▶ 브라우저에서 http://127.0.0.1:8080
./head_view_up.sh down     # 정리
```

```
vision-3090                        SSH 터널              local5090
 RealSense D435 → MJPEG :8080 ──────────────────→ 127.0.0.1:8080 → 브라우저
                (127.0.0.1 에만 바인딩)
```

화면에 **중앙 십자선**과 **중앙 영역 깊이(m)** 를 겹쳐 그린다 — 눈대중이 아니라 숫자로
조준한다.

### 왜 X11 포워딩이 아닌가

- vision-3090 sshd 에 `X11Forwarding` 설정이 없어 **기본값(no)** 이다 →
  `X11 forwarding request failed on channel 0`. 켜려면 sudo + sshd 재시작이 필요하다
- 켜더라도 `realsense-viewer` 는 **OpenGL 앱**이라 포워딩하면 느리다

MJPEG + 터널은 sudo 도, 설정 변경도, 네트워크 노출도 없다.

> ★**원격에서 `pkill -f <스크립트이름>` 을 쓰지 말 것.** ssh 명령줄에 그 이름이
> 들어 있어 **자기 자신을 죽인다.** 증상은 "출력이 아예 없음"이라 원인을 찾기 어렵다.
> 브래킷 트릭(`[s]tream...`)도 경로가 같은 줄에 있으면 안 통한다. 이름을 원격 제어
> 스크립트(`head_stream_ctl.sh`)로 빼거나 포트/PID 파일로 찾는다.

---

## 4. 실행

작업 중에는 창이 둘이다:

```
[터미널 · local5090]  head_compliant_hold.py  ← 손으로 모터 돌리기
[브라우저 · local5090] http://127.0.0.1:8080   ← §3.5 의 라이브 화면
```

```bash
cd ~/rl_ws/sim2real/scripts
./head_view_up.sh                                    # 화면 먼저
./dxl.sh scan --port /dev/ttyUSB0                    # ★baud 를 먼저 확인
../.venv/bin/python head_compliant_hold.py \
    --ids 1,2 --names pan,tilt \
    --goal-current 10 \
    --save ../config/head_hand_set.yaml
```

화면:

```
pan   +12.40°(cur  +8.1mA) [HOLD] · tilt  +31.02°(cur +19.4mA) [FOLLOW]
```

- `[HOLD]` 손이 없다 · `[FOLLOW]` 밀고 있다 · `[LATCH]` 방금 확정
- `[FROZEN]` 시작 위치에서 120° 넘게 벗어나 그 관절의 추종을 멈췄다 (`--max-travel-deg`)
- `cur` 이 `--goal-current` 에 계속 붙어 있으면 상한 포화 — 힘겨루기 중이다

**Ctrl-C 로 종료하면 토크를 반드시 끈다.** 종료 경로는 예외가 나도 막히지 않는다.

---

## 5. 중단 기준

즉시 Ctrl-C:

- 🔥 온도 경고 (`{name}: 55°C`) — 도구가 1초마다 확인해 출력한다
- 화면이 `[FOLLOW]` 인데 손을 대지 않았다 — 중력에 흘러내리는 중
- 소리·발열·LED 이상

---

## 6. ★기준 자세 — local5090 기본값

**`config/head_home.yaml` 이 단일 진실원천이다.** 자세·baud·게인이 한곳에 있다.

```bash
cd ~/rl_ws/sim2real/scripts
./head_home.py                # 계획만 (아무것도 쓰지 않는다)
sg dialout -c "../.venv/bin/python head_home.py --execute"    # 적용 + 검증
```

| | 값 | 근거 |
|---|---|---|
| pan | **0.00°** (id 1) | 손으로 맞춘 +2.24° 를 반올림 |
| tilt | **−20.00°** (id 2) | 손으로 맞춘 −18.86° 를 반올림 |
| Position I Gain | **400** | 아래 §6.1 |
| baud | 1 Mbps | 실측 |
| 모드 | 3 (Position Control) | 자세 유지용 |

2026-09-01 검증: `pan +0.132° · tilt +0.000° · σ 0.000° · I=400 ✓`

> ★**Position I Gain 은 RAM 이다 — 전원을 끄면 사라진다.** bringup 마다
> `head_home.py --execute` 를 돌린다. 손 게인(`apply_hand_gains.py`)과 같은 성질이다.

### 6.1 왜 I 게인이 필요한가

XC330 은 Position I Gain 이 **0 으로 출하**된다. P 만으로는 중력처럼 일정한 부하를
이기지 못해 정상상태 오차가 남는다. tilt 가 −20° 지령에서 **+1.5° 처졌다**
(0.56 m 거리에서 약 15 mm — 캘리브에 그대로 들어가는 오차다). pan 은 수직축이라
중력 부하가 없어 I 없이도 0.13° 다.

`probe_head_position_i_gain.py --id 2 --target-deg -20` 실측:

| I | 오차 | σ(진동) |
|---|---|---|
| 0 (출하) | +0.44° | 0.000° |
| 200 | +0.09° | 0.000° |
| **400** | **+0.00°** | **0.000°** |
| 800 | +0.00° | 0.000° |

800 까지도 진동이 없어 여유가 크다. **400** 을 쓴다.

### 6.2 ★★쓰기 순서 — 여기서 두 번 물렸다

```
토크off → Operating Mode → 게인 → 프로파일 → 토크on → 목표
```

두 가지 펌웨어 동작 때문이고, **둘 다 조용히 실패한다**(에러가 안 난다):

1. **Operating Mode 를 쓰면 제어 게인이 그 모드의 기본값으로 리셋된다.**
   `apply_head_gains.py` 로 I=400 을 넣은 뒤 `head_goto_hold.py`(모드 3 으로 전환)를
   돌렸더니 I 가 **0 으로 돌아가** tilt 가 +1.56° 처졌다. → 게인은 모드 **뒤에**.
2. **Torque Enable 이 0→1 이 되면 Goal Position 이 Present Position 으로 덮어써진다**
   (급격한 점프 방지). 토크가 꺼진 동안 tilt 가 중력에 떨어졌고, 그 자리(tick 1871)가
   목표가 되어 **오차 +4.22°** 로 굳었다. 모터는 자기 목표를 완벽히 지키고 있었다 —
   지령이 잘린 것이다. → 목표는 토크 **뒤에**.

이 순서는 `test_head_home.py` 의 순서 테스트 3개가 고정한다.

### 6.3 다음 단계

이 자세에서 카메라 extrinsics 교정(`find_head_tilt.py` → ChArUco)으로 이어진다.
자세를 다시 손으로 잡고 싶으면 §4 의 순응 홀드를 쓰고, 결과를 `head_home.yaml` 에
반영한다.

## 6.4 ★sim ↔ real 카메라 위치 불일치 (2026-09-01 확인)

URDF 가 두는 head 카메라 자리와 **실제 컬러 광학 중심이 59.5 mm 어긋난다.**

같은 optical 회전을 쓰고 병진만 바꿔 재투영해 정량화했다(25프레임 · 553코너):

| | RMS 재투영 | 0.64 m 에서 |
|---|---|---|
| 캘리브된 `T_neck_cam` | **1.214 px** | 1.3 mm |
| sim 명목 (URDF `head_cam_view`) | **50.268 px** | **53 mm** |

기준 자세에서 위치 차이 = `[+43.7, +38.1, −13.3] mm`.

**규약 문제가 아니다.** 생성기 문서가 `head_cam_view: camera view origin frame
(+X = viewing direction, +Z up)` 라고 정의하므로 이 프레임은 **카메라 시점 원점으로
의도된 것**이고, 위 비교는 회전을 동일하게 두고 병진만 본 것이다.

★**FK 자체는 맞다** — 영점 자세에서 `head_fk_chain.py` 가 `[0.0372, 0.0000, 0.8525]`
를 내고, 이는 태스크 코드 주석의 `head_cam_view [0.037, 0, 0.852]` 와 일치한다.
어긋나는 것은 FK 가 아니라 **자산이 두는 카메라 위치**다.

### 지금 당장은 문제가 아니다

sim 의 `TiledCameraCfg` 는 `prim_path="/World/envs/env_.*/Camera"` — **월드 고정**이고
head 링크에 붙어 있지 않다. 그래서 학습된 distillation 정책은 head 카메라를 쓰지 않고,
이 불일치의 영향을 받지 않는다.

영향을 받는 것은 **실기 head 카메라를 쓰는 쪽**(cup_pose · FoundationPose)이고,
그쪽은 이미 `config/head_extrinsics.yaml`(캘리브값)을 쓰면 된다.

### 고치려면

`head_j_cam_view` 의 origin 은 **생성된 자산**이다(`urdf/tools/generate_rl_urdf.py:612`).
바꾸려면 생성기를 고치고 URDF·USD 를 다시 만들어야 한다 — 자산 재빌드가 따르므로
사용자 판단이 필요하다. 고치지 않고 두어도 되며, 그 경우 **sim 의 head 카메라 시야는
실제와 다르다**는 점만 알고 있으면 된다.

---

## 6.5 ★실기 카메라를 sim 에 이식 — SIM = REAL

**자산의 `head_cam_view` 는 건드리지 않는다.** 대신 실측 `T_neck_cam` 으로 `head_camera`
(tilt 링크)에 카메라를 **새로** 붙인다. 그러면 목이 돌 때 sim 카메라도 똑같이 따라가고,
실기에서 본 물체를 sim 에 같은 자리에 놓을 수 있다.

```bash
python sim_head_camera.py            # Isaac 설정 조각 출력
python sim_head_camera.py --json ../config/head_camera_sim.json
```

이식하는 것 셋:

| | 값 |
|---|---|
| **extrinsics** | `T_neck_cam` (hand-eye 실측) → `OffsetCfg(pos, rot, convention="ros")` |
| **intrinsics** | 실측 K → `PinholeCameraCfg.from_intrinsic_matrix(...)` · 640×480 · fx 606.60 fy 605.65 · FOV 55.63°×43.23° |
| **프레임 규약** | `T_neck_cam` 의 목적지가 optical 프레임이고, 그것이 곧 Isaac 의 `convention="ros"` |

### 검증 (`probe_sim_head_camera.py`)

```
head_j_tilt   지령 -20.000° → sim -20.000°
link head_camera  sim [0.02250, -0.01450, 0.81600]
                  FK  [0.02250, -0.01450, 0.81600]
link_err 0.000 mm · 0.0000°          PASS
```

FK ↔ sim 기구학이 **정확히** 일치한다. 카메라 오프셋은 설정값이므로 이로써 sim 카메라는
실측 자리에 선다.

### ★부딪힌 것 넷

1. **씬 cfg 에 카메라를 넣으면 안 된다.** `grasp_s2r` 은 DirectRLEnv 라 로봇을
   `_setup_scene()` 에서 추가하는데, 씬 cfg 의 센서는 **그보다 먼저** 만들어져
   `head_camera` prim 이 아직 없다 → `Unable to find source prim path`.
   **env 를 만든 뒤** `Camera(cfg)` 로 붙인다.
2. **센서 초기화 콜백이 다시 안 뜬다.** sim play 이벤트에 걸려 있는데 env 가 이미
   play 중이다. `camera._initialize_impl()` 을 직접 태운다.
3. **`camera.data.pos_w` 가 0 으로 남는다**(fabric 에서 안 채워짐). 판정에 쓰지 말고
   **링크 자세**(`robot.data.body_pos_w`)로 검증한다 — 어차피 카메라 오프셋은 우리가
   넣은 설정값이라 확인할 것은 FK ↔ sim 기구학이다.
4. **★sim 의 head 액추에이터가 약하다.** 상태를 한 번만 써 넣으면 물리 스텝마다
   **1.6°씩** 기본 목표(0)로 끌려간다(−20° 지령이 12스텝 뒤 −7.4°). sim 에서 목 자세를
   유지하려면 **매 스텝 다시 명령**하거나 액추에이터 강성을 올려야 한다.

### 실기 목 각도를 sim 에 넣을 때

**부호 변환을 거쳐야 한다** — `head_fk_chain.urdf_from_encoder(pan, tilt)`.
pan 은 부호가 반대다(§6.4의 근거). 그냥 인코더 값을 넣으면 pan 이 반대로 돈다.

---

## 6.6 실기 물체를 sim 으로 — 전체 고리 검증

실기 카메라로 본 물체를 base 좌표로 옮겨 sim 에 소환하면 sim 카메라가 **같은 화소**에
그린다. 2026-09-01 실측:

```
실기 카메라 → 캘리브 → base 좌표 → sim 소환 → Isaac 렌더 → 다시 화소
평균 1.52 px · 중앙값 1.51 · 최대 1.71   (0.64 m 에서 1.6 mm)
```

절차(ChArUco 보드를 예로):

```bash
# 1) 실기 프레임 → 보드의 base 좌표
#    T_base_board = T_base_neck(pan,tilt) ∘ T_neck_cam ∘ T_cam_board
# 2) 그 자리에 소환하고 렌더
./isaaclab.sh -p probe_sim_head_camera.py --headless \
    --pan-deg 0.132 --tilt-deg -20.0 --board-npy T_base_board_live.npy
```

라이브 단일 프레임으로 잰 보드 위치 `[+0.1662, −0.0975, +0.2268]` 는 25프레임 캘리브의
`[+0.1707, −0.0841, +0.2264]` 와 **14.1 mm** 차이였다(z 는 0.4 mm) — 독립 확인이 된다.

> ★**소환한 마커를 물체의 자식으로 만들지 말 것.** `/World/RealBoard/c_i_j` 로 두면
> translation 이 보드 변환에 **한 번 더** 곱해져 엉뚱한 데로 간다. 형제
> (`/World/RealCorners/...`)로 둔다.

## 6.7 태스크 배선

`hdgp/source/openarm/openarm/sensors/head_camera.py` 가 함정 넷을 안에서 처리한다.

```python
from openarm.sensors.head_camera import attach_head_camera, urdf_head_angles

camera = attach_head_camera(env)             # env 를 만든 **뒤에**
pan_urdf, tilt_urdf = urdf_head_angles(pan_encoder, tilt_encoder)
```

캘리브값은 `sim2real/config/head_camera_sim.json` 에서 읽는다(생성:
`sim2real/scripts/calib/sim_head_camera.py`). 테스트 7개가 **캘리브 파일 존재**와
**부호 규약**을 고정한다.

★학습 태스크(`grasp_s2r` 등)는 **고치지 않았다** — 재학습 중이라 건드리면 재생이
조용히 달라진다. 붙일 자리는 사용하는 쪽에서 정한다.

---

## 6.8 컵 좌표 경로 — 목이 돌아도 맞게

`cup_pose_relay.py` 는 `T_base_body = T_base_cam ∘ T_cam_cad ∘ T_cad_body` 로
FoundationPose 출력을 base 프레임으로 옮긴다. 문제는 `T_base_cam` 이 **정적**이라는 점이다.

### ① 정적값 갱신 (2026-09-01 완료)

`global_camera_extrinsics.yaml` 의 `camera:` 블록을 hand-eye 값으로 교체했다.

| | 구 (2026-08-02) | 신 (2026-09-01) |
|---|---|---|
| position | `[0.06385, 0.04053, 0.82714]` | **`[0.06748, 0.03812, 0.84200]`** |
| 방법 | 보드 CAD 고정 · 1프레임 | hand-eye · 25프레임 · 553코너 |

구값이 낡은 이유 셋: ⓐ목 모터 교체로 인코더 영점이 다르다 ⓑ교정 자세
`pan −90/tilt 280` 이 지금 좌표계에 없다 ⓒ방법이 다르다.
차이는 **위치 15.5 mm · 회전 4.19°** — 0.6 m 컵이면 회전만으로 **약 44 mm** 다.

★이 값은 **`pan 0 / tilt −20` 에서만** 유효한 스냅샷이다. `head_home.py --execute`
로 그 자세를 재현한 뒤에 쓴다.

### ② 목 각도 인식 모드

목이 돌면 정적값은 통째로 틀린다 — `pan 15°` 면 카메라가 11 mm 이동 + 큰 회전이다.

```bash
# local5090: 목 각도를 ROS 로 (읽기 전용 — 토크·게인을 안 건드린다)
python head_joint_publisher.py                      # 20 Hz → /head/joint_states

# 릴레이: 목 각도로 T_base_cam 을 매번 계산
python cup_pose_relay.py --head-joint-topic /head/joint_states
```

```
T_base_cam(pan, tilt) = T_base_neck(pan, tilt) ∘ T_neck_cam
```

**정합 확인**: 기준 자세에서 계산값이 정적 yaml 과 **0.14 mm** 차이다(실기 실측).
그래서 모드를 켜도 컵 좌표가 점프하지 않는다 — 테스트가 이걸 고정한다.

### ★규약 — 여기서 부호를 틀리기 쉽다

| 어디 | 규약 |
|---|---|
| `head_home.yaml` · 캘리브 · 사람 | **인코더 각(deg)** — pan 0 / tilt −20 |
| ROS `joint_states` | **URDF 라디안** — 토픽이 URDF 관절 이름을 쓰므로 |

변환은 `head_fk_chain.urdf_from_encoder` / `encoder_from_urdf` 가 한다. **손으로 뒤집지
말 것** — pan 은 부호가 반대라 언젠가 반대로 넣는다. 왕복 테스트가 이걸 고정한다.

### ★목 각도가 오래되면 **발행하지 않는다**

`--head-max-age`(기본 1.0 s)를 넘으면 릴레이가 `cup_pose` 를 내보내지 않고 경고만 낸다.
틀린 컵 좌표는 없느니만 못하다 — 정책이 그걸 믿고 손을 뻗는다.

---

## 7. 안전 장치 요약

| 장치 | 값 | 하는 일 |
|---|---|---|
| Goal Current 정책 상한 | 150 mA | 넘으면 **거부**(clamp 아님) |
| EEPROM Current Limit 확인 | 모터에서 읽음 | 상한 초과 지령을 모터가 조용히 자르는 것을 막음 |
| 시작 처짐 측정 | ±3° / 2s | 전류가 중력에 모자라면 **추종에 들어가지 않음** |
| 1주기 추종 제한 | 180°/s | 통신 글리치가 목표를 순간이동시키지 못함 |
| 이동 한계 | ±120° | 시작 위치에서 너무 벗어나면 그 관절 추종 정지 |
| 온도 감시 | 55°C | 1초마다 경고 |
| SIGINT 직접 수신 | — | 어떤 경로로 끝나도 토크 해제 |
