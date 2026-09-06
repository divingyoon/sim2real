#!/usr/bin/env python3
"""손 JTC PID 게인을 **한 번의 서비스 호출**로 적용한다 (bringup 이후 매번).

★★2026-09-06 사용자 확정: 학습·제어 전부 **벤더 기준 PD 게인만** 쓴다.
   DG-5F 손의 벤더값은 `p=1.5 · i=0 · d=0`(dg5f_driver PID yaml, 40 관절 동일)이고
   그것이 이 스크립트의 기본값이다. 드라이버가 이미 그 값으로 뜨므로 평소에는
   **적용할 것이 없다** — 이 도구는 이제 "드라이버가 벤더값인지 확인·복원"이 주 용도다.

   구 09.01 튜닝값 `p=4.5`(폐기, `--p 4.5` 로만 도달 가능)의 실측 근거는 기록으로 남긴다:
     · 벤더 기본 p=1.5 는 4 s 램프 주먹에서 지령의 82 % 까지만 간다.
     · p=6 부터 진동(다관절 σ 0.099 → 0.160°), p=4.5 는 도달률 98~101 %·σ 0.095°.
     · d 는 0~0.05 에서 진동 무관, 단일 관절 d>0.02 는 오버슈트 ⇒ 벤더처럼 d=0.
   ⇒ 벤더값으로 돌아가면 **파지 도달률이 82 % 대로 떨어진다**. sim 도 같은 1.5 로
     학습하므로 sim↔실기는 일치하지만, 파지 성능은 재확인이 필요하다.

★진동은 **다관절 동시**에만 나타난다. 관절 하나로 시험하면 σ 가 0 이라 놓친다.

    python3 apply_hand_gains.py            # 현재 값 확인만
    python3 apply_hand_gains.py --execute  # 벤더 p=1.5 적용(=드라이버 기본)
    python3 apply_hand_gains.py --p 4.5 --execute   # ★구 튜닝값 — 벤더 규칙 위반
"""

from __future__ import annotations

import argparse

CTRL = "/dg5f_right/dg5f_right_controller"
#: 벤더 드라이버 PID(dg5f_both_pid_all_controller.yaml, 40 관절 전부 같다) = 기본값.
VENDOR_P, VENDOR_D = 1.5, 0.0
#: 구 09.01 튜닝값 — 2026-09-06 벤더 전용 규칙으로 폐기. `--p 4.5` 로만 쓸 수 있다.
RETIRED_TUNED_P, RETIRED_TUNED_D = 4.5, 0.0
TUNED_P, TUNED_D = VENDOR_P, VENDOR_D
JOINTS = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=None, help="기본 1.5(벤더). 4.5 는 폐기된 튜닝값")
    parser.add_argument("--d", type=float, default=None, help="기본 0.0")
    parser.add_argument("--restore", action="store_true", help="벤더 기본(1.5/0) — 이제 기본값과 같다")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.restore:
        p, d = VENDOR_P, VENDOR_D
    else:
        p = TUNED_P if args.p is None else args.p
        d = TUNED_D if args.d is None else args.d

    import rclpy
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import GetParameters, SetParameters
    from rclpy.node import Node

    rclpy.init()
    node = Node("apply_hand_gains")
    try:
        getter = node.create_client(GetParameters, f"{CTRL}/get_parameters")
        if not getter.wait_for_service(timeout_sec=10.0):
            print(f"❌ {CTRL} 가 없다 — 손 bringup 확인")
            return 1
        names = [f"gains.{j}.{k}" for j in JOINTS for k in ("p", "d")]
        req = GetParameters.Request()
        req.names = names
        fut = getter.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=20.0)
        if fut.result() is None:
            print("❌ 현재 값을 못 읽었다")
            return 1
        cur = [v.double_value for v in fut.result().values]
        print(f"현재  p {sorted({round(x,4) for x in cur[0::2]})} "
              f"· d {sorted({round(x,4) for x in cur[1::2]})}")
        print(f"목표  p {p} · d {d}")
        if not args.execute:
            print("\nDRY RUN — 실제로 적용하려면 --execute")
            return 0

        setter = node.create_client(SetParameters, f"{CTRL}/set_parameters")
        if not setter.wait_for_service(timeout_sec=10.0):
            print("❌ set_parameters 서비스가 없다")
            return 1
        # ★40개를 한 번에. `ros2 param set` 을 40번 부르면 수십 초가 걸리고
        #   09.01 에 타임아웃으로 게인이 중간 상태로 남을 뻔했다.
        sreq = SetParameters.Request()
        for joint in JOINTS:
            for key, value in (("p", p), ("d", d)):
                sreq.parameters.append(Parameter(
                    name=f"gains.{joint}.{key}",
                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                         double_value=float(value))))
        sfut = setter.call_async(sreq)
        rclpy.spin_until_future_complete(node, sfut, timeout_sec=20.0)
        res = sfut.result()
        if res is None:
            print("❌ 설정 응답이 없다")
            return 1
        ok = sum(1 for r in res.results if r.successful)
        print(f"적용 {ok}/{len(sreq.parameters)}")
        return 0 if ok == len(sreq.parameters) else 1
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
