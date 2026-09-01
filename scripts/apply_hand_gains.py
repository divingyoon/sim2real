#!/usr/bin/env python3
"""손 JTC PID 게인을 **한 번의 서비스 호출**로 적용한다 (bringup 이후 매번).

09.01 확정값 `p=4.5 · i=0 · d=0`. 근거는 `docs/R2S_FRAMEWORK.md` §손:

  · 벤더 기본 `p=1.5` 는 4 s 램프 주먹에서도 지령의 82 % 밖에 못 간다.
  · **p=6 부터 진동**이 시작된다(다관절 σ 0.099 → 0.160°, p=12 는 육안 확인).
  · `p=4.5` 는 완전 주먹에서 σ 0.095°(기저 0.09°와 같음)·도달률 98~101 %·
    정상오차 0.39° — 진동 없이 얻을 수 있는 최선이다.
  · `d` 는 0~0.05 에서 진동에 영향이 없고(σ 0.095~0.099°), 단일 관절에서는
    d>0.02 가 오버슈트를 만든다 ⇒ 벤더처럼 **d=0** 을 유지한다.

★진동은 **다관절 동시**에만 나타난다. 관절 하나로 시험하면 σ 가 0 이라 놓친다.

    python3 apply_hand_gains.py            # 현재 값 확인만
    python3 apply_hand_gains.py --execute  # p=4.5 적용
    python3 apply_hand_gains.py --restore --execute   # 벤더 기본으로 되돌림
"""

from __future__ import annotations

import argparse

CTRL = "/dg5f_right/dg5f_right_controller"
TUNED_P, TUNED_D = 4.5, 0.0
VENDOR_P, VENDOR_D = 1.5, 0.0
JOINTS = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=None, help="기본 4.5")
    parser.add_argument("--d", type=float, default=None, help="기본 0.0")
    parser.add_argument("--restore", action="store_true", help="벤더 기본(1.5/0)")
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
