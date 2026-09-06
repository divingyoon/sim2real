#!/usr/bin/env python3
"""Build `deploy_contract.json` — from a training run dump, or control-only from a robot asset.

    # policy contract from a run (numbers come from params/env.yaml, agent.yaml, nn/*.pth)
    python3 policy_control/tools/build_deploy_contract.py --run logs/policy/left_v2B25 --grasp-band v1
    python3 policy_control/tools/build_deploy_contract.py --run logs/policy/right_g1 --gains <control_gains.yaml>
    # the same run re-based onto the 09.05 bimanual DG-5F-M asset (fabric URDF/params of that asset)
    python3 policy_control/tools/build_deploy_contract.py --run logs/policy/right_g1 --asset openarm_dg5f-m_bi_rl
    # control-only contract (no policy) for pd/fabric tests, one arm at a time
    python3 policy_control/tools/build_deploy_contract.py --asset openarm_dg5f-m_bi_rl --home zero \
        --out logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_control import contract as C  # noqa: E402
from policy_control.contract_assets import ASSETS, DEFAULT_ASSET, build_asset_contract  # noqa: E402
from policy_control.contract_build import build_contract  # noqa: E402


def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=None, help="run dir holding params/ and nn/ (omit for asset-only)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="explicit .pth (default: the only one in nn/)")
    ap.add_argument("--out", type=Path, default=None, help="default <run>/deploy_contract.json")
    ap.add_argument("--gains", type=Path, default=None, help="control_gains.yaml to compare trained kp/kd with")
    ap.add_argument("--grasp-band", default=None,
                    help="gripper_left only: v1 | v2 | lo,hi (table-height m). v2B25 was trained with v1")
    ap.add_argument("--asset", default=None, choices=sorted(ASSETS),
                    help=f"bind to an hdgp/assets/robot asset; without --run: control-only contract (default {DEFAULT_ASSET})")
    ap.add_argument("--sides", default="right,left", help="asset-only: sides to include (comma list)")
    ap.add_argument("--primary", default="right", help="asset-only: side mirrored into the legacy top-level sections")
    ap.add_argument("--home", default="zero", help="asset-only: zero | run:<run dir> (init_state, mirrored)")
    args = ap.parse_args(argv)
    if args.run is None and args.out is None:
        ap.error("--out is required without --run")
    return args


def main(argv=None) -> int:
    args = _parse(argv)
    if args.run is not None:
        c = build_contract(args.run, checkpoint=args.checkpoint, grasp_band=args.grasp_band, asset=args.asset)
        out = args.out or (args.run / "deploy_contract.json")
    else:
        c = build_asset_contract(args.asset or DEFAULT_ASSET, sides=tuple(args.sides.split(",")),
                                 primary=args.primary, home=args.home)
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    C.save_contract(c, out)
    asset = c.asset.name if c.asset else "run asset (training-time fabric URDF)"
    print(f"[contract] {c.run.task} · obs {c.policy.obs_dim} / act {c.policy.action_dim} · "
          f"{c.rate.policy_hz:.0f} Hz · gravity {c.pd.gravity.mode} · sides {c.side_names} (primary {c.primary_side}) · "
          f"asset {asset} → {out}")
    if args.gains:
        return _report_gains(c, args.gains)
    return 0


def _report_gains(c: C.DeployContract, gains: Path) -> int:
    rc = 0
    for side in c.side_names:
        rep = C.compare_gains(c, gains, side=side)
        print(f"[contract] {side} gains {'OK' if rep.ok else 'MISMATCH'}"
              + ("" if rep.ok else ": " + "; ".join(rep.reasons))
              + (f"\n  kd note: {rep.kd_note}" if rep.kd_note else ""))
        rc = rc or (0 if rep.ok else 3)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
