#!/usr/bin/env python3
"""
csc_v2_gate.py -- pre-registered validation gate for Cityscapes-C v2
(results/preregistration_step_curve.md, addendum).

Gate: on the 8 deterministic corruption types x 5 severities (40 conditions), v2 vanilla
mIoU must reproduce the published v1 pertype_vanilla_* values within |delta| <= 0.05 mIoU.
Any miss -> exit 1 (stop and debug). Also reported, with no threshold: the |delta|
distribution over the 11 stochastic types (55 conditions).

Usage: python csc_v2_gate.py <pool>   (reads stepcurve/<pool>/van_v2pertype_<type>.json)
"""
import json, os, sys
import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
POOL = sys.argv[1] if len(sys.argv) > 1 else "v100"
TOL = 0.05
DET = ["brightness", "contrast", "defocus_blur", "gaussian_blur", "jpeg_compression",
       "pixelate", "saturate", "zoom_blur"]
STO = ["elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
       "motion_blur", "shot_noise", "snow", "spatter", "speckle_noise"]


def sevs(path):
    return {r["severity"]: 100 * r["miou"] for r in json.load(open(path))["severity_results"]}


def deltas(types):
    out, missing = [], []
    for t in types:
        p2 = os.path.join(RES, "stepcurve", POOL, f"van_v2pertype_{t}.json")
        if not os.path.exists(p2):
            missing.append(t); continue
        v1, v2 = sevs(os.path.join(RES, f"pertype_vanilla_{t}.json")), sevs(p2)
        out += [(t, s, v1[s], v2[s], v2[s] - v1[s]) for s in sorted(v1)]
    return out, missing


def main():
    det, miss_d = deltas(DET)
    sto, miss_s = deltas(STO)
    if miss_d:
        print("GATE NOT EVALUABLE: v2 vanilla missing for", miss_d); sys.exit(2)
    print(f"Deterministic types ({len(det)} conditions), pool={POOL}, tol={TOL}:")
    fails = [d for d in det if abs(d[4]) > TOL]
    for t, s, a, b, d in det:
        flag = "  <-- MISS" if abs(d) > TOL else ""
        print(f"  {t:17s} sev{s}  v1 {a:6.2f}  v2 {b:6.2f}  delta {d:+.3f}{flag}")
    ad = np.abs([d[4] for d in det])
    print(f"  max |delta| {ad.max():.3f}, mean {ad.mean():.3f}")
    if sto:
        a = np.abs([d[4] for d in sto])
        print(f"\nStochastic types ({len(sto)} conditions; reported, no threshold): |delta| "
              f"median {np.median(a):.2f}, mean {a.mean():.2f}, p90 {np.percentile(a, 90):.2f}, max {a.max():.2f}")
        per = {}
        for t, s, _, _, d in sto: per.setdefault(t, []).append(d)
        for t, v in per.items():
            print(f"  {t:17s} " + " ".join(f"{x:+.2f}" for x in v))
        if miss_s: print("  (missing:", miss_s, ")")
    out = {"pool": POOL, "tol": TOL, "deterministic": det, "stochastic": sto,
           "passed": not fails, "n_fail": len(fails)}
    json.dump(out, open(os.path.join(RES, f"csc_v2_gate_{POOL}.json"), "w"), indent=2)
    print("\nGATE", "PASSED" if not fails else f"FAILED ({len(fails)}/{len(det)} misses) -> stop and debug")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
