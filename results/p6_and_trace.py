#!/usr/bin/env python3
"""
p6_and_trace.py -- two pre-registered real-suite analyses (results/preregistration_step_curve.md).

P6  Sky->building across the step axis: per real condition, the share of GT-sky pixels predicted as
    building and the sky IoU, at steps {1,2,3,5,10} (lr 1e-3), for PDA (h200) and TENT (v100),
    each against its own pool's vanilla. Prediction: on ACDC fog the rate is non-decreasing in s.
Trace  Objective vs accuracy: mean pre-update loss at each step (mean_loss_trace of the 10-step run)
    next to mIoU at s in {1,2,3,5,10}, real vs synthetic (synthetic: fog sev3, the ACDC-fog match).

Usage: python p6_and_trace.py <stepcurve_dir>
"""
import json, os, sys
import numpy as np

ROOT = sys.argv[1]
SKY, BLD = 10, 2
STEPS = [1, 2, 3, 5, 10]
CONDS = ["acdc_fog", "acdc_rain", "acdc_snow", "fz_fog", "acdc_night", "bdd_rainy", "bdd_snowy", "bdd_clear"]
POOL = {"pda": "h200", "tent": "v100"}


def cm(pool, tag, c):
    p = os.path.join(ROOT, pool, "per_image_cm", f"{tag}_{c}.npz")
    return np.load(p)["cm"].astype(np.float64).sum(0) if os.path.exists(p) else None


def sky_stats(M):
    rate = 100 * M[SKY, BLD] / M[SKY].sum()
    tp = M[SKY, SKY]
    iou = 100 * tp / (M[SKY].sum() + M[:, SKY].sum() - tp)
    return rate, iou


print("P6  GT-sky -> building (%) and sky IoU, by adaptation steps (lr 1e-3); vanilla of the same pool first")
for m in ["pda", "tent"]:
    pool = POOL[m]
    print(f"\n[{m.upper()} on {pool}]  {'cond':11s} {'van':>12s} " + " ".join(f"{'s'+str(s):>12s}" for s in STEPS) + "   monotone?")
    for c in CONDS:
        v = cm(pool, "van", c)
        if v is None:
            continue
        cells = [sky_stats(v)]
        for s in STEPS:
            M = cm(pool, f"{m}_s{s}_lr0.001", c)
            cells.append(sky_stats(M) if M is not None else (np.nan, np.nan))
        rates = [r for r, _ in cells[1:]]
        mono = all(b >= a - 0.5 for a, b in zip(rates, rates[1:])) if not np.isnan(rates).any() else None
        print(f"{'':16s} {c:11s} " + " ".join(f"{r:5.1f}/{i:5.1f} " for r, i in cells)
              + f"  {'yes' if mono else ('no' if mono is not None else '--')}")

print("\nTrace  objective (pre-update loss per step, 10-step run) vs mIoU at each budget")
for m in ["pda", "tent"]:
    pool = POOL[m]
    for label, f10, fstep in [
        ("ACDC fog (real)", f"{m}_s10_lr0.001_acdc_fog.json", lambda s: f"{m}_s{s}_lr0.001_acdc_fog.json"),
    ]:
        j = json.load(open(os.path.join(ROOT, pool, f10)))
        tr = j["metrics"].get("mean_loss_trace")
        van = json.load(open(os.path.join(ROOT, pool, "van_acdc_fog.json")))["metrics"]["miou"] * 100
        mi = []
        for s in STEPS:
            p = os.path.join(ROOT, pool, fstep(s))
            mi.append(json.load(open(p))["metrics"]["miou"] * 100 if os.path.exists(p) else np.nan)
        print(f"  {m.upper():4s} {label:16s} loss s1..s10: " + (" ".join(f"{x:.3f}" for x in tr) if tr else "n/a"))
        print(f"  {'':4s} {'':16s} mIoU van {van:.1f} | " + " ".join(f"s{s} {x:.1f}" for s, x in zip(STEPS, mi)))
    # synthetic fog sev3 (matched difficulty), from the per-severity or per-type files
    d = os.path.join(ROOT, pool)
    def syn(tag):
        p1 = os.path.join(d, f"{tag}_v2pertype_fog_sev3.json")
        if os.path.exists(p1):
            return json.load(open(p1))["metrics"]
        p2 = os.path.join(d, f"{tag}_v2pertype_fog.json")
        if os.path.exists(p2):
            return [r for r in json.load(open(p2))["severity_results"] if r["severity"] == 3][0]
        return None
    r10 = syn(f"{m}_s10_lr0.001"); rv = syn("van")
    if r10 and rv:
        tr = r10.get("mean_loss_trace")
        mi = [syn(f"{m}_s{s}_lr0.001") for s in STEPS]
        print(f"  {m.upper():4s} {'syn fog sev3':16s} loss s1..s10: " + (" ".join(f"{x:.3f}" for x in tr) if tr else "n/a"))
        print(f"  {'':4s} {'':16s} mIoU van {100*rv['miou']:.1f} | " + " ".join(
            f"s{s} {100*x['miou']:.1f}" if x else f"s{s} --" for s, x in zip(STEPS, mi)))
