#!/usr/bin/env python3
"""
stepcurve_analysis.py -- primary statistic of results/preregistration_step_curve.md.

For each method and setting (steps, lr), on Cityscapes-C v2 and within one GPU pool:
  refit benefit ~ vanilla mIoU (OLS, all 95 synthetic conditions, nothing excluded),
  place the 7 real adverse conditions against it (BDD clear reported separately),
and report mean synthetic benefit, mean real benefit, mean shortfall, #below/7, slope, R^2.
Benefit is always adapted minus the vanilla run of the SAME pool on identical inputs.

Usage: python stepcurve_analysis.py <stepcurve_dir> [--json out.json]
  <stepcurve_dir> holds h200/ and v100/ as produced by the runs (results/stepcurve).
Settings whose synthetic grid is incomplete are reported with n<95 and flagged.
"""
import argparse, json, os
import numpy as np

TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
         "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
         "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter",
         "speckle_noise", "zoom_blur"]
ADVERSE = ["acdc_fog", "acdc_night", "acdc_rain", "acdc_snow", "bdd_rainy", "bdd_snowy", "fz_fog"]
PROBE = "bdd_clear"
SETTINGS = [(1, "0.001"), (2, "0.001"), (3, "0.001"), (5, "0.001"), (10, "0.001"),
            (1, "0.0003"), (1, "0.0001")]
POOL_OF = {"pda": "h200", "ora": "h200", "tent": "v100"}


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def synth(d, tag, t):
    """{severity: mIoU*100} for one type; per-type file (all sev) or per-severity files."""
    out = {}
    whole = J(os.path.join(d, f"{tag}_v2pertype_{t}.json"))
    if whole:
        out = {r["severity"]: 100 * r["miou"] for r in whole["severity_results"]}
    for s in range(1, 6):
        one = J(os.path.join(d, f"{tag}_v2pertype_{t}_sev{s}.json"))
        if one:
            out[s] = 100 * one["metrics"]["miou"]
    return out


def real(d, tag, c):
    r = J(os.path.join(d, f"{tag}_{c}.json"))
    return None if r is None else 100 * r["metrics"]["miou"]


def analyse(root, method, steps, lr):
    d = os.path.join(root, POOL_OF[method])
    tag = f"{method}_s{steps}_lr{lr}"
    X, Y = [], []
    for t in TYPES:
        v, a = synth(d, "van", t), synth(d, tag, t)
        for s in sorted(set(v) & set(a)):
            X.append(v[s]); Y.append(a[s] - v[s])
    row = {"method": method, "pool": POOL_OF[method], "steps": steps, "lr": float(lr), "n_synth": len(X)}
    if len(X) < 10:
        return row
    X, Y = np.array(X), np.array(Y)
    b, a = np.polyfit(X, Y, 1)
    res = Y - (a + b * X)
    sd = res.std(ddof=1)
    r2 = 1 - (res ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    resid, bens = {}, {}
    for c in ADVERSE + [PROBE]:
        v, ad = real(d, "van", c), real(d, tag, c)
        if v is None or ad is None:
            continue
        bens[c] = ad - v
        resid[c] = bens[c] - (a + b * v)
    adv = [c for c in ADVERSE if c in resid]
    row.update({
        "slope": float(b), "intercept": float(a), "r2": float(r2), "resid_sd": float(sd),
        "mean_synth_benefit": float(Y.mean()),
        "n_real": len(adv),
        "mean_real_benefit": float(np.mean([bens[c] for c in adv])) if adv else None,
        "mean_shortfall": float(np.mean([resid[c] for c in adv])) if adv else None,
        "mean_shortfall_sd": float(np.mean([resid[c] for c in adv]) / sd) if adv else None,
        "n_below": int(sum(resid[c] < 0 for c in adv)),
        "probe_resid": float(resid[PROBE]) if PROBE in resid else None,
        "per_condition_resid": {c: float(resid[c]) for c in resid},
        "per_condition_benefit": {c: float(bens[c]) for c in bens},
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = [analyse(args.root, m, s, lr) for m in ["pda", "tent", "ora"] for s, lr in SETTINGS]
    print(f"{'method':5s} {'pool':5s} {'steps':>5s} {'lr':>7s} {'nSyn':>4s} {'slope':>7s} {'R2':>5s} "
          f"{'synBen':>7s} {'realBen':>8s} {'shortfall':>10s} {'(SD)':>6s} {'below':>6s} {'probe':>6s}")
    for r in rows:
        if "slope" not in r:
            print(f"{r['method']:5s} {r['pool']:5s} {r['steps']:5d} {r['lr']:7.4f} {r['n_synth']:4d}  (not enough data yet)")
            continue
        flag = "" if r["n_synth"] == 95 and r["n_real"] == 7 else "  <- incomplete"
        f = lambda x, w=7: f"{x:+{w}.2f}" if x is not None else " " * (w - 2) + "--"
        print(f"{r['method']:5s} {r['pool']:5s} {r['steps']:5d} {r['lr']:7.4f} {r['n_synth']:4d} "
              f"{r['slope']:+7.3f} {r['r2']:5.2f} {f(r['mean_synth_benefit'])} {f(r['mean_real_benefit'], 8)} "
              f"{f(r['mean_shortfall'], 10)} {f(r['mean_shortfall_sd'], 6)} {r['n_below']:>3d}/{r['n_real']} "
              f"{f(r['probe_resid'], 6)}{flag}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
