#!/usr/bin/env python3
"""
build_v2_data.py — feed fig4_stepcurve.py from the step-curve run outputs (results/stepcurve/{h200,v100}).

Writes into data/:
  v2_vanilla_{pool}_{type|cond}.json   vanilla per GPU pool (PDA/oracle pair with h200, TENT with v100), so every
                                       benefit is adapted - vanilla on the same pool and identical inputs
  v2s{K}_{method}_{type}.json          synthetic, severity_results (+ per-severity loss_trace); written only when all 5
                                       severities exist, so an incomplete setting drops out instead of being guessed
  v2s{K}_{method}_{cond}.json          real conditions, metrics (+ loss_trace)
  v2s1_lr{1e-4|3e-4}_{method}_*.json   LR axis, same layout
  stepcurve_ci.json                    paired joint-bootstrap CIs (copied verbatim)
methods: pdaonly (h200 'pda'), tent (v100 'tent'), oracle (h200 'ora').

Usage: python build_v2_data.py <results/stepcurve dir> [--ci <stepcurve_ci.json>]
"""
import argparse, json, os, shutil

TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
         "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
         "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
REAL = ["acdc_fog", "acdc_night", "acdc_rain", "acdc_snow", "bdd_rainy", "bdd_snowy", "bdd_clear", "fz_fog"]
METHODS = {"pdaonly": ("h200", "pda"), "tent": ("v100", "tent"), "oracle": ("h200", "ora")}
SETTINGS = [(k, "0.001", f"v2s{k}") for k in (1, 2, 3, 5, 10)] + \
           [(1, "0.0001", "v2s1_lr1e-4"), (1, "0.0003", "v2s1_lr3e-4")]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sev_entry(sev, m):
    e = {"severity": sev, "miou": m["miou"]}
    if m.get("mean_loss_trace"):
        e["loss_trace"] = m["mean_loss_trace"]
    return e


def synth(pool_dir, tag, t):
    """severity_results for one type from a per-type file or 5 per-severity files; None unless all 5 exist."""
    whole = J(os.path.join(pool_dir, f"{tag}_v2pertype_{t}.json"))
    if whole:
        rows = [sev_entry(r["severity"], r) for r in whole["severity_results"]]
    else:
        rows = []
        for s in range(1, 6):
            one = J(os.path.join(pool_dir, f"{tag}_v2pertype_{t}_sev{s}.json"))
            if one:
                rows.append(sev_entry(s, one["metrics"]))
    return rows if len({r["severity"] for r in rows}) == 5 else None


def real(pool_dir, tag, c):
    r = J(os.path.join(pool_dir, f"{tag}_{c}.json"))
    if r is None:
        return None
    m = {"miou": r["metrics"]["miou"]}
    if r["metrics"].get("mean_loss_trace"):
        m["loss_trace"] = r["metrics"]["mean_loss_trace"]
    return {"metrics": m}


def dump(name, obj):
    json.dump(obj, open(os.path.join(D, name), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--ci", default=None)
    a = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    n = 0
    for pool in ("h200", "v100"):
        pd = os.path.join(a.root, pool)
        for t in TYPES:
            rows = synth(pd, "van", t)
            if rows:
                dump(f"v2_vanilla_{pool}_{t}.json", {"severity_results": rows}); n += 1
        for c in REAL:
            r = real(pd, "van", c)
            if r:
                dump(f"v2_vanilla_{pool}_{c}.json", r); n += 1
    report = []
    for m, (pool, short) in METHODS.items():
        pd = os.path.join(a.root, pool)
        for k, lr, prefix in SETTINGS:
            tag = f"{short}_s{k}_lr{lr}"
            ns = nr = 0
            for t in TYPES:
                rows = synth(pd, tag, t)
                if rows:
                    dump(f"{prefix}_{m}_{t}.json", {"severity_results": rows}); ns += 1
            for c in REAL:
                r = real(pd, tag, c)
                if r:
                    dump(f"{prefix}_{m}_{c}.json", r); nr += 1
            n += ns + nr
            report.append(f"  {prefix:12s} {m:8s} synthetic types {ns:2d}/19  real {nr}/8")
    if a.ci:
        shutil.copyfile(a.ci, os.path.join(D, "stepcurve_ci.json"))   # content-only copy
    print(f"wrote {n} files to data/"); print("\n".join(report))


if __name__ == "__main__":
    main()
