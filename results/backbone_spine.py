#!/usr/bin/env python3
"""
backbone_spine.py -- does the synthetic-to-real shortfall reproduce on other
BatchNorm-rich architectures?

Each backbone is analysed entirely on its own terms: its own vanilla baseline,
its own synthetic difficulty fit over the 95 Cityscapes-C conditions, and its
own residual std. Nothing is borrowed from the primary backbone, so a
replication here is a genuine independent instance of the claim rather than a
re-scoring of the same numbers.

Backbones (all BatchNorm-rich; BN-layer counts measured on the built models):
  dlv3_r50  : DeepLabV3+/ResNet-50   60 BN   (primary, from the main tables)
  r101      : DeepLabV3+/ResNet-101 111 BN   (depth axis)
  fcn50     : FCN/ResNet-50           54 BN  (head axis, no ASPP)

Run from ~/robustseg_wacv/results/ inside a Slurm allocation.
"""
import json, os
import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
         "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
         "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
         "snow", "spatter", "speckle_noise", "zoom_blur"]
ADVERSE = [("ACDC fog", "acdc_fog", "ACDC"), ("ACDC night", "acdc_night", "ACDC"),
           ("ACDC rain", "acdc_rain", "ACDC"), ("ACDC snow", "acdc_snow", "ACDC"),
           ("BDD rainy", "bdd_rainy", "BDD"), ("BDD snowy", "bdd_snowy", "BDD"),
           ("Foggy Zurich fog", "fz_fog", "FZ")]
PROBE = ("BDD clear", "bdd_clear", "BDD")

# name -> (synthetic file pattern, real file pattern, BN count, description)
BACKBONES = {
    "DeepLabV3+/R50 (primary)": ("pertype_{m}_{t}", None, 60, "ASPP head, depth 50"),
    "DeepLabV3+/R101":          ("r101_pertype_{m}_{t}", "r101_{m}_{tag}", 111, "ASPP head, depth 101"),
    "FCN/R50":                  ("fcn50_pertype_{m}_{t}", "fcn50_{m}_{tag}", 54, "FCN head, depth 50"),
}


def L(p): return json.load(open(os.path.join(RES, p)))
def sevs(p): return {r["severity"]: r["miou"] * 100 for r in L(p)["severity_results"]}
def scalar(p): return L(p)["metrics"]["miou"] * 100


def ols(X, Y):
    n = len(X); mx = X.mean(); my = Y.mean()
    b = ((X - mx) * (Y - my)).sum() / ((X - mx) ** 2).sum()
    a = my - b * mx
    r = Y - (a + b * X)
    return a, b, float(r.std(ddof=1)), float(1 - (r ** 2).sum() / ((Y - my) ** 2).sum())


def main():
    out = {}
    for name, (synpat, realpat, nbn, desc) in BACKBONES.items():
        X, Y, missing = [], [], []
        for t in TYPES:
            fv = synpat.format(m="vanilla", t=t) + ".json"
            fa = synpat.format(m="pdaonly", t=t) + ".json"
            if not (os.path.exists(os.path.join(RES, fv)) and os.path.exists(os.path.join(RES, fa))):
                missing.append(t); continue
            v, a = sevs(fv), sevs(fa)
            for s in sorted(v):
                if s in a: X.append(v[s]); Y.append(a[s] - v[s])
        if missing:
            print(f"!! {name}: missing {len(missing)} types -> {missing}")
        X, Y = np.array(X), np.array(Y)
        a, b, sd, r2 = ols(X, Y)
        print("=" * 96)
        print(f"{name}   [{desc}, {nbn} BN layers]")
        print(f"  own synthetic fit: benefit = {b:+.3f}*vanilla + {a:.2f}   "
              f"std {sd:.2f}, R2 {r2:.3f}, n={len(X)}, mean benefit {Y.mean():+.2f}")

        if realpat is None:   # primary: real numbers already in the main tables
            print("  (real-condition residuals are Table 2 of the paper)\n")
            out[name] = {"fit": {"a": a, "b": b, "std": sd, "r2": r2, "n": int(len(X)),
                                 "mean_benefit": float(Y.mean())}, "bn_layers": nbn}
            continue

        print(f"\n  {'condition':18s} {'van':>6s} {'PDA':>7s} {'benefit':>8s} {'pred':>7s} "
              f"{'resid':>7s} {'SD':>6s}")
        rows, sds = [], []
        for label, tag, ds in ADVERSE + [PROBE]:
            fv = realpat.format(m="vanilla", tag=tag) + ".json"
            fa = realpat.format(m="pdaonly", tag=tag) + ".json"
            if not (os.path.exists(os.path.join(RES, fv)) and os.path.exists(os.path.join(RES, fa))):
                print(f"  {label:18s} [missing]"); continue
            van, ada = scalar(fv), scalar(fa)
            pred = a + b * van
            res = (ada - van) - pred
            kind = "probe" if label == PROBE[0] else "adverse"
            rows.append({"condition": label, "kind": kind, "dataset": ds, "vanilla": van,
                         "adapted": ada, "benefit": ada - van, "pred": pred,
                         "resid": res, "resid_sd": res / sd})
            if kind == "adverse": sds.append(res / sd)
            print(f"  {label:18s} {van:6.2f} {ada:7.2f} {ada-van:+8.2f} {pred:+7.2f} "
                  f"{res:+7.2f} {res/sd:+6.2f}{'   (probe)' if kind=='probe' else ''}")
        adv = [r for r in rows if r["kind"] == "adverse"]
        below = sum(1 for s in sds if s < 0)
        nob = [r for r in adv if r["dataset"] != "BDD"]
        print(f"\n  adverse: {below}/{len(sds)} below, mean {np.mean(sds):+.2f} SD, "
              f"mean raw {np.mean([r['resid'] for r in adv]):+.2f} mIoU")
        print(f"  without BDD: {sum(1 for r in nob if r['resid']<0)}/{len(nob)} below, "
              f"mean raw {np.mean([r['resid'] for r in nob]):+.2f} mIoU")
        print(f"  => {'REPLICATES' if below == len(sds) else 'DOES NOT fully replicate - report honestly'}\n")
        out[name] = {"fit": {"a": a, "b": b, "std": sd, "r2": r2, "n": int(len(X)),
                             "mean_benefit": float(Y.mean())},
                     "bn_layers": nbn, "points": rows,
                     "n_below": below, "n_adverse": len(sds),
                     "mean_resid_sd": float(np.mean(sds)),
                     "mean_resid_raw": float(np.mean([r["resid"] for r in adv]))}

    json.dump(out, open(os.path.join(RES, "backbone_spine.json"), "w"), indent=1)
    print("wrote backbone_spine.json")


if __name__ == "__main__":
    main()
