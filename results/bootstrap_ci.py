#!/usr/bin/env python3
"""
bootstrap_ci.py -- paired bootstrap CIs on the spine residuals.

For each real condition we hold per-image confusion matrices for the SAME image
set under vanilla / PDA-only / TENT (results/per_image_cm/<tag>_<method>.npz).
A resample draws image (or batch) indices once and applies them to BOTH runs,
so vanilla mIoU, adapted mIoU, the benefit, and the fit's prediction all move
together -- which is the point: the residual's noise comes from both terms.

The synthetic fit is treated as FIXED (the pinned primary fit (a), n=95).
A secondary variant additionally resamples the 95 synthetic points, propagating
the fit's own uncertainty into the residual interval.

"""
import json, os, glob
import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
CMDIR = os.path.join(RES, "per_image_cm")
B = 10000
RNG = np.random.default_rng(0)

CONDITIONS = [
    # label, tag, kind ("adverse" or "probe"), n_expected
    ("ACDC fog",         "acdc_fog",   "adverse"),
    ("ACDC night",       "acdc_night", "adverse"),
    ("ACDC rain",        "acdc_rain",  "adverse"),
    ("ACDC snow",        "acdc_snow",  "adverse"),
    ("BDD rainy",        "bdd_rainy",  "adverse"),
    ("BDD snowy",        "bdd_snowy",  "adverse"),
    ("Foggy Zurich fog", "fz_fog",     "adverse"),
    ("BDD clear",        "bdd_clear",  "probe"),
]


def miou_from_cm(cm):
    """mIoU from a (…,19,19) confusion matrix, matching SegmentationMetrics."""
    tp = np.einsum("...ii->...i", cm).astype(np.float64)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = np.where(valid, tp / (denom + 1e-8), 0.0)
    return np.where(valid.any(-1), iou.sum(-1) / np.maximum(valid.sum(-1), 1), 0.0) * 100


def load(tag, method):
    z = np.load(os.path.join(CMDIR, f"{tag}_{method}.npz"))
    return z["cm"].astype(np.float64), z["batch_id"]


def sevs(path):
    d = json.load(open(path))
    out = {}
    for s in d["severity_results"]:
        k = [x for x in s if "miou" in x.lower()][0]
        out[s["severity"]] = s[k] * 100 if s[k] < 1.5 else s[k]
    return out


def synthetic_points(method):
    X, Y = [], []
    for fv in sorted(glob.glob(os.path.join(RES, "pertype_vanilla_*.json"))):
        c = os.path.basename(fv)[len("pertype_vanilla_"):-5]
        fm = os.path.join(RES, f"pertype_{method}_{c}.json")
        if not os.path.exists(fm):
            continue
        v, a = sevs(fv), sevs(fm)
        for s in sorted(v):
            if s in a:
                X.append(v[s]); Y.append(a[s] - v[s])
    return np.array(X), np.array(Y)


def fit_line(X, Y):
    b, a = np.polyfit(X, Y, 1)
    res = Y - (a + b * X)
    return a, b, res.std(ddof=1)


def resample_weights(n_units, unit_sizes=None, groups=None):
    """Multinomial counts over resampling units, expanded to per-image weights."""
    counts = RNG.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=B)  # (B, n_units)
    if groups is None:
        return counts.astype(np.float64)
    W = np.zeros((B, len(groups)), dtype=np.float64)
    for u in range(n_units):
        W[:, groups == u] += counts[:, u][:, None]
    return W


def main():
    fits = {m: fit_line(*synthetic_points(m)) for m in ("pdaonly", "tent", "dua", "pseudo")}
    for m, (a, b, s) in fits.items():
        print(f"synthetic fit [{m}]: benefit = {b:+.3f}*vanilla + {a:.2f}, resid std {s:.2f}")

    rows = []
    for method in ("pdaonly", "tent", "dua", "pseudo"):
        a, b, std = fits[method]
        Xs, Ys = synthetic_points(method)
        print(f"\n{'='*104}\n{method}\n{'='*104}")
        print(f"{'condition':18s} {'n':>4s} {'van':>6s} {'ada':>6s} {'benefit':>8s} "
              f"{'resid':>7s} {'SD':>6s}   {'resid 95% CI (image)':>24s}  {'(batch)':>18s}  excl0")
        for label, tag, kind in CONDITIONS:
            cm_v, bid = load(tag, "vanilla")
            cm_a, bid_a = load(tag, method)
            assert cm_v.shape == cm_a.shape, f"{tag}: shape mismatch {cm_v.shape} vs {cm_a.shape}"
            assert (bid == bid_a).all(), f"{tag}: batch layout differs between runs"
            n = cm_v.shape[0]
            flat_v, flat_a = cm_v.reshape(n, -1), cm_a.reshape(n, -1)

            van = miou_from_cm(cm_v.sum(0)); ada = miou_from_cm(cm_a.sum(0))
            resid = (ada - van) - (a + b * van)

            out = {}
            for scheme in ("image", "batch"):
                if scheme == "image":
                    W = resample_weights(n)
                else:
                    ub = np.unique(bid); g = np.searchsorted(ub, bid)
                    W = resample_weights(len(ub), groups=g)
                bv = miou_from_cm((W @ flat_v).reshape(B, 19, 19))
                ba = miou_from_cm((W @ flat_a).reshape(B, 19, 19))
                r = (ba - bv) - (a + b * bv)
                out[scheme] = np.percentile(r, [2.5, 97.5])

            # secondary: also resample the 95 synthetic points (fit uncertainty)
            W = resample_weights(n)
            bv = miou_from_cm((W @ flat_v).reshape(B, 19, 19))
            ba = miou_from_cm((W @ flat_a).reshape(B, 19, 19))
            idx = RNG.integers(0, len(Xs), size=(B, len(Xs)))
            aa = np.empty(B); bb = np.empty(B)
            for i in range(B):
                bb[i], aa[i] = np.polyfit(Xs[idx[i]], Ys[idx[i]], 1)
            r_fit = (ba - bv) - (aa + bb * bv)
            ci_fit = np.percentile(r_fit, [2.5, 97.5])

            excl = "YES" if out["image"][1] < 0 else "no"
            print(f"{label:18s} {n:4d} {van:6.2f} {ada:6.2f} {ada-van:+8.2f} "
                  f"{resid:+7.2f} {resid/std:+6.2f}   [{out['image'][0]:+6.2f}, {out['image'][1]:+6.2f}]"
                  f"      [{out['batch'][0]:+6.2f}, {out['batch'][1]:+6.2f}]   {excl}")
            rows.append({"condition": label, "tag": tag, "kind": kind, "method": method,
                         "n_images": int(n), "vanilla": van, "adapted": ada,
                         "benefit": ada - van, "resid": resid, "resid_sd": resid / std,
                         "ci_image": list(out["image"]), "ci_batch": list(out["batch"]),
                         "ci_image_plus_fit": list(ci_fit)})

        adv = [r for r in rows if r["method"] == method and r["kind"] == "adverse"]
        below = sum(1 for r in adv if r["resid"] < 0)
        print(f"  adverse only: {below}/{len(adv)} below, mean {np.mean([r['resid_sd'] for r in adv]):+.2f} SD, "
              f"raw shortfall {min(r['resid'] for r in adv):+.1f} to {max(r['resid'] for r in adv):+.1f} mIoU")

    json.dump({"fits": {m: {"a": f[0], "b": f[1], "std": f[2]} for m, f in fits.items()},
               "B": B, "points": rows},
              open(os.path.join(RES, "bootstrap_ci.json"), "w"), indent=1)
    print("\nwrote bootstrap_ci.json")


if __name__ == "__main__":
    main()
