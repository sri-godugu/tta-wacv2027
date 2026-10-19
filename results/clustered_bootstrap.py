#!/usr/bin/env python3
"""
clustered_bootstrap.py -- corruption-family-clustered bootstrap for the spine.

Reviewer objection (PL6X, M9tV): the 95 synthetic conditions are 19 corruption
TYPES x 5 severities, so conditions within a type are correlated and treating
them as 95 independent points overstates the precision of the difficulty fit
and the significance of the real-vs-trend shortfall.

This answers it two ways, both resampling at the granularity of the corruption
TYPE (the cluster), not the condition or the image:

  (1) Cluster bootstrap. Resample the 19 types with replacement (each type
      carries its 5 severities as a block), refit the OLS difficulty line on
      the resampled synthetic grid, recompute every real condition's residual,
      and record the mean over the 7 adverse conditions. Report the 95% CI of
      that mean and the fraction of resamples in which all 7 stay below trend.

  (2) Leave-one-corruption-type-out (LOCTO) jackknife. Drop each of the 19
      types in turn, refit, recompute the mean real shortfall; report the full
      range. If the conclusion depended on any single corruption family, this
      range would straddle zero.

Drop this in ~/robustseg_wacv/results/ (same dir as fit_robustness.py) and run.
Needs only the per-condition JSONs already used by the spine:
  pertype_vanilla_{type}.json, pertype_pdaonly_{type}.json  (primary R50)
Switch PREFIX_V / PREFIX_A to the r101_/fcn50_ variants to run the replication
backbones.
"""
import json, os
import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)           # fixed seed -> reproducible CI
B = 10000

TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
         "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
         "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
         "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
         "zoom_blur"]
ADVERSE = [("ACDC fog", "acdc_fog"), ("ACDC night", "acdc_night"),
           ("ACDC rain", "acdc_rain"), ("ACDC snow", "acdc_snow"),
           ("BDD rainy", "bdd_rainy"), ("BDD snowy", "bdd_snowy"),
           ("Foggy Zurich fog", "fz_fog")]

# --- primary spine. For the backbones use "r101_pertype_" / "fcn50_pertype_"
PREFIX_V = "pertype_vanilla_"
PREFIX_A = "pertype_pdaonly_"
REAL_V = "cm_{tag}_vanilla.json"         # primary real vanilla
REAL_A = "cm_{tag}_pdaonly.json"         # primary real adapted


def L(p):
    return json.load(open(os.path.join(RES, p)))


def sevs(p):
    return {r["severity"]: r["miou"] * 100 for r in L(p)["severity_results"]}


def scalar(p):
    return L(p)["metrics"]["miou"] * 100


def load_grid_by_type():
    """Return {type: (vanilla[], benefit[])} over its available severities."""
    g = {}
    for t in TYPES:
        v = sevs(f"{PREFIX_V}{t}.json")
        a = sevs(f"{PREFIX_A}{t}.json")
        xs = sorted(s for s in v if s in a)
        g[t] = (np.array([v[s] for s in xs]),
                np.array([a[s] - v[s] for s in xs]))
    return g


def load_real():
    out = []
    for label, tag in ADVERSE:
        van = scalar(REAL_V.format(tag=tag))
        ada = scalar(REAL_A.format(tag=tag))
        out.append((label, van, ada - van))
    return out


def fit(X, Y):
    b, a = np.polyfit(X, Y, 1)
    return a, b


def mean_shortfall(a, b, real):
    return np.mean([ben - (a + b * van) for _, van, ben in real])


def main():
    g = load_grid_by_type()
    real = load_real()

    # point estimate on the full grid
    X = np.concatenate([g[t][0] for t in TYPES])
    Y = np.concatenate([g[t][1] for t in TYPES])
    a0, b0 = fit(X, Y)
    pt = mean_shortfall(a0, b0, real)
    print(f"point estimate: slope {b0:+.4f}, mean real shortfall {pt:+.3f} mIoU "
          f"(n_grid={len(X)}, n_types={len(TYPES)})")

    # (1) cluster bootstrap over the 19 corruption types
    means, all7 = [], 0
    for _ in range(B):
        pick = rng.integers(0, len(TYPES), len(TYPES))
        Xb = np.concatenate([g[TYPES[i]][0] for i in pick])
        Yb = np.concatenate([g[TYPES[i]][1] for i in pick])
        a, b = fit(Xb, Yb)
        res = [ben - (a + b * van) for _, van, ben in real]
        means.append(np.mean(res))
        all7 += int(all(r < 0 for r in res))
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"(1) type-clustered bootstrap: mean shortfall "
          f"{np.mean(means):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"excludes 0: {hi < 0}")
    print(f"    all-7-below-trend in {100*all7/B:.1f}% of resamples")

    # (2) leave-one-corruption-type-out jackknife
    loco = []
    for drop in TYPES:
        keep = [t for t in TYPES if t != drop]
        Xk = np.concatenate([g[t][0] for t in keep])
        Yk = np.concatenate([g[t][1] for t in keep])
        a, b = fit(Xk, Yk)
        loco.append(mean_shortfall(a, b, real))
    print(f"(2) leave-one-type-out: mean shortfall in "
          f"[{min(loco):+.3f}, {max(loco):+.3f}] across the 19 drops")


if __name__ == "__main__":
    main()
