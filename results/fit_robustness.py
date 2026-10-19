#!/usr/bin/env python3
"""
fit_robustness.py -- two pre-emptive robustness checks on the spine.

(A) DATASET LEAVE-OUT. Does the result survive dropping BDD100K entirely (the
    dataset with the annotation-policy confound)? Reports the ACDC+Foggy-Zurich
    subset separately, and every single-dataset leave-one-out.

(B) TREND SPECIFICATION. The matched-difficulty argument rests on a fitted
    synthetic trend. If the shortfall is an artifact of assuming linearity it
    should weaken under a more flexible or more robust fit. We re-place every
    real condition against: OLS linear (primary), OLS quadratic, Theil-Sen
    (robust to the high-leverage saturate points), and LOESS (nonparametric).
    Residual std is recomputed per specification so SD units stay comparable.

Run from ~/robustseg_wacv/results/ inside a Slurm allocation.
"""
import json, os, glob
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
METHODS = {"pdaonly": "pertype_pdaonly", "tent": "pertype_tent",
           "dua": "pertype_dua", "pseudo": "pertype_pseudo"}
# how each method's REAL-condition result files are named
REALFILE = {"pdaonly": "cm_{t}_pdaonly", "tent": "cm_{t}_tent",
            "dua": "dua_{t}", "pseudo": "pseudo_{t}"}


def L(p): return json.load(open(os.path.join(RES, p)))
def sevs(p): return {r["severity"]: r["miou"] * 100 for r in L(p)["severity_results"]}
def scalar(p): return L(p)["metrics"]["miou"] * 100


def grid(prefix):
    X, Y = [], []
    for t in TYPES:
        v = sevs(f"pertype_vanilla_{t}.json"); a = sevs(f"{prefix}_{t}.json")
        for s in sorted(v):
            if s in a: X.append(v[s]); Y.append(a[s] - v[s])
    return np.array(X), np.array(Y)


def real(method):
    out = []
    for label, tag, ds in ADVERSE:
        van = scalar(f"cm_{tag}_vanilla.json")
        ada = scalar(REALFILE[method].format(t=tag) + ".json")
        out.append({"label": label, "dataset": ds, "vanilla": van, "benefit": ada - van})
    return out


# ---- trend specifications: each returns predict(x_new) and in-sample residual std
def spec_linear(X, Y):
    b, a = np.polyfit(X, Y, 1)
    f = lambda x: a + b * np.asarray(x)
    return f, float(np.std(Y - f(X), ddof=1))


def spec_quadratic(X, Y):
    c2, c1, c0 = np.polyfit(X, Y, 2)
    f = lambda x: c0 + c1 * np.asarray(x) + c2 * np.asarray(x) ** 2
    return f, float(np.std(Y - f(X), ddof=1))


def spec_theilsen(X, Y):
    """Median-of-pairwise-slopes: insensitive to the high-leverage saturate points."""
    idx = np.triu_indices(len(X), k=1)
    dx = X[idx[1]] - X[idx[0]]
    ok = np.abs(dx) > 1e-9
    slopes = (Y[idx[1]] - Y[idx[0]])[ok] / dx[ok]
    b = float(np.median(slopes))
    a = float(np.median(Y - b * X))
    f = lambda x: a + b * np.asarray(x)
    return f, float(np.std(Y - f(X), ddof=1))


def spec_loess(X, Y, frac=0.5):
    """Local linear regression, tricube weights -- no functional form assumed."""
    order = np.argsort(X); Xs, Ys = X[order], Y[order]
    k = max(3, int(frac * len(Xs)))

    def f(xq):
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        out = []
        for x0 in xq:
            d = np.abs(Xs - x0)
            h = np.sort(d)[k - 1] or 1e-9
            w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
            W = np.diag(w)
            A = np.vstack([np.ones_like(Xs), Xs]).T
            beta = np.linalg.pinv(A.T @ W @ A) @ (A.T @ W @ Ys)
            out.append(beta[0] + beta[1] * x0)
        return np.array(out) if len(out) > 1 else out[0]
    resid = Ys - np.array([f(x) for x in Xs])
    return f, float(np.std(resid, ddof=1))


SPECS = {"linear (primary)": spec_linear, "quadratic": spec_quadratic,
         "Theil-Sen (robust)": spec_theilsen, "LOESS (nonparametric)": spec_loess}


def main():
    report = {}
    for method, prefix in METHODS.items():
        X, Y = grid(prefix)
        pts = real(method)
        print("=" * 92)
        print(f"{method}   (synthetic grid n={len(X)})")
        print("=" * 92)
        report[method] = {}

        # ---------- (B) trend specification ----------
        print(f"\n  {'specification':22s} {'std':>6s}  {'7 adverse':>22s}   {'ACDC+FZ only (no BDD)':>24s}")
        for name, fn in SPECS.items():
            f, sd = fn(X, Y)
            res = [(p, p["benefit"] - float(f(p["vanilla"]))) for p in pts]
            allr = [r for _, r in res]
            nob = [r for p, r in res if p["dataset"] != "BDD"]
            print(f"  {name:22s} {sd:6.2f}  {sum(1 for r in allr if r<0)}/7 below, "
                  f"mean {np.mean(allr):+5.2f} mIoU ({np.mean(allr)/sd:+5.2f} SD)   "
                  f"{sum(1 for r in nob if r<0)}/5 below, mean {np.mean(nob):+5.2f} mIoU")
            report[method][name] = {
                "resid_std": sd,
                "n_below_7": int(sum(1 for r in allr if r < 0)),
                "mean_resid": float(np.mean(allr)),
                "mean_resid_sd": float(np.mean(allr) / sd),
                "n_below_no_bdd": int(sum(1 for r in nob if r < 0)),
                "mean_resid_no_bdd": float(np.mean(nob)),
                "per_condition": {p["label"]: float(r) for p, r in res},
            }

        # ---------- (A) dataset leave-one-out, under the primary fit ----------
        f, sd = spec_linear(X, Y)
        res = [(p, p["benefit"] - float(f(p["vanilla"]))) for p in pts]
        print(f"\n  dataset leave-one-out (primary linear fit):")
        for drop in ("BDD", "ACDC", "FZ"):
            keep = [(p, r) for p, r in res if p["dataset"] != drop]
            rr = [r for _, r in keep]
            print(f"    drop {drop:5s} -> {sum(1 for r in rr if r<0)}/{len(rr)} below, "
                  f"mean {np.mean(rr):+5.2f} mIoU ({np.mean(rr)/sd:+5.2f} SD)")
            report[method][f"drop_{drop}"] = {
                "n_below": int(sum(1 for r in rr if r < 0)), "n": len(rr),
                "mean_resid": float(np.mean(rr)), "mean_resid_sd": float(np.mean(rr) / sd)}
        print()

    json.dump(report, open(os.path.join(RES, "fit_robustness.json"), "w"), indent=1)
    print("wrote fit_robustness.json")


if __name__ == "__main__":
    main()
