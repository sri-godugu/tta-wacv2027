#!/usr/bin/env python3
"""
stepcurve_ci.py -- uncertainty for the step curve (inference on the registered primary statistic).

For each method and each complete budget s (lr 1e-3), the mean real shortfall against the synthetic trend refit at
that budget, with a JOINT bootstrap:
  - synthetic side: resample the 19 corruption TYPES with replacement (each carries its 5 severities), refit OLS;
  - real side: resample IMAGES within each real adverse condition (per-image confusion matrices), recompute
    vanilla and adapted mIoU on the same draw (paired), hence benefit and residual.
The same type and image draws are reused across budgets, so shortfall(s) - shortfall(1) gets a paired interval:
that is the "is the growth with budget real?" test.

Checks first that summed per-image CMs reproduce each run's reported mIoU.
Usage: python stepcurve_ci.py <stepcurve_dir> [--B 2000] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stepcurve_analysis as sa

STEPS = [1, 2, 3, 5, 10]


def miou_from_cm(M):
    tp = np.diag(M); den = M.sum(0) + M.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(den > 0, tp / den, np.nan)
    return 100 * np.nanmean(iou)


def load_cms(d, tag, c):
    return np.load(os.path.join(d, "per_image_cm", f"{tag}_{c}.npz"))["cm"].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--json", default=None); ap.add_argument("--methods", nargs="+", default=["pda", "tent"])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for m in a.methods:
        d = os.path.join(a.root, sa.POOL_OF[m])
        steps = [s for s in STEPS if (lambda r: r.get("n_synth") == 95 and r.get("n_real") == 7)(sa.analyse(a.root, m, s, "0.001"))]
        if not steps:
            continue
        # synthetic: per setting, per type -> arrays over severities (aligned)
        syn = {}
        for s in steps:
            tag = f"{m}_s{s}_lr0.001"; per = []
            for t in sa.TYPES:
                v, ad = sa.synth(d, "van", t), sa.synth(d, tag, t)
                sev = sorted(set(v) & set(ad))
                per.append((np.array([v[k] for k in sev]), np.array([ad[k] - v[k] for k in sev])))
            syn[s] = per
        # real: per-image CMs
        van = {c: load_cms(d, "van", c) for c in sa.ADVERSE}
        ada = {s: {c: load_cms(d, f"{m}_s{s}_lr0.001", c) for c in sa.ADVERSE} for s in steps}
        # consistency check against reported mIoU
        worst = 0.0
        for c in sa.ADVERSE:
            worst = max(worst, abs(miou_from_cm(van[c].sum(0)) - sa.real(d, "van", c)))
            for s in steps:
                worst = max(worst, abs(miou_from_cm(ada[s][c].sum(0)) - sa.real(d, f"{m}_s{s}_lr0.001", c)))
        assert worst < 1e-3, f"per-image CMs do not reproduce reported mIoU (max diff {worst})"

        def shortfall(type_idx, img_idx):
            res = {}
            for s in steps:
                X = np.concatenate([syn[s][i][0] for i in type_idx]); Y = np.concatenate([syn[s][i][1] for i in type_idx])
                b, a0 = np.polyfit(X, Y, 1)
                r = []
                for c in sa.ADVERSE:
                    w = img_idx[c]
                    vm = miou_from_cm(np.tensordot(w, van[c], 1)); am = miou_from_cm(np.tensordot(w, ada[s][c], 1))
                    r.append((am - vm) - (a0 + b * vm))
                res[s] = (np.mean(r), sum(x < 0 for x in r))
            return res

        full_t = np.arange(len(sa.TYPES))
        full_i = {c: np.ones(len(van[c])) for c in sa.ADVERSE}
        point = shortfall(full_t, full_i)
        draws = {s: [] for s in steps}; all7 = {s: 0 for s in steps}
        for _ in range(a.B):
            ti = rng.integers(0, len(sa.TYPES), len(sa.TYPES))
            ii = {c: np.bincount(rng.integers(0, len(van[c]), len(van[c])), minlength=len(van[c])).astype(float)
                  for c in sa.ADVERSE}
            r = shortfall(ti, ii)
            for s in steps:
                draws[s].append(r[s][0]); all7[s] += int(r[s][1] == 7)
        print(f"\n===== {m.upper()} ({sa.POOL_OF[m]})  joint bootstrap B={a.B} (types x images, paired across budgets); "
              f"CM check max |diff| {worst:.1e}")
        print(f"{'steps':>5s} {'shortfall':>10s} {'95% CI':>18s} {'all 7 below':>12s} | {'vs 1 step':>10s} {'95% CI':>18s} {'P(diff<0)':>9s}")
        res = {}
        base = np.array(draws[1]) if 1 in steps else None
        for s in steps:
            x = np.array(draws[s]); lo, hi = np.percentile(x, [2.5, 97.5])
            row = {"point": float(point[s][0]), "ci": [float(lo), float(hi)], "all7_pct": 100 * all7[s] / a.B}
            line = f"{s:5d} {point[s][0]:+10.2f} [{lo:+7.2f}, {hi:+7.2f}] {100*all7[s]/a.B:11.1f}%"
            if base is not None and s != 1:
                dd = x - base; dlo, dhi = np.percentile(dd, [2.5, 97.5])
                row.update({"diff_vs_1": float(point[s][0] - point[1][0]), "diff_ci": [float(dlo), float(dhi)],
                            "p_diff_lt0": float((dd < 0).mean())})
                line += f" | {point[s][0]-point[1][0]:+10.2f} [{dlo:+7.2f}, {dhi:+7.2f}] {(dd<0).mean():9.3f}"
            print(line); res[s] = row
        out[m] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
