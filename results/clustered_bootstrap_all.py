#!/usr/bin/env python3
"""
clustered_bootstrap_all.py -- run clustered_bootstrap.py's type-clustered bootstrap and
leave-one-type-out jackknife for every approach and replication backbone (Tab. S_x).

Imports clustered_bootstrap.py unmodified and swaps its file prefixes per configuration;
the RNG is re-seeded (seed 0, as in the original) before each one so every row is
reproducible on its own. Writes clustered_bootstrap_all.json next to this file.
"""
import io, json, os, contextlib
import numpy as np
import clustered_bootstrap as cb

# (label, synthetic vanilla prefix, synthetic adapted prefix, real vanilla, real adapted)
CONFIGS = [
    ("PDA (R50)",          "pertype_vanilla_",       "pertype_pdaonly_",       "cm_{tag}_vanilla.json", "cm_{tag}_pdaonly.json"),
    ("TENT (R50)",         "pertype_vanilla_",       "pertype_tent_",          "cm_{tag}_vanilla.json", "cm_{tag}_tent.json"),
    ("DUA (R50)",          "pertype_vanilla_",       "pertype_dua_",           "cm_{tag}_vanilla.json", "cm_{tag}_dua.json"),
    ("Self-training (R50)","pertype_vanilla_",       "pertype_pseudo_",        "cm_{tag}_vanilla.json", "cm_{tag}_pseudo.json"),
    ("PDA (R101)",         "r101_pertype_vanilla_",  "r101_pertype_pdaonly_",  "r101_vanilla_{tag}.json", "r101_pdaonly_{tag}.json"),
    ("PDA (FCN-R50)",      "fcn50_pertype_vanilla_", "fcn50_pertype_pdaonly_", "fcn50_vanilla_{tag}.json", "fcn50_pdaonly_{tag}.json"),
]


def run(label, pv, pa, rv, ra):
    cb.PREFIX_V, cb.PREFIX_A, cb.REAL_V, cb.REAL_A = pv, pa, rv, ra
    g, real = cb.load_grid_by_type(), cb.load_real()
    X = np.concatenate([g[t][0] for t in cb.TYPES]); Y = np.concatenate([g[t][1] for t in cb.TYPES])
    a0, b0 = cb.fit(X, Y)
    rng = np.random.default_rng(0)
    means, all7 = [], 0
    for _ in range(cb.B):
        pick = rng.integers(0, len(cb.TYPES), len(cb.TYPES))
        a, b = cb.fit(np.concatenate([g[cb.TYPES[i]][0] for i in pick]),
                      np.concatenate([g[cb.TYPES[i]][1] for i in pick]))
        res = [ben - (a + b * van) for _, van, ben in real]
        means.append(np.mean(res)); all7 += int(all(r < 0 for r in res))
    loco = []
    for drop in cb.TYPES:
        keep = [t for t in cb.TYPES if t != drop]
        a, b = cb.fit(np.concatenate([g[t][0] for t in keep]), np.concatenate([g[t][1] for t in keep]))
        loco.append(cb.mean_shortfall(a, b, real))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"label": label, "n_grid": int(len(X)), "slope": float(b0),
            "point_shortfall": float(cb.mean_shortfall(a0, b0, real)),
            "boot_mean": float(np.mean(means)), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(hi < 0), "all7_below_pct": 100.0 * all7 / cb.B,
            "loto_range": [float(min(loco)), float(max(loco))]}


def main():
    rows = [run(*c) for c in CONFIGS]
    print(f"{'config':20s} {'slope':>7s} {'point':>7s} {'boot 95% CI':>18s} {'all7<0':>7s} {'LOTO range':>18s}")
    for r in rows:
        print(f"{r['label']:20s} {r['slope']:+7.3f} {r['point_shortfall']:+7.2f} "
              f"[{r['ci95'][0]:+6.2f}, {r['ci95'][1]:+6.2f}] {r['all7_below_pct']:6.1f}% "
              f"[{r['loto_range'][0]:+6.2f}, {r['loto_range'][1]:+6.2f}]")
    out = os.path.join(cb.RES, "clustered_bootstrap_all.json")
    json.dump({"B": cb.B, "seed": 0, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
