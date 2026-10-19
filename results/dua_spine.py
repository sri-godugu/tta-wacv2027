#!/usr/bin/env python3
"""
dua_spine.py -- places the real conditions against DUA's OWN synthetic
difficulty fit, exactly as done for PDA and TENT.

DUA (Mirza et al., CVPR 2022) adapts BatchNorm *statistics* by forward passes
rather than *affine parameters* by gradients, so it is a second adaptation
mechanism. If the shortfall replicates here, the claim strengthens from
"two methods, one family" to "three methods, two mechanisms".
Run from ~/robustseg_wacv/results/ inside a Slurm allocation.
"""
import json, os, glob

RES = os.path.dirname(os.path.abspath(__file__))
ADVERSE = [("ACDC fog", "acdc_fog"), ("ACDC night", "acdc_night"), ("ACDC rain", "acdc_rain"),
           ("ACDC snow", "acdc_snow"), ("BDD rainy", "bdd_rainy"), ("BDD snowy", "bdd_snowy"),
           ("Foggy Zurich fog", "fz_fog")]
PROBE = [("BDD clear", "bdd_clear")]


def L(p): return json.load(open(os.path.join(RES, p)))
def sevs(p): return {r["severity"]: r["miou"] * 100 for r in L(p)["severity_results"]}
def scalar(p): return L(p)["metrics"]["miou"] * 100


def ols(X, Y):
    n = len(X); mx = sum(X) / n; my = sum(Y) / n
    b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    a = my - b * mx
    r = [y - (a + b * x) for x, y in zip(X, Y)]
    ss = sum(v * v for v in r)
    return a, b, (ss / (n - 1)) ** 0.5, 1 - ss / sum((y - my) ** 2 for y in Y)


def main():
    X, Y = [], []
    for fv in sorted(glob.glob(os.path.join(RES, "pertype_vanilla_*.json"))):
        t = os.path.basename(fv)[len("pertype_vanilla_"):-5]
        fd = os.path.join(RES, f"pertype_dua_{t}.json")
        if not os.path.exists(fd):
            print(f"  [missing] pertype_dua_{t}.json"); continue
        v, d = sevs(f"pertype_vanilla_{t}.json"), sevs(f"pertype_dua_{t}.json")
        for s in sorted(v):
            if s in d: X.append(v[s]); Y.append(d[s] - v[s])
    a, b, sd, r2 = ols(X, Y)
    print(f"DUA synthetic difficulty fit: benefit = {b:+.3f}*vanilla + {a:.2f}")
    print(f"  residual std {sd:.2f}, R2 {r2:.3f}, n={len(X)}, mean benefit {sum(Y)/len(Y):+.2f} mIoU")

    print(f"\n  {'condition':18s} {'van':>6s} {'DUA':>7s} {'benefit':>8s} {'pred':>7s} "
          f"{'resid':>7s} {'SD':>6s}  {'nbrs':>5s}")
    rows, sds = [], []
    for label, tag in ADVERSE + PROBE:
        van = scalar(f"cm_{tag}_vanilla.json")
        ada = scalar(f"dua_{tag}.json")
        pred = a + b * van
        res = (ada - van) - pred
        near = sum(1 for x in X if abs(x - van) <= 3)
        kind = "adverse" if (label, tag) in ADVERSE else "probe"
        rows.append({"condition": label, "kind": kind, "vanilla": van, "adapted": ada,
                     "benefit": ada - van, "pred": pred, "resid": res, "resid_sd": res / sd,
                     "neighbors_pm3": near})
        if kind == "adverse": sds.append(res / sd)
        print(f"  {label:18s} {van:6.2f} {ada:7.2f} {ada-van:+8.2f} {pred:+7.2f} "
              f"{res:+7.2f} {res/sd:+6.2f}  {near:5d}")
    below = sum(1 for s in sds if s < 0)
    print(f"\n  adverse: {below}/{len(sds)} below the line, mean {sum(sds)/len(sds):+.2f} SD")
    print(f"  raw shortfall {min(r['resid'] for r in rows if r['kind']=='adverse'):+.1f} to "
          f"{max(r['resid'] for r in rows if r['kind']=='adverse'):+.1f} mIoU")
    print(f"\n  => {'REPLICATES: three methods, two mechanisms' if below == len(sds) else 'DOES NOT fully replicate -- report honestly'}")
    json.dump({"fit": {"a": a, "b": b, "std": sd, "r2": r2, "n": len(X)}, "points": rows},
              open(os.path.join(RES, "dua_spine.json"), "w"), indent=1)
    print("\nwrote dua_spine.json")


if __name__ == "__main__":
    main()
