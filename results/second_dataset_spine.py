#!/usr/bin/env python3
"""
second_dataset_spine.py -- places the second real-dataset points (BDD100K,
Foggy Zurich) against the synthetic difficulty line, exactly as spine_scrutiny.py
does for ACDC.

Fit rule = primary/pinned (a): all 95 synthetic points (19 CS-C corruption types
x 5 severities), nothing excluded, residual std with ddof=1.
Run from ~/robustseg_wacv/results/.
"""
import json, glob, os

RES = os.path.dirname(os.path.abspath(__file__))


def _miou(d):
    k = [x for x in d if "miou" in x.lower()][0]
    v = d[k]
    return v * 100 if v < 1.5 else v


def scalar(tag):
    d = json.load(open(os.path.join(RES, tag + ".json")))
    return _miou(d["metrics"])


def sevs(tag):
    d = json.load(open(os.path.join(RES, tag + ".json")))
    return {s["severity"]: _miou(s) for s in d["severity_results"]}


def ols(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    std = (sum(r * r for r in res) / (n - 1)) ** 0.5
    r2 = 1 - sum(r * r for r in res) / sum((y - my) ** 2 for y in ys)
    return a, b, std, r2


def synthetic_fit(method):
    X, Y = [], []
    for fv in sorted(glob.glob(os.path.join(RES, "pertype_vanilla_*.json"))):
        ctype = os.path.basename(fv)[len("pertype_vanilla_"):-5]
        if not os.path.exists(os.path.join(RES, f"pertype_{method}_{ctype}.json")):
            continue
        van, ada = sevs(f"pertype_vanilla_{ctype}"), sevs(f"pertype_{method}_{ctype}")
        for s in sorted(van):
            if s in ada:
                X.append(van[s]); Y.append(ada[s] - van[s])
    a, b, std, r2 = ols(X, Y)
    return {"a": a, "b": b, "std": std, "r2": r2, "n": len(X), "X": X}


def place(fit, van, ada):
    pred = fit["a"] + fit["b"] * van
    res = (ada - van) - pred
    return {"vanilla": van, "adapted": ada, "benefit": ada - van, "pred": pred,
            "resid": res, "sd": res / fit["std"],
            "neighbors_pm3": sum(1 for x in fit["X"] if abs(x - van) <= 3)}


# real condition points: label -> (vanilla_tag, method_tag_fmt, n_images)
ACDC = [("ACDC fog",   "acdc_vanilla_cond_fog",   {"pdaonly": "acdc_acdc_pdaonly_jsd_fog",   "tent": "acdc_tent_cond_fog"},   100),
        ("ACDC night", "acdc_vanilla_cond_night", {"pdaonly": "acdc_acdc_pdaonly_jsd_night", "tent": "acdc_tent_cond_night"}, 106),
        ("ACDC rain",  "acdc_vanilla_cond_rain",  {"pdaonly": "acdc_acdc_pdaonly_jsd_rain",  "tent": "acdc_tent_cond_rain"},  100),
        ("ACDC snow",  "acdc_vanilla_cond_snow",  {"pdaonly": "acdc_acdc_pdaonly_jsd_snow",  "tent": "acdc_tent_cond_snow"},  100)]

NEW = [("BDD rainy",       "bdd_rainy_vanilla", {"pdaonly": "bdd_rainy_pdaonly", "tent": "bdd_rainy_tent"},  41),
       ("BDD snowy",       "bdd_snowy_vanilla", {"pdaonly": "bdd_snowy_pdaonly", "tent": "bdd_snowy_tent"}, 110),
       ("BDD clear",       "bdd_clear_vanilla", {"pdaonly": "bdd_clear_pdaonly", "tent": "bdd_clear_tent"},  82),
       ("Foggy Zurich fog", "fz_fog_vanilla",   {"pdaonly": "fz_fog_pdaonly",    "tent": "fz_fog_tent"},     40)]

P70 = [("BDD rainy p70", "bdd_rainy_p70_vanilla", {"pdaonly": "bdd_rainy_p70_pdaonly", "tent": "bdd_rainy_p70_tent"}, 34),
       ("BDD snowy p70", "bdd_snowy_p70_vanilla", {"pdaonly": "bdd_snowy_p70_pdaonly", "tent": "bdd_snowy_p70_tent"}, 93),
       ("BDD clear p70", "bdd_clear_p70_vanilla", {"pdaonly": "bdd_clear_p70_pdaonly", "tent": "bdd_clear_p70_tent"}, 65)]


def report(group, name, fits, out):
    for method in ("pdaonly", "tent"):
        fit = fits[method]; sds = []
        print(f"\n[{name} | {method}]  fit: benefit = {fit['b']:+.3f}*vanilla + {fit['a']:.2f}, "
              f"resid std {fit['std']:.2f}, n={fit['n']}")
        for label, vtag, mtags, n in group:
            p = place(fit, scalar(vtag), scalar(mtags[method]))
            sds.append(p["sd"])
            print(f"  {label:18s} n={n:4d}  van {p['vanilla']:5.2f}  ada {p['adapted']:5.2f}  "
                  f"benefit {p['benefit']:+5.2f}  pred {p['pred']:+5.2f}  "
                  f"resid {p['resid']:+5.2f} = {p['sd']:+5.2f} SD  [{p['neighbors_pm3']} nbrs]")
            out.append({"group": name, "method": method, "condition": label, "n_images": n, **{k: v for k, v in p.items() if k != "X"}})
        print(f"  --> mean {sum(sds)/len(sds):+.2f} SD, {sum(1 for s in sds if s < 0)}/{len(sds)} below the line")


if __name__ == "__main__":
    fits = {m: synthetic_fit(m) for m in ("pdaonly", "tent")}
    rows = []
    report(ACDC, "ACDC (published)", fits, rows)
    report(NEW,  "NEW real datasets", fits, rows)
    report(P70,  "BDD purity>=0.7 robustness", fits, rows)

    bs2, bs1 = scalar("bdd_clear_vanilla"), scalar("bdd_clear_vanilla_bs1")
    print(f"\n[parity] vanilla BDD clear: bs=2 {bs2:.4f} vs bs=1 {bs1:.4f}  (delta {bs2-bs1:+.4f} mIoU)")

    payload = {"fits": {m: {k: v for k, v in f.items() if k != "X"} for m, f in fits.items()},
               "points": rows,
               "vanilla_batch_parity": {"bs2": bs2, "bs1": bs1, "delta": bs2 - bs1}}
    with open(os.path.join(RES, "second_dataset_spine.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print("\nwrote second_dataset_spine.json")
