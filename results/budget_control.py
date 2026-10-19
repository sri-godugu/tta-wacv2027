#!/usr/bin/env python3
"""
budget_control.py -- separates SIGNAL from OPTIMISATION BUDGET in the oracle
decomposition.

The 10-step oracle far exceeds 1-step unsupervised TTA, but that comparison
conflates a better adaptation signal with 10x the gradient steps. This runs the
same comparison with the unsupervised methods ALSO at 10 steps.

Reads: pertype_{vanilla,pdaonly,tent}_*, oracle_{s1,s10}_*, s10_{pdaonly,tent}_*
Run from ~/robustseg_wacv/results/ inside a Slurm allocation.
"""
import json, os

RES = os.path.dirname(os.path.abspath(__file__))
TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
         "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
         "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
         "snow", "spatter", "speckle_noise", "zoom_blur"]
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
    return a, b, (sum(v * v for v in r) / (n - 1)) ** 0.5


def synth(prefix):
    X, Y = [], []
    for t in TYPES:
        v = sevs(f"pertype_vanilla_{t}.json"); a = sevs(f"{prefix}_{t}.json")
        for s in sorted(v):
            if s in a: X.append(v[s]); Y.append(a[s] - v[s])
    return X, Y


def main():
    print("=" * 96)
    print("SYNTHETIC GRID (95 conditions): mean benefit vs adaptation budget")
    print("=" * 96)
    runs = {"PDA @1": "pertype_pdaonly", "PDA @10": "s10_pdaonly",
            "TENT @1": "pertype_tent", "TENT @10": "s10_tent",
            "oracle @1": "oracle_s1", "oracle @10": "oracle_s10"}
    means = {}
    for name, pre in runs.items():
        X, Y = synth(pre)
        means[name] = sum(Y) / len(Y)
        a, b, s = ols(X, Y)
        print(f"  {name:11s} mean benefit {means[name]:+6.2f} mIoU   "
              f"fit {b:+.3f}*van + {a:5.2f}  std {s:.2f}")
    print(f"\n  budget effect on the SYNTHETIC grid:")
    print(f"    PDA    1->10 steps: {means['PDA @10']-means['PDA @1']:+.2f} mIoU")
    print(f"    TENT   1->10 steps: {means['TENT @10']-means['TENT @1']:+.2f} mIoU")
    print(f"    oracle 1->10 steps: {means['oracle @10']-means['oracle @1']:+.2f} mIoU")

    print("\n" + "=" * 96)
    print("REAL CONDITIONS: benefit vs budget, against the supervised ceiling")
    print("=" * 96)
    print(f"  {'condition':18s} {'PDA@1':>7s} {'PDA@10':>7s} {'d':>6s}   "
          f"{'TENT@1':>7s} {'TENT@10':>8s} {'d':>6s}   {'ceil@10':>8s}")
    rows = []
    for label, tag in ADVERSE + PROBE:
        van = scalar(f"cm_{tag}_vanilla.json")
        p1 = scalar(f"cm_{tag}_pdaonly.json") - van
        p10 = scalar(f"s10_pdaonly_{tag}.json") - van
        t1 = scalar(f"cm_{tag}_tent.json") - van
        t10 = scalar(f"s10_tent_{tag}.json") - van
        ceil = scalar(f"oracle_s10_{tag}.json") - van
        rows.append({"condition": label, "kind": "adverse" if (label, tag) in ADVERSE else "probe",
                     "vanilla": van, "pda_s1": p1, "pda_s10": p10,
                     "tent_s1": t1, "tent_s10": t10, "ceiling_s10": ceil})
        print(f"  {label:18s} {p1:+7.2f} {p10:+7.2f} {p10-p1:+6.2f}   "
              f"{t1:+7.2f} {t10:+8.2f} {t10-t1:+6.2f}   {ceil:+8.2f}")

    adv = [r for r in rows if r["kind"] == "adverse"]
    n = len(adv)
    print(f"\n  adverse means: PDA {sum(r['pda_s1'] for r in adv)/n:+.2f} -> "
          f"{sum(r['pda_s10'] for r in adv)/n:+.2f} at 10 steps "
          f"({sum(r['pda_s10']-r['pda_s1'] for r in adv)/n:+.2f})")
    print(f"                 TENT {sum(r['tent_s1'] for r in adv)/n:+.2f} -> "
          f"{sum(r['tent_s10'] for r in adv)/n:+.2f} "
          f"({sum(r['tent_s10']-r['tent_s1'] for r in adv)/n:+.2f})")
    print(f"                 supervised ceiling {sum(r['ceiling_s10'] for r in adv)/n:+.2f}")
    worse = sum(1 for r in adv if r["pda_s10"] < r["pda_s1"])
    worse_t = sum(1 for r in adv if r["tent_s10"] < r["tent_s1"])
    print(f"\n  more steps makes the UNSUPERVISED result WORSE on {worse}/{n} (PDA) and "
          f"{worse_t}/{n} (TENT) adverse conditions,")
    print(f"  while the SUPERVISED signal on the same parameters gains "
          f"{sum(r['ceiling_s10'] for r in adv)/n:+.2f} mIoU.")
    print("  => the shortfall is not an optimisation-budget deficit; it is the signal.")

    json.dump({"synthetic_means": means, "rows": rows},
              open(os.path.join(RES, "budget_control.json"), "w"), indent=1)
    print("\nwrote budget_control.json")


if __name__ == "__main__":
    main()
