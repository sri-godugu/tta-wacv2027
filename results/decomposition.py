#!/usr/bin/env python3
"""
decomposition.py -- oracle-ceiling decomposition of the synthetic-to-real gap.
Read fixed in advance: see preregistration_oracle_ceiling.md.

    recoverability gap = predicted benefit - ceiling   (unrecoverable by this family)
    signal gap         = ceiling - actual benefit      (recoverable, signal missed it)
    total shortfall    = predicted - actual            (identity: sum of the two)

Two oracle budgets: s1 (1 step, budget-matched to the unsupervised runs, as
pre-registered) and s10 (10 steps, budget-unconstrained ceiling -- added
because the pre-registered sanity gate fired: a 1-step supervised update is
not a supremum).
Run from ~/robustseg_wacv/results/ inside a Slurm allocation.
"""
import json, glob, os

RES = os.path.dirname(os.path.abspath(__file__))
TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
         "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
         "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
         "snow", "spatter", "speckle_noise", "zoom_blur"]
REAL = [("ACDC fog", "acdc_fog", "adverse"), ("ACDC night", "acdc_night", "adverse"),
        ("ACDC rain", "acdc_rain", "adverse"), ("ACDC snow", "acdc_snow", "adverse"),
        ("BDD rainy", "bdd_rainy", "adverse"), ("BDD snowy", "bdd_snowy", "adverse"),
        ("Foggy Zurich fog", "fz_fog", "adverse"), ("BDD clear", "bdd_clear", "probe")]


def L(p):
    return json.load(open(os.path.join(RES, p)))


def sevs(p):
    return {r["severity"]: r["miou"] * 100 for r in L(p)["severity_results"]}


def scalar(p):
    return L(p)["metrics"]["miou"] * 100


def ols(X, Y):
    n = len(X); mx = sum(X) / n; my = sum(Y) / n
    b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    a = my - b * mx
    r = [y - (a + b * x) for x, y in zip(X, Y)]
    return a, b, (sum(v * v for v in r) / (n - 1)) ** 0.5


def synth(method_prefix):
    """(vanilla, benefit) over the 95 synthetic points for a given adapted run prefix."""
    X, Y = [], []
    for t in TYPES:
        v = sevs(f"pertype_vanilla_{t}.json")
        a = sevs(f"{method_prefix}_{t}.json")
        for s in sorted(v):
            if s in a:
                X.append(v[s]); Y.append(a[s] - v[s])
    return X, Y


def main():
    # benefit fit (the pinned spine fit) and ceiling fits
    fit_pda = ols(*synth("pertype_pdaonly"))
    print(f"benefit fit (PDA):  {fit_pda[1]:+.3f}*van + {fit_pda[0]:.2f}   std {fit_pda[2]:.2f}")

    for budget in ("s1", "s10"):
        Xc, Yc = synth(f"oracle_{budget}")
        ac, bc, sc_ = ols(Xc, Yc)
        print(f"\n{'='*100}\nORACLE CEILING, budget = {budget}"
              f"   synthetic ceiling fit: {bc:+.3f}*van + {ac:.2f}, std {sc_:.2f}")

        # sanity gate: ceiling below achieved unsupervised benefit?
        Xb, Yb = synth("pertype_pdaonly")
        viol = sum(1 for yc, yb in zip(Yc, Yb) if yc < yb)
        print(f"  gate: ceiling < achieved PDA benefit on {viol}/{len(Yc)} synthetic points "
              f"({100*viol/len(Yc):.0f}%)")

        print(f"\n  {'condition':18s} {'van':>6s} {'actual':>7s} {'ceiling':>8s} {'pred':>7s} "
              f"{'recov.gap':>10s} {'signal.gap':>11s} {'ceil resid SD':>14s}")
        rows = []
        for label, tag, kind in REAL:
            van = scalar(f"cm_{tag}_vanilla.json")
            act = scalar(f"cm_{tag}_pdaonly.json") - van
            ceil = scalar(f"oracle_{budget}_{tag}.json") - van
            pred = fit_pda[0] + fit_pda[1] * van
            recov, signal = pred - ceil, ceil - act
            cres = (ceil - (ac + bc * van)) / sc_
            rows.append({"condition": label, "kind": kind, "budget": budget, "vanilla": van,
                         "actual": act, "ceiling": ceil, "pred": pred,
                         "recoverability_gap": recov, "signal_gap": signal,
                         "ceiling_resid_sd": cres})
            print(f"  {label:18s} {van:6.2f} {act:+7.2f} {ceil:+8.2f} {pred:+7.2f} "
                  f"{recov:+10.2f} {signal:+11.2f} {cres:+14.2f}")

        adv = [r for r in rows if r["kind"] == "adverse"]
        n = len(adv)
        mr = sum(r["recoverability_gap"] for r in adv) / n
        ms = sum(r["signal_gap"] for r in adv) / n
        shares = [r["recoverability_gap"] / (r["recoverability_gap"] + r["signal_gap"])
                  for r in adv if abs(r["recoverability_gap"] + r["signal_gap"]) > 1e-9]
        shares.sort()
        med = shares[len(shares) // 2] if len(shares) % 2 else (shares[len(shares)//2 - 1] + shares[len(shares)//2]) / 2
        print(f"\n  adverse means: recoverability gap {mr:+.2f}, signal gap {ms:+.2f} mIoU"
              f"   (total {mr+ms:+.2f})")
        print(f"  median share of the shortfall carried by RECOVERABILITY: {100*med:.0f}%"
              f"   -> H1 {'SUPPORTED' if med > 0.5 else 'NOT supported'} at this budget")
        print(f"  ceiling residuals below the synthetic ceiling line: "
              f"{sum(1 for r in adv if r['ceiling_resid_sd'] < 0)}/{n}, "
              f"mean {sum(r['ceiling_resid_sd'] for r in adv)/n:+.2f} SD")

        json.dump({"budget": budget, "ceiling_fit": {"a": ac, "b": bc, "std": sc_},
                   "benefit_fit": {"a": fit_pda[0], "b": fit_pda[1], "std": fit_pda[2]},
                   "rows": rows},
                  open(os.path.join(RES, f"decomposition_{budget}.json"), "w"), indent=1)
    print("\nwrote decomposition_{s1,s10}.json")


if __name__ == "__main__":
    main()
