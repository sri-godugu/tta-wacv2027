#!/usr/bin/env python3
"""
policy_eval.py -- selective (confidence-gated) adaptation, fit on synthetic only.

The measurement study says TTA benefit is largely predicted by headroom, and
that mean max-softmax confidence recovers most of the oracle difficulty signal
without labels. This turns that observation into a deployable rule:

    adapt this condition  iff  mean confidence < tau

tau is fitted on the 95 SYNTHETIC points only (19 CS-C types x 5 severities),
then FROZEN and applied to the 8 real condition-points. No real data informs the
threshold, so the real-side numbers are an honest held-out evaluation.

Rows reported over the real conditions: never-adapt / always-adapt / gated,
plus an oracle-gate upper reference (choose the better of the two per condition,
which requires labels and is not a method).

Run from ~/robustseg_wacv/results/ inside a Slurm allocation.
"""
import json, os

RES = os.path.dirname(os.path.abspath(__file__))

TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
         "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
         "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
         "snow", "spatter", "speckle_noise", "zoom_blur"]

REAL = [  # label, tag, kind
    ("ACDC fog", "acdc_fog", "adverse"), ("ACDC night", "acdc_night", "adverse"),
    ("ACDC rain", "acdc_rain", "adverse"), ("ACDC snow", "acdc_snow", "adverse"),
    ("BDD rainy", "bdd_rainy", "adverse"), ("BDD snowy", "bdd_snowy", "adverse"),
    ("Foggy Zurich fog", "fz_fog", "adverse"), ("BDD clear", "bdd_clear", "probe"),
]


def _load(p):
    return json.load(open(os.path.join(RES, p)))


def synthetic_grid(method):
    """[(vanilla_miou, adapted_miou, confidence)] over the 95 synthetic points."""
    out = []
    for t in TYPES:
        van = {r["severity"]: r["miou"] * 100 for r in _load(f"pertype_vanilla_{t}.json")["severity_results"]}
        ada = {r["severity"]: r["miou"] * 100 for r in _load(f"pertype_{method}_{t}.json")["severity_results"]}
        conf = {r["severity"]: r.get("mean_confidence")
                for r in _load(f"pertype_vanconf_{t}.json")["severity_results"]}
        for s in sorted(van):
            if s in ada and conf.get(s) is not None:
                out.append((van[s], ada[s], conf[s]))
    return out


def real_points(method):
    out = []
    for label, tag, kind in REAL:
        v = _load(f"cm_{tag}_vanilla.json")["metrics"]
        a = _load(f"cm_{tag}_{method}.json")["metrics"]
        out.append({"label": label, "kind": kind,
                    "vanilla": v["miou"] * 100, "adapted": a["miou"] * 100,
                    "conf": v["mean_confidence"]})
    return out


def fit_tau(grid):
    """tau maximising total realised benefit on the synthetic grid (adapt iff conf < tau)."""
    cands = sorted(set(c for _, _, c in grid))
    # candidate thresholds sit between observed confidences, plus the two extremes
    mids = [cands[0] - 1e-6] + [(x + y) / 2 for x, y in zip(cands, cands[1:])] + [cands[-1] + 1e-6]
    best, best_tau = None, None
    for tau in mids:
        gain = sum((a - v) for v, a, c in grid if c < tau)
        if best is None or gain > best:
            best, best_tau = gain, tau
    return best_tau, best


def main():
    for method in ("pdaonly", "tent"):
        grid = synthetic_grid(method)
        tau, gain = fit_tau(grid)
        n_adapt = sum(1 for _, _, c in grid if c < tau)
        print(f"\n{'='*92}\n{method}   tau fitted on SYNTHETIC ONLY = {tau:.4f}  "
              f"(adapts {n_adapt}/{len(grid)} synthetic points, total realised gain {gain:+.1f} mIoU)")

        pts = real_points(method)
        print(f"\n  {'condition':18s} {'conf':>7s}  {'gate':>6s}  {'vanilla':>8s} {'adapt':>8s} {'gated':>8s}  {'benefit':>8s}")
        for p in pts:
            p["adapt"] = p["conf"] < tau
            p["gated"] = p["adapted"] if p["adapt"] else p["vanilla"]
            print(f"  {p['label']:18s} {p['conf']:7.4f}  {'ADAPT' if p['adapt'] else 'skip':>6s}  "
                  f"{p['vanilla']:8.2f} {p['adapted']:8.2f} {p['gated']:8.2f}  {p['adapted']-p['vanilla']:+8.2f}")

        for scope, keep in (("7 adverse conditions", lambda p: p["kind"] == "adverse"),
                            ("all 8 (incl. clear probe)", lambda _: True)):
            sel = [p for p in pts if keep(p)]
            n = len(sel)
            never = sum(p["vanilla"] for p in sel) / n
            always = sum(p["adapted"] for p in sel) / n
            gated = sum(p["gated"] for p in sel) / n
            orac = sum(max(p["vanilla"], p["adapted"]) for p in sel) / n
            print(f"\n  --- {scope} (mean mIoU over conditions) ---")
            print(f"      never-adapt        {never:6.2f}")
            print(f"      always-adapt       {always:6.2f}   ({always-never:+.2f} vs never)")
            print(f"      confidence-gated   {gated:6.2f}   ({gated-never:+.2f} vs never, "
                  f"{gated-always:+.2f} vs always)")
            print(f"      [oracle gate]      {orac:6.2f}   (needs labels; upper reference)")

        json.dump({"tau": tau, "method": method, "points": pts},
                  open(os.path.join(RES, f"policy_eval_{method}.json"), "w"), indent=1)

    print("\nwrote policy_eval_{pdaonly,tent}.json")


if __name__ == "__main__":
    main()
