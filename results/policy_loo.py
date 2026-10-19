#!/usr/bin/env python3
"""
policy_loo.py -- does real-condition calibration buy a working adapt-or-not rule?

Pre-registered read (written before running):
  LOO >= always-adapt  -> few-shot real-world calibration restores a useful gate;
                          synthetic-only calibration cannot.
  LOO ~= always-adapt  -> even condition-level real calibration is insufficient at
                          this granularity; gating needs per-image signals.
Sanity expectation: the corrected rule should abstain on ACDC fog and adapt on
ACDC night and Foggy Zurich; BDD snowy is the costly abstention to watch.

Three rules, all deciding WITHOUT the target condition's labels:
  (a) synthetic-only gate : adapt iff confidence < tau, tau fitted on the 95
                            synthetic conditions  (reported previously: degenerate)
  (b) LOO corrected gate  : adapt iff  pred(van_hat) - shortfall_LOO > 0, where
                            van_hat is predicted from confidence via a synthetic
                            fit (label-free) and shortfall_LOO is the mean spine
                            residual of the OTHER real conditions
  (c) oracle gate         : per-condition best of {adapt, don't} -- needs labels,
                            upper reference only
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


def L(p): return json.load(open(os.path.join(RES, p)))
def sevs(p, key="miou"): return {r["severity"]: r[key] for r in L(p)["severity_results"]}
def scalar(p, key="miou"): return L(p)["metrics"][key]


def ols(X, Y):
    n = len(X); mx = sum(X) / n; my = sum(Y) / n
    b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    a = my - b * mx
    r = [y - (a + b * x) for x, y in zip(X, Y)]
    return a, b, (sum(v * v for v in r) / (n - 1)) ** 0.5


def synthetic(method):
    """(vanilla, benefit, confidence) over the 95 synthetic conditions."""
    out = []
    for t in TYPES:
        v = sevs(f"pertype_vanilla_{t}.json")
        a = sevs(f"pertype_{method}_{t}.json")
        c = sevs(f"pertype_vanconf_{t}.json", "mean_confidence")
        for s in sorted(v):
            if s in a and c.get(s) is not None:
                out.append((v[s] * 100, (a[s] - v[s]) * 100, c[s]))
    return out


def main():
    for method in ("pdaonly", "tent", "dua"):
        grid = synthetic(method)
        # benefit-vs-difficulty fit and the label-free confidence->difficulty fit,
        # both estimated on synthetic data only
        fit_ben = ols([g[0] for g in grid], [g[1] for g in grid])
        fit_van = ols([g[2] for g in grid], [g[0] for g in grid])
        print(f"\n{'='*98}\n{method}")
        print(f"  benefit fit   : {fit_ben[1]:+.3f}*van + {fit_ben[0]:.2f}")
        print(f"  conf->vanilla : {fit_van[1]:+.1f}*conf + {fit_van[0]:.1f}  (label-free difficulty proxy)")

        pts = []
        for label, tag in ADVERSE:
            van = scalar(f"cm_{tag}_vanilla.json") * 100
            ada = scalar(f"cm_{tag}_{method}.json") * 100
            conf = scalar(f"cm_{tag}_vanilla.json", "mean_confidence")
            van_hat = fit_van[0] + fit_van[1] * conf
            pred = fit_ben[0] + fit_ben[1] * van
            pts.append({"label": label, "vanilla": van, "adapted": ada, "conf": conf,
                        "van_hat": van_hat, "resid": (ada - van) - pred})

        print(f"\n  {'condition':18s} {'conf':>6s} {'van':>6s} {'van_hat':>8s} "
              f"{'benefit':>8s} {'corr.pred':>10s}  {'decision':>8s}")
        for i, p in enumerate(pts):
            loo_shortfall = sum(q["resid"] for j, q in enumerate(pts) if j != i) / (len(pts) - 1)
            corrected = (fit_ben[0] + fit_ben[1] * p["van_hat"]) + loo_shortfall
            p["corrected_pred"] = corrected
            p["adapt"] = corrected > 0
            p["gated"] = p["adapted"] if p["adapt"] else p["vanilla"]
            print(f"  {p['label']:18s} {p['conf']:6.4f} {p['vanilla']:6.2f} {p['van_hat']:8.2f} "
                  f"{p['adapted']-p['vanilla']:+8.2f} {corrected:+10.2f}  "
                  f"{'ADAPT' if p['adapt'] else 'skip':>8s}")

        n = len(pts)
        never = sum(p["vanilla"] for p in pts) / n
        always = sum(p["adapted"] for p in pts) / n
        loo = sum(p["gated"] for p in pts) / n
        orac = sum(max(p["vanilla"], p["adapted"]) for p in pts) / n
        print(f"\n  --- mean mIoU over the 7 real adverse conditions ---")
        print(f"      never-adapt              {never:6.2f}")
        print(f"      always-adapt             {always:6.2f}   ({always-never:+.2f})")
        print(f"      synthetic-calibrated gate{always:6.2f}   ({always-never:+.2f})  [degenerate: adapts everywhere]")
        print(f"      LOO-corrected gate       {loo:6.2f}   ({loo-never:+.2f} vs never, {loo-always:+.2f} vs always)")
        print(f"      [oracle gate]            {orac:6.2f}   ({orac-always:+.2f} vs always; needs labels)")
        verdict = "LOO >= always -> real calibration restores a useful gate" if loo >= always - 1e-9 \
                  else "LOO < always -> condition-level real calibration is NOT sufficient"
        print(f"      => {verdict}")
        json.dump({"method": method, "never": never, "always": always, "loo": loo,
                   "oracle": orac, "points": pts},
                  open(os.path.join(RES, f"policy_loo_{method}.json"), "w"), indent=1)
    print("\nwrote policy_loo_{pdaonly,tent}.json")


if __name__ == "__main__":
    main()
