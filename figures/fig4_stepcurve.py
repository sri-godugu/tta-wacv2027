#!/usr/bin/env python3
"""
Fig. 4 — the budget dissociation as a curve.

Consumes the overnight campaign's outputs:
  synthetic per-type : data/v2s{K}_{method}_{type}.json      (severity_results)
  real conditions    : data/v2s{K}_{method}_{cond}.json       (metrics)
  methods in {pdaonly, tent, oracle, vanilla}; K in {1,2,3,5,10}
  optional per-run   : "loss_trace": [per-step mean objective]
  optional LR axis   : data/v2s1_lr{1e-4|3e-4}_{method}_{cond}.json

Two modes, decided per panel by what exists on disk:
  FULL  (v2 files present): left = mean synthetic benefit vs steps (95-grid,
        per-step vanilla-paired on the same pool); right = mean REAL SHORTFALL
        vs steps, where the synthetic difficulty trend is REFIT AT EVERY STEP
        so the comparison stays difficulty-controlled at every budget.
  PREVIEW (no v2 files): endpoints from data/v1_endpoints.json, drawn as
        dashed two-point segments and watermarked — composition check only.

Nothing is ever interpolated as if measured: lines connect measured steps
only, and the mode is printed on the figure.
"""
import glob, json, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pubstyle import (use_pubstyle, despine, chip, ACCENT, BLUE, GREEN, INK,
                      MUTED, CLOUD, FITLN, SURF)

use_pubstyle(7.2)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
STEPS = [1, 2, 3, 5, 10]
REAL = ["acdc_fog", "acdc_night", "acdc_rain", "acdc_snow",
        "bdd_rainy", "bdd_snowy", "fz_fog"]
METH = {"pdaonly": dict(c=ACCENT, m="o", lab="PDA (disagreement)"),
        "tent":    dict(c=GREEN,  m="^", lab="TENT (entropy)"),
        "oracle":  dict(c=BLUE,   m="s", lab="supervised ref.")}

def load(p):
    return json.load(open(p))

# Vanilla is paired PER GPU POOL (PDA and the supervised reference ran on H200, TENT on V100), so every
# benefit is adapted minus vanilla on the same hardware and identical inputs. build_v2_data.py writes these.
POOL = {"pdaonly": "h200", "tent": "v100", "oracle": "h200"}

def _van(method, stem):
    return os.path.join(D, f"v2_vanilla_{POOL[method]}_{stem}.json")

def syn_points(method, k):
    """(vanilla[], benefit[]) over the 95 v2 conditions, same-pool vanilla-paired; None unless all 19 types exist."""
    V, B = [], []
    pref = os.path.join(D, f"v2_vanilla_{POOL[method]}_")
    types = [os.path.basename(p)[len(os.path.basename(pref)):-5] for p in sorted(glob.glob(pref + "*.json"))]
    types = [t for t in types if t not in REAL and t != "bdd_clear"]
    if len(types) != 19:
        return None
    for t in types:
        pm = os.path.join(D, f"v2s{k}_{method}_{t}.json")
        if not os.path.exists(pm):
            return None
        sv = {r["severity"]: r["miou"] for r in load(_van(method, t))["severity_results"]}
        sm = {r["severity"]: r["miou"] for r in load(pm)["severity_results"]}
        for sev in sorted(set(sv) & set(sm)):
            V.append(sv[sev] * 100); B.append((sm[sev] - sv[sev]) * 100)
    return (np.asarray(V), np.asarray(B)) if len(V) else None

def real_points(method, k):
    out = []
    for c in REAL:
        pv, pm = _van(method, c), os.path.join(D, f"v2s{k}_{method}_{c}.json")
        if not (os.path.exists(pv) and os.path.exists(pm)):
            return None
        v = load(pv)["metrics"]["miou"] * 100
        a = load(pm)["metrics"]["miou"] * 100
        out.append((v, a - v))
    return out

def collect():
    """Per method: steps -> (syn_mean, real_mean, real_shortfall_mean)."""
    res = {m: {} for m in METH}
    any_v2 = False
    for m in METH:
        for k in STEPS:
            syn = syn_points(m, k)
            rl = real_points(m, k)
            entry = {}
            if syn is not None:
                any_v2 = True
                V, B = syn
                entry["syn_mean"] = float(B.mean())
                b, a = np.polyfit(V, B, 1)
                entry["fit"] = (a, b)
            if rl is not None:
                any_v2 = True
                entry["real_mean"] = float(np.mean([b for _, b in rl]))
                if "fit" in entry:
                    a, b = entry["fit"]
                    entry["shortfall"] = float(np.mean(
                        [ben - (a + b * v) for v, ben in rl]))
            if entry:
                res[m][k] = entry
    return res, any_v2

def preview():
    ep = load(os.path.join(D, "v1_endpoints.json"))
    res = {m: {} for m in METH}
    for m in METH:
        for k in STEPS:
            e = {}
            sk = ep["synthetic_means"].get(m, {}).get(str(k))
            rk = ep["real_means"].get(m, {}).get(str(k))
            if sk is not None: e["syn_mean"] = sk
            if rk is not None: e["real_mean"] = rk
            if e: res[m][k] = e
    return res

res, full = collect()
if not full:
    res = preview()

# ---- panel (c): descent trace on fog (objective vs accuracy) -------------
def _trace(path):
    """Return (objective_after_K_steps, miou*100) for a run file, or None.
    loss_trace may sit at top level (real files) or inside the severity-3
    entry (synthetic per-type files)."""
    if not os.path.exists(path):
        return None
    d = load(path)
    if "metrics" in d:
        tr = d.get("loss_trace") or d["metrics"].get("loss_trace")
        mi = d["metrics"]["miou"] * 100
    else:
        e = next((r for r in d["severity_results"] if r["severity"] == 3), None)
        if e is None:
            return None
        tr = e.get("loss_trace") or d.get("loss_trace")
        mi = e["miou"] * 100
    if not tr:
        return None
    return float(tr[-1]), float(mi), float(tr[0])

def descent_series():
    """{(method, domain): [(steps, obj_after, miou)...], 'van': {...}}"""
    out = {}
    van = {}
    for dom, stem in [("real", "acdc_fog"), ("syn", "fog")]:
        for m in ("pdaonly", "tent"):
            pv = _van(m, stem)                      # same-pool vanilla for this method
            if os.path.exists(pv):
                dv = load(pv)
                van[(m, dom)] = (dv["metrics"]["miou"] * 100 if "metrics" in dv else
                                 next(r for r in dv["severity_results"]
                                      if r["severity"] == 3)["miou"] * 100)
            pts = []
            obj0 = None
            for k in STEPS:
                t = _trace(os.path.join(D, f"v2s{k}_{m}_{stem}.json"))
                if t:
                    pts.append((k, t[0], t[1]))
                    if k == 1:
                        obj0 = t[2]
            if len(pts) >= 2:
                out[(m, dom)] = (pts, obj0)
    return out, van

DES, VAN = descent_series()
three = len(DES) >= 2

if three:
    fig, (axL, axR, axD) = plt.subplots(
        1, 3, figsize=(6.875, 2.15),
        gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.82]))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.78, bottom=0.185,
                        wspace=0.36)
else:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.875, 2.15), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.80, bottom=0.185,
                        wspace=0.16)

_cip = os.path.join(D, "stepcurve_ci.json")
CI = {}
if os.path.exists(_cip):
    _c = load(_cip)
    CI = {"pdaonly": _c.get("pda", {}), "tent": _c.get("tent", {})}

def draw(ax, key, ylab, title):
    for m, st in METH.items():
        ks = [k for k in STEPS if key in res[m].get(k, {})]
        if not ks:
            continue
        ys = [res[m][k][key] for k in ks]
        ls = "-" if (full and len(ks) >= 3) else (0, (3, 2))
        lw, z = 1.4, 5
        if key == "shortfall" and m == "oracle":   # on this axis it nearly coincides with TENT: dash it so both show
            ls, lw, z = (0, (3.2, 2.2)), 1.1, 6
        ax.plot(ks, ys, color=st["c"], marker=st["m"], ms=4.2, lw=lw, ls=ls,
                mec=SURF, mew=0.5, zorder=z, label=st["lab"])
        if key == "shortfall" and CI.get(m):
            ci = [CI[m].get(str(k)) for k in ks]
            if all(c is not None for c in ci):           # per-budget 95% CI (joint types x images bootstrap)
                ax.fill_between(ks, [c["ci"][0] for c in ci], [c["ci"][1] for c in ci],
                                color=st["c"], alpha=0.13, lw=0, zorder=2)
        # nudge this end label off any label already placed nearby (e.g. TENT -8.2 vs supervised -8.3)
        _ylab = ys[-1]
        _span = np.diff(ax.get_ylim())[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1.0
        _placed = getattr(ax, "_end_labels", [])
        for _yp in _placed:
            if abs(_ylab - _yp) < 0.06 * abs(_span):
                _ylab = _yp + (0.065 * abs(_span) if ys[-1] >= _yp else -0.065 * abs(_span))
        ax._end_labels = _placed + [_ylab]
        ax.annotate(f"{ys[-1]:+.1f}".replace("-", "\u2212"), (ks[-1], _ylab), xytext=(4, 0),
                    textcoords="offset points", fontsize=5.6, color=st["c"],
                    va="center", zorder=6)
    ax.axhline(0, color="#B9C1CB", lw=0.9, zorder=1)
    ax.set_xscale("log"); ax.set_xticks(STEPS)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(0.85, 13.5)
    ax.set_ylabel(ylab, labelpad=1.5)
    ax.set_title(title, fontsize=7.0, loc="left", pad=3)
    despine(ax)

mode_syn = "mean benefit (mIoU)"
if full and any("shortfall" in res[m].get(k, {}) for m in METH for k in STEPS):
    draw(axL, "syn_mean", "mean benefit (mIoU)",
         "(a) 95 synthetic: holds or rises")
    # Panel (b) plots each curve in the quantity the argument uses: REAL mean benefit (one axis, one quantity),
    # so the dissociation reads directly: PDA and TENT fall, the supervised reference rises ("capacity intact").
    # The difficulty-controlled shortfall (with paired CIs) is carried by the caption and Sec. 6. On a shortfall
    # axis the supervised curve also falls (vs. its own steep synthetic trend), which is a disclosed caveat, not the
    # claim. FIG4_B=shortfall renders that version (fig_stepcurve_shortfall.*) for the supplement.
    if os.environ.get("FIG4_B") == "shortfall":
        draw(axR, "shortfall", "mean shortfall (mIoU)",
             "(b) 7 real adverse: shortfall deepens")
    else:
        draw(axR, "real_mean", "mean real benefit (mIoU)",
             "(b) 7 real adverse: only supervision gains")
    if CI and os.environ.get("FIG4_B") == "shortfall":
        chip(axR, 0.04, 0.05, ["band: per-budget 95% CI",
                               "growth vs. 1 step: every",
                               "paired 95% CI excludes 0"],
             loc="axes", ha="left", va="bottom", fontsize=5.2)
else:
    draw(axL, "syn_mean", "mean benefit (mIoU)",
         "(a) 95 synthetic conditions: more steps help or are harmless")
    draw(axR, "real_mean", "mean benefit (mIoU)",
         "(b) 7 real adverse: the same budget hurts")

# legend above the panels (inside panel (a) it sat on the PDA line)
_h, _l = axL.get_legend_handles_labels()
fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(_h), fontsize=6.0,
           handlelength=1.8, columnspacing=1.4, frameon=False)
# centred under panels (a)+(b) only; panel (c) carries its own x-label
_x0, _x1 = axL.get_position().x0, axR.get_position().x1
fig.supxlabel("adaptation budget (gradient steps, log scale)", fontsize=7.0,
              y=0.02, x=(_x0 + _x1) / 2, color=INK)

if three:
    # (c) objective descent vs accuracy, ACDC fog vs synthetic fog sev-3.
    # x = objective after K steps, normalized to its pre-update value, so
    # the two objectives (JSD, entropy) share one axis; descent runs left.
    for (m, dom), (pts, obj0) in DES.items():
        st = METH[m]
        xs = [p[1] / obj0 for p in pts] if obj0 else None
        if xs is None:
            continue
        ys = [p[2] for p in pts]
        if (m, dom) in VAN:
            xs = [1.0] + xs
            ys = [VAN[(m, dom)]] + ys
        alpha, lw = (1.0, 1.5) if dom == "real" else (0.45, 1.2)
        axD.plot(xs, ys, color=st["c"], lw=lw, alpha=alpha, zorder=5,
                 marker=st["m"], ms=3.2, mec=SURF, mew=0.4)
        axD.annotate("", xy=(xs[-1], ys[-1]),
                     xytext=(xs[-2], ys[-2]), zorder=6,
                     arrowprops=dict(arrowstyle="-|>", color=st["c"],
                                     lw=0.0, alpha=alpha, mutation_scale=7))
    axD.invert_xaxis()
    axD.set_xlabel("objective (fraction of initial)", labelpad=1.5)
    axD.set_ylabel("mIoU on fog", labelpad=1.5)
    axD.set_title("(c) descent vs. accuracy", fontsize=7.0, loc="left", pad=3)
    axD.text(0.97, 0.05, "faint: synthetic fog s3\nsolid: real ACDC fog",   # empty lower-right corner
             transform=axD.transAxes, fontsize=5.3, color=MUTED, va="bottom", ha="right",
             linespacing=1.35)
    despine(axD)

if not full:
    fig.text(0.5, 0.5, "PREVIEW: v1 endpoints only; replaced by v2s* runs",
             ha="center", va="center", fontsize=11, color="#C9CFD8",
             alpha=0.7, rotation=8, zorder=0)

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
_stem = "fig_stepcurve_shortfall" if os.environ.get("FIG4_B") == "shortfall" else "fig_stepcurve"
fig.savefig(os.path.join(HERE, "out", _stem + ".pdf"))
fig.savefig(os.path.join(HERE, "out", _stem + ".png"), dpi=300)
print("wrote out/fig_stepcurve.{pdf,png}  mode:",
      "FULL v2" if full else "PREVIEW (v1 endpoints)")
