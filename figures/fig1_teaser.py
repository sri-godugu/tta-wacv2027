#!/usr/bin/env python3
"""
Fig. 1 (teaser, full width): the audit protocol as a pipeline (left),
the real-vs-synthetic spine on the primary backbone (right),
and a verdict strip (bottom). All plotted values are the released run
outputs; nothing is synthesized.
"""
import json, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
from pubstyle import (use_pubstyle, despine, chip, stage_card, arrow, ols,
                      ACCENT, BLUE, GREEN, INK, MUTED, CLOUD, FITLN, BAND,
                      GRID, SURF, P_GREEN, P_LAV, P_PEACH, P_BLUE)

use_pubstyle(7.2)
D = "./data"
grid = json.load(open(f"{D}/grid_primary_pda.json"))
meta = json.load(open(f"{D}/spine_meta.json"))
fit  = meta["fits"]["primary"]
real = meta["real"]["primary"]

types = sorted(grid["vanilla"])
V = np.concatenate([np.asarray(grid["vanilla"][t]) * 100 for t in types])
B = np.concatenate([(np.asarray(grid["pda"][t]) - np.asarray(grid["vanilla"][t])) * 100
                    for t in types])
labels = [(t, s + 1) for t in types for s in range(5)]
a, b, r2, sd = fit["a"], fit["b"], fit["r2"], fit["std"]

fig = plt.figure(figsize=(6.875, 2.85))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.52], height_ratios=[1.0, 0.145],
                      left=0.005, right=0.995, top=0.985, bottom=0.01,
                      wspace=0.115, hspace=0.38)

# ================================================================ schematic
axS = fig.add_subplot(gs[0, 0]); axS.set_xlim(0, 1); axS.set_ylim(0, 1)
axS.axis("off")

# dashed grouping frame
axS.add_patch(mpl.patches.FancyBboxPatch((0.01, 0.015), 0.98, 0.955,
    boxstyle="round,pad=0.008,rounding_size=0.022", facecolor="#FBFCFD",
    edgecolor="#AEB6C2", linewidth=0.9, linestyle=(0, (4, 2.4)), zorder=1))
axS.text(0.045, 0.935, "THE AUDIT", fontsize=7.0, weight="bold", color=INK, zorder=4)
axS.text(0.955, 0.935, "a reusable evaluation protocol", fontsize=5.8,
         color=MUTED, style="italic", ha="right", zorder=4)

cw, ch, cx = 0.90, 0.245, 0.045
ys = [0.635, 0.340, 0.045]
stage_card(axS, cx, ys[0], cw, ch, *P_GREEN, number="1",
           title="Fit the difficulty–benefit law",
           body="95 synthetic conditions (Cityscapes-C)\n"
                "benefit = −0.19·mIoU + 13.3 · R² = 0.44")
stage_card(axS, cx, ys[1], cw, ch, *P_LAV, number="2",
           title="Place natural shifts against it",
           body="7 real adverse conditions · 3 datasets\n"
                "matched difficulty, interpolation only")
stage_card(axS, cx, ys[2], cw, ch, *P_PEACH, number="3",
           title="Separate error from signal",
           body="supervised ref.: same params, budget\n"
                "recoverability gap  vs.  signal gap")
for y0, y1 in [(ys[0], ys[1] + ch), (ys[1], ys[2] + ch)]:
    arrow(axS, 0.5, y0 - 0.004, 0.5, y1 + 0.004, color="#8D95A3", lw=1.2)

# micro-glyphs (drawn from the real numbers, miniature scale)
def to_card(x, y, bx, by, bw, bh, xr, yr):
    xs = bx + (np.asarray(x) - xr[0]) / (xr[1] - xr[0]) * bw
    ys_ = by + (np.asarray(y) - yr[0]) / (yr[1] - yr[0]) * bh
    return xs, ys_
xr, yr = (5, 73), (-7, 33)
g1 = dict(bx=cx + 0.620, by=ys[0] + 0.030, bw=0.255, bh=ch - 0.125)
sx, sy = to_card(V[::3], B[::3], **g1, xr=xr, yr=yr)
axS.scatter(sx, sy, s=1.6, color=CLOUD, lw=0, zorder=5)
lx = np.array([10, 70]); ly = a + b * lx
lx_, ly_ = to_card(lx, ly, **g1, xr=xr, yr=yr)
axS.plot(lx_, ly_, color=FITLN, lw=1.0, zorder=6)

g2 = dict(bx=cx + 0.620, by=ys[1] + 0.030, bw=0.255, bh=ch - 0.125)
axS.fill_between(*to_card(lx, ly + sd, **g2, xr=xr, yr=yr),
                 to_card(lx, ly - sd, **g2, xr=xr, yr=yr)[1],
                 color=BAND, alpha=0.12, lw=0, zorder=4)
axS.plot(*to_card(lx, ly, **g2, xr=xr, yr=yr), color=FITLN, lw=0.9, zorder=5)
rv = [r["vanilla"] for r in real if r["kind"] == "adverse"]
rb = [r["benefit"] for r in real if r["kind"] == "adverse"]
axS.scatter(*to_card(rv, rb, **g2, xr=xr, yr=yr), marker="*", s=20,
            color=ACCENT, edgecolor=SURF, linewidth=0.3, zorder=6)

g3 = dict(bx=cx + 0.620, by=ys[2] + 0.038, bw=0.255, bh=ch - 0.135)
for i, (act, ref) in enumerate([(-2.49, 3.41), (1.63, 9.04)]):
    yy = g3["by"] + (0.80 - 0.56 * i) * g3["bh"]
    x0, _ = to_card([act], [0], **g3, xr=(-4, 10), yr=(0, 1))
    x1, _ = to_card([ref], [0], **g3, xr=(-4, 10), yr=(0, 1))
    axS.plot([x0[0], x1[0]], [yy, yy], color=ACCENT, lw=2.6, alpha=0.30,
             solid_capstyle="butt", zorder=5)
    axS.scatter([x0[0]], [yy], s=8, color=ACCENT, zorder=6)
    axS.scatter([x1[0]], [yy], s=8, marker="s", color=BLUE, zorder=6)

# ================================================================ spine
ax = fig.add_subplot(gs[0, 1])
xx = np.linspace(6, 72.5, 200)
ax.fill_between(xx, a + b * xx - sd, a + b * xx + sd, color=BAND, alpha=0.10,
                lw=0, zorder=1)
ax.plot(xx, a + b * xx, color=FITLN, lw=1.3, zorder=3)
ax.scatter(V, B, s=9, color=CLOUD, lw=0, zorder=2)

# flag the BN-recalibration regime (saturate s4/s5)
sat = [(v, bb) for (v, bb), lab in zip(zip(V, B), labels)
       if lab == ("saturate", 4) or lab == ("saturate", 5)]
for v, bb in sat:
    ax.scatter([v], [bb], s=34, facecolor="none", edgecolor=MUTED,
               linewidth=0.8, zorder=3)
ax.annotate("extreme saturation:\nBN-statistics regime (Sec. 8)",
            xy=(sat[0][0] + 1.2, sat[0][1]), xytext=(48.8, 21.0),
            fontsize=5.8, color=MUTED, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                            shrinkA=0, shrinkB=2,
                            connectionstyle="arc3,rad=-0.15"), zorder=4)

# real conditions
adv = [r for r in real if r["kind"] == "adverse"]
pro = [r for r in real if r["kind"] == "probe"][0]
ax.scatter([r["vanilla"] for r in adv], [r["benefit"] for r in adv],
           marker="*", s=86, color=ACCENT, edgecolor=SURF, linewidth=0.5,
           zorder=6)
ax.scatter([pro["vanilla"]], [pro["benefit"]], marker="o", s=30,
           facecolor="none", edgecolor=ACCENT, linewidth=1.2, zorder=6)

# direct labels with leaders (fanned to avoid the 38-44 cluster collision)
lab_pos = {"ACDC night": (17.0, 4.6, "center"), "Foggy Zurich fog": (27.4, 5.2, "center"),
           "BDD rainy": (33.2, -5.2, "center"), "ACDC snow": (40.8, -7.2, "center"),
           "ACDC rain": (47.6, -7.2, "center"), "BDD snowy": (35.4, 9.6, "center"),
           "ACDC fog": (55.0, -6.3, "center")}
short = {"ACDC night": "night", "Foggy Zurich fog": "FZ fog", "BDD rainy": "rainy",
         "ACDC snow": "snow", "ACDC rain": "rain", "BDD snowy": "snowy",
         "ACDC fog": "fog"}
for r in adv:
    tx, ty, ha = lab_pos[r["condition"]]
    ax.annotate(short[r["condition"]], xy=(r["vanilla"], r["benefit"]),
                xytext=(tx, ty), fontsize=5.9, color=ACCENT, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.5, alpha=0.55,
                                shrinkA=1, shrinkB=2.5), zorder=5)
# BDD clear is the only open marker and is named in the legend; a direct label here
# would have to cross the BDD-snowy star (41.3, 1.6 vs 41.6, 0.5), so it is identified by the legend alone.

# the fog clincher
fog = next(r for r in adv if r["condition"] == "ACDC fog")
ax.annotate("real fog: trend predicts +2.1,\nadaptation delivers −2.5",
            xy=(fog["vanilla"] + 0.6, fog["benefit"]), xytext=(60.5, 12.0),
            fontsize=6.0, color=INK, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.9,
                            shrinkA=2, shrinkB=3, mutation_scale=7,
                            connectionstyle="arc3,rad=0.22"), zorder=7)

chip(ax, 0.985, 0.975, ["7/7 adverse below trend",
                        "mean −5.4 mIoU (−1.33 SD)",
                        "all bootstrap CIs exclude 0"],
     loc="axes", ha="right", va="top", fontsize=6.0)

# legend (custom, one row)
h = [mpl.lines.Line2D([], [], marker="o", ls="", ms=3.4, color=CLOUD,
                      label="95 synthetic conditions"),
     mpl.lines.Line2D([], [], color=FITLN, lw=1.3,
                      label="difficulty fit ±1 SD"),
     mpl.lines.Line2D([], [], marker="*", ls="", ms=7.5, color=ACCENT,
                      markeredgecolor=SURF, label="7 real adverse"),
     mpl.lines.Line2D([], [], marker="o", ls="", ms=4.6, mfc="none",
                      color=ACCENT, label="BDD clear (probe)")]
_pos = ax.get_position()                       # free a band above the plot for the legend
ax.set_position([_pos.x0, _pos.y0, _pos.width, _pos.height * 0.885])
leg = ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
                handletextpad=0.25, columnspacing=0.75, borderaxespad=0.0,
                frameon=False)

ax.set_xlim(6, 72.5); ax.set_ylim(-9, 35.5)
ax.set_xlabel("baseline mIoU on the condition (difficulty)", labelpad=1.5)
ax.set_ylabel("adaptation benefit (mIoU)", labelpad=1.5)
despine(ax)

# ================================================================ verdict strip
axV = fig.add_subplot(gs[1, :]); axV.set_xlim(0, 1); axV.set_ylim(0, 1)
axV.axis("off")
axV.add_patch(mpl.patches.FancyBboxPatch((0.0, 0.04), 1.0, 0.92,
    boxstyle="round,pad=0.002,rounding_size=0.012",
    facecolor="#F4F6F8", edgecolor="#D4DAE1", linewidth=0.8, zorder=1))
seg = [("MEASURE", ACCENT, 0.062,
        "28/28 real condition×approach pairs\nbelow trend (2.4–5.4 mIoU)"),
       ("DIAGNOSE", BLUE, 0.066,
        "supervised ref. +5.8 ≈ +5.7 predicted:\nthe signal fails, not the capacity"),
       ("ACT", GREEN, 0.040,
        "synthetic-τ gate never abstains; real-\ncalibrated gate recovers 82–94%")]
xs = [0.010, 0.345, 0.690]
for (tag, col, tw, txt), x0 in zip(seg, xs):
    axV.add_patch(mpl.patches.FancyBboxPatch((x0 + 0.004, 0.24), tw, 0.52,
        boxstyle="round,pad=0.002,rounding_size=0.01",
        facecolor=col, edgecolor="none", zorder=2))
    axV.text(x0 + 0.004 + tw / 2, 0.51, tag, ha="center", va="center",
             fontsize=5.2, weight="bold", color="white", zorder=3)
    axV.text(x0 + tw + 0.014, 0.50, txt, ha="left", va="center",
             fontsize=5.45, color=INK, linespacing=1.35, zorder=3)
for xd in (0.338, 0.683):
    axV.plot([xd, xd], [0.16, 0.84], color="#D4DAE1", lw=0.8, zorder=2)

fig.savefig("./out/fig_teaser.pdf")
fig.savefig("./out/fig_teaser.png", dpi=300)
print("wrote out/fig_teaser.{pdf,png}")
