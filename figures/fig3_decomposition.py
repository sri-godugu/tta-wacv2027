#!/usr/bin/env python3
"""
Fig. 3 (column width): the shortfall decomposition at the 10-step budget.
Per condition: achieved PDA benefit (circle) -> supervised reference (square),
with the signal gap shaded in the accent; the synthetic-trend prediction is a
tick. Reads decomposition_s10.json (released run output).
"""
import json, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects
from pubstyle import (use_pubstyle, despine, chip, ACCENT, BLUE, GREEN,
                      INK, MUTED, SURF, GRID)

use_pubstyle(7.2)
D = json.load(open("./data/decomposition_s10.json"))
rows = D["rows"]
adv = [r for r in rows if r["kind"] == "adverse"]
pro = [r for r in rows if r["kind"] == "probe"][0]
order = adv + [pro]
short = {"ACDC fog": "ACDC fog", "ACDC night": "ACDC night",
         "ACDC rain": "ACDC rain", "ACDC snow": "ACDC snow",
         "BDD rainy": "BDD rainy", "BDD snowy": "BDD snowy",
         "Foggy Zurich fog": "Foggy Zurich", "BDD clear": "BDD clear*"}

fig, ax = plt.subplots(figsize=(3.3125, 2.58))
fig.subplots_adjust(left=0.215, right=0.985, top=0.90, bottom=0.145)

N = len(order)
ys = np.arange(N)[::-1].astype(float)
ys[-1] -= 0.45  # drop the probe a little below a divider

for r, y in zip(order, ys):
    act, ref, pred = r["actual"], r["ceiling"], r["pred"]
    pa = dict(alpha=0.42) if r["kind"] == "probe" else dict(alpha=1.0)
    # signal gap: achieved -> supervised reference
    ax.plot([act, ref], [y, y], color=ACCENT, lw=4.6, alpha=0.22 * pa["alpha"] / 1.0
            if r["kind"] == "adverse" else 0.10,
            solid_capstyle="butt", zorder=2)
    ax.plot([act, ref], [y, y], color="#C9CFD8", lw=0.8, zorder=3,
            alpha=pa["alpha"])
    # recoverability sliver: reference -> prediction (only visible when pred>ref)
    if pred > ref:
        ax.plot([ref, pred], [y, y], color=MUTED, lw=1.6, alpha=0.75 * pa["alpha"],
                linestyle=(0, (1.6, 1.4)), zorder=3)
    # the prediction tick draws above the markers with a thin surface outline, so it stays visible
    # where it coincides with the reference square (e.g. ACDC rain: 4.96 vs 5.05)
    ax.plot([pred], [y], marker="|", ms=8.5, mew=1.5, color=GREEN,
            alpha=pa["alpha"], zorder=7,
            path_effects=[mpl.patheffects.withStroke(linewidth=2.6, foreground=SURF)])
    ax.plot([act], [y], marker="o", ms=4.8, color=ACCENT, mec=SURF, mew=0.5,
            alpha=pa["alpha"], zorder=6)
    ax.plot([ref], [y], marker="s", ms=4.4, color=BLUE, mec=SURF, mew=0.5,
            alpha=pa["alpha"], zorder=6)
    # right-margin signal-gap value
    ax.text(10.55, y, f"+{r['signal_gap']:.1f}", ha="right", va="center",
            fontsize=5.8, color=ACCENT, alpha=min(1.0, pa["alpha"] + 0.15),
            zorder=6)

ax.text(10.55, ys[0] + 0.62, "signal\ngap", ha="right", va="bottom",
        fontsize=5.4, color=ACCENT, linespacing=1.2)

# divider before the probe
ax.axhline(ys[-1] + 0.72, color="#D4DAE1", lw=0.8, ls=(0, (3, 2)), zorder=1)

ax.axvline(0, color="#B9C1CB", lw=0.9, zorder=1)
ax.set_yticks(ys)
ax.set_yticklabels([short[r["condition"]] for r in order], fontsize=6.4)
for tl, r in zip(ax.get_yticklabels(), order):
    tl.set_color(INK if r["kind"] == "adverse" else MUTED)
ax.set_xlim(-3.6, 10.8)
ax.set_ylim(ys[-1] - 0.65, ys[0] + 1.30)
ax.set_xlabel("benefit over no adaptation (mIoU)", labelpad=1.5)
ax.grid(axis="y", visible=False)
despine(ax, keep=("bottom",))
ax.tick_params(axis="y", length=0)

# legend row (top, custom)
h = [mpl.lines.Line2D([], [], marker="o", ls="", ms=4.6, color=ACCENT,
                      mec=SURF, label="achieved (PDA)"),
     mpl.lines.Line2D([], [], marker="s", ls="", ms=4.2, color=BLUE,
                      mec=SURF, label="supervised ref. @10"),
     mpl.lines.Line2D([], [], marker="|", ls="", ms=7.5, mew=1.5, color=GREEN,
                      label="trend predicts")]
leg = ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 0.995),
                ncol=3, columnspacing=0.9, handletextpad=0.25,
                borderaxespad=0.0, fontsize=5.9)

# summary as two muted lines in the free strip above the first row (left); no position inside the
# rows fits between the bars (end ~5.4) and the signal-gap value column (starts ~9.9)
ax.text(-3.5, ys[0] + 0.42, "mean recoverability gap −0.10\nmean signal gap +5.47",
        ha="left", va="bottom", fontsize=5.6, color=MUTED, linespacing=1.35, zorder=8,
        bbox=dict(boxstyle="square,pad=0.15", facecolor=SURF, edgecolor="none"))   # masks the grid lines behind it
ax.text(0.005, 0.015, "*non-adverse probe", transform=ax.transAxes,
        fontsize=5.2, color=MUTED)

fig.savefig("./out/fig_decomp.pdf")
fig.savefig("./out/fig_decomp.png", dpi=300)
print("wrote out/fig_decomp.{pdf,png}")
