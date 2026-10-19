#!/usr/bin/env python3
"""
Fig. 2 (full width): architecture replication. Each panel is that
architecture's own 95-condition fit (exact released parameters) with the same
seven real conditions placed against it. The synthetic cloud for a panel is
drawn when its grid file exists; the R101/FCN clouds are assembled by running
make_grid.py (or this script) inside results/ where the per-type JSONs live.
Nothing is ever synthesized: a missing cloud renders as a note, not fake dots.
"""
import json, os, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
from pubstyle import (use_pubstyle, despine, chip, ACCENT, BLUE, GREEN, INK,
                      MUTED, CLOUD, FITLN, BAND, SURF)

use_pubstyle(7.0)
D = "./data"
meta = json.load(open(f"{D}/spine_meta.json"))

PANELS = [("primary", "(a) DeepLabV3+/ResNet-50 · primary", "grid_primary_pda.json"),
          ("r101",    "(b) DeepLabV3+/ResNet-101 · depth",   "grid_r101_pda.json"),
          ("fcn50",   "(c) FCN/ResNet-50 · head",            "grid_fcn50_pda.json")]

fig, axes = plt.subplots(1, 3, figsize=(6.875, 2.08), sharey=True)
fig.subplots_adjust(left=0.052, right=0.995, top=0.80, bottom=0.175, wspace=0.055)

for ax, (key, title, gridfile) in zip(axes, PANELS):
    f = meta["fits"][key]
    a, b, sd, r2 = f["a"], f["b"], f["std"], f["r2"]
    xx = np.linspace(5, 73, 200)
    ax.fill_between(xx, a + b * xx - sd, a + b * xx + sd, color=BAND,
                    alpha=0.10, lw=0, zorder=1)
    ax.plot(xx, a + b * xx, color=FITLN, lw=1.2, zorder=3)

    gp = os.path.join(D, gridfile)
    if os.path.exists(gp):
        g = json.load(open(gp))
        V = np.concatenate([np.asarray(g["vanilla"][t]) * 100 for t in sorted(g["vanilla"])])
        B = np.concatenate([(np.asarray(g["pda"][t]) - np.asarray(g["vanilla"][t])) * 100
                            for t in sorted(g["vanilla"])])
        ax.scatter(V, B, s=6.5, color=CLOUD, lw=0, zorder=2)
    else:
        ax.text(0.5, 0.86, "95-condition cloud renders from results/\n"
                f"({key}_pertype_*.json)", transform=ax.transAxes,
                ha="center", va="center", fontsize=5.2, color="#AEB6C2",
                style="italic", zorder=2)

    adv = [r for r in meta["real"][key] if r["kind"] == "adverse"]
    pro = [r for r in meta["real"][key] if r["kind"] == "probe"][0]
    ax.scatter([r["vanilla"] for r in adv], [r["benefit"] for r in adv],
               marker="*", s=64, color=ACCENT, edgecolor=SURF, linewidth=0.45,
               zorder=6)
    ax.scatter([pro["vanilla"]], [pro["benefit"]], marker="o", s=22,
               facecolor="none", edgecolor=ACCENT, linewidth=1.1, zorder=6)

    m = lambda x, spec: format(x, spec).replace("-", "\u2212")   # true minus signs, as in the rest of the figure
    # stats live on a muted line under the title, outside the plot, so no chip competes with data or annotations
    ax.text(0.0, 1.012, f"slope {m(b, '+.3f')} · R² {f['r2']:.2f} · mean shortfall {m(f['mean_shortfall'], '+.2f')} · "
            f"{f['n_below']}/7 below", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, color=MUTED)

    ax.set_title(title, fontsize=7.0, loc="left", pad=11, color=INK)
    ax.set_xlim(5, 73); ax.set_ylim(-10.5, 33.5)
    despine(ax)

# panel-specific annotations
axA, axB, axC = axes
fogA = next(r for r in meta["real"]["primary"] if r["condition"] == "ACDC fog")
axA.annotate("fog: −2.5 where the\ntrend predicts +2.1",
             xy=(fogA["vanilla"], fogA["benefit"]), xytext=(44.5, 14.5),
             fontsize=5.5, color=INK, ha="center",
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8,
                             shrinkA=2, shrinkB=3, mutation_scale=6,
                             connectionstyle="arc3,rad=0.22"), zorder=7)
fzC = next(r for r in meta["real"]["fcn50"] if r["condition"] == "Foggy Zurich fog")
axC.annotate("Foggy Zurich: +4.1 delivered,\n+7.7 predicted: below-trend\nis not adaptation failure",
             xy=(fzC["vanilla"], fzC["benefit"]), xytext=(47.0, 24.0),
             fontsize=5.5, color=INK, ha="center",
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8,
                             shrinkA=2, shrinkB=3, mutation_scale=6,
                             connectionstyle="arc3,rad=-0.22"), zorder=7)
nB = next(r for r in meta["real"]["r101"] if r["condition"] == "ACDC night")
axB.annotate("night: −10.9 below a\ntighter law (R² = 0.69)",
             xy=(nB["vanilla"], nB["benefit"]), xytext=(42.0, 26.5),
             fontsize=5.5, color=INK, ha="center",
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8,
                             shrinkA=2, shrinkB=3, mutation_scale=6,
                             connectionstyle="arc3,rad=-0.2"), zorder=7)

axA.set_ylabel("adaptation benefit (mIoU)", labelpad=1.5)
fig.supxlabel("baseline mIoU on the condition (difficulty)", fontsize=7.0,
              y=0.015, color=INK)

h = [mpl.lines.Line2D([], [], marker="o", ls="", ms=3.2, color=CLOUD,
                      label="95 synthetic conditions (per architecture)"),
     mpl.lines.Line2D([], [], color=FITLN, lw=1.2,
                      label="own difficulty fit ±1 SD"),
     mpl.lines.Line2D([], [], marker="*", ls="", ms=7, color=ACCENT,
                      markeredgecolor=SURF, label="7 real adverse"),
     mpl.lines.Line2D([], [], marker="o", ls="", ms=4.2, mfc="none",
                      color=ACCENT, label="BDD clear (probe)")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=4,
           columnspacing=1.1, handletextpad=0.35, fontsize=6.0, frameon=False)

fig.savefig("./out/fig_arch_v2.pdf")
fig.savefig("./out/fig_arch_v2.png", dpi=300)
print("wrote out/fig_arch_v2.{pdf,png}")
