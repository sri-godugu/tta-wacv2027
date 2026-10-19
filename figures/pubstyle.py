"""
pubstyle.py — shared figure style for the UrbanState submission.

Design language:
pastel stage cards with micro-captions, dashed grouping frames, one reserved
accent for the finding, per-panel stat chips, recessive axes, direct labels.

Palette: Okabe–Ito trio (CVD-validated: worst adjacent-pair dE 18.0 protan,
18.7 normal; all contrast >= 3:1 on near-white surface) + neutral context
layers. Identity is never color-alone: marker shape is the secondary encoding
(cloud dots vs. stars vs. squares vs. ticks).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ------------------------------------------------------------------ colors
ACCENT  = "#D55E00"   # the finding: real conditions / signal gap
BLUE    = "#0072B2"   # supervised reference
GREEN   = "#009E73"   # synthetic-trend prediction accents
INK     = "#20262E"   # primary text
MUTED   = "#5B6472"   # secondary text
CLOUD   = "#AEB6C2"   # synthetic-condition dots (context layer)
FITLN   = "#39434F"   # difficulty-fit line
BAND    = "#39434F"   # +-1 SD band (low alpha)
SURF    = "#FFFFFF"
GRID    = "#E4E7EC"

# pastel stage cards (schematic)
P_GREEN = ("#EAF5EC", "#BFDCC6")
P_LAV   = ("#F0EBF7", "#CFC2E6")
P_PEACH = ("#FBEDE0", "#EFC9A4")
P_BLUE  = ("#E8F1F9", "#BDD6EC")
P_CREAM = ("#FBF5DF", "#E8D9A0")

CONDITION_ORDER = ["ACDC fog", "ACDC night", "ACDC rain", "ACDC snow",
                   "BDD rainy", "BDD snowy", "Foggy Zurich fog"]

# ------------------------------------------------------------------ rc
def use_pubstyle(base=7.2):
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base, "axes.titlesize": base + 0.8,
        "axes.labelsize": base, "xtick.labelsize": base - 0.4,
        "ytick.labelsize": base - 0.4, "legend.fontsize": base - 0.4,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#C9CFD8", "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.55,
        "axes.axisbelow": True,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False,
        "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in the PDF
    })

def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)

def chip(ax, x, y, lines, loc="data", fc="#F4F6F8", ec="#D4DAE1",
         fontsize=None, ha="left", va="top", pad=0.32, lw=0.7,
         text_colors=None, weight_first=True, zorder=10):
    """Rounded stat chip; `lines` is a list of strings."""
    fs = fontsize or mpl.rcParams["font.size"] - 0.6
    txt = "\n".join(lines)
    t = ax.text(x, y, txt, transform=ax.transAxes if loc == "axes" else ax.transData,
                fontsize=fs, ha=ha, va=va, color=INK, linespacing=1.45, zorder=zorder,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor=fc,
                          edgecolor=ec, linewidth=lw))
    return t

def stage_card(ax, x, y, w, h, fill, edge, number=None, title="", body="",
               tfs=6.6, bfs=5.4, num_color=INK, lw=1.0, body_w=1.0):
    """Rounded pastel card: badge + bold title on the top row, muted body
    anchored below it (va=top). body_w limits body to the left fraction
    so a right-side glyph never collides."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.028",
                         facecolor=fill, edgecolor=edge, linewidth=lw,
                         mutation_aspect=1.0, zorder=2)
    ax.add_patch(box)
    ty = y + h - 0.066
    tx = x + 0.040
    if number is not None:
        ax.add_patch(FancyBboxPatch((x + 0.028, ty - 0.040), 0.062, 0.078,
                     boxstyle="round,pad=0.004,rounding_size=0.012",
                     facecolor=SURF, edgecolor=edge, linewidth=0.9, zorder=3))
        ax.text(x + 0.059, ty, number, ha="center", va="center",
                fontsize=tfs - 0.4, color=num_color, weight="bold", zorder=4)
        tx = x + 0.118
    ax.text(tx, ty, title, ha="left", va="center",
            fontsize=tfs, weight="bold", color=INK, zorder=4)
    if body:
        ax.text(x + 0.040, ty - 0.078, body, ha="left", va="top",
                fontsize=bfs, color=MUTED, linespacing=1.45, zorder=4,
                wrap=False)
    return box

def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.1, ls="-", z=5, arc=0.0,
          mutation=7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1,
                                mutation_scale=mutation,
                                connectionstyle=f"arc3,rad={arc}"))

def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    yhat = a + b * np.asarray(x, float)
    r = np.asarray(y, float) - yhat
    r2 = 1 - (r @ r) / float(((y - np.mean(y)) ** 2).sum())
    return a, b, r2, float(np.std(r, ddof=1))
