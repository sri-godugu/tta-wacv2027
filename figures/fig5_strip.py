#!/usr/bin/env python3
"""
render_strip.py -- qualitative strip for ACDC fog (pre-registered indices 14 / 42 / 51 = 25th / 50th / 75th
percentile of per-image sky-IoU drop under PDA@1; see preregistration_step_curve.md).

Predictions are the argmax maps saved from the SAME eval pass that produced the reported metrics (--save_preds).
Before drawing, every saved label map is checked against the ground-truth file loaded here, so index i is provably
the image shown.

Writes  strip_semantic.png/.pdf  (A: Cityscapes palette)
        strip_skyfate.png/.pdf   (B: every GT-sky pixel coloured by its prediction: sky / building / other)
Usage: python render_strip.py <strip_dir> <out_dir>
"""
import os, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
IDX = [14, 42, 51]
FILES = {14: "GOPR0476_frame_000869", 42: "GOPR0476_frame_001033", 51: "GP010476_frame_000022"}
PCT = {14: "25th", 42: "50th", 51: "75th"}
COLS = [("Vanilla", "h200_strip_van_acdc_fog"),
        ("PDA, 1 step", "h200_strip_pda_s1_lr0.001_acdc_fog"),
        ("PDA, 10 steps", "h200_strip_pda_s10_lr0.001_acdc_fog"),
        ("TENT, 10 steps", "v100_strip_tent_s10_lr0.001_acdc_fog"),
        ("Supervised ref., 10 steps", "h200_strip_ora_s10_lr0.001_acdc_fog")]
SKY, BLD = 10, 2

# Cityscapes labelId -> trainId (standard 19-class mapping)
ID2TRAIN = np.full(256, 255, np.uint8)
for lid, tid in {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}.items():
    ID2TRAIN[lid] = tid
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]],
                   np.uint8)
# Sky-fate colours: categorical slots 1-3 of the validated palette (light mode), fixed order.
FATE = {"sky": (0x2a, 0x78, 0xd6), "building": (0xeb, 0x68, 0x34), "other": (0x1b, 0xaf, 0x7a)}
INK, INK2 = "#0b0b0b", "#52514e"


def load_preds(stem):
    p = os.path.join(SRC, stem + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {int(i): (z["pred"][k], z["label"][k]) for k, i in enumerate(z["idx"])}


def colourise(m):
    out = np.zeros(m.shape + (3,), np.uint8)
    ok = m < 19
    out[ok] = PALETTE[m[ok]]
    return out


def skyfate(img, gt, pred):
    grey = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    base = (255 - 0.45 * (255 - grey)).astype(np.uint8)          # faded greyscale context
    out = np.repeat(base[..., None], 3, axis=2)
    sky = gt == SKY
    for name, mask in [("sky", sky & (pred == SKY)), ("building", sky & (pred == BLD)),
                       ("other", sky & (pred != SKY) & (pred != BLD))]:
        out[mask] = FATE[name]
    return out


def sky_stats(gt, pred):
    sky = gt == SKY
    tp = (sky & (pred == SKY)).sum(); den = sky.sum() + (pred == SKY).sum() - tp
    return 100 * tp / max(den, 1), 100 * (sky & (pred == BLD)).sum() / max(sky.sum(), 1)


preds = {name: load_preds(stem) for name, stem in COLS}
# Everything is drawn and scored at the resolution the eval pipeline scored at (labels resized to the crop size,
# nearest), using the saved label as ground truth. The GT files are used only to prove which image index i is.
ref = next(p for p in preds.values() if p is not None)
H, W = ref[IDX[0]][1].shape


def gt_file(i):
    g = ID2TRAIN[np.asarray(Image.open(os.path.join(SRC, FILES[i] + "_gt_labelIds.png")))]
    return np.asarray(Image.fromarray(g).resize((W, H), Image.NEAREST))


files_gt = {i: gt_file(i) for i in IDX}
rows = []
for i in IDX:
    img = np.asarray(Image.open(os.path.join(SRC, FILES[i] + "_rgb_anon.png")).convert("RGB").resize((W, H), Image.BILINEAR))
    for name, p in preds.items():
        if p is None:
            continue
        lab = p[i][1]
        own = (lab == files_gt[i]).mean()
        others = max((lab == files_gt[j]).mean() for j in IDX if j != i)
        assert own > 0.995 and own - others > 0.05, \
            f"index {i} / '{name}': own-file agreement {own:.4f}, best other file {others:.4f}"
    print(f"index {i}: saved labels match {FILES[i]} ({own:.4f}) vs best other file ({others:.4f})")
    rows.append((i, img, ref[i][1]))
print(f"index check passed; drawing at the scored resolution {H}x{W}")


def draw(kind):
    ncol = 2 + len(COLS)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.55 * ncol, 1.36 * len(rows) + (0.55 if kind == "skyfate" else 0.3)), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    for r, (i, img, gt) in enumerate(rows):
        panels = [("Input (ACDC fog)", img), ("Ground truth", colourise(gt) if kind == "semantic" else skyfate(img, gt, gt))]
        for name, _ in COLS:
            p = preds[name]
            if p is None:
                panels.append((name, None)); continue
            pr = p[i][0]
            panels.append((name, (colourise(pr) if kind == "semantic" else skyfate(img, gt, pr), sky_stats(gt, pr))))
        for c, (title, content) in enumerate(panels):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7.5, color=INK, pad=4)
            if content is None:   # same-shape placeholder so the row stays aligned
                ax.imshow(np.full((H, W, 3), 236, np.uint8))
                ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=7, color=INK2, transform=ax.transAxes)
                continue
            if isinstance(content, tuple):
                arr, (iou, s2b) = content
                ax.imshow(arr)
                ax.text(0.02, 0.04, f"sky IoU {iou:.0f}   sky→bld {s2b:.0f}%", transform=ax.transAxes, fontsize=5.6,
                        color=INK, bbox=dict(boxstyle="square,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
            else:
                ax.imshow(content)
            if c == 0:
                ax.set_ylabel(f"#{i} ({PCT[i]} pct.)", fontsize=6.5, color=INK2)
    if kind == "skyfate":
        handles = [Patch(fc="#%02x%02x%02x" % FATE[k], label=lab) for k, lab in
                   [("sky", "GT sky → predicted sky"), ("building", "GT sky → predicted building"),
                    ("other", "GT sky → any other class")]]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, labelcolor=INK,
                   bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.025, right=0.997, top=0.93, bottom=0.09 if kind == "skyfate" else 0.01, wspace=0.025, hspace=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"strip_{kind}.{ext}"), facecolor=fig.get_facecolor())
    plt.close(fig)


draw("semantic"); draw("skyfate")
print("wrote", [f for f in sorted(os.listdir(OUT)) if f.startswith("strip_")])
