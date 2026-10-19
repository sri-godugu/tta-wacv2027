#!/usr/bin/env python3
"""
sky_luminance_correlate.py -- pre-registered luminance correlate (results/preregistration_step_curve.md).

Per ACDC val image with >=1% GT-sky and >=1% GT-building pixels:
  contrast = |mean Y(GT sky) - mean Y(GT building)|  (Rec.601 luma, full-resolution image and labels)
  outcome  = sky IoU(adapted) - sky IoU(same-pool vanilla), from the per-image confusion matrices.
Prediction: Spearman rho(contrast, delta sky IoU) > 0.
Image i of a run's per-image CM file is image i of ACDCDataset(...).image_paths (eval order, shuffle off).

Run on a compute node from code/:  python ../results/sky_luminance_correlate.py
Writes results/sky_luminance_correlate.json.
"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

R = os.environ.get("ROBUSTSEG_ROOT", ".")   # repository root
sys.path.insert(0, os.path.join(R, "code"))
from data.datasets import ACDCDataset  # noqa: E402

SKY_ID, BLD_ID = 23, 11          # Cityscapes labelIds (ACDC gt_labelIds)
SKY, BLD = 10, 2                 # trainIds used in the confusion matrices
SC = os.path.join(R, "results", "stepcurve")
RUNS = [("PDA@1", "h200", "pda_s1_lr0.001"), ("PDA@10", "h200", "pda_s10_lr0.001"),
        ("TENT@1", "v100", "tent_s1_lr0.001"), ("TENT@10", "v100", "tent_s10_lr0.001")]


def sky_iou_per_image(path):
    A = np.load(path)["cm"].astype(np.float64)
    tp = A[:, SKY, SKY]
    den = A[:, SKY, :].sum(1) + A[:, :, SKY].sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)


def contrasts(cond):
    ds = ACDCDataset(root=os.path.join(R, "datasets", "acdc"), condition=cond, split="val")
    out = []
    for ip, lp in zip(ds.image_paths, ds.label_paths):
        img = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float64)
        lab = np.asarray(Image.open(lp))
        y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        sky, bld = lab == SKY_ID, lab == BLD_ID
        n = lab.size
        if sky.sum() < 0.01 * n or bld.sum() < 0.01 * n:
            out.append(np.nan)
        else:
            out.append(abs(y[sky].mean() - y[bld].mean()))
    return np.array(out)


def main():
    res = {}
    for cond in ["fog", "rain", "snow"]:
        c = contrasts(cond)
        print(f"\nACDC {cond}: {np.isfinite(c).sum()}/{len(c)} images with >=1% sky and building; "
              f"contrast median {np.nanmedian(c):.1f} (0-255 luma)")
        for label, pool, tag in RUNS:
            van = sky_iou_per_image(os.path.join(SC, pool, "per_image_cm", f"van_acdc_{cond}.npz"))
            ada = sky_iou_per_image(os.path.join(SC, pool, "per_image_cm", f"{tag}_acdc_{cond}.npz"))
            assert len(van) == len(ada) == len(c), (len(van), len(ada), len(c))
            d = 100 * (ada - van)
            ok = np.isfinite(c) & np.isfinite(d)
            rho, p = spearmanr(c[ok], d[ok])
            lo, hi = np.median(d[ok][c[ok] <= np.median(c[ok])]), np.median(d[ok][c[ok] > np.median(c[ok])])
            print(f"  {label:8s} n={ok.sum():3d}  Spearman rho(contrast, d sky IoU) = {rho:+.3f}  p={p:.2g}   "
                  f"median d sky IoU: low-contrast half {lo:+.1f} vs high-contrast half {hi:+.1f}")
            res[f"{cond}/{label}"] = {"n": int(ok.sum()), "rho": float(rho), "p": float(p),
                                      "median_d_low_contrast": float(lo), "median_d_high_contrast": float(hi)}
    json.dump(res, open(os.path.join(R, "results", "sky_luminance_correlate.json"), "w"), indent=2)
    print("\nwrote results/sky_luminance_correlate.json")


if __name__ == "__main__":
    main()
