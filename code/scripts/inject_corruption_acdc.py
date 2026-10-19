#!/usr/bin/env python3
"""
inject_corruption_acdc.py — causal intervention: apply a synthetic
Cityscapes-C-style corruption on top of real ACDC adverse-weather images,
to test whether raising measured disagreement (JSD) recovers PDA benefit
on real-world images. Reuses imagecorruptions + the same compatibility
shims as generate_cityscapes_c.py.

Output layout mirrors ACDC's rgb_anon structure so the existing
ACDCDataset loader works unchanged against --data_root <out_root>, with
gt/ symlinked to the original ACDC root (labels are unaffected by the
corruption injection).
"""
import argparse
import os

import numpy as np
from PIL import Image

if not hasattr(np, "float_"):
    np.float_ = np.float64

import skimage.filters as _skf
_orig_gaussian = _skf.gaussian


def _gaussian_compat(image, sigma=1, multichannel=None, channel_axis=None, **kwargs):
    if multichannel is not None:
        channel_axis = -1 if multichannel else None
    return _orig_gaussian(image, sigma=sigma, channel_axis=channel_axis, **kwargs)


_skf.gaussian = _gaussian_compat

from imagecorruptions import corrupt  # noqa: E402

CONDITIONS = ["fog", "night", "rain", "snow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    gt_link = os.path.join(args.out_root, "gt")
    if not os.path.exists(gt_link):
        os.symlink(os.path.join(args.acdc_root, "gt"), gt_link)

    n_total, n_written = 0, 0
    for cond in CONDITIONS:
        img_base = os.path.join(args.acdc_root, "rgb_anon", cond, args.split)
        if not os.path.isdir(img_base):
            continue
        for scene in sorted(os.listdir(img_base)):
            scene_dir = os.path.join(img_base, scene)
            if not os.path.isdir(scene_dir):
                continue
            out_scene_dir = os.path.join(args.out_root, "rgb_anon", cond, args.split, scene)
            os.makedirs(out_scene_dir, exist_ok=True)
            for fname in sorted(os.listdir(scene_dir)):
                if not fname.endswith(".png"):
                    continue
                n_total += 1
                out_path = os.path.join(out_scene_dir, fname)
                if os.path.exists(out_path):
                    continue
                img = np.array(Image.open(os.path.join(scene_dir, fname)).convert("RGB"))
                corrupted = corrupt(img, corruption_name=args.corruption, severity=args.severity)
                Image.fromarray(corrupted.astype(np.uint8)).save(out_path)
                n_written += 1
                if n_written % 50 == 0:
                    print(f"[inject] {n_written} written / {n_total} seen so far", flush=True)

    print(f"[inject] DONE. total={n_total} written={n_written}", flush=True)


if __name__ == "__main__":
    main()
