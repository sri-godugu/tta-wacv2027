#!/usr/bin/env python3
"""
generate_cityscapes_c.py — generate the Cityscapes-C corruption benchmark
from the Cityscapes val split, using the `imagecorruptions` package
(Hendrycks' 19 standard corruptions). Output layout matches what
data/datasets.py's CityscapesCDataset expects:

    <out_root>/<corruption_type>/<severity>/<city>/<filename>.png

Labels are not duplicated — CityscapesCDataset reads them from the
original gtFine/val directory via --cs_root at eval time.

`imagecorruptions` (last released ~2020) breaks against modern numpy>=2.0
(`np.float_` removed) and modern scikit-image (`gaussian(..., multichannel=)`
removed in favor of `channel_axis=`); both are shimmed below before import.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# --- compatibility shims (must run before importing imagecorruptions) ------
if not hasattr(np, "float_"):
    np.float_ = np.float64

import skimage.filters as _skf
_orig_gaussian = _skf.gaussian


def _gaussian_compat(image, sigma=1, multichannel=None, channel_axis=None, **kwargs):
    if multichannel is not None:
        channel_axis = -1 if multichannel else None
    return _orig_gaussian(image, sigma=sigma, channel_axis=channel_axis, **kwargs)


_skf.gaussian = _gaussian_compat

from imagecorruptions import corrupt, get_corruption_names  # noqa: E402


def process_image(task):
    img_path, out_root, corruption_types, severities = task
    city = os.path.basename(os.path.dirname(img_path))
    fname = os.path.basename(img_path)

    # Skip entirely if every output for this image already exists (resume support)
    if all(
        os.path.exists(os.path.join(out_root, c, str(s), city, fname))
        for c in corruption_types for s in severities
    ):
        return fname, 0

    img = np.array(Image.open(img_path).convert("RGB"))
    written = 0
    for c in corruption_types:
        for s in severities:
            out_dir = os.path.join(out_root, c, str(s), city)
            out_path = os.path.join(out_dir, fname)
            if os.path.exists(out_path):
                continue
            os.makedirs(out_dir, exist_ok=True)
            corrupted = corrupt(img, corruption_name=c, severity=s)
            Image.fromarray(corrupted.astype(np.uint8)).save(out_path)
            written += 1
    return fname, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs_root", required=True, help="Cityscapes root (contains leftImg8bit/val)")
    ap.add_argument("--out_root", required=True, help="Cityscapes-C output root")
    ap.add_argument("--corruptions", default="all", help="'all' or comma-separated corruption names")
    ap.add_argument("--severities", default="1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    corruption_types = (
        get_corruption_names("all") if args.corruptions == "all"
        else args.corruptions.split(",")
    )
    severities = [int(s) for s in args.severities.split(",")]

    val_dir = os.path.join(args.cs_root, "leftImg8bit", "val")
    img_paths = []
    for city in sorted(os.listdir(val_dir)):
        city_dir = os.path.join(val_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if fname.endswith(".png"):
                img_paths.append(os.path.join(city_dir, fname))

    total_outputs = len(img_paths) * len(corruption_types) * len(severities)
    print(f"[gen] {len(img_paths)} val images x {len(corruption_types)} corruptions x "
          f"{len(severities)} severities = {total_outputs} total outputs", flush=True)

    tasks = [(p, args.out_root, corruption_types, severities) for p in img_paths]

    done_images = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_image, t) for t in tasks]
        for fut in as_completed(futures):
            fname, written = fut.result()
            done_images += 1
            if done_images % 25 == 0 or done_images == len(tasks):
                print(f"[gen] {done_images}/{len(tasks)} images done", flush=True)

    print("[gen] DONE", flush=True)


if __name__ == "__main__":
    main()
