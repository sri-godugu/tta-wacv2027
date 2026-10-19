#!/usr/bin/env python3
"""
generate_cityscapes_c_v2.py — SEEDED, order-independent Cityscapes-C generation.

v1 (generate_cityscapes_c.py) drew from NumPy's global RNG with no seed across a
process pool, so its realizations cannot be reproduced. v2 seeds every work item
independently:

    seed(img_idx, corruption, severity) =
        SeedSequence([SEED0, img_idx, TYPE_IDX[corruption], severity]).generate_state(1)[0]

and sets it immediately before each corrupt() call, both for NumPy's global RNG and
for skimage.util.random_noise (impulse_noise), which in scikit-image >= 0.19 draws
from its own default_rng and ignores np.random.seed. Any single image can therefore
be regenerated in isolation, in any order, on any number of workers.

img_idx is the position of the image in sorted(city) / sorted(filename) order over
<cs_root>/leftImg8bit/val; TYPE_IDX is the position in get_corruption_names("all").

Outputs (same layout as v1, so CityscapesCDataset reads it unchanged):
    <out_root>/<corruption>/<severity>/<city>/<filename>.png
    <out_root>/MANIFEST_seeds_<types>.csv   img_idx, city/filename, corruption, severity, seed, sha256
    <out_root>/MANIFEST_env.json    SEED0, scheme, package + codec versions
A determinism self-check regenerates a sample of items in a different order and in a
fresh process and requires byte-identical PNGs; the run fails loudly otherwise.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

# --- compatibility shims (must run before importing imagecorruptions) ------
if not hasattr(np, "float_"):
    np.float_ = np.float64

import skimage
import skimage.filters as _skf
import skimage.util as _sku

_orig_gaussian = _skf.gaussian


def _gaussian_compat(image, sigma=1, multichannel=None, channel_axis=None, **kwargs):
    if multichannel is not None:
        channel_axis = -1 if multichannel else None
    return _orig_gaussian(image, sigma=sigma, channel_axis=channel_axis, **kwargs)


_skf.gaussian = _gaussian_compat

# random_noise ignores np.random.seed; route the current item seed into it.
_ITEM_SEED = None
_orig_random_noise = _sku.random_noise


def _random_noise_seeded(image, mode="gaussian", rng=None, clip=True, **kwargs):
    if rng is None:
        rng = _ITEM_SEED
    return _orig_random_noise(image, mode=mode, rng=rng, clip=clip, **kwargs)


_sku.random_noise = _random_noise_seeded

from imagecorruptions import corrupt, get_corruption_names  # noqa: E402

SCHEME = ("SeedSequence([SEED0, img_idx, TYPE_IDX[corruption], severity]).generate_state(1)[0]; "
          "np.random.seed(seed) and skimage.util.random_noise(rng=seed) before each corrupt()")
TYPES = list(get_corruption_names("all"))
TYPE_IDX = {c: i for i, c in enumerate(TYPES)}


def item_seed(seed0, img_idx, corruption, severity):
    return int(np.random.SeedSequence(
        [seed0, img_idx, TYPE_IDX[corruption], severity]).generate_state(1)[0])


def render(img, seed, corruption, severity):
    """Corrupt one image under one item seed; return the PNG bytes that get written."""
    global _ITEM_SEED
    _ITEM_SEED = seed
    np.random.seed(seed)
    out = corrupt(img, corruption_name=corruption, severity=severity)
    buf = io.BytesIO()
    Image.fromarray(out.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def process_image(task):
    """All corruptions x severities for one image. Returns manifest rows."""
    img_idx, img_path, rel, out_root, types, sevs, seed0 = task
    img = np.array(Image.open(img_path).convert("RGB"))
    rows = []
    for c in types:
        for s in sevs:
            seed = item_seed(seed0, img_idx, c, s)
            out_path = os.path.join(out_root, c, str(s), rel)
            if os.path.exists(out_path):          # resume: re-hash what is there
                data = open(out_path, "rb").read()
            else:
                data = render(img, seed, c, s)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp = out_path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, out_path)         # never leave a half-written PNG
            rows.append((img_idx, rel, c, s, seed, hashlib.sha256(data).hexdigest()))
    return rows


def recheck(task):
    """Regenerate one item from scratch (fresh worker) and return its sha256."""
    img_idx, img_path, c, s, seed0 = task
    img = np.array(Image.open(img_path).convert("RGB"))
    return hashlib.sha256(render(img, item_seed(seed0, img_idx, c, s), c, s)).hexdigest()


def env_info(seed0):
    import cv2, scipy, PIL, imagecorruptions
    from PIL import features
    try:
        from importlib.metadata import version
        ic_ver = version("imagecorruptions")
    except Exception:
        ic_ver = getattr(imagecorruptions, "__version__", "unknown")
    return {
        "SEED0": seed0, "scheme": SCHEME, "type_order": TYPES,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scikit-image": skimage.__version__, "scipy": scipy.__version__,
        "pillow": PIL.__version__, "libjpeg": features.version("jpg"),
        "zlib": features.version("zlib"), "opencv": cv2.__version__,
        "imagecorruptions": ic_ver,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs_root", required=True, help="Cityscapes root (contains leftImg8bit/val)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--corruptions", default="all")
    ap.add_argument("--severities", default="1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_recheck", type=int, default=60,
                    help="items regenerated in shuffled order for the determinism self-check")
    args = ap.parse_args()

    types = TYPES if args.corruptions == "all" else args.corruptions.split(",")
    sevs = [int(s) for s in args.severities.split(",")]
    os.makedirs(args.out_root, exist_ok=True)

    env = env_info(args.seed0)
    with open(os.path.join(args.out_root, "MANIFEST_env.json"), "w") as f:
        json.dump(env, f, indent=2)
    print("[gen] env:", json.dumps(env), flush=True)

    val_dir = os.path.join(args.cs_root, "leftImg8bit", "val")
    imgs = []
    for city in sorted(os.listdir(val_dir)):
        cdir = os.path.join(val_dir, city)
        if os.path.isdir(cdir):
            for fn in sorted(os.listdir(cdir)):
                if fn.endswith(".png"):
                    imgs.append((os.path.join(cdir, fn), f"{city}/{fn}"))
    print(f"[gen] {len(imgs)} images x {len(types)} corruptions x {len(sevs)} severities "
          f"= {len(imgs) * len(types) * len(sevs)} outputs, SEED0={args.seed0}", flush=True)

    tasks = [(i, p, rel, args.out_root, types, sevs, args.seed0) for i, (p, rel) in enumerate(imgs)]
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(process_image, tasks, chunksize=1):
            rows.extend(r)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"[gen] {done}/{len(tasks)} images", flush=True)

    rows.sort(key=lambda r: (r[0], TYPE_IDX[r[2]], r[3]))
    # One manifest per invocation, so per-type jobs running in parallel never collide.
    tag = "all" if args.corruptions == "all" else args.corruptions.replace(",", "+")
    with open(os.path.join(args.out_root, f"MANIFEST_seeds_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_idx", "image", "corruption", "severity", "seed", "sha256"])
        w.writerows(rows)

    # Determinism self-check: random items, shuffled order, fresh worker pool.
    rnd = random.Random(args.seed0)
    sample = rnd.sample(rows, min(args.n_recheck, len(rows)))
    rnd.shuffle(sample)
    checks = [(r[0], imgs[r[0]][0], r[2], r[3], args.seed0) for r in sample]
    with ProcessPoolExecutor(max_workers=min(args.workers, 8)) as ex:
        again = list(ex.map(recheck, checks))
    bad = [(r[1], r[2], r[3]) for r, h in zip(sample, again) if h != r[5]]
    print(f"[gen] determinism self-check: {len(sample) - len(bad)}/{len(sample)} byte-identical", flush=True)
    if bad:
        print("[gen] NON-DETERMINISTIC ITEMS:", bad[:20], flush=True)
        sys.exit(1)
    print("[gen] DONE", flush=True)


if __name__ == "__main__":
    main()
