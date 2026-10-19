# Cityscapes-C v2 — seeded, per-image reproducible

A seeded regeneration of Cityscapes-C (19 corruption types × 5 severities × the 500 Cityscapes val images = 47,500 images),
made with `imagecorruptions` 1.1.2. Unlike an unseeded generation, **every image can be regenerated on its own, in any
order, on any number of workers**, and checked byte-for-byte against this manifest.

## Why a v2

The commonly used unseeded generation draws the 11 stochastic corruptions (gaussian/shot/impulse/speckle noise,
glass/motion blur, fog, frost, snow, spatter, elastic) from NumPy's global RNG, so its realizations can't be reproduced.
Re-drawing them moves per-condition vanilla mIoU by a median 0.16 and up to 0.88 mIoU (our measurement, 55 conditions,
DeepLabV3+/ResNet-50). The 8 deterministic types reproduce our original v1 numbers within 0.019 mIoU.

## Seeding scheme

    seed = SeedSequence([SEED0, img_idx, TYPE_IDX[corruption], severity]).generate_state(1)[0]
    np.random.seed(seed)                       # global NumPy RNG, used by most corruptions
    skimage.util.random_noise(..., rng=seed)   # impulse_noise: scikit-image >= 0.19 ignores np.random.seed

`SEED0 = 20261018`. `img_idx` is the image's position in sorted(city) / sorted(filename) order over
`leftImg8bit/val`. `TYPE_IDX` is its position in `imagecorruptions.get_corruption_names("all")` (listed in
`MANIFEST_env.json`). Both seeds are set immediately before each `corrupt()` call.

## Files

| file | content |
|---|---|
| `MANIFEST.csv` | one row per output image: `img_idx, image, corruption, severity, seed, sha256` (sha256 of the written PNG) |
| `MANIFEST_env.json` | SEED0, scheme, type order, and the exact package / codec versions used (Pillow, libjpeg, zlib, OpenCV, …) |
| `generate_cityscapes_c_v2.py` | the generator (writes `<out>/<corruption>/<severity>/<city>/<file>.png`) |
| `SHA256SUMS` | checksums of the three files above |

## Regenerate and verify

    python generate_cityscapes_c_v2.py --cs_root <cityscapes> --out_root <out> --seed0 20261018 [--corruptions fog]

The generator checks determinism itself: it regenerates a random sample of items in shuffled order in a fresh process
pool and requires byte-identical PNGs. To check a single image, compute the sha256 of
`<out>/<corruption>/<severity>/<image>` and compare it with its `MANIFEST.csv` row.

**Byte-identity holds only within the pinned environment** in `MANIFEST_env.json`. PNG encoding (zlib) and the JPEG
codec used by `jpeg_compression`/`pixelate` can differ across library versions; a different environment should reproduce
the *seeds* exactly but may not reproduce the bytes.
