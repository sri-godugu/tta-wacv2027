#!/usr/bin/env python3
"""
xdataset_contrast_exploratory.py -- EXPLORATORY (post hoc; not in preregistration_step_curve.md).

Question: does sky-vs-building luminance contrast differ ACROSS datasets/conditions in a way that lines up with
where the sky->building merge appears at 1 step (ACDC fog/rain/snow) vs only with budget (BDD)?
Per condition: distribution of per-image contrast |Y_sky - Y_building| (images with >=1% of each class),
alongside vanilla / PDA@1 / PDA@10 sky->building rates, plus the within-condition Spearman rho for the
non-ACDC conditions (also exploratory). Supplement / rebuttal use only.

Run on a compute node from code/:  python ../results/xdataset_contrast_exploratory.py
Writes results/xdataset_contrast_exploratory.json.
"""
import json, os, sys
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

R = os.environ.get("ROBUSTSEG_ROOT", ".")   # repository root
sys.path.insert(0, os.path.join(R, "code"))
from data.datasets import ACDCDataset, BDD100KDataset, FoggyZurichDataset, _LABEL_MAP  # noqa: E402

SKY, BLD = 10, 2
SC = os.path.join(R, "results", "stepcurve", "h200", "per_image_cm")
DS = os.path.join(R, "datasets")
ATTR = os.path.join(DS, "bdd100k", "bdd_segval_attrs.json")
CONDS = [
    ("acdc_fog",   lambda: ACDCDataset(os.path.join(DS, "acdc"), condition="fog", split="val"), None),
    ("acdc_rain",  lambda: ACDCDataset(os.path.join(DS, "acdc"), condition="rain", split="val"), None),
    ("acdc_snow",  lambda: ACDCDataset(os.path.join(DS, "acdc"), condition="snow", split="val"), None),
    ("acdc_night", lambda: ACDCDataset(os.path.join(DS, "acdc"), condition="night", split="val"), None),
    ("fz_fog",     lambda: FoggyZurichDataset(os.path.join(DS, "foggy_zurich", "Foggy_Zurich")), None),
    ("bdd_rainy",  lambda: BDD100KDataset(os.path.join(DS, "bdd100k"), split="val", condition="rainy", attr_json=ATTR), 40),
    ("bdd_snowy",  lambda: BDD100KDataset(os.path.join(DS, "bdd100k"), split="val", condition="snowy", attr_json=ATTR), None),
    ("bdd_clear",  lambda: BDD100KDataset(os.path.join(DS, "bdd100k"), split="val", condition="clear", attr_json=ATTR), None),
]


def cms(tag, c):
    return np.load(os.path.join(SC, f"{tag}_{c}.npz"))["cm"].astype(np.float64)


def sky_iou(A):
    tp = A[:, SKY, SKY]; den = A[:, SKY, :].sum(1) + A[:, :, SKY].sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)


def rate(A):
    M = A.sum(0); return 100 * M[SKY, BLD] / M[SKY].sum()


def main():
    out = {}
    print(f"{'condition':11s} {'n_ok/n':>7s} {'contrast med [IQR]':>22s} | {'sky->bld %: van  PDA1  PDA10':>29s} | "
          f"{'rho(c, dSkyIoU) PDA1 / PDA10':>28s}")
    for name, mk, keep in CONDS:
        ds = mk()
        pairs = list(zip(ds.image_paths, ds.label_paths))
        if keep is not None:
            pairs = pairs[:keep]          # drop_last at bs=2 scored only the first `keep` images
        con = []
        for ip, lp in pairs:
            img = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float64)
            lab = np.asarray(Image.open(lp)).astype(np.int64)
            if lab.ndim == 3:
                lab = lab[..., 0]
            if getattr(ds, "_apply_cs_mapping", True):
                lab = _LABEL_MAP[lab]
            if lab.shape != img.shape[:2]:
                lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize(img.shape[1::-1], Image.NEAREST)).astype(np.int64)
            y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            sky, bld = lab == SKY, lab == BLD
            con.append(abs(y[sky].mean() - y[bld].mean()) if sky.mean() >= 0.01 and bld.mean() >= 0.01 else np.nan)
        con = np.array(con)
        V, P1, P10 = cms("van", name), cms("pda_s1_lr0.001", name), cms("pda_s10_lr0.001", name)
        assert len(V) == len(con), (name, len(V), len(con))
        ok = np.isfinite(con)
        rhos = []
        for A in (P1, P10):
            d = sky_iou(A) - sky_iou(V); m = ok & np.isfinite(d)
            rhos.append(spearmanr(con[m], d[m])[0] if m.sum() > 5 else np.nan)
        q1, med, q3 = np.nanpercentile(con, [25, 50, 75])
        rv, r1, r10 = rate(V), rate(P1), rate(P10)
        print(f"{name:11s} {ok.sum():3d}/{len(con):<3d} {med:7.1f} [{q1:5.1f}, {q3:5.1f}]    | "
              f"{rv:13.1f} {r1:5.1f} {r10:5.1f} | {rhos[0]:+12.3f} / {rhos[1]:+.3f}")
        out[name] = {"n_ok": int(ok.sum()), "n": len(con), "contrast_median": float(med), "contrast_iqr": [float(q1), float(q3)],
                     "sky_to_bld": {"van": rv, "pda1": r1, "pda10": r10}, "rho_pda1": float(rhos[0]), "rho_pda10": float(rhos[1])}
    json.dump({"status": "EXPLORATORY", "rows": out},
              open(os.path.join(R, "results", "xdataset_contrast_exploratory.json"), "w"), indent=2)
    print("\nwrote results/xdataset_contrast_exploratory.json  [EXPLORATORY]")


if __name__ == "__main__":
    main()
