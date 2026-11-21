#!/usr/bin/env python3
"""
lr0_control_analysis.py -- the registered BN-statistics control (P7a/P7b; preregistration_step_curve.md, addendum
2026-10-19 05:20 EDT). PDA at one step with lr 0 (tag pda_s1_lr0, pool h200), so only BatchNorm running statistics
adapt, against full PDA@1 (pda_s1_lr0.001, h200) and the same-pool vanilla (van_*, h200; not rerun).

 0. Validity: every lr-0 cell must carry an identity record (eval_lr0_identity.py) showing that every parameter was
    bit-identical after every adapt() call, and that the running statistics moved. Otherwise the run is void.
 1. Primary statistics (stepcurve_analysis.analyse, unchanged): synthetic trend refit on the lr-0 grid (95),
    mean synthetic benefit x, mean real benefit over the 7 adverse, mean real shortfall y, below/7, BDD-clear probe.
 2. Registered test: R_syn = x / 4.07 (P7a: >= 0.80, i.e. x >= +3.26); Delta_short = y - (-5.38)
    (P7b: |Delta_short| <= 1.0, i.e. y in [-6.38, -4.38]). Both -> H_BN supported.
 3. Registered interval: paired joint bootstrap, B=2000 (types x images), same draw sequence as stepcurve_ci.py
    (seed 0), so PDA@1's interval reproduces the published one exactly; lr 0 and PDA@1 share every draw.
 4. Supplementary interval: paired image-level bootstrap, B=10^4 (bootstrap_ci.py scheme: each setting's synthetic
    fit held fixed, images resampled within each real condition, the same draw applied to vanilla, lr 0, PDA@1).
 5. Sky->building on ACDC fog: share of GT-sky pixels predicted as building (summed per-image CMs), same-pool.
 6. Pathway split: lr 0 = statistics pathway; PDA@1 = statistics + gradient; difference = the gradient step's
    contribution, per population, with paired joint-bootstrap intervals.
Bootstrap fractions are printed as exact counts over B, never rounded.

Usage: python lr0_control_analysis.py <stepcurve_dir> [--json lr0_control.json]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stepcurve_analysis as sa

POOL = "h200"
LR0, FULL = "pda_s1_lr0", "pda_s1_lr0.001"
SKY, BLD = 10, 2
REG_X_DEN, REG_Y_FULL = 4.07, -5.38   # as written in the registration


def miou_from_cm(cm):
    """bootstrap_ci.py's mIoU (matches SegmentationMetrics); works on (...,19,19)."""
    tp = np.einsum("...ii->...i", cm).astype(np.float64)
    denom = cm.sum(-2) + cm.sum(-1) - tp
    valid = denom > 0
    iou = np.where(valid, tp / (denom + 1e-8), 0.0)
    return np.where(valid.any(-1), iou.sum(-1) / np.maximum(valid.sum(-1), 1), 0.0) * 100


def load_cm(d, tag, c):
    z = np.load(os.path.join(d, "per_image_cm", f"{tag}_{c}.npz"))
    return z["cm"].astype(np.float64), (z["batch_id"] if "batch_id" in z.files else None)


def frac(k, n):
    return f"{k}/{n} = {k / n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = os.path.join(a.root, POOL)
    out = {"pool": POOL, "lr0_tag": LR0, "full_tag": FULL}

    # ---- 0. validity --------------------------------------------------------------------------------------------
    cells = [f"v2pertype_{t}_sev{s}" for t in sa.TYPES for s in range(1, 6)] + sa.ADVERSE + [sa.PROBE]
    bad, missing, inactive = [], [], []
    for c in cells:
        p = os.path.join(d, "identity", f"{LR0}_{c}.json")
        if not os.path.exists(p):
            missing.append(c); continue
        r = json.load(open(p))
        if not r.get("valid"):
            bad.append(c)
        if not r.get("stats_pathway_active"):
            inactive.append(c)
    calls = sum(json.load(open(p))["adapt_calls"] for p in glob.glob(os.path.join(d, "identity", f"{LR0}_*.json")))
    out["identity"] = {"cells": len(cells), "missing": missing, "not_bit_identical": bad,
                       "stats_not_moved": inactive, "adapt_calls_checked": calls}
    print(f"[0] identity check: {len(cells) - len(missing)}/{len(cells)} cells recorded, {calls} adapt() calls checked; "
          f"not bit-identical: {bad or 'none'}; running stats unmoved: {inactive or 'none'}; missing: {missing or 'none'}")
    if bad or missing:
        print("    -> CONTROL VOID or incomplete under the registered sanity check; nothing below is a verdict.")

    # ---- 1. primary statistics (unchanged machinery) ------------------------------------------------------------
    r0, r1 = sa.analyse(a.root, "pda", 1, "0"), sa.analyse(a.root, "pda", 1, "0.001")
    for name, r in (("lr 0 (stats only)", r0), ("PDA@1 lr 1e-3", r1)):
        print(f"[1] {name:18s} nSyn {r['n_synth']} nReal {r.get('n_real')}  slope {r['slope']:+.3f}  R2 {r['r2']:.2f}  "
              f"syn {r['mean_synth_benefit']:+.2f}  real {r['mean_real_benefit']:+.2f}  shortfall {r['mean_shortfall']:+.2f}"
              f"  below {r['n_below']}/{r['n_real']}  probe resid {r['probe_resid']:+.2f}")
    out["lr0"], out["full"] = r0, r1
    complete = r0["n_synth"] == 95 and r0["n_real"] == 7

    # ---- 2. registered test -------------------------------------------------------------------------------------
    x, y = r0["mean_synth_benefit"], r0["mean_shortfall"]
    R_reg, D_reg = x / REG_X_DEN, y - REG_Y_FULL
    R_unr, D_unr = x / r1["mean_synth_benefit"], y - r1["mean_shortfall"]
    p7a, p7b = R_reg >= 0.80, abs(D_reg) <= 1.0
    agree = (p7a == (R_unr >= 0.80)) and (p7b == (abs(D_unr) <= 1.0))
    out["test"] = {"x": x, "y": y, "R_syn": R_reg, "Delta_short": D_reg, "R_syn_unrounded_den": R_unr,
                   "Delta_short_unrounded_ref": D_unr, "P7a": bool(p7a), "P7b": bool(p7b),
                   "H_BN": bool(p7a and p7b), "same_verdict_with_unrounded_reference": bool(agree)}
    print(f"[2] P7a: R_syn = {x:+.2f} / 4.07 = {R_reg:.3f} (threshold 0.80; x >= +3.26) -> {'HOLDS' if p7a else 'FAILS'}")
    print(f"    P7b: Delta_short = {y:+.2f} - (-5.38) = {D_reg:+.2f} (|.| <= 1.0; y in [-6.38, -4.38]) -> {'HOLDS' if p7b else 'FAILS'}")
    print(f"    H_BN {'SUPPORTED' if p7a and p7b else 'NOT SUPPORTED'}"
          f"   (unrounded reference: R {R_unr:.3f}, Delta {D_unr:+.2f}; same verdict: {agree})")

    # ---- per-image CMs (paired) ---------------------------------------------------------------------------------
    van = {c: load_cm(d, "van", c) for c in sa.ADVERSE + [sa.PROBE]}
    ada = {t: {c: load_cm(d, t, c) for c in sa.ADVERSE + [sa.PROBE]} for t in (FULL, LR0)}
    worst = 0.0
    for c in sa.ADVERSE + [sa.PROBE]:
        for t, M in (("van", van[c]), (FULL, ada[FULL][c]), (LR0, ada[LR0][c])):
            assert M[0].shape == van[c][0].shape, f"{t} {c}: image set differs"
            if M[1] is not None and van[c][1] is not None:
                assert (M[1] == van[c][1]).all(), f"{t} {c}: batch layout differs"
            worst = max(worst, abs(miou_from_cm(M[0].sum(0)) - sa.real(d, t, c)))
    assert worst < 1e-3, f"per-image CMs do not reproduce reported mIoU (max {worst})"
    print(f"    per-image CMs reproduce every reported real mIoU (max |diff| {worst:.1e})")

    syn = {}
    for t in (FULL, LR0):
        per = []
        for ty in sa.TYPES:
            v, ad = sa.synth(d, "van", ty), sa.synth(d, t, ty)
            sev = sorted(set(v) & set(ad))
            per.append((np.array([v[k] for k in sev]), np.array([ad[k] - v[k] for k in sev])))
        syn[t] = per

    # ---- 3. registered joint bootstrap (B=2000, stepcurve_ci.py draw sequence) ----------------------------------
    def stats(type_idx, w):
        res = {}
        for t in (FULL, LR0):
            X = np.concatenate([syn[t][i][0] for i in type_idx]); Y = np.concatenate([syn[t][i][1] for i in type_idx])
            b, a0 = np.polyfit(X, Y, 1)
            rr, bb = [], []
            for c in sa.ADVERSE:
                vm = miou_from_cm(np.tensordot(w[c], van[c][0], 1)); am = miou_from_cm(np.tensordot(w[c], ada[t][c][0], 1))
                bb.append(am - vm); rr.append((am - vm) - (a0 + b * vm))
            res[t] = {"syn": Y.mean(), "real": np.mean(bb), "short": np.mean(rr), "below": sum(v < 0 for v in rr)}
        return res

    B = 2000
    rng = np.random.default_rng(0)
    full_w = {c: np.ones(len(van[c][0])) for c in sa.ADVERSE}
    pt = stats(np.arange(len(sa.TYPES)), full_w)
    dr = {t: {k: [] for k in ("syn", "real", "short")} for t in (FULL, LR0)}
    all7 = {FULL: 0, LR0: 0}
    for _ in range(B):
        ti = rng.integers(0, len(sa.TYPES), len(sa.TYPES))
        ii = {c: np.bincount(rng.integers(0, len(van[c][0]), len(van[c][0])), minlength=len(van[c][0])).astype(float)
              for c in sa.ADVERSE}
        r = stats(ti, ii)
        for t in (FULL, LR0):
            for k in ("syn", "real", "short"):
                dr[t][k].append(r[t][k])
            all7[t] += int(r[t]["below"] == 7)
    ci = lambda v: [float(q) for q in np.percentile(np.asarray(v), [2.5, 97.5])]
    joint = {}
    print(f"[3] registered paired joint bootstrap, B={B} (types x images; seed 0, stepcurve_ci.py sequence)")
    for t, lab in ((FULL, "PDA@1"), (LR0, "lr 0")):
        joint[t] = {k: {"point": float(pt[t][k]), "ci": ci(dr[t][k])} for k in ("syn", "real", "short")}
        joint[t]["all7_below"] = {"count": all7[t], "B": B, "fraction": all7[t] / B}
        print(f"    {lab:6s} shortfall {pt[t]['short']:+.2f} [{joint[t]['short']['ci'][0]:+.2f}, {joint[t]['short']['ci'][1]:+.2f}]"
              f"  syn {pt[t]['syn']:+.2f} [{joint[t]['syn']['ci'][0]:+.2f}, {joint[t]['syn']['ci'][1]:+.2f}]"
              f"  real {pt[t]['real']:+.2f} [{joint[t]['real']['ci'][0]:+.2f}, {joint[t]['real']['ci'][1]:+.2f}]"
              f"  all 7 below: {frac(all7[t], B)}")
    # ---- 6. pathway split (gradient contribution = PDA@1 - lr 0), paired ----------------------------------------
    split = {}
    print("[6] pathway split: statistics (lr 0) | statistics + gradient (PDA@1) | gradient contribution, paired 95% CI")
    for k, lab in (("syn", "synthetic mean benefit"), ("real", "real mean benefit (7)"), ("short", "real shortfall")):
        dd = np.asarray(dr[FULL][k]) - np.asarray(dr[LR0][k])
        split[k] = {"stats_only": float(pt[LR0][k]), "stats_plus_gradient": float(pt[FULL][k]),
                    "gradient_contribution": float(pt[FULL][k] - pt[LR0][k]), "ci": ci(dd),
                    "share_of_full_from_stats": float(pt[LR0][k] / pt[FULL][k]) if pt[FULL][k] else None}
        print(f"    {lab:24s} {pt[LR0][k]:+6.2f} | {pt[FULL][k]:+6.2f} | {pt[FULL][k] - pt[LR0][k]:+6.2f} "
              f"[{split[k]['ci'][0]:+.2f}, {split[k]['ci'][1]:+.2f}]")
    dsh = np.asarray(dr[LR0]["short"]) - np.asarray(dr[FULL]["short"])
    joint["Delta_short_lr0_minus_full"] = {"point": float(pt[LR0]["short"] - pt[FULL]["short"]), "ci": ci(dsh)}
    print(f"    Delta_short (lr 0 - PDA@1) {pt[LR0]['short'] - pt[FULL]['short']:+.2f} "
          f"[{joint['Delta_short_lr0_minus_full']['ci'][0]:+.2f}, {joint['Delta_short_lr0_minus_full']['ci'][1]:+.2f}]")
    out["joint_bootstrap_B2000"], out["pathway_split"] = joint, split

    # ---- 4. supplementary image-level bootstrap (B=10^4, bootstrap_ci.py scheme) -------------------------------
    BI = 10000
    rng = np.random.default_rng(0)
    fit = {t: (r["intercept"], r["slope"]) for t, r in ((LR0, r0), (FULL, r1))}
    res = {t: np.zeros((BI, len(sa.ADVERSE))) for t in (FULL, LR0)}
    per_cond = {}
    for j, c in enumerate(sa.ADVERSE):
        n = len(van[c][0])
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=BI).astype(np.float64)
        bv = miou_from_cm((W @ van[c][0].reshape(n, -1)).reshape(BI, 19, 19))
        for t in (FULL, LR0):
            ba = miou_from_cm((W @ ada[t][c][0].reshape(n, -1)).reshape(BI, 19, 19))
            a0, b = fit[t]
            res[t][:, j] = (ba - bv) - (a0 + b * bv)
        per_cond[c] = {"resid": r0["per_condition_resid"][c], "ci": ci(res[LR0][:, j])}
    img = {}
    for t in (FULL, LR0):
        m = res[t].mean(1); k = int((res[t] < 0).all(1).sum())
        img[t] = {"point": float((r0 if t == LR0 else r1)["mean_shortfall"]), "ci": ci(m),
                  "all7_below": {"count": k, "B": BI, "fraction": k / BI}}
    dimg = res[LR0].mean(1) - res[FULL].mean(1)
    img["Delta_short_lr0_minus_full"] = {"point": float(r0["mean_shortfall"] - r1["mean_shortfall"]), "ci": ci(dimg)}
    img["per_condition_lr0"] = per_cond
    out["image_bootstrap_B10000"] = img
    print(f"[4] supplementary paired image-level bootstrap, B={BI} (fits fixed)")
    for t, lab in ((FULL, "PDA@1"), (LR0, "lr 0")):
        print(f"    {lab:6s} shortfall {img[t]['point']:+.2f} [{img[t]['ci'][0]:+.2f}, {img[t]['ci'][1]:+.2f}]"
              f"  all 7 below: {frac(img[t]['all7_below']['count'], BI)}")
    print(f"    Delta_short (lr 0 - PDA@1) {img['Delta_short_lr0_minus_full']['point']:+.2f} "
          f"[{img['Delta_short_lr0_minus_full']['ci'][0]:+.2f}, {img['Delta_short_lr0_minus_full']['ci'][1]:+.2f}]")
    for c in sa.ADVERSE:
        print(f"      {c:11s} lr0 resid {per_cond[c]['resid']:+6.2f} [{per_cond[c]['ci'][0]:+6.2f}, {per_cond[c]['ci'][1]:+6.2f}]")

    # ---- 5. sky -> building, ACDC fog (and every real condition) -------------------------------------------------
    print("[5] GT-sky -> building (%) / sky IoU, summed per-image CMs, same pool (h200)")
    sky = {}
    for c in sa.ADVERSE + [sa.PROBE]:
        row = {}
        for t, M in (("van", van[c][0]), (FULL, ada[FULL][c][0]), (LR0, ada[LR0][c][0])):
            S = M.sum(0); tp = S[SKY, SKY]
            row[t] = {"sky_to_building_pct": float(100 * S[SKY, BLD] / S[SKY].sum()),
                      "sky_iou": float(100 * tp / (S[SKY].sum() + S[:, SKY].sum() - tp))}
        # per image: share of images (with GT sky) whose sky->building rate exceeds vanilla's
        V, L = van[c][0], ada[LR0][c][0]
        has = V[:, SKY].sum(1) > 0
        rv = V[has, SKY, BLD] / V[has, SKY].sum(1); rl = L[has, SKY, BLD] / L[has, SKY].sum(1)
        row["lr0_images_rate_above_vanilla"] = {"count": int((rl > rv).sum()), "n": int(has.sum())}
        sky[c] = row
        print(f"    {c:11s} van {row['van']['sky_to_building_pct']:5.1f}/{row['van']['sky_iou']:5.1f}  "
              f"lr0 {row[LR0]['sky_to_building_pct']:5.1f}/{row[LR0]['sky_iou']:5.1f}  "
              f"PDA@1 {row[FULL]['sky_to_building_pct']:5.1f}/{row[FULL]['sky_iou']:5.1f}  "
              f"lr0 images above van: {row['lr0_images_rate_above_vanilla']['count']}/{row['lr0_images_rate_above_vanilla']['n']}")
    print("    (published v1 ACDC fog: vanilla 11.9%, PDA@1 26.0%, DUA 28.7%)")
    out["sky_to_building"] = sky
    out["complete"] = bool(complete and not bad and not missing)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
