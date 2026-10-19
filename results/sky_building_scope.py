"""Sky->building scope check (prereg addendum): per real condition, the share of GT-sky pixels
predicted as building, vanilla vs each 1-step adaptor, plus sky IoU and per-image drop rate."""
import sys, os, numpy as np
D = sys.argv[1]
SKY, BLD = 10, 2
CONDS = ["acdc_fog", "acdc_night", "acdc_rain", "acdc_snow", "bdd_rainy", "bdd_snowy", "fz_fog", "bdd_clear"]
def L(c, m): return np.load(os.path.join(D, f"{c}_{m}.npz"))["cm"].astype(np.float64)
def iou(cm, k): tp = cm[k, k]; return tp / (cm[k].sum() + cm[:, k].sum() - tp)
def per_img_iou(A, k):
    tp = A[:, k, k]; den = A[:, k, :].sum(1) + A[:, :, k].sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)
print(f"{'condition':11s} {'sky%px':>6s} | {'sky->bld %':>34s} | {'sky IoU':>27s} | PDA img drop")
print(f"{'':11s} {'':>6s} | {'van':>6s} {'PDA':>6s} {'TENT':>6s} {'DUA':>6s} {'dPDA':>6s} | {'van':>5s} {'PDA':>6s} {'TENT':>6s} {'DUA':>6s} |")
n_pattern = 0
for c in CONDS:
    M = {m: L(c, m) for m in ["vanilla", "pdaonly", "tent", "dua"]}
    S = {m: A.sum(0) for m, A in M.items()}
    gt_sky = S["vanilla"][SKY].sum(); tot = S["vanilla"].sum()
    rate = {m: 100 * S[m][SKY, BLD] / S[m][SKY].sum() for m in S}
    io = {m: 100 * iou(S[m], SKY) for m in S}
    has = M["vanilla"][:, SKY, :].sum(1) > 0
    d = per_img_iou(M["pdaonly"], SKY)[has] - per_img_iou(M["vanilla"], SKY)[has]
    dp = rate["pdaonly"] - rate["vanilla"]
    if c != "bdd_clear" and dp >= 5: n_pattern += 1
    print(f"{c:11s} {100*gt_sky/tot:6.1f} | {rate['vanilla']:6.1f} {rate['pdaonly']:6.1f} {rate['tent']:6.1f} {rate['dua']:6.1f} {dp:+6.1f} | "
          f"{io['vanilla']:5.1f} {io['pdaonly']-io['vanilla']:+6.1f} {io['tent']-io['vanilla']:+6.1f} {io['dua']-io['vanilla']:+6.1f} | "
          f"{np.mean(d < 0)*100:3.0f}% of {has.sum()} (median {np.median(d)*100:+.1f})")
print(f"\nadverse conditions with PDA sky->building increase >= 5 pts: {n_pattern}/7 "
      f"(prereg: >=3 -> 'pattern', else 'fog diagnostic case')")
