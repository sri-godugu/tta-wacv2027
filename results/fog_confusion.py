import sys, os, numpy as np
D = sys.argv[1]
C = ["road","sidewalk","building","wall","fence","pole","t.light","t.sign","veg","terrain",
     "sky","person","rider","car","truck","bus","train","motorcyc","bicycle"]
def load(m):
    z = np.load(os.path.join(D, f"acdc_fog_{m}.npz"))
    print(m, {k: z[k].shape for k in z.files})
    k = [k for k in z.files if z[k].ndim == 3][0]
    return z[k].astype(np.float64)          # (N, 19, 19) rows = GT, cols = pred (assumed)
def iou(cm):
    tp = np.diag(cm); return tp / (cm.sum(0) + cm.sum(1) - tp)
V = load("vanilla"); P = load("pdaonly"); T = load("tent"); U = load("dua")
cv, cp = V.sum(0), P.sum(0)
print("row-sum (GT px) of sky:", cv[10].sum(), " col-sum (pred px) of sky vanilla/pda:", cv[:,10].sum(), cp[:,10].sum())
iv, ip = iou(cv), iou(cp)
print(f"mIoU check van {np.nanmean(iv)*100:.2f}  pda {np.nanmean(ip)*100:.2f}")
d = (ip - iv) * 100 / 19
order = np.argsort(d)
print("contribution of each class to the mIoU change (pda1 - van), most negative first:")
for i in order[:6]: print(f"  {C[i]:9s} {d[i]:+.2f}")
print(f"  sum of all = {d.sum():+.2f}; sky+building = {d[10]+d[2]:+.2f}")
def where(cm, gt):
    r = cm[gt] / cm[gt].sum(); o = np.argsort(-r)[:4]
    return ", ".join(f"{C[j]} {100*r[j]:.1f}%" for j in o)
for name, cm in [("vanilla", cv), ("PDA@1", cp), ("TENT@1", T.sum(0)), ("DUA", U.sum(0))]:
    print(f"\n[{name}] GT sky ->", where(cm, 10))
    print(f"[{name}] GT building ->", where(cm, 2))
    # what is predicted as sky
    col = cm[:,10] / cm[:,10].sum(); o = np.argsort(-col)[:3]
    print(f"[{name}] pred sky comes from GT:", ", ".join(f"{C[j]} {100*col[j]:.1f}%" for j in o))
# per-image: fraction of images where sky IoU drops
def per_img_iou(A, c):
    tp = A[:, c, c]; den = A[:, c, :].sum(1) + A[:, :, c].sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)
sv, sp = per_img_iou(V, 10), per_img_iou(P, 10)
ok = ~np.isnan(sv) & ~np.isnan(sp) & (V[:,10,:].sum(1) > 0)
dd = sp[ok] - sv[ok]
print(f"\nper-image sky IoU change (images with sky GT, n={ok.sum()}): median {np.median(dd)*100:+.1f}, "
      f"drops in {np.mean(dd<0)*100:.0f}%, drops >10pt in {np.mean(dd<-0.10)*100:.0f}%")
