#!/usr/bin/env python3
"""Checksum the transcribed primary PDA grid against the published fit.
Expected (backbone_spine.json, DeepLabV3+/R50 primary):
  slope b = -0.19192926461392887, intercept a = 13.298413239434046,
  R2 = 0.44420369814376626, resid std = 4.028407208402297,
  mean benefit = 4.05026188022212, n = 95.
"""
import json, sys
import numpy as np

d = json.load(open("./data/grid_primary_pda.json"))
types = sorted(d["vanilla"].keys())
X, Y, labels = [], [], []
for t in types:
    v = np.asarray(d["vanilla"][t], dtype=np.float64) * 100.0
    p = np.asarray(d["pda"][t], dtype=np.float64) * 100.0
    assert v.shape == (5,) and p.shape == (5,), t
    X.append(v); Y.append(p - v)
    labels += [(t, s + 1) for s in range(5)]
X = np.concatenate(X); Y = np.concatenate(Y)
n = X.size
b, a = np.polyfit(X, Y, 1)
pred = a + b * X
resid = Y - pred
ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((Y - Y.mean())**2))
r2 = 1.0 - ss_res / ss_tot
std = float(np.std(resid, ddof=0))
mb = float(Y.mean())

exp = dict(a=13.298413239434046, b=-0.19192926461392887,
           r2=0.44420369814376626, std=4.028407208402297,
           mean_benefit=4.05026188022212)
got = dict(a=a, b=b, r2=r2, std=std, mean_benefit=mb)
print(f"n={n}")
ok = n == 95
for k in exp:
    # std might be ddof=1 in their script; check both
    pass
tol = dict(a=5e-3, b=5e-5, r2=5e-4, std=5e-3, mean_benefit=5e-3)
std1 = float(np.std(resid, ddof=1))
for k in exp:
    gv = got[k]
    d_ = abs(gv - exp[k])
    alt = ""
    if k == "std" and d_ > tol[k]:
        d1 = abs(std1 - exp[k])
        if d1 < d_:
            gv, d_ = std1, d1; alt = " (ddof=1)"
    flag = "OK " if d_ <= tol[k] else "FAIL"
    print(f"{flag} {k:13s} got {gv:+.6f}{alt}  exp {exp[k]:+.6f}  |d|={d_:.2e}")
    ok &= d_ <= tol[k]
mx = int(np.argmax(Y))
print(f"max benefit {Y[mx]:+.2f} at {labels[mx]} (expect ~+31.2 saturate s5)")
mn = int(np.argmin(Y))
print(f"min benefit {Y[mn]:+.2f} at {labels[mn]}")
print("CHECKSUM:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
