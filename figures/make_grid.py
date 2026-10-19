#!/usr/bin/env python3
"""
make_grid.py — build grid_{primary,r101,fcn50}_pda.json from the released
per-condition outputs. Run from inside results/ (or pass --results PATH).
Each grid is checksummed against the published fit before it is written;
a failing checksum aborts the write (no silent wrong figures).

  cd results/ && python make_grid.py --out /path/to/figkit/data
"""
import argparse, json, os, sys
import numpy as np

TYPES = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
         "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
         "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
         "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
         "zoom_blur"]

BACKBONES = {
    "primary": dict(prefix="",        label="DeepLabV3+/ResNet-50",
                    exp=dict(a=13.298413239434046, b=-0.19192926461392887,
                             r2=0.44420369814376626, std=4.028407208402297)),
    "r101":    dict(prefix="r101_",   label="DeepLabV3+/ResNet-101",
                    exp=dict(a=18.45050516380383, b=-0.2727172199575733,
                             r2=0.6894349361654708, std=3.4059903472734154)),
    "fcn50":   dict(prefix="fcn50_",  label="FCN/ResNet-50",
                    exp=dict(a=13.063301561388368, b=-0.19337825198111636,
                             r2=0.41064912584015123, std=4.342145242259533)),
}

def sevs(path):
    d = json.load(open(path))
    return {r["severity"]: float(r["miou"]) for r in d["severity_results"]}

def build(results, key, spec, out_dir):
    van, pda = {}, {}
    for t in TYPES:
        pv = os.path.join(results, f"{spec['prefix']}pertype_vanilla_{t}.json")
        pp = os.path.join(results, f"{spec['prefix']}pertype_pdaonly_{t}.json")
        if not (os.path.exists(pv) and os.path.exists(pp)):
            print(f"[{key}] missing {t} — skipping this backbone")
            return False
        v, p = sevs(pv), sevs(pp)
        common = sorted(set(v) & set(p))
        van[t] = [v[s] for s in common]
        pda[t] = [p[s] for s in common]
    V = np.concatenate([np.asarray(van[t]) * 100 for t in TYPES])
    B = np.concatenate([(np.asarray(pda[t]) - np.asarray(van[t])) * 100 for t in TYPES])
    b, a = np.polyfit(V, B, 1)
    r = B - (a + b * V)
    r2 = 1 - (r @ r) / float(((B - B.mean()) ** 2).sum())
    std = float(np.std(r, ddof=1))
    e = spec["exp"]
    ok = (V.size == 95 and abs(a - e["a"]) < 5e-3 and abs(b - e["b"]) < 5e-5
          and abs(r2 - e["r2"]) < 5e-4 and abs(std - e["std"]) < 5e-3)
    print(f"[{key}] n={V.size} slope={b:+.4f} a={a:+.3f} R2={r2:.4f} "
          f"sd={std:.3f} -> {'CHECKSUM PASS' if ok else 'CHECKSUM FAIL'}")
    if not ok:
        return False
    out = os.path.join(out_dir, f"grid_{key}_pda.json")
    json.dump({"backbone": spec["label"], "method": "PDA",
               "vanilla": van, "pda": pda}, open(out, "w"), indent=1)
    print(f"[{key}] wrote {out}")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=".")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    good = [build(args.results, k, s, args.out) for k, s in BACKBONES.items()]
    sys.exit(0 if all(good) else 1)
