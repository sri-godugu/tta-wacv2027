import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

NOISE = {"gaussian_noise","shot_noise","impulse_noise","speckle_noise"}
BLUR  = {"defocus_blur","gaussian_blur","motion_blur","glass_blur","zoom_blur"}

grid = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    van_by_sev = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    for r in dp["severity_results"]:
        sev = r["severity"]
        pda_miou = r["miou"]*100
        van_miou = van_by_sev[sev]
        grid.append({"type": t, "sev": sev, "vanilla": van_miou, "benefit": pda_miou-van_miou})

xs = [g["vanilla"] for g in grid]
ys = [g["benefit"] for g in grid]
n = len(xs)
mx, my = statistics.mean(xs), statistics.mean(ys)
sxx = sum((x-mx)**2 for x in xs)
sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
slope = sxy/sxx
intercept = my - slope*mx

for g in grid:
    g["predicted"] = slope*g["vanilla"] + intercept
    g["residual"] = g["benefit"] - g["predicted"]

resid_std = statistics.stdev([g["residual"] for g in grid])
print(f"Overall fit: benefit = {slope:.4f}*vanilla + {intercept:.4f}, residual std={resid_std:.2f}")

def family_stats(name, members):
    pts = [g for g in grid if g["type"] in members]
    resids = [g["residual"] for g in pts]
    print(f"\n{name} (n={len(pts)}): mean residual = {statistics.mean(resids):+.2f} mIoU "
          f"({statistics.mean(resids)/resid_std:+.2f} SD), "
          f"std={statistics.stdev(resids):.2f}")
    for g in sorted(pts, key=lambda g: -g['residual'])[:3]:
        print(f"    top: {g['type']} sev{g['sev']}: residual={g['residual']:+.2f}")
    return resids

noise_resid = family_stats("NOISE family", NOISE)
blur_resid  = family_stats("BLUR family", BLUR)
other_resid = family_stats("OTHER (remaining 10 types)", set(TYPES)-NOISE-BLUR)

# two-sample t-test-ish comparison (Welch, manual)
def welch_t(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = (va/na + vb/nb)**0.5
    return (ma-mb)/se

t_noise_blur = welch_t(noise_resid, blur_resid)
print(f"\nWelch t-stat, noise residual vs blur residual: {t_noise_blur:.2f}")
