import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

grid = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    van_by_sev = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    for r in dp["severity_results"]:
        sev = r["severity"]
        pda_miou = r["miou"]*100
        van_miou = van_by_sev[sev]
        grid.append((van_miou, pda_miou - van_miou))

xs = [g[0] for g in grid]
ys = [g[1] for g in grid]
n = len(xs)
mx, my = statistics.mean(xs), statistics.mean(ys)
sxx = sum((x-mx)**2 for x in xs)
sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
slope = sxy/sxx
intercept = my - slope*mx
print(f"Linear fit (synthetic, n={n}): benefit = {slope:.4f} * vanilla_mIoU + {intercept:.4f}")

resid = [y - (slope*x+intercept) for x,y in zip(xs,ys)]
resid_std = statistics.stdev(resid)
print(f"Residual std (synthetic scatter around the difficulty trend): {resid_std:.2f} mIoU")

# ACDC actual point
acdc_vanilla = 41.39
acdc_benefit = 0.11
predicted = slope*acdc_vanilla + intercept
residual_acdc = acdc_benefit - predicted
print(f"\nAt ACDC's vanilla mIoU ({acdc_vanilla}), synthetic-trend predicted benefit: {predicted:.2f}")
print(f"ACDC's actual benefit: {acdc_benefit}")
print(f"ACDC residual (actual - predicted): {residual_acdc:.2f}  ({residual_acdc/resid_std:.2f} synthetic-residual std devs)")
