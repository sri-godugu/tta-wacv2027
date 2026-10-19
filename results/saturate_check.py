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
        grid.append({"type": t, "sev": sev, "vanilla": van_by_sev[sev],
                     "pda": r["miou"]*100, "benefit": r["miou"]*100 - van_by_sev[sev]})

print("=== SATURATE per-severity (monotonicity check) ===")
for g in [x for x in grid if x["type"]=="saturate"]:
    print(f"  sev{g['sev']}: vanilla={g['vanilla']:.2f}  pda={g['pda']:.2f}  benefit={g['benefit']:+.2f}")

def fit(points):
    xs=[p["vanilla"] for p in points]; ys=[p["benefit"] for p in points]
    mx,my=statistics.mean(xs),statistics.mean(ys)
    sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    slope=sxy/sxx; intercept=my-slope*mx
    resid=[p["benefit"]-(slope*p["vanilla"]+intercept) for p in points]
    return slope, intercept, statistics.stdev(resid)

ACDC_VAN, ACDC_BEN = 41.39, 0.11
def acdc_sd(points, label):
    slope,intercept,rstd = fit(points)
    pred = slope*ACDC_VAN+intercept
    resid = ACDC_BEN-pred
    print(f"\n{label} (n={len(points)}):")
    print(f"  fit: benefit={slope:.4f}*vanilla+{intercept:.4f}, residual std={rstd:.2f}")
    print(f"  ACDC predicted benefit={pred:.2f}, actual={ACDC_BEN}, residual={resid:.2f} = {resid/rstd:.2f} SD")

acdc_sd(grid, "(a) ALL 95 points")
acdc_sd([g for g in grid if not (g["type"]=="saturate" and g["sev"]>=4)], "(b) DROP saturate sev4-5")
acdc_sd([g for g in grid if g["type"]!="saturate"], "(c) DROP all saturate")

# Also: what are the top-leverage positive residuals overall?
slope,intercept,rstd = fit(grid)
for g in grid: g["resid"]=g["benefit"]-(slope*g["vanilla"]+intercept)
print("\n=== Top 6 positive residuals (all 95) ===")
for g in sorted(grid,key=lambda g:-g["resid"])[:6]:
    print(f"  {g['type']} sev{g['sev']}: vanilla={g['vanilla']:.1f} benefit={g['benefit']:+.2f} resid={g['resid']:+.2f}")
