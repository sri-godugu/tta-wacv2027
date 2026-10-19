import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

# Synthetic fit, saturate sev4-5 excluded (per pre-registration)
pts = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    vb = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    for r in dp["severity_results"]:
        s=r["severity"]
        if t=="saturate" and s>=4: continue
        pts.append((vb[s], r["miou"]*100 - vb[s]))
xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
mx,my=statistics.mean(xs),statistics.mean(ys)
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
intercept=my-slope*mx
rstd=statistics.stdev([y-(slope*x+intercept) for x,y in zip(xs,ys)])
print(f"Synthetic fit (saturate sev4-5 excluded, n={len(pts)}): benefit={slope:.4f}*vanilla+{intercept:.4f}, residual std={rstd:.2f}")

# 4 ACDC conditions: vanilla from new runs, PDA-only from earlier JSD run
print("\n=== 4 real ACDC conditions placed against synthetic line ===")
n_below = 0
resids = []
for cond in ["fog","night","rain","snow"]:
    dv = json.load(open(f"acdc_vanilla_cond_{cond}.json"))
    dp = json.load(open(f"acdc_acdc_pdaonly_jsd_{cond}.json"))
    van = dv["metrics"]["miou"]*100
    pda = dp["metrics"]["miou"]*100
    benefit = pda - van
    pred = slope*van + intercept
    resid = benefit - pred
    resids.append(resid)
    below = resid < 0
    if below: n_below += 1
    print(f"  {cond:6s}: vanilla={van:.2f} pda={pda:.2f} benefit={benefit:+.2f} | predicted={pred:+.2f} residual={resid:+.2f} = {resid/rstd:+.2f} SD {'BELOW' if below else 'above'}")

print(f"\nDirectional: {n_below}/4 conditions BELOW the synthetic line")
print(f"Mean residual across 4 conditions: {statistics.mean(resids):+.2f} mIoU = {statistics.mean(resids)/rstd:+.2f} SD")
