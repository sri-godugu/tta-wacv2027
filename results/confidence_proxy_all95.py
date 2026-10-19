import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]
DROP_SATURATE_HIGH = False  # per pre-registration: saturate sev4-5 are distinct-mechanism outliers

def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx = (sum((x-mx)**2 for x in xs))**0.5
    sy = (sum((y-my)**2 for y in ys))**0.5
    return cov/(sx*sy)

def partial(a, b, c):
    # partial corr of a,b controlling for c
    rab, rac, rbc = pearson(a,b), pearson(a,c), pearson(b,c)
    return (rab - rac*rbc) / (((1-rac**2)**0.5)*((1-rbc**2)**0.5))

grid = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    dc = json.load(open(f"pertype_vanconf_{t}.json"))
    van_by_sev = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    conf_by_sev = {r["severity"]: r.get("mean_confidence") for r in dc["severity_results"]}
    for r in dp["severity_results"]:
        s = r["severity"]
        if DROP_SATURATE_HIGH and t=="saturate" and s>=4:
            continue
        grid.append({"type":t,"sev":s,"vanilla":van_by_sev[s],
                     "benefit":r["miou"]*100-van_by_sev[s],"conf":conf_by_sev[s]})

grid = [g for g in grid if g["conf"] is not None]
print(f"n = {len(grid)} (saturate sev4-5 excluded={DROP_SATURATE_HIGH})")

ben = [g["benefit"] for g in grid]
conf = [g["conf"] for g in grid]
van = [g["vanilla"] for g in grid]

print(f"\nraw Pearson(confidence, benefit): {pearson(conf,ben):.3f}")
print(f"raw Pearson(vanilla_mIoU, benefit): {pearson(van,ben):.3f}")
print(f"Pearson(confidence, vanilla_mIoU): {pearson(conf,van):.3f}  <- how well confidence proxies difficulty")
print(f"\nPARTIAL Pearson(confidence, benefit | vanilla_mIoU): {partial(conf,ben,van):.3f}")
print("  [pre-reg: |ρ|<0.3 => confidence is a label-free RESTATEMENT of headroom, not independent]")

# proxy-vs-oracle: predict benefit from confidence alone vs from mIoU alone
def r2_predict(x, y):
    return pearson(x,y)**2
print(f"\nProxy-vs-oracle:")
print(f"  R^2(benefit ~ vanilla_mIoU) [oracle]: {r2_predict(van,ben):.3f}")
print(f"  R^2(benefit ~ confidence)   [proxy]:  {r2_predict(conf,ben):.3f}")
print(f"  => confidence recovers {r2_predict(conf,ben)/r2_predict(van,ben)*100:.0f}% of oracle predictive power")
