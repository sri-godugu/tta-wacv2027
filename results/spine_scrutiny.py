import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

# full 95-point grid
grid = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    vb = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    for r in dp["severity_results"]:
        s=r["severity"]
        grid.append({"type":t,"sev":s,"vanilla":vb[s],"benefit":r["miou"]*100-vb[s]})

def fit(points):
    xs=[p["vanilla"] for p in points]; ys=[p["benefit"] for p in points]
    mx,my=statistics.mean(xs),statistics.mean(ys)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    intercept=my-slope*mx
    rstd=statistics.stdev([p["benefit"]-(slope*p["vanilla"]+intercept) for p in points])
    return slope,intercept,rstd

# ACDC 4 conditions
acdc=[]
for cond in ["fog","night","rain","snow"]:
    v=json.load(open(f"acdc_vanilla_cond_{cond}.json"))["metrics"]["miou"]*100
    p=json.load(open(f"acdc_acdc_pdaonly_jsd_{cond}.json"))["metrics"]["miou"]*100
    acdc.append((cond,v,p-v))

fits = {
  "(a) ALL 95 pts [nothing excluded, CONSERVATIVE]": grid,
  "(b) drop saturate sev4-5 [BN-recalib mechanism, TENT-confirmed]": [g for g in grid if not (g["type"]=="saturate" and g["sev"]>=4)],
  "(c) drop all saturate": [g for g in grid if g["type"]!="saturate"],
}
for label, pts in fits.items():
    slope,intercept,rstd = fit(pts)
    print(f"\n=== {label} (n={len(pts)}) ===")
    print(f"    fit: benefit={slope:.4f}*vanilla+{intercept:.4f}, residual std={rstd:.2f}")
    sds=[]
    for cond,v,b in acdc:
        pred=slope*v+intercept; resid=b-pred; sd=resid/rstd; sds.append(sd)
        print(f"    {cond:6s}: vanilla={v:.1f} benefit={b:+.2f} pred={pred:+.2f} resid={resid:+.2f} = {sd:+.2f} SD {'BELOW' if resid<0 else 'above'}")
    nbelow=sum(1 for s in sds if s<0)
    print(f"    -> {nbelow}/4 below; mean = {statistics.mean(sds):+.2f} SD")
