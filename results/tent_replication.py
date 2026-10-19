import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

def fit(points):
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    mx,my=statistics.mean(xs),statistics.mean(ys)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    intercept=my-slope*mx
    rstd=statistics.stdev([y-(slope*x+intercept) for x,y in zip(xs,ys)])
    return slope,intercept,rstd

# TENT synthetic difficulty fit (95 pts): TENT benefit vs vanilla mIoU
pts=[]
for t in TYPES:
    dv=json.load(open(f"pertype_vanilla_{t}.json"))
    dt=json.load(open(f"pertype_tent_{t}.json"))
    vb={r["severity"]:r["miou"]*100 for r in dv["severity_results"]}
    for r in dt["severity_results"]:
        s=r["severity"]; pts.append((vb[s], r["miou"]*100-vb[s]))
slope,intercept,rstd=fit(pts)
print(f"TENT synthetic fit (n={len(pts)}): benefit={slope:.4f}*vanilla+{intercept:.4f}, residual std={rstd:.2f}")

print("\n=== 4 ACDC conditions placed against TENT synthetic line ===")
nbelow=0; sds=[]
for c in ["fog","night","rain","snow"]:
    v=json.load(open(f"acdc_vanilla_cond_{c}.json"))["metrics"]["miou"]*100
    tt=json.load(open(f"acdc_tent_cond_{c}.json"))["metrics"]["miou"]*100
    b=tt-v; pred=slope*v+intercept; resid=b-pred; sd=resid/rstd; sds.append(sd)
    if resid<0: nbelow+=1
    print(f"  {c:6s}: vanilla={v:.1f} tent={tt:.1f} benefit={b:+.2f} pred={pred:+.2f} resid={resid:+.2f} = {sd:+.2f} SD {'BELOW' if resid<0 else 'above'}")
print(f"\n-> TENT: {nbelow}/4 below synthetic line; mean = {statistics.mean(sds):+.2f} SD")
print("VERDICT: if TENT also 4/4 below -> gap replicates across methods -> method-class claim -> bigger title")
