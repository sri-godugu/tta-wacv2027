import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

grid = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    vb = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    for r in dp["severity_results"]:
        s=r["severity"]
        grid.append({"vanilla":vb[s],"benefit":r["miou"]*100-vb[s]})

xs=[g["vanilla"] for g in grid]; ys=[g["benefit"] for g in grid]
mx,my=statistics.mean(xs),statistics.mean(ys)
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/((sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**0.5)
r = pear(xs,ys)
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
intercept=my-slope*mx
rstd=statistics.stdev([y-(slope*x+intercept) for x,y in zip(xs,ys)])
print("=== FIT (a) — PRIMARY, nothing excluded (n=95) — PIN THESE ===")
print(f"  benefit = {slope:.4f}*vanilla + {intercept:.4f}")
print(f"  R^2(benefit ~ vanilla_mIoU) = {r**2:.3f}   |   residual std = {rstd:.2f}")

print("\n=== Local synthetic density around each ACDC condition (is 'matched' real or extrapolated?) ===")
acdc = {"fog":58.25,"night":21.51,"rain":43.45,"snow":42.81}
xs_sorted = sorted(xs)
print(f"  synthetic vanilla-mIoU range: [{min(xs):.1f}, {max(xs):.1f}]")
for cond,v in acdc.items():
    within3 = sum(1 for x in xs if abs(x-v)<=3)
    within5 = sum(1 for x in xs if abs(x-v)<=5)
    below = sum(1 for x in xs if x < v)
    edge = "EXTRAPOLATED-EDGE" if v < min(xs) or v > max(xs) else ("near-edge" if below<5 or (95-below)<5 else "interior")
    print(f"  {cond:6s} vanilla={v:.1f}: {within3} synth pts within +-3, {within5} within +-5 | {below} synth below, {95-below} above -> {edge}")
