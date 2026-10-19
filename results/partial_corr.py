import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

# Build the 95-point grid: (type, severity) -> vanilla_miou, benefit, jsd
grid = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    van_by_sev = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
    for r in dp["severity_results"]:
        sev = r["severity"]
        pda_miou = r["miou"]*100
        van_miou = van_by_sev[sev]
        jsd = r.get("mean_jsd")
        if jsd is None:
            continue
        grid.append({
            "type": t, "severity": sev,
            "vanilla": van_miou, "pda": pda_miou,
            "benefit": pda_miou - van_miou, "jsd": jsd,
        })

print(f"Grid size: {len(grid)} points (expected 95)")

def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx = (sum((x-mx)**2 for x in xs))**0.5
    sy = (sum((y-my)**2 for y in ys))**0.5
    return cov/(sx*sy)

benefit = [g["benefit"] for g in grid]
jsd = [g["jsd"] for g in grid]
vanilla = [g["vanilla"] for g in grid]

r_jb = pearson(jsd, benefit)
r_jv = pearson(jsd, vanilla)
r_bv = pearson(benefit, vanilla)

partial = (r_jb - r_jv*r_bv) / (((1-r_jv**2)**0.5) * ((1-r_bv**2)**0.5))

print(f"\nRaw Pearson(JSD, benefit) over 95 points: {r_jb:.3f}")
print(f"Pearson(JSD, vanilla_mIoU): {r_jv:.3f}")
print(f"Pearson(benefit, vanilla_mIoU): {r_bv:.3f}")
print(f"\nPARTIAL correlation (JSD, benefit | vanilla_mIoU): {partial:.3f}")

# Within-difficulty-bin analysis: bin by vanilla mIoU, width ~4
bin_width = 4
bins = {}
for g in grid:
    b = int(g["vanilla"] // bin_width)
    bins.setdefault(b, []).append(g)

print(f"\n--- Within-difficulty-bin JSD-benefit correlation (bin width={bin_width} mIoU) ---")
for b in sorted(bins):
    pts = bins[b]
    if len(pts) < 4:
        print(f"  bin vanilla~[{b*bin_width}-{(b+1)*bin_width}): n={len(pts)} (too few to correlate)")
        continue
    bj = [p["jsd"] for p in pts]
    bb = [p["benefit"] for p in pts]
    try:
        r = pearson(bj, bb)
    except ZeroDivisionError:
        r = float('nan')
    print(f"  bin vanilla~[{b*bin_width}-{(b+1)*bin_width}): n={len(pts)}  r(JSD,benefit)={r:.3f}  types={[p['type']+':'+str(p['severity']) for p in pts]}")
