import json, statistics

TYPES = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost",
         "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
         "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
         "snow","spatter","speckle_noise","zoom_blur"]

rows = []
for t in TYPES:
    dv = json.load(open(f"pertype_vanilla_{t}.json"))
    dp = json.load(open(f"pertype_pdaonly_{t}.json"))
    van_miou = sum(r["miou"] for r in dv["severity_results"])/5*100
    pda_miou = sum(r["miou"] for r in dp["severity_results"])/5*100
    jsd_vals = [r["mean_jsd"] for r in dp["severity_results"] if r.get("mean_jsd") is not None]
    jsd = sum(jsd_vals)/len(jsd_vals) if jsd_vals else None
    benefit = pda_miou - van_miou
    rows.append((t, van_miou, pda_miou, benefit, jsd))

rows.sort(key=lambda r: -r[3])
print(f"{'type':<20}{'vanilla':>9}{'pda':>9}{'benefit':>10}{'mean_jsd':>10}")
for t, v, p, b, j in rows:
    print(f"{t:<20}{v:9.2f}{p:9.2f}{b:10.2f}{j:10.4f}")

benefits = [r[3] for r in rows]
jsds = [r[4] for r in rows]
print()
print("mean benefit across 19 types:", round(statistics.mean(benefits),2))
print("mean vanilla:", round(statistics.mean([r[1] for r in rows]),2))
print("mean pda:", round(statistics.mean([r[2] for r in rows]),2))

def rankdata(vals):
    sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0]*len(vals)
    for rank, i in enumerate(sorted_idx):
        ranks[i] = rank+1
    return ranks

rb = rankdata(benefits)
rj = rankdata(jsds)
n = len(benefits)
d2 = sum((a-b)**2 for a,b in zip(rb,rj))
spearman = 1 - 6*d2/(n*(n**3-1))
print("Spearman rank correlation (benefit vs mean_jsd):", round(spearman,3))

mb, mj = statistics.mean(benefits), statistics.mean(jsds)
cov = sum((b-mb)*(j-mj) for b,j in zip(benefits,jsds))
sb = (sum((b-mb)**2 for b in benefits))**0.5
sj = (sum((j-mj)**2 for j in jsds))**0.5
pearson = cov/(sb*sj)
print("Pearson correlation:", round(pearson,3))

print()
print("Top 5 by benefit:", [r[0] for r in rows[:5]])
print("Bottom 5 by benefit:", [r[0] for r in rows[-5:]])
