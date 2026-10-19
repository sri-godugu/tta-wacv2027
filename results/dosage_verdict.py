import json
def mean_miou(fn):
    d = json.load(open(fn))
    sr = d["severity_results"]
    return {r["severity"]: r["miou"]*100 for r in sr}, sum(r["miou"] for r in sr)/len(sr)*100

print(f"{'config':<28}{'S1':>7}{'S2':>7}{'S3':>7}{'S4':>7}{'S5':>7}{'mean':>8}")
for label, fn in [
    ("lambda=1.0 (dosage)", "cityscapes_c_dosage_lambda1p0.json"),
    ("lambda=3.0 (dosage)", "cityscapes_c_dosage_lambda3p0.json"),
    ("lambda=0.3 (fixedthresh s0)", "cityscapes_c_dl_ft_s0_ct.json"),
]:
    sv, m = mean_miou(fn)
    print(f"{label:<28}" + "".join(f"{sv[s]:7.2f}" for s in range(1,6)) + f"{m:8.2f}")
print(f"\nReference: PDA-only (no TCPR) mean = 51.75")
print(f"Reference: lambda=0.3 3-seed mean = 51.67 +/- 0.47")
