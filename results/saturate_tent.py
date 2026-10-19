import json
dp = json.load(open("pertype_pdaonly_saturate.json"))
dt = json.load(open("pertype_tent_saturate.json"))
dv = json.load(open("pertype_vanilla_saturate.json"))
vb = {r["severity"]: r["miou"]*100 for r in dv["severity_results"]}
pb = {r["severity"]: r["miou"]*100 for r in dp["severity_results"]}
tb = {r["severity"]: r["miou"]*100 for r in dt["severity_results"]}
print("saturate: does TENT recover sev4-5 like PDA? (confirms BN-recalibration mechanism)")
print(f"{'sev':>4}{'vanilla':>9}{'PDA':>9}{'TENT':>9}{'PDA-ben':>9}{'TENT-ben':>9}")
for s in range(1,6):
    print(f"{s:>4}{vb[s]:9.2f}{pb[s]:9.2f}{tb[s]:9.2f}{pb[s]-vb[s]:+9.2f}{tb[s]-vb[s]:+9.2f}")
