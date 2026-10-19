import json, sys, os
D = sys.argv[1]
C = ["road","sidewalk","building","wall","fence","pole","t.light","t.sign","veg","terrain",
     "sky","person","rider","car","truck","bus","train","motorcyc","bicycle"]
def L(f): return json.load(open(os.path.join(D, f + ".json")))
def real(f): m = L(f)["metrics"]; return m["miou"]*100, [x*100 for x in m["per_class_iou"]]
def syn(f, sev):
    for r in L(f)["severity_results"]:
        if r["severity"] == sev: return r["miou"]*100, [x*100 for x in r["per_class_iou"]]
print("keys of severity_results:", list(L("s10_pdaonly_fog")["severity_results"][0].keys()))
# synthetic fog severity closest in vanilla mIoU to ACDC fog (58.3)
vs = {r["severity"]: r["miou"]*100 for r in L("pertype_vanilla_fog")["severity_results"]}
print("synthetic fog vanilla by severity:", {k: round(v,1) for k,v in vs.items()})
sev = min(vs, key=lambda s: abs(vs[s]-58.26))
print("matched synthetic fog severity:", sev)
rows = {
 "R van": real("cm_acdc_fog_vanilla"), "R pda1": real("cm_acdc_fog_pdaonly"),
 "R pda10": real("s10_pdaonly_acdc_fog"), "R tent1": real("cm_acdc_fog_tent"),
 "R tent10": real("s10_tent_acdc_fog"), "R ora10": real("oracle_s10_acdc_fog"),
 "S van": syn("pertype_vanilla_fog", sev), "S pda1": syn("pertype_pdaonly_fog", sev),
 "S pda10": syn("s10_pdaonly_fog", sev), "S tent1": syn("pertype_tent_fog", sev),
 "S tent10": syn("s10_tent_fog", sev), "S ora10": syn("oracle_s10_fog", sev)}
print("\nmIoU: " + "  ".join(f"{k}={v[0]:.1f}" for k,v in rows.items()))
def d(a,b): return [x-y for x,y in zip(rows[a][1], rows[b][1])]
cols = [("R pda1","R van"),("R pda10","R van"),("R tent10","R van"),("R ora10","R van"),
        ("S pda1","S van"),("S pda10","S van"),("S tent10","S van"),("S ora10","S van")]
print("\nper-class IoU delta vs own vanilla (R = ACDC fog, S = synthetic fog sev%d)" % sev)
print(f"{'class':9s} {'Rvan':>5s} " + " ".join(f"{a.replace(' ','')[:7]:>7s}" for a,_ in cols))
for i,c in enumerate(C):
    print(f"{c:9s} {rows['R van'][1][i]:5.1f} " + " ".join(f"{d(a,b)[i]:+7.1f}" for a,b in cols))
