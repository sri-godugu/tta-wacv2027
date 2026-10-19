import json
names=["road","sidewalk","building","wall","fence","pole","tlight","tsign","veg","terrain","sky","person","rider","car","truck","bus","train","moto","bike"]
dv=json.load(open("pertype_vanilla_saturate.json"))
dp=json.load(open("pertype_pdaonly_saturate.json"))
for tag,d in [("VANILLA",dv),("PDA",dp)]:
    r=[x for x in d["severity_results"] if x["severity"]==5][0]
    miou=r["miou"]*100; pa=r["pixel_accuracy"]*100; ma=r["mean_accuracy"]*100
    print(f"--- saturate sev5 {tag}: mIoU={miou:.2f} pixAcc={pa:.2f} meanAcc={ma:.2f}")
    pc=r["per_class_iou"]
    zeros=sum(1 for v in pc if v<0.01)
    print(f"    classes with ~0 IoU: {zeros}/19")
    print("    per-class:", " ".join(f"{n}:{v*100:.0f}" for n,v in zip(names,pc)))
