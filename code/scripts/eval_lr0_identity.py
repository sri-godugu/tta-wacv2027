#!/usr/bin/env python3
"""
eval_lr0_identity.py -- runs scripts/evaluate.py unchanged, plus the bit-identity check registered for the
BN-statistics control (preregistration_step_curve.md, addendum 2026-10-19 05:20 EDT):

    "Before scoring, a sanity check confirms that the affine parameters after adaptation are bit-identical to the
     checkpoint's (otherwise lr 0 is not a stats-only control and the run is void)."

The check is done inside the scored pass itself. PerturbationDisagreementAdaptor.adapt is wrapped (observation
only, no numeric effect): before every adapt() call every model parameter is snapshotted, and after the step
(before reset()) each one is compared bitwise with torch.equal. It also records whether the BatchNorm running
statistics moved during adapt(), since that is the pathway the control isolates.

Usage: python scripts/eval_lr0_identity.py --identity_json OUT.json <evaluate.py arguments>
The identity record is written even if evaluation fails; "valid" is true only if every adapt() call left every
parameter bit-identical and finite.
"""
import json, os, runpy, sys

import torch

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from adaptation import pda as _pda  # same module object evaluation/evaluate.py imports

i = sys.argv.index("--identity_json")
OUT = sys.argv[i + 1]
del sys.argv[i:i + 2]

rec = {"adapt_calls": 0, "calls_all_params_identical": 0, "calls_affine_identical": 0,
       "affine_tensors_per_call": None, "params_changed_total": 0, "nonfinite_params_total": 0,
       "calls_running_stats_moved": 0, "max_abs_running_mean_delta": 0.0, "max_abs_running_var_delta": 0.0}
_orig_adapt = _pda.PerturbationDisagreementAdaptor.adapt


def _adapt(self, x):
    named = [(n, p) for n, p in self.model.named_parameters()]
    before = [p.detach().clone() for _, p in named]
    affine = set()
    for mn, m in self.model.named_modules():
        if isinstance(m, self._norm_classes):
            for pn in ("weight", "bias"):
                if getattr(m, pn, None) is not None:
                    affine.add(f"{mn}.{pn}" if mn else pn)
    bns = [m for m in self.model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.running_mean is not None]
    rm0 = [m.running_mean.clone() for m in bns]
    rv0 = [m.running_var.clone() for m in bns]

    out = _orig_adapt(self, x)

    with torch.no_grad():
        changed = [n for (n, p), b in zip(named, before) if not torch.equal(p.detach(), b)]
        nonfinite = sum(int(not torch.isfinite(p).all()) for _, p in named)
        dm = max(float((m.running_mean - a).abs().max()) for m, a in zip(bns, rm0)) if bns else 0.0
        dv = max(float((m.running_var - a).abs().max()) for m, a in zip(bns, rv0)) if bns else 0.0
    rec["adapt_calls"] += 1
    rec["affine_tensors_per_call"] = len(affine)
    rec["calls_all_params_identical"] += int(not changed)
    rec["calls_affine_identical"] += int(not (set(changed) & affine))
    rec["params_changed_total"] += len(changed)
    rec["nonfinite_params_total"] += nonfinite
    rec["calls_running_stats_moved"] += int(dm > 0 or dv > 0)
    rec["max_abs_running_mean_delta"] = max(rec["max_abs_running_mean_delta"], dm)
    rec["max_abs_running_var_delta"] = max(rec["max_abs_running_var_delta"], dv)
    if changed and "first_changed" not in rec:
        rec["first_changed"] = changed[:10]
    return out


_pda.PerturbationDisagreementAdaptor.adapt = _adapt
status = "crashed"
try:
    runpy.run_path(os.path.join(CODE, "scripts", "evaluate.py"), run_name="__main__")
    status = "ok"
finally:
    n = rec["adapt_calls"]
    rec["eval_status"] = status
    rec["valid"] = bool(status == "ok" and n > 0 and rec["calls_all_params_identical"] == n
                        and rec["nonfinite_params_total"] == 0)
    rec["stats_pathway_active"] = bool(n > 0 and rec["calls_running_stats_moved"] == n)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"IDENTITY valid={rec['valid']} calls={n} identical={rec['calls_all_params_identical']} "
          f"stats_moved={rec['calls_running_stats_moved']} max_drm={rec['max_abs_running_mean_delta']:.3g}")
