#!/usr/bin/env python3
"""
prereg_p2_p4_p5.py -- pre-registered predictions P1, P2, P4, P5 (results/preregistration_step_curve.md),
evaluated per method on whatever settings are complete (n_synth == 95 and 7 real). Builds on stepcurve_analysis.

P1  shortfall < 0 at every s in {1,2,3,5,10}, >= 6/7 below at each s.
P2  synthetic benefit(s=5) >= synthetic benefit(s=1) - 0.5, AND real benefit(s=5) < real benefit(s=1).
P4a deployable stopping point: s* = argmax mean SYNTHETIC benefit over s (lr 1e-3); real shortfall at s* < 0.
P4b label-selected upper bound, applied symmetrically: every synthetic condition AND every real condition takes its
    own best s (by its own benefit, i.e. using labels); refit the trend on the synthetic best-s benefits; mean real
    shortfall < 0. Uses labels; never presented as deployable (same best-iterate selection we criticise in DUA).
P5  shortfall < 0 at lr 1e-4 and 3e-4 (s = 1).

Usage: python prereg_p2_p4_p5.py <stepcurve_dir> [--methods pda tent ora] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stepcurve_analysis as sa

STEPS = [1, 2, 3, 5, 10]


def complete(r):
    return r.get("n_synth") == 95 and r.get("n_real") == 7


def per_condition(root, method, steps, lr):
    """synthetic {(type,sev): (van, benefit)} and real {cond: (van, benefit)} for one setting."""
    d = os.path.join(root, sa.POOL_OF[method]); tag = f"{method}_s{steps}_lr{lr}"
    syn = {}
    for t in sa.TYPES:
        v, a = sa.synth(d, "van", t), sa.synth(d, tag, t)
        for s in set(v) & set(a):
            syn[(t, s)] = (v[s], a[s] - v[s])
    real = {}
    for c in sa.ADVERSE:
        v, a = sa.real(d, "van", c), sa.real(d, tag, c)
        if v is not None and a is not None:
            real[c] = (v, a - v)
    return syn, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--methods", nargs="+", default=["pda", "tent", "ora"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for m in a.methods:
        rows = {s: sa.analyse(a.root, m, s, "0.001") for s in STEPS}
        lrs = {lr: sa.analyse(a.root, m, 1, lr) for lr in ["0.0003", "0.0001"]}
        done = [s for s in STEPS if complete(rows[s])]
        print(f"\n===== {m.upper()} ({sa.POOL_OF[m]}) — complete step settings: {done}; "
              f"LR: {[lr for lr in lrs if complete(lrs[lr])]}")
        res = {}
        # P1
        if done:
            p1 = all(rows[s]["mean_shortfall"] < 0 and rows[s]["n_below"] >= 6 for s in done)
            print("P1 " + ("SUPPORTED" if p1 else "NOT SUPPORTED") + f" on s={done}: " + ", ".join(
                f"s{s} {rows[s]['mean_shortfall']:+.2f} ({rows[s]['n_below']}/7)" for s in done)
                + ("" if len(done) == 5 else "  [partial]"))
            res["P1"] = {"supported": p1, "steps": done}
        # P2
        if complete(rows[1]) and complete(rows[5]):
            sy1, sy5 = rows[1]["mean_synth_benefit"], rows[5]["mean_synth_benefit"]
            re1, re5 = rows[1]["mean_real_benefit"], rows[5]["mean_real_benefit"]
            p2 = (sy5 >= sy1 - 0.5) and (re5 < re1)
            print(f"P2 {'SUPPORTED' if p2 else 'NOT SUPPORTED'}: synthetic s1 {sy1:+.2f} -> s5 {sy5:+.2f} "
                  f"(need >= {sy1 - 0.5:+.2f}); real s1 {re1:+.2f} -> s5 {re5:+.2f} (need lower)")
            res["P2"] = {"supported": p2, "syn1": sy1, "syn5": sy5, "real1": re1, "real5": re5}
        # P4a
        if len(done) == 5:
            sstar = max(done, key=lambda s: rows[s]["mean_synth_benefit"])
            sf = rows[sstar]["mean_shortfall"]
            print(f"P4a {'SUPPORTED' if sf < 0 else 'NOT SUPPORTED'}: synthetic-selected s* = {sstar} "
                  f"(mean synthetic benefit {rows[sstar]['mean_synth_benefit']:+.2f}); real shortfall at s* {sf:+.2f} "
                  f"({rows[sstar]['n_below']}/7 below); real benefit at s* {rows[sstar]['mean_real_benefit']:+.2f}")
            res["P4a"] = {"supported": sf < 0, "s_star": sstar, "shortfall": sf}
            # P4b symmetric label-selected best step per condition
            per = {s: per_condition(a.root, m, s, "0.001") for s in done}
            keys = set.intersection(*[set(per[s][0]) for s in done])
            X = np.array([per[1][0][k][0] for k in keys])
            Y = np.array([max(per[s][0][k][1] for s in done) for k in keys])
            b, a0 = np.polyfit(X, Y, 1)
            resid = {}
            for c in sa.ADVERSE:
                van = per[1][1][c][0]
                best = max(per[s][1][c][1] for s in done)
                resid[c] = best - (a0 + b * van)
            ms = float(np.mean(list(resid.values())))
            nb = int(sum(v < 0 for v in resid.values()))
            print(f"P4b {'SUPPORTED' if ms < 0 else 'NOT SUPPORTED'} [LABEL-SELECTED UPPER BOUND, not deployable]: "
                  f"best-s synthetic fit slope {b:+.3f}, mean best-s synthetic benefit {Y.mean():+.2f}; "
                  f"mean real shortfall {ms:+.2f} ({nb}/7 below)")
            print("     real best-s benefit per condition: " + ", ".join(
                f"{c} {max(per[s][1][c][1] for s in done):+.2f}@s{max(done, key=lambda s: per[s][1][c][1])}" for c in sa.ADVERSE))
            res["P4b"] = {"supported": ms < 0, "shortfall": ms, "n_below": nb, "slope": float(b)}
        # P5
        lr_done = [lr for lr in lrs if complete(lrs[lr])]
        if lr_done:
            p5 = all(lrs[lr]["mean_shortfall"] < 0 for lr in lr_done)
            print(f"P5 {'SUPPORTED' if p5 else 'NOT SUPPORTED'} on lr {lr_done}: " + ", ".join(
                f"lr {lr}: shortfall {lrs[lr]['mean_shortfall']:+.2f} ({lrs[lr]['n_below']}/7), "
                f"synthetic {lrs[lr]['mean_synth_benefit']:+.2f}, real {lrs[lr]['mean_real_benefit']:+.2f}" for lr in lr_done)
                + ("" if len(lr_done) == 2 else "  [partial]"))
            res["P5"] = {"supported": p5, "lrs": lr_done}
        out[m] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
