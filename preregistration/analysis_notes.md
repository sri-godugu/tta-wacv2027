# Analysis notes — step curve / sky→building mechanism (workshop revision)

Every entry has exactly one status tag:
- **REGISTERED**: tests a prediction written in `preregistration_step_curve.md` before the data existed.
- **OBSERVED**: a descriptive measurement reported as-is, with no claim of a prior prediction.
- **EXPLORATORY**: a post-hoc analysis chosen after seeing data. Supplement or rebuttal only, never main text, and never
  promoted to a prediction.

Entries are appended the moment they print and never edited afterwards. Corrections are added as new entries.

---

### 2026-10-19 ~04:10 — Luminance correlate, 1 step, ACDC fog, PDA — **REGISTERED** (direction fixed in prereg first)
Spearman ρ(|Y_sky − Y_building|, Δ sky IoU) = **+0.294, p = 0.0042**, n = 93. PASSED.
Supporting cells (same run, descriptive): snow PDA@1 +0.293 (p=0.0045), TENT@1 fog +0.420, snow +0.345; rain PDA@1 +0.110 (n.s.),
TENT@1 +0.128 (n.s.). Source: `results/sky_luminance_correlate.json`.

### 2026-10-19 ~04:10 — Luminance correlate at 10 steps — **EXPLORATORY**
Not pre-registered at 10 steps. ACDC fog PDA@10 ρ=+0.342 (p=0.0008), TENT@10 +0.274 (p=0.008); snow PDA@10 +0.137 (n.s.),
TENT@10 +0.242 (p=0.02); rain PDA@10 −0.011, TENT@10 −0.032 (both n.s.). The expected attenuation as the merge saturates
appears on snow and rain but NOT on fog, where ρ does not weaken. Reported as observed, no interpretation promoted.

### 2026-10-19 ~04:25 — Where GT-sky pixels go, by budget — **OBSERVED**
ACDC fog, % of GT sky (vanilla → PDA s1 → s5 → s10): building 11.9 → 26.0 → 50.3 → 49.3; classes other than sky or building
2.2 → 4.5 → 15.5 → **21.6** (road 0.4 → 10.9, vegetation 1.4 → 6.2). Rain and snow behave the same (19.7% and 18.9% beyond
building at s10). TENT s10 stays a two-way sky↔building merge (fog: building 51.2, sky 41.6, beyond-building 7.2; rain/snow
6.3–6.4). This measures the cause of the P6 departure (PDA plateau 50.3 → 49.3 at s10, which stays reported as **not
supported** under "non-decreasing"): building saturates by s5 and further loss goes to road/vegetation. Failure topology
is method-specific.

### Mechanism claims allowed in the main text (as of 2026-10-19)
1. Within-condition contrast predicts which images merge first (REGISTERED; moderate; fog + snow; rain null).
2. With more steps the merge spreads to every dataset tested, BDD clear probe included (OBSERVED; P6 data).
3. Failure topology is method-specific: TENT two-way, PDA spills into road/vegetation (OBSERVED).
Not claimed: any cross-dataset "contrast predicts onset" statement. The data contradict it: ACDC merges at 1 step at
normal contrast, while pre-collapsed night and Foggy Zurich do not.

### 2026-10-19 ~04:45 — Cross-dataset contrast vs merge onset — **EXPLORATORY**
Source: `results/xdataset_contrast_exploratory.json`. Per condition: contrast median [IQR]; sky→building % (vanilla / PDA@1 / PDA@10);
within-condition ρ(contrast, Δ sky IoU) at PDA@1 / PDA@10.

| condition | n_ok/n | contrast med [IQR] | sky→bld van / s1 / s10 | ρ PDA@1 / PDA@10 |
|---|---|---|---|---|
| acdc_fog | 93/100 | 116.0 [100.0, 135.3] | 11.9 / 26.0 / 49.3 | +0.294 / +0.342 |
| acdc_rain | 96/100 | 133.8 [124.3, 150.8] | 7.6 / 21.2 / 45.7 | +0.110 / −0.011 |
| acdc_snow | 92/100 | 110.4 [98.9, 120.9] | 12.4 / 22.7 / 45.5 | +0.293 / +0.137 |
| acdc_night | 97/106 | 17.7 [6.9, 25.9] | 69.8 / 46.8 / 38.7 | +0.107 / +0.168 |
| fz_fog | 28/40 | 63.3 [56.1, 72.5] | 76.1 / 82.9 / 59.2 | −0.328 / −0.275 |
| bdd_rainy | 29/40 | 134.2 [123.4, 148.1] | 8.7 / 11.3 / 22.1 | −0.130 / −0.228 |
| bdd_snowy | 85/110 | 107.3 [47.4, 134.3] | 6.1 / 8.6 / 26.7 | −0.576 / −0.273 |
| bdd_clear | 57/82 | 58.8 [25.8, 106.5] | 7.5 / 10.0 / 26.6 | −0.251 / +0.193 |

Readings (descriptive only):
(a) Cross-dataset contrast does NOT track 1-step merge onset. BDD rainy has ACDC-level contrast (134) but little 1-step merge
(8.7→11.3); BDD clear has low contrast (59), also with little 1-step merge; ACDC rain has the highest contrast and a strong merge.
This supports the earlier decision not to make any cross-dataset onset claim.
(b) Low-contrast conditions (night 18, FZ 63) are the ones vanilla already merges (70–76%). Consistent with grey-on-grey for
the unadapted model; not promoted.
(c) **The within-condition sign does not generalise beyond ACDC fog/snow.** On BDD and Foggy Zurich the per-image ρ is negative
at PDA@1 (BDD snowy −0.576 at n=85; FZ −0.328; BDD clear −0.251; BDD rainy −0.130). So mechanism claim (1) must be scoped exactly
as registered: *on ACDC fog (and snow)*. It must not be written as a general property of real shift. Caveat: BDD uses a
different label ontology and capture pipeline.

### 2026-10-19 ~05:00 — Interpretation guard for the 04:45 cross-dataset table — **NOTE (no new data)**
The eight within-condition ρ values in that table are **not eight estimates of one quantity**, and must not be pooled,
averaged, or compared as if they were:
- ACDC fog/snow: measured where the 1-step effect is active (vanilla 12% → PDA@1 23–26% sky→building).
- Foggy Zurich (and ACDC night): measured under saturation (vanilla already 70–76% merged), so the response is range-restricted.
- BDD rainy/snowy/clear: measured where the 1-step merge is essentially absent (+2.5 points), so the Δ-sky variance being correlated
  comes from whatever residual pathway moves sky there. BDD also has a different label ontology and capture pipeline.
Correlations computed in absent-effect or saturated-effect regimes are **different estimands**. The table is reported, not interpreted.
**Explicitly declined:** any "reverse mechanism" reading of the negative BDD/FZ values (e.g. BDD snowy ρ = −0.576, n = 85). That is
a post-hoc story on an exploratory, regime-mismatched estimate. It is not to be written in the paper, the supplement, or a rebuttal
as a finding.

### 2026-10-19 04:23 (cluster clock) — Correction to timestamps above — **NOTE**
The "~HH:MM" times on earlier entries were estimates, and several are later than the real clock: the guard entry above
was written at about 04:22, not "~05:00", and the cross-dataset entry at about 04:15, not "~04:45". The authoritative order
is the order of entries in this file (append-only). The prereg's luminance-direction addendum was written and synced to the compute cluster
before `sky_luminance_correlate.py` first ran, which is the ordering that matters for the REGISTERED tag. From here on, entry
times are taken from the cluster clock.

### 2026-10-19 04:44 (cluster clock) — Uncertainty on the step curve — **REGISTERED statistic, inference method chosen now**
The statistic is the pre-registered primary (mean real shortfall against the trend refit at each budget). The interval method
was chosen at this point: a joint bootstrap, B=2000, resampling corruption types (clusters of 5 severities) and images within
each real condition, with paired draws across budgets. Per-image CMs reproduce the reported mIoU (max |diff| < 1e-5).
PDA: s1 −5.38 [−7.74, −3.60]; s2 −7.54 [−10.02, −5.48]; s3 −8.68 [−11.22, −6.61]; s5 −10.69 [−13.13, −8.49]; s10 −11.94
[−14.25, −9.71]; all 7 below in 100% of draws at every budget. Growth vs s1 (paired): −2.16 [−2.60, −1.66], −3.30
[−3.99, −2.64], −5.31 [−6.02, −4.36], −6.56 [−7.39, −5.22]; P(diff<0) = 1.000 for each.
TENT (s10 pending): s1 −2.62 [−4.38, −1.50] (all-7 99.1%); s2 −3.52 [−5.44, −2.17]; s3 −4.36 [−6.45, −2.85]; s5 −5.90
[−8.14, −4.13]. Growth vs s1: −0.90 [−1.33, −0.46], −1.74 [−2.45, −1.06], −3.28 [−4.31, −2.19]; P = 1.000 each.
Wording rule: per-budget intervals overlap, so claim growth via the paired differences, not via non-overlap.
Source: `results/stepcurve_ci.json`, `stepcurve_ci.py`.

### 2026-10-19 05:13 (cluster clock) — PDA complete (665/665): pre-registered verdicts — **REGISTERED**
Full grid, Cityscapes-C v2, H200 pool, same-pool vanilla, trend refit at every setting.
| setting | synthetic benefit | real benefit | shortfall [95% CI, joint] | below |
|---|---|---|---|---|
| s1 lr1e-3 | +4.07 | +0.35 | −5.38 [−7.74, −3.60] | 7/7 |
| s2 | +4.47 | −0.93 | −7.54 [−10.02, −5.48] | 7/7 |
| s3 | +4.34 | −2.00 | −8.68 [−11.22, −6.61] | 7/7 |
| s5 | +4.11 | −4.04 | −10.69 [−13.13, −8.49] | 7/7 |
| s10 | +3.85 | −5.57 | −11.94 [−14.25, −9.71] | 7/7 |
| s1 lr3e-4 | +4.03 | +0.32 | −5.39 | 7/7 |
| s1 lr1e-4 | +4.02 | +0.31 | −5.39 | 7/7 |
- **P1 SUPPORTED:** shortfall < 0 and 7/7 below at every s.
- **P2 SUPPORTED:** synthetic s5 +4.11 ≥ s1 +4.07 − 0.5; real s5 −4.04 < s1 +0.35.
- **P4a SUPPORTED:** the synthetic-selected budget s* = 2 gives real shortfall −7.54 (worse than s1); tuning the budget on the
  benchmark worsens deployment.
- **P4b SUPPORTED** (label-selected upper bound, never deployable): symmetric best-s gives shortfall −7.43, 7/7. Six of seven
  real conditions are best at s1, the smallest tested budget; night is best at s2 (+1.53).
- **P5 SUPPORTED:** lr 3e-4 and 1e-4 both give −5.39, 7/7.
v2 endpoints reproduce v1 (s1: +4.07 / −5.38 vs v1 +4.05 / −5.37).

### 2026-10-19 05:13 — PDA's 1-step result is insensitive to learning rate — **OBSERVED**
Over a 10× LR range at one step, synthetic benefit +4.07 / +4.03 / +4.02 and real +0.35 / +0.32 / +0.31. Plausible reading,
not tested: at one step most of PDA's effect comes from BatchNorm running statistics updated during the train-mode
perturbed forwards (read in the eval-mode clean pass), not from the affine gradient. That links to the DUA/BN-recalibration
results. Stated as observed; a stats-only control (no gradient step) would be needed to claim it.

### 2026-10-19 13:12 (cluster clock) — Oracle and TENT complete: remaining pre-registered verdicts — **REGISTERED**
Oracle finished 08:12 (665/665 synthetic, 56/56 real, 0 failed). TENT 133/133 on V100. Pools: oracle H200, TENT V100, same-pool vanilla.
**P3 (supervised reference) SUPPORTED:** mean real benefit is non-decreasing in s: +1.16 / +1.77 / +2.29 / +3.19 / +5.82 (s1..s10). Synthetic
benefit also rises monotonically (+2.92 → +12.36), so the Fig. 3 caption's "the supervised curve rises in both regimes" is verified. The v1
handshake is exact (synthetic s1 +2.92 vs +2.93, s10 +12.36 vs +12.34, real s10 +5.82 vs +5.82).
**TENT:** P1 SUPPORTED (shortfall s1..s10: −2.62, −3.52, −4.36, −5.90, −8.22, all 7/7). P2 SUPPORTED (synthetic +2.37 → +4.34; real +0.73 → +0.15).
P4a SUPPORTED: synthetic-selected **s* = 10**; at s* real benefit is −1.18 (vs +0.73 at s1) and the shortfall −8.22. Paired growth vs s1 is
−5.60 [−7.12, −4.07]. P4b SUPPORTED (label-selected bound): −6.34, 7/7. P5 SUPPORTED: lr 3e-4 −2.58, lr 1e-4 −2.57.
TENT paired growth vs s1: s2 −0.90 [−1.33, −0.46], s3 −1.74 [−2.45, −1.06], s5 −3.28 [−4.31, −2.19], s10 −5.60 [−7.12, −4.07]; P = 1.000 each.
**Scope note:** P1/P2/P4/P5 were registered for PDA and TENT only. The analysis script also printed them for the oracle (incl. "P2 not
supported"); those oracle lines are descriptive and are NOT verdicts. The oracle's registered prediction is P3.

### 2026-10-19 13:12 — Oracle shortfall vs its own synthetic trend also deepens — **OBSERVED**
−2.64 / −3.51 / −4.31 / −5.83 / −8.32 (7/7 at every s), because its synthetic benefit grows faster than its real benefit (+2.92 → +12.36 vs
+1.16 → +5.82). Consistent with v1 (real ceilings below the synthetic ceiling line). Consequence for Fig. 3: a shortfall-axis panel would show
the supervised line falling alongside PDA/TENT. The dissociation the figure is meant to show (unsupervised real benefit falls, supervised real
benefit rises) appears on the real-BENEFIT axis.
