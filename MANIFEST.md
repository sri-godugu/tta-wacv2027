# Claim → file → script

Each paper claim, the aggregate file it is read from (in `results/` unless stated) and the script that produces it.
Scripts read only the released per-condition outputs; run them from `results/`.

## Main paper

| Section | Claim | File | Script |
|---|---|---|---|
| §4, Eq. 1 | difficulty law, all 95 conditions: slope −0.192, R² 0.44, residual SD 4.03 | `fit_robustness.json` (linear) | `fit_robustness.py` |
| §4 | slopes / R² across architectures (−0.192 / −0.273 / −0.193; 0.44 / 0.69 / 0.41) | `backbone_spine.json` | `backbone_spine.py` |
| §4 | R² by mechanism (PDA 0.44, TENT 0.35, DUA 0.16) | `fit_robustness.json`, `dua_spine.json` | `fit_robustness.py`, `dua_spine.py` |
| §5.1, Tab. 1 | 7/7 adverse below trend; per-condition residuals and paired bootstrap CIs (28 intervals) | `bootstrap_ci.json` | `bootstrap_ci.py` |
| §5.1 | type-clustered bootstrap, all approaches: every CI excludes 0, all-7-below 98–100% | `clustered_bootstrap_all.json` | `clustered_bootstrap.py` (routine), `clustered_bootstrap_all.py` (all approaches) |
| §5.1 | matched difficulty is interpolation (4–9 neighbours within ±3 mIoU) | (printed) | `density_and_pin.py` |
| §5.1 | ACDC fog stable across three seeds (−2.52 / −2.42 / −2.50) | `acdc_fog_pdarep2.json`, `acdc_fog_pdarep3.json`, `cm_acdc_fog_pdaonly.json` | — |
| §5.2 | TENT / DUA / self-training replicate (means −2.63 / −4.92 / −2.42) | `fit_robustness.json`, `dua_spine.json` | `fit_robustness.py`, `dua_spine.py` |
| §5.2 | robustness to dataset removal and trend specification (linear/quadratic/Theil–Sen/LOESS) | `fit_robustness.json` | `fit_robustness.py` |
| §5.3 | architecture replication: R101 −6.54 (7/7), FCN −4.88 (7/7); FCN Foggy Zurich +4.11 vs +7.69 | `backbone_spine.json` | `backbone_spine.py` |
| §6 | supervised reference: recoverability gap −0.10, signal gap +5.47; 1-step gate 61/95, 10-step 2/95 | `decomposition_s10.json`, `decomposition_s1.json` | `decomposition.py` |
| §6 | budget control at ten steps (TENT +2.40 synthetic vs −1.90 real; PDA −0.22 vs −5.94) | `budget_control.json` | `budget_control.py` |
| §6, Fig. 3 | step curve {1,2,3,5,10} and LR axis on Cityscapes-C v2: benefit and shortfall per budget | `stepcurve_final_all.json` | `stepcurve_analysis.py` |
| §6 | paired budget differences (every increase over one step excludes 0) | `stepcurve_ci.json` | `stepcurve_ci.py` |
| §6 | synthetic-selected budget (PDA 2, TENT 10), label-selected bound (−7.43 / −6.34), LR control | `prereg_pda_final.json`, `prereg_tent_ora_final.json` | `prereg_p2_p4_p5.py` |
| §6 | supervised reference rises with budget (+1.16 → +5.82 real) | `stepcurve_final_all.json` | `stepcurve_analysis.py` |
| §6 | sky→building at one step (11.9 / 26.0 / 28.7%; 100/100 images; −1.47 of −2.49) | `per_image_cm/acdc_fog_*.npz` | `fog_confusion.py`, `sky_building_scope.py` |
| §6 | sky→building across steps; leakage past building (2.2 → 21.6%, road 0.4 → 10.9) | `stepcurve/*/per_image_cm/*.npz` | `p6_and_trace.py`, `perclass_fog.py` |
| §6 | objective descends while accuracy diverges (JSD 0.304 → 0.260; entropy 0.266 → 0.169) | `stepcurve/*/` (`mean_loss_trace`) | `p6_and_trace.py` |
| §6 | luminance correlate, registered: fog ρ = +0.29, p = 0.004 | `sky_luminance_correlate.json` | `sky_luminance_correlate.py` |
| §7 | confidence proxy R² 0.53 vs 0.44; corr with vanilla mIoU 0.81; partial −0.43 / −0.09 | (printed) | `confidence_proxy_all95.py`, `confidence_proxy.py` |
| §7 | synthetic-calibrated gate degenerate (τ = 0.974) | `policy_eval_pdaonly.json`, `policy_eval_tent.json` | `policy_eval.py` |
| §7 | leave-one-condition-out gate: PDA 82%, DUA 94% of the oracle margin; TENT −0.10 | `policy_loo_*.json` | `policy_loo.py` |
| §8 | JSD partial correlation given difficulty −0.18 | (printed) | `partial_corr.py` |
| §8 | perturbation-consistency regularizer dose (51.75 / 51.67 / 51.11 / 51.06) | `cityscapes_c_dosage_lambda*.json`, `cityscapes_c_dl_ft_*` | `dosage_verdict.py` |
| §8 | saturation regime: BN recalibration (PDA +31.2, TENT +24.1, DUA +33.3 at severity 5) | `pertype_*_saturate.json`, `pertype_tent_saturate.json` | `saturate_check.py`, `saturate_tent.py`, `saturate_degen.py` |
| §9 | SegFormer leverage check (+0.76 PDA, +0.06 TENT on Cityscapes-C) | `cityscapes_c_sf_*` | — |
| Artifacts | Cityscapes-C v2 validation gate: deterministic max \|Δ\| 0.019; stochastic shift median 0.16, max 0.88 | `csc_v2_gate_h200.json`, `csc_v2_gate_v100.json` | `csc_v2_gate.py` |

## Supplementary only

| Claim | File | Script |
|---|---|---|
| second real dataset placement (BDD100K, Foggy Zurich) | `second_dataset_spine.py` output (printed) | `second_dataset_spine.py` |
| cross-dataset contrast check (**EXPLORATORY**, not a registered claim) | `xdataset_contrast_exploratory.json` | `xdataset_contrast_exploratory.py` |
| per-figure inputs | `figures/data/` | `figures/build_v2_data.py`, `figures/make_grid.py` |
| qualitative strips (ACDC fog, pre-registered indices 14 / 42 / 51), with the image-identity provenance check | `figures/data/strip/` (saved predictions + ground truth) | `figures/render_strip.py` |

## Raw run naming

| Prefix | Contents |
|---|---|
| `pertype_{vanilla,pdaonly,tent,dua,pseudo}_<type>.json` | primary backbone, 19 Cityscapes-C types × 5 severities |
| `cm_<condition>_<method>.json` | real conditions, seeded runs behind the bootstrap CIs (with `per_image_cm/`) |
| `r101_*`, `fcn50_*` | architecture replications (synthetic per type, real per condition) |
| `oracle_s{1,10}_*`, `s10_{pdaonly,tent}_*` | supervised reference and budget control (v1) |
| `stepcurve/h200/{van,pda,ora}_*`, `stepcurve/v100/{van,tent}_*` | step curve / LR axis on Cityscapes-C v2; `_v2pertype_<type>_sev<s>` (synthetic), `_<condition>` (real) |

Earlier exploratory runs from the project's method-development phase (e.g. SegFormer variants, RobustNet) are kept for
provenance; only the files listed above back claims in the paper.
