# When Does Test-Time Adaptation Help? The Synthetic-to-Real Gap in Urban Driving Segmentation

> **📄 Supplementary material: [`wacv_urbanstate_supplementary_ID25.pdf`](wacv_urbanstate_supplementary_ID25.pdf)**
> (11 pages: pre-registration protocol, per-condition tables for every approach and architecture, robustness grids,
> gating tables, the budget campaign with paired intervals, the sky-to-building analyses, qualitative strips, and
> reproduction details). The main paper's references to "the supplementary" point to this document.

## Quick navigation

| What | Where |
|---|---|
| Supplementary PDF | [`wacv_urbanstate_supplementary_ID25.pdf`](wacv_urbanstate_supplementary_ID25.pdf) |
| Paper claim → file → script | [`MANIFEST.md`](MANIFEST.md) |
| Pre-registration documents and analysis log | [`preregistration/`](preregistration/) |
| Per-condition results and analysis scripts | [`results/`](results/) |
| Step-curve / learning-rate runs | [`results/stepcurve/`](results/stepcurve/) |
| Seeded Cityscapes-C v2 (seeds, sha256, generator) | [`benchmark/cityscapes_c_v2/`](benchmark/cityscapes_c_v2/) |
| Adaptation methods and evaluation code | [`code/`](code/) |
| Figure scripts and rendered figures | [`figures/`](figures/) |

### Names used in the supplementary

| Supplementary says | In this repository |
|---|---|
| `clustered_bootstrap.py` | [`results/clustered_bootstrap.py`](results/clustered_bootstrap.py) (the routine; primary PDA run). The per-approach and per-architecture intervals come from [`results/clustered_bootstrap_all.py`](results/clustered_bootstrap_all.py), which imports it unmodified and writes `clustered_bootstrap_all.json` |
| `stepcurve_ci.json` | [`results/stepcurve_ci.json`](results/stepcurve_ci.json) (from `stepcurve_ci.py`); `build_v2_data.py` copies it verbatim to `figures/data/stepcurve_ci.json` |
| `render_strip.py` (with the provenance checks) | [`figures/render_strip.py`](figures/render_strip.py); every saved label map is checked against its ground truth before drawing. `figures/fig5_strip.py` is an alias |
| `csc_v2_release/` | [`benchmark/cityscapes_c_v2/`](benchmark/cityscapes_c_v2/) (generator, root seed, 47,500-row seed + SHA-256 manifest, environment record, README, SHA256SUMS) |

Code, per-condition results, analysis scripts, pre-registration record and the seeded Cityscapes-C v2 benchmark
for the paper of the same title (anonymous submission). The paper audits test-time adaptation (TTA) for semantic
segmentation. It measures how adaptation benefit depends on baseline difficulty across 95 Cityscapes-C conditions,
places seven real adverse conditions (ACDC fog/night/rain/snow, BDD100K rainy/snowy, Foggy Zurich) against that
synthetic trend, and decomposes the shortfall into recoverable error versus adaptation-signal failure with a
supervised reference.

Every number in the paper and supplementary traces to a released result file and the script that produced it. `MANIFEST.md` maps each claim to its file and script.

## Layout

```
wacv_urbanstate_supplementary_ID25.pdf   supplementary material (11 pages)
code/                      adaptation methods, models, data loaders, evaluation
  adaptation/              pda.py (PDA), tent.py, dua.py, pseudolabel.py (self-training), oracle.py (supervised reference)
  scripts/evaluate.py      single entry point for every evaluation run (flags recorded in each result's "config")
  scripts/generate_cityscapes_c_v2.py   seeded Cityscapes-C generator (see benchmark/)
  scripts/stepcurve_worker.sh           how the step-curve runs were invoked
results/                   one flat directory, exactly as the analysis scripts expect it:
  *.json                   per-condition evaluation outputs (one per run; "config" = exact flags) and the aggregate
                           analysis outputs the paper's numbers are read from (no "config" block)
  *.py                     the analysis scripts that turn the runs into the aggregate outputs
  per_image_cm/            per-image confusion matrices behind the image-level bootstrap intervals
  stepcurve/{h200,v100}/   step-curve / learning-rate runs on Cityscapes-C v2 (synthetic per severity, real per condition,
                           real per-image confusion matrices, qualitative-strip predictions); one GPU pool per method
benchmark/cityscapes_c_v2/ seed + sha256 manifest, environment record and generator for the seeded benchmark
                           (called csc_v2_release/ in the supplementary)
preregistration/           dated pre-registration documents and the append-only analysis notes
figures/                   figure scripts (fig1-fig4, render_strip.py), their input data, and the rendered PDFs
```

## Setup

```bash
pip install -r requirements.txt
```

Datasets are not redistributed: Cityscapes, ACDC, BDD100K and Foggy Zurich are available from their providers under their
own licenses; Cityscapes-C v2 is regenerated from Cityscapes val with the seeded generator (below). Paths in result
`config` blocks are relative to the repository root (`datasets/...`, `runs/...`). Model checkpoints are not included in
this repository because of their size (DeepLabV3+/ResNet-50, ResNet-101 and FCN/ResNet-50, trained on clean Cityscapes,
80 epochs, ImageNet-pretrained backbone).

## Reproducing results

Single evaluation (every run in `results/` was produced this way; its exact flags are in its `config` field):

```bash
cd code
python scripts/evaluate.py --checkpoint <ckpt> --pretrained --backbone deeplabv3plus \
  --dataset acdc --data_root ../datasets/acdc --condition fog --split val \
  --use_pda --K 4 --clean_final --n_adapt_steps 1 --batch_size 2 --seed 0 --device cuda \
  --results_json out.json --per_image_cm out.npz
```

Adaptation is episodic: parameters reset to the vanilla checkpoint before each batch. Benefit is always computed against a
vanilla run on identical inputs and the same GPU type.

Analyses (CPU, seconds to minutes) run from `results/`. Each regenerates its aggregate output in place; all of the
following were checked to reproduce the shipped files exactly:

```bash
cd results
python bootstrap_ci.py                    # per-condition paired bootstrap intervals (needs per_image_cm/)
python fit_robustness.py                  # leave-one-dataset-out and trend-specification robustness
python backbone_spine.py                  # architecture replication (ResNet-101, FCN)
python dua_spine.py                       # DUA difficulty law and shortfall
python decomposition.py                   # supervised-reference decomposition (1 and 10 steps)
python budget_control.py                  # unsupervised methods at the supervised reference's budget
python policy_eval.py; python policy_loo.py   # synthetic-calibrated and leave-one-condition-out gates
python clustered_bootstrap.py             # corruption-type-clustered bootstrap (primary PDA run)
python clustered_bootstrap_all.py         # the same routine, imported unmodified, for every approach and architecture
python stepcurve_analysis.py stepcurve --json stepcurve_final_all.json   # step curve, trend refit at every budget
python prereg_p2_p4_p5.py stepcurve       # pre-registered step-curve predictions
python stepcurve_ci.py stepcurve --B 2000 --json stepcurve_ci.json # joint bootstrap, paired budget differences
```

Figures: `cd figures && python build_v2_data.py ../results/stepcurve --ci ../results/stepcurve_ci.json`
(from `figures/`), then `python fig1_teaser.py`, `fig2_arch.py`, `fig3_decomposition.py`, `fig4_stepcurve.py`, `render_strip.py data/strip out` (`fig5_strip.py` is an alias).

## Cityscapes-C v2 (seeded, per-image reproducible)

`benchmark/cityscapes_c_v2/` holds a seeded regeneration of Cityscapes-C (19 corruptions × 5 severities × 500 val images).
Each image is seeded independently:

```
seed = SeedSequence([SEED0, img_idx, TYPE_IDX[corruption], severity]).generate_state(1)[0]   # SEED0 = 20261018
```

applied to NumPy's global RNG and to `skimage.util.random_noise` (which ignores the global RNG in scikit-image >= 0.19)
immediately before each `corrupt()` call. Any single image can be regenerated alone and checked against its sha256 in
`MANIFEST.csv`. Byte identity holds within the pinned environment in `MANIFEST_env.json`. Unseeded generation moves
stochastic per-condition vanilla mIoU by a median 0.16 and up to 0.88 mIoU; the 8 deterministic types reproduce the
original within 0.02.

## Pre-registration record

`preregistration/` contains the dated documents written before the corresponding runs (supervised-reference experiment;
step curve, learning-rate axis, sky-to-building mechanism, Cityscapes-C v2 validation gate, and one post-submission
control) and `analysis_notes.md`, an append-only log in which every result is tagged REGISTERED, OBSERVED or EXPLORATORY.
For anonymity, names of the compute infrastructure and absolute file-system paths in these documents were replaced with
generic terms; no other text was changed.

## Notes

- `lambda_cons` in evaluation `config` blocks is an unused training-time default of the evaluation script; all reported
  checkpoints are trained with `lambda_cons = 0` (the perturbation-consistency regularizer is evaluated only in its
  dose-response control).
- `--use_oracle` is the supervised reference: transductive by construction (it sees the labels of the batch it is scored on)
  and never proposed as a method.
- Adaptation runs are single runs per condition, seeded. The step-curve endpoints were re-run on regenerated data and on
  different GPU types and reproduce the originals within 0.02 mIoU (0.21 for PDA at ten steps).
