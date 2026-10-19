# Pre-registration — oracle-adaptation ceiling and the shortfall decomposition

**Written 2026-08-13, BEFORE any oracle run exists.** No oracle number has been
computed at the time of writing. Fixed in advance: the estimand, the
decomposition, the hypothesis, and what each possible outcome licenses us to
claim. Registered because the whole point is that either branch is publishable,
so the read must not be chosen after seeing the numbers.

## What is run

The existing adaptation loop with the **loss swapped**, nothing else changed:
same adapted parameters (normalization affine, `norm_types=bn`), same optimiser
(Adam, `lr=1e-3`), same `n_steps=1`, same `batch_size=2`, same checkpoint
(`tcpr_deeplab_pdaonly`, true vanilla), same 512x1024 protocol. The unsupervised
signal (PDA's cross-perturbation disagreement / TENT's entropy) is replaced by
**cross-entropy against ground truth** on the same batch.

This is an **oracle / ceiling**, not a method: it is transductive (it sees the
test labels it is scored on). It answers one question only — *how much could
normalization-parameter adaptation recover on this condition if the adaptation
signal were perfect?*

Grid: 19 CS-C corruption types x 5 severities (95 synthetic points) + 8 real
condition-points (ACDC fog/night/rain/snow, BDD rainy/snowy/clear, Foggy Zurich).

## Definitions (fixed now)

For each condition, with `pred = -0.192 * vanilla + 13.30` (the pinned primary
PDA fit (a), n=95, nothing excluded):

- `ceiling`            = oracle mIoU - vanilla mIoU
- `recoverability gap` = pred - ceiling      (benefit the fit predicts that even a
                                              perfect signal cannot deliver via
                                              these parameters)
- `signal gap`         = ceiling - actual    (recoverable benefit that the
                                              unsupervised signal failed to exploit)
- `total shortfall`    = pred - actual = recoverability gap + signal gap   (identity)

`actual` = PDA-only benefit (primary). The same decomposition is computed with
TENT as `actual` in a secondary table.

## Hypothesis (locked)

**H1 (primary): at matched difficulty, real conditions show a larger
recoverability gap than synthetic ones.** I.e. the shortfall documented in the
spine is carried mainly by the *recoverability* term, not the *signal* term —
synthetic-corruption error is disproportionately normalization-recoverable and
real-shift error is not.

Operational read: regress `ceiling` on `vanilla` over the 95 synthetic points
exactly as the benefit fit was built; place the 8 real points as residuals in SD
units. H1 predicts the real ceilings sit **below** the synthetic ceiling line,
and that the ceiling residual accounts for the majority of the benefit residual
already measured (median share > 50% across the 7 adverse conditions).

**H0 / alternative outcomes, all pre-committed:**

- **(A) Ceilings low on real at matched difficulty** (H1 confirmed) -> mechanism
  is the *error structure*: no unsupervised TTA of this family could close the
  gap. Paper claims: benchmarks overestimate benefit *because* synthetic
  corruption error is normalization-recoverable and real-shift error is not.
- **(B) Ceilings comparable on real, but PDA/TENT fall short of them** -> mechanism
  is the *adaptation signal*: entropy/disagreement mislead under real shift.
  Equally reportable, and it names what future methods must fix.
- **(C) Mixed across conditions** -> report the per-condition decomposition; the
  claim becomes conditional and the table is the contribution.

No outcome is a null result; the section is written either way. Which of A/B/C
holds decides only *which mechanism sentence* is written.

## Analysis plan (fixed now)

1. Sanity gate: `ceiling >= actual` for essentially all conditions. A ceiling
   materially **below** the achieved unsupervised benefit means the oracle is
   under-optimised (1 gradient step is not a true supremum), not that the method
   beats the ceiling. If this fires on >10% of conditions, the ceiling is
   reported as "1-step supervised reference", not "ceiling", and H1 is judged on
   the paired comparison only.
2. Paired bootstrap CIs (B=10000, image-level, same machinery as
   `bootstrap_ci.py`) on the ceiling residual for the 8 real points.
3. Report `recoverability gap` / `signal gap` per condition, plus the adverse-only
   mean split.
4. The BDD-clear probe stays separate from the 7-adverse headline, as agreed.

## Known limitations to disclose regardless of outcome

- The ceiling is specific to **these parameters** (BN affine) and **this budget**
  (1 step, bs=2). It is a ceiling for *this family of adaptation*, not for
  adaptation in general. A larger budget would raise it.
- The oracle is transductive and is never proposed as a method.
- One backbone (DeepLabV3+/ResNet-50), as elsewhere in the paper.
