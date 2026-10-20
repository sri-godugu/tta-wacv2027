# UrbanState figure kit

Style language: numbered pastel stage-cards + dashed grouping frame (protocol),
one reserved accent for the finding (#D55E00), per-panel stat chips, shaded
±1 SD bands, direct labels with leaders, verdict strip. Palette is the
Okabe–Ito trio — CVD-validated (worst adjacent-pair ΔE 18.0 under protanopia,
all contrasts ≥ 3:1); identity is never color-alone (shapes: dots / stars /
squares / ticks). Fonts embedded as Type 42 (editable, no rasterized text).

Every plotted value is a released run output. `fig2_arch.py` draws a panel's
95-dot cloud only when its grid file exists; it never synthesizes points
(a missing cloud renders as a note).

## Files

```
pubstyle.py            shared style module (colors, chips, cards, arrows)
make_grid.py           builds data/grid_{primary,r101,fcn50}_pda.json from results/
                       (each write is checksummed against the published fit)
fig1_teaser.py         Fig. 1  — audit protocol + real spine + verdict strip (full width)
fig2_arch.py           Fig. 2  — three-panel architecture replication (full width)
fig3_decomposition.py  Fig. 3  — signal/recoverability dumbbells (column width)
checksum_grid.py       standalone checksum for the primary grid
data/                  exact run numbers (grid_primary verified to machine precision)
out/                   rendered PDFs (vector) + 300-dpi PNG previews
```

## On your machine (completes Fig. 2's R101/FCN clouds)

```bash
cp -r figkit ~/robustseg_wacv/figkit
cd ~/robustseg_wacv/results
python ../figkit/make_grid.py --out ../figkit/data   # prints 3× CHECKSUM PASS
cd ../figkit && python fig1_teaser.py && python fig2_arch.py && python fig3_decomposition.py
cp out/fig_teaser.pdf out/fig_arch_v2.pdf out/fig_decomp.pdf <paper-dir>/
```

## LaTeX

**Fig. 1 (page 1, replaces nothing — the teaser iWbd asked for):**
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{fig_teaser.pdf}
\caption{\textbf{The audit, and its headline measurement.}
\emph{Left}: the protocol — fit the synthetic difficulty--benefit law,
place natural shifts against it at matched difficulty, and separate
unrecoverable error from signal failure with a supervised reference.
\emph{Right}: on DeepLabV3+/ResNet-50, \pdaname\ benefit across the 95
Cityscapes-C conditions falls linearly with baseline accuracy
($R^2{=}0.44$, $\pm1$ residual-SD band); all seven real adverse conditions
(stars) fall below the band at matched difficulty, and real fog is
\emph{harmed} where the trend predicts a gain. Open marker: the non-adverse
BDD-clear probe; ringed dots: the extreme-saturation BN-recalibration
regime (\cref{sec:saturate}).}
\label{fig:teaser}
\end{figure*}
```

**Fig. 2 (replaces the current `fig_arch.pdf` figure, same caption works):**
```latex
\includegraphics[width=\linewidth]{fig_arch_v2.pdf}
```

**Fig. 3 (column width, in \cref{sec:mechanism}; if the 8-page budget is
tight it moves to the supplement unchanged):**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_decomp.pdf}
\caption{\textbf{The gap is the signal, not the error} (10-step budget).
Per condition: achieved \pdaname\ benefit (circle) vs.\ the supervised
reference (square) adapting the \emph{same} parameters; the tick marks the
benefit the synthetic trend predicts. The shaded span — the signal gap —
dominates everywhere (mean $+5.47$), while the reference sits near the
prediction on average (mean recoverability gap $-0.10$); dashed segments
mark the two conditions (night, Foggy Zurich) where supervision falls short
of the prediction.}
\label{fig:decomp}
\end{figure}
```

## Fig. 4 slot (tonight's step-curve runs)

`pubstyle.py` already carries everything the dissociation figure needs
(ACCENT vs. FITLN pair, chips, bands). When the step-{2,3,5} JSONs land,
the figure is: x = adaptation steps {1,2,3,5,10}, two panels
(synthetic-matched subset | 7 real), one line per method, supervised
reference as the blue dashed upper curve on real. Send the JSONs and I'll
ship `fig4_stepcurve.py` in minutes.

## Nit I noticed while mining the result files (worth one look)

Every eval `config` block echoes `"lambda_cons": 0.3` (an eval-script
default) while the paper states the checkpoint is trained with
$\lambda_{\mathrm{cons}}=0$ and the README says `--lambda_cons 0`. The flag
is unused at eval time, but an artifact auditor could read it as a
contradiction — one sentence in the release README ("`lambda_cons` in eval
configs is an unused training-time default; all reported checkpoints are
$\lambda=0$") closes it.

## Changes 2026-10-19 (originals in `_backup_20261019/`)

- **Fig. 1 teaser:** legend moved above the plot (it hid the ringed +31 saturation point); y headroom to 35.5; wider gap
  so tick labels clear the dashed frame; "snowy" label moved up-left; the BDD-clear direct label dropped (its leader had
  to cross the BDD-snowy star; it is the only open marker and is named in the legend).
- **Fig. 2 architecture:** R101 and FCN clouds now render (`make_grid.py` on the mirrored raw results, both CHECKSUM PASS);
  stats moved from in-plot chips to a muted line under each title; true minus signs; (b)/(c) annotations moved into the
  freed upper band; title double space fixed.
- **Fig. 3 decomposition:** the summary moved from a chip over the ACDC-fog row to two muted lines above the rows (left);
  the trend-prediction tick now draws above the markers with a surface outline (it was hidden behind the ACDC-rain square).
- **Fig. 4 step curve:** fed by `build_v2_data.py` (run outputs → `data/`, no hand-copied numbers); vanilla paired **per GPU
  pool** (PDA/oracle ↔ H200, TENT ↔ V100); per-budget 95% CI bands from `data/stepcurve_ci.json` plus a chip stating the
  paired test; legend above the panels; shorter titles and y-label; (c) key moved to its empty corner. Re-run
  `python build_v2_data.py <results/stepcurve> --ci <stepcurve_ci.json> && python fig4_stepcurve.py` as runs land;
  incomplete settings drop out instead of being drawn.
- **Fig. 5 qualitative strip (new):** `python render_strip.py data/strip out` (alias: `fig5_strip.py`). Pre-registered ACDC-fog indices 14/42/51,
  predictions from the same pass as the scored metrics, index→image proven against the GT files; the sky-fate palette is
  validated (CVD ΔE ≥ 9.2). The supervised@10 column fills in once `strip_ora_s10…npz` is in `data/strip/`.
- `fig4_interim.py` / `fig_stepcurve_interim.pdf` are superseded by `fig4_stepcurve.py` and kept only for reference.
- **Fig. 3 / `fig4_stepcurve.py` panel (b) is now REAL mean benefit (default), 2026-10-19 14:20.** PDA falls (+0.35→−5.57),
  TENT falls (+0.73→−1.18), the supervised reference rises (+1.16→+5.82): the dissociation on one axis, in one quantity. The
  difficulty-controlled shortfall with paired CIs lives in the caption and §6. On a shortfall axis the supervised curve also
  falls (−2.64→−8.32, vs its own steep synthetic trend). That is the disclosed §6 caveat, not the claim, so it is not plotted.
  The shortfall version renders on request: `FIG4_B=shortfall python fig4_stepcurve.py` → `out/fig_stepcurve_shortfall.*`
  (supplement).
