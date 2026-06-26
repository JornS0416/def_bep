# Appendix — Evaluation metrics

This appendix documents how the severity-classification metrics are computed and
why, so every protocol (LOSO, MIDA, LODO, CDA) and every backbone (VideoMAE,
DINOv2, V-JEPA 1/2) stays mutually comparable and faithful to the Care-PD paper.
The single implementation lives in
[`scripts/_evaluation/carepd_metrics.py`](../scripts/_evaluation/carepd_metrics.py);
this file is the prose companion referenced from the evaluation scripts.

## The three reported metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| **F1_{0-3}** | macro-F1 over the UPDRS-gait classes **present** in the data | `macro_f1_03` |
| **F1_{0-2}** | macro-F1 restricted to labels {0,1,2} that occur in the truth | `macro_f1_02`; drops the rare/absent severe class 3 |
| **weighted F1** | support-weighted F1 over present classes | `weighted_f1`; a BEP extra, not a Care-PD metric |

All three use `zero_division=0` and **no forced label set**, so a class that does
not occur is never injected as an F1 = 0 term.

## What the Care-PD reference does

Source: `TaatiTeam/CARE-PD` (`test.py` + `const/const.py`).

1. **Pooling.** Predictions from every fold are accumulated into two flat lists
   (`total_outs_last`, `total_gts`) and the score is computed **once** over the
   pooled arrays *after* the fold loop — not as an average of per-fold F1 scores.
2. **Present classes only.** `classification_report` is called without a fixed
   `labels=[0,1,2,3]`, so the macro average runs over
   `unique_labels(y_true, y_pred)`. The only deliberate label restriction is
   `LABELS_INCLUDED_IN_F1_CALCULATION = [0, 1, 2]`, which defines F1_{0-2}.

## The bug these helpers replace

Earlier per-script code computed a per-fold macro-F1 with the full label set
forced in (`labels=[0,1,2,3]`) and then averaged across folds.

In LOSO/MIDA each fold is a single held-out subject, so within a fold `y_true` is
often a single constant label. Forcing all four labels caps a *perfect* fold at
1/4 = 0.25 (one correct class out of four), and averaging those capped folds
deflated the reported within-dataset F1 by 2–3× versus Care-PD.

**Worked example.** A subject whose every walk is truly class 1 and is predicted
perfectly:

- *forced-label, per-fold:* `f1_score(y_true=[1,1], y_pred=[1,1], labels=[0,1,2,3])`
  → classes 0, 2, 3 contribute F1 = 0 → macro-F1 = 1/4 = **0.25**.
- *present-class, pooled:* the same subject pooled with the rest of the dataset
  contributes its true positives to class 1's F1 with no phantom zeros → the
  pooled macro-F1 reflects the **actual** per-class performance.

Pooling + present-class averaging removes both artefacts.

## How each protocol applies it

- **LOSO / MIDA** (per-subject folds): accumulate every out-of-fold, walk-level
  `(true, pred)` pair across all folds, then call the helpers **once** on the
  pooled arrays for the headline score. Per-fold values are kept only for a
  between-subject SD (Care-PD Appendix D.4 variability reporting).
- **LODO / CDA** (each fold/cell is already a whole held-out dataset with several
  classes): call the helpers directly on that test set's `(true, pred)` — no
  cross-fold pooling needed; the per-target value is the comparable unit.
