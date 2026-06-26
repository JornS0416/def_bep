"""
Care-PD–faithful severity-classification metrics
================================================
Single source of truth for how UPDRS-gait macro-F1 is computed across every
evaluation protocol (LOSO, MIDA, LODO, CDA) and every backbone (VideoMAE,
DINOv2, V-JEPA). Importing these helpers everywhere guarantees the four
protocols stay mutually comparable and match the reference implementation in
the Care-PD paper.

─────────────────────────────────────────────────────────────────────────────
WHAT THE CARE-PD REFERENCE ACTUALLY DOES
─────────────────────────────────────────────────────────────────────────────
Source: TaatiTeam/CARE-PD (branch master), `test.py` + `const/const.py`.

1. POOLING.  Inside `test__hypertune`, predictions from every fold are
   accumulated into two flat lists:

       total_outs_last.extend(outs_last)     # predicted classes
       total_gts.extend(gts)                 # ground-truth classes

   and the reported score is computed ONCE over the pooled arrays, AFTER the
   fold loop (output file: `last_report_allfolds.txt`):

       rep      = classification_report(total_gts, total_outs_last)               # F1_{0-3}
       rep012   = classification_report(total_gts, total_outs_last,
                      labels=[l for l in LABELS_INCLUDED_IN_F1_CALCULATION
                              if l in total_gts])                                  # F1_{0-2}
       macro_f1 = classification_report(..., output_dict=True)['macro avg']['f1-score']

   They do NOT average a per-fold F1 across folds. (A per-fold F1 exists in
   `train.py`, but only as the *validation* score for early-stopping / Optuna
   model selection — never as the reported test metric.)

2. NO FORCED ABSENT CLASSES.  `classification_report` is called WITHOUT a fixed
   `labels=[0,1,2,3]`, so the macro average runs only over the classes actually
   present (`unique_labels(y_true, y_pred)`). An absent UPDRS class is never
   injected as an F1=0 term. The only label restriction is the deliberate
   class-3 handling for the rare/absent severe class:

       const.LABELS_INCLUDED_IN_F1_CALCULATION = [0, 1, 2]

   which defines the second reported metric F1_{0-2}.

─────────────────────────────────────────────────────────────────────────────
WHY THIS MATTERS (the bug these helpers replace)
─────────────────────────────────────────────────────────────────────────────
The previous per-script code computed a per-fold macro-F1 with the full label
set forced in (`labels=[0,1,2,3]`) and then averaged across folds. In LOSO/MIDA
every fold is a single held-out subject, so within a fold `y_true` is (often) a
single constant UPDRS label. Forcing all four labels then caps a *perfect* fold
at m/4 (e.g. 0.25 for a single-label subject), and averaging those capped folds
deflated the reported within-dataset F1 by 2–3× relative to Care-PD. Pooling +
present-class averaging removes both artefacts. See the methodology appendix
(docs/appendix_evaluation_metrics.md) for the full derivation and a worked
example.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
For LOSO / MIDA  (per-subject folds): accumulate every out-of-fold walk-level
(true, pred) pair across all folds, then call these once on the POOLED arrays
for the headline score. Keep the per-fold values only for a between-subject SD
(Care-PD Appendix D.4 variability reporting).

For LODO / CDA  (each fold/cell is already a whole held-out dataset that
contains several classes): just call these on that test set's (true, pred) —
no cross-fold pooling needed, the per-target value is the comparable unit.
"""

from sklearn.metrics import f1_score

# Identical to Care-PD const.LABELS_INCLUDED_IN_F1_CALCULATION.
LABELS_INCLUDED_IN_F1_CALCULATION = [0, 1, 2]


def _present(y_true):
    """Set of integer labels that actually occur in the ground truth."""
    return {int(v) for v in y_true}


def macro_f1_03(y_true, y_pred):
    """
    F1_{0-3}: macro-F1 over the UPDRS classes PRESENT in the data.

    Mirrors Care-PD `classification_report(y_true, y_pred)['macro avg']` — with
    no fixed label set, scikit-learn averages over `unique_labels(y_true, y_pred)`,
    i.e. the classes present in the truth OR the predictions. Absent classes are
    never forced into the average. `zero_division=0` matches the reference.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def macro_f1_02(y_true, y_pred):
    """
    F1_{0-2}: macro-F1 restricted to labels [0, 1, 2] that occur in y_true.

    Mirrors Care-PD exactly:
        labels = [l for l in LABELS_INCLUDED_IN_F1_CALCULATION if l in total_gts]
        classification_report(total_gts, total_outs_last, labels=labels)
    The rare/absent severe class 3 is dropped from the average so it cannot
    deflate the metric on datasets that lack it (e.g. BMClab, T-SDU-PD). Returns
    NaN if none of {0,1,2} occur in y_true.
    """
    present = _present(y_true)
    labels = [l for l in LABELS_INCLUDED_IN_F1_CALCULATION if l in present]
    if not labels:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=labels, zero_division=0))


def weighted_f1(y_true, y_pred):
    """
    Support-weighted F1 over present classes (a BEP extra, not a Care-PD metric).
    No forced label set, so absent classes are not injected as F1=0 terms.
    """
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
