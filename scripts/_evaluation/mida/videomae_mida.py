"""
VideoMAE — Multi-dataset In-domain Adaptation (MIDA) Evaluation
=================================================================
Re-runs the within-dataset LOSO protocol on the target dataset, but augments
EVERY fold's training set with the full pool of the other 3 ("auxiliary")
datasets. Tests on the same held-out subject as standard LOSO.

Matches the Care-PD paper MIDA protocol exactly (sec. 4.2 + Appendix D.4):
  "Starting from the LODO checkpoint, we fine-tune the probe (but keep the
   encoder frozen) on the target cohort's training split — again under
   LOSO — and test on its held-out subjects."
  "(iii) MIDA: we re-run LOSO after adding external data to the training
   split, so n and the computation are identical to (i) [LOSO]."
  Table D.2: "MIDA (LOSO: train on target train split + auxiliary datasets,
   test on target test split)"

─────────────────────────────────────────────────────────────────────────────
WHY A SINGLE-STAGE FIT INSTEAD OF "LODO CHECKPOINT → FINE-TUNE"
─────────────────────────────────────────────────────────────────────────────
The paper describes MIDA procedurally as starting from a LODO-trained probe
and then fine-tuning it on (target_train_fold ∪ auxiliary). For OUR probes
(LogisticRegression / Ridge — both convex, L2-regularised, solved to
convergence with lbfgs / Cholesky), the converged solution is independent of
the initialisation: training from scratch on the combined set
(target_train_fold ∪ auxiliary) yields the EXACT SAME fitted model as
"initialise at the LODO solution, then continue training on that same
combined set". We therefore fit directly on the union — mathematically
identical end result, far simpler code, and no risk of an under-converged
"fine-tune" step silently producing a worse (non-equivalent) model.

(Sequential "train on source, then continue training on target-only" would
NOT be equivalent — that converges to the target-only solution, discarding
the source signal. That is why the paper's "fine-tune on the target split"
step explicitly KEEPS the auxiliary data in the loss — it is not target-only
fine-tuning, it is joint training on the union. See run_mida_dataset().)

─────────────────────────────────────────────────────────────────────────────
WHY NO SPECIAL SAMPLE WEIGHTING
─────────────────────────────────────────────────────────────────────────────
Each LOSO training fold already contains (n_subjects - 1)/n_subjects of the
target cohort (e.g. ~97% for PD-GaM's 30 subjects) — roughly comparable in
scale to the auxiliary pool. The target signal is not "drowned out", so we
pool the data directly and rely on the existing class_weight="balanced"
(for UPDRS-label imbalance) — exactly as in LOSO/LODO/CDA, and consistent
with how the paper describes MIDA (no mention of domain-balancing weights).

─────────────────────────────────────────────────────────────────────────────
COMPARISON FRAME — LODO vs. MIDA vs. LOSO
─────────────────────────────────────────────────────────────────────────────
This is exactly the 3-way comparison the Care-PD paper itself uses
(Fig. 4, Table D.2, Fig. D.9):
  - LODO  = zero target-domain supervision (lower bound / generalisation gap)
  - MIDA  = target supervision + auxiliary cohorts (adaptation)
  - LOSO  = target-only supervision (in-domain ceiling, no auxiliary data)
"Comparing MIDA to LODO quantifies how much performance can be recovered by
a modest amount of in-domain supervision" (paper, sec. 4.2). "Comparing MIDA
to within-dataset LOSO highlights the value of [additional, diverse] data in
boosting performance" (paper, sec. 5.1).

Usage
-----
    cd main_project
    python scripts/_evaluation/mida/videomae_mida.py

Prerequisite: run scripts/videomae/extract_features.py first.
"""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    cohen_kappa_score, confusion_matrix,
)
from scipy.stats import spearmanr

# ── Care-PD–faithful metrics (shared module) ──────────────────────────────────
# Pooled, present-class macro-F1. Replaces the old per-fold, forced-label F1
# that deflated the LOSO-on-target MIDA scores 2-3×. See
# scripts/_evaluation/carepd_metrics.py and docs/appendix_evaluation_metrics.md.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from carepd_metrics import macro_f1_03, macro_f1_02, weighted_f1
from run_paths import output_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[3]
FEATURES_PATH = PROJECT_ROOT / "assets/datasets/fabricated_datasets/videomae_features_multilayer.pkl"
RESULTS_DIR   = output_root() / "results" / "mida" / "videomae"

DATASETS     = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]
LR_C         = 1.0      # fixed — same rationale as LOSO/LODO/CDA
RIDGE_ALPHA  = 1.0
RANDOM_STATE = 42
UPDRS_MIN, UPDRS_MAX = 0, 3
# ─────────────────────────────────────────────────────────────────────────────


def clip_and_round(x):
    return np.clip(np.round(x), UPDRS_MIN, UPDRS_MAX).astype(int)


def majority_vote(clip_preds, clip_labels, walk_ids):
    """Mode vote per walk. Ties broken by lowest label (conservative)."""
    walk_true, walk_pred = [], []
    for wid in np.unique(walk_ids):
        mask = walk_ids == wid
        values, counts = np.unique(clip_preds[mask], return_counts=True)
        walk_pred.append(int(values[np.argmax(counts)]))
        walk_true.append(int(clip_labels[mask][0]))
    return np.array(walk_true), np.array(walk_pred)


# macro_f1_excl3 removed — replaced by the shared carepd_metrics.macro_f1_02
# (present-class F1_{0-2}); see import block above.


def run_mida_dataset(
    X_target: np.ndarray,
    y_target: np.ndarray,
    groups_target: np.ndarray,
    walk_ids_target: np.ndarray,
    X_aux: np.ndarray,
    y_aux: np.ndarray,
    dataset: str,
) -> dict:
    """
    MIDA evaluation for one target dataset.

    Re-uses the EXACT LOSO fold structure (GroupKFold(n_splits=n_subjects)
    on the target dataset's subjects), but each fold's training set is the
    union of:
      - that fold's target-domain training subjects (LOSO train split), and
      - the FULL auxiliary pool (the other 3 datasets, all clips).
    Test set = the held-out target subject (identical to standard LOSO).

    This mirrors the paper's description verbatim: "train on target train
    split + auxiliary datasets, test on target test split... n and the
    computation are identical to LOSO".
    """
    n_subjects = len(np.unique(groups_target))
    log.info("── MIDA %s | %d target subjects | %d target clips + %d auxiliary clips | %d folds ──",
             dataset, n_subjects, len(y_target), len(y_aux), n_subjects)

    all_labels   = list(range(UPDRS_MIN, UPDRS_MAX + 1))
    gkf          = GroupKFold(n_splits=n_subjects)
    fold_results = []

    # Accumulate all out-of-fold walk-level predictions for the aggregate
    # confusion matrix AND for a single pooled Spearman correlation (see note
    # below the loop — per-fold Spearman is mathematically undefined in this
    # LOSO-on-target fold structure and was producing NaN for every fold,
    # exactly like in the *_loso.py scripts).
    all_walk_true = []
    all_walk_pred = []          # linear-probe classifier predictions (for F1/acc)
    all_walk_pred_ridge = []    # ridge ordinal predictions (for pooled MAE/QWK)
    all_walk_cont = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X_target, y_target, groups_target), start=1
    ):
        test_subject = groups_target[test_idx][0]

        # ── Augmented training pool: target LOSO-fold train ∪ full auxiliary ──
        X_train = np.concatenate([X_target[train_idx], X_aux], axis=0)
        y_train = np.concatenate([y_target[train_idx], y_aux], axis=0)
        X_test  = X_target[test_idx]
        y_test  = y_target[test_idx]

        # ── Linear probe (fixed C) ────────────────────────────────────────────
        linear_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=LR_C,
                max_iter=1000, tol=1e-3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
        linear_pipe.fit(X_train, y_train)
        pred_clips = linear_pipe.predict(X_test)
        walk_true, walk_pred = majority_vote(pred_clips, y_test, walk_ids_target[test_idx])

        # Per-fold values: present-class macro-F1 (kept only for the
        # between-subject SD reported alongside the pooled headline below).
        acc         = accuracy_score(walk_true, walk_pred)
        f1_macro    = macro_f1_03(walk_true, walk_pred)
        f1_macro02  = macro_f1_02(walk_true, walk_pred)
        f1_weighted = weighted_f1(walk_true, walk_pred)

        # ── Ridge probe (ordinal) ─────────────────────────────────────────────
        ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        ridge_pipe.fit(X_train, y_train)
        pred_cont  = ridge_pipe.predict(X_test)
        ridge_pred = clip_and_round(pred_cont)
        _, walk_pred_ridge = majority_vote(ridge_pred, y_test, walk_ids_target[test_idx])
        walk_cont = np.array([
            pred_cont[walk_ids_target[test_idx] == wid].mean()
            for wid in np.unique(walk_ids_target[test_idx])
        ])

        mae = mean_absolute_error(walk_true, walk_pred_ridge)
        try:
            qwk = cohen_kappa_score(walk_true, walk_pred_ridge, weights="quadratic")
        except ValueError:
            qwk = float("nan")
        # NOTE: per-fold Spearman is intentionally NOT computed here — see the
        # explanation below the loop for why (constant walk_true ⇒ NaN).

        all_walk_true.extend(walk_true.tolist())
        all_walk_pred.extend(walk_pred.tolist())
        all_walk_pred_ridge.extend(walk_pred_ridge.tolist())
        all_walk_cont.extend(walk_cont.tolist())

        fold_results.append({
            "fold":                 fold,
            "test_subject":         test_subject,
            "n_target_train_clips": int(len(train_idx)),
            "n_aux_train_clips":    int(len(X_aux)),
            "n_train_clips_total":  int(len(X_train)),
            "n_test_clips":         int(len(test_idx)),
            "n_test_walks":         int(len(walk_true)),
            "f1_macro_03":          round(f1_macro,    4),
            "f1_macro_02":          round(f1_macro02, 4) if f1_macro02 == f1_macro02 else None,
            "f1_weighted":          round(f1_weighted, 4),
            "accuracy":             round(acc,         4),
            "mae":                  round(mae,         4),
            "qwk":                  round(qwk, 4) if not np.isnan(qwk) else None,
            # no "spearman" key here on purpose (see note below the loop)
        })

    # ── Spearman: computed ONCE, pooled over every out-of-fold walk ──────────
    # GroupKFold(n_splits=n_subjects) on the target makes each fold's test set
    # ≈ one subject, so within a fold walk_true is CONSTANT (a single UPDRS
    # label) and spearmanr(walk_true, walk_cont) is mathematically undefined →
    # scipy raises ConstantInputWarning and returns NaN. Averaging n_subjects
    # NaNs then silently turned spearman_mean/_std into NaN for every dataset
    # (same root cause as in *_loso.py — MIDA re-runs the LOSO fold structure
    # on the target). Fix: pool all out-of-fold (walk_true, walk_cont) pairs
    # first — exactly like the aggregate confusion matrix just below — THEN
    # compute Spearman once over the pooled arrays, where labels vary across
    # subjects and the correlation is well-defined.
    if len(set(all_walk_true)) > 1:
        spearman_pooled, _ = spearmanr(all_walk_true, all_walk_cont)
        spearman_pooled = round(float(spearman_pooled), 4)
    else:
        spearman_pooled = None

    # ── Summary — Care-PD style: POOL every out-of-fold walk prediction, then
    #    compute ONE score over the pooled arrays (mirrors test.py's
    #    `last_report_allfolds` report). `{metric}_mean` holds the POOLED
    #    headline value; `{metric}_std` holds the between-subject SD of the
    #    per-fold values (Care-PD Appendix D.4 variability), NOT a fold mean.
    at = np.asarray(all_walk_true)
    ap = np.asarray(all_walk_pred)          # linear-probe classifier predictions
    ar = np.asarray(all_walk_pred_ridge)    # ridge ordinal predictions (MAE/QWK)

    pooled = {
        "f1_macro_03": macro_f1_03(at, ap),
        "f1_macro_02": macro_f1_02(at, ap),
        "f1_weighted": weighted_f1(at, ap),
        "accuracy":    accuracy_score(at, ap),
        "mae":         mean_absolute_error(at, ar),
        "qwk":         (cohen_kappa_score(at, ar, weights="quadratic")
                        if len(set(at.tolist())) > 1 else float("nan")),
    }

    def _fold_std(key):
        vals = [f[key] for f in fold_results if f[key] is not None]
        return round(float(np.std(vals)), 4) if vals else None

    summary = {}
    log.info("  %s MIDA summary — POOLED over %d out-of-fold walks (± between-subject SD):",
             dataset, len(all_walk_true))
    for metric, label in [
        ("f1_macro_03", "Macro F1 (0-3)"),
        ("f1_macro_02", "Macro F1 (0-2)"),
        ("f1_weighted", "Weighted F1   "),
        ("accuracy",    "Accuracy      "),
        ("mae",         "MAE           "),
        ("qwk",         "QWK           "),
    ]:
        val = pooled[metric]
        summary[f"{metric}_mean"] = round(float(val), 4) if val == val else None
        summary[f"{metric}_std"]  = _fold_std(metric)
        log.info("    %s: %s ± %s", label,
                 f"{val:.3f}" if val == val else "n/a",
                 f"{summary[f'{metric}_std']:.3f}"
                 if summary[f'{metric}_std'] is not None else "n/a")

    # Spearman is reported as a SINGLE pooled value (see note above), stored
    # under the same "{metric}_mean"/"_std" keys for schema compatibility with
    # the other metrics and with aggregate_results.py — but "_std" is None
    # because it isn't a fold-average (there is only one pooled value).
    summary["spearman_mean"] = spearman_pooled
    summary["spearman_std"]  = None
    log.info("    %s: %s  (pooled over all %d out-of-fold walks — not a fold-average)",
             "Spearman      ",
             f"{spearman_pooled:.3f}" if spearman_pooled is not None else "n/a",
             len(all_walk_true))

    cm = confusion_matrix(all_walk_true, all_walk_pred, labels=all_labels)
    cm_df = pd.DataFrame(
        cm,
        index  =[f"true_{l}" for l in all_labels],
        columns=[f"pred_{l}" for l in all_labels],
    )
    log.info("  Aggregate confusion matrix:\n%s", cm_df.to_string())

    return {
        "dataset":            dataset,
        "n_subjects":         n_subjects,
        "n_target_clips":     int(len(y_target)),
        "n_auxiliary_clips":  int(len(y_aux)),
        "n_folds":            n_subjects,
        "updrs_dist_target":  {int(k): int(v) for k, v in
                               zip(*np.unique(y_target, return_counts=True))},
        "folds":              fold_results,
        "summary":            summary,
        "cm_df":              cm_df,
    }


def main() -> None:
    log.info("Loading VideoMAE features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS})

    dataset_results = {}

    for dataset in DATASETS:
        target_rows = [r for r in records if r["dataset"] == dataset]
        aux_rows    = [r for r in records if r["dataset"] != dataset]

        if not target_rows:
            log.warning("No records for target dataset %s — skipping.", dataset)
            continue
        if not aux_rows:
            log.warning("No auxiliary records for target %s — skipping (MIDA "
                        "requires at least one other dataset).", dataset)
            continue

        X_target        = np.stack([r["feature"]    for r in target_rows])
        y_target        = np.array([r["UPDRS_GAIT"] for r in target_rows], dtype=int)
        groups_target   = np.array([r["subject"]    for r in target_rows])
        walk_ids_target = np.array([f"{r['subject']}_{r['trial']}" for r in target_rows])

        X_aux = np.stack([r["feature"]    for r in aux_rows])
        y_aux = np.array([r["UPDRS_GAIT"] for r in aux_rows], dtype=int)

        result = run_mida_dataset(
            X_target, y_target, groups_target, walk_ids_target,
            X_aux, y_aux, dataset,
        )
        dataset_results[dataset] = result

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "VideoMAE (MCG-NJU/videomae-base), frozen, layers 9-12",
        "evaluation": "MIDA — re-run LOSO on target dataset with each fold's "
                      "training set augmented by the full auxiliary pool "
                      "(the other 3 datasets). n and computation identical "
                      "to LOSO (Care-PD paper, sec. 4.2 / Appendix D.4).",
        "probe":      f"LogisticRegression (C={LR_C}, class_weight=balanced, fixed) "
                      f"+ Ridge (alpha={RIDGE_ALPHA})",
        "rationale": {
            "mida": "Matches Care-PD MIDA protocol verbatim: same LOSO folds "
                    "as within-dataset LOSO, training set = target train "
                    "split ∪ auxiliary datasets, test on target held-out "
                    "subjects.",
            "single_stage_fit": "The paper describes MIDA as 'fine-tuning "
                    "from a LODO checkpoint'. For our convex linear probes "
                    "(LogisticRegression/Ridge solved to convergence), the "
                    "converged solution is independent of initialisation, so "
                    "fitting directly on (target_train ∪ auxiliary) yields "
                    "the mathematically identical model — see module docstring.",
            "no_sample_weighting": "Each LOSO training fold already contains "
                    "~(n-1)/n of the target cohort — comparable in scale to "
                    "the auxiliary pool, so no domain-balancing weights are "
                    "needed; class_weight='balanced' (for UPDRS-label "
                    "imbalance) is applied as in LOSO/LODO/CDA.",
            "comparison_frame": "LODO (zero target supervision) vs. MIDA "
                    "(target + auxiliary) vs. LOSO (target-only) — identical "
                    "to the framing used in the Care-PD paper (Fig. 4, "
                    "Table D.2, Fig. D.9).",
        },
        "datasets": {
            ds: {k: v for k, v in res.items() if k != "cm_df"}
            for ds, res in dataset_results.items()
        },
    }

    json_path = RESULTS_DIR / f"results_videomae_mida_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", json_path)

    for ds, res in dataset_results.items():
        if res.get("cm_df") is not None:
            cm_path = RESULTS_DIR / f"results_videomae_mida_{ds}_cm_{run_ts}.csv"
            res["cm_df"].to_csv(cm_path)
            log.info("Confusion matrix (%s) saved to %s", ds, cm_path)

    # ── Cross-dataset summary ─────────────────────────────────────────────────
    log.info("\n===== MIDA SUMMARY ACROSS DATASETS =====")
    log.info("%-10s  %s  %s  %s  %s",
             "Dataset", "F1_0-3", "F1_0-2", "Acc   ", "QWK   ")
    for ds, res in dataset_results.items():
        s = res["summary"]
        log.info("%-10s  %.3f±%.3f  %.3f±%.3f  %.3f±%.3f  %.3f±%.3f",
                 ds,
                 s["f1_macro_03_mean"], s["f1_macro_03_std"],
                 s["f1_macro_02_mean"], s["f1_macro_02_std"],
                 s["accuracy_mean"],    s["accuracy_std"],
                 s["qwk_mean"],         s["qwk_std"])


if __name__ == "__main__":
    main()
