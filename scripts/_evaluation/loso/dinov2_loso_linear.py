"""
DINOv2 + Linear Probe — Leave-One-Subject-Out (LOSO) Evaluation   ["opposite" arm]
===================================================================================
Companion / contrast script to dinov2_loso.py.

dinov2_loso.py evaluates DINOv2 within-dataset using a BiLSTM+Attention over
the full (81, 768) per-clip sequence (with per-dataset HP search, "Option C").
This script runs the EXACT SAME within-dataset LOSO protocol, but with the
"opposite" architecture: mean-pool (81, 768) → (768,) and fit a FIXED linear
probe (LogisticRegression + Ridge) — identical to the probe used by
videomae_loso.py / vjepa_loso.py, and to DINOv2's own cross-cohort scripts
(dinov2_lodo.py / dinov2_cda.py / dinov2_mida.py).

─────────────────────────────────────────────────────────────────────────────
WHY BUILD THIS — WHAT IT TESTS
─────────────────────────────────────────────────────────────────────────────
We argued (see dinov2_lodo.py / dinov2_cda.py / dinov2_mida.py docstrings)
that:
  - WITHIN one cohort, temporal position is consistent enough for a BiLSTM
    to exploit → BiLSTM is the right choice for dinov2_loso.py.
  - ACROSS cohorts, temporal position means different things in different
    cohorts (different cadence/FPS/recording setups) → a BiLSTM trained on
    a mixed pool cannot learn one coherent temporal mapping, so we use a
    linear probe on pooled features for LODO/CDA/MIDA instead.

That argument makes a TESTABLE PREDICTION: within one cohort, the BiLSTM
(which can exploit genuine temporal dynamics) should be competitive with —
or outperform — a linear probe on mean-pooled features (which discards all
temporal information). Running the linear probe *within* a cohort gives us
the missing reference point to confirm or challenge that prediction
empirically, instead of resting on the theoretical argument alone.

This is the direct counterpart of:
    dinov2_lodo_bilstm.py  — runs the BiLSTM where we said it should struggle
    dinov2_cda_bilstm.py   — idem
    dinov2_mida_bilstm.py  — idem
Together, the four scripts let you build a 2×4 comparison grid
(architecture × evaluation protocol) for DINOv2.

─────────────────────────────────────────────────────────────────────────────
DESIGN — KEPT IDENTICAL TO THE RELEVANT REFERENCE SCRIPTS
─────────────────────────────────────────────────────────────────────────────
- Fold structure & metrics: identical to dinov2_loso.py's run_loso_dataset
  (GroupKFold(n_splits=n_subjects), per dataset, walk-level majority vote,
  same metric set, aggregate confusion matrix).
- Probe: identical to videomae_loso.py / vjepa_loso.py — FIXED
  LogisticRegression(C=1.0, class_weight="balanced") + Ridge(alpha=1.0) on
  StandardScaler-ed features. NO hyperparameter search: a linear probe with
  fixed L2 regularisation is far less HP-sensitive than a BiLSTM (this is
  exactly the asymmetry that justified Option C for the BiLSTM but a fixed
  C/alpha for VideoMAE/V-JEPA — see the LOSO-comparison discussion). Using
  the same fixed configuration here keeps this script directly comparable
  to those two models' LOSO runs as well.
- Feature aggregation: mean-pool (81, 768) → (768,), identical to
  dinov2_lodo.py / dinov2_cda.py / dinov2_mida.py.

Usage
-----
    cd main_project
    python scripts/_evaluation/loso/dinov2_loso_linear.py
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
# that deflated within-dataset LOSO scores 2-3×. See
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
FEATURES_PATH = PROJECT_ROOT / "assets/datasets/fabricated_datasets/dinov2_features_81f.pkl"
RESULTS_DIR   = output_root() / "results" / "loso" / "dinov2_linear"

DATASETS     = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]
LR_C         = 1.0
RIDGE_ALPHA  = 1.0
RANDOM_STATE = 42
UPDRS_MIN, UPDRS_MAX = 0, 3
# ─────────────────────────────────────────────────────────────────────────────


def clip_and_round(x):
    return np.clip(np.round(x), UPDRS_MIN, UPDRS_MAX).astype(int)


def majority_vote(clip_preds, clip_labels, walk_ids):
    walk_true, walk_pred = [], []
    for wid in np.unique(walk_ids):
        mask = walk_ids == wid
        values, counts = np.unique(clip_preds[mask], return_counts=True)
        walk_pred.append(int(values[np.argmax(counts)]))
        walk_true.append(int(clip_labels[mask][0]))
    return np.array(walk_true), np.array(walk_pred)


def run_loso_dataset_linear(X, y, groups, walk_ids, dataset):
    """
    Plain within-dataset LOSO on mean-pooled (768,) features with a FIXED
    linear probe. Same fold structure / metrics as dinov2_loso.py's
    run_loso_dataset — only the model differs (linear probe, no HP search).
    """
    n_subjects = len(np.unique(groups))
    log.info("── %s LOSO (linear probe) | %d subjects | %d clips ──",
             dataset, n_subjects, len(y))

    all_labels   = list(range(UPDRS_MIN, UPDRS_MAX + 1))
    gkf          = GroupKFold(n_splits=n_subjects)
    fold_results = []
    # Accumulate all out-of-fold walk-level predictions for the aggregate
    # confusion matrix AND for a single pooled Spearman correlation (see note
    # below the loop — per-fold Spearman is mathematically undefined in LOSO
    # and was producing NaN for every fold).
    all_walk_true, all_walk_pred, all_walk_cont = [], [], []
    all_walk_pred_ridge = []    # ridge ordinal predictions (for pooled MAE/QWK)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups), start=1
    ):
        test_subject = groups[test_idx][0]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        linear_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=LR_C, max_iter=1000, tol=1e-3,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
        )
        linear_pipe.fit(X_train, y_train)
        pred_clips = linear_pipe.predict(X_test)
        walk_true, walk_pred = majority_vote(pred_clips, y_test, walk_ids[test_idx])

        # Per-fold values: present-class macro-F1 (kept only for the
        # between-subject SD reported alongside the pooled headline below).
        acc         = accuracy_score(walk_true, walk_pred)
        f1_macro    = macro_f1_03(walk_true, walk_pred)
        f1_macro02  = macro_f1_02(walk_true, walk_pred)
        f1_weighted = weighted_f1(walk_true, walk_pred)

        ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        ridge_pipe.fit(X_train, y_train)
        pred_cont  = ridge_pipe.predict(X_test)
        ridge_pred = clip_and_round(pred_cont)
        _, walk_pred_ridge = majority_vote(ridge_pred, y_test, walk_ids[test_idx])
        walk_cont = np.array([
            pred_cont[walk_ids[test_idx] == wid].mean()
            for wid in np.unique(walk_ids[test_idx])
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
            "fold":          fold,
            "test_subject":  test_subject,
            "n_train_clips": int(len(train_idx)),
            "n_test_clips":  int(len(test_idx)),
            "n_test_walks":  int(len(walk_true)),
            "f1_macro_03":   round(f1_macro,    4),
            "f1_macro_02":   round(f1_macro02, 4) if f1_macro02 == f1_macro02 else None,
            "f1_weighted":   round(f1_weighted, 4),
            "accuracy":      round(acc,         4),
            "mae":           round(mae,         4),
            "qwk":           round(qwk, 4) if not np.isnan(qwk) else None,
            # no "spearman" key here on purpose (see note below the loop)
        })

    # ── Spearman: computed ONCE, pooled over every out-of-fold walk ──────────
    # GroupKFold(n_splits=n_subjects) makes each fold's test set ≈ one subject,
    # so within a fold walk_true is CONSTANT (a single UPDRS label) and
    # spearmanr(walk_true, walk_cont) is mathematically undefined → scipy
    # raises ConstantInputWarning and returns NaN. Averaging n_subjects NaNs
    # then silently turned spearman_mean/_std into NaN for every dataset.
    # Fix: pool all out-of-fold (walk_true, walk_cont) pairs first — exactly
    # like the aggregate confusion matrix just below — THEN compute Spearman
    # once over the pooled arrays, where labels vary across subjects and the
    # correlation is well-defined.
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
    log.info("  %s LOSO (linear) summary — POOLED over %d out-of-fold walks (± between-subject SD):",
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
        "dataset":    dataset,
        "n_subjects": n_subjects,
        "n_clips":    int(len(y)),
        "n_folds":    n_subjects,
        "updrs_dist": {int(k): int(v) for k, v in
                       zip(*np.unique(y, return_counts=True))},
        "folds":      fold_results,
        "summary":    summary,
        "cm_df":      cm_df,
    }


def main() -> None:
    log.info("Loading DINOv2 features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS})

    dataset_results = {}

    for dataset in DATASETS:
        rows = [r for r in records if r["dataset"] == dataset]
        if not rows:
            log.warning("No records for %s — skipping.", dataset)
            continue

        # Mean-pool (81, 768) → (768,) for the linear probe
        X        = np.stack([r["feature"].mean(axis=0) for r in rows])
        y        = np.array([r["UPDRS_GAIT"] for r in rows], dtype=int)
        groups   = np.array([r["subject"]    for r in rows])
        walk_ids = np.array([f"{r['subject']}_{r['trial']}" for r in rows])

        dataset_results[dataset] = run_loso_dataset_linear(
            X, y, groups, walk_ids, dataset
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "DINOv2 (facebook/dinov2-base), frozen, mean-pool over 81 CLS tokens",
        "evaluation": "LOSO per dataset — GroupKFold(n_splits=n_subjects). "
                      "'Opposite-architecture' counterpart to dinov2_loso.py: "
                      "linear probe on mean-pooled features instead of "
                      "BiLSTM+Attention on the full sequence. See module "
                      "docstring for what this comparison is meant to test.",
        "probe":      f"LogisticRegression (C={LR_C}, class_weight=balanced, fixed) "
                      f"+ Ridge (alpha={RIDGE_ALPHA}, fixed) — NO per-dataset HP search "
                      f"(unlike dinov2_loso.py's Option C; a linear probe is far less "
                      f"HP-sensitive, and a fixed config matches videomae_loso.py / "
                      f"vjepa_loso.py).",
        "note":       "Feature aggregation: mean over (81, 768) → (768,), "
                      "identical to dinov2_lodo.py / dinov2_cda.py / dinov2_mida.py. "
                      "Compare against dinov2_loso.py (BiLSTM, same folds, same "
                      "dataset) to test whether the BiLSTM's extra temporal "
                      "modelling capacity actually pays off within a single cohort.",
        "datasets": {
            ds: {k: v for k, v in res.items() if k != "cm_df"}
            for ds, res in dataset_results.items()
        },
    }

    json_path = RESULTS_DIR / f"results_dinov2_loso_linear_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", json_path)

    for ds, res in dataset_results.items():
        if res.get("cm_df") is not None:
            cm_path = RESULTS_DIR / f"results_dinov2_loso_linear_{ds}_cm_{run_ts}.csv"
            res["cm_df"].to_csv(cm_path)
            log.info("Confusion matrix (%s) saved to %s", ds, cm_path)

    log.info("\n===== LOSO (linear probe) SUMMARY ACROSS DATASETS =====")
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
