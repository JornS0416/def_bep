"""
V-JEPA 2 — Leave-One-Subject-Out (LOSO) Evaluation
==================================================
Evaluates frozen V-JEPA 2 ViT-L/16 mean-pool features using LOSO
cross-validation, applied per dataset separately.

Input:  vjepa2_features.pkl — "feature" field: (1024,) mean-pool per clip
Output: outputs/runs/<run>/results/loso/vjepa2/results_vjepa2_loso_{ts}.json

Structurally identical to videomae_loso.py. The only differences are:
  - Feature file and dimension (1024 vs 3072)
  - Results directory (vjepa vs videomae)

See videomae_loso.py for full rationale on:
  - Why LOSO over 5-fold
  - Why per dataset
  - Why fixed C

Usage:
    cd main_project
    python scripts/_evaluation/loso/vjepa_loso.py

Prerequisite: run scripts/vjepa/extract_features.py first.
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

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
FEATURES_PATH = PROJECT_ROOT / "assets/datasets/fabricated_datasets/vjepa2_features.pkl"
RESULTS_DIR   = output_root() / "results" / "loso" / "vjepa2"

DATASETS     = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]
LR_C         = 1.0
RIDGE_ALPHA  = 1.0
RANDOM_STATE = 42
UPDRS_MIN    = 0
UPDRS_MAX    = 3


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


def run_loso_dataset(X, y, groups, walk_ids, dataset):
    """LOSO for one dataset. See videomae_loso.py for full documentation."""
    n_subjects = len(np.unique(groups))
    log.info("── %s | %d subjects | %d clips | %d folds ──",
             dataset, n_subjects, len(y), n_subjects)

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
        y_train = y[train_idx]
        y_test  = y[test_idx]

        linear_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=LR_C, max_iter=1000, tol=1e-3,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
        )
        linear_pipe.fit(X[train_idx], y_train)
        pred_clips = linear_pipe.predict(X[test_idx])
        walk_true, walk_pred = majority_vote(pred_clips, y_test, walk_ids[test_idx])

        # Per-fold values: present-class macro-F1 (kept only for the
        # between-subject SD reported alongside the pooled headline below).
        acc        = accuracy_score(walk_true, walk_pred)
        f1_macro   = macro_f1_03(walk_true, walk_pred)
        f1_macro02 = macro_f1_02(walk_true, walk_pred)
        f1_weighted= weighted_f1(walk_true, walk_pred)

        ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        ridge_pipe.fit(X[train_idx], y_train)
        pred_cont  = ridge_pipe.predict(X[test_idx])
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
    log.info("  %s LOSO summary — POOLED over %d out-of-fold walks (± between-subject SD):",
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
    log.info("Loading V-JEPA 2 features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records", len(records))

    dataset_results = {}

    for dataset in DATASETS:
        rows = [r for r in records if r["dataset"] == dataset]
        if not rows:
            log.warning("No records for %s — skipping.", dataset)
            continue

        X        = np.stack([r["feature"]    for r in rows])
        y        = np.array([r["UPDRS_GAIT"] for r in rows], dtype=int)
        groups   = np.array([r["subject"]    for r in rows])
        walk_ids = np.array([f"{r['subject']}_{r['trial']}" for r in rows])

        dataset_results[dataset] = run_loso_dataset(X, y, groups, walk_ids, dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "V-JEPA 2 ViT-L/16 (facebookresearch/jepa), frozen",
        "evaluation": "LOSO per dataset — GroupKFold(n_splits=n_subjects)",
        "probe":      f"LogisticRegression (C={LR_C}, class_weight=balanced, fixed)",
        "rationale": {
            "loso":      "Matches Care-PD paper within-dataset evaluation protocol.",
            "per_dataset":"Avoids conflating within- and cross-dataset generalisation.",
            "fixed_C":   f"C={LR_C} fixed — see videomae_loso.py for full rationale.",
        },
        "datasets": {
            ds: {k: v for k, v in res.items() if k != "cm_df"}
            for ds, res in dataset_results.items()
        },
    }

    json_path = RESULTS_DIR / f"results_vjepa_loso_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", json_path)

    for ds, res in dataset_results.items():
        if res.get("cm_df") is not None:
            cm_path = RESULTS_DIR / f"results_vjepa_loso_{ds}_cm_{run_ts}.csv"
            res["cm_df"].to_csv(cm_path)

    log.info("\n===== LOSO SUMMARY ACROSS DATASETS =====")
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
