"""
VideoMAE — Leave-One-Dataset-Out (LODO) Evaluation
====================================================
Trains a linear probe on D-1 datasets, evaluates on the held-out dataset.
Repeats for all 4 datasets → 4 LODO folds.

Matches the Care-PD paper LODO protocol (sec. 4.2):
  "we train on the union of D-1 cohorts and evaluate on the held-out cohort"

Design choices
--------------
- Fixed C=1.0: same rationale as LOSO (lightweight probe, comparable to paper).
- Walk-level majority vote: same as LOSO.
- Metrics: Macro-F1 (0-3), Macro-F1 (0-2), Accuracy, MAE, QWK, Spearman.
  F1 (0-2) is the primary metric to match the Care-PD paper (label 3 is rare
  and absent in some datasets, e.g. T-SDU-PD only has 0/1/2).
- class_weight="balanced": corrects for UPDRS imbalance in training pool.
- BMClab note: _down0…_down4 variants produce identical video features after
  re-rendering at 30 FPS. This gives BMClab ~5× more training clips than other
  datasets. Consider deduplication if BMClab dominates training.

Usage
-----
    cd main_project
    python scripts/_evaluation/lodo/videomae_lodo.py
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
# Present-class macro-F1 (no forced absent classes). The held-out test set
# already contains several classes, so each fold's F1 is computed over its own
# pooled walks — the fix here is only to stop injecting absent UPDRS classes as
# F1=0 terms (esp. class 3 on datasets that lack it). See
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
RESULTS_DIR   = output_root() / "results" / "lodo" / "videomae"

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


# macro_f1_excl3 removed — replaced by the shared carepd_metrics.macro_f1_02
# (present-class F1_{0-2}); see import block above.


def run_lodo(records: list, datasets: list) -> dict:
    """
    For each held-out dataset: train on union of remaining, test on held-out.
    Returns dict with one entry per fold (= per held-out dataset).
    """
    all_labels   = list(range(UPDRS_MIN, UPDRS_MAX + 1))
    fold_results = []
    all_walk_true, all_walk_pred = [], []

    for held_out in datasets:
        train_rows = [r for r in records if r["dataset"] != held_out]
        test_rows  = [r for r in records if r["dataset"] == held_out]

        if not test_rows:
            log.warning("No records for held-out dataset %s — skipping.", held_out)
            continue

        X_train = np.stack([r["feature"]    for r in train_rows])
        y_train = np.array([r["UPDRS_GAIT"] for r in train_rows], dtype=int)
        X_test  = np.stack([r["feature"]    for r in test_rows])
        y_test  = np.array([r["UPDRS_GAIT"] for r in test_rows],  dtype=int)

        train_ds_counts = {ds: sum(1 for r in train_rows if r["dataset"] == ds)
                           for ds in datasets if ds != held_out}
        log.info("Fold: test=%-10s | train=%s | n_test=%d",
                 held_out, train_ds_counts, len(test_rows))

        walk_ids = np.array([f"{r['subject']}_{r['trial']}" for r in test_rows])

        # ── Linear probe ──────────────────────────────────────────────────────
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
        pred_clips            = linear_pipe.predict(X_test)
        walk_true, walk_pred  = majority_vote(pred_clips, y_test, walk_ids)

        acc        = accuracy_score(walk_true, walk_pred)
        f1_macro   = macro_f1_03(walk_true, walk_pred)
        f1_macro02 = macro_f1_02(walk_true, walk_pred)
        f1_w       = weighted_f1(walk_true, walk_pred)

        # ── Ridge probe (ordinal) ─────────────────────────────────────────────
        ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        ridge_pipe.fit(X_train, y_train)
        pred_cont  = ridge_pipe.predict(X_test)
        ridge_pred = clip_and_round(pred_cont)
        _, walk_pred_ridge = majority_vote(ridge_pred, y_test, walk_ids)
        walk_cont  = np.array([
            pred_cont[walk_ids == wid].mean()
            for wid in np.unique(walk_ids)
        ])

        mae = mean_absolute_error(walk_true, walk_pred_ridge)
        try:
            qwk = cohen_kappa_score(walk_true, walk_pred_ridge, weights="quadratic")
        except ValueError:
            qwk = float("nan")
        rho, _ = spearmanr(walk_true, walk_cont)

        all_walk_true.extend(walk_true.tolist())
        all_walk_pred.extend(walk_pred.tolist())

        log.info("  F1(0-3)=%.3f  F1(0-2)=%.3f  Acc=%.3f  MAE=%.3f  QWK=%.3f",
                 f1_macro, f1_macro02, acc, mae, qwk if not np.isnan(qwk) else -1)

        fold_results.append({
            "held_out":       held_out,
            "n_train_clips":  int(len(X_train)),
            "n_test_clips":   int(len(X_test)),
            "n_test_walks":   int(len(walk_true)),
            "train_datasets": train_ds_counts,
            "updrs_dist_test": {int(k): int(v) for k, v in
                                zip(*np.unique(y_test, return_counts=True))},
            "f1_macro_03":    round(f1_macro,    4),
            "f1_macro_02":    round(f1_macro02,  4),
            "f1_weighted":    round(f1_w,         4),
            "accuracy":       round(acc,          4),
            "mae":            round(mae,           4),
            "qwk":            round(qwk, 4) if not np.isnan(qwk) else None,
            "spearman":       round(float(rho),   4),
        })

    # ── Summary over all folds ────────────────────────────────────────────────
    def _ms(key):
        vals = [f[key] for f in fold_results if f[key] is not None]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    summary = {}
    log.info("\n===== LODO SUMMARY (mean ± std over %d folds) =====", len(fold_results))
    for metric, label in [
        ("f1_macro_03", "F1 (0-3)  "),
        ("f1_macro_02", "F1 (0-2)  "),
        ("f1_weighted", "F1 weighted"),
        ("accuracy",    "Accuracy  "),
        ("mae",         "MAE       "),
        ("qwk",         "QWK       "),
        ("spearman",    "Spearman  "),
    ]:
        m, s = _ms(metric)
        summary[f"{metric}_mean"] = m
        summary[f"{metric}_std"]  = s
        log.info("  %s: %.3f ± %.3f", label, m, s)

    cm = confusion_matrix(all_walk_true, all_walk_pred, labels=all_labels)
    cm_df = pd.DataFrame(
        cm,
        index  =[f"true_{l}" for l in all_labels],
        columns=[f"pred_{l}" for l in all_labels],
    )
    log.info("Aggregate confusion matrix:\n%s", cm_df.to_string())

    return {"folds": fold_results, "summary": summary, "cm_df": cm_df}


def main() -> None:
    log.info("Loading VideoMAE features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS})

    result = run_lodo(records, DATASETS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "VideoMAE (MCG-NJU/videomae-base), frozen, layers 9-12",
        "evaluation": "LODO — train on D-1 datasets, test on held-out",
        "probe":      f"LogisticRegression (C={LR_C}, class_weight=balanced)",
        "datasets":   DATASETS,
        "folds":      result["folds"],
        "summary":    result["summary"],
    }

    json_path = RESULTS_DIR / f"results_videomae_lodo_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved to %s", json_path)

    cm_path = RESULTS_DIR / f"results_videomae_lodo_cm_{run_ts}.csv"
    result["cm_df"].to_csv(cm_path)
    log.info("Confusion matrix saved to %s", cm_path)


if __name__ == "__main__":
    main()
