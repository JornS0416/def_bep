"""
V-JEPA — Cross-Dataset Analysis (CDA)
======================================
Structurally identical to videomae_cda.py.
Only differences: feature file, embed dim (1024 vs 3072), results dir.

See videomae_cda.py for full rationale (why CDA, why no diagonal, why a
single train/test split per pair, why the probe is fit once per source, etc.)

Usage
-----
    cd main_project
    python scripts/_evaluation/cda/vjepa_cda.py
"""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    cohen_kappa_score, confusion_matrix,
)
from scipy.stats import spearmanr

# ── Care-PD–faithful metrics (shared module) ──────────────────────────────────
# Present-class macro-F1 (no forced absent classes); each source→target cell is
# a whole target dataset, so the fix is only to stop injecting absent UPDRS
# classes (esp. class 3) as F1=0 terms. See scripts/_evaluation/carepd_metrics.py
# and docs/appendix_evaluation_metrics.md.
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
FEATURES_PATH = PROJECT_ROOT / "assets/datasets/fabricated_datasets/vjepa_features.pkl"
RESULTS_DIR   = output_root() / "results" / "cda" / "vjepa"

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


def evaluate_pair(linear_pipe, ridge_pipe, X_test, y_test, walk_ids):
    """Evaluate a fitted (linear, ridge) probe pair on one target dataset."""
    all_labels = list(range(UPDRS_MIN, UPDRS_MAX + 1))

    pred_clips           = linear_pipe.predict(X_test)
    walk_true, walk_pred = majority_vote(pred_clips, y_test, walk_ids)

    acc        = accuracy_score(walk_true, walk_pred)
    f1_macro   = macro_f1_03(walk_true, walk_pred)
    f1_macro02 = macro_f1_02(walk_true, walk_pred)
    f1_w       = weighted_f1(walk_true, walk_pred)

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

    cm = confusion_matrix(walk_true, walk_pred, labels=all_labels)

    return {
        "n_test_clips":    int(len(X_test)),
        "n_test_walks":    int(len(walk_true)),
        "updrs_dist_test": {int(k): int(v) for k, v in
                            zip(*np.unique(y_test, return_counts=True))},
        "f1_macro_03":     round(f1_macro,   4),
        "f1_macro_02":     round(f1_macro02, 4),
        "f1_weighted":     round(f1_w,       4),
        "accuracy":        round(acc,        4),
        "mae":             round(mae,        4),
        "qwk":             round(qwk, 4) if not np.isnan(qwk) else None,
        "spearman":        round(float(rho), 4),
        "cm":              cm,
    }


def run_cda(records: list, datasets: list) -> dict:
    """
    For each ordered (source, target) pair with source != target:
      - fit a probe on the FULL source dataset (once per source)
      - evaluate on the FULL target dataset
    Returns the 12 pair results plus per-metric source × target matrices.
    """
    pair_results = []

    metrics = ["f1_macro_03", "f1_macro_02", "f1_weighted",
               "accuracy", "mae", "qwk", "spearman"]
    matrices = {m: pd.DataFrame(np.nan, index=datasets, columns=datasets)
                for m in metrics}

    for source in datasets:
        train_rows = [r for r in records if r["dataset"] == source]
        if not train_rows:
            log.warning("No records for source dataset %s — skipping.", source)
            continue

        X_train = np.stack([r["feature"]    for r in train_rows])
        y_train = np.array([r["UPDRS_GAIT"] for r in train_rows], dtype=int)

        log.info("Source: %-10s | n_train_clips=%d | dist=%s",
                 source, len(X_train),
                 {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})

        linear_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=LR_C, max_iter=1000, tol=1e-3,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
        )
        linear_pipe.fit(X_train, y_train)

        ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        ridge_pipe.fit(X_train, y_train)

        for target in datasets:
            if target == source:
                continue

            test_rows = [r for r in records if r["dataset"] == target]
            if not test_rows:
                log.warning("No records for target dataset %s — skipping.", target)
                continue

            X_test   = np.stack([r["feature"]    for r in test_rows])
            y_test   = np.array([r["UPDRS_GAIT"] for r in test_rows], dtype=int)
            walk_ids = np.array([f"{r['subject']}_{r['trial']}" for r in test_rows])

            res = evaluate_pair(linear_pipe, ridge_pipe, X_test, y_test, walk_ids)
            cm  = res.pop("cm")

            log.info("  %-10s → %-10s | F1(0-3)=%.3f  F1(0-2)=%.3f  Acc=%.3f  MAE=%.3f  QWK=%s",
                     source, target, res["f1_macro_03"], res["f1_macro_02"],
                     res["accuracy"], res["mae"],
                     f"{res['qwk']:.3f}" if res["qwk"] is not None else "nan")

            for m in metrics:
                matrices[m].loc[source, target] = res[m]

            pair_results.append({
                "source": source,
                "target": target,
                "n_train_clips": int(len(X_train)),
                **res,
                "cm": cm.tolist(),
            })

    return {"pairs": pair_results, "matrices": matrices}


def main() -> None:
    log.info("Loading V-JEPA features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS})

    result = run_cda(records, DATASETS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "V-JEPA (ViT-L/16), frozen, mean-pool over 1568 tokens",
        "evaluation": "CDA — train on single source dataset, evaluate on each "
                      "other dataset separately (12 source→target pairs). "
                      "Diagonal (in-domain) intentionally omitted — use LOSO "
                      "results for that; see videomae_cda.py docstring for rationale.",
        "probe":      f"LogisticRegression (C={LR_C}, class_weight=balanced) "
                      f"+ Ridge (alpha={RIDGE_ALPHA})",
        "datasets":   DATASETS,
        "pairs":      result["pairs"],
    }

    json_path = RESULTS_DIR / f"results_vjepa_cda_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved to %s", json_path)

    for metric, mat in result["matrices"].items():
        mat_path = RESULTS_DIR / f"results_vjepa_cda_matrix_{metric}_{run_ts}.csv"
        mat.to_csv(mat_path)
    log.info("Source × target matrices (one CSV per metric) saved to %s", RESULTS_DIR)

    log.info("\n===== CDA — F1 (0-2) transfer matrix (rows=source, cols=target) =====\n%s",
             result["matrices"]["f1_macro_02"].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
