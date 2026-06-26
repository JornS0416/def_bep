"""
DINOv2 + BiLSTM+Attention — Multi-dataset In-domain Adaptation (MIDA)  ["opposite" arm]
========================================================================================
Companion / contrast script to dinov2_mida.py.

dinov2_mida.py mean-pools (81, 768) → (768,) and uses a linear probe, for the
same cross-cohort reasons as dinov2_lodo.py / dinov2_cda.py. This script runs
the "opposite" setup: re-run the within-dataset LOSO fold structure on the
target dataset, but train a BiLSTM+Attention DIRECTLY on the full (81, 768)
sequences of {target LOSO-fold training subjects} ∪ {full auxiliary pool of
the other 3 datasets} — no pooling — and test on the held-out target subject.

─────────────────────────────────────────────────────────────────────────────
WHY THIS IS THE MOST DEMANDING OF THE THREE BiLSTM CROSS-COHORT EXPERIMENTS
─────────────────────────────────────────────────────────────────────────────
It stacks BOTH stress factors we identified, simultaneously:

  1. A heterogeneous, mixed-cohort training pool — target-train ∪ the full
     auxiliary pool of the other 3 datasets — the same issue probed by
     dinov2_lodo_bilstm.py (different cadence/FPS/recording setups ⇒ "frame
     index t" means something different per source cohort).
  2. Per-fold retraining — n_subjects folds per target dataset, each with
     its own (large, heterogeneous) pool — the same cost structure as
     dinov2_loso.py's LOSO, but now on a substantially larger and more
     heterogeneous pool each time (target-train ∪ auxiliary, vs. just
     target-train).

If dinov2_lodo_bilstm.py shows the BiLSTM struggling with mixed pools, this
script shows whether that problem gets WORSE when the mixed pool also
includes a sizeable slice of the target cohort itself (as MIDA's pool does)
— i.e., whether "more in-domain data, but packaged inside a noisier mixed
pool" helps, hurts, or roughly cancels out, relative to LODO.

⚠⚠ RUNTIME WARNING — READ BEFORE RUNNING ⚠⚠
This is, by a wide margin, the most expensive script in the entire evaluation
suite: n_subjects folds × (training one BiLSTM, up to 100 epochs with early
stopping, on a pool of MANY THOUSANDS of heterogeneous clips). For PD-GaM or
BMClab (dozens of subjects) this could plausibly take many hours to >1 day on
a single machine. STRONGLY consider:
  - running it first on the smallest datasets (3DGait, T-SDU-PD) as a
    sanity check / rough signal,
  - or temporarily lowering NUM_EPOCHS / EARLY_STOP_PATIENCE for a smoke test,
  - or running it overnight / over a weekend for the larger cohorts.

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETERS & EARLY STOPPING
─────────────────────────────────────────────────────────────────────────────
Fixed DEFAULT_HP (hidden_size=256, dropout=0.3, lr=1e-3 — identical to the
dinov2_loso.py 3DGait fallback / train_lstm.py defaults / the other two
"opposite-arm" BiLSTM scripts) for every fold of every dataset. A per-fold
search here would multiply an already very expensive computation by ~8× and
would, again, be confounded by the heterogeneity it's meant to measure the
impact of. Using one fixed configuration also keeps this comparable to
dinov2_mida.py's linear probe (fixed C=1.0 / alpha=1.0 everywhere).

Early stopping uses the held-out target-test fold as the validation set —
identical to dinov2_loso.py's (and dinov2_lodo_bilstm.py's) approach, and the
established pattern for LOSO-style per-fold training in this codebase. This
keeps the fold structure "n and computation identical to LOSO", exactly as
the Care-PD paper specifies for MIDA (see dinov2_mida.py's docstring for the
verbatim quote and the full MIDA rationale).

Usage
-----
    cd main_project
    python scripts/_evaluation/mida/dinov2_mida_bilstm.py
"""

import json
import pickle
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    cohen_kappa_score, confusion_matrix,
)
from scipy.stats import spearmanr
from tqdm import tqdm

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
FEATURES_PATH = PROJECT_ROOT / "assets/datasets/fabricated_datasets/dinov2_features_81f.pkl"
RESULTS_DIR   = output_root() / "results" / "mida" / "dinov2_bilstm"

DATASETS = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]

# Fixed HP — NOT searched per fold (see module docstring for why)
DEFAULT_HP = {"hidden_size": 256, "dropout": 0.3, "learning_rate": 1e-3}

INPUT_SIZE  = 768
NUM_CLASSES = 4
NUM_LAYERS  = 1
BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR_PATIENCE = 10
LR_FACTOR   = 0.5
EARLY_STOP_PATIENCE = 15

RANDOM_STATE = 42
# Subject-disjoint fraction of the TARGET training fold held out for leak-free
# early-stopping checkpoint selection (never the held-out test subject).
VAL_FRACTION = 0.15
UPDRS_MIN, UPDRS_MAX = 0, 3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset / Model / utilities — identical to dinov2_loso.py ────────────────

class GaitSeqDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BiLSTMAttention(nn.Module):
    """BiLSTM + temporal attention classifier — identical to dinov2_loso.py."""
    def __init__(self, hidden_size=256, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE, hidden_size=hidden_size,
            num_layers=NUM_LAYERS, batch_first=True,
            bidirectional=True,
            dropout=dropout if NUM_LAYERS > 1 else 0.0,
        )
        self.attn       = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        embed = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(self.dropout(embed))


def compute_class_weights(labels):
    counts = Counter(labels.tolist())
    n, k   = len(labels), NUM_CLASSES
    return torch.tensor(
        [n / (k * counts.get(c, 1)) for c in range(k)],
        dtype=torch.float32, device=DEVICE,
    )


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


def train_one_epoch(model, loader, criterion, optimiser):
    model.train()
    total = 0.0
    for seqs, labels in loader:
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(seqs), labels)
        loss.backward()
        optimiser.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, probs, labels_out = [], [], []
    for seqs, labels in loader:
        seqs = seqs.to(DEVICE)
        logits = model(seqs)
        preds.append(logits.argmax(-1).cpu().numpy())
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        labels_out.append(labels.numpy())
    return (
        np.concatenate(preds),
        np.concatenate(probs),
        np.concatenate(labels_out),
    )


def train_and_eval(train_feats, train_labels, val_feats, val_labels,
                   hidden_size, dropout, learning_rate, tag=""):
    """Train BiLSTM+Attention, early-stop on val loss. Identical to dinov2_loso.py."""
    tr_ds  = GaitSeqDataset(train_feats, train_labels)
    val_ds = GaitSeqDataset(val_feats,   val_labels)
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMAttention(hidden_size=hidden_size, dropout=dropout).to(DEVICE)
    wts   = compute_class_weights(np.array(train_labels))
    crit  = nn.CrossEntropyLoss(weight=wts)
    opt   = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc=tag, leave=False):
        loss = train_one_epoch(model, tr_loader, crit, opt)
        sched.step(loss)

        val_preds, _, val_labels_arr = predict(model, val_loader)
        val_loss = float(nn.CrossEntropyLoss()(
            torch.tensor(
                [[float(p == c) for c in range(NUM_CLASSES)] for p in val_preds],
                dtype=torch.float32,
            ),
            torch.tensor(val_labels_arr, dtype=torch.long),
        ))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    val_preds, _, val_labels_arr = predict(model, val_loader)
    val_f1 = f1_score(val_labels_arr, val_preds, average="macro",
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    return model, val_f1


# ── MIDA with BiLSTM ──────────────────────────────────────────────────────────

def run_mida_dataset_bilstm(
    X_target_list, y_target, groups_target, walk_ids_target,
    X_aux_list, y_aux, dataset,
):
    """
    MIDA for one target dataset, training a BiLSTM+Attention on
    {target LOSO-fold train} ∪ {full auxiliary pool} — full (81,768)
    sequences, no pooling. Same fold structure / metrics as
    dinov2_mida.py's run_mida_dataset; only the model differs.
    """
    n_subjects = len(np.unique(groups_target))
    log.info("── MIDA-BiLSTM %s | %d target subjects | %d target clips + "
             "%d auxiliary clips | %d folds | HP=%s ──",
             dataset, n_subjects, len(y_target), len(y_aux), n_subjects, DEFAULT_HP)

    all_labels   = list(range(UPDRS_MIN, UPDRS_MAX + 1))
    gkf          = GroupKFold(n_splits=n_subjects)
    fold_results = []
    # Accumulate all out-of-fold walk-level predictions for the aggregate
    # confusion matrix AND for a single pooled Spearman correlation (see note
    # below the loop — per-fold Spearman is mathematically undefined in this
    # LOSO-on-target fold structure and was producing NaN for every fold,
    # exactly like in the *_loso.py / other *_mida.py scripts).
    all_walk_true, all_walk_pred, all_walk_cont = [], [], []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(y_target)), y_target, groups_target), start=1
    ):
        test_subject = groups_target[test_idx][0]

        # ── Leak-free early stopping ──────────────────────────────────────────
        # Select the checkpoint on held-out TARGET subjects (never the test
        # subject); keep ALL auxiliary data in training. GroupShuffleSplit on the
        # target training fold's subjects guarantees subject-disjointness.
        es_tr, es_val = next(GroupShuffleSplit(
            n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_STATE
        ).split(np.arange(len(train_idx)), y_target[train_idx], groups_target[train_idx]))
        tgt_tr_idx  = train_idx[es_tr]
        tgt_val_idx = train_idx[es_val]

        train_feats  = [X_target_list[i] for i in tgt_tr_idx] + list(X_aux_list)
        train_labels = y_target[tgt_tr_idx].tolist() + y_aux.tolist()
        val_feats    = [X_target_list[i] for i in tgt_val_idx]
        val_labels   = y_target[tgt_val_idx].tolist()
        test_feats   = [X_target_list[i] for i in test_idx]
        test_labels  = y_target[test_idx]

        model, _ = train_and_eval(
            train_feats, train_labels, val_feats, val_labels,
            hidden_size=DEFAULT_HP["hidden_size"],
            dropout=DEFAULT_HP["dropout"],
            learning_rate=DEFAULT_HP["learning_rate"],
            tag=f"MIDA-BiLSTM {dataset} fold {fold}/{n_subjects}",
        )

        test_loader = DataLoader(
            GaitSeqDataset(test_feats, test_labels.tolist()),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        clip_preds, clip_probs, _ = predict(model, test_loader)
        del model

        walk_true, walk_pred = majority_vote(
            clip_preds, test_labels, walk_ids_target[test_idx]
        )

        # Per-fold values: present-class macro-F1 (kept only for the
        # between-subject SD reported alongside the pooled headline below).
        acc         = accuracy_score(walk_true, walk_pred)
        f1_macro    = macro_f1_03(walk_true, walk_pred)
        f1_macro02  = macro_f1_02(walk_true, walk_pred)
        f1_weighted = weighted_f1(walk_true, walk_pred)

        expected  = (clip_probs * np.array([0, 1, 2, 3])).sum(axis=1)
        walk_cont = np.array([
            expected[walk_ids_target[test_idx] == wid].mean()
            for wid in np.unique(walk_ids_target[test_idx])
        ])

        mae = mean_absolute_error(walk_true, walk_pred)
        try:
            qwk = cohen_kappa_score(walk_true, walk_pred, weights="quadratic")
        except ValueError:
            qwk = float("nan")
        # NOTE: per-fold Spearman is intentionally NOT computed here — see the
        # explanation below the loop for why (constant walk_true ⇒ NaN).

        all_walk_true.extend(walk_true.tolist())
        all_walk_pred.extend(walk_pred.tolist())
        all_walk_cont.extend(walk_cont.tolist())

        fold_results.append({
            "fold":                 fold,
            "test_subject":         test_subject,
            "n_target_train_clips": int(len(train_idx)),
            "n_aux_train_clips":    int(len(X_aux_list)),
            "n_train_clips_total":  int(len(train_feats)),
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
    # (same root cause as in *_loso.py / the other *_mida.py scripts). Fix:
    # pool all out-of-fold (walk_true, walk_cont) pairs first — exactly like
    # the aggregate confusion matrix below — THEN compute Spearman once over
    # the pooled arrays, where labels vary across subjects and the correlation
    # is well-defined.
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
    #    This BiLSTM variant has no ridge probe, so MAE/QWK pool the same
    #    walk-level classifier predictions used for F1.
    at = np.asarray(all_walk_true)
    ap = np.asarray(all_walk_pred)

    pooled = {
        "f1_macro_03": macro_f1_03(at, ap),
        "f1_macro_02": macro_f1_02(at, ap),
        "f1_weighted": weighted_f1(at, ap),
        "accuracy":    accuracy_score(at, ap),
        "mae":         mean_absolute_error(at, ap),
        "qwk":         (cohen_kappa_score(at, ap, weights="quadratic")
                        if len(set(at.tolist())) > 1 else float("nan")),
    }

    def _fold_std(key):
        vals = [f[key] for f in fold_results if f[key] is not None]
        return round(float(np.std(vals)), 4) if vals else None

    summary = {}
    log.info("  %s MIDA-BiLSTM summary — POOLED over %d out-of-fold walks (± between-subject SD):",
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
        "dataset":           dataset,
        "n_subjects":        n_subjects,
        "n_target_clips":    int(len(y_target)),
        "n_auxiliary_clips": int(len(y_aux)),
        "n_folds":           n_subjects,
        "updrs_dist_target": {int(k): int(v) for k, v in
                              zip(*np.unique(y_target, return_counts=True))},
        "folds":             fold_results,
        "summary":           summary,
        "cm_df":             cm_df,
    }


def main() -> None:
    log.info("Loading DINOv2 features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s | device: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS},
             DEVICE)
    log.info("⚠ This is the most expensive script in the suite — see module "
             "docstring's RUNTIME WARNING before committing to a full run.")

    dataset_results = {}

    for dataset in DATASETS:
        target_rows = [r for r in records if r["dataset"] == dataset]
        aux_rows    = [r for r in records if r["dataset"] != dataset]

        if not target_rows or not aux_rows:
            log.warning("Skipping %s — missing target or auxiliary rows.", dataset)
            continue

        X_target_list   = [r["feature"]    for r in target_rows]
        y_target        = np.array([r["UPDRS_GAIT"] for r in target_rows], dtype=int)
        groups_target   = np.array([r["subject"]    for r in target_rows])
        walk_ids_target = np.array([f"{r['subject']}_{r['trial']}" for r in target_rows])

        X_aux_list = [r["feature"]    for r in aux_rows]
        y_aux      = np.array([r["UPDRS_GAIT"] for r in aux_rows], dtype=int)

        dataset_results[dataset] = run_mida_dataset_bilstm(
            X_target_list, y_target, groups_target, walk_ids_target,
            X_aux_list, y_aux, dataset,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "DINOv2 ViT-B/14 (facebook/dinov2-base), frozen + BiLSTM+Attention",
        "evaluation": "MIDA — re-run LOSO on target dataset with each fold's "
                      "training set augmented by the full auxiliary pool (full "
                      "(81,768) sequences, NO pooling). 'Opposite-architecture' "
                      "counterpart to dinov2_mida.py (mean-pool + linear probe). "
                      "n and computation identical to LOSO (Care-PD paper, "
                      "sec. 4.2 / Appendix D.4) — see module docstring & "
                      "dinov2_mida.py for the full MIDA rationale.",
        "hp":         f"FIXED for every fold of every dataset: {DEFAULT_HP} — "
                      f"NOT searched (prohibitively expensive here, and would be "
                      f"confounded by the heterogeneity under test). Matches "
                      f"dinov2_mida.py's single fixed C=1.0 / alpha=1.0.",
        "note":       "No feature pooling — full (81, 768) sequences fed to the "
                      "BiLSTM, identical input format to dinov2_loso.py. Compare "
                      "against dinov2_mida.py (linear probe, same folds, same "
                      "augmented pools) AND against dinov2_lodo_bilstm.py (same "
                      "architecture, smaller/less-target-heavy pool) to separate "
                      "'does mixing cohorts hurt the BiLSTM' from 'does adding "
                      "more in-domain data offset that hurt'.",
        "rationale": {
            "mida":              "See dinov2_mida.py / videomae_mida.py for the "
                                 "full MIDA rationale (verbatim Care-PD protocol).",
            "comparison_frame":  "This 2×4 grid (architecture × {LOSO, LODO, CDA, "
                                 "MIDA}) lets you see whether the BiLSTM-vs-linear "
                                 "gap (if any) within a cohort persists, "
                                 "shrinks, or reverses as the training pool "
                                 "becomes progressively more cross-cohort.",
        },
        "datasets": {
            ds: {k: v for k, v in res.items() if k != "cm_df"}
            for ds, res in dataset_results.items()
        },
    }

    json_path = RESULTS_DIR / f"results_dinov2_mida_bilstm_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", json_path)

    for ds, res in dataset_results.items():
        if res.get("cm_df") is not None:
            cm_path = RESULTS_DIR / f"results_dinov2_mida_bilstm_{ds}_cm_{run_ts}.csv"
            res["cm_df"].to_csv(cm_path)
            log.info("Confusion matrix (%s) saved to %s", ds, cm_path)

    log.info("\n===== MIDA-BiLSTM SUMMARY ACROSS DATASETS =====")
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
