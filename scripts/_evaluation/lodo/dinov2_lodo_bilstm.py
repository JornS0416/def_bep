"""
DINOv2 + BiLSTM+Attention — Leave-One-Dataset-Out (LODO) Evaluation  ["opposite" arm]
======================================================================================
Companion / contrast script to dinov2_lodo.py.

dinov2_lodo.py mean-pools (81, 768) → (768,) and uses a linear probe, arguing
that a BiLSTM trained on a POOL OF HETEROGENEOUS COHORTS (different cadence,
FPS, recording/rendering setups) cannot learn one coherent temporal mapping —
"frame index t" represents a different point in the gait cycle, and a
different real-world time span, depending on which cohort a clip came from.

This script runs the "opposite" setup: train a BiLSTM+Attention DIRECTLY on
the full (81, 768) sequences of the pooled D-1 training cohorts (NO pooling),
and test on the held-out dataset's sequences. It gives an empirical answer to
"does mixing cohorts actually break the BiLSTM, or was that only a theoretical
concern?" — the direct counterpart to dinov2_loso_linear.py (which tests the
flip side: does the linear probe lose anything by ignoring temporal info
*within* a single, homogeneous cohort?).

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETERS — FIXED, NOT SEARCHED
─────────────────────────────────────────────────────────────────────────────
Unlike dinov2_loso.py's per-dataset Option C search, this script uses a
SINGLE FIXED configuration (DEFAULT_HP = hidden_size=256, dropout=0.3,
lr=1e-3 — identical to the dinov2_loso.py 3DGait fallback / train_lstm.py
defaults) for all 4 LODO folds. Reasons:

  1. The LODO training pool is, by construction, a MIX of 3 heterogeneous
     cohorts. There is no single "right" temporal scale to search for — an
     HP search on this mix would itself be confounded by the very
     cross-cohort heterogeneity this experiment is meant to probe.
  2. An 8-combination grid search × 4 LODO folds × pools of several thousand
     clips would be prohibitively expensive (many hours).
  3. Using ONE fixed architecture for all 4 folds isolates the variable we
     actually care about — does cross-cohort pooling break the BiLSTM? —
     from HP-search noise, and keeps this directly comparable to
     dinov2_lodo.py's linear probe (also a single fixed configuration:
     C=1.0 / alpha=1.0).

Early stopping uses the held-out dataset's clips as the validation set —
identical to how dinov2_loso.py uses the LOSO test fold for early stopping
(the established pattern in this codebase). Note this means the reported
metrics and the early-stopping checkpoint selection both involve the held-out
set; this is the SAME (mild) leakage already present and accepted in
dinov2_loso.py, kept here for direct comparability.

⚠ Runtime: 4 LODO folds, each training a BiLSTM+Attention on a pool of
  several thousand heterogeneous clips for up to 100 epochs (with early
  stopping, patience=15). Expect this to take roughly as long as a full
  per-dataset LOSO run, multiplied by ~4 — i.e., potentially many hours.
  Consider a quick smoke-test (e.g., temporarily lower NUM_EPOCHS) first.

Usage
-----
    cd main_project
    python scripts/_evaluation/lodo/dinov2_lodo_bilstm.py
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    cohen_kappa_score, confusion_matrix,
)
from scipy.stats import spearmanr
from tqdm import tqdm

# ── Care-PD–faithful metrics (shared module) ──────────────────────────────────
# Present-class macro-F1 (no forced absent classes); each held-out test set is a
# whole dataset, so the fix is only to stop injecting absent UPDRS classes
# (esp. class 3) as F1=0 terms. See scripts/_evaluation/carepd_metrics.py and
# docs/appendix_evaluation_metrics.md.
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
RESULTS_DIR   = output_root() / "results" / "lodo" / "dinov2_bilstm"

DATASETS = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]

# Fixed HP — NOT searched (see module docstring for why)
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
# Subject-disjoint fraction of the D-1 TRAINING cohorts held out for leak-free
# early-stopping checkpoint selection (never the held-out test dataset).
VAL_FRACTION = 0.15
UPDRS_MIN, UPDRS_MAX = 0, 3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset / Model — identical to dinov2_loso.py ────────────────────────────

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


# ── Utilities — identical to dinov2_loso.py ──────────────────────────────────

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


# ── LODO with BiLSTM ──────────────────────────────────────────────────────────

def run_lodo_bilstm(records: list, datasets: list) -> dict:
    """
    For each held-out dataset:
      - train a BiLSTM+Attention DIRECTLY on the full (81, 768) sequences of
        the pooled D-1 cohorts (no pooling — the "opposite" of dinov2_lodo.py)
      - test on the held-out dataset's sequences
    Uses DEFAULT_HP for every fold (see module docstring for why).
    """
    all_labels   = list(range(UPDRS_MIN, UPDRS_MAX + 1))
    fold_results = []
    all_walk_true, all_walk_pred = [], []

    for held_out in datasets:
        train_rows = [r for r in records if r["dataset"] != held_out]
        test_rows  = [r for r in records if r["dataset"] == held_out]

        if not test_rows or not train_rows:
            log.warning("Skipping held-out=%s — missing train or test rows.", held_out)
            continue

        train_feats  = [r["feature"]    for r in train_rows]
        train_labels = [r["UPDRS_GAIT"] for r in train_rows]
        test_feats   = [r["feature"]    for r in test_rows]
        test_labels  = np.array([r["UPDRS_GAIT"] for r in test_rows], dtype=int)
        walk_ids     = np.array([f"{r['subject']}_{r['trial']}" for r in test_rows])

        train_ds_counts = {ds: sum(1 for r in train_rows if r["dataset"] == ds)
                           for ds in datasets if ds != held_out}
        log.info("── LODO (BiLSTM) fold: test=%-10s | train=%s (%d clips total) | HP=%s ──",
                 held_out, train_ds_counts, len(train_feats), DEFAULT_HP)

        # ── Leak-free early stopping ──────────────────────────────────────────
        # Select the checkpoint on held-out subjects from the D-1 TRAINING
        # cohorts (subjects namespaced by dataset so IDs don't collide across
        # cohorts) — NEVER on the held-out test dataset.
        train_groups = np.array([f"{r['dataset']}__{r['subject']}" for r in train_rows])
        es_tr, es_val = next(GroupShuffleSplit(
            n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_STATE
        ).split(np.arange(len(train_feats)), train_labels, train_groups))
        es_train_feats  = [train_feats[i]  for i in es_tr]
        es_train_labels = [train_labels[i] for i in es_tr]
        es_val_feats    = [train_feats[i]  for i in es_val]
        es_val_labels   = [train_labels[i] for i in es_val]

        model, _ = train_and_eval(
            es_train_feats, es_train_labels, es_val_feats, es_val_labels,
            hidden_size=DEFAULT_HP["hidden_size"],
            dropout=DEFAULT_HP["dropout"],
            learning_rate=DEFAULT_HP["learning_rate"],
            tag=f"LODO-BiLSTM held_out={held_out}",
        )

        test_loader = DataLoader(
            GaitSeqDataset(test_feats, test_labels.tolist()),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        clip_preds, clip_probs, _ = predict(model, test_loader)
        del model

        walk_true, walk_pred = majority_vote(clip_preds, test_labels, walk_ids)

        acc         = accuracy_score(walk_true, walk_pred)
        f1_macro    = macro_f1_03(walk_true, walk_pred)
        f1_macro02  = macro_f1_02(walk_true, walk_pred)
        f1_weighted = weighted_f1(walk_true, walk_pred)

        expected  = (clip_probs * np.array([0, 1, 2, 3])).sum(axis=1)
        walk_cont = np.array([
            expected[walk_ids == wid].mean()
            for wid in np.unique(walk_ids)
        ])

        mae = mean_absolute_error(walk_true, walk_pred)
        try:
            qwk = cohen_kappa_score(walk_true, walk_pred, weights="quadratic")
        except ValueError:
            qwk = float("nan")
        rho, _ = spearmanr(walk_true, walk_cont)

        all_walk_true.extend(walk_true.tolist())
        all_walk_pred.extend(walk_pred.tolist())

        log.info("  F1(0-3)=%.3f  F1(0-2)=%.3f  Acc=%.3f  MAE=%.3f  QWK=%s",
                 f1_macro, f1_macro02, acc, mae,
                 f"{qwk:.3f}" if not np.isnan(qwk) else "nan")

        fold_results.append({
            "held_out":        held_out,
            "n_train_clips":   int(len(train_feats)),
            "n_test_clips":    int(len(test_feats)),
            "n_test_walks":    int(len(walk_true)),
            "train_datasets":  train_ds_counts,
            "updrs_dist_test": {int(k): int(v) for k, v in
                                zip(*np.unique(test_labels, return_counts=True))},
            "f1_macro_03":     round(f1_macro,    4),
            "f1_macro_02":     round(f1_macro02,  4),
            "f1_weighted":     round(f1_weighted, 4),
            "accuracy":        round(acc,         4),
            "mae":             round(mae,         4),
            "qwk":             round(qwk, 4) if not np.isnan(qwk) else None,
            "spearman":        round(float(rho),  4),
        })

    def _ms(key):
        vals = [f[key] for f in fold_results if f[key] is not None]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    summary = {}
    log.info("\n===== LODO (BiLSTM) SUMMARY (mean ± std over %d folds) =====", len(fold_results))
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
    log.info("Loading DINOv2 features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s | device: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS},
             DEVICE)

    result = run_lodo_bilstm(records, DATASETS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "DINOv2 ViT-B/14 (facebook/dinov2-base), frozen + BiLSTM+Attention",
        "evaluation": "LODO — train on D-1 datasets (pooled, full (81,768) "
                      "sequences, NO pooling), test on held-out dataset. "
                      "'Opposite-architecture' counterpart to dinov2_lodo.py "
                      "(which uses mean-pooling + linear probe). See module "
                      "docstring for what this comparison is meant to test.",
        "hp":         f"FIXED for all folds: {DEFAULT_HP} — NOT searched per fold "
                      f"(an HP search on a heterogeneous mixed-cohort pool would be "
                      f"confounded by the very thing we're testing; also "
                      f"prohibitively expensive). See module docstring.",
        "note":       "No feature pooling — full (81, 768) sequences fed to the "
                      "BiLSTM, identical input format to dinov2_loso.py. Compare "
                      "against dinov2_lodo.py (linear probe, same folds, same "
                      "held-out datasets) to test whether mixing heterogeneous "
                      "cohorts actually breaks the BiLSTM's temporal assumptions.",
        "datasets":   DATASETS,
        "folds":      result["folds"],
        "summary":    result["summary"],
    }

    json_path = RESULTS_DIR / f"results_dinov2_lodo_bilstm_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved to %s", json_path)

    cm_path = RESULTS_DIR / f"results_dinov2_lodo_bilstm_cm_{run_ts}.csv"
    result["cm_df"].to_csv(cm_path)
    log.info("Confusion matrix saved to %s", cm_path)


if __name__ == "__main__":
    main()
