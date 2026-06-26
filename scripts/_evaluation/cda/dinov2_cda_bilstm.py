"""
DINOv2 + BiLSTM+Attention — Cross-Dataset Analysis (CDA)   ["opposite" arm]
============================================================================
Companion / contrast script to dinov2_cda.py.

dinov2_cda.py mean-pools (81, 768) → (768,) and uses a linear probe, for the
same cross-cohort reasons as dinov2_lodo.py / dinov2_mida.py. This script
runs the "opposite" setup: train a BiLSTM+Attention DIRECTLY on the full
(81, 768) sequences of a single SOURCE dataset, then evaluate it on each
OTHER (target) dataset's sequences — the 12 source→target pairs, exactly as
in dinov2_cda.py, just with the BiLSTM instead of the linear probe.

─────────────────────────────────────────────────────────────────────────────
WHY CDA IS THE "CLEANEST" OF THE THREE BiLSTM CROSS-COHORT EXPERIMENTS
─────────────────────────────────────────────────────────────────────────────
Unlike LODO and MIDA, CDA's TRAINING pool for any given source is a SINGLE,
homogeneous cohort — exactly the situation in which we argued the BiLSTM is
the right architecture (dinov2_loso.py)! Nothing about the *training* setup
differs from LOSO. The only thing that changes is the *test* distribution:
instead of held-out subjects from the SAME cohort, the model is evaluated on
a DIFFERENT cohort's sequences (different cadence/FPS/recording setup).

This makes CDA-with-BiLSTM the most surgical of the three experiments — it
isolates exactly ONE question:

    "Does a BiLSTM's learned temporal mapping for cohort A transfer when
     applied to cohort B's (different) temporal structure?"

— without also confounding it with "trained on an already-mixed pool" (that
confound is what dinov2_lodo_bilstm.py / dinov2_mida_bilstm.py introduce, on
top of this same transfer question).

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETERS & VALIDATION SPLIT
─────────────────────────────────────────────────────────────────────────────
Fixed DEFAULT_HP (hidden_size=256, dropout=0.3, lr=1e-3 — identical to the
dinov2_loso.py 3DGait fallback / train_lstm.py defaults) for every source.
Deliberately NOT per-source-tuned: tuning each source separately would
conflate "transfer ability" with "how well we tuned source X vs. source Y".
A single fixed configuration isolates the transfer question and matches
dinov2_cda.py's linear probe (also one fixed C=1.0 / alpha=1.0 for every
source).

Early stopping needs a validation set, but — unlike dinov2_loso.py, where the
LOSO test fold doubles as the validation set — we must NOT use any CDA target
for that purpose here (that would leak target information into model
selection and would also force a separate model per target, breaking the
"fit once per source" design we use in dinov2_cda.py). Instead we carve out
a small subject-disjoint validation split FROM THE SOURCE itself
(GroupShuffleSplit, 15% of subjects, same RANDOM_STATE as the rest of the
pipeline) purely for early-stopping checkpoint selection — mirroring the
80/20 subject-disjoint split dinov2_loso.py already uses for its HP search
(search_hp_for_dataset), just repurposed here for early stopping rather than
HP selection. The final model is then evaluated, untouched, on each full
target dataset — exactly as dinov2_cda.py evaluates its linear probe on each
full target dataset.

⚠ Runtime: 4 sources, each training one BiLSTM (up to 100 epochs, early
  stopping patience=15), then evaluated on 3 targets each — comparable to a
  bit more than one full per-dataset LOSO run (4 trainings instead of
  n_subjects, but on full-source-sized pools).

Usage
-----
    cd main_project
    python scripts/_evaluation/cda/dinov2_cda_bilstm.py
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
FEATURES_PATH = PROJECT_ROOT / "assets/datasets/fabricated_datasets/dinov2_features_81f.pkl"
RESULTS_DIR   = output_root() / "results" / "cda" / "dinov2_bilstm"

DATASETS = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]

# Fixed HP — NOT searched per source (see module docstring for why)
DEFAULT_HP = {"hidden_size": 256, "dropout": 0.3, "learning_rate": 1e-3}

# Subject-disjoint validation split carved out of the source, for early
# stopping ONLY (never used for evaluation — targets remain fully held out)
VAL_FRACTION = 0.15

INPUT_SIZE  = 768
NUM_CLASSES = 4
NUM_LAYERS  = 1
BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR_PATIENCE = 10
LR_FACTOR   = 0.5
EARLY_STOP_PATIENCE = 15

RANDOM_STATE = 42
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


def evaluate_on_target(model, X_feats, y_test, walk_ids):
    """Evaluate a trained BiLSTM on one target dataset's full sequences."""
    all_labels = list(range(UPDRS_MIN, UPDRS_MAX + 1))

    loader = DataLoader(
        GaitSeqDataset(X_feats, y_test.tolist()),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    clip_preds, clip_probs, _ = predict(model, loader)
    walk_true, walk_pred = majority_vote(clip_preds, y_test, walk_ids)

    acc        = accuracy_score(walk_true, walk_pred)
    f1_macro   = macro_f1_03(walk_true, walk_pred)
    f1_macro02 = macro_f1_02(walk_true, walk_pred)
    f1_w       = weighted_f1(walk_true, walk_pred)

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

    cm = confusion_matrix(walk_true, walk_pred, labels=all_labels)

    return {
        "n_test_clips":    int(len(X_feats)),
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


def run_cda_bilstm(records: list, datasets: list) -> dict:
    """
    For each source dataset:
      - carve out a subject-disjoint 15% validation split (early stopping only)
      - train ONE BiLSTM+Attention on the remaining ~85% (full sequences)
      - evaluate that single trained model on each of the 3 other datasets
    Returns the 12 pair results plus per-metric source × target matrices.
    """
    pair_results = []
    metrics = ["f1_macro_03", "f1_macro_02", "f1_weighted",
               "accuracy", "mae", "qwk", "spearman"]
    matrices = {m: pd.DataFrame(np.nan, index=datasets, columns=datasets)
                for m in metrics}

    for source in datasets:
        rows = [r for r in records if r["dataset"] == source]
        if not rows:
            log.warning("No records for source dataset %s — skipping.", source)
            continue

        X_list = [r["feature"]    for r in rows]
        y      = np.array([r["UPDRS_GAIT"] for r in rows], dtype=int)
        groups = np.array([r["subject"]    for r in rows])

        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_STATE)
        train_idx, val_idx = next(gss.split(np.arange(len(y)), y, groups))

        train_feats  = [X_list[i] for i in train_idx]
        train_labels = y[train_idx].tolist()
        val_feats    = [X_list[i] for i in val_idx]
        val_labels   = y[val_idx].tolist()

        log.info("── CDA-BiLSTM source=%-10s | train=%d clips (%d subj) | "
                 "early-stop val=%d clips (%d subj) | HP=%s ──",
                 source, len(train_idx), len(np.unique(groups[train_idx])),
                 len(val_idx), len(np.unique(groups[val_idx])), DEFAULT_HP)

        model, val_f1 = train_and_eval(
            train_feats, train_labels, val_feats, val_labels,
            hidden_size=DEFAULT_HP["hidden_size"],
            dropout=DEFAULT_HP["dropout"],
            learning_rate=DEFAULT_HP["learning_rate"],
            tag=f"CDA-BiLSTM source={source}",
        )
        log.info("  %s: trained (early-stop val macro-F1=%.3f)", source, val_f1)

        for target in datasets:
            if target == source:
                continue
            test_rows = [r for r in records if r["dataset"] == target]
            if not test_rows:
                log.warning("No records for target dataset %s — skipping.", target)
                continue

            X_test   = [r["feature"]    for r in test_rows]
            y_test   = np.array([r["UPDRS_GAIT"] for r in test_rows], dtype=int)
            walk_ids = np.array([f"{r['subject']}_{r['trial']}" for r in test_rows])

            res = evaluate_on_target(model, X_test, y_test, walk_ids)
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
                "n_train_clips": int(len(train_idx)),
                "early_stop_val_macro_f1": round(float(val_f1), 4),
                **res,
                "cm": cm.tolist(),
            })

        del model

    return {"pairs": pair_results, "matrices": matrices}


def main() -> None:
    log.info("Loading DINOv2 features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records across datasets: %s | device: %s",
             len(records),
             {ds: sum(1 for r in records if r["dataset"] == ds) for ds in DATASETS},
             DEVICE)

    result = run_cda_bilstm(records, DATASETS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "DINOv2 ViT-B/14 (facebook/dinov2-base), frozen + BiLSTM+Attention",
        "evaluation": "CDA — train ONE BiLSTM per source dataset (full (81,768) "
                      "sequences, NO pooling), evaluate on each other dataset "
                      "separately (12 source→target pairs). 'Opposite-architecture' "
                      "counterpart to dinov2_cda.py (mean-pool + linear probe). "
                      "Diagonal intentionally omitted (use dinov2_loso.py results). "
                      "See module docstring for what this comparison is meant to test.",
        "hp":         f"FIXED for every source: {DEFAULT_HP} — NOT per-source-tuned "
                      f"(isolates the transfer question; matches dinov2_cda.py's "
                      f"single fixed C=1.0 / alpha=1.0).",
        "validation": f"Early-stopping checkpoint selected on a {VAL_FRACTION:.0%} "
                      f"subject-disjoint split CARVED OUT OF THE SOURCE itself "
                      f"(GroupShuffleSplit) — never on a CDA target. Targets remain "
                      f"fully held out, exactly as in dinov2_cda.py.",
        "note":       "No feature pooling — full (81, 768) sequences fed to the "
                      "BiLSTM, identical input format to dinov2_loso.py. Compare "
                      "against dinov2_cda.py (linear probe, same source→target "
                      "pairs) to test whether a BiLSTM's cohort-specific temporal "
                      "mapping transfers across cohorts.",
        "datasets":   DATASETS,
        "pairs":      result["pairs"],
    }

    json_path = RESULTS_DIR / f"results_dinov2_cda_bilstm_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved to %s", json_path)

    for metric, mat in result["matrices"].items():
        mat_path = RESULTS_DIR / f"results_dinov2_cda_bilstm_matrix_{metric}_{run_ts}.csv"
        mat.to_csv(mat_path)
    log.info("Source × target matrices (one CSV per metric) saved to %s", RESULTS_DIR)

    log.info("\n===== CDA-BiLSTM — F1 (0-2) transfer matrix (rows=source, cols=target) =====\n%s",
             result["matrices"]["f1_macro_02"].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
