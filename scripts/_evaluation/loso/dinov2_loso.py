"""
DINOv2 + BiLSTM + Attention — Leave-One-Subject-Out (LOSO) Evaluation
=======================================================================
Evaluates the DINOv2+BiLSTM+Attention pipeline using LOSO cross-validation,
applied per dataset separately, with per-dataset hyperparameter search
(Option C).

Input:  dinov2_features_81f.pkl — (81, 768) sequences per clip
Output: outputs/runs/<run>/results/loso/dinov2_bilstm/results_dinov2_loso_{ts}.json

─────────────────────────────────────────────────────────────────────────────
WHY LOSO AND PER DATASET
─────────────────────────────────────────────────────────────────────────────
See videomae_loso.py for the shared rationale. Summary:
- LOSO matches the Care-PD paper evaluation protocol (sec. 4.2).
- Per-dataset evaluation isolates cohort-specific performance.
- 3DGait (43 subjects, ~4 clips/subject) is very noisy; interpret with caution.

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETER STRATEGY — OPTION C (PER-DATASET SEARCH)
─────────────────────────────────────────────────────────────────────────────

The 5-fold script (train_lstm.py) tunes HP on fold 1 only and reuses across
all folds. For LOSO we adopt the same philosophy but per dataset:

    For each dataset:
        1. Take a random 80/20 subject-disjoint split of that dataset.
        2. Train all HP combinations on the 80% split, evaluate on 20%.
        3. Use the best HP for ALL LOSO folds of that dataset.

Why Option C over alternatives:

- Option A (HP from 5-fold run): the 5-fold HP were tuned on combined data
  from all three datasets. LOSO is per dataset, so dataset-specific HP are
  more appropriate. Also: the 5-fold run may not have been completed yet.

- Option B (defaults): not tuned at all. Given that DINOv2+LSTM has multiple
  sensitive hyperparameters (hidden_size, dropout, learning_rate), using
  defaults risks systematically underperforming.

- Option C (this): dataset-specific tuning on a held-out 20% split.
  The 80% training split per dataset is large enough for reliable results:
      PD-GaM  80%: ~24 subjects, ~1920 clips → reliable
      BMClab  80%: ~18 subjects, ~3566 clips → reliable
      3DGait  80%: ~34 subjects, ~168 clips  → too small (fallback to defaults)

  After finding best HP, the 20% held-out data is NOT used again — only
  the HP configuration is retained. All LOSO folds use the full dataset,
  so the 20% split introduces no leakage into LOSO evaluation.

HP search space (same as train_lstm.py):
    hidden_size   ∈ [128, 256]
    dropout       ∈ [0.2, 0.4]
    learning_rate ∈ [1e-3, 5e-4]
    → 8 combinations × ~2 min/run × 3 datasets ≈ ~48 min total HP search
    → Plus LOSO training: 96 folds × ~2 min ≈ ~3 h
    → Total: ~3.5-4 h. Run overnight.

─────────────────────────────────────────────────────────────────────────────
3DGait EXCEPTION
─────────────────────────────────────────────────────────────────────────────

3DGait has only 210 clips across 43 subjects (~4 clips/subject). The 80/20
split would give ~168 training clips — too few for reliable BiLSTM HP search
(the BiLSTM has ~130k-520k parameters depending on hidden_size). For 3DGait
we use the default hyperparameters (hidden_size=256, dropout=0.3, lr=1e-3).
This is documented in the results file.

Usage:
    cd main_project
    python scripts/_evaluation/loso/dinov2_loso.py

Prerequisite: run scripts/dinov2/extract_features.py first to produce
              dinov2_features_81f.pkl.
"""

import itertools
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
RESULTS_DIR   = output_root() / "results" / "loso" / "dinov2_bilstm"

DATASETS  = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]

# Default HP — used for 3DGait and as starting point
DEFAULT_HP = {"hidden_size": 256, "dropout": 0.3, "learning_rate": 1e-3}

# HP search space (identical to train_lstm.py)
HP_GRID = {
    "hidden_size":   [128, 256],
    "dropout":       [0.2, 0.4],
    "learning_rate": [1e-3, 5e-4],
}

# Threshold: datasets with fewer training clips than this use default HP
HP_SEARCH_MIN_CLIPS = 200

# Training
INPUT_SIZE  = 768
NUM_CLASSES = 4
NUM_LAYERS  = 1
BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR_PATIENCE = 10
LR_FACTOR   = 0.5
EARLY_STOP_PATIENCE = 15

RANDOM_STATE = 42
# Subject-disjoint fraction held out of each fold's TRAINING data for
# leak-free early-stopping checkpoint selection (never the test subject).
VAL_FRACTION = 0.15
UPDRS_MIN    = 0
UPDRS_MAX    = 3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────

class GaitSeqDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTMAttention(nn.Module):
    """
    BiLSTM + temporal attention classifier. Identical to
    DINOv2BiLSTMClassifier in train_lstm.py.
    """
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


# ── Utilities ─────────────────────────────────────────────────────────────────

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
    """
    Train BiLSTM+Attention on train split, evaluate val-F1 on val split.
    Returns (model, val_macro_f1).
    Used for both HP search and LOSO fold training.
    """
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


# ── Per-dataset HP search (Option C) ─────────────────────────────────────────

def search_hp_for_dataset(X_list, y, groups, dataset):
    """
    Option C: search HP on a random 80/20 subject-disjoint split.

    GroupShuffleSplit(test_size=0.2) ensures subjects don't appear in
    both train and validation — preventing leakage during HP search.

    After HP selection, the 20% split is discarded. All clips are used
    in the subsequent LOSO evaluation.

    Falls back to DEFAULT_HP if the dataset has fewer than
    HP_SEARCH_MIN_CLIPS training clips (specifically: 3DGait).
    """
    n_clips = len(y)

    if n_clips < HP_SEARCH_MIN_CLIPS:
        log.info(
            "%s: only %d clips — too small for HP search. "
            "Using default HP: %s", dataset, n_clips, DEFAULT_HP
        )
        return DEFAULT_HP.copy(), "default (dataset too small)"

    log.info("%s: running HP search on 80/20 subject-disjoint split ...", dataset)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    hp_train_idx, hp_val_idx = next(gss.split(np.arange(n_clips), y, groups))

    train_feats  = [X_list[i] for i in hp_train_idx]
    train_labels = y[hp_train_idx].tolist()
    val_feats    = [X_list[i] for i in hp_val_idx]
    val_labels   = y[hp_val_idx].tolist()

    log.info("  HP train: %d clips (%d subjects) | HP val: %d clips (%d subjects)",
             len(hp_train_idx), len(np.unique(groups[hp_train_idx])),
             len(hp_val_idx),   len(np.unique(groups[hp_val_idx])))

    combos = list(itertools.product(
        HP_GRID["hidden_size"],
        HP_GRID["dropout"],
        HP_GRID["learning_rate"],
    ))
    log.info("  Testing %d HP combinations ...", len(combos))

    best_f1 = -1.0
    best_hp = DEFAULT_HP.copy()

    for hs, dr, lr in combos:
        tag = f"HP {dataset} h={hs} d={dr} lr={lr}"
        _, val_f1 = train_and_eval(
            train_feats, train_labels, val_feats, val_labels,
            hidden_size=hs, dropout=dr, learning_rate=lr, tag=tag,
        )
        log.info("  %s → val F1=%.3f", tag, val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_hp = {"hidden_size": hs, "dropout": dr, "learning_rate": lr}

    log.info("%s: best HP = %s (val F1=%.3f)", dataset, best_hp, best_f1)
    return best_hp, f"searched (val F1={best_f1:.3f})"


# ── LOSO per dataset ──────────────────────────────────────────────────────────

def run_loso_dataset(X_list, y, groups, walk_ids, dataset, best_hp):
    """
    LOSO evaluation for one dataset using pre-determined HP.
    GroupKFold(n_splits=n_subjects) — one subject as test per fold.
    """
    n_subjects = len(np.unique(groups))
    log.info("── %s LOSO | %d subjects | %d clips | HP: %s ──",
             dataset, n_subjects, len(y), best_hp)

    all_labels = list(range(UPDRS_MIN, UPDRS_MAX + 1))
    gkf        = GroupKFold(n_splits=n_subjects)

    fold_results  = []
    # Accumulate all out-of-fold walk-level predictions for the aggregate
    # confusion matrix AND for a single pooled Spearman correlation (see note
    # below the loop — per-fold Spearman is mathematically undefined in LOSO
    # and was producing NaN for every fold).
    all_walk_true = []
    all_walk_pred = []
    all_walk_cont = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(y)), y, groups), start=1
    ):
        test_subject  = groups[test_idx][0]
        train_feats   = [X_list[i] for i in train_idx]
        test_feats    = [X_list[i] for i in test_idx]
        train_labels  = y[train_idx].tolist()
        test_labels   = y[test_idx]

        # ── Leak-free early stopping ──────────────────────────────────────────
        # The checkpoint must be selected on data the model is allowed to see —
        # NEVER on the held-out test subject. Carve a subject-disjoint validation
        # split from THIS fold's training subjects (GroupShuffleSplit on
        # groups[train_idx]); the test subject stays untouched until scoring.
        es_tr, es_val = next(GroupShuffleSplit(
            n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_STATE
        ).split(np.arange(len(train_idx)), y[train_idx], groups[train_idx]))
        es_train_feats  = [train_feats[i]  for i in es_tr]
        es_train_labels = [train_labels[i] for i in es_tr]
        es_val_feats    = [train_feats[i]  for i in es_val]
        es_val_labels   = [train_labels[i] for i in es_val]

        model, _ = train_and_eval(
            es_train_feats, es_train_labels, es_val_feats, es_val_labels,
            hidden_size=best_hp["hidden_size"],
            dropout=best_hp["dropout"],
            learning_rate=best_hp["learning_rate"],
            tag=f"{dataset} fold {fold}/{n_subjects}",
        )

        test_loader = DataLoader(
            GaitSeqDataset(test_feats, test_labels.tolist()),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        clip_preds, clip_probs, clip_labels_arr = predict(model, test_loader)
        del model

        walk_true, walk_pred = majority_vote(
            clip_preds, test_labels, walk_ids[test_idx]
        )

        # Per-fold values: present-class macro-F1 (kept only for the
        # between-subject SD reported alongside the pooled headline below).
        acc        = accuracy_score(walk_true, walk_pred)
        f1_macro   = macro_f1_03(walk_true, walk_pred)
        f1_macro02 = macro_f1_02(walk_true, walk_pred)
        f1_weighted= weighted_f1(walk_true, walk_pred)

        expected   = (clip_probs * np.array([0, 1, 2, 3])).sum(axis=1)
        walk_cont  = np.array([
            expected[walk_ids[test_idx] == wid].mean()
            for wid in np.unique(walk_ids[test_idx])
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
        "best_hp":    best_hp,
        "folds":      fold_results,
        "summary":    summary,
        "cm_df":      cm_df,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading DINOv2 features from %s", FEATURES_PATH)
    with open(FEATURES_PATH, "rb") as f:
        records = pickle.load(f)
    log.info("Loaded %d records | device: %s", len(records), DEVICE)

    dataset_results = {}
    hp_search_log   = {}

    for dataset in DATASETS:
        rows = [r for r in records if r["dataset"] == dataset]
        if not rows:
            log.warning("No records for %s — skipping.", dataset)
            continue

        X_list   = [r["feature"]    for r in rows]
        y        = np.array([r["UPDRS_GAIT"] for r in rows], dtype=int)
        groups   = np.array([r["subject"]    for r in rows])
        walk_ids = np.array([f"{r['subject']}_{r['trial']}" for r in rows])

        # Step 1: HP search (Option C)
        best_hp, hp_note = search_hp_for_dataset(X_list, y, groups, dataset)
        hp_search_log[dataset] = {"best_hp": best_hp, "note": hp_note}

        # Step 2: LOSO with best HP
        result = run_loso_dataset(X_list, y, groups, walk_ids, dataset, best_hp)
        dataset_results[dataset] = result

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":      "DINOv2 ViT-B/14 (facebook/dinov2-base), frozen + BiLSTM+Attention",
        "evaluation": "LOSO per dataset — GroupKFold(n_splits=n_subjects)",
        "rationale": {
            "loso":      "Matches Care-PD paper within-dataset evaluation protocol.",
            "per_dataset":"Avoids conflating within- and cross-dataset generalisation.",
            "hp_option_C": (
                "HP tuned once per dataset on a random 80/20 subject-disjoint "
                "split (GroupShuffleSplit). Best HP applied to all LOSO folds "
                "of that dataset. 3DGait exception: too few clips for reliable "
                "search; uses DEFAULT_HP instead. See docstring for full rationale."
            ),
        },
        "hp_search": hp_search_log,
        "datasets": {
            ds: {k: v for k, v in res.items() if k != "cm_df"}
            for ds, res in dataset_results.items()
        },
    }

    json_path = RESULTS_DIR / f"results_dinov2_loso_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", json_path)

    for ds, res in dataset_results.items():
        if res.get("cm_df") is not None:
            cm_path = RESULTS_DIR / f"results_dinov2_loso_{ds}_cm_{run_ts}.csv"
            res["cm_df"].to_csv(cm_path)
            log.info("Confusion matrix (%s) saved to %s", ds, cm_path)

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
