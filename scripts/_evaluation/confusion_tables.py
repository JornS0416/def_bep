"""
Confusion-matrix statistics tables
====================================
Bouwt confusion-statistieken voor alle vier de protocollen (LOSO, MIDA, LODO, CDA).
LOSO/MIDA/LODO leveren een *_cm_*.csv per dataset; CDA bewaart zijn confusion-
matrices in de run-JSON (één per source→target paar) en wordt per TARGET-dataset
gesommeerd, zodat het op dezelfde manier in onderstaande tabellen meekomt:

  Table 1 — Pooled confusion matrix per (model × protocol)
             Ruwe tellingen, gesommeerd over datasets.

  Table 2 — Per-class prediction statistics
             Per (model, protocol, dataset, UPDRS_class):
             N_true, N_pred, N_correct, precision, recall, F1, bias_ratio.

  Table 3 — Ordinal error profile
             Per (model, protocol): exact%, ±1%, ≥2%, under%, over%.

  Table 4 — Class confusion tendency
             Welke klasse wordt het vaakst verward met welke andere klasse?
             Off-diagonal percentages per (model, protocol).

  Table 5 — Per-dataset difficulty summary
             Per dataset: gemiddeld recall per UPDRS-klasse, over alle modellen.

Uitvoer: CSV-bestanden onder outputs/runs/<run>/tables/
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import output_root

RESULTS_ROOT = output_root() / "results"
TABLES_DIR   = output_root() / "tables"

METHODS  = ["loso", "mida", "lodo", "cda"]
DATASETS = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]
ALL_MODELS = ["videomae", "dinov2", "vjepa", "vjepa2"]
MODEL_LABELS = {
    "videomae":      "VideoMAE",
    "dinov2":        "DINOv2 (BiLSTM)",
    "vjepa":         "V-JEPA",
    "vjepa2":        "V-JEPA 2",
}

_TRUE = [f"true_{i}" for i in range(4)]
_PRED = [f"pred_{i}" for i in range(4)]
TS_RE = re.compile(r"(\d{8}_\d{6})")
DS_RE  = re.compile(r"_(" + "|".join(re.escape(d) for d in DATASETS) + r")_cm_")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_cm(path: Path) -> np.ndarray:
    m = pd.read_csv(path, index_col=0).reindex(index=_TRUE, columns=_PRED)
    return m.fillna(0).to_numpy(dtype=float)


def _folder_for_model(method: str, model_key: str):
    """Return the result folder that matches model_key for this protocol."""
    proto = RESULTS_ROOT / method
    if not proto.is_dir():
        return None
    # dinov2 BiLSTM sits in dinov2_bilstm for all protocols
    folder_name = "dinov2_bilstm" if model_key == "dinov2" else model_key
    candidate = proto / folder_name
    return candidate if candidate.is_dir() else None


def _cms_by_dataset(method: str, model_key: str) -> dict:
    """{dataset_or_'ALL': 4×4 ndarray} of confusion counts for this (method, model).

    LOSO/MIDA store one *_cm_*.csv per dataset; LODO stores a single pooled CM
    (keyed 'ALL'). CDA has no per-dataset CSV — its confusion matrices live in the
    run JSON, one per source→target pair — so they are read from there and summed
    per TARGET dataset, making CDA directly comparable to the other protocols.
    """
    folder = _folder_for_model(method, model_key)
    if folder is None:
        return {}
    if method == "cda":
        return _cda_cms_by_target(folder)
    by_ds, latest_ts = {}, {}
    for p in folder.glob("*_cm_*.csv"):
        m = DS_RE.search(p.name)
        ds = m.group(1) if m else "ALL"
        tm = TS_RE.search(p.stem)
        ts = tm.group(1) if tm else "0"
        if ds not in latest_ts or ts > latest_ts[ds]:
            latest_ts[ds] = ts
            by_ds[ds] = _read_cm(p)
    return by_ds


def _cda_cms_by_target(folder: Path) -> dict:
    """Sum the latest CDA run JSON's per-pair confusion matrices per target dataset."""
    js = sorted(folder.glob("*_cda*.json"))
    if not js:
        return {}
    pairs = json.load(open(js[-1])).get("pairs", [])
    by_tgt: dict = {}
    for p in pairs:
        tgt = p["target"]
        by_tgt[tgt] = by_tgt.get(tgt, np.zeros((4, 4))) + np.array(p["cm"], float)
    return by_tgt


def _pooled_cm(method: str, model_key: str) -> np.ndarray:
    mat = np.zeros((4, 4))
    for cm in _cms_by_dataset(method, model_key).values():
        mat += cm
    return mat


def _safe_div(a, b):
    return a / b if b > 0 else float("nan")


# ── Table 1: Pooled confusion matrix per (model × protocol) ──────────────────
def build_table1() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for model_key in ALL_MODELS:
            mat = _pooled_cm(method, model_key)
            if mat.sum() == 0:
                continue
            for ti in range(4):
                for pi in range(4):
                    rows.append({
                        "protocol":   method.upper(),
                        "model":      MODEL_LABELS[model_key],
                        "true_UPDRS": ti,
                        "pred_UPDRS": pi,
                        "count":      int(mat[ti, pi]),
                    })
    return pd.DataFrame(rows)


# ── Table 2: Per-class prediction statistics ──────────────────────────────────
def build_table2() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for model_key in ALL_MODELS:
            cms = _cms_by_dataset(method, model_key)   # {ds: 4×4 ndarray}
            if not cms:
                continue
            mat_pool = sum(cms.values(), np.zeros((4, 4)))
            sources = {**cms}
            if mat_pool.sum() > 0:
                sources["ALL (pooled)"] = mat_pool

            for ds, mat in sources.items():
                if mat.sum() == 0:
                    continue

                col_sums = mat.sum(axis=0)   # predicted per class
                row_sums = mat.sum(axis=1)   # true per class

                for cls in range(4):
                    n_true    = int(row_sums[cls])
                    n_pred    = int(col_sums[cls])
                    n_correct = int(mat[cls, cls])
                    recall    = _safe_div(n_correct, n_true)
                    precision = _safe_div(n_correct, n_pred)
                    f1 = _safe_div(2 * precision * recall, precision + recall) \
                         if (not np.isnan(precision) and not np.isnan(recall)) else float("nan")
                    bias = _safe_div(n_pred, n_true)   # >1 = over-predicted, <1 = under-predicted

                    rows.append({
                        "protocol":    method.upper(),
                        "model":       MODEL_LABELS[model_key],
                        "dataset":     ds,
                        "UPDRS_class": cls,
                        "N_true":      n_true,
                        "N_pred":      n_pred,
                        "N_correct":   n_correct,
                        "recall":      round(recall, 4)    if not np.isnan(recall)    else float("nan"),
                        "precision":   round(precision, 4) if not np.isnan(precision) else float("nan"),
                        "F1":          round(f1, 4)        if not np.isnan(f1)        else float("nan"),
                        "bias_ratio":  round(bias, 4)      if not np.isnan(bias)      else float("nan"),
                    })
    return pd.DataFrame(rows)


# ── Table 3: Ordinal error profile ───────────────────────────────────────────
def build_table3() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for model_key in ALL_MODELS:
            mat = _pooled_cm(method, model_key)
            if mat.sum() == 0:
                continue
            yt, yp = [], []
            for ti in range(4):
                for pi in range(4):
                    n = int(round(mat[ti, pi]))
                    yt += [ti] * n
                    yp += [pi] * n
            yt, yp = np.array(yt), np.array(yp)
            err = yp - yt
            n = len(err)
            rows.append({
                "protocol":     method.upper(),
                "model":        MODEL_LABELS[model_key],
                "N_total":      n,
                "exact_%":      round(100 * np.mean(err == 0), 1),
                "off_by_1_%":   round(100 * np.mean(np.abs(err) == 1), 1),
                "off_by_ge2_%": round(100 * np.mean(np.abs(err) >= 2), 1),
                "under_%":      round(100 * np.mean(err < 0), 1),   # pred < true
                "over_%":       round(100 * np.mean(err > 0), 1),   # pred > true
                "mean_abs_err": round(float(np.mean(np.abs(err))), 4),
            })
    return pd.DataFrame(rows)


# ── Table 4: Class confusion tendency (off-diagonal percentages) ──────────────
def build_table4() -> pd.DataFrame:
    """
    For each (model, protocol, true_UPDRS_class): what fraction of walks is
    predicted as each other class?  Reveals systematic confusions like 1→2, 2→1.
    """
    rows = []
    for method in METHODS:
        for model_key in ALL_MODELS:
            mat = _pooled_cm(method, model_key)
            if mat.sum() == 0:
                continue
            row_sums = mat.sum(axis=1)
            for ti in range(4):
                if row_sums[ti] == 0:
                    continue
                for pi in range(4):
                    rows.append({
                        "protocol":       method.upper(),
                        "model":          MODEL_LABELS[model_key],
                        "true_UPDRS":     ti,
                        "pred_UPDRS":     pi,
                        "count":          int(mat[ti, pi]),
                        "row_%":          round(100 * mat[ti, pi] / row_sums[ti], 1),
                        "correct":        ti == pi,
                    })
    return pd.DataFrame(rows)


# ── Table 5: Per-dataset difficulty summary ───────────────────────────────────
def build_table5() -> pd.DataFrame:
    """
    Per (dataset, UPDRS_class): mean recall and mean precision over all models
    (pooled over protocols).  Reveals which class is structurally hardest
    in each cohort, independent of the encoder.
    """
    # Accumulate recall/precision per (dataset, class) across all models & protocols
    acc: dict = {}   # key=(ds, cls) → list of (recall, precision)
    for method in METHODS:
        for model_key in ALL_MODELS:
            cms = _cms_by_dataset(method, model_key)
            for ds, mat in cms.items():
                if ds == "ALL":
                    continue
                if mat.sum() == 0:
                    continue
                col_sums = mat.sum(axis=0)
                row_sums = mat.sum(axis=1)
                for cls in range(4):
                    rec  = _safe_div(mat[cls, cls], row_sums[cls])
                    prec = _safe_div(mat[cls, cls], col_sums[cls])
                    acc.setdefault((ds, cls), []).append((rec, prec))

    rows = []
    for (ds, cls), vals in sorted(acc.items()):
        recs  = [v[0] for v in vals if not np.isnan(v[0])]
        precs = [v[1] for v in vals if not np.isnan(v[1])]
        rows.append({
            "dataset":         ds,
            "UPDRS_class":     cls,
            "N_observations":  len(recs),
            "mean_recall":     round(np.mean(recs), 4)  if recs  else float("nan"),
            "std_recall":      round(np.std(recs), 4)   if recs  else float("nan"),
            "mean_precision":  round(np.mean(precs), 4) if precs else float("nan"),
            "std_precision":   round(np.std(precs), 4)  if precs else float("nan"),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building Table 1: Pooled confusion matrices …")
    t1 = build_table1()
    t1.to_csv(TABLES_DIR / "table1_pooled_cm.csv", index=False)
    print(f"  → {len(t1)} rows")

    print("Building Table 2: Per-class prediction statistics …")
    t2 = build_table2()
    t2.to_csv(TABLES_DIR / "table2_perclass_stats.csv", index=False)
    print(f"  → {len(t2)} rows")

    print("Building Table 3: Ordinal error profile …")
    t3 = build_table3()
    t3.to_csv(TABLES_DIR / "table3_ordinal_errors.csv", index=False)
    print(f"  → {len(t3)} rows")
    print()
    print(t3.to_string(index=False))

    print("\nBuilding Table 4: Class confusion tendency …")
    t4 = build_table4()
    t4.to_csv(TABLES_DIR / "table4_confusion_tendency.csv", index=False)
    print(f"  → {len(t4)} rows")

    print("\nBuilding Table 5: Per-dataset difficulty summary …")
    t5 = build_table5()
    t5.to_csv(TABLES_DIR / "table5_dataset_difficulty.csv", index=False)
    print(f"  → {len(t5)} rows")
    print()
    print(t5.to_string(index=False))

    # ── Console summary: per-class stats for ALL (pooled), LOSO only ─────────
    print("\n── Table 2 preview: ALL-pooled, LOSO ─────────────────────────────────")
    preview = t2[
        (t2["protocol"] == "LOSO") & (t2["dataset"] == "ALL (pooled)")
    ][["model", "UPDRS_class", "N_true", "N_pred", "N_correct",
       "recall", "precision", "F1", "bias_ratio"]]
    print(preview.to_string(index=False))

    print(f"\nAll tables written to {TABLES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
