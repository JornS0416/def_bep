"""
Visualize key metric comparisons across models, protocols & architectures
==========================================================================
Companion to aggregate_results.py: instead of one big table, this script
renders the most informative comparisons as figures (PNG), focused on the
two headline classification metrics (per your request):
    f1_macro_02  — Macro F1, labels (0,1,2)   [Care-PD's primary metric]
    f1_macro_03  — Macro F1, labels (0,1,2,3)

Figures produced (one subdirectory per letter under outputs/figures/):
  A — model comparison per protocol            grouped bars: dataset × encoder,
                                                one panel per protocol
  B — protocol comparison / generalisation gap line chart: how much performance
                                                drops LOSO → LODO/CDA, and
                                                whether MIDA recovers it
  D — within+cross-dataset transfer matrices   heatmap per encoder: diagonal =
                                                LOSO (within), off-diagonal =
                                                CDA (cross) — à la Care-PD's
                                                per-model train×test figure
  E — dataset-difficulty profile               grouped bars: which cohorts are
                                                intrinsically hard, regardless
                                                of encoder (averaged over the 3)
  G — LOSO/MIDA fold-to-fold stability         bars with mean ± std error bars
                                                (only LOSO/MIDA aggregate folds)
  I — does auxiliary data help? (MIDA vs LOSO) grouped bars per encoder:
                                                same target cohort, with vs
                                                without the mixed auxiliary pool
  H — per-protocol "all models at a glance"    heatmap: model × dataset,
                                                side-by-side for both F1
                                                variants — à la Care-PD's
                                                own LODO comparison figure

Usage
-----
    cd main_project
    python scripts/_evaluation/plot_results.py
"""

import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import output_root          # honours $BEP_OUTPUT_ROOT (per-run folder)
from carepd_metrics import macro_f1_02      # present-class macro-F1(0-2), same as eval
RESULTS_ROOT = output_root() / "results"
FIGURES_DIR  = output_root() / "figures"

METHODS = ["loso", "lodo", "cda", "mida"]
METHOD_LABELS = {"loso": "LOSO", "lodo": "LODO", "cda": "CDA", "mida": "MIDA"}
ALL_MODELS = ["videomae", "dinov2", "vjepa", "vjepa2"]   # everything that exists
MODELS     = ["videomae", "dinov2", "vjepa2"]             # shown in the MAIN plots;
#   V-JEPA 2 is the MAIN V-JEPA encoder; V-JEPA v1 appears ONLY in
#   fig_C_vjepa_comparison. Every main figure iterates MODELS, so v1 is excluded
#   everywhere except that one comparison.
MODEL_LABELS = {"videomae": "VideoMAE", "dinov2": "DINOv2",
                "vjepa": "V-JEPA", "vjepa2": "V-JEPA 2"}
DATASETS = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]
VARIANT_ORDER = ["default"]

METRICS = ["f1_macro_02", "f1_macro_03"]
METRIC_LABELS = {
    "f1_macro_02": "Macro F1 — labels (0,1,2)",
    "f1_macro_03": "Macro F1 — labels (0,1,2,3)",
}

TS_RE = re.compile(r"(\d{8}_\d{6})")
sns.set_theme(style="whitegrid", font_scale=1.4)

# Global text enlargement for thesis-ready figures (all text scaled up).
plt.rcParams.update({
    "font.size":        15,
    "axes.titlesize":   17,
    "axes.labelsize":   15,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  13,
    "figure.titlesize": 18,
})# ─────────────────────────────────────────────────────────────────────────────


# ── Loading (lean re-derivation of aggregate_results.py's logic — only the
#    columns this script actually plots) ──────────────────────────────────────
def parse_model_variant(dirname, method=None):
    """Return (model, variant) for a result-folder name.

    DINOv2's BiLSTM is the probe we report, and now lives in a consistently
    named folder across *all* protocols:
      • folder ``dinov2_bilstm`` → ("dinov2", "default")  [BiLSTM — the probe used]
      • folder ``dinov2``        → ("dinov2", "linear")   [ridge probe; dropped below]
    (LOSO only has the BiLSTM run.)
    """
    for model in ALL_MODELS:
        if dirname == model:
            # bare 'dinov2' is the ridge/linear-probe run; relabel so it is dropped.
            if model == "dinov2":
                return model, "linear"
            return model, "default"
        if dirname.startswith(model + "_"):
            suffix = dirname[len(model) + 1:]
            if model == "dinov2" and suffix == "bilstm":
                return model, "default"
            return model, suffix
    return dirname, "default"


def latest_per_group(json_paths):
    candidates = {}
    for p in json_paths:
        method = p.parents[1].name
        model, variant = parse_model_variant(p.parents[0].name, method=method)
        m = TS_RE.search(p.stem)
        ts = m.group(1) if m else "00000000_000000"
        candidates.setdefault((method, model, variant), []).append((p, ts))
    keepers = {}
    for key, entries in candidates.items():
        entries.sort(key=lambda e: e[1])
        keepers[key] = entries[-1][0]
    return keepers


def _rows_loso_or_mida(d, method, model, variant):
    rows = []
    for ds_name, res in d.get("datasets", {}).items():
        if not isinstance(res, dict) or "summary" not in res:
            continue
        s = res["summary"]
        row = {"method": method, "model": model, "variant": variant,
               "scope": ds_name, "scope_kind": "dataset"}
        for m in METRICS:
            row[m] = s.get(f"{m}_mean")
            row[f"{m}_std"] = s.get(f"{m}_std")
        rows.append(row)
    return rows


def _rows_lodo(d, method, model, variant):
    rows = []
    for fold in d.get("folds", []):
        row = {"method": method, "model": model, "variant": variant,
               "scope": fold.get("held_out"), "scope_kind": "dataset"}
        for m in METRICS:
            row[m] = fold.get(m)
            row[f"{m}_std"] = None
        rows.append(row)
    return rows


def _rows_cda(d, method, model, variant):
    rows = []
    for pair in d.get("pairs", []):
        row = {"method": method, "model": model, "variant": variant,
               "scope": f"{pair.get('source')} → {pair.get('target')}",
               "scope_kind": "pair",
               "source": pair.get("source"), "target": pair.get("target")}
        for m in METRICS:
            row[m] = pair.get(m)
            row[f"{m}_std"] = None
        rows.append(row)
    return rows


def _parse_file(path, method, model, variant):
    with open(path) as f:
        d = json.load(f)
    datasets_field = d.get("datasets")
    if isinstance(datasets_field, dict) and any(
        isinstance(v, dict) and "summary" in v for v in datasets_field.values()
    ):
        return _rows_loso_or_mida(d, method, model, variant)
    if "pairs" in d:
        return _rows_cda(d, method, model, variant)
    if d.get("folds") and "held_out" in d["folds"][0]:
        return _rows_lodo(d, method, model, variant)
    log.warning("Unrecognised result structure in %s — skipped.", path)
    return []


def load_master_df():
    json_paths = sorted(p for method in METHODS for p in (RESULTS_ROOT / method).glob("*/*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No result files found under {RESULTS_ROOT} — "
                                f"run the evaluation scripts first.")
    keepers = latest_per_group(json_paths)
    rows = []
    for (method, model, variant), path in sorted(keepers.items()):
        rows.extend(_parse_file(path, method, model, variant))
    df = pd.DataFrame(rows)
    df["model"]  = pd.Categorical(df["model"],  categories=ALL_MODELS,  ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)

    # DINOv2 uses BiLSTM as its sole probe — drop the linear-probe variant
    # so it never appears in any figure.
    dropped = (df["model"] == "dinov2") & (df["variant"] == "linear")
    if dropped.any():
        log.info("Dropping %d DINOv2 linear-probe rows (BiLSTM is the only probe used)",
                 dropped.sum())
        df = df[~dropped].reset_index(drop=True)

    log.info("Loaded master dataframe: %d rows from %d (method,model,variant) result files",
             len(df), len(keepers))
    for (method, model, variant) in sorted(keepers):
        if model == "dinov2" and variant == "linear":
            continue   # excluded above
        log.info("    %-5s | %-10s | %s", method, model, variant)
    return df


# ── Small shared helpers ──────────────────────────────────────────────────────
def _save(fig, subdir, name):
    out_dir = FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("    saved → %s", path.relative_to(PROJECT_ROOT))


def _scope_value_table(df, method, variant):
    """
    Unify the four protocols' native granularity into one tidy
    (model, dataset, f1_macro_02, f1_macro_03) table:
      - loso/mida : already one row per dataset (fold-aggregated mean)
      - lodo      : one row per held-out dataset (the top-level "ALL" summary
                    row is dropped here — it would double-count)
      - cda       : 12 source→target rows -> averaged PER TARGET, so "dataset"
                    means "this cohort as the held-out/target cohort" — directly
                    comparable to LODO's framing ("how well do we do ON X when
                    X wasn't used for [in-domain] training")
    """
    sub = df[(df["method"] == method) & (df["variant"] == variant)]
    if sub.empty:
        return pd.DataFrame(columns=["model", "dataset"] + METRICS)

    if method in ("loso", "mida"):
        out = (sub[sub["scope_kind"] == "dataset"][["model", "scope"] + METRICS]
               .rename(columns={"scope": "dataset"}))
    elif method == "lodo":
        out = (sub[(sub["scope_kind"] == "dataset") & (sub["scope"] != "ALL (LODO mean)")]
               [["model", "scope"] + METRICS].rename(columns={"scope": "dataset"}))
    else:  # cda
        out = (sub[sub["scope_kind"] == "pair"]
               .groupby(["model", "target"], observed=True)[METRICS].mean()
               .reset_index().rename(columns={"target": "dataset"}))
    return out.reset_index(drop=True)


def _take_legend(ax):
    """Grab (handles, labels) from an axes' current legend, then remove it —
    used to build ONE shared figure-level legend from small-multiple panels."""
    h, l = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return h, l


# ── A — model comparison per protocol ─────────────────────────────────────────
def fig_A_model_comparison(df):
    """Which encoder performs best, per protocol & cohort? Each encoder in its
    default setup: BiLSTM for DINOv2 across all protocols, fixed linear probe
    for VideoMAE and V-JEPA."""
    log.info("Model comparison per protocol")
    for metric in METRICS:
        fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.5), sharey=True)
        handles = labels = None
        for ax, method in zip(axes, METHODS):
            view = _scope_value_table(df, method, "default")
            if view.empty:
                ax.set_title(f"{METHOD_LABELS[method]}\n(no data)")
                ax.axis("off")
                continue
            sns.barplot(data=view, x="dataset", y=metric, hue="model",
                        order=DATASETS, hue_order=MODELS, palette="Set2", ax=ax)
            ax.set_title(METHOD_LABELS[method], fontsize=15.9, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric] if ax is axes[0] else "")
            ax.tick_params(axis="x", rotation=28)
            ax.set_ylim(0, 1)
            h, l = _take_legend(ax)
            if handles is None and h:
                handles, labels = h, l
        if handles:
            fig.legend(handles, [MODEL_LABELS.get(x, x) for x in labels], title="Encoder",
                       loc="upper center", ncol=len(MODELS), bbox_to_anchor=(0.5, 1.06), frameon=False)
        fig.suptitle(f"Encoder comparison per evaluation protocol   ({METRIC_LABELS[metric]})",
                     fontweight="bold", fontsize=17.4, y=1.13)
        _save(fig, "A_model_comparison", f"A_model_comparison_{metric}")


# ── B — protocol comparison / generalisation gap ──────────────────────────────
def fig_B_protocol_comparison(df):
    """How much performance drops going from within-cohort (LOSO) to
    cross-cohort (LODO/CDA), and whether MIDA's auxiliary-data augmentation
    recovers (some of) that drop — averaged over cohorts, per encoder."""
    log.info("Figure B — protocol comparison / generalisation gap")
    for metric in METRICS:
        rows = []
        for method in METHODS:
            view = _scope_value_table(df, method, "default")
            if view.empty:
                continue
            g = view.groupby("model", observed=True)[metric].mean().reset_index()
            g["method"] = method
            rows.append(g)
        if not rows:
            continue
        plot_df = pd.concat(rows, ignore_index=True)
        plot_df["method"] = pd.Categorical(plot_df["method"], categories=METHODS, ordered=True)
        plot_df = plot_df.sort_values(["model", "method"])

        fig, ax = plt.subplots(figsize=(9.1, 6.0))
        sns.lineplot(data=plot_df, x="method", y=metric, hue="model", hue_order=MODELS,
                     marker="o", markersize=9, linewidth=2.2, palette="Set2", ax=ax)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
        ax.set_xlabel("Evaluation protocol")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0, 1)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, [MODEL_LABELS.get(x, x) for x in l], title="Encoder", frameon=True)
        ax.set_title(f"Generalisation across evaluation protocols   ({METRIC_LABELS[metric]})\n"
                     f"Mean over 4 cohorts", fontsize=17.4, fontweight="bold")
        _save(fig, "B_protocol_comparison", f"B_protocol_comparison_{metric}")


# ── D — combined within + cross-dataset transfer matrices ─────────────────────
def _build_transfer_matrix(df, model, variant_loso, variant_cda, metric):
    """Train(rows)×Test(cols) matrix: diagonal = LOSO (within-dataset),
    off-diagonal = CDA (cross-dataset, source→target)."""
    mat = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
    loso = df[(df["method"] == "loso") & (df["model"] == model)
              & (df["variant"] == variant_loso) & (df["scope_kind"] == "dataset")]
    for _, r in loso.iterrows():
        if r["scope"] in DATASETS:
            mat.loc[r["scope"], r["scope"]] = r[metric]
    cda = df[(df["method"] == "cda") & (df["model"] == model)
             & (df["variant"] == variant_cda) & (df["scope_kind"] == "pair")]
    for _, r in cda.iterrows():
        if r["source"] in DATASETS and r["target"] in DATASETS:
            mat.loc[r["source"], r["target"]] = r[metric]
    return mat


def _plot_transfer_heatmap(ax, mat, title):
    sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                cbar=False, linewidths=0.6, linecolor="white", square=True,
                annot_kws={"fontsize": 18, "fontweight": "bold"}, ax=ax)
    for i in range(len(DATASETS)):                      # frame the "home turf" diagonal
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2.2))
    ax.set_xlabel("Test set (target)", fontsize=15)
    ax.set_ylabel("Train set (source)", fontsize=15)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", rotation=0, labelsize=14)
    ax.tick_params(axis="x", labelsize=14)


def fig_D_transfer_matrices(df):
    """Combined within-dataset (LOSO, framed diagonal — "home turf") and
    cross-dataset (CDA, off-diagonal — genuine transfer) matrix per encoder.
    Mirrors Care-PD's per-model train×test heatmap (e.g. their MixSTE figure)."""
    log.info("Figure D — combined within+cross-dataset transfer matrices")
    for model in MODELS:
        if df[(df["method"] == "loso") & (df["model"] == model)].empty \
           and df[(df["method"] == "cda") & (df["model"] == model)].empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14.3, 6.5))
        for ax, metric in zip(axes, METRICS):
            mat = _build_transfer_matrix(df, model, "default", "default", metric)
            _plot_transfer_heatmap(ax, mat, METRIC_LABELS[metric])
        fig.subplots_adjust(top=0.90)
        fig.suptitle(f"{MODEL_LABELS[model]}: LOSO (diagonal) and CDA (off-diagonal) transfer",
                     fontweight="bold", fontsize=17.4, y=0.97)
        _save(fig, "D_transfer_matrices", f"D_transfer_matrix_{model}")



# ── D (combined) — all three transfer matrices in one figure ─────────────
def fig_D_combined_transfer_matrices(df):
    """Single thesis-ready figure: all encoders (rows) × both F1 metrics
    (columns) in one grid. Each cell is the combined within-dataset
    (LOSO, framed diagonal) + cross-dataset (CDA, off-diagonal) matrix."""
    log.info("Figure D (combined) — all transfer matrices in one figure")

    # Check that at least some data exists
    if df[(df["method"].isin(["loso", "cda"]))].empty:
        log.warning("  no LOSO/CDA data — skipping combined transfer matrix figure")
        return

    # Short row labels so the left-margin annotation stays compact
    SHORT_METRIC = {"f1_macro_02": "Macro F1\n(0–2)", "f1_macro_03": "Macro F1\n(0–3)"}

    n_ds = len(DATASETS)
    # 2 rows (metrics) × N columns (one per model — scales with len(MODELS))
    # square=True forces equal cell width/height; figure size controls physical cell size.
    fig, axes = plt.subplots(
        2, len(MODELS),
        figsize=(6.2 * len(MODELS), 13.5),
        gridspec_kw={"hspace": 0.18, "wspace": 0.15},
    )

    for row_i, metric in enumerate(METRICS):
        for col_j, model in enumerate(MODELS):
            ax = axes[row_i, col_j]
            mat = _build_transfer_matrix(df, model, "default", "default", metric)

            # ── heatmap ───────────────────────────────────────────────────────
            sns.heatmap(
                mat.astype(float), annot=True, fmt=".2f",
                cmap="Blues", vmin=0, vmax=1,
                cbar=False, linewidths=0.8, linecolor="white",
                square=True,
                annot_kws={"fontsize": 20, "fontweight": "bold"}, ax=ax,
            )
            for i in range(n_ds):
                ax.add_patch(
                    Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2.2)
                )

            ax.tick_params(axis="y", rotation=0, labelsize=14)
            ax.tick_params(axis="x", rotation=30, labelsize=14)

            # ── column header: model name (top row only) ──────────────────────
            if row_i == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=20,
                             fontweight="bold", pad=10)
            else:
                ax.set_title("")

            # ── y-axis: left column keeps tick labels; no ylabel (row annotation serves as label) ───
            if col_j == 0:
                ax.set_ylabel("Train set (source)", fontsize=15, labelpad=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            # ── x-axis label: bottom row only ─────────────────────────────────
            if row_i == len(METRICS) - 1:
                ax.set_xlabel("Test set (target)", fontsize=15)
            else:
                ax.set_xlabel("")

            # ── row label: compact metric name to the left of the left panel ─
            # Placed further left (-0.45) to clear the tick labels and ylabel.
            if col_j == 0:
                ax.annotate(
                    SHORT_METRIC[metric],
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-0.45, 0.5), textcoords="axes fraction",
                    fontsize=15, fontweight="bold",
                    ha="center", va="center", rotation=90,
                    annotation_clip=False,
                )

    fig.subplots_adjust(left=0.13, top=0.95)
    fig.suptitle(
        "Within-dataset LOSO (framed diagonal) and cross-dataset CDA (off-diagonal)",
        fontweight="bold", fontsize=20, y=0.99,
    )
    _save(fig, "D_transfer_matrices", "D_combined_transfer_matrix")


# ── E — dataset-difficulty profile ────────────────────────────────────────────
def fig_E_dataset_difficulty(df):
    """Which cohorts are intrinsically hard, REGARDLESS of which encoder you
    use? (mean over the 3 encoders' native/default setup, shown per protocol —
    a cohort that's hard everywhere is a property of the data, not the model)."""
    log.info("Figure E — dataset-difficulty profile")
    for metric in METRICS:
        rows = []
        for method in METHODS:
            view = _scope_value_table(df, method, "default")
            if view.empty:
                continue
            g = view.groupby("dataset", observed=True)[metric].mean().reset_index()
            g["method"] = method
            rows.append(g)
        if not rows:
            continue
        plot_df = pd.concat(rows, ignore_index=True)

        fig, ax = plt.subplots(figsize=(9.75, 6.0))
        sns.barplot(data=plot_df, x="dataset", y=metric, hue="method",
                    order=DATASETS, hue_order=METHODS, palette="flare", ax=ax)
        ax.set_xlabel("Cohort")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(0, 1)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, [METHOD_LABELS.get(x, x) for x in l], title="Protocol")
        ax.set_title(f"Per-cohort difficulty profile   ({METRIC_LABELS[metric]})\n"
                     f"Averaged over 3 encoders",
                     fontsize=17.4, fontweight="bold")
        _save(fig, "E_dataset_difficulty", f"E_dataset_difficulty_{metric}")


# ── G — LOSO/MIDA fold-to-fold stability (mean ± std) ─────────────────────────
def fig_G_fold_stability(df):
    """Reliability of the POOLED macro-F1(0-2): bar = pooled score (from the
    confusion matrix), error bar = 95% bootstrap CI (resampling walks). Replaces
    the old between-subject SD — which was huge purely because each LOSO/MIDA fold
    tests a single subject (constant label → near-binary per-fold F1), NOT because
    the headline score is unstable. The bootstrap CI shows the real reliability."""
    log.info("Figure G — pooled macro-F1(0-2) with 95%% bootstrap CI (LOSO/MIDA)")
    metric = "f1_macro_02"
    colors = dict(zip(MODELS, sns.color_palette("Set2", len(MODELS))))
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 6.2), sharey=True)
    any_data = False
    for ax, method in zip(axes, ["loso", "mida"]):
        n = len(MODELS)
        width = 0.8 / n
        drew = False
        for i, model in enumerate(MODELS):
            cms = _cm_paths(method, model)
            if not cms:
                continue
            base, lo, hi = [], [], []
            for ds in DATASETS:
                b, l, h = (_bootstrap_ci(*_cm_to_pairs(_read_cm(cms[ds])))
                           if ds in cms else (np.nan, np.nan, np.nan))
                base.append(b); lo.append(l); hi.append(h)
            base, lo, hi = map(np.array, (base, lo, hi))
            x = np.arange(len(DATASETS)) + (i - (n - 1) / 2) * width
            yerr = np.vstack([base - lo, hi - base])
            ax.bar(x, base, width=width, yerr=yerr, capsize=3,
                   color=colors[model], label=MODEL_LABELS[model])
            drew = True
        if not drew:
            ax.set_title(f"{METHOD_LABELS[method]}\n(no data)"); ax.axis("off"); continue
        any_data = True
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS, rotation=20)
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(METRIC_LABELS[metric] if ax is axes[0] else "")
    if any_data:
        axes[0].legend(title="Encoder")
    fig.subplots_adjust(top=0.91)
    fig.suptitle(f"Reliability of pooled performance — 95% bootstrap CI   "
                 f"({METRIC_LABELS[metric]})", fontweight="bold", fontsize=17.4, y=0.97)
    _save(fig, "G_bootstrap_ci", f"G_bootstrap_ci_{metric}")


# ── I — does heterogeneous auxiliary data help? (MIDA vs LOSO) ────────────────
def fig_I_mida_vs_loso(df):
    """MIDA re-runs the EXACT SAME LOSO fold structure on the target cohort,
    just with each fold's training set additionally augmented by the full
    auxiliary pool of the other 3 cohorts ("n and computation identical to
    LOSO" — Care-PD's verbatim MIDA protocol). So any LOSO→MIDA difference is
    cleanly attributable to that auxiliary data — this figure shows it
    directly, per encoder and per target cohort."""
    log.info("Figure I — MIDA vs LOSO: does auxiliary data help?")
    for metric in METRICS:
        rows = []
        for method in ("loso", "mida"):
            view = _scope_value_table(df, method, "default")
            if view.empty:
                continue
            view = view.copy()
            view["method"] = method
            rows.append(view)
        if not rows:
            continue
        plot_df = pd.concat(rows, ignore_index=True)
        plot_df["method"] = pd.Categorical(plot_df["method"], categories=["loso", "mida"], ordered=True)

        models_present = [m for m in MODELS if m in set(plot_df["model"])]
        if not models_present:
            continue
        fig, axes = plt.subplots(1, len(models_present),
                                 figsize=(5.9 * len(models_present), 5.7), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models_present):
            msub = plot_df[plot_df["model"] == model]
            sns.barplot(data=msub, x="dataset", y=metric, hue="method",
                        order=DATASETS, hue_order=["loso", "mida"],
                        palette=["#4C72B0", "#55A868"], ax=ax)
            ax.set_title(MODEL_LABELS[model], fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric] if ax is axes[0] else "")
            ax.tick_params(axis="x", rotation=20)
            ax.set_ylim(0, 1)
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, [METHOD_LABELS.get(x, x) for x in l], title="Protocol")
        fig.subplots_adjust(top=0.85)
        fig.suptitle(f"Effect of auxiliary data augmentation: MIDA vs. LOSO   ({METRIC_LABELS[metric]})",
                     fontweight="bold", fontsize=15.5, y=0.99)
        _save(fig, "I_mida_vs_loso", f"I_mida_vs_loso_{metric}")

        mida_default_models = set(plot_df.loc[plot_df["method"] == "mida", "model"]) \
                              if "mida" in set(plot_df["method"]) else set()
        mida_any_models = set(df.loc[df["method"] == "mida", "model"])
        missing = [m for m in MODELS if m not in mida_default_models]
        if not missing:
            continue
        if mida_any_models - mida_default_models:
            log.info("    ('default'-variant MIDA missing for %s — a MIDA run exists for %s "
                     "but as '%s'-variant, intentionally excluded here for a fair "
                     "apples-to-apples LOSO-vs-MIDA comparison; re-run once the "
                     "'default' (BiLSTM) MIDA results are available)",
                     ", ".join(MODEL_LABELS[m] for m in missing),
                     ", ".join(MODEL_LABELS[m] for m in (mida_any_models - mida_default_models)),
                     df.loc[(df["method"] == "mida") & (df["model"].isin(mida_any_models - mida_default_models)), "variant"].iloc[0])
        else:
            log.warning("    (no MIDA results yet for %s — re-run once "
                        "videomae_mida.py / vjepa_mida.py / dinov2_mida.py have produced output)",
                        ", ".join(MODEL_LABELS[m] for m in missing))


# ── H — per-protocol "all models at a glance" heatmap ─────────────────────────
def fig_H_protocol_heatmaps(df):
    """One heatmap per protocol (model × cohort), F1(0-2) and F1(0-3) side
    by side — directly mirrors Care-PD's own model-comparison figure for LODO.
    Each model appears as a single row using its default probe (BiLSTM for
    DINOv2, fixed linear probe for VideoMAE and V-JEPA)."""
    log.info("Figure H — per-protocol model×dataset heatmaps")
    for method in ("mida", "loso", "lodo"):
        view = df[(df["method"] == method) & (df["scope_kind"] == "dataset")
                  & (df["scope"] != "ALL (LODO mean)")].copy()
        if view.empty:
            log.info("    %s — no data, skipped", method.upper())
            continue

        view["row"] = view.apply(
            lambda r: MODEL_LABELS[r["model"]] if r["variant"] == "default"
            else f"{MODEL_LABELS[r['model']]} ({r['variant']})", axis=1)
        variant_rank = {v: i for i, v in enumerate(VARIANT_ORDER)}
        row_order = (view[["model", "variant", "row"]].drop_duplicates()
                     .assign(_vr=lambda d: d["variant"].map(variant_rank).fillna(99))
                     .sort_values(["model", "_vr"])["row"].tolist())

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 0.72 * len(row_order) + 2.8), sharey=True)
        for ax, metric in zip(axes, METRICS):
            mat = view.pivot_table(index="row", columns="scope", values=metric, aggfunc="first")
            mat = mat.reindex(index=row_order, columns=[d for d in DATASETS if d in mat.columns])
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                        cbar=False, linewidths=0.6, linecolor="white",
                        square=True, annot_kws={"fontsize": 16, "fontweight": "bold"}, ax=ax)
            ax.set_title(METRIC_LABELS[metric], fontsize=16)
            ax.set_xlabel("Test cohort", fontsize=15)
            ax.tick_params(axis="y", rotation=0, labelsize=14)
            ax.tick_params(axis="x", labelsize=14)
            if ax is axes[0]:
                ax.set_ylabel("Model", fontsize=15)
            else:
                # Rows are identical across both panels — repeating the (long)
                # "<Model> (<variant>)" labels here only crowds the gap between
                # the two heatmaps and visually overlaps the left panel's cells.
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
        fig.subplots_adjust(wspace=0.05, top=0.91)
        fig.suptitle(f"{METHOD_LABELS[method]}: all models and variants at a glance",
                     fontweight="bold", fontsize=17.4, y=0.97)
        _save(fig, "H_protocol_heatmaps", f"H_{method}_model_dataset_heatmap")


# ── K — Dataset difficulty overview ──────────────────────────────────────────
# Hard-coded structural stats extracted from the LOSO result JSONs:
#   subjects, clips, clips/subject, UPDRS class distribution, per-protocol
#   performance (averaged over the three encoders' default probes).
_DS_STATS = {
    #           n_subj  n_clips  cls_dist (proportions 0-3)
    "PD-GaM":   (30,    2401,    [0.338, 0.358, 0.288, 0.015]),
    "BMClab":   (23,    893,     [0.382, 0.310, 0.308, 0.000]),
    "3DGait":   (43,    210,     [0.162, 0.424, 0.210, 0.205]),
    "T-SDU-PD": (14,    931,     [0.192, 0.287, 0.521, 0.000]),
}
_DS_LABELS = {
    "PD-GaM":   "PD-GaM",
    "BMClab":   "BMClab",
    "3DGait":   "3DGait",
    "T-SDU-PD": "T-SDU-PD",
}
_CLASS_COLORS = ["#4393c3", "#74c476", "#fd8d3c", "#d73027"]   # 0 1 2 3
_CLASS_LABELS = ["UPDRS 0", "UPDRS 1", "UPDRS 2", "UPDRS 3"]


def fig_K_dataset_overview(df):
    """Three-panel dataset characterisation figure intended to precede the model
    evaluation sections in the thesis.

    Panel A — stacked bar: UPDRS severity class composition per cohort.
    Panel B — dot/bar: subjects (filled marker) and clips/subject (open marker).
    Panel C — grouped bar: mean F1(0-2) per cohort × protocol (default probes,
               averaged over the three encoders), showing how difficulty ranking
               changes across protocols.
    """
    log.info("Figure K — dataset difficulty overview (3 panels)")

    # ── Panel C data: mean F1(0-2) per dataset × protocol ────────────────────
    proto_means = {ds: {} for ds in DATASETS}
    for method in ["loso", "lodo", "mida", "cda"]:
        scope_kind = "pair" if method == "cda" else "dataset"
        sub = df[(df["method"] == method) & (df["variant"] == "default")
                 & (df["scope_kind"] == scope_kind)]
        for ds in DATASETS:
            if method == "cda":
                vals = sub.loc[sub["target"] == ds, "f1_macro_02"]
            else:
                vals = sub.loc[sub["scope"] == ds, "f1_macro_02"]
            proto_means[ds][method] = float(vals.mean()) if not vals.empty else float("nan")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18.5, 6.2))
    gs  = fig.add_gridspec(1, 3, wspace=0.38,
                           width_ratios=[1.15, 0.85, 1.35])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    ds_order = DATASETS   # left→right: PD-GaM, BMClab, 3DGait, T-SDU-PD
    x        = np.arange(len(ds_order))
    bar_w    = 0.55

    # ── Panel A: stacked class distribution ───────────────────────────────────
    bottoms = np.zeros(len(ds_order))
    for ci, (color, label) in enumerate(zip(_CLASS_COLORS, _CLASS_LABELS)):
        vals = np.array([_DS_STATS[ds][2][ci] for ds in ds_order])
        ax_a.bar(x, vals * 100, bar_w, bottom=bottoms,
                        color=color, label=label, edgecolor="white", linewidth=0.5)
        # Annotate non-trivial slices (≥ 5 %)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.05:
                ax_a.text(xi, b + v * 50, f"{v*100:.0f}%",
                          ha="center", va="center", fontsize=10.9,
                          color="white", fontweight="bold")
        bottoms += vals * 100

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([_DS_LABELS[d] for d in ds_order], fontsize=13)
    ax_a.set_ylabel("Proportion of clips (%)", fontsize=13)
    ax_a.set_ylim(0, 105)
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax_a.legend(fontsize=11.6, loc="upper right", frameon=True,
                handlelength=1.0, handletextpad=0.4)
    ax_a.set_title("Severity class composition", fontsize=14.5, fontweight="bold")

    # ── Panel B: subjects + clips/subject ─────────────────────────────────────
    n_subj         = [_DS_STATS[ds][0] for ds in ds_order]
    clips_per_subj = [_DS_STATS[ds][1] / _DS_STATS[ds][0] for ds in ds_order]

    ax_b2 = ax_b.twinx()

    c_subj = "#5e81ac"
    c_clip = "#bf616a"
    ax_b.bar(x, n_subj, bar_w, color=c_subj, alpha=0.85,
             label="Subjects", edgecolor="white", linewidth=0.5)
    ax_b2.plot(x, clips_per_subj, "D--", color=c_clip, ms=7,
               lw=1.6, label="Clips / subject", zorder=3)

    for xi, (s, c) in enumerate(zip(n_subj, clips_per_subj)):
        ax_b.text(xi, s + 0.6, str(s), ha="center", va="bottom",
                  fontsize=12.3, color=c_subj, fontweight="bold")
        ax_b2.text(xi, c + 1.8, f"{c:.0f}", ha="center", va="bottom",
                   fontsize=12.3, color=c_clip, fontweight="bold")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([_DS_LABELS[d] for d in ds_order], fontsize=13)
    ax_b.set_ylabel("Number of subjects", fontsize=13, color=c_subj)
    ax_b.tick_params(axis="y", labelcolor=c_subj)
    ax_b2.set_ylabel("Clips per subject", fontsize=13, color=c_clip)
    ax_b2.tick_params(axis="y", labelcolor=c_clip)
    ax_b.set_ylim(0, 58)
    ax_b2.set_ylim(0, 110)

    lines_a, labs_a = ax_b.get_legend_handles_labels()
    lines_b, labs_b = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines_a + lines_b, labs_a + labs_b,
                fontsize=11.6, loc="upper right", frameon=True,
                handlelength=1.2, handletextpad=0.4)
    ax_b.set_title("Scale and recording density", fontsize=14.5, fontweight="bold")

    # ── Panel C: F1(0-2) per cohort × protocol ────────────────────────────────
    protocols  = ["loso", "mida", "cda", "lodo"]
    proto_lbls = {"loso": "LOSO", "lodo": "LODO", "cda": "CDA (col)", "mida": "MIDA"}
    n_proto    = len(protocols)
    group_w    = 0.72
    bw         = group_w / n_proto
    offsets    = np.linspace(-(group_w - bw) / 2, (group_w - bw) / 2, n_proto)
    palette    = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for pi, (proto, color) in enumerate(zip(protocols, palette)):
        vals = [proto_means[ds].get(proto, float("nan")) for ds in ds_order]
        ax_c.bar(x + offsets[pi], vals, bw * 0.92,
                        color=color, label=proto_lbls[proto],
                        edgecolor="white", linewidth=0.4)
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax_c.text(x[xi] + offsets[pi], v + 0.008, f"{v:.2f}",
                          ha="center", va="bottom", fontsize=9.9, color=color,
                          fontweight="bold", rotation=90)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([_DS_LABELS[d] for d in ds_order], fontsize=13)
    ax_c.set_ylabel("Macro F1  (labels 0–2),  averaged over 3 encoders", fontsize=13)
    ax_c.set_ylim(0, 0.72)
    ax_c.legend(fontsize=12.3, loc="upper right", frameon=True,
                ncol=2, handlelength=1.0, handletextpad=0.4)
    ax_c.set_title("Performance profile across protocols", fontsize=14.5, fontweight="bold")
    ax_c.axhline(0, color="black", linewidth=0.6)

    fig.suptitle("Dataset characterisation: structure and difficulty profile",
                 fontweight="bold", fontsize=17.4, y=1.02)
    fig.tight_layout()
    _save(fig, "K_dataset_overview", "K_dataset_overview")


# ── J — LODO vs LOSO delta heatmap ───────────────────────────────────────────
def fig_J_lodo_loso_delta(df):
    """Diverging heatmap of Δ = LODO − LOSO per cohort (rows) and encoder (cols).
    Blue cells = LODO outperforms LOSO; red cells = LOSO outperforms LODO.
    Symmetric colour scale ±0.35; annotations show the signed delta."""
    log.info("Figure J — LODO vs. LOSO delta heatmap")

    loso = df[(df["method"] == "loso") & (df["variant"] == "default")
              & (df["scope_kind"] == "dataset")]
    lodo = df[(df["method"] == "lodo") & (df["variant"] == "default")
              & (df["scope_kind"] == "dataset")]
    if loso.empty or lodo.empty:
        log.warning("  LOSO or LODO data missing — skipping figure J")
        return

    for metric in METRICS:
        # Build rows=cohort, cols=model delta matrix
        delta = pd.DataFrame(index=DATASETS, columns=MODELS, dtype=float)
        for model in MODELS:
            for ds in DATASETS:
                lo = loso.loc[(loso["model"] == model) & (loso["scope"] == ds), metric]
                ld = lodo.loc[(lodo["model"] == model) & (lodo["scope"] == ds), metric]
                if not lo.empty and not ld.empty:
                    delta.loc[ds, model] = float(ld.iloc[0]) - float(lo.iloc[0])

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        # Signed annotations: "+0.27" / "−0.23"
        annot = delta.apply(
            lambda col: col.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
        )

        sns.heatmap(
            delta.astype(float), annot=annot, fmt="",
            cmap="RdBu", vmin=-0.35, vmax=0.35, center=0,
            linewidths=0.8, linecolor="white", square=True,
            annot_kws={"fontsize": 16, "fontweight": "bold"},
            cbar_kws={"label": "Δ F1  (LODO − LOSO)", "shrink": 0.85},
            ax=ax,
        )

        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                           fontsize=14.5, rotation=25, ha="right")
        ax.set_yticklabels(DATASETS, rotation=0, fontsize=14.5)
        ax.set_xlabel("Encoder", fontsize=14.5, labelpad=6)
        ax.set_ylabel("Held-out cohort", fontsize=14.5, labelpad=6)
        ax.tick_params(axis="x", bottom=False)
        ax.tick_params(axis="y", left=False)

        ax.set_title(
            f"LODO vs. LOSO performance delta   ({METRIC_LABELS[metric]})",
            fontsize=17.4, fontweight="bold", pad=10,
        )
        fig.tight_layout()
        _save(fig, "J_lodo_loso_delta", f"J_lodo_loso_delta_{metric}")


# ── L — LOSO | LODO | Δ combined (1 × 3) ────────────────────────────────────
def fig_L_loso_lodo_combined(df):
    """Thesis-ready 1×3 combined figure:
      [LOSO heatmap] | [LODO heatmap] | [Δ LODO−LOSO diverging heatmap]

    All three panels share the same axis orientation (rows = encoder,
    cols = cohort) so that any row or column can be read continuously
    across the full figure.

    Panels 1/2 replicate fig_H style (Blues, cbar=False, linewidths 0.6).
    Panel 3 replicates fig_J style (RdBu, signed annotations, cbar).
    One figure per metric (F1 0-2 and F1 0-3)."""
    log.info("Figure L — combined LOSO | LODO | delta (1×3)")

    loso_sub = df[(df["method"] == "loso") & (df["variant"] == "default")
                  & (df["scope_kind"] == "dataset")]
    lodo_sub = df[(df["method"] == "lodo") & (df["variant"] == "default")
                  & (df["scope_kind"] == "dataset")]
    if loso_sub.empty or lodo_sub.empty:
        log.warning("  LOSO or LODO data missing — skipping figure L")
        return

    row_labels = [MODEL_LABELS[m] for m in MODELS]

    for metric in METRICS:
        # Build rows=encoders, cols=datasets matrices
        loso_mat  = pd.DataFrame(index=row_labels, columns=DATASETS, dtype=float)
        lodo_mat  = pd.DataFrame(index=row_labels, columns=DATASETS, dtype=float)
        delta_mat = pd.DataFrame(index=row_labels, columns=DATASETS, dtype=float)

        for model in MODELS:
            lbl = MODEL_LABELS[model]
            for ds in DATASETS:
                lo = loso_sub.loc[(loso_sub["model"] == model) & (loso_sub["scope"] == ds), metric]
                ld = lodo_sub.loc[(lodo_sub["model"] == model) & (lodo_sub["scope"] == ds), metric]
                if not lo.empty:
                    loso_mat.loc[lbl, ds]  = float(lo.iloc[0])
                if not ld.empty:
                    lodo_mat.loc[lbl, ds]  = float(ld.iloc[0])
                if not lo.empty and not ld.empty:
                    delta_mat.loc[lbl, ds] = float(ld.iloc[0]) - float(lo.iloc[0])

        # ── Layout ───────────────────────────────────────────────────────────
        # Panel 3 is slightly wider to accommodate the colorbar
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3,
            figsize=(18.5, 0.72 * len(MODELS) + 3.0),
            gridspec_kw={"wspace": 0.30, "width_ratios": [1, 1, 1.28]},
        )

        # ── Shared kwargs matching fig_H exactly ─────────────────────────────
        h_kw = dict(annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, linewidths=0.6, linecolor="white",
                    square=True, annot_kws={"fontsize": 16, "fontweight": "bold"})

        # ── Panel 1: LOSO ────────────────────────────────────────────────────
        sns.heatmap(loso_mat.astype(float), ax=ax1, **h_kw)
        ax1.set_title("LOSO", fontsize=17.4, fontweight="bold", pad=6)
        ax1.set_xlabel("Test cohort", fontsize=15)
        ax1.set_ylabel("Encoder", fontsize=15)
        ax1.tick_params(axis="y", rotation=0, labelsize=14)
        ax1.tick_params(axis="x", rotation=25, labelsize=14)

        # ── Panel 2: LODO ────────────────────────────────────────────────────
        sns.heatmap(lodo_mat.astype(float), ax=ax2, **h_kw)
        ax2.set_title("LODO", fontsize=17.4, fontweight="bold", pad=6)
        ax2.set_xlabel("Test cohort", fontsize=15)
        ax2.set_ylabel("")
        ax2.tick_params(axis="y", left=False, labelleft=False)
        ax2.tick_params(axis="x", rotation=25, labelsize=14)

        # ── Panel 3: Δ LODO − LOSO (RdBu, matching fig_J) ───────────────────
        delta_annot = delta_mat.apply(
            lambda col: col.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
        )
        sns.heatmap(
            delta_mat.astype(float), annot=delta_annot, fmt="",
            cmap="RdBu", vmin=-0.35, vmax=0.35, center=0,
            linewidths=0.8, linecolor="white", square=True,
            annot_kws={"fontsize": 16, "fontweight": "bold"},
            cbar_kws={"label": "Δ F1  (LODO − LOSO)", "shrink": 0.82},
            ax=ax3,
        )
        ax3.set_title("LODO − LOSO  (Δ)", fontsize=17.4, fontweight="bold", pad=6)
        ax3.set_xlabel("Test cohort", fontsize=15)
        ax3.set_ylabel("")
        ax3.tick_params(axis="y", left=False, labelleft=False, labelsize=14)
        ax3.tick_params(axis="x", rotation=25, bottom=False, labelsize=14)

        fig.suptitle(
            f"Within-dataset (LOSO) and leave-one-dataset-out (LODO) performance"
            f"   ({METRIC_LABELS[metric]})",
            fontweight="bold", fontsize=17.4, y=0.99,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(fig, "L_loso_lodo_combined", f"L_loso_lodo_combined_{metric}")


# ── Confusion-matrix analysis (reads the *_cm_*.csv files) ────────────────────
_TRUE = [f"true_{i}" for i in range(4)]
_PRED = [f"pred_{i}" for i in range(4)]


def _read_cm(path):
    """One confusion-matrix CSV → 4×4 float array (rows=true, cols=pred)."""
    m = pd.read_csv(path, index_col=0).reindex(index=_TRUE, columns=_PRED)
    return m.fillna(0).to_numpy(dtype=float)


def _cm_paths(method, model, variant="default"):
    """Latest CM CSV per dataset for (method, model, variant) → {dataset: path}.
    Picks the folder whose name parses to this model+variant (so DINOv2 resolves
    to its BiLSTM folder, consistent with the rest of this script)."""
    proto = RESULTS_ROOT / method
    if not proto.is_dir():
        return {}
    folder = next((d for d in proto.iterdir()
                   if d.is_dir() and parse_model_variant(d.name, method) == (model, variant)), None)
    if folder is None:
        return {}
    ds_re = re.compile(r"_(" + "|".join(re.escape(d) for d in DATASETS) + r")_cm_")
    by_ds = {}
    for p in folder.glob("*_cm_*.csv"):
        mds = ds_re.search(p.name)
        ds = mds.group(1) if mds else "ALL"      # LODO writes one pooled CM (no dataset tag)
        tm = TS_RE.search(p.stem)
        ts = tm.group(1) if tm else "0"
        if ds not in by_ds or ts > by_ds[ds][1]:
            by_ds[ds] = (p, ts)
    return {ds: pq[0] for ds, pq in by_ds.items()}


def _pooled_cm(method, model, variant="default"):
    """Sum CMs for (method, model) into one 4×4 array.

    LOSO/MIDA/LODO store one *_cm_*.csv per dataset. CDA has no per-dataset CM
    file — its confusion matrices live inside the run JSON, one per source→target
    pair — so for CDA we sum the per-pair `cm` arrays straight from that JSON
    (pooled over all 12 transfer pairs), giving the same 4×4 pooled view."""
    if method == "cda":
        return _pooled_cm_cda(model, variant)
    mat = np.zeros((4, 4))
    for p in _cm_paths(method, model, variant).values():
        mat += _read_cm(p)
    return mat


def _pooled_cm_cda(model, variant="default"):
    """Sum the per-pair confusion matrices from a CDA run JSON into one 4×4."""
    proto = RESULTS_ROOT / "cda"
    if not proto.is_dir():
        return np.zeros((4, 4))
    folder = next((d for d in proto.iterdir()
                   if d.is_dir() and parse_model_variant(d.name, "cda") == (model, variant)), None)
    if folder is None:
        return np.zeros((4, 4))
    js = sorted(folder.glob("*_cda*.json"))
    if not js:
        return np.zeros((4, 4))
    data = json.load(open(js[-1]))
    mat = np.zeros((4, 4))
    for p in data.get("pairs", []):
        mat += np.array(p["cm"], dtype=float)
    return mat


def _cm_to_pairs(mat):
    """4×4 counts → flat (y_true, y_pred) integer arrays."""
    yt, yp = [], []
    for t in range(4):
        for p in range(4):
            n = int(round(mat[t, p]))
            yt += [t] * n
            yp += [p] * n
    return np.array(yt), np.array(yp)


def _bootstrap_ci(yt, yp, n_boot=1000, seed=0):
    """Pooled macro-F1(0-2) + 95% bootstrap CI (resampling walks with replacement)."""
    base = macro_f1_02(yt, yp) if len(yt) else float("nan")
    if len(yt) == 0:
        return base, base, base
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yt))
    vals = [macro_f1_02(yt[s], yp[s])
            for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return base, lo, hi


# ── CM1 — confusion matrices per encoder ──────────────────────────────────────
def fig_CM_confusion(df, method="loso"):
    """Where does each encoder confuse severity grades? Row-normalised confusion
    matrix (colour = recall per true grade) with raw walk counts annotated,
    pooled across cohorts. Reveals adjacent (1↔2) confusions and whether the
    extreme grades (0, 3) are ever predicted."""
    log.info("Figure CM — confusion matrices per encoder (%s)", method)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 5.0))
    lab = ["0", "1", "2", "3"]
    drew = False
    for ax, model in zip(axes, MODELS):
        mat = _pooled_cm(method, model)
        if mat.sum() == 0:
            ax.set_title(f"{MODEL_LABELS[model]}\n(no data)"); ax.axis("off"); continue
        drew = True
        row = mat.sum(1, keepdims=True)
        norm = np.divide(mat, row, out=np.zeros_like(mat), where=row > 0)
        sns.heatmap(norm, annot=mat.astype(int), fmt="d", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, xticklabels=lab, yticklabels=lab,
                    linewidths=0.6, linecolor="white", square=True,
                    annot_kws={"fontsize": 18, "fontweight": "bold"}, ax=ax)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("Predicted UPDRS")
        ax.set_ylabel("True UPDRS" if ax is axes[0] else "")
        ax.tick_params(axis="y", rotation=0)
    if not drew:
        plt.close(fig); return
    fig.subplots_adjust(top=0.88)
    fig.suptitle(f"Confusion matrices — {METHOD_LABELS[method]}   "
                 f"(colour = recall, number = walk count)",
                 fontweight="bold", fontsize=16, y=0.97)
    _save(fig, "CM_confusion", f"CM_confusion_{method}")


# ── CM_all — every encoder × protocol in one grid ─────────────────────────────
def fig_CM_confusion_all(df, methods=("loso", "mida", "lodo", "cda")):
    """All confusion matrices in a single figure: rows = encoder, cols = protocol.
    Row-normalised (colour = recall per true grade), raw walk counts annotated.
    UPDRS tick labels are drawn only on the outer edges (bottom row, left column),
    not on every panel. One shared colour bar."""
    log.info("Figure CM_all — combined confusion grid (%d models × %d protocols)",
             len(MODELS), len(methods))
    nrow, ncol = len(MODELS), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.6 * nrow),
                             squeeze=False)
    lab = ["0", "1", "2", "3"]
    mesh = None
    for ri, model in enumerate(MODELS):
        for ci, method in enumerate(methods):
            ax = axes[ri][ci]
            mat = _pooled_cm(method, model)
            if mat.sum() == 0:
                ax.set_title(f"{METHOD_LABELS[method]}\n(no data)" if ri == 0 else "(no data)")
                ax.axis("off"); continue
            row = mat.sum(1, keepdims=True)
            norm = np.divide(mat, row, out=np.zeros_like(mat), where=row > 0)
            hm = sns.heatmap(norm, annot=mat.astype(int), fmt="d", cmap="Blues",
                             vmin=0, vmax=1, cbar=False,
                             xticklabels=(lab if ri == nrow - 1 else False),
                             yticklabels=(lab if ci == 0 else False),
                             linewidths=0.6, linecolor="white", square=True,
                             annot_kws={"fontsize": 14, "fontweight": "bold"}, ax=ax)
            mesh = hm.collections[0]
            # protocol names only on the top row
            if ri == 0:
                ax.set_title(METHOD_LABELS[method], fontweight="bold", fontsize=15)
            # encoder label + axis title only on the left column
            if ci == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n\nTrue UPDRS",
                              fontweight="bold", fontsize=13)
            else:
                ax.set_ylabel("")
            # x-axis title only on the bottom row
            ax.set_xlabel("Predicted UPDRS" if ri == nrow - 1 else "")
            ax.tick_params(axis="y", rotation=0)
    if mesh is None:
        plt.close(fig); return
    fig.suptitle("Confusion matrices — all encoders × protocols   "
                 "(colour = recall per true grade, number = walk count)",
                 fontweight="bold", fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.93, 0.985])
    cax = fig.add_axes([0.945, 0.12, 0.013, 0.74])
    fig.colorbar(mesh, cax=cax, label="Recall (row-normalised)")
    _save(fig, "CM_confusion", "CM_confusion_all")


# ── CM2 — ordinal error analysis (off-by-k + direction) ───────────────────────
def fig_err_distance(df, method="loso"):
    """Two ordinal-error views per encoder (pooled over cohorts):
    LEFT  — how far off: exact / ±1 grade / ≥2 grades (is it 'usually one off'?)
    RIGHT — direction: under- vs over-grading (does it systematically under-rate?)."""
    log.info("Figure ERR — ordinal error distance/direction (%s)", method)
    rows = []
    for model in MODELS:
        yt, yp = _cm_to_pairs(_pooled_cm(method, model))
        if len(yt) == 0:
            continue
        err = yp - yt
        rows.append({"model": MODEL_LABELS[model],
                     "exact": np.mean(err == 0), "±1": np.mean(np.abs(err) == 1),
                     "≥2": np.mean(np.abs(err) >= 2),
                     "under": np.mean(err < 0), "over": np.mean(err > 0)})
    if not rows:
        return
    d = pd.DataFrame(rows).set_index("model")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    d[["exact", "±1", "≥2"]].plot(kind="bar", stacked=True, ax=ax1,
                                  color=["#2c7bb6", "#fdae61", "#d7191c"])
    ax1.set_title("How far off  (|pred − true|)", fontweight="bold")
    ax1.set_ylabel("fraction of walks"); ax1.set_ylim(0, 1); ax1.tick_params(axis="x", rotation=15)
    d[["under", "over"]].plot(kind="bar", ax=ax2, color=["#8073ac", "#e08214"])
    ax2.set_title("Direction  (under- vs over-grading)", fontweight="bold")
    ax2.set_ylabel("fraction of walks"); ax2.set_ylim(0, 1); ax2.tick_params(axis="x", rotation=15)
    ax2.legend(["under (pred<true)", "over (pred>true)"])
    fig.subplots_adjust(top=0.91)
    fig.suptitle(f"Ordinal error analysis — {METHOD_LABELS[method]}",
                 fontweight="bold", fontsize=16, y=0.97)
    _save(fig, "ERR_ordinal", f"ERR_ordinal_{method}")


# ── CM3 — per-cohort ordinal error profile (why is a cohort hard?) ────────────
def fig_dataset_errors(df, method="loso"):
    """Per-cohort ordinal error profile, pooled over ALL encoders: which cohorts
    are intrinsically hard — lower exact rate / more ≥2-grade errors? (e.g. 3DGait)
    Only protocols with per-dataset CMs (LOSO/MIDA)."""
    log.info("Figure ERR-by-dataset — per-cohort error profile (%s)", method)
    rows = []
    for ds in DATASETS:
        mat = np.zeros((4, 4))
        for model in MODELS:
            cms = _cm_paths(method, model)
            if ds in cms:
                mat += _read_cm(cms[ds])
        yt, yp = _cm_to_pairs(mat)
        if len(yt) == 0:
            continue
        e = yp - yt
        rows.append({"dataset": ds, "exact": np.mean(e == 0),
                     "±1": np.mean(np.abs(e) == 1), "≥2": np.mean(np.abs(e) >= 2)})
    if not rows:
        return
    d = pd.DataFrame(rows).set_index("dataset")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    d[["exact", "±1", "≥2"]].plot(kind="bar", stacked=True, ax=ax,
                                  color=["#2c7bb6", "#fdae61", "#d7191c"])
    ax.set_title(f"Per-cohort ordinal error profile — {METHOD_LABELS[method]} "
                 f"(all encoders pooled)", fontweight="bold")
    ax.set_ylabel("fraction of walks"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=15)
    ax.legend(["exact", "±1", "≥2"], title="|pred − true|")
    _save(fig, "ERR_by_dataset", f"ERR_by_dataset_{method}")


# ── C — V-JEPA v1 vs V-JEPA 2 ─────────────────────────────────────────────────
def fig_C_vjepa_comparison(df):
    """Head-to-head: does the newer V-JEPA 2 backbone beat V-JEPA v1? Same plot
    style as A, but restricted to the two V-JEPA encoders, one panel per
    protocol, per cohort."""
    log.info("Figure C — V-JEPA v1 vs V-JEPA 2")
    pair = ["vjepa", "vjepa2"]
    palette = {"vjepa": "#8da0cb", "vjepa2": "#fc8d62"}
    for metric in METRICS:
        fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.5), sharey=True)
        handles = labels = None
        for ax, method in zip(axes, METHODS):
            view = _scope_value_table(df, method, "default")
            view = view[view["model"].isin(pair)]
            if view.empty:
                ax.set_title(f"{METHOD_LABELS[method]}\n(no data)")
                ax.axis("off")
                continue
            sns.barplot(data=view, x="dataset", y=metric, hue="model",
                        order=DATASETS, hue_order=pair, palette=palette, ax=ax)
            ax.set_title(METHOD_LABELS[method], fontsize=15.9, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric] if ax is axes[0] else "")
            ax.tick_params(axis="x", rotation=28)
            ax.set_ylim(0, 1)
            h, l = _take_legend(ax)
            if handles is None and h:
                handles, labels = h, l
        if handles:
            fig.legend(handles, [MODEL_LABELS.get(x, x) for x in labels], title="Encoder",
                       loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
        fig.suptitle(f"V-JEPA v1 vs V-JEPA 2 per protocol   ({METRIC_LABELS[metric]})",
                     fontweight="bold", fontsize=17.4, y=1.13)
        _save(fig, "C_vjepa_comparison", f"C_vjepa_comparison_{metric}")


# ── C2 — V-JEPA 2: 16 vs 64 frames ────────────────────────────────────────────
def fig_vjepa2_frames(df):
    """V-JEPA 2 at 16 frames (stride 4 — fits the 81-frame clip, matched to
    VideoMAE/v1) vs its 64-frame run (which fell back to dense linspace). Shows
    whether the input frame count matters at all. Needs both result folders:
    results/{method}/vjepa2 (=16f, variant 'default') and vjepa2_64f ('64f')."""
    log.info("Figure C2 — V-JEPA 2: 16 vs 64 frames")
    pairs = [("default", "16 frames"), ("64f", "64 frames")]
    colors = {"16 frames": "#fc8d62", "64 frames": "#8da0cb"}
    for metric in METRICS:
        fig, axes = plt.subplots(1, len(METHODS), figsize=(20.8, 5.5), sharey=True)
        any_data = False
        for ax, method in zip(axes, METHODS):
            x = np.arange(len(DATASETS)); w = 0.38; drew = False
            for i, (variant, label) in enumerate(pairs):
                t = _scope_value_table(df, method, variant)
                t = t[t["model"] == "vjepa2"].set_index("dataset")
                if t.empty:
                    continue
                vals = [t[metric].get(d, np.nan) if d in t.index else np.nan for d in DATASETS]
                ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=colors[label])
                drew = True
            if not drew:
                ax.set_title(f"{METHOD_LABELS[method]}\n(no data)"); ax.axis("off"); continue
            any_data = True
            ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=25)
            ax.set_title(METHOD_LABELS[method], fontweight="bold")
            ax.set_ylim(0, 1)
            ax.set_ylabel(METRIC_LABELS[metric] if ax is axes[0] else "")
        if not any_data:
            plt.close(fig); continue
        axes[0].legend(title="V-JEPA 2 input")
        fig.suptitle(f"V-JEPA 2 — 16 vs 64 frames per protocol   ({METRIC_LABELS[metric]})",
                     fontweight="bold", fontsize=15.5, y=1.02)
        _save(fig, "C2_vjepa2_frames", f"C2_vjepa2_frames_{metric}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_master_df()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_K_dataset_overview(df)
    fig_A_model_comparison(df)
    fig_C_vjepa_comparison(df)
    fig_vjepa2_frames(df)
    fig_B_protocol_comparison(df)
    fig_D_transfer_matrices(df)
    fig_D_combined_transfer_matrices(df)
    fig_E_dataset_difficulty(df)
    fig_G_fold_stability(df)
    for _m in ("loso", "mida", "lodo", "cda"):
        fig_CM_confusion(df, _m)
        fig_err_distance(df, _m)
    fig_CM_confusion_all(df)
    for _m in ("loso", "mida"):
        fig_dataset_errors(df, _m)
    fig_I_mida_vs_loso(df)
    fig_J_lodo_loso_delta(df)
    fig_L_loso_lodo_combined(df)
    fig_H_protocol_heatmaps(df)

    n_png = sum(1 for _ in FIGURES_DIR.rglob("*.png"))
    log.info("\nDone — %d figures written under %s", n_png, FIGURES_DIR)


if __name__ == "__main__":
    main()
