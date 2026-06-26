#!/usr/bin/env python
"""
baseline_comparison.py
=======================
Compare this project's frozen visual-foundation-model (VFM) macro-F1(0-2)
results against the Care-PD paper (frozen *pose* encoders, their Fig. 3) for
two evaluation protocols: LOSO (within-dataset) and CDA (cross-dataset).

Care-PD only publishes its F1(0-2) numbers as heat-maps (Fig. 3), never as a
table.  The 4x4 transfer matrices below were transcribed from that figure.
  rows  = train / source cohort   (BMClab, PD-GaM, T-SDU-PD, 3DGait)
  cols  = test  / target cohort   (BMClab, PD-GaM, T-SDU-PD, 3DGait)
  diagonal      = LOSO  (within-dataset)
  off-diagonal  = cross-dataset; per-target CDA = mean of that column's
                  three off-diagonal cells (matches how our own CDA is
                  aggregated: one score per target, averaged over sources).

Outputs (separately for CDA and LOSO)
-------------------------------------
1. Care-PD baseline values:
     assets/baseline/baseline_F1(0_2)_values_<PROTOCOL>.csv
2. Comparison (our 3 VFMs + Care-PD encoder mean + MoMask), per dataset:
     outputs/runs/<run>/results/_baseline_comparison/baseline_F1(0_2)_comparison_<PROTOCOL>.csv

Usage
-----
    cd main_project
    python scripts/_evaluation/baseline_comparison.py
"""
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import output_root          # honours $BEP_OUTPUT_ROOT (per-run folder)
ASSETS_BASE  = PROJECT_ROOT / "assets" / "baseline"
OVERVIEW_DIR = output_root() / "aggregate"            # reads the aggregated overview xlsx
OUT_DIR      = output_root() / "results" / "_baseline_comparison"

# ── Cohorts ───────────────────────────────────────────────────────────────────
DATASETS  = ["PD-GaM", "BMClab", "3DGait", "T-SDU-PD"]   # our display order
CPD_ORDER = ["BMClab", "PD-GaM", "T-SDU-PD", "3DGait"]    # Care-PD Fig.3 axis order

# ── Our encoders & DINOv2 variant selection per protocol ──────────────────────
#   VideoMAE / V-JEPA  -> variant "default"
#   DINOv2 (BiLSTM)    -> "default" under LOSO, "bilstm" under CDA
# V-JEPA 2 is intentionally excluded here — the v1-vs-v2 comparison lives in its
# own dedicated plot (plot_results.py fig_C), not in the Care-PD baseline table.
OUR_MODELS = {"videomae": "VideoMAE", "dinov2": "DINOv2", "vjepa2": "V-JEPA 2"}

# ── Care-PD Fig. 3 transfer matrices (macro-F1 0-2) ───────────────────────────
CAREPD_MATRICES = {
    "MixSTE":         [[0.67, 0.47, 0.37, 0.09],
                       [0.45, 0.67, 0.25, 0.09],
                       [0.46, 0.44, 0.41, 0.27],
                       [0.21, 0.26, 0.30, 0.43]],
    "MotionAGFormer": [[0.67, 0.51, 0.38, 0.12],
                       [0.37, 0.65, 0.27, 0.09],
                       [0.48, 0.42, 0.39, 0.15],
                       [0.30, 0.29, 0.30, 0.32]],
    "MotionBERT":     [[0.68, 0.49, 0.34, 0.11],
                       [0.35, 0.66, 0.24, 0.09],
                       [0.56, 0.41, 0.36, 0.15],
                       [0.31, 0.26, 0.28, 0.38]],
    "PoseFormerV2":   [[0.60, 0.52, 0.31, 0.14],
                       [0.40, 0.66, 0.23, 0.09],
                       [0.40, 0.42, 0.38, 0.28],
                       [0.32, 0.35, 0.20, 0.26]],
    "POTR":           [[0.45, 0.44, 0.35, 0.31],
                       [0.41, 0.49, 0.34, 0.34],
                       [0.27, 0.38, 0.38, 0.29],
                       [0.21, 0.23, 0.29, 0.34]],
    "MoMask":         [[0.57, 0.25, 0.31, 0.30],
                       [0.67, 0.64, 0.44, 0.45],
                       [0.44, 0.46, 0.54, 0.44],
                       [0.57, 0.41, 0.33, 0.35]],
    "MotionCLIP":     [[0.54, 0.49, 0.33, 0.27],
                       [0.44, 0.65, 0.27, 0.10],
                       [0.37, 0.45, 0.37, 0.18],
                       [0.24, 0.30, 0.25, 0.24]],
    # handcrafted baseline — kept for reference, excluded from the encoder mean
    "GaitFeatures+RF":[[0.66, 0.46, 0.41, 0.35],
                       [0.44, 0.38, 0.34, 0.33],
                       [0.42, 0.32, 0.30, 0.48],
                       [0.17, 0.18, 0.11, 0.28]],
}
LEARNED_ENCODERS = ["MixSTE", "MotionAGFormer", "MotionBERT", "PoseFormerV2",
                    "POTR", "MoMask", "MotionCLIP"]


# ── Care-PD value extraction ──────────────────────────────────────────────────
def _carepd_per_encoder(protocol):
    """Return {encoder: {dataset: value}} for 'loso' or 'cda'."""
    out = {}
    for name, mat in CAREPD_MATRICES.items():
        a = np.array(mat, dtype=float)
        d = {}
        for j, target in enumerate(CPD_ORDER):
            if protocol == "loso":
                d[target] = a[j, j]                         # diagonal
            else:  # cda = mean of off-diagonal cells in this target column
                d[target] = np.mean([a[i, j] for i in range(4) if i != j])
        out[name] = d
    return out


def build_carepd_values(protocol):
    """DataFrame: rows = encoders (+ Mean), cols = DATASETS."""
    per = _carepd_per_encoder(protocol)
    rows = {enc: [per[enc][ds] for ds in DATASETS] for enc in CAREPD_MATRICES}
    df = pd.DataFrame(rows, index=DATASETS).T          # encoders x datasets
    df = df.loc[list(CAREPD_MATRICES.keys())]          # keep declared order
    # encoder mean (7 learned encoders, excl. handcrafted baseline)
    df.loc["Mean (7 encoders)"] = df.loc[LEARNED_ENCODERS].mean(axis=0)
    return df.round(3)


# ── Our model values from the latest overview workbook ────────────────────────
def _latest_overview():
    files = sorted(glob.glob(str(OVERVIEW_DIR / "overview_all_results_*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No overview workbook in {OVERVIEW_DIR}")
    return files[-1]


def _select_dinov2_variant(df, protocol):
    """DINOv2 uses BiLSTM everywhere; the raw folder naming differs per protocol."""
    return "default" if protocol == "loso" else "bilstm"


def build_our_values(protocol):
    """DataFrame: rows = our 3 VFMs, cols = DATASETS (macro-F1 0-2)."""
    xf = _latest_overview()
    log.info("Reading our results from %s", Path(xf).name)
    ar = pd.read_excel(xf, sheet_name="all_results")
    sub = ar[ar["method"] == protocol].copy()

    rows = {}
    for model, label in OUR_MODELS.items():
        variant = _select_dinov2_variant(sub, protocol) if model == "dinov2" else "default"
        d = sub[(sub["model"] == model) & (sub["variant"] == variant)].copy()
        vals = {}
        if protocol == "loso":
            for ds in DATASETS:
                v = d[d["scope"] == ds]["f1_macro_02"]
                vals[ds] = float(v.iloc[0]) if len(v) else np.nan
        else:  # cda: scope is 'source → target'; aggregate per target
            d["target"] = d["scope"].astype(str).apply(
                lambda s: s.split("→")[-1].strip() if "→" in s else s)
            for ds in DATASETS:
                v = d[d["target"] == ds]["f1_macro_02"]
                vals[ds] = float(v.mean()) if len(v) else np.nan
        rows[label] = [vals[ds] for ds in DATASETS]

    return pd.DataFrame(rows, index=DATASETS).T[DATASETS].round(3)


# ── Comparison assembly ───────────────────────────────────────────────────────
def build_comparison(protocol):
    ours    = build_our_values(protocol)
    carepd  = build_carepd_values(protocol)
    comp = pd.concat([
        ours,
        carepd.loc[["Mean (7 encoders)"]].rename(index={"Mean (7 encoders)":
                                                         "Care-PD mean (7 enc.)"}),
        carepd.loc[["MoMask"]].rename(index={"MoMask": "Care-PD MoMask"}),
    ])
    comp.index.name = "method"
    return comp.round(3)


def render_tables_figure(out_path):
    """Render the LOSO and CDA comparison tables as one Times-New-Roman figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # Register Times New Roman explicitly (macOS path) so the figure is portable.
    for fp in ["/System/Library/Fonts/Supplemental/Times New Roman.ttf",
               "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
    plt.rcParams["font.family"]     = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    comps   = {"LOSO": build_comparison("loso"), "CDA": build_comparison("cda")}
    mean_row = "Care-PD mean (7 enc.)"

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4))
    for ax, (proto, df) in zip(axes, comps.items()):
        ax.axis("off")
        ax.set_title(f"{proto} — macro-F1 (labels 0–2)", fontsize=14,
                     fontweight="bold", pad=10)

        col_labels = [""] + list(df.columns)
        cell_text = []
        for method, row in df.iterrows():
            txt = [method] + [f"{v:.2f}" for v in row]
            cell_text.append(txt)

        n_data = len(df.columns)
        col_widths = [0.30] + [0.70 / n_data] * n_data   # wide method column
        table = ax.table(cellText=cell_text, colLabels=col_labels,
                         colWidths=col_widths,
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.55)

        mean_vals = df.loc[mean_row]
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#b0b0b0")
            cell.set_linewidth(0.6)
            # header row
            if r == 0:
                cell.set_facecolor("#2f4b7c")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
                continue
            method = cell_text[r - 1][0]
            # first column = method names, left-aligned & bold
            if c == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_fontweight("bold")
            # Care-PD reference rows shaded
            if method.startswith("Care-PD"):
                cell.set_facecolor("#eef1f6")
            else:
                cell.set_facecolor("#ffffff")
                # bold + colour our values that meet/beat the Care-PD mean
                if c >= 1:
                    ds = df.columns[c - 1]
                    if float(cell_text[r - 1][c]) >= float(mean_vals[ds]) - 1e-9:
                        cell.get_text().set_fontweight("bold")
                        cell.get_text().set_color("#1a7a3c")

    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("saved comparison figure   -> %s", out_path)


def _to_markdown(df):
    cols = list(df.columns)
    head = "| " + " | ".join([df.index.name or ""] + cols) + " |"
    sep  = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    body = []
    for idx, r in df.iterrows():
        body.append("| " + " | ".join([str(idx)] + [f"{v:.2f}" for v in r]) + " |")
    return "\n".join([head, sep] + body)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ASSETS_BASE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for protocol in ["CDA", "LOSO"]:
        key = protocol.lower()

        # 1) Care-PD baseline values  -> assets/baseline/
        carepd = build_carepd_values(key)
        carepd.index.name = "encoder"
        base_csv = ASSETS_BASE / f"baseline_F1(0_2)_values_{protocol}.csv"
        carepd.to_csv(base_csv)
        log.info("saved Care-PD %s values  -> %s", protocol, base_csv)

        # 2/3) Comparison table       -> outputs/.../baseline_comparison/
        comp = build_comparison(key)
        comp_csv = OUT_DIR / f"baseline_F1(0_2)_comparison_{protocol}.csv"
        comp.to_csv(comp_csv)
        comp_md = OUT_DIR / f"baseline_F1(0_2)_comparison_{protocol}.md"
        comp_md.write_text(f"### {protocol} — macro-F1(0-2)\n\n{_to_markdown(comp)}\n")
        log.info("saved %s comparison       -> %s", protocol, comp_csv)

        print(f"\n================  {protocol}  —  macro-F1(0-2)  ================")
        print(_to_markdown(comp))

    # combined figure (both tables) for direct pasting into the thesis
    render_tables_figure(OUT_DIR / "baseline_F1(0_2)_comparison_tables.png")

    print("\nNote: Care-PD values read from Fig. 3 (no table published); treat as "
          "+/-0.01. Encoder mean = 7 learned encoders (excl. handcrafted "
          "GaitFeatures+RF baseline). CDA per target = mean over source cohorts.")


if __name__ == "__main__":
    main()
