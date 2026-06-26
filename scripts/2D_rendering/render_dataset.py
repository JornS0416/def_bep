"""
Render Dataset to 2D Silhouette Video
======================================
Renders SMPL mesh sequences from a Care-PD dataset into 2D silhouette MP4s.

Usage:
    Set DATASET_NAME below to any of the available datasets, then run:
        python main_project/scripts/2D_rendering/render_dataset.py
"""

import os
from pathlib import Path

# Run from the repo root (parent of main_project) so the "main_project/..."
# relative paths below resolve regardless of where this script is launched.
os.chdir(Path(__file__).resolve().parents[3])
import pyrender
from src.data.data_loading import load_dataset
from src.rendering.smpl_utils import create_smpl_model
from src.rendering.render_dataset import render_dataset

# ── Config ────────────────────────────────────────────────────────────────────
# Change DATASET_NAME to switch datasets. Must match a file in DATASETS_DIR.
# Available: PD-GaM, BMClab, 3DGait, DNE, E-LC, KUL-DT-T, T-LTC, T-SDU-PD, T-SDU
DATASET_NAME = "E-LC"

DATASETS_DIR  = Path("main_project/assets/datasets/base_sets")
SMPL_MODEL_PATH = "main_project/utils/smpl/models/smpl_new"
OUTPUT_BASE   = Path("main_project/assets/datasets/rendered_videos")
# ─────────────────────────────────────────────────────────────────────────────


def main():
    dataset_path = DATASETS_DIR / f"{DATASET_NAME}.pkl"
    output_dir   = OUTPUT_BASE / DATASET_NAME

    if not dataset_path.exists():
        available = [p.stem for p in DATASETS_DIR.glob("*.pkl")]
        raise FileNotFoundError(
            f"Dataset '{DATASET_NAME}' not found at {dataset_path}.\n"
            f"Available datasets: {available}"
        )

    print(f"Dataset  : {dataset_path}")
    print(f"Output   : {output_dir}")

    data     = load_dataset(str(dataset_path))
    model    = create_smpl_model(SMPL_MODEL_PATH)
    renderer = pyrender.OffscreenRenderer(224, 224)

    render_dataset(
        data=data,
        model=model,
        renderer=renderer,
        output_dir=output_dir,
        max_trials=None,
        show_progress=True,
        show_frame_progress=False,
    )
    renderer.delete()

if __name__ == "__main__":
    main()
