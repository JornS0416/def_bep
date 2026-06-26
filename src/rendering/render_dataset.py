"""
Dataset Renderer
================
Iterates over all trials in a Care-PD dataset and renders each to an MP4.

Design choices:
- subject_id and trial_id are cast to str: Care-PD datasets use inconsistent
  key types (strings in PD-GaM, integers in 3DGait/BMClab). str() normalises
  this so Path concatenation works uniformly across all subsets.
- Skip-if-exists check: allows resuming interrupted renders without reprocessing
  completed trials.
- Output structure: assets/datasets/rendered_videos/<dataset>/<subject>/<trial>.mp4
  mirrors the Care-PD dataset structure and is expected by extract_features.py.
"""

from pathlib import Path
from tqdm import tqdm
from src.data.data_loading import iter_trials
from src.rendering.render_trial import render_trial


def render_dataset(data, model, renderer, output_dir,
                   max_trials=None, show_progress=True, show_frame_progress=False):
    """
    Render all trials in a dataset to MP4 files.

    Args:
        data:               loaded dataset dict.
        model:              smplx SMPL model instance (reused across all trials).
        renderer:           pyrender.OffscreenRenderer instance.
        output_dir:         root output directory; structure: <output_dir>/<subject>/<trial>.mp4
        max_trials:         optional cap for debugging (None = render all).
        show_progress:      tqdm bar over trials.
        show_frame_progress: tqdm bar over frames within each trial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_count = 0

    for subject_id, trial_id, trial in tqdm(iter_trials(data), desc="Trials", disable=not show_progress):
        trial_count += 1
        if max_trials is not None and trial_count > max_trials:
            break

        # str() cast: normalises int keys (3DGait, BMClab) and str keys (PD-GaM)
        subject_dir = output_dir / str(subject_id)
        subject_dir.mkdir(parents=True, exist_ok=True)
        output_path = subject_dir / f"{str(trial_id)}.mp4"

        if output_path.exists():
            print(f"Skipping existing: {output_path}")
            continue

        # Downsample to 30 FPS if the trial was captured at a higher rate.
        # Only applied when fps is an exact integer multiple of 30 (e.g. 150 → ÷5).
        # This ensures clip_id indices from scored_clip_rows (built at 30 FPS)
        # align with the rendered video frame count.
        fps = trial.get("fps", 30)
        if fps > 30 and fps % 30 == 0:
            stride = fps // 30
            trial  = {**trial,
                      "pose":  trial["pose"][::stride],
                      "trans": trial["trans"][::stride]}
            fps = 30

        render_trial(
            trial=trial,
            model=model,
            renderer=renderer,
            output_path=str(output_path),
            fps=fps,
            show_progress=show_frame_progress,
        )
