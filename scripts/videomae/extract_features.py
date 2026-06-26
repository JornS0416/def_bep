"""
VideoMAE Feature Extraction
============================
Extracts clip-level feature vectors from rendered 2D SMPL-mesh videos using
a frozen VideoMAE encoder, and saves them for downstream classification.

Pipeline per clip:
    MP4 → frames → uniform sampling (16 frames) → VideoMAE
        → multi-layer mean pool (layers 9-12 concatenated) → (3072,)

Design choices:
- VideoMAE (MCG-NJU/videomae-base): pretrained on Kinetics-400 via masked
  autoencoding. Captures spatio-temporal motion patterns without task-specific
  supervision. Used frozen to evaluate zero-shot representation quality,
  matching the Care-PD paper protocol (sec. 4.1).

- 16 frames per clip: VideoMAE's required input length. Frames are sampled
  with stride 4 (VideoMAE Kinetics-400 pretraining spec: 16 × stride 4 =
  60-frame span ≈ 2.0 s at 30 FPS), centered in the 81-frame clip window.

- Multi-layer feature extraction (layers 9, 10, 11, 12):
  VideoMAE-base has 12 transformer layers. The last layer is optimised for the
  pixel-reconstruction pretext task and may not be optimal for classification.
  Intermediate layers (9-12) retain richer semantic structure alongside
  motion patterns. Each selected layer is mean-pooled over 1568 tokens → (768,);
  the four vectors are concatenated → (3072,).
  Computational cost: zero extra — output_hidden_states=True returns all layers
  from the same forward pass. Feature file grows from ~21 MB to ~84 MB.
  Change LAYERS_TO_USE = [12] to revert to single-layer (768,).

- Clip length = 81 frames: set in ds_combiner.py (~2.7s at 30 FPS).

- RENDERED_DATASETS: PD-GaM, BMClab, 3DGait — all three have rendered MP4s.

- Single-video RAM strategy: one MP4 held in memory at a time. Batch size = 8.

- Labels cast to int and validated against {0,1,2,3} at extraction time.

- Output: videomae_features_multilayer.pkl  (keeps original videomae_features.pkl
  intact for comparison).

"""

import pickle
import logging
import re
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import cv2
import torch
from transformers import VideoMAEModel, AutoImageProcessor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS_PATH   = PROJECT_ROOT / "assets/datasets/fabricated_datasets/scored_clip_rows.pkl"
VIDEO_ROOT   = PROJECT_ROOT / "assets/datasets/rendered_videos"
FEATURES_OUT = PROJECT_ROOT / "assets/datasets/fabricated_datasets/videomae_features_multilayer.pkl"

MODEL_NAME   = "MCG-NJU/videomae-base"   # pretrained on Kinetics-400, no task head
NUM_FRAMES     = 16                      # VideoMAE requires exactly 16 frames
TEMPORAL_STRIDE = 4                     # Kinetics-400 pretraining stride (VideoMAE paper, Sec. 3)
CLIP_LEN       = 81                     # frames per clip (set in ds_combiner.py)
BATCH_SIZE   = 8                         # clips per forward pass; lower if OOM
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-layer feature extraction.
# VideoMAE-base has 12 transformer layers + 1 embedding layer = 13 hidden states
# (indexed 0..12 in output.hidden_states; 0 = patch embedding, 1..12 = transformer layers).
# Using only the last layer (index 12) is the default. Concatenating the last
# few layers captures both high-level semantics (last layer) and intermediate
# motion patterns (earlier layers), which often improves transfer to new tasks.
#
# LAYERS_TO_USE = [12]          → last layer only → (768,)    [original]
# LAYERS_TO_USE = [9, 10, 11, 12] → last 4 layers → (3072,)   [multi-layer]
#
# Computational cost: zero extra — output_hidden_states=True returns all layers
# from the same single forward pass. Feature file grows from ~21 MB to ~84 MB.
LAYERS_TO_USE = [9, 10, 11, 12]   # last 4 layers; change to [12] to revert

# Datasets for which rendered MP4s exist.
# Extend this set as more datasets are rendered.
RENDERED_DATASETS = {"PD-GaM", "BMClab", "3DGait", "T-SDU-PD"}
# ─────────────────────────────────────────────────────────────────────────────


def load_video_frames(video_path: Path) -> Optional[np.ndarray]:
    """
    Read all frames of an MP4 into a (T, H, W, 3) uint8 array using cv2.
    cv2 is ~3-5× faster than imageio for sequential frame reads.
    Returns None if the file is missing or unreadable.
    """
    if not video_path.exists():
        log.warning("Video not found: %s", video_path)
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Could not open %s", video_path)
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        log.warning("No frames read from %s", video_path)
        return None
    return np.stack(frames)   # (T, H, W, 3)


def sample_clip_frames(
    all_frames: np.ndarray,
    clip_id: int,
    clip_len: int,
    n_out: int,
    stride: int = TEMPORAL_STRIDE,
) -> List[np.ndarray]:
    """
    Sample n_out frames from clip window [clip_id*clip_len : (clip_id+1)*clip_len]
    using a fixed temporal stride, centered within the window.

    Stride = 4 matches VideoMAE Kinetics-400 pretraining exactly:
    16 frames × stride 4 = 64-frame span ≈ 2.0 s at 30 FPS.
    The 64-frame window is centered in the 81-frame clip (offset = (81-64)//2 = 8),
    so both ends of the gait cycle are equally represented.

    If the clip window is shorter than n_out * stride (e.g. a truncated final
    clip), falls back to uniform linspace to always return exactly n_out frames.
    If the window start exceeds video length, black frames are returned.
    """
    T     = len(all_frames)
    start = clip_id * clip_len
    end   = min(start + clip_len, T)

    if start >= T:
        log.warning("clip_id=%d start=%d >= video length %d; returning black frames", clip_id, start, T)
        h, w = all_frames.shape[1:3]
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        return [blank] * n_out

    window     = all_frames[start:end]
    win_len    = len(window)
    span       = (n_out - 1) * stride + 1   # e.g. 15*4+1 = 61 frames needed

    if win_len >= span:
        # Center the stride-4 window: shift right by half the unused tail
        offset  = (win_len - span) // 2     # e.g. (81-61)//2 = 10 → indices 10..70
        indices = offset + np.arange(n_out) * stride
    else:
        # Fallback for clips shorter than the required span
        indices = np.linspace(0, win_len - 1, n_out, dtype=int)

    return [window[i] for i in indices]


def main() -> None:
    # ── Load clip metadata ────────────────────────────────────────────────────
    log.info("Loading clips from %s", CLIPS_PATH)
    with open(CLIPS_PATH, "rb") as f:
        scored_rows = pickle.load(f)
    log.info("Loaded %d scored clips", len(scored_rows))

    # Determine which datasets still need extraction.
    # Datasets already present in the existing features file are skipped entirely
    # to avoid redundant computation.
    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            already_done = {r["dataset"] for r in pickle.load(f)}
    else:
        already_done = set()

    todo_datasets = RENDERED_DATASETS - already_done
    if not todo_datasets:
        log.info("All rendered datasets already extracted. Nothing to do.")
        return

    if already_done:
        log.info("Already extracted: %s  — skipping.", already_done)
    log.info("Extracting: %s", todo_datasets)

    scored_rows = [r for r in scored_rows if r["dataset"] in todo_datasets]
    log.info("Clips to process: %d", len(scored_rows))

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading VideoMAE: %s  (device=%s)", MODEL_NAME, DEVICE)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = VideoMAEModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # ── Feature extraction ────────────────────────────────────────────────────
    # Load one video at a time (low peak RAM); re-read only when trial changes.
    last_video_path: Optional[Path]       = None
    last_frames:     Optional[np.ndarray] = None

    # Batch buffers: accumulate BATCH_SIZE clips before each forward pass
    batch_pixel_values: List[torch.Tensor] = []
    batch_meta:         List[dict]         = []

    features = []
    skipped  = 0

    def flush_batch() -> None:
        if not batch_pixel_values:
            return
        pv = torch.cat(batch_pixel_values, dim=0).to(DEVICE)  # (B, 16, 3, 224, 224)
        with torch.no_grad():
            out = model(pixel_values=pv, output_hidden_states=True)

        # Multi-layer feature extraction:
        # out.hidden_states is a tuple of 13 tensors, each (B, 1568, 768).
        # For each selected layer: mean-pool over 1568 tokens → (B, 768).
        # Concatenate across selected layers → (B, 768 * len(LAYERS_TO_USE)).
        #
        # Example with LAYERS_TO_USE = [9, 10, 11, 12]:
        #   layer 9  mean → (B, 768)
        #   layer 10 mean → (B, 768)
        #   layer 11 mean → (B, 768)
        #   layer 12 mean → (B, 768)
        #   concat       → (B, 3072)
        # Mean-pool: average over 1568 tokens per layer → concatenate
        layer_vecs_mean = [
            out.hidden_states[i].mean(dim=1)   # (B, 768) per layer
            for i in LAYERS_TO_USE
        ]
        vecs_mean = torch.cat(layer_vecs_mean, dim=-1).cpu().numpy()  # (B, 3072)

        # Max-pool: element-wise max over 1568 tokens per layer → concatenate
        # Computed from the same forward pass at no extra cost.
        layer_vecs_max = [
            out.hidden_states[i].max(dim=1).values   # (B, 768) per layer
            for i in LAYERS_TO_USE
        ]
        vecs_max = torch.cat(layer_vecs_max, dim=-1).cpu().numpy()  # (B, 3072)

        for meta, vmean, vmax in zip(batch_meta, vecs_mean, vecs_max):
            features.append({**meta, "feature": vmean, "feature_max": vmax})
        batch_pixel_values.clear()
        batch_meta.clear()

    for row in tqdm(scored_rows, desc="VideoMAE feature extraction"):
        dataset  = row["dataset"]
        subject  = row["subject"]
        trial    = row["trial"]
        clip_id  = row["clip_id"]
        label    = int(row["UPDRS_GAIT"])  # cast: UPDRS-GAIT is strictly {0,1,2,3}

        # Some datasets (e.g. BMClab) store trial names with a _downX suffix
        # in scored_clip_rows (added by ds_combiner during downsampling).
        # The rendered video uses the base trial name without this suffix.
        video_trial = re.sub(r"_down\d+$", "", trial)
        video_path  = VIDEO_ROOT / dataset / subject / f"{video_trial}.mp4"

        if video_path != last_video_path:
            last_frames     = load_video_frames(video_path)
            last_video_path = video_path

        if last_frames is None:
            skipped += 1
            continue

        clip_frames = sample_clip_frames(last_frames, clip_id, CLIP_LEN, NUM_FRAMES)
        inputs      = processor(clip_frames, return_tensors="pt")   # (1, 16, 3, 224, 224)

        batch_pixel_values.append(inputs["pixel_values"])
        batch_meta.append({
            "dataset":    dataset,
            "subject":    subject,
            "trial":      trial,
            "clip_id":    clip_id,
            "UPDRS_GAIT": label,
        })

        if len(batch_pixel_values) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    log.info("Extracted features for %d clips", len(features))
    log.info("Skipped (missing video): %d", skipped)
    if skipped > 0:
        log.warning("%d clips skipped. Check that render_dataset was run for all datasets.", skipped)
    if features:
        log.info("Feature vector shape: %s", features[0]["feature"].shape)

    # ── Merge with existing features and save ─────────────────────────────────
    # Load existing features (if any) and merge by replacing entries for
    # datasets that were just processed, then appending new ones.
    # This allows running the script incrementally as new datasets are rendered
    # without re-extracting already-processed datasets.
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)

    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            existing = pickle.load(f)
        # Drop entries belonging to datasets we just (re-)processed
        new_datasets = {r["dataset"] for r in features}
        existing = [r for r in existing if r["dataset"] not in new_datasets]
        merged = existing + features
        log.info(
            "Merged with existing file: kept %d existing + %d new = %d total",
            len(existing), len(features), len(merged),
        )
    else:
        merged = features

    with open(FEATURES_OUT, "wb") as f:
        pickle.dump(merged, f)
    log.info("Saved %d feature vectors to %s", len(merged), FEATURES_OUT)


if __name__ == "__main__":
    main()
