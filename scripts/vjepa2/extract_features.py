"""
V-JEPA 2 Feature Extraction (ViT-L/16, 256px)
=============================================
Extracts frozen V-JEPA **2** embeddings per clip and saves them for the same
downstream probes used by V-JEPA v1, VideoMAE and DINOv2.

    MP4 → frames → 32-frame clip → V-JEPA 2 encoder → mean-pool tokens → (1024,)

Why this exists / how it differs from scripts/vjepa/extract_features.py (v1)
---------------------------------------------------------------------------
- v1 used Meta's cloned `model/jepa/` repo + a manual 5 GB `.pth.tar` checkpoint
  and hand-rolled preprocessing. V-JEPA 2 ships in HuggingFace `transformers`,
  so it loads with the SAME clean `AutoModel` / `AutoVideoProcessor` pattern as
  the VideoMAE and DINOv2 extractors — no repo clone, no checkpoint surgery.
- ViT-L V-JEPA 2 has hidden_size = 1024, identical to v1's ViT-L, so the output
  schema and feature dimension are a drop-in: downstream eval code is unchanged.
- v1 is kept intact; this writes a SEPARATE file (vjepa2_features.pkl) so the two
  can be compared directly.

Design choices (kept consistent with v1 for a fair v1-vs-v2 comparison)
----------------------------------------------------------------------
- Same clip metadata (scored_clip_rows.pkl), same rendered MP4s, same 81-frame
  clip windows. V-JEPA 2 supports a variable number of frames at inference. The
  default here is 32 frames: MPS-safe on an Apple M1 (64 frames exceed the MPS
  4 GB per-tensor attention limit) and already 2× v1's temporal context. On a
  CUDA GPU set NUM_FRAMES=64 — V-JEPA 2's native fpc64 length — for maximum
  performance. v1 stays at 16; the comparison is model-vs-model.
- Spatial preprocessing (resize→256, centre-crop, normalisation) is delegated to
  the official VJEPA2VideoProcessor — this is the recommended, exact pipeline.
- Pooling: mean over all encoder tokens → (1024,) for the linear/ridge probes
  (the metric reported in the thesis). feature_max and feature_temporal (per
  tubelet-step) are saved too, at no real extra cost, for optional ablations.
- skip_predictor=True: we only need the encoder output, so the predictor head is
  skipped for speed.

Environment
-----------
V-JEPA 2 needs transformers >= 4.53 (Python >= 3.10). Run this with the dedicated
`vjepa2` conda env, NOT carepd38 (which is Python 3.8). The downstream eval still
runs in carepd38 — it only reads the pickle this script writes.

    ~/anaconda3/envs/vjepa2/bin/python scripts/vjepa2/extract_features.py

Prerequisite: render_dataset.py must have produced the rendered MP4s, and
ds_combiner.py must have produced scored_clip_rows.pkl.
"""

import gc
import re
import pickle
import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS_PATH   = PROJECT_ROOT / "assets/datasets/fabricated_datasets/scored_clip_rows.pkl"
VIDEO_ROOT   = PROJECT_ROOT / "assets/datasets/rendered_videos"
FEATURES_OUT = PROJECT_ROOT / "assets/datasets/fabricated_datasets/vjepa2_features.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L, 1024-dim (drop-in vs v1)
NUM_FRAMES      = 16   # 16 frames × stride 4 → span 61 ≤ the 81-frame clip, so it
                       # fits with a REAL stride. (64 would need a 253-frame span and
                       # silently fall back to dense linspace.) Same temporal window
                       # as VideoMAE / V-JEPA v1 → fair, matched encoder comparison.
TEMPORAL_STRIDE = 4    # 16×4 spans 61 of 81 frames, centred — matches VideoMAE/v1
TUBELET_SIZE    = 2    # V-JEPA 2 temporal patch size (config default)
CLIP_LEN        = 81   # frames per clip window (set in ds_combiner.py)
EMBED_DIM       = 1024 # ViT-L hidden size
BATCH_SIZE      = 8    # clips per forward pass; safe on M1. Raise on a CUDA GPU.
USE_FP16        = True  # float32 fits at 32 frames on MPS. On a CUDA GPU set True
                         # for ~2× speed; for 64 frames on MPS it is REQUIRED.
SAVE_TEMPORAL   = False  # (NUM_FRAMES/2, 1024) per-tubelet features bloat the pickle
                         # ~17× at 64 frames and are only used by the dropped 5-fold
                         # attentive probe. The 4 Care-PD protocols use the mean only.

RENDERED_DATASETS = {"PD-GaM", "BMClab", "3DGait", "T-SDU-PD"}

# CUDA first: on a cloud GPU this is what you want. 64 frames (8192 tokens) exceed
# Apple-MPS's 4 GB single-buffer attention limit, so MPS only works at <=32 frames.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


def load_video_frames(video_path: Path) -> Optional[np.ndarray]:
    """Read all frames of an MP4 into a (T, H, W, 3) uint8 RGB array (cv2)."""
    if not video_path.exists():
        log.warning("Video not found: %s", video_path)
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Could not open: %s", video_path)
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
    return np.stack(frames)


def sample_clip_frames(all_frames: np.ndarray, clip_id: int) -> Optional[np.ndarray]:
    """
    Sample NUM_FRAMES frames from clip window
    [clip_id*CLIP_LEN : (clip_id+1)*CLIP_LEN] using a fixed temporal stride,
    centred in the window — identical temporal selection to v1. Spatial
    preprocessing is left to the processor, so raw RGB frames are returned.

    Returns (NUM_FRAMES, H, W, 3) uint8, or None if the window starts past the
    end of the video.
    """
    T = len(all_frames)
    start = clip_id * CLIP_LEN
    if start >= T:
        log.warning("clip_id=%d: start=%d >= video length %d; skipping", clip_id, start, T)
        return None
    window  = all_frames[start:min(start + CLIP_LEN, T)]
    win_len = len(window)
    span    = (NUM_FRAMES - 1) * TEMPORAL_STRIDE + 1
    if win_len >= span:
        offset  = (win_len - span) // 2
        indices = offset + np.arange(NUM_FRAMES) * TEMPORAL_STRIDE
    else:
        indices = np.linspace(0, win_len - 1, NUM_FRAMES, dtype=int)
    return window[indices]   # (NUM_FRAMES, H, W, 3)


def main() -> None:
    # ── Load clip metadata ────────────────────────────────────────────────────
    log.info("Loading clip metadata from %s", CLIPS_PATH)
    with open(CLIPS_PATH, "rb") as f:
        scored_rows = pickle.load(f)
    log.info("Total scored clips: %d", len(scored_rows))

    # Incremental: skip datasets already present in the output file.
    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            already_done = {r["dataset"] for r in pickle.load(f)}
    else:
        already_done = set()
    todo = RENDERED_DATASETS - already_done
    if not todo:
        log.info("All datasets already extracted. Nothing to do.")
        return
    if already_done:
        log.info("Already extracted: %s — skipping.", already_done)
    scored_rows = [r for r in scored_rows if r["dataset"] in todo]
    # Sort so all clips of one MP4 are consecutive (each video read once).
    scored_rows.sort(key=lambda r: (r["dataset"], r["subject"],
                                    re.sub(r"_down\d+$", "", r["trial"])))
    log.info("Extracting: %s  |  clips: %d  |  device: %s", todo, len(scored_rows), DEVICE)

    # ── Load V-JEPA 2 (encoder) + processor ───────────────────────────────────
    log.info("Loading %s ...", MODEL_NAME)
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    if USE_FP16:
        model = model.half()
    model.eval()
    log.info("Model ready — %dM params | hidden=%d",
             sum(p.numel() for p in model.parameters()) // 1_000_000,
             getattr(model.config, "hidden_size", EMBED_DIM))

    N_TEMPORAL = NUM_FRAMES // TUBELET_SIZE   # 8 tubelet steps

    # ── Streaming batched inference ───────────────────────────────────────────
    last_video_path: Optional[Path]       = None
    last_frames:     Optional[np.ndarray] = None
    batch_pv:   List[torch.Tensor] = []
    batch_meta: List[dict]         = []
    features = []
    skipped  = 0

    @torch.no_grad()
    def flush() -> None:
        if not batch_pv:
            return
        pv = torch.cat(batch_pv, dim=0).to(DEVICE)        # (B, T, C, H, W)
        if USE_FP16:
            pv = pv.half()
        tokens = model(pixel_values_videos=pv, skip_predictor=True).last_hidden_state
        tok = tokens.float().cpu().numpy()                # (B, N, 1024)
        B, N, D = tok.shape
        fmean = tok.mean(axis=1)                          # (B, 1024)
        fmax  = tok.max(axis=1)                           # (B, 1024)
        ftemp = None
        if SAVE_TEMPORAL and N % N_TEMPORAL == 0:          # (B, N_TEMPORAL, 1024)
            ftemp = tok.reshape(B, N_TEMPORAL, N // N_TEMPORAL, D).mean(axis=2)
        for i, meta in enumerate(batch_meta):
            rec = {**meta, "feature": fmean[i], "feature_max": fmax[i]}
            if ftemp is not None:
                rec["feature_temporal"] = ftemp[i]
            features.append(rec)
        batch_pv.clear(); batch_meta.clear()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    for row in tqdm(scored_rows, desc="V-JEPA 2 extraction"):
        dataset, subject, trial, clip_id = (row["dataset"], row["subject"],
                                            row["trial"], row["clip_id"])
        label = int(row["UPDRS_GAIT"])
        if label not in {0, 1, 2, 3}:
            log.warning("Unexpected UPDRS %d — %s/%s/%s skipped", label, dataset, subject, trial)
            skipped += 1
            continue

        video_trial = re.sub(r"_down\d+$", "", trial)
        video_path  = VIDEO_ROOT / dataset / subject / f"{video_trial}.mp4"
        if video_path != last_video_path:
            last_frames     = load_video_frames(video_path)
            last_video_path = video_path
        if last_frames is None:
            skipped += 1
            continue

        clip_frames = sample_clip_frames(last_frames, clip_id)
        if clip_frames is None:
            skipped += 1
            continue

        # Processor: resize→256, centre-crop, normalise → (1, T, C, H, W).
        pv = processor(list(clip_frames), return_tensors="pt")["pixel_values_videos"]
        batch_pv.append(pv)
        batch_meta.append({"dataset": dataset, "subject": subject, "trial": trial,
                           "clip_id": clip_id, "UPDRS_GAIT": label})
        if len(batch_pv) >= BATCH_SIZE:
            flush()
    flush()

    log.info("Extracted %d clips | skipped %d", len(features), skipped)
    if features:
        log.info("feature shape: %s | temporal saved: %s",
                 features[0]["feature"].shape, "feature_temporal" in features[0])

    # ── Merge with existing output and save ───────────────────────────────────
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            existing = pickle.load(f)
        new_ds   = {r["dataset"] for r in features}
        existing = [r for r in existing if r["dataset"] not in new_ds]
        merged   = existing + features
        log.info("Merged: %d existing + %d new = %d", len(existing), len(features), len(merged))
    else:
        merged = features
    with open(FEATURES_OUT, "wb") as f:
        pickle.dump(merged, f)
    log.info("Saved to %s", FEATURES_OUT)


if __name__ == "__main__":
    main()
