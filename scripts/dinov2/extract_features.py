"""
DINOv2 Feature Extraction
==========================
Extracts per-frame CLS-token embeddings from rendered 2D silhouette videos
using a frozen DINOv2 ViT-B/14 encoder, and saves them for downstream LSTM
training.

Pipeline per clip:
    MP4 → frames → uniform sampling + resize (48 × 224×224)
        → DINOv2 ViT-B/14 (streaming cross-clip batching, frozen)
        → CLS token per frame
        → saved as (48, 768) array per clip

Design choices:
- DINOv2 ViT-B/14 (facebook/dinov2-base): self-supervised ViT trained via
  self-distillation on LVD-142M. Used frozen to match the Care-PD evaluation
  protocol (sec. 4.1). ViT-B/14 (86M params, 768-dim) matches VideoMAE-base
  in scale and embedding dimensionality for fair comparison.

- CLS token (768,) per frame: global semantic embedding. Patch tokens (256×768)
  are discarded — silhouette input has no texture, so patch-level detail adds
  little signal while multiplying cost 256×.

- 48 frames per clip: ~1.6 s at 30 FPS, covering at least one full gait cycle
  at PD walking cadence. The downstream BiLSTM exploits the full sequence.

- (T, 768) saved per clip: the temporal sequence is preserved for the LSTM,
  unlike VideoMAE which collapses to a single (768,) vector via mean pooling.

- Manual numpy preprocessing instead of AutoImageProcessor:
  AutoImageProcessor uses PIL internally — single-threaded, ~3 s per batch of
  128 frames. Replaced by cv2 + numpy at sampling time: resize shortest edge to
  256 (INTER_CUBIC = BICUBIC), center-crop to 224×224, then vectorised numpy
  normalisation in flush_buffer. Pipeline matches AutoImageProcessor exactly
  (BitImageProcessor: shortest_edge=256, crop=224, resample=BICUBIC, ImageNet
  mean/std).

- Streaming cross-clip batching (GLOBAL_BATCH_SIZE=128):
  A rolling buffer accumulates frames from any clips until full, then runs one
  DINOv2 forward pass. Raw frames are freed immediately; only CLS tokens
  (~3 KB each) are kept. Reduces forward passes from ~48 000 (8 frames/pass)
  to ~3 000 (128 frames/pass) — 16× fewer calls.
  Memory: frame buffer ≈ 19 MB (constant), CLS storage ≈ 1.1 GB (final).

- fp16 inference (USE_FP16=True):
  Model weights cast to float16 before inference. MPS executes fp16 matrix ops
  on dedicated GPU matrix engines — measured ~22% faster than float32 on M1.
  Stored embeddings are cast back to float32 so the LSTM receives consistent
  input regardless of inference dtype.

- MPS device with bilinear positional-encoding patch:
  DINOv2 uses bicubic interpolation to resize its positional embeddings from
  the training grid (37×37) to the inference grid (16×16 at 224×224 input).
  MPS does not support 'aten::upsample_bicubic2d'. The patch replaces only
  that one interpolation call with bilinear — model weights are unchanged,
  quality difference on a 37×37 positional matrix is negligible.

- scored_rows sorted by video path:
  The video cache holds only the last loaded video. Sorting ensures all clips
  from the same MP4 are processed consecutively, so each file is read once.

Usage:
    cd main_project
    python scripts/dinov2/extract_features.py

Prerequisite: render_dataset.py must have been run for the target datasets.
"""

import math
import pickle
import logging
import re
import types
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

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
FEATURES_OUT = PROJECT_ROOT / "assets/datasets/fabricated_datasets/dinov2_features_81f.pkl"

MODEL_NAME        = "facebook/dinov2-base"
NUM_FRAMES        = 81    # full clip — all frames passed to LSTM (no uniform sampling).
                          # Using all 81 frames gives the LSTM maximum temporal coverage.
                          # Changed from 48: with attention aggregation the LSTM benefits
                          # from seeing every frame rather than a sampled subset.
CLIP_LEN          = 81    # clip window in video frames (set by ds_combiner.py)
MODEL_INPUT_SIZE  = 224   # DINOv2 ViT-B/14 inference resolution (px)
GLOBAL_BATCH_SIZE = 128   # frames per DINOv2 forward pass; raise to 256 if memory allows

# fp16: ~22% faster inference on MPS; stored embeddings are cast back to float32.
USE_FP16 = True

# ImageNet normalisation constants — identical to AutoImageProcessor for DINOv2.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Device: MPS (Apple Silicon) > CUDA > CPU.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

RENDERED_DATASETS = {"PD-GaM", "BMClab", "3DGait", "T-SDU-PD"}
# ─────────────────────────────────────────────────────────────────────────────


# ── MPS compatibility patch ───────────────────────────────────────────────────

def patch_dinov2_bicubic_for_mps(model: AutoModel) -> None:
    """
    Replace DINOv2's bicubic positional-encoding interpolation with bilinear
    to enable MPS execution on Apple Silicon.

    DINOv2 interpolates its learned position embeddings from the training grid
    (37×37) to the inference grid (16×16) using bicubic resampling. MPS does
    not support 'aten::upsample_bicubic2d'. This patch replaces only that call
    with bilinear — model weights are untouched, quality difference on a
    37×37 positional matrix is negligible for downstream classification.
    No-op on non-MPS devices.
    """
    if str(DEVICE) != "mps":
        return

    embeddings_module = model.embeddings

    def _bilinear_interpolate_pos_encoding(self, embeddings_tensor, height, width):
        num_patches   = embeddings_tensor.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings_tensor.shape[-1]

        h0 = height // self.config.patch_size
        w0 = width  // self.config.patch_size
        h0, w0 = h0 + 0.1, w0 + 0.1  # small offset matching original DINOv2 code

        patch_pos_embed = patch_pos_embed.reshape(
            1,
            int(math.sqrt(num_positions)),
            int(math.sqrt(num_positions)),
            dim,
        ).permute(0, 3, 1, 2)

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            scale_factor=(
                h0 / math.sqrt(num_positions),
                w0 / math.sqrt(num_positions),
            ),
            mode="bilinear",        # was "bicubic"; bilinear is MPS-compatible
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    embeddings_module.interpolate_pos_encoding = types.MethodType(
        _bilinear_interpolate_pos_encoding, embeddings_module
    )
    log.info("MPS patch applied: positional encoding uses bilinear interpolation.")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_batch(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of (224, 224, 3) uint8 RGB frames to a normalised tensor.

    Replaces AutoImageProcessor (PIL-based, ~3 s/batch) with vectorised numpy
    operations (~0.05 s/batch). Steps match AutoImageProcessor exactly:
      1. Stack into (B, H, W, 3) float32, scale to [0, 1]
      2. Subtract ImageNet channel means, divide by channel stds
      3. Transpose to channel-first (B, 3, H, W)

    Frames must already be resized to MODEL_INPUT_SIZE — done in sample_clip_frames.
    Returns float32 tensor; dtype conversion to fp16 happens in flush_buffer.
    """
    batch = np.stack(frames).astype(np.float32) / 255.0
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    batch = batch.transpose(0, 3, 1, 2)                 # (B, 3, H, W)
    return torch.from_numpy(batch)


# ── Video utilities ───────────────────────────────────────────────────────────

def load_video_frames(video_path: Path) -> Optional[np.ndarray]:
    """
    Read all frames of an MP4 into a (T, H, W, 3) uint8 RGB array using cv2.
    Returns None if the file is missing or unreadable.
    """
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
    return np.stack(frames)  # (T, H, W, 3)


def sample_clip_frames(
    all_frames: np.ndarray,
    clip_id: int,
    clip_len: int,
    n_out: int,
) -> List[np.ndarray]:
    """
    Uniformly sample n_out frames from clip window
    [clip_id*clip_len : (clip_id+1)*clip_len] and apply the DINOv2 preprocessing
    pipeline to each frame.

    Spatial pipeline (matches AutoImageProcessor for facebook/dinov2-base exactly):
      1. Resize shortest edge → 256 px using BICUBIC interpolation (resample=3)
      2. Center-crop to 224×224

    For square 224×224 rendered frames this means:
      resize(224→256, BICUBIC) → crop[16:240, 16:240] → 224×224
    i.e. the model sees the central 87.5% of the rendered frame, identical to
    how DINOv2 was pretrained on natural images via BitImageProcessor.

    Done at sampling time (once per frame) to avoid redundant resizing during
    batch assembly. If the window exceeds video length, black frames are returned.
    """
    # Intermediate size for resize-then-crop (matches processor shortest_edge=256)
    RESIZE_SIZE = 256
    CROP_SIZE   = MODEL_INPUT_SIZE  # 224

    T     = len(all_frames)
    start = clip_id * clip_len
    end   = min(start + clip_len, T)

    if start >= T:
        log.warning(
            "clip_id=%d: start=%d >= video length %d; using black frames",
            clip_id, start, T,
        )
        blank = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        return [blank] * n_out

    window  = all_frames[start:end]
    indices = np.linspace(0, len(window) - 1, n_out, dtype=int)

    result = []
    for i in indices:
        # Step 1 — resize shortest edge to 256 with BICUBIC (processor resample=3)
        frame = cv2.resize(
            window[i],
            (RESIZE_SIZE, RESIZE_SIZE),
            interpolation=cv2.INTER_CUBIC,   # BICUBIC = PILImageResampling(3)
        )
        # Step 2 — center crop to 224×224
        top  = (RESIZE_SIZE - CROP_SIZE) // 2   # = 16
        left = (RESIZE_SIZE - CROP_SIZE) // 2   # = 16
        frame = frame[top:top + CROP_SIZE, left:left + CROP_SIZE]
        result.append(frame)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load and filter clip metadata ─────────────────────────────────────────
    log.info("Loading clips from %s", CLIPS_PATH)
    with open(CLIPS_PATH, "rb") as f:
        scored_rows = pickle.load(f)
    log.info("Loaded %d scored clips", len(scored_rows))

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
        log.info("Already extracted: %s — skipping.", already_done)
    log.info("Extracting: %s  |  device: %s", todo_datasets, DEVICE)

    scored_rows = [r for r in scored_rows if r["dataset"] in todo_datasets]

    # Sort so all clips from the same MP4 are adjacent — each video loaded once.
    scored_rows.sort(key=lambda r: (
        r["dataset"], r["subject"], re.sub(r"_down\d+$", "", r["trial"])
    ))
    log.info("Clips to process: %d", len(scored_rows))

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading DINOv2: %s", MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    patch_dinov2_bicubic_for_mps(model)

    if USE_FP16:
        model = model.half()
        log.info("fp16 enabled.")

    log.info(
        "Model ready — %dM parameters  |  device: %s",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        DEVICE,
    )

    # ── Streaming batched inference ───────────────────────────────────────────
    # Rolling buffer: accumulate GLOBAL_BATCH_SIZE frames from any clips, then
    # run one DINOv2 forward pass. Raw frames freed after each flush.
    # CLS tokens stored until reassembly (~1.1 GB total).

    frame_buffer: List[tuple] = []          # [(clip_key, frame_idx, frame_np)]
    cls_storage:  dict        = defaultdict(dict)  # clip_key → {frame_idx: vec}
    clip_order:   List[tuple] = []          # [(clip_key, meta_dict)] in order

    last_video_path: Optional[Path]       = None
    last_frames:     Optional[np.ndarray] = None
    skipped = 0

    def flush_buffer() -> None:
        """Run DINOv2 on buffered frames, store CLS tokens, clear buffer."""
        if not frame_buffer:
            return

        frames = [item[2] for item in frame_buffer]

        # Preprocess + send to device (fp16 cast fused into .to() call)
        dtype        = torch.float16 if USE_FP16 else torch.float32
        pixel_values = preprocess_batch(frames).to(device=DEVICE, dtype=dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # Cast back to float32 for storage — LSTM always receives float32
        cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

        for (clip_key, frame_idx, _), cls_vec in zip(frame_buffer, cls_tokens):
            cls_storage[clip_key][frame_idx] = cls_vec

        frame_buffer.clear()

    log.info(
        "Starting extraction — %d frames/batch, ~%d passes total",
        GLOBAL_BATCH_SIZE,
        math.ceil(len(scored_rows) * NUM_FRAMES / GLOBAL_BATCH_SIZE),
    )

    for row in tqdm(scored_rows, desc="Extracting"):
        dataset = row["dataset"]
        subject = row["subject"]
        trial   = row["trial"]
        clip_id = row["clip_id"]
        label   = int(row["UPDRS_GAIT"])

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

        clip_frames = sample_clip_frames(last_frames, clip_id, CLIP_LEN, NUM_FRAMES)
        clip_key    = (dataset, subject, trial, clip_id)

        clip_order.append((clip_key, {
            "dataset":    dataset,
            "subject":    subject,
            "trial":      trial,
            "clip_id":    clip_id,
            "UPDRS_GAIT": label,
        }))

        for frame_idx, frame in enumerate(clip_frames):
            frame_buffer.append((clip_key, frame_idx, frame))
            if len(frame_buffer) >= GLOBAL_BATCH_SIZE:
                flush_buffer()

    flush_buffer()  # final partial batch
    log.info("Inference done — %d clips, %d skipped.", len(clip_order), skipped)

    # ── Reassemble (T, 768) sequences per clip ────────────────────────────────
    log.info("Reassembling per-clip sequences...")
    features = []

    for clip_key, meta in clip_order:
        frame_dict = cls_storage.get(clip_key, {})
        if len(frame_dict) != NUM_FRAMES:
            log.warning(
                "Clip %s: expected %d frames, got %d — skipping",
                clip_key, NUM_FRAMES, len(frame_dict),
            )
            continue
        seq = np.stack([frame_dict[i] for i in range(NUM_FRAMES)])  # (48, 768)
        features.append({**meta, "feature": seq})

    log.info(
        "%d sequences assembled. Shape per clip: %s",
        len(features),
        features[0]["feature"].shape if features else "N/A",
    )

    # ── Merge with existing output and save ───────────────────────────────────
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)

    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            existing = pickle.load(f)
        new_ds   = {r["dataset"] for r in features}
        existing = [r for r in existing if r["dataset"] not in new_ds]
        merged   = existing + features
        log.info(
            "Merged: %d existing + %d new = %d total",
            len(existing), len(features), len(merged),
        )
    else:
        merged = features

    with open(FEATURES_OUT, "wb") as f:
        pickle.dump(merged, f)
    log.info("Saved to %s", FEATURES_OUT)


if __name__ == "__main__":
    main()
