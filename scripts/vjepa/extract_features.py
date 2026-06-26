"""
V-JEPA Feature Extraction
==========================
Extracts clip-level feature vectors from rendered 2D SMPL-mesh videos using
a frozen V-JEPA ViT-L/16 encoder, and saves them for downstream classification.

Pipeline per clip:
    MP4 → frames → uniform sampling (16 frames) → V-JEPA encoder → mean pool → (1024,)

─────────────────────────────────────────────────────────────────────────────
SETUP (run once before this script):
─────────────────────────────────────────────────────────────────────────────

1.  Clone the Meta jepa repo into main_project/model/:

        cd main_project
        git clone https://github.com/facebookresearch/jepa model/jepa

2.  Download the pretrained ViT-L/16 checkpoint (~1.2 GB):

        mkdir -p assets/checkpoints/vjepa
        curl -L https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar \
             -o assets/checkpoints/vjepa/vitl16.pth.tar

    (Or use wget if curl is unavailable.)

After setup your directory should contain:
    main_project/
        model/jepa/src/models/vision_transformer.py   ← Meta ViT code
        assets/checkpoints/vjepa/vitl16.pth.tar       ← pretrained weights

─────────────────────────────────────────────────────────────────────────────
Design choices:
- V-JEPA ViT-L/16 (facebookresearch/jepa): self-supervised video model trained
  on VideoMix2M via feature-space prediction (JEPA objective). Chosen because it
  captures spatio-temporal representations without pixel reconstruction,
  theoretically better suited to abstract motion patterns like parkinsonian gait
  than VideoMAE's pixel-level masking. Used frozen (zero-shot evaluation),
  matching the Care-PD paper protocol and the VideoMAE/DINOv2 pipelines.

- checkpoint key = 'target_encoder': V-JEPA saves both a context encoder and a
  target encoder (EMA of context). The target encoder is more stable and is
  the recommended feature extractor (used in all official evals).

- Input format (B, 3, T, H, W): V-JEPA expects channel-first, time as 3rd dim.
  This differs from VideoMAE which expects (B, T, 3, H, W). The transposition
  happens in preprocess_clip() before the forward pass.

- 16 frames per clip: V-JEPA ViT-L/16 was pretrained with tubelet_size=2 and
  num_frames=16, giving 8 temporal × 14×14 spatial = 1568 tokens. Feeding
  exactly 16 frames avoids temporal positional-embedding interpolation.
  Frames are sampled with stride 4 (vitl16.yaml: sampling_rate=4), centered
  in the 81-frame clip: 16 × stride 4 = 61-frame span ≈ 2.0 s at 30 FPS.

- Mean pooling 1568 tokens → (1024,): collapses spatiotemporal tokens into
  a single clip embedding. Same strategy as VideoMAE (768-dim). This is the
  standard approach for linear probing on frozen V-JEPA features.
  Alternative (max pool) is computed in parallel at no extra cost and saved
  as a separate field for ablation.

- ImageNet normalisation (mean/std): V-JEPA ViT-L was trained with standard
  ImageNet pixel normalisation, identical to DINOv2. Preprocessing is done
  via fast numpy ops (same pattern as dinov2/extract_features.py).

- Input tensor shape: (B, 3, 16, 224, 224) — video clips with 16 RGB frames
  at 224×224 pixels.

- Batch size = 1: V-JEPA ViT-L (305M params) is 3.5× larger than VideoMAE-base
  (86M). On M1 (8 GB unified RAM) with other processes running, batch_size=4
  caused OOM crashes. Batch_size=1 keeps peak MPS memory under ~2 GB with fp16.

- MPS device: Apple Silicon GPU. V-JEPA's interpolate_pos_encoding uses
  trilinear interpolation — MPS supports this (unlike DINOv2's bicubic issue).
  No patch needed.

- RENDERED_DATASETS: now includes PD-GaM, BMClab, and 3DGait (all three have
  rendered MP4s). Extend further as more datasets are rendered.

- Incremental extraction: datasets already present in the output file are
  skipped to allow re-running without redundant computation.

Usage:
    cd main_project
    python scripts/vjepa/extract_features.py

Memory optimisations (M1-specific):
- fp16 inference (USE_FP16=True): halves weight memory from ~1.2 GB to ~600 MB.
  Stored embeddings cast back to float32. Identical strategy to DINOv2 pipeline.
- Checkpoint freed immediately after loading (del ckpt; gc.collect()) to avoid
  holding 1.2 GB fp32 weights alongside the fp16 model simultaneously.
- torch.mps.empty_cache() + gc.collect() after every batch.
- Run from Terminal (not PyCharm) to save ~2 GB RAM from IDE overhead.

Prerequisite: setup steps above + render_dataset.py run for all three datasets.
"""

import sys
import gc
import pickle
import logging
import re
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import cv2
import torch

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
JEPA_REPO      = PROJECT_ROOT / "model" / "jepa"          # git clone target
CHECKPOINT     = PROJECT_ROOT / "assets/checkpoints/vjepa/vitl16.pth.tar"
CLIPS_PATH     = PROJECT_ROOT / "assets/datasets/fabricated_datasets/scored_clip_rows.pkl"
VIDEO_ROOT     = PROJECT_ROOT / "assets/datasets/rendered_videos"
FEATURES_OUT   = PROJECT_ROOT / "assets/datasets/fabricated_datasets/vjepa_features.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
NUM_FRAMES      = 16   # V-JEPA ViT-L/16 pretrained with 16 frames; avoids pos-embed interpolation
TEMPORAL_STRIDE = 4    # sampling_rate in vitl16.yaml pretrain config (same as VideoMAE K400)
TUBELET_SIZE    = 2    # temporal patch size used during V-JEPA pretraining
CLIP_LEN        = 81   # frames per clip (set in ds_combiner.py)
IMG_SIZE        = 224  # spatial resolution expected by ViT-L/16
RESIZE_SIZE     = 256  # intermediate resize target (official eval: int(224*256/224)=256)
BATCH_SIZE   = 1      # 1 clip per forward pass — ViT-L is 305M params; keeps peak
                      # MPS memory under ~2 GB with fp16. Raise to 2 only if stable.
EMBED_DIM    = 1024   # ViT-L hidden dimension

# fp16: halves weight memory (~1.2 GB → ~600 MB) and activation memory.
# Stored embeddings are cast back to float32 so downstream code is unaffected.
# Identical strategy to dinov2/extract_features.py which measured ~22% speedup.
USE_FP16 = True

# ImageNet normalisation — same constants used by V-JEPA during pretraining.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Datasets with rendered MP4s available.
RENDERED_DATASETS = {"PD-GaM", "BMClab", "3DGait", "T-SDU-PD"}

# Device: MPS (Apple Silicon) > CUDA > CPU.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1: Add Meta jepa repo to Python path ─────────────────────────────────
# The jepa repo is not a proper package — it uses relative imports internally.
# We add the repo root to sys.path so that `from src.models...` resolves.

def setup_jepa_path() -> None:
    """Add the cloned jepa repo root to sys.path if not already present."""
    jepa_str = str(JEPA_REPO)
    if jepa_str not in sys.path:
        sys.path.insert(0, jepa_str)
    if not JEPA_REPO.exists():
        raise FileNotFoundError(
            f"V-JEPA repo not found at {JEPA_REPO}.\n"
            "Run: git clone https://github.com/facebookresearch/jepa model/jepa"
        )
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}.\n"
            "Run: curl -L https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar "
            "-o assets/checkpoints/vjepa/vitl16.pth.tar"
        )


# ── Step 2: Load V-JEPA ViT-L/16 encoder ─────────────────────────────────────
# V-JEPA saves two encoders in each checkpoint:
#   'encoder'        → context encoder (processes unmasked patches during training)
#   'target_encoder' → EMA of context encoder (used in all official evaluations)
# We load 'target_encoder' — it is more stable and is the recommended extractor.

def load_vjepa_encoder() -> torch.nn.Module:
    """
    Build a V-JEPA ViT-L/16 encoder and load pretrained weights.

    Architecture:
        VisionTransformer(embed_dim=1024, depth=24, num_heads=16)
        Input:  (B, 3, 16, 224, 224)   — batch, channels, frames, H, W
        Output: (B, 1568, 1024)        — batch, spatiotemporal tokens, dim
                (8 temporal × 14×14 spatial = 1568 tokens)

    Returns the encoder in eval mode on DEVICE.
    """
    from src.models.vision_transformer import vit_large  # from jepa repo

    log.info("Building V-JEPA ViT-L/16 (embed_dim=1024, depth=24)...")
    model = vit_large(
        patch_size=16,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        img_size=IMG_SIZE,
        uniform_power=False,   # setting used during ViT-L/16 pretraining
    )

    log.info("Loading checkpoint from %s", CHECKPOINT)
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)

    # The checkpoint stores the target encoder under 'target_encoder'.
    # It may be wrapped in an extra dict layer depending on the save format.
    if "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
    elif "encoder" in ckpt:
        log.warning("'target_encoder' not found; falling back to 'encoder'.")
        state_dict = ckpt["encoder"]
    else:
        # Assume the checkpoint IS the state dict
        state_dict = ckpt

    # Strip 'module.' prefix that appears when saved with DataParallel/FSDP.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Free the full checkpoint dict immediately — it's 1.2 GB in fp32.
    # Without this, ckpt stays in memory alongside the model weights.
    del ckpt
    gc.collect()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        log.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])

    # Free state_dict now that weights are loaded into the model.
    del state_dict
    gc.collect()

    model = model.to(DEVICE)

    # Cast to fp16 to halve weight + activation memory on MPS.
    # Embeddings are cast back to float32 in encode_batch() before saving.
    if USE_FP16:
        model = model.half()
        log.info("fp16 enabled — weight memory ~600 MB instead of ~1.2 GB.")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) // 1_000_000
    log.info("V-JEPA ViT-L/16 ready — %dM parameters | device: %s", n_params, DEVICE)
    return model


# ── Step 3: Video loading ─────────────────────────────────────────────────────

def load_video_frames(video_path: Path) -> Optional[np.ndarray]:
    """
    Read all frames of an MP4 into a (T, H, W, 3) uint8 RGB array using cv2.
    Returns None if the file is missing or unreadable.

    cv2 is ~3-5× faster than imageio for sequential frame reads (same as
    VideoMAE and DINOv2 pipelines).
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


# ── Step 4: Clip sampling ─────────────────────────────────────────────────────

def sample_clip_frames(
    all_frames: np.ndarray,
    clip_id: int,
    clip_len: int,
    n_out: int,
    stride: int = TEMPORAL_STRIDE,
) -> Optional[np.ndarray]:
    """
    Sample n_out frames from clip window [clip_id*clip_len : (clip_id+1)*clip_len]
    using a fixed temporal stride and apply the official V-JEPA spatial pipeline.

    Temporal sampling:
        stride=4 matches vitl16.yaml pretrain config ('sampling_rate: 4').
        16 frames × stride 4 = 61-frame span ≈ 2.0 s at 30 FPS, centered in the
        81-frame clip window (offset = (81-61)//2 = 10).
        Falls back to uniform linspace for clips shorter than the required span.

    Spatial pipeline (matches VideoTransform.eval_transform in
    evals/video_classification_frozen/utils.py exactly):
        1. Resize shortest edge → 256 px, bilinear  (int(224*256/224) = 256)
        2. Center-crop to 224×224

    Returns an (n_out, 224, 224, 3) uint8 array, or None if the clip window
    starts beyond the end of the video.
    """
    T     = len(all_frames)
    start = clip_id * clip_len
    end   = min(start + clip_len, T)

    if start >= T:
        log.warning(
            "clip_id=%d: start=%d >= video length %d; skipping",
            clip_id, start, T,
        )
        return None

    window  = all_frames[start:end]
    win_len = len(window)
    span    = (n_out - 1) * stride + 1   # 15*4+1 = 61 frames needed

    if win_len >= span:
        # Center the stride-4 window: shift right by half the unused tail
        offset  = (win_len - span) // 2   # e.g. (81-61)//2 = 10
        indices = offset + np.arange(n_out) * stride
    else:
        # Fallback for clips shorter than the required span
        indices = np.linspace(0, win_len - 1, n_out, dtype=int)

    result = []
    for i in indices:
        # Step 1 — resize shortest edge to 256, bilinear (matches V-JEPA Resize transform)
        frame = cv2.resize(
            window[i],
            (RESIZE_SIZE, RESIZE_SIZE),
            interpolation=cv2.INTER_LINEAR,   # bilinear = V-JEPA Resize default
        )
        # Step 2 — center crop to 224×224
        top   = (RESIZE_SIZE - IMG_SIZE) // 2   # = 16
        left  = (RESIZE_SIZE - IMG_SIZE) // 2   # = 16
        frame = frame[top:top + IMG_SIZE, left:left + IMG_SIZE]
        result.append(frame)

    return np.stack(result)  # (n_out, 224, 224, 3)


# ── Step 5: Preprocessing ─────────────────────────────────────────────────────

def preprocess_clip(frames: np.ndarray) -> torch.Tensor:
    """
    Convert (16, 224, 224, 3) uint8 RGB numpy array to a normalised tensor
    in V-JEPA's expected input format: (1, 3, 16, 224, 224) float32.

    Steps:
      1. Scale pixels to [0, 1]
      2. Subtract ImageNet channel means, divide by stds
         (same constants used during V-JEPA pretraining)
      3. Transpose from (T, H, W, C) to (C, T, H, W)  ← V-JEPA channel-first format
      4. Add batch dimension → (1, 3, T, H, W)

    Note on input format:
        VideoMAE expects (B, T, 3, H, W) — time before channels.
        V-JEPA expects  (B, 3, T, H, W) — channels before time.
    The transpose in step 3 handles this difference.
    """
    x = frames.astype(np.float32) / 255.0             # (T, H, W, 3), [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD            # normalise per channel
    x = x.transpose(3, 0, 1, 2)                       # (3, T, H, W)
    x = torch.from_numpy(x).unsqueeze(0)              # (1, 3, T, H, W)
    return x


# ── Step 6: Batch forward pass ────────────────────────────────────────────────

def encode_batch(
    model: torch.nn.Module,
    batch_tensors: List[torch.Tensor],
) -> np.ndarray:
    """
    Run V-JEPA encoder on a batch of clip tensors.

    Args:
        model:         V-JEPA encoder (eval mode, on DEVICE)
        batch_tensors: list of (1, 3, 16, 224, 224) tensors

    Returns:
        features_mean:     (B, 1024)    float32 — mean over all 1568 tokens
        features_max:      (B, 1024)    float32 — max  over all 1568 tokens
        features_temporal: (B, 8, 1024) float32 — mean over 196 spatial patches
                                                    per temporal step (8 steps)

    V-JEPA forward pass (no masks):
        Input  (B, 3, 16, 224, 224)
        → PatchEmbed3D → (B, 1568, 1024) spatiotemporal tokens
        → 24 Transformer blocks
        → LayerNorm
        → Output (B, 1568, 1024)

    Token layout: 1568 = 8 temporal × 196 spatial (14×14 patches).
    Mean/max pooling collapse all structure → (B, 1024).
    Temporal pooling preserves time axis → (B, 8, 1024) for attentive probe.
    """
    dtype = torch.float16 if USE_FP16 else torch.float32
    pv = torch.cat(batch_tensors, dim=0).to(device=DEVICE, dtype=dtype)  # (B, 3, 16, 224, 224)

    with torch.no_grad():
        tokens = model(pv)                             # (B, 1568, 1024)

    # Cast back to float32 for storage.
    tokens_cpu = tokens.cpu().float().numpy()          # (B, 1568, 1024)

    # ── Global pooling (for linear probe / ordinal_regression.py) ─────────────
    features_mean = tokens_cpu.mean(axis=1)            # (B, 1024)
    features_max  = tokens_cpu.max(axis=1)             # (B, 1024)

    # ── Temporal pooling (for attentive probe / train_attention.py) ───────────
    # V-JEPA ViT-L/16 with 16 input frames (tubelet_size=2) produces:
    #   8 temporal steps × 14×14 spatial patches = 1568 tokens
    # We mean-pool over the 196 spatial patches per time step to obtain
    # 8 temporal feature vectors: (B, 8, 1024).
    # This preserves temporal structure at feasible storage cost (~230 MB total).
    # The attentive probe in train_attention.py learns weights over these 8 steps.
    # Saving the full 1568 tokens would cost ~45 GB — not feasible.
    N_TEMPORAL  = 8    # 16 frames / tubelet_size 2
    N_SPATIAL   = 196  # 14 × 14 patches
    features_temporal = (
        tokens_cpu
        .reshape(-1, N_TEMPORAL, N_SPATIAL, EMBED_DIM)  # (B, 8, 196, 1024)
        .mean(axis=2)                                    # (B, 8, 1024)
    )

    return features_mean, features_max, features_temporal


# ── Step 7: Main extraction loop ──────────────────────────────────────────────

def main() -> None:

    # ── 7a. Verify setup ──────────────────────────────────────────────────────
    # Checks that the jepa repo and checkpoint exist before loading anything.
    setup_jepa_path()

    # ── 7b. Load clip metadata ────────────────────────────────────────────────
    # scored_clip_rows.pkl was produced by ds_combiner.py.
    # Each row contains: dataset, subject, trial, clip_id, x (keypoints), UPDRS_GAIT.
    # We use dataset/subject/trial to locate the rendered MP4; clip_id to find
    # the correct 81-frame window within that video.
    log.info("Loading clip metadata from %s", CLIPS_PATH)
    with open(CLIPS_PATH, "rb") as f:
        scored_rows = pickle.load(f)
    log.info("Total scored clips: %d", len(scored_rows))

    # ── 7c. Skip already-extracted datasets ───────────────────────────────────
    # Allows incremental re-runs: if PD-GaM is already in the output file,
    # only BMClab and 3DGait will be processed (or vice versa).
    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            already_done = {r["dataset"] for r in pickle.load(f)}
    else:
        already_done = set()

    todo_datasets = RENDERED_DATASETS - already_done
    if not todo_datasets:
        log.info("All datasets already extracted. Nothing to do.")
        return
    if already_done:
        log.info("Already extracted: %s — skipping.", already_done)
    log.info("Extracting: %s", todo_datasets)

    scored_rows = [r for r in scored_rows if r["dataset"] in todo_datasets]
    log.info("Clips to process: %d", len(scored_rows))

    # ── 7d. Load model ────────────────────────────────────────────────────────
    model = load_vjepa_encoder()

    # ── 7e. Extraction loop ───────────────────────────────────────────────────
    # Strategy: accumulate BATCH_SIZE clips, then do one forward pass.
    # Videos are loaded one at a time; the last loaded video is cached so that
    # consecutive clips from the same trial don't re-read the MP4.
    last_video_path: Optional[Path]       = None
    last_frames:     Optional[np.ndarray] = None

    batch_tensors: List[torch.Tensor] = []   # clip tensors waiting for forward pass
    batch_meta:    List[dict]         = []   # metadata for each clip in the batch

    features = []
    skipped  = 0

    def flush_batch() -> None:
        """Forward-pass the current batch, append results to features."""
        if not batch_tensors:
            return
        feat_mean, feat_max, feat_temporal = encode_batch(model, batch_tensors)
        for meta, fmean, fmax, ftemporal in zip(
            batch_meta, feat_mean, feat_max, feat_temporal
        ):
            features.append({
                **meta,
                "feature":          fmean,      # (1024,)  mean-pooled — linear probe
                "feature_max":      fmax,        # (1024,)  max-pooled  — ablation
                "feature_temporal": ftemporal,   # (8, 1024) temporal   — attentive probe
            })
        batch_tensors.clear()
        batch_meta.clear()
        # Free MPS/CUDA memory after every batch to prevent gradual OOM.
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    for row in tqdm(scored_rows, desc="V-JEPA feature extraction"):
        dataset = row["dataset"]
        subject = row["subject"]
        trial   = row["trial"]
        clip_id = row["clip_id"]
        label   = int(row["UPDRS_GAIT"])

        # Validate label before processing (catches data issues early).
        if label not in {0, 1, 2, 3}:
            log.warning("Unexpected UPDRS %d in %s/%s/%s — skipped", label, dataset, subject, trial)
            skipped += 1
            continue

        # Some datasets (BMClab) append a _downX suffix in scored_clip_rows
        # from downsampling in ds_combiner.py. The rendered video uses the
        # base trial name without this suffix.
        video_trial = re.sub(r"_down\d+$", "", trial)
        video_path  = VIDEO_ROOT / dataset / subject / f"{video_trial}.mp4"

        # Load video only when the trial changes (one read per MP4).
        if video_path != last_video_path:
            last_frames     = load_video_frames(video_path)
            last_video_path = video_path

        if last_frames is None:
            skipped += 1
            continue

        # Sample 16 uniformly-spaced frames from this clip's 81-frame window.
        clip_frames = sample_clip_frames(last_frames, clip_id, CLIP_LEN, NUM_FRAMES)
        if clip_frames is None:
            skipped += 1
            continue

        # Normalise + reshape to V-JEPA input format (1, 3, 16, 224, 224).
        tensor = preprocess_clip(clip_frames)

        batch_tensors.append(tensor)
        batch_meta.append({
            "dataset":    dataset,
            "subject":    subject,
            "trial":      trial,
            "clip_id":    clip_id,
            "UPDRS_GAIT": label,
        })

        # When the batch is full, run a forward pass.
        if len(batch_tensors) >= BATCH_SIZE:
            flush_batch()

    # Process any remaining clips that didn't fill a full batch.
    flush_batch()

    log.info("Extracted features for %d clips", len(features))
    log.info("Skipped (missing video or invalid label): %d", skipped)
    if features:
        log.info("Mean-pool feature shape: %s", features[0]["feature"].shape)
        log.info("Max-pool  feature shape: %s", features[0]["feature_max"].shape)

    # ── 7f. Merge with existing output and save ───────────────────────────────
    # Append newly extracted datasets to the existing file (if any), replacing
    # any entries that belong to the just-processed datasets (re-extraction).
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)

    if FEATURES_OUT.exists():
        with open(FEATURES_OUT, "rb") as f:
            existing = pickle.load(f)
        new_datasets = {r["dataset"] for r in features}
        existing     = [r for r in existing if r["dataset"] not in new_datasets]
        merged       = existing + features
        log.info(
            "Merged with existing: kept %d + added %d = %d total",
            len(existing), len(features), len(merged),
        )
    else:
        merged = features

    with open(FEATURES_OUT, "wb") as f:
        pickle.dump(merged, f)
    log.info("Saved %d feature vectors to %s", len(merged), FEATURES_OUT)


if __name__ == "__main__":
    main()
