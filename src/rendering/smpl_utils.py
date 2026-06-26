"""
SMPL Utilities
==============
Wraps the smplx library for converting SMPL parameters to 3D mesh vertices.

Design choices:
- gender='neutral': Care-PD subjects span both sexes; a single neutral model
  avoids the need for per-subject gender labels and keeps preprocessing uniform.
- use_pca=False: use the full 69-dim body pose instead of a PCA-reduced space,
  preserving all joint degrees of freedom present in the Care-PD annotations.
- pose_to_vertices_batched: processes all T frames of a trial in one forward
  pass by instantiating a model with batch_size=T. This avoids the O(T)
  overhead of T separate forward passes and is the primary render speed-up.
  beta (shape) is shared per subject and tiled to match the batch dimension.
"""

import numpy as np
import torch
import smplx


def create_smpl_model(model_path, batch_size=1):
    """Instantiate a neutral SMPL model. batch_size=1 is used for single-frame
    inference; use pose_to_vertices_batched for full-trial batching."""
    return smplx.create(
        model_path=model_path,
        model_type="smpl",
        gender="neutral",
        use_pca=False,
        batch_size=batch_size,
    )


def split_pose(pose_tensor):
    """Split SMPL 72-dim pose into global orientation (3) and body pose (69)."""
    return pose_tensor[:, :3], pose_tensor[:, 3:]


def pose_to_vertices_batched(model, pose, trans, beta):
    """
    Compute SMPL vertices for all T frames of a trial in a single forward pass.

    Args:
        model: existing smplx SMPL model instance (batch_size=1 at creation is fine;
               smplx accepts any batch size when explicit tensors are passed).
        pose:  (T, 72) array of SMPL pose parameters.
        trans: (T,  3) array of root translations (already normalized).
        beta:  (1, 10) or (10,) array of shape coefficients (per subject).

    Returns:
        vertices: (T, 6890, 3) float32 array.

    Note: the model is reused across trials — no re-instantiation overhead.
    smplx ignores its internal batch_size when explicit input tensors are given.
    """
    T = len(pose)

    pose_tensor  = torch.tensor(pose,  dtype=torch.float32)
    trans_tensor = torch.tensor(trans, dtype=torch.float32)

    # beta is defined per subject (shape=(1,10) or (10,)); tile to (T, 10)
    beta_np = np.array(beta)
    if beta_np.ndim == 1:
        beta_np = beta_np[np.newaxis, :]
    beta_tensor = torch.tensor(np.repeat(beta_np, T, axis=0), dtype=torch.float32)

    global_orient, body_pose = split_pose(pose_tensor)

    with torch.no_grad():
        output = model(
            betas=beta_tensor,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=trans_tensor,
        )
    return output.vertices.detach().cpu().numpy()  # (T, 6890, 3)


def pose_to_vertices(model, pose, trans, beta):
    """Single-frame SMPL forward pass. Kept for compatibility."""
    pose_tensor  = torch.tensor(pose,  dtype=torch.float32)
    trans_tensor = torch.tensor(trans, dtype=torch.float32)
    beta_tensor  = torch.tensor(beta,  dtype=torch.float32)
    global_orient, body_pose = split_pose(pose_tensor)
    output = model(
        betas=beta_tensor,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=trans_tensor,
    )
    return output.vertices.detach().cpu().numpy()
