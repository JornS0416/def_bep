"""
Trial Renderer
==============
Converts one Care-PD trial (SMPL parameters) into a 2D silhouette MP4.

Pipeline per trial:
    SMPL params → 3D vertices (batch) → pyrender frames → cv2 MP4

Design choices:
- Silhouette rendering (black body on white bg): removes appearance variation
  (clothing, skin colour, lighting) and forces the model to rely purely on
  body shape and motion. Consistent with the proposal (sec. 5, Fig. 1) and
  with prior work showing side-view silhouettes are optimal for gait analysis
  (Chen et al., 2025).
- Side-view camera (fixed): the side view maximally exposes spatiotemporal
  gait features such as stride length, arm swing, and trunk lean (Chen et al.,
  2025). Camera pose is fixed across all trials for cross-dataset consistency.
- Batch SMPL forward pass: all T frames of a trial are processed in one call
  to pose_to_vertices_batched, avoiding T separate model invocations. This is
  the main rendering speed-up (~10-30× vs frame-by-frame).
- Pyrender scene reuse: the camera and light nodes are created once; only the
  mesh node is swapped each frame. Avoids scene construction overhead per frame.
- cv2 VideoWriter (mp4v codec): faster than imageio for sequential frame writes
  and produces smaller files than the default imageio ffmpeg backend.
- Resolution 224×224: matches VideoMAE's expected input size, eliminating the
  need for a resize step during feature extraction.
"""

import numpy as np
import pyrender
import trimesh
import cv2
from tqdm import tqdm
from src.rendering.preprocessing import normalize_translation
from src.rendering.rendering import create_side_camera, create_light
from src.rendering.smpl_utils import pose_to_vertices_batched


def render_trial(trial, model, renderer, output_path, fps=30, show_progress=False):
    """
    Render all frames of a single trial to an MP4 file.

    Args:
        trial:       dict with keys 'pose' (T,72), 'trans' (T,3), 'beta' (1,10).
        model:       smplx SMPL model instance (reused across trials).
        renderer:    pyrender.OffscreenRenderer (224×224).
        output_path: destination .mp4 path.
        fps:         frame rate; taken from trial metadata (default 30).
    """
    pose  = trial["pose"]
    trans = normalize_translation(trial["trans"])
    beta  = trial["beta"]

    # Single batched SMPL forward pass for all T frames — model is reused
    all_vertices = pose_to_vertices_batched(model, pose, trans, beta)

    # Build static scene elements once
    camera, camera_pose = create_side_camera()
    light               = create_light()
    scene               = pyrender.Scene(bg_color=[1, 1, 1, 1])  # white background
    scene.add(camera, pose=camera_pose)
    scene.add(light,  pose=camera_pose)

    frames    = []
    mesh_node = None

    for i in tqdm(range(len(all_vertices)), disable=not show_progress):
        # Swap mesh node each frame; reuse everything else
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        tm        = trimesh.Trimesh(vertices=all_vertices[i], faces=model.faces, process=False)
        mesh_node = scene.add(pyrender.Mesh.from_trimesh(tm))
        color, _  = renderer.render(scene)
        frames.append(color)

    # Write video: cv2 is faster than imageio for sequential writes
    h, w   = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
