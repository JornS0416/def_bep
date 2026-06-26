import os
from pathlib import Path

# Run from the repo root (parent of main_project) so the "main_project/..."
# relative paths below resolve regardless of where this script is launched.
os.chdir(Path(__file__).resolve().parents[3])

from src.data.data_loading import load_dataset, get_trial
from src.rendering.smpl_utils import create_smpl_model
from src.rendering.render_trial import render_trial
import pyrender

data = load_dataset("main_project/assets/datasets/base_sets/PD-GaM.pkl")
trial = get_trial(data, "007", "007-13-000661_wid00_0")

model = create_smpl_model("main_project/utils/smpl/models/smpl_new")
renderer = pyrender.OffscreenRenderer(512, 512)

out_dir = Path("main_project/outputs/_scratch")
out_dir.mkdir(parents=True, exist_ok=True)

render_trial(
    trial=trial,
    model=model,
    renderer=renderer,
    output_path=str(out_dir / "single_trial_preview.mp4")
)