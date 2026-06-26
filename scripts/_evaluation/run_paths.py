"""
Output-path indirection for the evaluation pipeline
===================================================
Single helper that decides WHERE evaluation artefacts are written, so the same
scripts can write either to a scratch area (standalone) or into a dated,
self-contained run folder (when launched via run_all.py).

`run_all.py` creates  outputs/runs/<YYYY-MM-DD_HHMMSS>/  and exports

    BEP_OUTPUT_ROOT=<that folder>

before launching each step. Every output-writing script routes its results,
aggregate tables and figures through `output_root()`, so a whole run lands in
one tidy, timestamped tree:

    outputs/runs/<ts>/results/{loso,mida,lodo,cda}/<model>/...
    outputs/runs/<ts>/aggregate/...
    outputs/runs/<ts>/figures/...

When BEP_OUTPUT_ROOT is not set (a script run by hand), outputs go to
`outputs/_scratch/` so they never clutter the restructured top-level tree.
"""

import os
from pathlib import Path

# main_project/  (run_paths.py lives in main_project/scripts/_evaluation/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def output_root() -> Path:
    """
    Base directory for all evaluation outputs.

    Returns $BEP_OUTPUT_ROOT if set (the per-run folder created by run_all.py),
    otherwise outputs/_scratch for ad-hoc standalone runs.
    """
    env = os.environ.get("BEP_OUTPUT_ROOT")
    if env:
        return Path(env)
    runs = _PROJECT_ROOT / "outputs" / "runs"
    if runs.is_dir():
        existing = [p for p in runs.iterdir() if p.is_dir()]
        if existing:
            return max(existing, key=lambda p: p.stat().st_mtime)
    return _PROJECT_ROOT / "outputs" / "_scratch"
