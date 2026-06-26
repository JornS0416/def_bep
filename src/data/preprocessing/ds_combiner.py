"""
Dataset provenance & integrity check
=====================================
Single source of truth for HOW the evaluated clip dataset is built and a
validator that proves the frozen artefact is consistent with the raw cohorts.

PROVENANCE (how `scored_clip_rows.pkl` came to be)
--------------------------------------------------
    assets/datasets/base_sets/<cohort>.pkl        # raw per-trial SMPL + labels
        → render to side-view silhouette MP4s     # src/rendering/*
          (per-cohort windowing + downsampling;   #   adds the _widNN / _downN
           baked into the MP4 file names under    #   suffixes seen in trial names)
           assets/datasets/rendered_videos/)
        → one row per 81-frame clip window of each MP4, joined back to its
          UPDRS-gait label  → assets/datasets/fabricated_datasets/scored_clip_rows.pkl

`scored_clip_rows.pkl` is metadata only — (dataset, subject, trial, clip_id,
UPDRS_GAIT) — and is the join key every feature extractor and every evaluation
script uses.

WHY THIS FILE NO LONGER REBUILDS THE PICKLE
-------------------------------------------
The render-time windowing/downsampling that produced the exact MP4 segments
(and therefore the exact clip counts) is a one-off preprocessing step that is
not checked into the repo, so the frozen `scored_clip_rows.pkl` cannot be
re-derived byte-for-byte from the raw cohorts alone. We therefore treat the
pickle as the frozen dataset artefact and VERIFY it here instead of pretending
to regenerate it: every clip is traced back to a `base_sets` trial and its
label is re-checked. This catches the only thing that can silently corrupt
results — a wrong or drifted label — without depending on the lost windowing.

Run:  python -m src.data.preprocessing.ds_combiner       (from main_project/)
"""

import pickle
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_SETS    = PROJECT_ROOT / "assets/datasets/base_sets"
FABRICATED   = PROJECT_ROOT / "assets/datasets/fabricated_datasets"

CLIP_LEN = 81   # frames per clip window — the unit every extractor samples within

# Evaluation cohort -> base_sets filename.
LABELED_COHORTS = {
    "PD-GaM":   "PD-GaM.pkl",
    "BMClab":   "BMClab.pkl",
    "3DGait":   "3DGait.pkl",
    "T-SDU-PD": "T-SDU-PD.pkl",
}


def _label_map(filename):
    """base_sets/<file> -> {subject(str): {trial(str): UPDRS_GAIT}}."""
    bs = pickle.load(open(BASE_SETS / filename, "rb"))
    out = defaultdict(dict)
    for subject, trials in bs.items():
        for trial, meta in trials.items():
            out[str(subject)][str(trial)] = meta.get("UPDRS_GAIT")
    return out


def _resolve(label_map, subject, trial_name):
    """Map a rendered trial name (e.g. 'SUB01_off_walk_1_down0') back to its
    base trial via longest-prefix match, returning (base_trial, label)."""
    trials = label_map.get(str(subject), {})
    cands = [t for t in trials if str(trial_name).startswith(t)]
    if not cands:
        return None, None
    base_trial = max(cands, key=len)
    return base_trial, trials[base_trial]


def validate():
    """Trace every clip in scored_clip_rows.pkl back to a base_sets trial and
    re-check its label. Returns True iff every clip resolves and agrees."""
    scored = pickle.load(open(FABRICATED / "scored_clip_rows.pkl", "rb"))
    maps = {ds: _label_map(fn) for ds, fn in LABELED_COHORTS.items()}

    per_ds = defaultdict(lambda: {"clips": 0, "unresolved": 0, "label_mismatch": 0})
    for r in scored:
        ds = r["dataset"]
        s = per_ds[ds]
        s["clips"] += 1
        _, label = _resolve(maps[ds], r["subject"], r["trial"])
        if label is None:
            s["unresolved"] += 1
        elif label != r["UPDRS_GAIT"]:
            s["label_mismatch"] += 1

    ok = True
    print(f"{'cohort':10s} {'clips':>7s} {'unresolved':>11s} {'label_mismatch':>15s}")
    for ds in LABELED_COHORTS:
        s = per_ds[ds]
        print(f"{ds:10s} {s['clips']:7d} {s['unresolved']:11d} {s['label_mismatch']:15d}")
        ok &= (s["unresolved"] == 0 and s["label_mismatch"] == 0)
    print("\nVALIDATION:", "PASS — every clip traces to a base_sets label"
          if ok else "FAIL — see non-zero columns above")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
