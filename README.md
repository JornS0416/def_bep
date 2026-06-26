# Video Foundation Models for Parkinson's Gait Severity

BEP research code. It evaluates **frozen** video/image foundation models —
VideoMAE, DINOv2, V-JEPA 1 and V-JEPA 2 — as feature extractors for UPDRS-gait
severity classification on the **Care-PD** benchmark. Subjects are represented as
2D side-view **silhouette videos** rendered from SMPL meshes, so the models see
gait shape and motion without identity or appearance cues.

Each backbone is evaluated with a lightweight probe under four cross-validation
protocols and compared against the Care-PD paper's published encoders.

| Protocol | Question it answers |
|----------|---------------------|
| **LOSO** | Within one cohort, can we grade an unseen subject? |
| **CDA**  | Does a probe trained on cohort A transfer to cohort B? |
| **LODO** | Train on all cohorts but one — does it generalise to the held-out cohort? |
| **MIDA** | LOSO + extra cohorts added to training — how much does in-domain data help? |

## Repository layout

```
main_project/
├── README.md                  ← you are here
├── requirements.txt
├── scripts/
│   ├── 2D_rendering/           SMPL meshes → side-view silhouette MP4s
│   ├── videomae/  dinov2/  vjepa/  vjepa2/    per-backbone feature extractors
│   └── _evaluation/            evaluation pipeline (see its own README)
│       ├── run_all.py          ← control panel: run evaluations + figures
│       ├── loso/ cda/ lodo/ mida/   one script per protocol × backbone
│       └── carepd_metrics.py, aggregate_results.py, plot_results.py, ...
├── src/                        rendering + data-loading library code
│   ├── data/                   dataset loading + provenance check (ds_combiner)
│   └── rendering/              SMPL → silhouette video utilities
├── model/jepa/                 vendored V-JEPA 1 code (Meta)
├── utils/smpl/                 SMPL loader code (models are NOT shipped — see Data)
├── assets/
│   ├── baseline/               Care-PD published baseline numbers (committed)
│   ├── datasets/               cohorts, rendered videos, feature pickles (NOT committed)
│   └── checkpoints/            backbone checkpoints (NOT committed)
├── docs/                       methodology appendix
└── outputs/runs/<date>/        one committed example run (results + figures)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements.txt
```

> **numpy 2.x is required.** The cached V-JEPA 2 feature pickles were created
> under numpy 2.x and only unpickle with `numpy >= 2.0`. This is the single
> environment that runs every step.

All scripts use project-root-relative paths; run them from the `main_project/`
directory. There are no machine-specific paths and no cloud dependencies.

## Data

The large data is **not** included in the repository (multi-GB, and the SMPL
body models are licence-restricted). To run end-to-end you need:

1. **Care-PD cohorts** → `assets/datasets/base_sets/<cohort>.pkl`
   (PD-GaM, BMClab, 3DGait, T-SDU-PD for evaluation). From the Care-PD benchmark.
2. **SMPL body models** → `utils/smpl/models/` (only needed for rendering).
   Obtain from <https://smpl.is.tue.mpg.de/> (their licence forbids redistribution).
3. **Backbone checkpoints** are downloaded on demand by the extractors; V-JEPA 1's
   ViT-L/16 is a manual one-liner documented in `scripts/vjepa/extract_features.py`.

For graders, the rendered videos and cached feature pickles can be supplied
separately so the evaluation can be reproduced without re-rendering.

## The pipeline

```
base_sets/*.pkl
   → scripts/2D_rendering/render_dataset.py        → rendered_videos/<cohort>/...
   → scripts/<backbone>/extract_features.py        → fabricated_datasets/<backbone>_features*.pkl
   → scripts/_evaluation/run_all.py                → outputs/runs/<date>/{results,aggregate,figures}
```

Each stage is independent and cached, so you can join the chain at whichever
artefact you already have (e.g. start from the feature pickles to reproduce all
results without rendering or re-extracting).

## Reproducing the results

The evaluation is driven by a single control panel. From `main_project/`:

```bash
# Full headline test: all protocols, V-JEPA 2 as the default V-JEPA (no V-JEPA 1)
python scripts/_evaluation/run_all.py --preset default --post all --yes

# Interactive menu instead:
python scripts/_evaluation/run_all.py
```

Results, aggregated tables and all figures land in a fresh
`outputs/runs/<timestamp>/`. **V-JEPA 1 is opt-in** (it is only needed for the
V-JEPA 1 vs 2 comparison): add it with `--preset all` or `--preset vjepa`.

See [`scripts/_evaluation/README.md`](scripts/_evaluation/README.md) for the full
menu, presets, the `run_all` vs `run_paths` roles, and the output layout.

## Notes

- A committed example run is kept under `outputs/runs/` so the expected tables
  and figures can be inspected without re-running anything.
- Evaluation metrics follow the Care-PD reference implementation; the derivation
  is in [`docs/appendix_evaluation_metrics.md`](docs/appendix_evaluation_metrics.md).
