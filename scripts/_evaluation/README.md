# Evaluation pipeline

This folder reproduces every result, table and figure in the thesis. All output
lands in one tidy, timestamped folder under `outputs/runs/<YYYY-MM-DD_HHMMSS>/`.

## Quickstart

Run from the project root (`main_project/`) with the evaluation environment:

```bash
# The full headline test (V-JEPA 2 as the default V-JEPA, i.e. without V-JEPA 1)
python scripts/_evaluation/run_all.py --preset default --post all --yes

# Or interactively — shows a menu to pick protocols/models:
python scripts/_evaluation/run_all.py
```

Results, tables and figures appear in `outputs/runs/<timestamp>/` and the
`outputs/latest` symlink points at the most recent run.

### V-JEPA 1 vs V-JEPA 2

By default only **V-JEPA 2** runs. V-JEPA 1 stays opt-in so you can still
reproduce the V-JEPA 1 ↔ 2 comparison:

```bash
python scripts/_evaluation/run_all.py --preset all   --post all --yes   # both
python scripts/_evaluation/run_all.py --preset vjepa --post all --yes   # V-JEPA 1 only
```

## What runs

Four cross-validation protocols × four models:

| Protocol | Meaning                                            |
|----------|----------------------------------------------------|
| **LOSO** | Leave-One-Subject-Out, within a single dataset     |
| **CDA**  | Cross-Dataset: train on one dataset, test on others|
| **LODO** | Leave-One-Dataset-Out: train on D−1, test on the held-out one |
| **MIDA** | LOSO folds + the other datasets added to training  |

Models: VideoMAE, V-JEPA (1), V-JEPA 2, DINOv2. Each protocol's scripts live in
its own subfolder (`loso/`, `cda/`, `lodo/`, `mida/`).

Presets for `--preset` / the menu:
`default` (all, with V-JEPA 2), `all`, `fast`, `slow`, `loso`/`cda`/`lodo`/`mida`,
and per-model `videomae`/`vjepa`(=V-JEPA 1)/`vjepa2`/`dinov2`.

## Post-processing (`--post`)

- `aggregate` — `aggregate_results.py`: collects every result JSON into overview
  tables (`outputs/runs/<ts>/aggregate/`).
- `figures` — `plot_results.py`: all thesis figures (`.../figures/`).
- `carepd` — `baseline_comparison.py`: comparison against the Care-PD paper's
  published encoders (`.../results/_baseline_comparison/`).
- `confusion` — `confusion_tables.py`: pooled confusion matrices and per-class
  statistics for all four protocols (`.../tables/`). CDA's per-pair confusion
  matrices are read from its run JSON and summed per target dataset, so it
  appears in the same tables as LOSO/MIDA/LODO.

## The two `run_*` files (they are not duplicates)

- **`run_all.py`** — the control panel you launch. It shows the menu, creates the
  dated run folder, runs each evaluation as a subprocess and then the
  post-processing steps. **This is the entry point.**
- **`run_paths.py`** — a small helper module (one function, `output_root()`)
  imported by every evaluation script to decide *where* to write. When launched
  via `run_all.py` it returns that run's folder (via the `BEP_OUTPUT_ROOT` env
  var run_all sets); when a script is run by hand it falls back to
  `outputs/_scratch/`. You never run this file directly.

## Data prerequisites

Evaluations read cached feature pickles from
`assets/datasets/fabricated_datasets/` (`*_features*.pkl`). Those are produced by
the feature-extraction scripts in `scripts/<model>/extract_features.py`, which in
turn read the rendered silhouette videos (`assets/datasets/rendered_videos/`),
rendered from the Care-PD base sets via `scripts/2D_rendering/render_dataset.py`.

Full chain: `base_sets → render_dataset.py → rendered_videos → extract_features.py
→ fabricated_datasets/*.pkl → run_all.py → outputs/runs/<ts>/`.

> **V-JEPA 1 re-extraction** needs the ViT-L/16 checkpoint, which is *not*
> shipped. Download it (the command is in `scripts/vjepa/extract_features.py`):
> ```bash
> mkdir -p assets/checkpoints/vjepa
> curl -L https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar \
>      -o assets/checkpoints/vjepa/vitl16.pth.tar
> ```
> Running the *evaluation* does not need it — it uses the cached
> `vjepa_features.pkl`.
