# Marketplace Policies Code

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Package manager](https://img.shields.io/badge/package%20manager-uv-5c4ee5.svg)](pyproject.toml)
[![Reproduction](https://img.shields.io/badge/reproduction-raw--data--to--paper-16a34a.svg)](#reproducibility-boundary)

This repository reproduces the empirical artifacts for the paper:

**Paper: From Auction Replay to Launch Readiness: A Decision-Support Framework for Ads Marketplace Policies**

The code is organized as a small, object-oriented Python package. It reads the original local iPinYou archive, rebuilds the bid-opportunity panels, runs the nuisance-model and assumption-aware off-policy evaluation workflow used in the paper, regenerates the paper figures and result tables, validates the decision claims, and creates a clean reproduction bundle that can be archived with the paper or uploaded to GitHub.

<p align="center">
  <img src="images/mermaid-diagram.png" alt="Marketplace policy reproduction architecture" width="70%">
</p>

## Pipeline Overview

The repo is built around one reproducibility contract: start from the original local iPinYou data, regenerate the analysis artifacts, and then regenerate the publication outputs.

| Component | Class | Output |
| --- | --- | --- |
| Raw data access | `IpinYouArchive` | Reads bz2 members inside the original iPinYou zip archive |
| Panel construction | `OpportunityPanelBuilder` | Builds season-two and season-three bid-opportunity panels |
| Nuisance modeling | `NuisanceModelTrainer` | Fits the LightGBM nuisance models and calibration diagnostics |
| Policy replay | `ReservePolicyCatalog`, `PolicyReplayAnalyzer` | Replays non-decreasing reserve/floor policies |
| Evidence synthesis | `DerivedEvidenceBuilder` | Runs the simulated logger, HistGradientBoosting OPE, cross-fitted DR, bootstrap ranking, heterogeneity, validation, scorecards, theory, and ablations |
| Figure generation | `FigureRenderer` | Rebuilds paper-facing diagnostic and validation figures |
| Claim checks | `ResultValidator` | Verifies policy, lift, holdout, ablation, and action claims |
| Full run | `RawToPaperPipeline`, `PaperReproductionPipeline` | Creates analysis artifacts, figures, tables, reports, manifest, and bundle |

## What This Repo Reproduces

The pipeline reproduces the paper-facing results from a local copy of the original iPinYou archive. It intentionally keeps the generated workspace narrow: after the raw analysis finishes, non-paper exploratory intermediates are pruned so the remaining artifacts correspond to manuscript figures, manuscript tables, checks, or the minimal data needed to regenerate them.

- auction price and outcome-density diagnostics
- LightGBM nuisance-model metrics and calibration diagnostics
- reserve/floor replay frontier and daily stability
- season-three out-of-time validation figures
- decision-rule ablation figures
- simulated-propensity IPS/SNIPS/DR diagnostics, support diagnostics, cross-fitted DR lower-tail ranking, downside-risk, heterogeneity, and sensitivity figures
- launch-readiness and validation-design figures
- paper tables selected by generated `final_table_selection.csv`
- headline decision checks used in the manuscript

No result artifacts are committed. The local `data/` and generated `artifacts/` folders are ignored by Git.

## Repository Layout

```text
marketplace-policies-code/
├── configs/default.toml
├── pyproject.toml
├── README.md
├── scripts/reproduce_paper.py
├── src/marketplace_policies_code/
│   ├── cli.py
│   ├── config.py
│   ├── figures.py
│   ├── pipeline.py
│   ├── raw_pipeline.py
│   ├── repository.py
│   └── results.py
└── tests/
```

## Quick Start

This repository assumes [`uv`](https://docs.astral.sh/uv/) is available for environment creation and package installation.

Place the original iPinYou archive here:

```text
data/ipinyou/archive.zip
```

From this folder:

```bash
uv sync --extra dev
uv run marketplace-policies reproduce --quick
```

The quick run uses a bounded subset of days and rows so the pipeline can be tested on a laptop. It writes generated analysis artifacts to `artifacts/workspace/` and publication outputs to `artifacts/`.

All commands print timestamped progress messages while they run, including the current pipeline stage, data shard, model, figure, and validation step. Add `--quiet` to suppress these messages.

For paper-scale regeneration, run:

```bash
uv run marketplace-policies reproduce --full
```

The full run reads every available season-two training day and the season-three validation window. It can take a long time and will create large local parquet files under `artifacts/workspace/data/processed/`.

## Python Dependencies

The main runtime packages are:

```text
pandas
numpy
pyarrow
matplotlib
seaborn
graphviz
scikit-learn
lightgbm
```

`scikit-learn` and `lightgbm` are required for the paper reproduction pipeline: LightGBM fits the nuisance models, and `HistGradientBoostingRegressor` powers the simulated-logger direct-method and doubly robust diagnostics. The Python `graphviz` package also needs the Graphviz `dot` executable available on the system path.

`uv sync --extra dev` installs the runtime and development dependencies declared in `pyproject.toml`.

## Commands

All commands are exposed through the `marketplace-policies` CLI after `uv sync --extra dev`.

| Command | Reads | Writes | Use when |
| --- | --- | --- | --- |
| `reproduce` | Raw iPinYou archive by default | Paper-facing `artifacts/workspace/` and `artifacts/` outputs | You want the complete raw-data-to-paper run |
| `build-artifacts` | Raw iPinYou archive | Paper-facing analysis artifacts under `artifacts/workspace/` | You only want to process raw data and stop before figures/tables/checks |
| `figures` | Existing generated artifacts under `--source-root` | PNG figures under `artifacts/figures/` | You changed figure code or already have analysis artifacts |
| `tables` | Existing generated artifacts under `--source-root` | Selected CSV tables under `artifacts/tables/` | You want the paper-facing result tables only |
| `check` | Existing generated artifacts under `--source-root` | Claim-check reports under `artifacts/reports/` | You want to verify headline decision claims |

### Full Reproduction

```bash
uv run marketplace-policies reproduce --full
```

This is the main paper-scale command. It expects `data/ipinyou/archive.zip`, builds the season-two development panel, fits nuisance models, replays reserve/floor policies, runs the simulated-logger OPE and cross-fitted DR diagnostics, validates the priority policy on season three, regenerates only the figures and tables used in the manuscript, validates headline claims, and writes an archival bundle.

Outputs:

- `artifacts/workspace/metadata/`: paper-used analysis CSVs and figure dependencies
- `artifacts/workspace/tables/`: manuscript table CSVs
- `artifacts/workspace/data/processed/`: minimal generated parquet sample needed for the price-distribution figure
- `artifacts/figures/`: regenerated PNG figures
- `artifacts/tables/`: selected paper-facing tables
- `artifacts/reports/`: claim checks and result summaries
- `artifacts/bundle/`: compact bundle of regenerated outputs

### Quick Smoke Test

```bash
uv run marketplace-policies reproduce --quick
```

This runs the same pipeline shape as `--full`, but on a bounded subset of days and rows. Use it first after cloning to confirm that the archive path, Python environment, Graphviz installation, model dependencies, and output folders are all working.

The quick run is useful for development checks, but it is not the final paper-scale reproduction.

### Build Analysis Artifacts Only

```bash
uv run marketplace-policies build-artifacts --quick
```

This reads the raw iPinYou archive and stops after generating the pruned paper-facing analysis workspace. It does not render figures, export selected tables, validate claim checks, or create the final bundle.

Use this when you want to inspect or debug the data-processing and estimation artifacts before running the full publication-output stage. Switch to `--full` when you want the paper-scale analysis artifacts:

```bash
uv run marketplace-policies build-artifacts --full
```

### Reuse Existing Analysis Artifacts

```bash
uv run marketplace-policies reproduce --skip-analysis --source-root artifacts/workspace
```

This skips raw-data processing and reuses an existing generated workspace. It is the fastest way to rerender figures, export selected tables, rerun claim checks, write summaries, and rebuild the output bundle after you have already run `build-artifacts` or a previous `reproduce` command.

### Render Figures Only

```bash
uv run marketplace-policies figures --source-root artifacts/workspace
```

This reads generated CSV/parquet artifacts from `--source-root` and writes only the PNG figures to `artifacts/figures/`. It does not rebuild raw panels or refit models.

Use this after changing plotting code, labels, colors, sizing, or figure selection.

### Export Tables Only

```bash
uv run marketplace-policies tables --source-root artifacts/workspace
```

This reads `final_table_selection.csv` from the generated metadata and copies the selected paper-facing CSV tables into `artifacts/tables/`. It also writes a `table_index.csv` so the exported files can be matched back to their role in the paper.

### Validate Headline Claims Only

```bash
uv run marketplace-policies check --source-root artifacts/workspace
```

This reads the final generated decision artifacts and verifies that the main manuscript claims are internally consistent. It checks the priority policy, the reader-facing policy name, the season-three validation result, the ablation result, and the final recommendation that the policy should go through online validation rather than direct launch.

Outputs are written to `artifacts/reports/`.

Equivalent script entry point:

```bash
uv run python scripts/reproduce_paper.py --quick
```

The script is a convenience wrapper around the CLI. If no command is provided, it defaults to `reproduce`, so the command above is equivalent to:

```bash
uv run marketplace-policies reproduce --quick
```

### Shared Options

- `--quick`: bounded smoke-test run for local verification.
- `--full`: paper-scale run over all available season-two and season-three rows.
- `--data-root`: folder containing `ipinyou/archive.zip`; defaults to `data`.
- `--source-root`: generated analysis workspace; defaults to `artifacts/workspace`.
- `--output-root`: publication-output folder; defaults to `artifacts`.
- `--skip-analysis`: reuse existing generated artifacts instead of rebuilding from raw data.
- `--quiet`: suppress timestamped progress messages.

## Expected Headline Checks

The `check` command verifies that generated artifacts are internally consistent:

- priority policy: `hybrid_q75_if_gap_100`
- reader-facing name: `Q75 Margin-Gated Floor`
- decision-rule ablation: simplified rules select the same policy but overclaim direct launch
- full DSS action: online validation rather than direct launch

## Data Notes

This repo is designed to be GitHub-friendly:

- full raw and processed iPinYou data are ignored by default
- generated outputs go under `artifacts/`
- no result CSV, parquet, or paper artifacts are committed
- all paper-facing figures and tables are regenerated from local raw data
- external generated artifacts can still be supplied with `--source-root /path/to/artifact/root`

The original iPinYou archive is not redistributed in this repository. Users should obtain it separately and place it under `data/ipinyou/archive.zip`.

## License

This code repository is released under the MIT License. See `LICENSE` for details.

## Development

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

## Reproducibility Boundary

The package is a raw-data-to-paper reproduction repo, not a repository of precomputed paper artifacts. A fresh clone can regenerate the analysis artifacts, figures, selected paper tables, checks, and reports after the user places the original iPinYou archive under `data/ipinyou/archive.zip`. The reproduction path covers raw panel construction, LightGBM nuisance-model diagnostics, reserve/floor replay, simulated known-propensity OPE, cross-fitted DR, conservative bootstrap ranking, season-three validation, decision-rule ablations, and final launch-readiness checks. The quick mode is for smoke testing; `--full` is the paper-scale run.
