# Marketplace Policies Code

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Reproduction](https://img.shields.io/badge/reproduction-artifact--level-16a34a.svg)](#reproducibility-boundary)

This repository reproduces the empirical artifacts for the paper:

**From Auction Replay to Launch Readiness: A Decision-Support Framework for Ads Marketplace Policies**

The code is organized as a small, object-oriented Python package. It regenerates the paper figures and result tables from the manuscript artifact layer, validates the headline numerical claims, and creates a clean reproduction bundle that can be archived with the paper or uploaded to GitHub.

<p align="center">
  <img src="images/mermaid-diagram.png" alt="Marketplace policy reproduction architecture" width="55%">
</p>

## Visual Overview

The repo is built around one reproducibility contract: start from the manuscript artifact layer, regenerate the publication outputs, and verify that the numerical claims used in the paper still hold.

<p align="center">
  <img src="docs/figures/evidence_stack.svg" alt="Evidence stack for launch readiness" width="100%">
</p>

At a glance:

| Component | Class | Output |
| --- | --- | --- |
| Artifact access | `ArtifactRepository` | Loads metadata, tables, parquet panels, and Overleaf sources |
| Figure generation | `FigureRenderer` | Rebuilds paper-facing diagnostic and validation figures |
| Claim checks | `ResultValidator` | Verifies policy, lift, holdout, ablation, and action claims |
| Full run | `PaperReproductionPipeline` | Creates figures, tables, reports, manifest, and bundle |

## What This Repo Reproduces

The pipeline reproduces the paper-facing results from the local `metadata/`, `tables/`, `figures/`, and `data/processed/` artifacts:

- auction price and outcome-density diagnostics
- nuisance-model calibration diagnostics
- reserve/floor replay frontier and daily stability
- season-three out-of-time validation figures
- decision-rule ablation figures
- OPE, support, lower-tail, and sensitivity figures
- launch-readiness and validation-design figures
- paper tables selected in `final_table_selection.csv`
- headline claim checks used in the manuscript

The raw iPinYou archives are intentionally not committed here. Place raw or processed data under a local `data/` directory, or point `--source-root` at the manuscript artifact directory that already contains the processed outputs.

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
│   ├── repository.py
│   └── results.py
└── tests/
```

## Quick Start

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
marketplace-policies reproduce --source-root .. --output-root artifacts
```

In this workspace, `--source-root ..` points to:

```text
notebooks/writing/ads_marketplace_auction_experimentation/
```

That directory contains the current `metadata/`, `tables/`, `figures/`, `data/processed/`, and `overleaf/` folders.

## Commands

Validate headline results only:

```bash
marketplace-policies check --source-root ..
```

Render figures only:

```bash
marketplace-policies figures --source-root .. --output-root artifacts
```

Export selected paper tables only:

```bash
marketplace-policies tables --source-root .. --output-root artifacts
```

Run the full artifact-level reproduction:

```bash
marketplace-policies reproduce --source-root .. --output-root artifacts
```

Equivalent script entry point:

```bash
python scripts/reproduce_paper.py --source-root .. --output-root artifacts
```

## Expected Headline Checks

The `check` command verifies that the local artifacts reproduce the headline results:

- priority policy: `hybrid_q75_if_gap_100`
- reader-facing name: `Q75 Margin-Gated Floor`
- season-two replay lift: approximately `47.7%`
- conservative cross-fitted DR lower-tail lift: approximately `45.8%`
- break-even marketplace response loss: approximately `32.3%`
- season-three holdout replay lift: approximately `43.9%`
- decision-rule ablation: simplified rules select the same policy but overclaim direct launch
- full DSS action: online validation rather than direct launch

## Data Notes

This repo is designed to be GitHub-friendly:

- large data files are ignored by default
- generated artifacts go under `artifacts/`
- source CSV/parquet artifacts can be supplied externally
- all figures are regenerated from tabular artifacts where possible

For a fully independent public release, upload the small artifact CSV files to a release asset or data repository and document the DOI or download URL here. If raw iPinYou logs are redistributed, confirm that redistribution is allowed by the dataset license.

## License

This code repository is released under the MIT License. See `LICENSE` for details.

## Development

Run tests:

```bash
pytest
```

Lint:

```bash
ruff check .
```

## Reproducibility Boundary

The package is an artifact-level reproduction repo. It regenerates the paper's figures, paper tables, checks, and bundles from the processed metadata/table layer. The original exploratory notebooks remain useful provenance, but they are no longer required to regenerate the manuscript-facing outputs.
