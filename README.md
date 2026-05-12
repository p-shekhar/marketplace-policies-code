# Marketplace Policies Code

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Package manager](https://img.shields.io/badge/package%20manager-uv-5c4ee5.svg)](pyproject.toml)
[![Workflow](https://img.shields.io/badge/workflow-notebook--first-16a34a.svg)](#notebook-workflow)

This repository contains the notebook-first reproduction code for the paper:

**From Auction Replay to Launch Readiness: A Decision-Support Framework for Ads Marketplace Policies**

The notebooks are the face of the repository. They walk through the analysis in the same order as the paper: data setup, exploratory diagnostics, nuisance-model checks, auction replay, guardrails, off-policy evaluation, robustness, out-of-time validation, launch-readiness design, and decision-rule ablation. The Python files in `src/` provide the reusable implementation used by those notebooks.

No generated data, figures, tables, reports, model artifacts, or paper PDFs are committed. The notebooks themselves are committed with their displayed outputs so readers can inspect the analysis narrative directly on GitHub.

<p align="center">
  <img src="images/mermaid-diagram.png" alt="Marketplace policy notebook architecture" width="82%">
</p>

## Reproducibility Boundary

This repo assumes that the user has a local copy of the original iPinYou data. The raw archive is not redistributed here.

Place the archive at:

```text
data/ipinyou/archive.zip
```

The notebooks generate local outputs under:

```text
artifacts/
```

Both `data/` and `artifacts/` are ignored by Git. This keeps the public repo lightweight while preserving a fully local, raw-data-to-results workflow.

## Notebook Workflow

Run the notebooks in numerical order. Each notebook is intentionally scoped to one part of the DSS rather than hiding the full analysis behind a single command.

| Notebook | Purpose | Main `src/` modules used |
| --- | --- | --- |
| `0_setup.ipynb` | Locate the raw iPinYou archive, build Season 2 and Season 3 panels, freeze the policy catalog | `data_access`, `panel_builder`, `policy_catalog` |
| `1_eda_figures.ipynb` | Produce the initial price and outcome-density EDA figures | `checks`, `figures` |
| `2_nuisance_model_diagnostics.ipynb` | Fit/check nuisance models and calibration diagnostics | `nuisance_models`, `figures` |
| `3_reserve_policy_tradeoff_and_daily_stability.ipynb` | Replay reserve policies, apply guardrails, inspect daily stability | `auction_replay`, `guardrails`, `figures` |
| `4_ope_estimator_diagnostics.ipynb` | Run simulated known-propensity OPE diagnostics and scorecard construction | `ope`, `figures` |
| `5_heterogeneity_and_downside_risk.ipynb` | Analyze segment heterogeneity and downside risk | `ope`, `figures` |
| `6_sensitivity_and_robustness.ipynb` | Run response-loss and support-collapse sensitivity checks | `theory_sensitivity`, `figures` |
| `7_external_season3_validation.ipynb` | Validate the frozen policy ranking on Season 3 | `external_validation`, `figures` |
| `8_validation_design_and_launch_readiness.ipynb` | Convert evidence into validation design and launch-readiness artifacts | `ope`, `decision_engine`, `figures` |
| `9_decision_rule_ablation.ipynb` | Compare simplified decision rules against the full DSS | `decision_engine`, `figures` |

## Repository Layout

```text
marketplace-policies-code/
├── configs/default.toml
├── images/
│   └── mermaid-diagram.png
├── notebooks/
│   ├── 0_setup.ipynb
│   ├── 1_eda_figures.ipynb
│   ├── 2_nuisance_model_diagnostics.ipynb
│   ├── 3_reserve_policy_tradeoff_and_daily_stability.ipynb
│   ├── 4_ope_estimator_diagnostics.ipynb
│   ├── 5_heterogeneity_and_downside_risk.ipynb
│   ├── 6_sensitivity_and_robustness.ipynb
│   ├── 7_external_season3_validation.ipynb
│   ├── 8_validation_design_and_launch_readiness.ipynb
│   └── 9_decision_rule_ablation.ipynb
├── src/
│   ├── auction_replay.py
│   ├── checks.py
│   ├── cli.py
│   ├── config.py
│   ├── data_access.py
│   ├── decision_engine.py
│   ├── external_validation.py
│   ├── figures.py
│   ├── guardrails.py
│   ├── nuisance_models.py
│   ├── ope.py
│   ├── panel_builder.py
│   ├── paper_tables.py
│   ├── policy_catalog.py
│   ├── progress.py
│   └── theory_sensitivity.py
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

## Installation

This repository assumes [`uv`](https://docs.astral.sh/uv/) is installed.

Install the runtime, development, and notebook dependencies:

```bash
uv sync --extra dev --extra notebooks
```

Start Jupyter from the repository root:

```bash
uv run jupyter lab notebooks
```

If you prefer to register the environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name marketplace-policies-code --display-name "Marketplace Policies Code"
```

## Python Dependencies

Core runtime packages:

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

`lightgbm` is used for nuisance-model prototypes. `scikit-learn` is used for calibration, preprocessing, and histogram-gradient-boosting diagnostics. The Python `graphviz` package also requires the Graphviz `dot` executable on the system path.

## Generated Files

Notebook execution writes generated files under `artifacts/`, including:

- processed Season 2 and Season 3 parquet panels
- metadata CSVs
- manuscript figure PNGs
- manuscript table CSVs
- claim-check reports

These files are local outputs and are ignored by Git. The repository commits the notebooks with outputs displayed, but not the generated files those notebooks write to disk.

Useful checks before pushing:

```bash
git status --short
git status --ignored --short
git check-ignore -v artifacts/figures/14_decision_rule_gate_matrix.png
git check-ignore -v data/ipinyou/archive.zip
```

The first command should show source files and notebooks you intend to commit. The ignored-status/check-ignore commands should confirm that local data and artifacts are not being tracked.

## Utility Commands

The primary workflow is the numbered notebooks. A small CLI remains for utility tasks after notebook artifacts already exist:

```bash
uv run marketplace-policies figures --source-root artifacts/workspace
uv run marketplace-policies tables --source-root artifacts/workspace
uv run marketplace-policies check --source-root artifacts/workspace
```

These commands do not run the full analysis. They only render figures, export selected tables, or validate headline claims from an existing generated workspace.

## Development

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

## GitHub Checklist

Before pushing, verify the following:

- `data/` is ignored.
- `artifacts/` is ignored.
- notebooks under `notebooks/` are tracked with outputs displayed.
- `images/mermaid-diagram.png` is tracked.
- `src/`, `tests/`, `configs/`, `README.md`, `pyproject.toml`, `.gitignore`, `LICENSE`, and notebooks are staged.
- no generated CSV, parquet, PNG figure output, model file, archive, or PDF is staged.

## License

This code repository is released under the MIT License. See `LICENSE` for details.
