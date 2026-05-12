from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import PaperConfig, ProjectPaths


@dataclass
class ArtifactRepository:
    """Access layer for generated paper artifacts."""

    paths: ProjectPaths

    def require_layout(self) -> None:
        required = [self.paths.metadata_dir, self.paths.tables_dir, self.paths.processed_data_dir]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required source folders: " + ", ".join(missing))

    def csv_path(self, name: str) -> Path:
        for folder in [self.paths.metadata_dir, self.paths.tables_dir]:
            path = folder / name
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find CSV artifact {name!r} in metadata/ or tables/")

    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.csv_path(name), **kwargs)

    def read_parquet(self, name: str, **kwargs) -> pd.DataFrame:
        path = self.paths.processed_data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Could not find parquet artifact {path}")
        return pd.read_parquet(path, **kwargs)

    def selected_figures(self) -> pd.DataFrame:
        return self.read_csv("final_figure_selection.csv")

    def selected_tables(self) -> pd.DataFrame:
        return self.read_csv("final_table_selection.csv")

    def copy_selected_tables(self, output_dir: Path) -> list[Path]:
        """Copy table artifacts referenced by final_table_selection.csv."""

        output_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for row in self.selected_tables().itertuples(index=False):
            relative_path = Path(str(row.relative_path))
            source = self.paths.source_root / relative_path
            if not source.exists():
                source = self.csv_path(str(row.filename))
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
        return copied

    def write_manifest(self, output_dir: Path) -> Path:
        """Write an inventory of generated files in output_dir."""

        rows = []
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "relative_path": str(path.relative_to(output_dir)),
                        "suffix": path.suffix,
                        "size_kb": round(path.stat().st_size / 1024, 2),
                    }
                )
        manifest = output_dir / "artifact_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return manifest


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    observed: object
    expected: object
    passed: bool
    tolerance: float | None = None


class ResultValidator:
    """Validate the headline numerical and decision claims in the paper."""

    def __init__(self, repository: ArtifactRepository, config: PaperConfig | None = None) -> None:
        self.repository = repository
        self.config = config or PaperConfig()

    @staticmethod
    def _close(observed: float, expected: float, tolerance: float = 5e-4) -> bool:
        return abs(float(observed) - float(expected)) <= tolerance

    def run(self) -> list[ClaimCheck]:
        final = self.repository.read_csv("final_policy_recommendation.csv").iloc[0]
        season3 = self.repository.read_csv("season3_priority_policy_validation.csv").iloc[0]
        ablation = self.repository.read_csv("decision_rule_ablation_summary.csv")
        scorecard = self.repository.read_csv("marketplace_scorecard.csv")
        full_dss = ablation.query("rule_id == 'full_dss'").iloc[0]
        simplified = ablation.query("rule_id != 'full_dss'")
        top_scorecard_policy = scorecard.sort_values("weighted_evidence_score", ascending=False).iloc[0].policy_id

        checks = [
            ClaimCheck(
                "priority_policy_id",
                final.policy_id,
                self.config.priority_policy_id,
                final.policy_id == self.config.priority_policy_id,
            ),
            ClaimCheck(
                "priority_policy_label",
                season3.policy_label,
                self.config.priority_policy_label,
                season3.policy_label == self.config.priority_policy_label,
            ),
            ClaimCheck(
                "priority_matches_scorecard_top",
                final.policy_id,
                top_scorecard_policy,
                final.policy_id == top_scorecard_policy,
            ),
            ClaimCheck(
                "season2_replay_lift_positive", final.replay_yield_lift, "> 0", float(final.replay_yield_lift) > 0
            ),
            ClaimCheck(
                "dr_lower_tail_lift_positive", final.p10_crossfit_dr_lift, "> 0", float(final.p10_crossfit_dr_lift) > 0
            ),
            ClaimCheck(
                "break_even_response_loss_in_unit_interval",
                final.break_even_market_response_loss_share,
                "[0, 1]",
                0 <= float(final.break_even_market_response_loss_share) <= 1,
            ),
            ClaimCheck(
                "season3_holdout_lift_positive",
                season3.season3_pct_yield_lift,
                "> 0",
                float(season3.season3_pct_yield_lift) > 0,
            ),
            ClaimCheck("season3_rank_top_three", int(season3.season3_rank), "<= 3", int(season3.season3_rank) <= 3),
            ClaimCheck(
                "simplified_rules_select_priority",
                bool(simplified["selected_policy_id"].eq(self.config.priority_policy_id).all()),
                True,
                bool(simplified["selected_policy_id"].eq(self.config.priority_policy_id).all()),
            ),
            ClaimCheck(
                "simplified_rules_overclaim_direct_launch",
                int(simplified["direct_launch_overclaim"].sum()),
                int(simplified.shape[0]),
                int(simplified["direct_launch_overclaim"].sum()) == int(simplified.shape[0]),
            ),
            ClaimCheck(
                "full_dss_recommends_validation",
                full_dss.recommended_action_under_rule,
                "validate_online",
                full_dss.recommended_action_under_rule == "validate_online",
            ),
        ]
        return checks

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.__dict__ for check in self.run()])

    def write_report(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        path = output_dir / "claim_checks.csv"
        frame.to_csv(path, index=False)
        markdown = output_dir / "claim_checks.md"
        markdown.write_text(self._markdown_report(frame), encoding="utf-8")
        return path

    @staticmethod
    def _markdown_report(frame: pd.DataFrame) -> str:
        lines = ["# Claim Checks", ""]
        for row in frame.itertuples(index=False):
            mark = "PASS" if row.passed else "FAIL"
            lines.append(f"- **{mark}** `{row.name}`: observed `{row.observed}`, expected `{row.expected}`")
        lines.append("")
        return "\n".join(lines)

    def assert_all_pass(self) -> None:
        failed = [check for check in self.run() if not check.passed]
        if failed:
            names = ", ".join(check.name for check in failed)
            raise AssertionError(f"Claim checks failed: {names}")
