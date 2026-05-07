from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import PaperConfig, ProjectPaths
from figures import FigureRenderer
from progress import ProgressLogger
from repository import ArtifactRepository
from results import ResultValidator


@dataclass
class PipelineResult:
    """Summary of a reproduction run."""

    figures: list[Path]
    tables: list[Path]
    reports: list[Path]
    bundle_files: list[Path]
    manifest: Path


class PaperReproductionPipeline:
    """Publication-output pipeline that consumes generated analysis artifacts."""

    def __init__(
        self, paths: ProjectPaths, config: PaperConfig | None = None, progress: ProgressLogger | None = None
    ) -> None:
        self.paths = paths
        self.config = config or PaperConfig()
        self.progress = progress or ProgressLogger(enabled=False)
        self.repository = ArtifactRepository(paths)

    def validate_inputs(self) -> None:
        self.progress.step("checking generated artifact layout")
        self.repository.require_layout()
        required_csv = [
            "final_policy_recommendation.csv",
            "season3_priority_policy_validation.csv",
            "decision_rule_ablation_summary.csv",
            "final_figure_selection.csv",
            "final_table_selection.csv",
        ]
        for name in required_csv:
            self.repository.csv_path(name)
        self.progress.done(f"checking generated artifact layout; found {len(required_csv)} required CSV files")

    def render_figures(self) -> list[Path]:
        renderer = FigureRenderer(self.repository, self.paths.figure_dir, self.config, progress=self.progress)
        figures = renderer.render_all()
        self.progress.done(f"rendering figures; wrote {len(figures)} PNG files")
        return figures

    def export_tables(self) -> list[Path]:
        self.progress.step("exporting selected paper tables")
        copied = self.repository.copy_selected_tables(self.paths.exported_table_dir)
        index_path = self.paths.exported_table_dir / "table_index.csv"
        self.repository.selected_tables().to_csv(index_path, index=False)
        self.progress.done(f"exporting selected paper tables; wrote {len(copied) + 1} CSV files")
        return [*copied, index_path]

    def validate_claims(self) -> list[Path]:
        self.progress.step("validating headline decision claims")
        validator = ResultValidator(self.repository, self.config)
        validator.assert_all_pass()
        csv_path = validator.write_report(self.paths.report_dir)
        self.progress.done("validating headline decision claims")
        return [csv_path, self.paths.report_dir / "claim_checks.md"]

    def write_result_summary(self) -> Path:
        self.progress.step("writing result summary")
        final = self.repository.read_csv("final_policy_recommendation.csv").iloc[0]
        season3 = self.repository.read_csv("season3_priority_policy_validation.csv").iloc[0]
        ablation = self.repository.read_csv("decision_rule_ablation_summary.csv")
        simplified = ablation.query("rule_id != 'full_dss'")
        summary = pd.DataFrame(
            [
                {
                    "priority_policy_id": final.policy_id,
                    "priority_policy_label": self.config.priority_policy_label,
                    "season2_replay_lift": final.replay_yield_lift,
                    "dr_lower_tail_lift": final.p10_crossfit_dr_lift,
                    "season3_holdout_lift": season3.season3_pct_yield_lift,
                    "simplified_rules_select_priority": bool(
                        simplified["selected_policy_id"].eq(final.policy_id).all()
                    ),
                    "simplified_rules_overclaim_direct_launch": int(simplified["direct_launch_overclaim"].sum()),
                    "full_dss_action": ablation.query("rule_id == 'full_dss'").iloc[0].recommended_action_under_rule,
                    "recommended_action": final.recommended_action,
                }
            ]
        )
        path = self.paths.report_dir / "result_summary.csv"
        summary.to_csv(path, index=False)
        markdown = self.paths.report_dir / "result_summary.md"
        row = summary.iloc[0]
        markdown.write_text(
            "\n".join(
                [
                    "# Result Summary",
                    "",
                    f"- Priority policy: `{row.priority_policy_id}` ({row.priority_policy_label})",
                    f"- Season-two replay lift: {row.season2_replay_lift:.1%}",
                    f"- Conservative DR lower-tail lift: {row.dr_lower_tail_lift:.1%}",
                    f"- Season-three holdout lift: {row.season3_holdout_lift:.1%}",
                    f"- Simplified rules select priority policy: {row.simplified_rules_select_priority}",
                    f"- Simplified direct-launch overclaims: {row.simplified_rules_overclaim_direct_launch}",
                    f"- Full DSS action: `{row.full_dss_action}`",
                    f"- Recommended action: {row.recommended_action}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.progress.done(f"writing result summary to {path}")
        return path

    def make_bundle(self) -> list[Path]:
        """Create an archival bundle of regenerated outputs."""

        self.progress.step("creating archival bundle of regenerated outputs")
        bundle = self.paths.bundle_dir
        bundle.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []

        regenerated_figures = bundle / "figures"
        regenerated_figures.mkdir(exist_ok=True)
        for figure in self.paths.figure_dir.glob("*.png"):
            destination = regenerated_figures / figure.name
            shutil.copy2(figure, destination)
            copied.append(destination)

        regenerated_tables = bundle / "tables"
        regenerated_tables.mkdir(exist_ok=True)
        for table in self.paths.exported_table_dir.glob("*.csv"):
            destination = regenerated_tables / table.name
            shutil.copy2(table, destination)
            copied.append(destination)

        for report in self.paths.report_dir.glob("*"):
            if report.is_file():
                destination = bundle / report.name
                shutil.copy2(report, destination)
                copied.append(destination)
        self.progress.done(f"creating archival bundle; copied {len(copied)} files")
        return copied

    def run(
        self,
        *,
        render_figures: bool = True,
        export_tables: bool = True,
        validate_claims: bool = True,
        make_bundle: bool = True,
    ) -> PipelineResult:
        self.progress.step("publication-output reproduction pipeline")
        self.paths.ensure_output_dirs()
        self.validate_inputs()
        figures = self.render_figures() if render_figures else []
        tables = self.export_tables() if export_tables else []
        reports = []
        if validate_claims:
            reports.extend(self.validate_claims())
        reports.append(self.write_result_summary())
        reports.append(self.paths.report_dir / "result_summary.md")
        bundle_files = self.make_bundle() if make_bundle else []
        self.progress.step("writing artifact manifest")
        manifest = self.repository.write_manifest(self.paths.output_root)
        self.progress.done(f"writing artifact manifest to {manifest}")
        self.progress.done("publication-output reproduction pipeline")
        return PipelineResult(
            figures=figures, tables=tables, reports=reports, bundle_files=bundle_files, manifest=manifest
        )
