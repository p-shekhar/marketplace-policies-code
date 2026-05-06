from __future__ import annotations

import argparse
from pathlib import Path

from marketplace_policies_code.config import PaperConfig, ProjectPaths
from marketplace_policies_code.figures import FigureRenderer
from marketplace_policies_code.pipeline import PaperReproductionPipeline
from marketplace_policies_code.raw_pipeline import RawPipelineConfig, RawToPaperPipeline
from marketplace_policies_code.repository import ArtifactRepository
from marketplace_policies_code.results import ResultValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce ads marketplace policy paper artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("check", "Validate headline paper claims."),
        ("figures", "Regenerate publication figures."),
        ("tables", "Export selected paper tables."),
        ("reproduce", "Run the raw-data-to-paper reproduction."),
        ("build-artifacts", "Build paper artifacts from the original local iPinYou archive."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--source-root",
            default="artifacts/workspace",
            help="Folder containing generated metadata/, tables/, data/processed/, and optional overleaf/.",
        )
        subparser.add_argument("--output-root", default="artifacts", help="Folder where generated artifacts should be written.")
        subparser.add_argument("--data-root", default="data", help="Local folder containing ipinyou/archive.zip.")
        subparser.add_argument("--full", action="store_true", help="Use all available season-two and season-three rows.")
        subparser.add_argument("--quick", action="store_true", help="Use a bounded smoke-test subset of the raw data.")
        subparser.add_argument("--skip-analysis", action="store_true", help="Skip raw-data analysis and reuse --source-root artifacts.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root)
    if args.command in {"reproduce", "build-artifacts"} and not args.skip_analysis:
        raw_config = RawPipelineConfig(
            data_root=Path(args.data_root),
            workspace_root=source_root,
            full_run=bool(args.full and not args.quick),
            clean_workspace=True,
        )
        built_root = RawToPaperPipeline(raw_config).run()
        source_root = built_root
        if args.command == "build-artifacts":
            print(f"Built raw-data artifacts at {built_root}")
            return

    paths = ProjectPaths.from_cli(source_root, args.output_root)
    config = PaperConfig()
    repository = ArtifactRepository(paths)

    if args.command == "check":
        validator = ResultValidator(repository, config)
        validator.assert_all_pass()
        paths.ensure_output_dirs()
        validator.write_report(paths.report_dir)
        print(f"Claim checks passed. Report written to {paths.report_dir}")
        return

    if args.command == "figures":
        paths.ensure_output_dirs()
        repository.require_layout()
        figures = FigureRenderer(repository, paths.figure_dir, config).render_all()
        print(f"Rendered {len(figures)} figures to {paths.figure_dir}")
        return

    if args.command == "tables":
        paths.ensure_output_dirs()
        tables = repository.copy_selected_tables(paths.exported_table_dir)
        print(f"Exported {len(tables)} selected tables to {paths.exported_table_dir}")
        return

    pipeline = PaperReproductionPipeline(paths, config)
    result = pipeline.run()
    print(f"Rendered figures: {len(result.figures)}")
    print(f"Exported tables: {len(result.tables)}")
    print(f"Reports: {len(result.reports)}")
    print(f"Bundle files: {len(result.bundle_files)}")
    print(f"Manifest: {result.manifest}")


if __name__ == "__main__":
    main()
