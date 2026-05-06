from __future__ import annotations

import argparse
from pathlib import Path

from marketplace_policies_code.config import PaperConfig, ProjectPaths
from marketplace_policies_code.figures import FigureRenderer
from marketplace_policies_code.pipeline import PaperReproductionPipeline
from marketplace_policies_code.repository import ArtifactRepository
from marketplace_policies_code.results import ResultValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce ads marketplace policy paper artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("check", "Validate headline paper claims."),
        ("figures", "Regenerate publication figures."),
        ("tables", "Export selected paper tables."),
        ("reproduce", "Run the full artifact-level reproduction."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--source-root",
            default="..",
            help="Folder containing metadata/, tables/, data/, figures/, and optional overleaf/.",
        )
        subparser.add_argument("--output-root", default="artifacts", help="Folder where generated artifacts should be written.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_cli(args.source_root, args.output_root)
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
