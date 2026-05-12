from __future__ import annotations

import argparse
from pathlib import Path

from config import PaperConfig, ProjectPaths
from progress import ProgressLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Utility commands for notebook-generated ads marketplace policy artifacts. "
            "Run the numbered notebooks for the full analysis."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("check", "Validate headline paper claims."),
        ("figures", "Regenerate publication figures."),
        ("tables", "Export selected paper tables."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--source-root",
            default="artifacts/workspace",
            help="Folder containing generated metadata/, tables/, and data/processed/.",
        )
        subparser.add_argument(
            "--output-root", default="artifacts", help="Folder where generated artifacts should be written."
        )
        subparser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    progress = ProgressLogger(enabled=not args.quiet)
    progress.log(f"Command `{args.command}` started.")
    source_root = Path(args.source_root)
    paths = ProjectPaths.from_cli(source_root, args.output_root)
    config = PaperConfig()

    from checks import ArtifactRepository, ResultValidator

    repository = ArtifactRepository(paths)

    if args.command == "check":
        progress.step("headline claim validation")
        validator = ResultValidator(repository, config)
        validator.assert_all_pass()
        paths.ensure_output_dirs()
        validator.write_report(paths.report_dir)
        progress.done(f"headline claim validation; report written to {paths.report_dir}")
        print(f"Claim checks passed. Report written to {paths.report_dir}")
        return

    if args.command == "figures":
        from figures import FigureRenderer

        progress.step("figure rendering")
        paths.ensure_output_dirs()
        repository.require_layout()
        figures = FigureRenderer(repository, paths.figure_dir, config, progress=progress).render_all()
        progress.done(f"figure rendering; wrote {len(figures)} figures to {paths.figure_dir}")
        print(f"Rendered {len(figures)} figures to {paths.figure_dir}")
        return

    if args.command == "tables":
        progress.step("selected table export")
        paths.ensure_output_dirs()
        tables = repository.copy_selected_tables(paths.exported_table_dir)
        progress.done(f"selected table export; wrote {len(tables)} tables to {paths.exported_table_dir}")
        print(f"Exported {len(tables)} selected tables to {paths.exported_table_dir}")
        return


if __name__ == "__main__":
    main()
