from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketplace_policies_code.config import ProjectPaths


@dataclass
class ArtifactRepository:
    """Typed-ish access layer for paper artifacts.

    The pipeline generates many CSV files. This class centralizes lookup rules so
    figure, table, and validation code do not hard-code folder details.
    """

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
