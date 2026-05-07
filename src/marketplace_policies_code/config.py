from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperConfig:
    """Paper-level names that should stay stable across the reproduction."""

    priority_policy_id: str = "hybrid_q75_if_gap_100"
    priority_policy_label: str = "Q75 Margin-Gated Floor"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for source artifacts and generated outputs."""

    source_root: Path
    output_root: Path

    @classmethod
    def from_cli(cls, source_root: str | Path, output_root: str | Path) -> ProjectPaths:
        return cls(
            source_root=Path(source_root).expanduser().resolve(), output_root=Path(output_root).expanduser().resolve()
        )

    @property
    def metadata_dir(self) -> Path:
        return self.source_root / "metadata"

    @property
    def tables_dir(self) -> Path:
        return self.source_root / "tables"

    @property
    def source_figure_dir(self) -> Path:
        return self.source_root / "figures"

    @property
    def data_dir(self) -> Path:
        return self.source_root / "data"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifact_dir(self) -> Path:
        return self.output_root

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def exported_table_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def report_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def bundle_dir(self) -> Path:
        return self.output_root / "bundle"

    def ensure_output_dirs(self) -> None:
        for folder in [self.output_root, self.figure_dir, self.exported_table_dir, self.report_dir, self.bundle_dir]:
            folder.mkdir(parents=True, exist_ok=True)
