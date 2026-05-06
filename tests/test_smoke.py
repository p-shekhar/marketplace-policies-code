from __future__ import annotations

from pathlib import Path

from marketplace_policies_code.config import ProjectPaths


def test_project_paths_resolve(tmp_path: Path) -> None:
    paths = ProjectPaths.from_cli(tmp_path, tmp_path / "artifacts")
    assert paths.source_root == tmp_path.resolve()
    assert paths.output_root == (tmp_path / "artifacts").resolve()
    assert paths.metadata_dir.name == "metadata"

