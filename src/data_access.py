from __future__ import annotations

import bz2
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import RawPipelineConfig
from progress import ProgressLogger

BID_COLUMNS = [
    "bid_id",
    "timestamp",
    "user_id",
    "user_agent",
    "ip",
    "region",
    "city",
    "ad_exchange",
    "domain",
    "url",
    "url_id",
    "slot_id",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_floor_price",
    "creative_id",
    "bid_price",
    "advertiser_id",
    "user_tags",
]

EVENT_COLUMNS = [
    "bid_id",
    "timestamp",
    "log_type",
    "user_id",
    "user_agent",
    "ip",
    "region",
    "city",
    "ad_exchange",
    "domain",
    "url",
    "url_id",
    "slot_id",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_floor_price",
    "creative_id",
    "bid_price",
    "pay_price",
    "key_page",
    "advertiser_id",
    "user_tags",
]

VALUE_PROXY_CONVERSION_WEIGHT = 10.0
BASELINE_POLICY_ID = "logged_floor_status_quo"
PRIORITY_POLICY_ID = "hybrid_q75_if_gap_100"
PRIORITY_POLICY_LABEL = "Q75 Margin-Gated Floor"


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def table_dir(self) -> Path:
        return self.root / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.root / "figures"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def season2_panel_dir(self) -> Path:
        return self.processed_dir / "season2_bid_opportunity_panel"

    @property
    def season3_panel_dir(self) -> Path:
        return self.processed_dir / "season3_bid_opportunity_panel"

    def prepare(self, clean: bool = False) -> None:
        if clean and self.root.exists():
            shutil.rmtree(self.root)
        for folder in [
            self.metadata_dir,
            self.table_dir,
            self.figure_dir,
            self.processed_dir,
            self.season2_panel_dir,
            self.season3_panel_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)




class IpinYouArchive:
    """Reader for the original compressed iPinYou contest archive."""

    pattern = re.compile(r"(training(?:1st|2nd|3rd))/(bid|imp|clk|conv)\.(\d{8})\.txt\.bz2$")

    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path.expanduser().resolve()
        if not self.archive_path.exists():
            raise FileNotFoundError(
                f"Could not find iPinYou archive at {self.archive_path}. "
                "Place the original archive at data/ipinyou/archive.zip or pass --data-root."
            )

    def inventory(self) -> pd.DataFrame:
        rows = []
        with zipfile.ZipFile(self.archive_path) as zf:
            for member in zf.namelist():
                match = self.pattern.search(member)
                if match:
                    season, file_kind, date_label = match.groups()
                    rows.append(
                        {
                            "season": season,
                            "file_kind": file_kind,
                            "date_label": date_label,
                            "member": member,
                            "compressed_size": zf.getinfo(member).compress_size,
                            "file_size": zf.getinfo(member).file_size,
                        }
                    )
        return pd.DataFrame(rows).sort_values(["season", "date_label", "file_kind"]).reset_index(drop=True)

    def read_tsv(
        self, member: str, columns: list[str], usecols: list[str] | None = None, nrows: int | None = None
    ) -> pd.DataFrame:
        with zipfile.ZipFile(self.archive_path) as zf:
            with zf.open(member) as compressed:
                with bz2.open(compressed, "rt", encoding="utf-8", errors="replace") as handle:
                    return pd.read_csv(
                        handle,
                        sep="\t",
                        names=columns,
                        usecols=usecols,
                        nrows=nrows,
                        dtype=str,
                        low_memory=False,
                    )




class ArtifactBuilder:
    """Shared CSV artifact I/O for DSS evidence-building components."""

    def __init__(
        self, workspace: Workspace, config: RawPipelineConfig, catalog=None, progress: ProgressLogger | None = None
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.catalog = catalog
        self.progress = progress or ProgressLogger(enabled=False)
        self.rng = np.random.default_rng(config.random_seed)

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.workspace.metadata_dir / name)

    def write(self, frame: pd.DataFrame, name: str, table: bool = False) -> pd.DataFrame:
        frame.to_csv(self.workspace.metadata_dir / name, index=False)
        if table:
            frame.to_csv(self.workspace.table_dir / name, index=False)
        return frame
