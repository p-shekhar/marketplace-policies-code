from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import RawPipelineConfig
from data_access import BID_COLUMNS, EVENT_COLUMNS, VALUE_PROXY_CONVERSION_WEIGHT, IpinYouArchive, Workspace
from progress import ProgressLogger


class OpportunityPanelBuilder:
    """Builds bid-opportunity panels from bid, impression, click, and conversion logs."""

    numeric_columns = [
        "region",
        "city",
        "ad_exchange",
        "slot_width",
        "slot_height",
        "slot_floor_price",
        "bid_price",
        "advertiser_id",
    ]
    categorical_columns = ["region", "city", "ad_exchange", "advertiser_id", "slot_visibility", "slot_format"]

    def __init__(
        self,
        archive: IpinYouArchive,
        workspace: Workspace,
        config: RawPipelineConfig,
        progress: ProgressLogger | None = None,
    ) -> None:
        self.archive = archive
        self.workspace = workspace
        self.config = config
        self.progress = progress or ProgressLogger(enabled=False)
        self.inventory = archive.inventory()

    def write_inventory(self) -> pd.DataFrame:
        self.inventory.to_csv(self.workspace.metadata_dir / "ipinyou_archive_inventory.csv", index=False)
        self.progress.log(f"Archived file inventory found {len(self.inventory):,} members.")
        return self.inventory

    def _file_matrix(self, season: str) -> pd.DataFrame:
        frame = self.inventory.query("season == @season").copy()
        return (
            frame.pivot_table(index="date_label", columns="file_kind", values="member", aggfunc="first")
            .reset_index()
            .sort_values("date_label")
        )

    def _members_for(self, season: str, date_label: str) -> dict[str, str]:
        matrix = self._file_matrix(season)
        row = matrix.loc[matrix["date_label"].astype(str).eq(str(date_label))]
        if row.empty:
            raise FileNotFoundError(f"No {season} archive members for {date_label}")
        return {
            kind: row.iloc[0][kind]
            for kind in ["bid", "imp", "clk", "conv"]
            if kind in row.columns and pd.notna(row.iloc[0][kind])
        }

    @staticmethod
    def _time_features(frame: pd.DataFrame) -> pd.DataFrame:
        timestamp_prefix = frame["timestamp"].astype(str).str.slice(0, 14)
        frame["event_time"] = pd.to_datetime(timestamp_prefix, format="%Y%m%d%H%M%S", errors="coerce")
        frame["event_date"] = frame["event_time"].dt.date.astype(str)
        frame["hour"] = frame["event_time"].dt.hour.fillna(-1).astype(int)
        frame["day_of_week"] = frame["event_time"].dt.dayofweek.fillna(-1).astype(int)
        return frame

    def engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = self._time_features(frame)
        frame["slot_area"] = frame["slot_width"] * frame["slot_height"]
        frame["bid_floor_gap"] = frame["bid_price"] - frame["slot_floor_price"]
        frame["floor_to_bid_ratio"] = np.where(
            frame["bid_price"].gt(0), frame["slot_floor_price"] / frame["bid_price"], 0.0
        )
        frame["log_bid_price"] = np.log1p(frame["bid_price"].clip(lower=0))
        frame["log_slot_floor_price"] = np.log1p(frame["slot_floor_price"].clip(lower=0))
        clean_tags = frame["user_tags"].fillna("").astype(str)
        clean_tags = clean_tags.where(~clean_tags.isin(["null", "nan", "None", "0"]), "")
        frame["user_tag_count"] = clean_tags.map(lambda value: 0 if value == "" else len(value.split(",")))
        frame["has_user_tags"] = frame["user_tag_count"].gt(0).astype(int)
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("string").fillna("missing")
        frame["support_cluster"] = (
            frame["ad_exchange"].astype(str)
            + "|"
            + frame["region"].astype(str)
            + "|"
            + frame["slot_visibility"].astype(str)
            + "|"
            + frame["slot_format"].astype(str)
        )
        return frame

    def build_day_panel(
        self, season: str, date_label: str, nrows: int | None = None
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        start = time.time()
        mode = "all rows" if nrows is None else f"up to {nrows:,} bid rows"
        self.progress.log(f"Building {season} panel for date {date_label} ({mode}).")
        members = self._members_for(season, date_label)
        bid_df = self.archive.read_tsv(members["bid"], BID_COLUMNS, nrows=nrows).drop_duplicates("bid_id", keep="first")
        imp_labels = self.archive.read_tsv(
            members["imp"], EVENT_COLUMNS, usecols=["bid_id", "pay_price"]
        ).drop_duplicates("bid_id", keep="last")
        imp_labels["pay_price"] = pd.to_numeric(imp_labels["pay_price"], errors="coerce")
        click_ids = (
            set(self.archive.read_tsv(members["clk"], EVENT_COLUMNS, usecols=["bid_id"])["bid_id"].dropna().astype(str))
            if "clk" in members
            else set()
        )
        conv_ids = (
            set(
                self.archive.read_tsv(members["conv"], EVENT_COLUMNS, usecols=["bid_id"])["bid_id"].dropna().astype(str)
            )
            if "conv" in members
            else set()
        )

        frame = bid_df.merge(imp_labels[["bid_id", "pay_price"]], on="bid_id", how="left", validate="one_to_one")
        frame["season"] = season
        frame["date_label"] = str(date_label)
        frame["filled"] = frame["pay_price"].notna().astype(int)
        frame["clicked"] = (frame["filled"].eq(1) & frame["bid_id"].isin(click_ids)).astype(int)
        frame["converted"] = (frame["filled"].eq(1) & frame["bid_id"].isin(conv_ids)).astype(int)
        frame["value_proxy_lambda_10"] = frame["clicked"] + VALUE_PROXY_CONVERSION_WEIGHT * frame["converted"]
        frame["source_member_bid"] = members["bid"]
        frame["source_member_imp"] = members["imp"]
        frame = self.engineer_features(frame)
        manifest = {
            "date_label": str(date_label),
            "season": season,
            "run_mode": "full" if nrows is None else "quick",
            "bid_member": members["bid"],
            "imp_member": members["imp"],
            "bid_rows_requested": "all" if nrows is None else nrows,
            "bid_rows_materialized": len(frame),
            "unique_imp_bid_ids": imp_labels["bid_id"].nunique(),
            "unique_click_bid_ids": len(click_ids),
            "unique_conversion_bid_ids": len(conv_ids),
            "filled_rows_in_panel": int(frame["filled"].sum()),
            "clicks_in_panel": int(frame["clicked"].sum()),
            "conversions_in_panel": int(frame["converted"].sum()),
            "fill_rate_in_panel": float(frame["filled"].mean()),
            "elapsed_seconds": round(time.time() - start, 2),
        }
        self.progress.log(
            f"Built {season} date {date_label}: {len(frame):,} opportunities, "
            f"{int(frame['filled'].sum()):,} fills, {int(frame['clicked'].sum()):,} clicks "
            f"in {manifest['elapsed_seconds']}s."
        )
        return frame, manifest

    def build_season2(self) -> pd.DataFrame:
        self.progress.step("season-two bid-opportunity panel construction")
        matrix = self._file_matrix("training2nd")
        dates = matrix.dropna(subset=["bid", "imp"])["date_label"].astype(str).tolist()
        if not self.config.full_run:
            dates = dates[: self.config.quick_season2_dates]
        matrix["used_in_run"] = matrix["date_label"].astype(str).isin(dates)
        matrix.to_csv(self.workspace.metadata_dir / "season2_file_matrix.csv", index=False)

        nrows = None if self.config.full_run else self.config.season2_rows_per_day_quick
        frames: list[pd.DataFrame] = []
        manifests = []
        for date in dates:
            frame, manifest = self.build_day_panel("training2nd", date, nrows=nrows)
            frames.append(frame)
            day_path = self.workspace.season2_panel_dir / f"season2_panel_{date}.parquet"
            manifest["panel_artifact"] = str(day_path.relative_to(self.workspace.root))
            manifests.append(manifest)

        panel = pd.concat(frames, ignore_index=True).sort_values(["event_time", "bid_id"]).reset_index(drop=True)
        row_position = np.arange(len(panel)) / max(len(panel) - 1, 1)
        panel["time_split"] = np.select(
            [row_position < 0.60, row_position < 0.80], ["train", "validation"], default="test"
        )
        for date in dates:
            panel.loc[panel["date_label"].astype(str).eq(str(date))].to_parquet(
                self.workspace.season2_panel_dir / f"season2_panel_{date}.parquet", index=False
            )
        panel.to_parquet(
            self.workspace.processed_dir / "season2_development_or_full_panel_current_scope.parquet", index=False
        )

        manifest_df = pd.DataFrame(manifests)
        manifest_df.to_csv(self.workspace.metadata_dir / "season2_panel_manifest.csv", index=False)
        manifest_df.to_csv(self.workspace.table_dir / "05_season2_panel_manifest.csv", index=False)

        density = (
            panel.groupby("event_date", observed=True)
            .agg(
                bid_opportunities=("bid_id", "size"),
                filled=("filled", "sum"),
                clicks=("clicked", "sum"),
                conversions=("converted", "sum"),
                fill_rate=("filled", "mean"),
                click_rate_per_opportunity=("clicked", "mean"),
                conversion_rate_per_opportunity=("converted", "mean"),
            )
            .reset_index()
        )
        density.to_csv(self.workspace.metadata_dir / "season2_outcome_density.csv", index=False)
        density.to_csv(self.workspace.table_dir / "05_season2_outcome_density.csv", index=False)

        sample_n = min(600_000, len(panel))
        sample = panel.sample(n=sample_n, random_state=self.config.random_seed) if len(panel) > sample_n else panel
        sample[
            [
                "bid_id",
                "slot_floor_price",
                "bid_price",
                "bid_floor_gap",
                "pay_price",
                "filled",
                "clicked",
                "converted",
                "event_date",
                "hour",
                "day_of_week",
                "ad_exchange",
                "region",
                "advertiser_id",
                "slot_width",
                "slot_height",
                "slot_area",
                "slot_visibility",
                "slot_format",
                "floor_to_bid_ratio",
                "user_tag_count",
                "has_user_tags",
                "support_cluster",
                "time_split",
                "value_proxy_lambda_10",
            ]
        ].to_parquet(self.workspace.processed_dir / "ipinyou_nuisance_prototype_opportunity_panel.parquet", index=False)
        self.progress.done(
            f"season-two panel construction; {len(panel):,} opportunities across {len(dates):,} date(s)"
        )
        return panel

    def build_season3(self) -> pd.DataFrame:
        self.progress.step("season-three holdout panel construction")
        matrix = self._file_matrix("training3rd")
        dates = matrix.dropna(subset=["bid", "imp"])["date_label"].astype(str).tolist()
        if not self.config.full_run:
            dates = dates[: self.config.quick_season3_dates]
        matrix["used_in_run"] = matrix["date_label"].astype(str).isin(dates)
        matrix.to_csv(self.workspace.metadata_dir / "season3_file_matrix.csv", index=False)

        nrows = None if self.config.full_run else self.config.season3_rows_per_day_quick
        frames: list[pd.DataFrame] = []
        manifests = []
        for date in dates:
            frame, manifest = self.build_day_panel("training3rd", date, nrows=nrows)
            frame["time_split"] = "holdout"
            frames.append(frame)
            day_path = self.workspace.season3_panel_dir / f"season3_panel_{date}.parquet"
            manifest["panel_artifact"] = str(day_path.relative_to(self.workspace.root))
            manifests.append(manifest)

        panel = pd.concat(frames, ignore_index=True).sort_values(["event_time", "bid_id"]).reset_index(drop=True)
        panel["time_split"] = "holdout"
        for date in dates:
            panel.loc[panel["date_label"].astype(str).eq(str(date))].to_parquet(
                self.workspace.season3_panel_dir / f"season3_panel_{date}.parquet", index=False
            )
        panel.to_parquet(self.workspace.processed_dir / "season3_holdout_panel_current_scope.parquet", index=False)

        manifest_df = pd.DataFrame(manifests)
        manifest_df.to_csv(self.workspace.metadata_dir / "season3_panel_manifest.csv", index=False)
        manifest_df.to_csv(self.workspace.table_dir / "13_season3_panel_manifest.csv", index=False)

        density = (
            panel.groupby("event_date", observed=True)
            .agg(
                bid_opportunities=("bid_id", "size"),
                filled=("filled", "sum"),
                clicks=("clicked", "sum"),
                conversions=("converted", "sum"),
                fill_rate=("filled", "mean"),
                click_rate_per_opportunity=("clicked", "mean"),
                conversion_rate_per_opportunity=("converted", "mean"),
            )
            .reset_index()
        )
        density.to_csv(self.workspace.metadata_dir / "season3_outcome_density.csv", index=False)
        density.to_csv(self.workspace.table_dir / "13_season3_outcome_density.csv", index=False)

        self.progress.done(
            f"season-three holdout panel construction; {len(panel):,} opportunities across {len(dates):,} date(s)"
        )
        return panel

