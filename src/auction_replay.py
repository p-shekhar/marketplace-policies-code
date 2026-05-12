from __future__ import annotations

import numpy as np
import pandas as pd

from data_access import BASELINE_POLICY_ID, VALUE_PROXY_CONVERSION_WEIGHT, Workspace  # noqa: F401
from guardrails import ReservePolicyGuardrails
from policy_catalog import ReservePolicyCatalog
from progress import ProgressLogger


class PolicyReplayAnalyzer:
    """Computes reserve/floor replay effects and guardrails."""

    replay_columns = [
        "event_date",
        "bid_price",
        "slot_floor_price",
        "bid_floor_gap",
        "filled",
        "pay_price",
        "clicked",
        "converted",
        "ad_exchange",
        "region",
        "advertiser_id",
    ]

    def __init__(
        self, workspace: Workspace, catalog: ReservePolicyCatalog, progress: ProgressLogger | None = None
    ) -> None:
        self.workspace = workspace
        self.catalog = catalog
        self.progress = progress or ProgressLogger(enabled=False)
        self.registry = catalog.registry()

    @staticmethod
    def _add_rates(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["fill_rate"] = result["retained_impressions"] / result["opportunities"].replace(0, np.nan)
        result["yield_per_opportunity"] = result["counterfactual_yield"] / result["opportunities"].replace(0, np.nan)
        result["yield_per_retained_impression"] = result["counterfactual_yield"] / result[
            "retained_impressions"
        ].replace(0, np.nan)
        result["click_rate_per_opportunity"] = result["retained_clicks"] / result["opportunities"].replace(0, np.nan)
        result["conversion_rate_per_opportunity"] = result["retained_conversions"] / result["opportunities"].replace(
            0, np.nan
        )
        result["value_proxy_per_opportunity"] = result["retained_value_proxy"] / result["opportunities"].replace(
            0, np.nan
        )
        result["retained_impression_share"] = result["retained_impressions"] / result[
            "observed_filled_impressions"
        ].replace(0, np.nan)
        return result

    def replay_metrics(self, frame: pd.DataFrame, policy_id: str) -> dict[str, object]:
        candidate_floor = self.catalog.floor(frame, policy_id)
        logged_floor = pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0).clip(lower=0).to_numpy(
            "float64"
        )
        bid_price = pd.to_numeric(frame["bid_price"], errors="coerce").fillna(-np.inf).to_numpy("float64")
        pay_price = pd.to_numeric(frame["pay_price"], errors="coerce").fillna(0).to_numpy("float64")
        filled = frame["filled"].to_numpy(bool)
        clicked = frame["clicked"].to_numpy("int64")
        converted = frame["converted"].to_numpy("int64")
        retained = filled & (bid_price >= candidate_floor)
        floor_changed = ~np.isclose(candidate_floor, logged_floor)
        cf_pay = np.where(retained, np.maximum(pay_price, candidate_floor), 0.0)
        cf_clicks = np.where(retained, clicked, 0)
        cf_conversions = np.where(retained, converted, 0)
        cf_value = cf_clicks + VALUE_PROXY_CONVERSION_WEIGHT * cf_conversions
        return {
            "policy_id": policy_id,
            "opportunities": len(frame),
            "observed_filled_impressions": int(filled.sum()),
            "retained_impressions": int(retained.sum()),
            "counterfactual_yield": float(cf_pay.sum()),
            "retained_clicks": int(cf_clicks.sum()),
            "retained_conversions": int(cf_conversions.sum()),
            "retained_value_proxy": int(cf_value.sum()),
            "mean_candidate_floor": float(np.mean(candidate_floor)),
            "p95_candidate_floor": float(np.percentile(candidate_floor, 95)),
            "floor_changed_opportunities": int(floor_changed.sum()),
            "floor_changed_share": float(np.mean(floor_changed)),
        }

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.progress.step(f"auction replay for {len(self.registry):,} reserve/floor policies")
        self.registry.to_csv(self.workspace.metadata_dir / "reserve_policy_registry.csv", index=False)
        self.registry.to_csv(self.workspace.table_dir / "06_reserve_policy_registry.csv", index=False)
        daily_rows = []
        shards = sorted(self.workspace.season2_panel_dir.glob("season2_panel_*.parquet"))
        for shard_index, shard in enumerate(shards, start=1):
            self.progress.log(f"Replaying policies on season-two shard {shard_index}/{len(shards)}: {shard.name}.")
            frame = pd.read_parquet(shard, columns=self.replay_columns)
            for policy_id in self.registry["policy_id"]:
                row = self.replay_metrics(frame, policy_id)
                row["event_date"] = str(frame["event_date"].iloc[0])
                daily_rows.append(row)
        daily = self._add_rates(pd.DataFrame(daily_rows))
        aggregate = daily.groupby("policy_id", as_index=False).agg(
            opportunities=("opportunities", "sum"),
            observed_filled_impressions=("observed_filled_impressions", "sum"),
            retained_impressions=("retained_impressions", "sum"),
            counterfactual_yield=("counterfactual_yield", "sum"),
            retained_clicks=("retained_clicks", "sum"),
            retained_conversions=("retained_conversions", "sum"),
            retained_value_proxy=("retained_value_proxy", "sum"),
            mean_candidate_floor=("mean_candidate_floor", "mean"),
            p95_candidate_floor=("p95_candidate_floor", "mean"),
            floor_changed_opportunities=("floor_changed_opportunities", "sum"),
        )
        aggregate["floor_changed_share"] = aggregate["floor_changed_opportunities"] / aggregate["opportunities"].replace(
            0, np.nan
        )
        effects = self._add_rates(aggregate)
        baseline = effects.query("policy_id == @BASELINE_POLICY_ID").iloc[0]
        for metric in [
            "fill_rate",
            "yield_per_opportunity",
            "click_rate_per_opportunity",
            "conversion_rate_per_opportunity",
            "value_proxy_per_opportunity",
        ]:
            effects[f"delta_{metric}_vs_baseline"] = effects[metric] - baseline[metric]
            effects[f"pct_delta_{metric}_vs_baseline"] = np.where(
                baseline[metric] != 0, effects[f"delta_{metric}_vs_baseline"] / baseline[metric], np.nan
            )
            base_map = daily.query("policy_id == @BASELINE_POLICY_ID").set_index("event_date")[metric].to_dict()
            daily[f"baseline_{metric}"] = daily["event_date"].map(base_map)
            daily[f"delta_{metric}_vs_daily_baseline"] = daily[metric] - daily[f"baseline_{metric}"]
            daily[f"pct_delta_{metric}_vs_daily_baseline"] = np.where(
                daily[f"baseline_{metric}"].ne(0),
                daily[f"delta_{metric}_vs_daily_baseline"] / daily[f"baseline_{metric}"],
                np.nan,
            )
        effects["click_retention"] = effects["retained_clicks"] / baseline["retained_clicks"]
        effects["conversion_retention"] = effects["retained_conversions"] / baseline["retained_conversions"]
        effects["value_proxy_retention"] = effects["retained_value_proxy"] / baseline["retained_value_proxy"]
        effects = effects.merge(
            self.registry.drop(columns=["policy_number", "policy_label"], errors="ignore"), on="policy_id", how="left"
        )
        effects = effects.sort_values("yield_per_opportunity", ascending=False)
        effects.to_csv(self.workspace.metadata_dir / "reserve_policy_effects.csv", index=False)
        effects.to_csv(self.workspace.table_dir / "06_reserve_policy_effects.csv", index=False)
        daily.to_csv(self.workspace.metadata_dir / "reserve_policy_daily_effects.csv", index=False)
        daily.to_csv(self.workspace.table_dir / "06_reserve_policy_daily_effects.csv", index=False)
        ReservePolicyGuardrails(self.workspace).evaluate(effects, daily)
        self.progress.done(f"auction replay; evaluated {len(effects):,} policies across {len(shards):,} shard(s)")
        return effects, daily
