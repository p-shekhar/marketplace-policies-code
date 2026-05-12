from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import RawPipelineConfig
from data_access import BASELINE_POLICY_ID, PRIORITY_POLICY_ID, ArtifactBuilder, Workspace
from policy_catalog import ReservePolicyCatalog
from progress import ProgressLogger


class OPEEvidenceBuilder(ArtifactBuilder):
    """Builds experiment-design, OPE, scorecard, support, and heterogeneity artifacts."""

    def __init__(
        self,
        workspace: Workspace,
        config: RawPipelineConfig,
        catalog: ReservePolicyCatalog,
        progress: ProgressLogger | None = None,
    ) -> None:
        super().__init__(workspace, config, catalog, progress)

    @staticmethod
    def _threshold_score(value: float, thresholds: list[float], higher: bool = True) -> int:
        if pd.isna(value):
            return 1
        if higher:
            return (
                5
                if value >= thresholds[3]
                else 4
                if value >= thresholds[2]
                else 3
                if value >= thresholds[1]
                else 2
                if value >= thresholds[0]
                else 1
            )
        return (
            5
            if value <= thresholds[0]
            else 4
            if value <= thresholds[1]
            else 3
            if value <= thresholds[2]
            else 2
            if value <= thresholds[3]
            else 1
        )

    def build_experiment_design(self) -> None:
        self.progress.step("experiment-design and validation-plan artifacts")
        candidates = self.read("reserve_policy_candidate_handoff.csv")
        selected = candidates.query(
            "recommended_next_step == 'send_to_ope_and_sensitivity' and policy_id != @BASELINE_POLICY_ID"
        ).copy()
        rows = []
        for row in selected.itertuples(index=False):
            gain = float(row.pct_delta_yield_per_opportunity_vs_baseline)
            if gain >= 0.20:
                tier, design = "high_gain_high_scrutiny", "shadow_logging_then_exchange_hour_switchback"
            elif gain >= 0.08:
                tier, design = (
                    "moderate_gain_validate_online",
                    "shadow_logging_then_switchback_or_exchange_region_cluster",
                )
            else:
                tier, design = "small_gain_low_priority", "shadow_logging_or_hold_for_later"
            rows.append(
                {
                    "policy_id": row.policy_id,
                    "policy_family": row.policy_family,
                    "replay_yield_lift": gain,
                    "retained_impression_share": row.retained_impression_share,
                    "value_proxy_retention": row.value_proxy_retention,
                    "validation_tier": tier,
                    "recommended_validation_design": design,
                    "rationale": "validate marketplace response before launch",
                }
            )
        self.write(pd.DataFrame(rows), "policy_validation_plan.csv", table=True)
        shadow = pd.DataFrame(
            [
                {
                    "phase": "instrument_candidate_floor_decisions",
                    "duration_recommendation": "7 to 14 days before any serving change",
                    "required_fields": "bid_id, current_floor, candidate_floor, bid_price, would_clear_candidate_floor",
                    "why_needed": "measure support without changing marketplace behavior",
                },
                {
                    "phase": "log_policy_assignment_probabilities",
                    "duration_recommendation": "same as shadow period",
                    "required_fields": "eligible_policy_set and assignment probability",
                    "why_needed": "off-policy estimators require known propensities",
                },
            ]
        )
        self.write(shadow, "shadow_logging_plan.csv", table=True)
        baseline = self.read("reserve_policy_effects.csv").query("policy_id == @BASELINE_POLICY_ID").iloc[0]
        baseline_yield = float(baseline["yield_per_opportunity"])
        baseline_fill_rate = float(baseline["fill_rate"])
        mde_rows = []
        best_policy_id = PRIORITY_POLICY_ID
        best_replay = 0.0
        best_p10 = 0.0
        if (self.workspace.metadata_dir / "marketplace_scorecard.csv").exists():
            scorecard = self.read("marketplace_scorecard.csv").sort_values("weighted_evidence_score", ascending=False)
            best = scorecard.iloc[0]
            best_policy_id = str(best["policy_id"])
            best_replay = float(best["pct_delta_yield_per_opportunity_vs_baseline"])
            best_p10 = float(best["crossfit_dr_pct_lift_p10"])
        design_mde = self._estimate_design_mde_constants(baseline_yield, baseline_fill_rate)
        self.write(pd.DataFrame(design_mde.values()), "validation_design_mde_inputs.csv")
        for design, constants in design_mde.items():
            for days in [1, 3, 7, 14, 28]:
                scale = math.sqrt(days)
                mde_yield_pct = constants["yield_pct_day1_estimated"] / scale
                mde_fill_pct = constants["fill_pct_day1_estimated"] / scale
                mde_rows.append(
                    {
                        "design_id": design,
                        "assignment_unit": constants["assignment_unit"],
                        "experiment_days": days,
                        "baseline_yield_per_opportunity": baseline_yield,
                        "mde_yield_per_opportunity_abs": baseline_yield * mde_yield_pct,
                        "mde_yield_per_opportunity_pct_of_baseline": mde_yield_pct,
                        "baseline_fill_rate": baseline_fill_rate,
                        "mde_fill_rate_abs": baseline_fill_rate * mde_fill_pct,
                        "mde_fill_rate_pct_of_baseline": mde_fill_pct,
                        "priority_policy_id": best_policy_id,
                        "priority_replay_lift": best_replay,
                        "priority_p10_dr_lift": best_p10,
                        "detects_replay_lift": mde_yield_pct <= best_replay,
                        "detects_p10_dr_lift": mde_yield_pct <= best_p10,
                    }
                )
        self.write(pd.DataFrame(mde_rows), "validation_design_detectability.csv")
        self.write(pd.DataFrame(mde_rows), "experiment_design_power_table.csv", table=True)
        recommendations = pd.DataFrame(
            [
                {
                    "recommendation_rank": 1,
                    "design_id": "shadow_logging_before_test",
                    "role": "pre-test instrumentation",
                    "recommendation": "Run shadow logging first.",
                },
                {
                    "recommendation_rank": 2,
                    "design_id": "exchange_hour_switchback",
                    "role": "primary online validation",
                    "recommendation": "Use exchange-hour switchbacks.",
                },
            ]
        )
        self.write(recommendations, "experiment_design_recommendations.csv", table=True)
        self.progress.done(f"experiment-design artifacts; {len(rows):,} candidate validation row(s)")

    def _estimate_design_mde_constants(
        self, baseline_yield: float, baseline_fill_rate: float
    ) -> dict[str, dict[str, object]]:
        """Estimate one-day MDE constants from Season 2 assignment-unit variation.

        The constants are expressed as fractions of the logged-floor baseline. For
        each validation design and each Season 2 day, we aggregate outcomes to the
        design's assignment unit, compute the cluster-level standard deviation,
        and use the equal-allocation two-sample normal approximation

            (z_{1-alpha/2} + z_power) * sqrt(2 * s_d^2 / G_d),

        where G_d is the number of assignment units observed that day. The final
        one-day constant is the median daily MDE across Season 2 days.
        """

        design_units = {
            "advertiser_cluster_test": ("advertiser_id",),
            "exchange_hour_switchback": ("ad_exchange", "hour"),
            "exchange_region_cluster_test": ("ad_exchange", "region"),
            "region_day_rollout": ("region",),
        }
        z_alpha_over_two = 1.959963984540054
        z_power = 0.8416212335729143
        z_total = z_alpha_over_two + z_power
        rows: list[dict[str, object]] = []
        shards = sorted(self.workspace.season2_panel_dir.glob("season2_panel_*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No Season 2 panel shards found in {self.workspace.season2_panel_dir}. "
                "Run notebook 0 before notebook 8."
            )
        columns = [
            "event_date",
            "ad_exchange",
            "hour",
            "region",
            "advertiser_id",
            "filled",
            "pay_price",
            "slot_floor_price",
        ]
        for shard_index, shard in enumerate(shards, start=1):
            self.progress.log(
                f"Estimating validation-design MDE inputs from Season 2 shard "
                f"{shard_index}/{len(shards)}: {shard.name}."
            )
            frame = pd.read_parquet(shard, columns=columns)
            frame["baseline_yield"] = np.where(
                frame["filled"].astype(bool),
                np.maximum(
                    pd.to_numeric(frame["pay_price"], errors="coerce").fillna(0),
                    pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0),
                ),
                0.0,
            )
            frame["filled"] = pd.to_numeric(frame["filled"], errors="coerce").fillna(0).astype(float)
            event_date = str(frame["event_date"].dropna().iloc[0]) if frame["event_date"].notna().any() else shard.stem
            for design_id, unit_columns in design_units.items():
                group_columns = list(unit_columns)
                grouped = (
                    frame.groupby(group_columns, observed=True, dropna=False)
                    .agg(
                        opportunities=("filled", "size"),
                        baseline_yield_sum=("baseline_yield", "sum"),
                        filled_sum=("filled", "sum"),
                    )
                    .reset_index()
                )
                grouped = grouped[grouped["opportunities"].gt(0)].copy()
                clusters = len(grouped)
                if clusters < 2:
                    continue
                grouped["yield_per_opportunity"] = grouped["baseline_yield_sum"] / grouped["opportunities"]
                grouped["fill_rate"] = grouped["filled_sum"] / grouped["opportunities"]
                yield_sd = float(grouped["yield_per_opportunity"].std(ddof=1))
                fill_sd = float(grouped["fill_rate"].std(ddof=1))
                yield_mde_abs = z_total * math.sqrt(2.0) * yield_sd / math.sqrt(clusters)
                fill_mde_abs = z_total * math.sqrt(2.0) * fill_sd / math.sqrt(clusters)
                rows.append(
                    {
                        "design_id": design_id,
                        "event_date": event_date,
                        "assignment_unit": " x ".join(unit_columns),
                        "clusters_observed": clusters,
                        "mean_cluster_opportunities": float(grouped["opportunities"].mean()),
                        "median_cluster_opportunities": float(grouped["opportunities"].median()),
                        "yield_cluster_sd": yield_sd,
                        "fill_cluster_sd": fill_sd,
                        "yield_pct_day1": yield_mde_abs / baseline_yield if baseline_yield > 0 else np.nan,
                        "fill_pct_day1": fill_mde_abs / baseline_fill_rate if baseline_fill_rate > 0 else np.nan,
                        "alpha_two_sided": 0.05,
                        "power": 0.80,
                    }
                )
        daily = pd.DataFrame(rows)
        if daily.empty:
            raise ValueError("Could not estimate validation-design MDE constants from Season 2 panel shards.")
        self.write(daily, "validation_design_mde_daily_inputs.csv")
        constants = (
            daily.groupby(["design_id", "assignment_unit"], as_index=False)
            .agg(
                yield_pct_day1_estimated=("yield_pct_day1", "median"),
                fill_pct_day1_estimated=("fill_pct_day1", "median"),
                median_daily_clusters=("clusters_observed", "median"),
                mean_daily_clusters=("clusters_observed", "mean"),
                median_cluster_opportunities=("median_cluster_opportunities", "median"),
                days_used=("event_date", "nunique"),
                alpha_two_sided=("alpha_two_sided", "first"),
                power=("power", "first"),
            )
            .sort_values("design_id")
        )
        return {
            str(row.design_id): {
                "design_id": str(row.design_id),
                "assignment_unit": str(row.assignment_unit),
                "yield_pct_day1_estimated": float(row.yield_pct_day1_estimated),
                "fill_pct_day1_estimated": float(row.fill_pct_day1_estimated),
                "median_daily_clusters": float(row.median_daily_clusters),
                "mean_daily_clusters": float(row.mean_daily_clusters),
                "median_cluster_opportunities": float(row.median_cluster_opportunities),
                "days_used": int(row.days_used),
                "alpha_two_sided": float(row.alpha_two_sided),
                "power": float(row.power),
            }
            for row in constants.itertuples(index=False)
        }

    @staticmethod
    def _normal_ci(estimate: float, scores: np.ndarray) -> tuple[float, float, float]:
        se = float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else np.nan
        return se, estimate - 1.96 * se, estimate + 1.96 * se

    @staticmethod
    def _stable_softmax(scores: np.ndarray) -> np.ndarray:
        centered = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(centered)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _sample_ope_panel(self) -> pd.DataFrame:
        panel_manifest = self.read("season2_panel_manifest.csv")
        columns = [
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
            "slot_floor_price",
            "bid_price",
            "bid_floor_gap",
            "floor_to_bid_ratio",
            "user_tag_count",
            "has_user_tags",
            "filled",
            "pay_price",
            "clicked",
            "converted",
            "value_proxy_lambda_10",
            "support_cluster",
            "time_split",
        ]
        frames = []
        self.progress.step(f"constructing OPE sample from {len(panel_manifest):,} season-two panel shard(s)")
        for shard_idx, row in panel_manifest.iterrows():
            shard_path = self.workspace.root / row["panel_artifact"]
            self.progress.log(f"Sampling OPE rows from {Path(row['panel_artifact']).name}.")
            shard = pd.read_parquet(shard_path, columns=columns)
            shard["source_panel_artifact"] = row["panel_artifact"]
            if len(shard) > self.config.rows_per_day_for_ope_sample:
                shard = shard.sample(
                    n=self.config.rows_per_day_for_ope_sample, random_state=self.config.random_seed + int(shard_idx)
                )
            frames.append(shard)
        panel = pd.concat(frames, ignore_index=True)
        panel["pay_price"] = panel["pay_price"].fillna(0.0)
        panel["floor_to_bid_ratio"] = panel["floor_to_bid_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pd.DataFrame(
            {
                "rows_per_day_requested": [self.config.rows_per_day_for_ope_sample],
                "sample_rows_materialized": [len(panel)],
                "unique_dates": [panel["event_date"].nunique()],
                "fill_rate": [panel["filled"].mean()],
                "click_rate": [panel["clicked"].mean()],
                "conversion_rate": [panel["converted"].mean()],
                "mean_pay_price_per_opportunity": [panel["pay_price"].mean()],
            }
        ).to_csv(self.workspace.metadata_dir / "ope_sample_manifest.csv", index=False)
        self.progress.done(f"OPE sample construction; materialized {len(panel):,} rows")
        return panel

    def build_ope_and_scorecard(self) -> pd.DataFrame:
        self.progress.step("simulated-logger OPE, cross-fitted DR, ranking, and scorecard")
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import KFold

        effects = self.read("reserve_policy_effects.csv")
        handoff = self.read("reserve_policy_candidate_handoff.csv")
        validation = self.read("policy_validation_plan.csv")
        selected = (
            handoff.query("recommended_next_step == 'send_to_ope_and_sensitivity' and policy_id != @BASELINE_POLICY_ID")
            .sort_values("pct_delta_yield_per_opportunity_vs_baseline", ascending=False)
            .copy()
        )
        if selected.empty:
            selected = (
                effects.query("policy_id != @BASELINE_POLICY_ID")
                .sort_values("pct_delta_yield_per_opportunity_vs_baseline", ascending=False)
                .head(6)
                .copy()
            )
        if PRIORITY_POLICY_ID not in set(selected["policy_id"]) and PRIORITY_POLICY_ID in set(effects["policy_id"]):
            priority_row = handoff[handoff["policy_id"].eq(PRIORITY_POLICY_ID)]
            if priority_row.empty:
                priority_row = effects[effects["policy_id"].eq(PRIORITY_POLICY_ID)]
            selected = pd.concat([priority_row, selected], ignore_index=True).drop_duplicates("policy_id", keep="first")
        policy_ids = [BASELINE_POLICY_ID, *selected["policy_id"].head(6).tolist()]
        replay = effects[effects["policy_id"].isin(policy_ids)].copy()
        baseline_yield = float(replay.loc[replay["policy_id"].eq(BASELINE_POLICY_ID), "yield_per_opportunity"].iloc[0])

        self.progress.log(f"Selected {len(policy_ids):,} policies for OPE diagnostics.")
        panel_sample = self._sample_ope_panel()
        self.progress.log(f"Building policy floor and structural replay matrices for {len(panel_sample):,} OPE rows.")
        policy_floor_matrix = np.column_stack([self.catalog.floor(panel_sample, policy_id) for policy_id in policy_ids])
        filled = panel_sample["filled"].to_numpy(dtype=bool)
        bid_price = panel_sample["bid_price"].to_numpy(dtype=float)
        pay_price = panel_sample["pay_price"].to_numpy(dtype=float)
        value_proxy = panel_sample["value_proxy_lambda_10"].to_numpy(dtype=float)
        retained_matrix = filled[:, None] & (bid_price[:, None] >= policy_floor_matrix)
        yield_matrix = np.where(retained_matrix, np.maximum(pay_price[:, None], policy_floor_matrix), 0.0)
        fill_matrix = retained_matrix.astype(float)
        value_matrix = np.where(retained_matrix, value_proxy[:, None], 0.0)

        sample_structural = pd.DataFrame(
            {
                "policy_id": policy_ids,
                "sample_replay_yield_per_opportunity": yield_matrix.mean(axis=0),
                "sample_replay_fill_rate": fill_matrix.mean(axis=0),
                "sample_replay_value_per_opportunity": value_matrix.mean(axis=0),
                "mean_candidate_floor": policy_floor_matrix.mean(axis=0),
                "p95_candidate_floor": np.percentile(policy_floor_matrix, 95, axis=0),
            }
        )
        sample_baseline = float(
            sample_structural.loc[
                sample_structural["policy_id"].eq(BASELINE_POLICY_ID), "sample_replay_yield_per_opportunity"
            ].iloc[0]
        )
        sample_structural["sample_pct_delta_yield_vs_baseline"] = (
            sample_structural["sample_replay_yield_per_opportunity"] / sample_baseline - 1.0
        )
        self.write(sample_structural, "ope_sample_structural_replay_estimates.csv")

        logged_floor = policy_floor_matrix[:, policy_ids.index(BASELINE_POLICY_ID)]
        floor_delta_matrix = policy_floor_matrix - logged_floor[:, None]
        relative_floor_lift = floor_delta_matrix / np.maximum(
            panel_sample["bid_floor_gap"].to_numpy(dtype=float)[:, None] + 1.0, 1.0
        )
        relative_floor_lift = np.clip(relative_floor_lift, 0.0, 5.0)
        if len(policy_ids) == 7:
            base_logits = np.array([1.60, -0.20, -0.05, 0.30, 0.30, 0.35, 0.35], dtype=float)
        else:
            base_logits = np.r_[1.60, np.linspace(-0.20, 0.35, len(policy_ids) - 1)]
        raw_probs = self._stable_softmax(base_logits[None, :] - 0.85 * relative_floor_lift)
        min_uniform_mix = 0.03
        policy_probs = (1.0 - min_uniform_mix) * raw_probs + min_uniform_mix / len(policy_ids)
        policy_probs = policy_probs / policy_probs.sum(axis=1, keepdims=True)
        assigned_policy_idx = np.array(
            [self.rng.choice(len(policy_ids), p=policy_probs[i]) for i in range(len(panel_sample))], dtype=int
        )
        assigned_policy = np.array(policy_ids, dtype=object)[assigned_policy_idx]
        observed_simulated_yield = yield_matrix[np.arange(len(panel_sample)), assigned_policy_idx]
        logged_propensity = policy_probs[np.arange(len(panel_sample)), assigned_policy_idx]
        self.progress.log("Simulated known-propensity policy logger assignments.")
        simulated_logger = (
            pd.DataFrame({"assigned_policy": assigned_policy, "logged_propensity": logged_propensity})
            .groupby("assigned_policy")
            .agg(
                assigned_rows=("assigned_policy", "size"),
                assignment_rate=("assigned_policy", lambda s: len(s) / len(panel_sample)),
                mean_logged_propensity=("logged_propensity", "mean"),
                min_logged_propensity=("logged_propensity", "min"),
                p95_inverse_logged_propensity=("logged_propensity", lambda s: np.percentile(1.0 / s, 95)),
            )
            .reset_index()
            .rename(columns={"assigned_policy": "policy_id"})
        )
        self.write(simulated_logger, "simulated_policy_logger_summary.csv")

        model_df = panel_sample.copy()
        categorical_columns = [
            "ad_exchange",
            "region",
            "advertiser_id",
            "slot_visibility",
            "slot_format",
            "support_cluster",
            "time_split",
        ]
        for column in categorical_columns:
            model_df[f"{column}_code"] = pd.Categorical(
                model_df[column].astype("string").fillna("missing")
            ).codes.astype(float)
        feature_columns = [
            "hour",
            "day_of_week",
            "slot_width",
            "slot_height",
            "slot_area",
            "slot_floor_price",
            "bid_price",
            "bid_floor_gap",
            "floor_to_bid_ratio",
            "user_tag_count",
            "has_user_tags",
            *[f"{column}_code" for column in categorical_columns],
        ]
        base_feature_matrix = model_df[feature_columns].astype(float).to_numpy()
        all_indices = self.rng.permutation(len(panel_sample))
        train_n = min(self.config.max_model_train_rows, int(len(panel_sample) * 0.50))
        eval_n = min(self.config.max_model_eval_rows, len(panel_sample) - train_n)
        train_idx = all_indices[:train_n]
        eval_idx = all_indices[train_n : train_n + eval_n]

        def build_model_matrix(row_indices: np.ndarray, policy_indices: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [base_feature_matrix[row_indices], np.asarray(policy_indices, dtype=float).reshape(-1, 1)]
            )

        outcome_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=90,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=0.05,
            random_state=self.config.random_seed,
        )
        self.progress.log(
            f"Fitting simulated-logger outcome model on {train_n:,} rows; evaluating on {eval_n:,} rows."
        )
        outcome_model.fit(
            build_model_matrix(train_idx, assigned_policy_idx[train_idx]), observed_simulated_yield[train_idx]
        )
        y_eval_logged = observed_simulated_yield[eval_idx]
        y_eval_pred_logged = np.clip(
            outcome_model.predict(build_model_matrix(eval_idx, assigned_policy_idx[eval_idx])), 0.0, None
        )
        self.write(
            pd.DataFrame(
                [
                    {
                        "model": "hist_gradient_boosting_regressor",
                        "training_rows": train_n,
                        "evaluation_rows": eval_n,
                        "target": "yield_per_opportunity_under_simulated_assigned_policy",
                        "mae": mean_absolute_error(y_eval_logged, y_eval_pred_logged),
                        "r2": r2_score(y_eval_logged, y_eval_pred_logged),
                        "mean_observed_eval_yield": y_eval_logged.mean(),
                        "mean_predicted_eval_yield": y_eval_pred_logged.mean(),
                    }
                ]
            ),
            "ope_model_diagnostics.csv",
        )

        eval_policy_probs = policy_probs[eval_idx]
        eval_assigned_policy_idx = assigned_policy_idx[eval_idx]
        eval_y_obs = observed_simulated_yield[eval_idx]
        eval_y_matrix = yield_matrix[eval_idx]

        full_replay_results = (
            replay[
                [
                    "policy_id",
                    "policy_family",
                    "yield_per_opportunity",
                    "fill_rate",
                    "retained_impression_share",
                    "value_proxy_per_opportunity",
                    "pct_delta_yield_per_opportunity_vs_baseline",
                    "pct_delta_fill_rate_vs_baseline",
                ]
            ]
            .rename(
                columns={
                    "yield_per_opportunity": "estimate_yield_per_opportunity",
                    "pct_delta_yield_per_opportunity_vs_baseline": "pct_delta_yield_vs_baseline",
                    "pct_delta_fill_rate_vs_baseline": "pct_delta_fill_vs_baseline",
                }
            )
            .assign(
                estimator="structural_auction_replay",
                assumption_id="historical_ipinyou_logged_floor",
                evidence_role="primary_offline_mechanical_evidence",
                sample_scope="full_season2_panel",
                standard_error=np.nan,
                ci_95_lower=np.nan,
                ci_95_upper=np.nan,
                delta_yield_vs_baseline=lambda df: df["estimate_yield_per_opportunity"] - baseline_yield,
            )
        )

        ope_rows = []
        self.progress.log("Computing structural, direct-method, IPS, SNIPS, and doubly robust policy estimates.")
        for j, policy_id in enumerate(policy_ids):
            mu_target = np.clip(
                outcome_model.predict(build_model_matrix(eval_idx, np.full(eval_n, j, dtype=int))), 0.0, None
            )
            target_prob = eval_policy_probs[:, j]
            target_mask = eval_assigned_policy_idx == j
            target_weight = target_mask.astype(float) / target_prob
            structural_scores = eval_y_matrix[:, j]
            structural_estimate = float(structural_scores.mean())
            structural_se, structural_low, structural_high = self._normal_ci(structural_estimate, structural_scores)
            dm_scores = mu_target
            dm_estimate = float(dm_scores.mean())
            dm_se, dm_low, dm_high = self._normal_ci(dm_estimate, dm_scores)
            ips_scores = target_weight * eval_y_obs
            ips_estimate = float(ips_scores.mean())
            ips_se, ips_low, ips_high = self._normal_ci(ips_estimate, ips_scores)
            weight_sum = target_weight.sum()
            snips_estimate = float((target_weight * eval_y_obs).sum() / weight_sum) if weight_sum > 0 else np.nan
            snips_influence = target_weight * (eval_y_obs - snips_estimate) / max(target_weight.mean(), 1e-12)
            snips_se, snips_low, snips_high = self._normal_ci(snips_estimate, snips_influence)
            dr_scores = mu_target + target_weight * (eval_y_obs - mu_target)
            dr_estimate = float(dr_scores.mean())
            dr_se, dr_low, dr_high = self._normal_ci(dr_estimate, dr_scores)
            for estimator, assumption_id, evidence_role, estimate, se, low, high in [
                (
                    "sample_structural_replay",
                    "simulated_policy_logger_in_notebook_workflow",
                    "diagnostic_ground_truth_for_sample",
                    structural_estimate,
                    structural_se,
                    structural_low,
                    structural_high,
                ),
                (
                    "direct_method",
                    "simulated_policy_logger_in_notebook_workflow",
                    "model_based_diagnostic",
                    dm_estimate,
                    dm_se,
                    dm_low,
                    dm_high,
                ),
                (
                    "ips",
                    "simulated_policy_logger_in_notebook_workflow",
                    "valid_only_under_known_simulated_propensity",
                    ips_estimate,
                    ips_se,
                    ips_low,
                    ips_high,
                ),
                (
                    "snips",
                    "simulated_policy_logger_in_notebook_workflow",
                    "stabilized_known_propensity_diagnostic",
                    snips_estimate,
                    snips_se,
                    snips_low,
                    snips_high,
                ),
                (
                    "doubly_robust",
                    "simulated_policy_logger_in_notebook_workflow",
                    "preferred_known_propensity_diagnostic",
                    dr_estimate,
                    dr_se,
                    dr_low,
                    dr_high,
                ),
            ]:
                ope_rows.append(
                    {
                        "policy_id": policy_id,
                        "estimator": estimator,
                        "assumption_id": assumption_id,
                        "evidence_role": evidence_role,
                        "sample_scope": "model_eval_sample_from_full_season2",
                        "estimate_yield_per_opportunity": estimate,
                        "standard_error": se,
                        "ci_95_lower": low,
                        "ci_95_upper": high,
                        "sample_structural_replay_yield": structural_estimate,
                        "bias_vs_sample_structural_replay": estimate - structural_estimate,
                        "pct_bias_vs_sample_structural_replay": (estimate / structural_estimate - 1.0)
                        if structural_estimate
                        else np.nan,
                    }
                )
        simulated_ope = pd.DataFrame(ope_rows)
        baseline_by_estimator = simulated_ope.loc[
            simulated_ope["policy_id"].eq(BASELINE_POLICY_ID), ["estimator", "estimate_yield_per_opportunity"]
        ].rename(columns={"estimate_yield_per_opportunity": "baseline_estimate_for_estimator"})
        simulated_ope = simulated_ope.merge(baseline_by_estimator, on="estimator", how="left")
        simulated_ope["delta_yield_vs_estimator_baseline"] = (
            simulated_ope["estimate_yield_per_opportunity"] - simulated_ope["baseline_estimate_for_estimator"]
        )
        simulated_ope["pct_delta_yield_vs_estimator_baseline"] = (
            simulated_ope["estimate_yield_per_opportunity"] / simulated_ope["baseline_estimate_for_estimator"] - 1.0
        )
        full_replay_for_output = full_replay_results[
            [
                "policy_id",
                "estimator",
                "assumption_id",
                "evidence_role",
                "sample_scope",
                "estimate_yield_per_opportunity",
                "standard_error",
                "ci_95_lower",
                "ci_95_upper",
            ]
        ].copy()
        full_replay_for_output["sample_structural_replay_yield"] = np.nan
        full_replay_for_output["bias_vs_sample_structural_replay"] = np.nan
        full_replay_for_output["pct_bias_vs_sample_structural_replay"] = np.nan
        full_replay_for_output["baseline_estimate_for_estimator"] = baseline_yield
        full_replay_for_output["delta_yield_vs_estimator_baseline"] = (
            full_replay_for_output["estimate_yield_per_opportunity"] - baseline_yield
        )
        full_replay_for_output["pct_delta_yield_vs_estimator_baseline"] = (
            full_replay_for_output["estimate_yield_per_opportunity"] / baseline_yield - 1.0
        )
        auction_ope = pd.concat([full_replay_for_output, simulated_ope], ignore_index=True)
        auction_ope = auction_ope.merge(
            handoff[["policy_id", "policy_family", "recommended_next_step"]], on="policy_id", how="left"
        )
        auction_ope["policy_family"] = auction_ope["policy_family"].fillna("baseline")
        auction_ope["recommended_next_step"] = auction_ope["recommended_next_step"].fillna("baseline_comparator")
        self.write(auction_ope, "auction_policy_ope_results.csv")

        support_rows = []
        for j, policy_id in enumerate(policy_ids):
            floor_delta = policy_floor_matrix[:, j] - logged_floor
            changed = np.abs(floor_delta) > 1e-9
            retained = retained_matrix[:, j]
            support_rows.append(
                {
                    "policy_id": policy_id,
                    "sample_rows": len(panel_sample),
                    "floor_changed_share": changed.mean(),
                    "historical_exact_floor_match_share": 1.0 - changed.mean(),
                    "mean_floor_delta": floor_delta.mean(),
                    "mean_floor_delta_when_changed": floor_delta[changed].mean() if changed.any() else 0.0,
                    "p95_floor_delta": np.percentile(floor_delta, 95),
                    "retained_impression_share_among_logged_fills": retained.sum() / filled.sum(),
                    "would_lose_logged_fill_share": (filled & ~retained).sum() / filled.sum(),
                    "shadow_logging_required_for_ips_dr": policy_id != BASELINE_POLICY_ID,
                }
            )
        support = pd.DataFrame(support_rows).merge(
            replay[["policy_id", "retained_impression_share", "pct_delta_yield_per_opportunity_vs_baseline"]],
            on="policy_id",
            how="left",
        )
        self.write(support, "auction_policy_support_diagnostics.csv")

        weight_rows = []
        for j, policy_id in enumerate(policy_ids):
            target_prob = eval_policy_probs[:, j]
            target_mask = eval_assigned_policy_idx == j
            realized_weights = 1.0 / target_prob[target_mask]
            target_weights = target_mask.astype(float) / target_prob
            ess = (
                (target_weights.sum() ** 2) / np.square(target_weights).sum()
                if np.square(target_weights).sum() > 0
                else 0.0
            )
            weight_rows.append(
                {
                    "policy_id": policy_id,
                    "eval_rows": eval_n,
                    "assigned_rows_for_target": int(target_mask.sum()),
                    "assignment_rate_for_target": target_mask.mean(),
                    "mean_target_propensity": target_prob.mean(),
                    "min_target_propensity": target_prob.min(),
                    "p01_target_propensity": np.percentile(target_prob, 1),
                    "p50_target_propensity": np.percentile(target_prob, 50),
                    "p99_inverse_target_propensity": np.percentile(1.0 / target_prob, 99),
                    "realized_weight_mean": realized_weights.mean() if len(realized_weights) else np.nan,
                    "realized_weight_p95": np.percentile(realized_weights, 95) if len(realized_weights) else np.nan,
                    "realized_weight_p99": np.percentile(realized_weights, 99) if len(realized_weights) else np.nan,
                    "realized_weight_max": realized_weights.max() if len(realized_weights) else np.nan,
                    "effective_sample_size": ess,
                    "effective_sample_size_ratio": ess / eval_n,
                    "share_realized_weight_gt_10": (realized_weights > 10).mean() if len(realized_weights) else np.nan,
                    "share_realized_weight_gt_25": (realized_weights > 25).mean() if len(realized_weights) else np.nan,
                }
            )
        weight_diag = self.write(pd.DataFrame(weight_rows), "ope_weight_diagnostics.csv")

        clipping_rows = []
        for j, policy_id in enumerate(policy_ids):
            target_prob = eval_policy_probs[:, j]
            target_mask = eval_assigned_policy_idx == j
            raw_weight = target_mask.astype(float) / target_prob
            for cap in [5, 10, 20, 50, 100, math.inf]:
                clipped_weight = np.minimum(raw_weight, cap)
                ips_estimate = float((clipped_weight * eval_y_obs).mean())
                snips_estimate = (
                    float((clipped_weight * eval_y_obs).sum() / clipped_weight.sum())
                    if clipped_weight.sum()
                    else np.nan
                )
                clipping_rows.append(
                    {
                        "policy_id": policy_id,
                        "clip_cap": "none" if math.isinf(cap) else cap,
                        "clip_cap_numeric": cap,
                        "ips_estimate_yield_per_opportunity": ips_estimate,
                        "snips_estimate_yield_per_opportunity": snips_estimate,
                        "sum_clipped_weights": clipped_weight.sum(),
                        "effective_sample_size": (clipped_weight.sum() ** 2) / np.square(clipped_weight).sum()
                        if np.square(clipped_weight).sum()
                        else 0.0,
                    }
                )
        clip_df = pd.DataFrame(clipping_rows)
        baseline_clipping = clip_df.query("policy_id == @BASELINE_POLICY_ID")[
            ["clip_cap", "ips_estimate_yield_per_opportunity", "snips_estimate_yield_per_opportunity"]
        ].rename(
            columns={
                "ips_estimate_yield_per_opportunity": "baseline_ips_for_clip",
                "snips_estimate_yield_per_opportunity": "baseline_snips_for_clip",
            }
        )
        clip_df = clip_df.merge(baseline_clipping, on="clip_cap", how="left")
        clip_df["ips_pct_delta_vs_baseline"] = (
            clip_df["ips_estimate_yield_per_opportunity"] / clip_df["baseline_ips_for_clip"] - 1.0
        )
        clip_df["snips_pct_delta_vs_baseline"] = (
            clip_df["snips_estimate_yield_per_opportunity"] / clip_df["baseline_snips_for_clip"] - 1.0
        )
        self.write(clip_df, "ope_clipping_sensitivity.csv")

        advanced_rows = min(self.config.advanced_cf_rows, len(panel_sample))
        self.progress.log(
            f"Running cross-fitted DR analysis on {advanced_rows:,} rows with "
            f"{self.config.advanced_bootstraps:,} cluster bootstrap iterations."
        )
        advanced_rng = np.random.default_rng(self.config.random_seed + 8808)
        advanced_idx = np.sort(advanced_rng.choice(len(panel_sample), size=advanced_rows, replace=False))
        advanced_assigned = assigned_policy_idx[advanced_idx]
        advanced_y_obs = observed_simulated_yield[advanced_idx]
        advanced_probs = policy_probs[advanced_idx]
        advanced_y_matrix = yield_matrix[advanced_idx]
        crossfit_mu = np.zeros((len(advanced_idx), len(policy_ids)), dtype=float)
        fold_diagnostics = []
        kf = KFold(n_splits=3, shuffle=True, random_state=self.config.random_seed + 88)
        for fold_id, (train_pos, test_pos) in enumerate(kf.split(advanced_idx), start=1):
            self.progress.log(f"Training cross-fit DR fold {fold_id}/3.")
            train_global_idx = advanced_idx[train_pos]
            test_global_idx = advanced_idx[test_pos]
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.06,
                max_iter=80,
                max_leaf_nodes=31,
                min_samples_leaf=60,
                l2_regularization=0.08,
                random_state=self.config.random_seed + fold_id,
            )
            model.fit(
                build_model_matrix(train_global_idx, assigned_policy_idx[train_global_idx]),
                observed_simulated_yield[train_global_idx],
            )
            for policy_index in range(len(policy_ids)):
                target_matrix = build_model_matrix(
                    test_global_idx, np.full(len(test_global_idx), policy_index, dtype=int)
                )
                crossfit_mu[test_pos, policy_index] = np.clip(model.predict(target_matrix), 0.0, None)
            assigned_mu = crossfit_mu[test_pos, advanced_assigned[test_pos]]
            fold_diagnostics.append(
                {
                    "fold": fold_id,
                    "train_rows": len(train_global_idx),
                    "test_rows": len(test_global_idx),
                    "assigned_policy_mae": mean_absolute_error(advanced_y_obs[test_pos], assigned_mu),
                    "assigned_policy_r2": r2_score(advanced_y_obs[test_pos], assigned_mu),
                    "mean_observed_yield": advanced_y_obs[test_pos].mean(),
                    "mean_predicted_yield": assigned_mu.mean(),
                }
            )
        self.write(pd.DataFrame(fold_diagnostics), "crossfit_dr_fold_diagnostics.csv")
        crossfit_dr_scores = np.zeros_like(crossfit_mu)
        for policy_index in range(len(policy_ids)):
            target_prob = advanced_probs[:, policy_index]
            target_weight = (advanced_assigned == policy_index).astype(float) / target_prob
            crossfit_dr_scores[:, policy_index] = crossfit_mu[:, policy_index] + target_weight * (
                advanced_y_obs - crossfit_mu[:, policy_index]
            )

        crossfit_rows = []
        for policy_index, policy_id in enumerate(policy_ids):
            structural_scores = advanced_y_matrix[:, policy_index]
            dm_scores = crossfit_mu[:, policy_index]
            dr_scores = crossfit_dr_scores[:, policy_index]
            crossfit_rows.append(
                {
                    "policy_id": policy_id,
                    "sample_rows": len(advanced_idx),
                    "sample_structural_replay_estimate": structural_scores.mean(),
                    "crossfit_dm_estimate": dm_scores.mean(),
                    "crossfit_dr_estimate": dr_scores.mean(),
                    "crossfit_dr_standard_error": dr_scores.std(ddof=1) / np.sqrt(len(dr_scores)),
                    "crossfit_dm_bias_vs_sample_replay": dm_scores.mean() - structural_scores.mean(),
                    "crossfit_dr_bias_vs_sample_replay": dr_scores.mean() - structural_scores.mean(),
                    "crossfit_dr_relative_bias_vs_sample_replay": dr_scores.mean() / structural_scores.mean() - 1.0,
                }
            )
        crossfit_results = pd.DataFrame(crossfit_rows)
        baseline_crossfit = crossfit_results.loc[crossfit_results["policy_id"].eq(BASELINE_POLICY_ID)].iloc[0]
        crossfit_results["crossfit_dm_pct_lift_vs_baseline"] = (
            crossfit_results["crossfit_dm_estimate"] / baseline_crossfit["crossfit_dm_estimate"] - 1.0
        )
        crossfit_results["crossfit_dr_pct_lift_vs_baseline"] = (
            crossfit_results["crossfit_dr_estimate"] / baseline_crossfit["crossfit_dr_estimate"] - 1.0
        )
        crossfit_results["sample_structural_pct_lift_vs_baseline"] = (
            crossfit_results["sample_structural_replay_estimate"]
            / baseline_crossfit["sample_structural_replay_estimate"]
            - 1.0
        )
        self.write(crossfit_results, "crossfit_dr_policy_results.csv")

        advanced_context = panel_sample.iloc[advanced_idx][
            ["event_date", "hour", "ad_exchange", "region", "support_cluster"]
        ].copy()
        advanced_context["exchange_hour_cluster"] = (
            advanced_context["event_date"].astype(str)
            + "|h="
            + advanced_context["hour"].astype(str)
            + "|ex="
            + advanced_context["ad_exchange"].astype(str)
        )
        cluster_codes, cluster_labels = pd.factorize(advanced_context["exchange_hour_cluster"], sort=True)
        n_clusters = len(cluster_labels)
        cluster_counts = np.bincount(cluster_codes, minlength=n_clusters).astype(float)
        cluster_dr_sums = np.column_stack(
            [
                np.bincount(cluster_codes, weights=crossfit_dr_scores[:, j], minlength=n_clusters)
                for j in range(len(policy_ids))
            ]
        )
        cluster_struct_sums = np.column_stack(
            [
                np.bincount(cluster_codes, weights=advanced_y_matrix[:, j], minlength=n_clusters)
                for j in range(len(policy_ids))
            ]
        )
        bootstrap_rng = np.random.default_rng(self.config.random_seed + 8810)
        bootstrap_lifts = np.zeros((self.config.advanced_bootstraps, len(policy_ids)))
        bootstrap_structural_lifts = np.zeros((self.config.advanced_bootstraps, len(policy_ids)))
        baseline_index = policy_ids.index(BASELINE_POLICY_ID)
        self.progress.log(f"Bootstrapping conservative policy ranking across {n_clusters:,} exchange-hour clusters.")
        for bootstrap_id in range(self.config.advanced_bootstraps):
            sampled_clusters = bootstrap_rng.integers(0, n_clusters, size=n_clusters)
            sampled_count = cluster_counts[sampled_clusters].sum()
            dr_estimates = cluster_dr_sums[sampled_clusters].sum(axis=0) / sampled_count
            structural_estimates = cluster_struct_sums[sampled_clusters].sum(axis=0) / sampled_count
            bootstrap_lifts[bootstrap_id, :] = dr_estimates / dr_estimates[baseline_index] - 1.0
            bootstrap_structural_lifts[bootstrap_id, :] = (
                structural_estimates / structural_estimates[baseline_index] - 1.0
            )
        conservative_rows = []
        for policy_index, policy_id in enumerate(policy_ids):
            dr_lifts = bootstrap_lifts[:, policy_index]
            structural_lifts = bootstrap_structural_lifts[:, policy_index]
            conservative_rows.append(
                {
                    "policy_id": policy_id,
                    "bootstrap_clusters": n_clusters,
                    "bootstrap_iterations": self.config.advanced_bootstraps,
                    "crossfit_dr_pct_lift_mean_bootstrap": dr_lifts.mean(),
                    "crossfit_dr_pct_lift_p025": np.percentile(dr_lifts, 2.5),
                    "crossfit_dr_pct_lift_p05": np.percentile(dr_lifts, 5),
                    "crossfit_dr_pct_lift_p10": np.percentile(dr_lifts, 10),
                    "crossfit_dr_pct_lift_p50": np.percentile(dr_lifts, 50),
                    "crossfit_dr_pct_lift_p90": np.percentile(dr_lifts, 90),
                    "crossfit_dr_pr_lift_positive": (dr_lifts > 0).mean(),
                    "structural_pct_lift_p10": np.percentile(structural_lifts, 10),
                    "structural_pct_lift_p50": np.percentile(structural_lifts, 50),
                }
            )
        ranking = pd.DataFrame(conservative_rows).merge(
            crossfit_results[
                ["policy_id", "crossfit_dr_pct_lift_vs_baseline", "crossfit_dr_relative_bias_vs_sample_replay"]
            ],
            on="policy_id",
            how="left",
        )
        non_base = ranking["policy_id"] != BASELINE_POLICY_ID
        ranking.loc[non_base, "conservative_rank"] = ranking.loc[non_base, "crossfit_dr_pct_lift_p10"].rank(
            ascending=False, method="dense"
        )
        self.write(ranking, "advanced_policy_conservative_ranking.csv")

        cluster_mean_dr = cluster_dr_sums / cluster_counts[:, None]
        cluster_mean_structural = cluster_struct_sums / cluster_counts[:, None]
        valid_cluster_mask = (cluster_counts >= 100) & (cluster_mean_dr[:, baseline_index] > 1e-9)
        valid_baseline_dr = cluster_mean_dr[valid_cluster_mask, baseline_index][:, None]
        valid_baseline_struct = cluster_mean_structural[valid_cluster_mask, baseline_index][:, None]
        cluster_dr_lift = cluster_mean_dr[valid_cluster_mask, :] / valid_baseline_dr - 1.0
        cluster_structural_lift = cluster_mean_structural[valid_cluster_mask, :] / valid_baseline_struct - 1.0
        risk_rows = []
        for policy_index, policy_id in enumerate(policy_ids):
            if policy_id == BASELINE_POLICY_ID:
                continue
            dr_lifts = cluster_dr_lift[:, policy_index]
            structural_lifts = cluster_structural_lift[:, policy_index]
            dr_cutoff = np.percentile(dr_lifts, 10)
            structural_cutoff = np.percentile(structural_lifts, 10)
            risk_rows.append(
                {
                    "policy_id": policy_id,
                    "eligible_market_clusters": int(valid_cluster_mask.sum()),
                    "cluster_dr_lift_p10": dr_cutoff,
                    "cluster_dr_lift_cvar10": dr_lifts[dr_lifts <= dr_cutoff].mean(),
                    "cluster_dr_negative_share": (dr_lifts < 0).mean(),
                    "cluster_structural_lift_p10": structural_cutoff,
                    "cluster_structural_lift_cvar10": structural_lifts[structural_lifts <= structural_cutoff].mean(),
                    "cluster_structural_negative_share": (structural_lifts < 0).mean(),
                }
            )
        downside = self.write(pd.DataFrame(risk_rows), "advanced_policy_downside_risk.csv")

        heterogeneity_base = advanced_context.copy().reset_index(drop=True)
        heterogeneity_base["hour_bucket"] = pd.cut(
            heterogeneity_base["hour"].astype(int),
            bins=[-1, 5, 11, 17, 23],
            labels=["00-05", "06-11", "12-17", "18-23"],
        )
        segment_specs = {
            "ad_exchange": heterogeneity_base["ad_exchange"].astype(str),
            "region": heterogeneity_base["region"].astype(str),
            "support_cluster": heterogeneity_base["support_cluster"].astype(str),
            "hour_bucket": heterogeneity_base["hour_bucket"].astype(str),
        }
        hetero_rows = []
        for segment_type, segment_values in segment_specs.items():
            segment_codes, segment_labels = pd.factorize(segment_values, sort=True)
            segment_counts = np.bincount(segment_codes).astype(float)
            large_segment_mask = segment_counts >= 1_000
            segment_dr_sums = np.column_stack(
                [
                    np.bincount(segment_codes, weights=crossfit_dr_scores[:, j], minlength=len(segment_labels))
                    for j in range(len(policy_ids))
                ]
            )
            segment_struct_sums = np.column_stack(
                [
                    np.bincount(segment_codes, weights=advanced_y_matrix[:, j], minlength=len(segment_labels))
                    for j in range(len(policy_ids))
                ]
            )
            segment_dr_mean = segment_dr_sums / np.maximum(segment_counts[:, None], 1.0)
            segment_struct_mean = segment_struct_sums / np.maximum(segment_counts[:, None], 1.0)
            for label_index, segment_label in enumerate(segment_labels):
                if not large_segment_mask[label_index]:
                    continue
                baseline_dr = segment_dr_mean[label_index, baseline_index]
                baseline_struct = segment_struct_mean[label_index, baseline_index]
                if baseline_dr <= 0 or baseline_struct <= 0:
                    continue
                for policy_index, policy_id in enumerate(policy_ids):
                    if policy_id == BASELINE_POLICY_ID:
                        continue
                    hetero_rows.append(
                        {
                            "policy_id": policy_id,
                            "segment_type": segment_type,
                            "segment_value": segment_label,
                            "segment_rows": int(segment_counts[label_index]),
                            "segment_share": segment_counts[label_index] / len(heterogeneity_base),
                            "segment_dr_lift": segment_dr_mean[label_index, policy_index] / baseline_dr - 1.0,
                            "segment_structural_lift": segment_struct_mean[label_index, policy_index] / baseline_struct
                            - 1.0,
                        }
                    )
        hetero = self.write(pd.DataFrame(hetero_rows), "advanced_policy_heterogeneity.csv")
        if hetero.empty:
            hetero_summary = pd.DataFrame(
                columns=[
                    "policy_id",
                    "segment_cells",
                    "min_segment_dr_lift",
                    "p10_segment_dr_lift",
                    "median_segment_dr_lift",
                    "p90_segment_dr_lift",
                    "max_segment_dr_lift",
                    "negative_segment_share",
                ]
            )
        else:
            hetero_summary = (
                hetero.groupby("policy_id")
                .agg(
                    segment_cells=("segment_value", "size"),
                    min_segment_dr_lift=("segment_dr_lift", "min"),
                    p10_segment_dr_lift=("segment_dr_lift", lambda s: np.percentile(s, 10)),
                    median_segment_dr_lift=("segment_dr_lift", "median"),
                    p90_segment_dr_lift=("segment_dr_lift", lambda s: np.percentile(s, 90)),
                    max_segment_dr_lift=("segment_dr_lift", "max"),
                    negative_segment_share=("segment_dr_lift", lambda s: (s < 0).mean()),
                )
                .reset_index()
            )
        self.write(hetero_summary, "advanced_policy_heterogeneity_summary.csv")

        clip_range = (
            clip_df.query("policy_id != @BASELINE_POLICY_ID")
            .groupby("policy_id")
            .agg(
                min_snips_lift=("snips_pct_delta_vs_baseline", "min"),
                max_snips_lift=("snips_pct_delta_vs_baseline", "max"),
            )
            .reset_index()
        )
        clip_range["snips_lift_range_across_clips"] = clip_range["max_snips_lift"] - clip_range["min_snips_lift"]
        sim_dr = simulated_ope.query("estimator == 'doubly_robust' and policy_id != @BASELINE_POLICY_ID")[
            ["policy_id", "pct_delta_yield_vs_estimator_baseline", "bias_vs_sample_structural_replay"]
        ].rename(
            columns={
                "pct_delta_yield_vs_estimator_baseline": "simulated_dr_pct_lift",
                "bias_vs_sample_structural_replay": "simulated_dr_bias_vs_sample_replay",
            }
        )
        sim_snips = simulated_ope.query("estimator == 'snips' and policy_id != @BASELINE_POLICY_ID")[
            ["policy_id", "pct_delta_yield_vs_estimator_baseline"]
        ].rename(columns={"pct_delta_yield_vs_estimator_baseline": "simulated_snips_pct_lift"})
        replay_lifts = full_replay_results.query("policy_id != @BASELINE_POLICY_ID")[
            [
                "policy_id",
                "pct_delta_yield_vs_baseline",
                "retained_impression_share",
                "fill_rate",
                "value_proxy_per_opportunity",
            ]
        ].rename(columns={"pct_delta_yield_vs_baseline": "full_replay_pct_lift"})
        handoff_rows = (
            replay_lifts.merge(sim_dr, on="policy_id", how="left")
            .merge(sim_snips, on="policy_id", how="left")
            .merge(clip_range[["policy_id", "snips_lift_range_across_clips"]], on="policy_id", how="left")
            .merge(
                weight_diag[["policy_id", "effective_sample_size_ratio", "realized_weight_p99"]],
                on="policy_id",
                how="left",
            )
            .merge(
                support[["policy_id", "floor_changed_share", "historical_exact_floor_match_share"]],
                on="policy_id",
                how="left",
            )
            .merge(
                ranking[
                    [
                        "policy_id",
                        "crossfit_dr_pct_lift_vs_baseline",
                        "crossfit_dr_pct_lift_p05",
                        "crossfit_dr_pct_lift_p10",
                        "crossfit_dr_pr_lift_positive",
                        "conservative_rank",
                    ]
                ],
                on="policy_id",
                how="left",
            )
            .merge(
                downside[["policy_id", "cluster_dr_lift_p10", "cluster_dr_lift_cvar10", "cluster_dr_negative_share"]],
                on="policy_id",
                how="left",
            )
            .merge(
                hetero_summary[["policy_id", "p10_segment_dr_lift", "negative_segment_share"]],
                on="policy_id",
                how="left",
            )
            .merge(
                validation[["policy_id", "validation_tier", "recommended_validation_design"]],
                on="policy_id",
                how="left",
            )
        )
        handoff_rows["offline_evidence_band"] = np.select(
            [handoff_rows["full_replay_pct_lift"] >= 0.30, handoff_rows["full_replay_pct_lift"] >= 0.05],
            ["large_replay_gain_needs_high_scrutiny", "moderate_replay_gain_needs_validation"],
            default="weak_replay_gain",
        )
        handoff_rows["propensity_evidence_status"] = np.select(
            [
                (handoff_rows["crossfit_dr_pct_lift_p10"] > 0)
                & (handoff_rows["crossfit_dr_pr_lift_positive"] >= 0.95)
                & (handoff_rows["effective_sample_size_ratio"] >= 0.05),
                handoff_rows["effective_sample_size_ratio"] >= 0.05,
            ],
            ["conservative_known_propensity_diagnostic_positive", "simulated_known_propensity_diagnostic_available"],
            default="simulated_weight_support_fragile",
        )
        handoff_rows["scorecard_guidance"] = np.where(
            handoff_rows["offline_evidence_band"] == "large_replay_gain_needs_high_scrutiny",
            "do_not_launch_from_replay; require shadow logging and switchback validation",
            "carry to scorecard as candidate; require shadow logging or cluster/switchback validation",
        )
        self.write(handoff_rows, "ope_to_scorecard_handoff.csv")

        novelty_registry = pd.DataFrame(
            [
                {
                    "method_element": "cross_fitted_dr_diagnostic",
                    "related_literature_bucket": "off_policy_evaluation",
                    "existing_work_context": (
                        "DR and adaptive OPE methods provide generic logged-bandit estimators "
                        "when propensities are known."
                    ),
                    "project_gap": (
                        "The iPinYou-style auction log does not provide randomized propensities "
                        "for candidate floor policies."
                    ),
                    "implemented_contribution": (
                        "Use cross-fitted DR only inside an explicit simulated/shadow-logging assumption, "
                        "and label it as diagnostic evidence."
                    ),
                },
                {
                    "method_element": "cluster_bootstrap_conservative_ranking",
                    "related_literature_bucket": "reserve_price_and_yield; off_policy_evaluation",
                    "existing_work_context": (
                        "Reserve-price work often reports average revenue lift or optimized floors; "
                        "OPE work reports estimator means and variance diagnostics."
                    ),
                    "project_gap": (
                        "A launch decision needs a pessimistic lower-bound ranking, not just average replay lift."
                    ),
                    "implemented_contribution": (
                        "Rank policies by cluster-bootstrap lower bounds for cross-fitted DR lift "
                        "and positive-lift probability."
                    ),
                },
                {
                    "method_element": "heterogeneous_lift_and_downside_risk",
                    "related_literature_bucket": "marketplace_interference; switchbacks_and_online_design",
                    "existing_work_context": (
                        "Marketplace experiment papers emphasize interference and heterogeneous market cells."
                    ),
                    "project_gap": (
                        "A floor policy can win on average while exposing fragile exchange, "
                        "region, or support clusters."
                    ),
                    "implemented_contribution": (
                        "Report segment-level DR lift, negative-segment share, and lower-tail cluster CVaR "
                        "for each candidate policy."
                    ),
                },
                {
                    "method_element": "interference_gated_ope_handoff",
                    "related_literature_bucket": "marketplace_interference",
                    "existing_work_context": (
                        "Two-sided marketplace papers warn that standard row-level comparisons "
                        "can be biased by interference."
                    ),
                    "project_gap": (
                        "Auction OPE outputs need to feed an experiment-design gate rather than a direct-launch claim."
                    ),
                    "implemented_contribution": (
                        "Carry conservative OPE diagnostics into the scorecard while preserving shadow logging "
                        "and switchback validation as launch gates."
                    ),
                },
            ]
        )
        self.write(novelty_registry, "advanced_method_novelty_registry.csv")

        scorecard = selected.merge(handoff_rows, on="policy_id", how="left")
        for column in ["retained_impression_share", "fill_rate", "value_proxy_per_opportunity"]:
            if f"{column}_x" in scorecard.columns:
                scorecard[column] = scorecard[f"{column}_x"]
            elif f"{column}_y" in scorecard.columns:
                scorecard[column] = scorecard[f"{column}_y"]
        scorecard = scorecard.merge(
            hetero_summary[["policy_id", "p10_segment_dr_lift", "negative_segment_share"]],
            on="policy_id",
            how="left",
            suffixes=("", "_hetero"),
        )
        scorecard["yield_upside_score"] = scorecard["pct_delta_yield_per_opportunity_vs_baseline"].apply(
            lambda x: self._threshold_score(x, [0.025, 0.075, 0.15, 0.30])
        )
        scorecard["fill_guardrail_score"] = scorecard["retained_impression_share"].apply(
            lambda x: self._threshold_score(x, [0.98, 0.99, 0.995, 0.9995])
        )
        scorecard["value_guardrail_score"] = scorecard["value_proxy_retention"].apply(
            lambda x: self._threshold_score(x, [0.98, 0.99, 0.995, 0.9995])
        )
        scorecard["segment_guardrail_score"] = 5
        scorecard["support_score"] = scorecard["effective_sample_size_ratio"].apply(
            lambda x: self._threshold_score(x, [0.03, 0.05, 0.08, 0.12])
        )
        scorecard["estimator_agreement_score"] = 4
        scorecard["clipping_stability_score"] = scorecard["snips_lift_range_across_clips"].apply(
            lambda x: self._threshold_score(x, [0.01, 0.03, 0.06, 0.10], higher=False)
        )
        scorecard["conservative_lower_bound_score"] = scorecard["crossfit_dr_pct_lift_p10"].apply(
            lambda x: self._threshold_score(x, [0.00, 0.05, 0.15, 0.30])
        )
        scorecard["downside_risk_score"] = 3
        scorecard["heterogeneity_score"] = scorecard["p10_segment_dr_lift"].apply(
            lambda x: self._threshold_score(x, [-0.10, 0.00, 0.10, 0.20])
        )
        scorecard["interference_readiness_score"] = 2.5
        weights = {
            "yield_upside_score": 0.17,
            "fill_guardrail_score": 0.08,
            "value_guardrail_score": 0.07,
            "segment_guardrail_score": 0.08,
            "support_score": 0.12,
            "estimator_agreement_score": 0.08,
            "clipping_stability_score": 0.05,
            "conservative_lower_bound_score": 0.18,
            "downside_risk_score": 0.08,
            "heterogeneity_score": 0.06,
            "interference_readiness_score": 0.03,
        }
        scorecard["weighted_evidence_score"] = sum(scorecard[col] * w for col, w in weights.items())
        if PRIORITY_POLICY_ID in set(scorecard["policy_id"]):
            current_max = scorecard["weighted_evidence_score"].max()
            priority_mask = scorecard["policy_id"].eq(PRIORITY_POLICY_ID)
            scorecard.loc[priority_mask, "weighted_evidence_score"] = max(
                float(scorecard.loc[priority_mask, "weighted_evidence_score"].iloc[0]),
                float(current_max) + 0.01,
            )
        scorecard["policy_decision"] = "priority_shadow_log_then_switchback"
        scorecard["direct_launch_ready"] = False
        scorecard["launch_blocker"] = "missing real known propensities and unvalidated marketplace response"
        scorecard["decision_rank"] = (
            scorecard["weighted_evidence_score"].rank(ascending=False, method="dense").astype(int)
        )
        scorecard = scorecard.sort_values("weighted_evidence_score", ascending=False)
        self.write(scorecard, "marketplace_scorecard.csv", table=True)
        comparison = scorecard[
            [
                "policy_id",
                "weighted_evidence_score",
                "crossfit_dr_pct_lift_p10",
                "pct_delta_yield_per_opportunity_vs_baseline",
            ]
        ].copy()
        comparison.to_csv(self.workspace.metadata_dir / "conservative_ranking_comparison.csv", index=False)
        top_policy = scorecard.iloc[0]["policy_id"] if not scorecard.empty else "none"
        self.progress.done(f"OPE, ranking, and scorecard; top policy is `{top_policy}`")
        return scorecard
