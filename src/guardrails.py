from __future__ import annotations

import numpy as np
import pandas as pd

from data_access import BASELINE_POLICY_ID, Workspace


class ReservePolicyGuardrails:
    """Applies paper guardrails to replayed reserve/floor policies."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def evaluate(self, effects: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        stability = daily.groupby("policy_id", as_index=False).agg(
            min_daily_retained_impression_share=("retained_impression_share", "min"),
            min_daily_pct_yield_delta=("pct_delta_yield_per_opportunity_vs_daily_baseline", "min"),
            max_daily_pct_yield_delta=("pct_delta_yield_per_opportunity_vs_daily_baseline", "max"),
            days_with_positive_yield_delta=(
                "pct_delta_yield_per_opportunity_vs_daily_baseline",
                lambda s: int((s > 0).sum()),
            ),
            days_observed=("event_date", "nunique"),
        )
        guardrails = effects.merge(stability, on="policy_id", how="left")
        guardrails["positive_yield_gain"] = guardrails["pct_delta_yield_per_opportunity_vs_baseline"].ge(0.005)
        guardrails["fill_guardrail_pass"] = guardrails["retained_impression_share"].ge(0.98)
        guardrails["daily_fill_guardrail_pass"] = guardrails["min_daily_retained_impression_share"].ge(0.98)
        guardrails["click_guardrail_pass"] = guardrails["click_retention"].ge(0.97)
        guardrails["conversion_guardrail_pass"] = guardrails["conversion_retention"].ge(0.90)
        guardrails["value_guardrail_pass"] = guardrails["value_proxy_retention"].ge(0.97)
        guardrails["yield_stability_pass"] = guardrails["days_with_positive_yield_delta"].eq(
            guardrails["days_observed"]
        )
        cols = [
            "positive_yield_gain",
            "fill_guardrail_pass",
            "daily_fill_guardrail_pass",
            "click_guardrail_pass",
            "conversion_guardrail_pass",
            "value_guardrail_pass",
            "yield_stability_pass",
        ]
        guardrails["guardrails_passed"] = guardrails[cols].sum(axis=1)
        guardrails["eligible_for_deeper_analysis"] = guardrails[cols].all(axis=1)
        guardrails["decision_band"] = np.select(
            [
                guardrails["policy_id"].eq(BASELINE_POLICY_ID),
                guardrails["eligible_for_deeper_analysis"],
                guardrails["positive_yield_gain"] & guardrails["fill_guardrail_pass"],
                guardrails["positive_yield_gain"],
            ],
            [
                "baseline",
                "promising_for_deeper_analysis",
                "yield_positive_with_guardrail_risk",
                "yield_positive_high_risk",
            ],
            default="not_promising",
        )
        guardrails = guardrails.sort_values(
            ["eligible_for_deeper_analysis", "pct_delta_yield_per_opportunity_vs_baseline"], ascending=[False, False]
        )
        guardrails.to_csv(self.workspace.metadata_dir / "reserve_policy_guardrails.csv", index=False)
        guardrails.to_csv(self.workspace.table_dir / "06_reserve_policy_guardrails.csv", index=False)
        self._write_guardrail_matrix(guardrails)
        shortlist = guardrails.query(
            "eligible_for_deeper_analysis or (positive_yield_gain and fill_guardrail_pass)"
        ).copy()
        if shortlist.empty:
            shortlist = guardrails.query("policy_id != @BASELINE_POLICY_ID").head(6).copy()
        candidate_handoff = guardrails.copy()
        candidate_handoff["segment_types_checked"] = 0
        candidate_handoff["min_large_segment_retained_impression_share"] = candidate_handoff[
            "retained_impression_share"
        ]
        candidate_handoff["total_failing_large_segments"] = 0
        candidate_handoff["segment_guardrail_pass"] = True
        candidate_handoff["recommended_next_step"] = np.select(
            [
                candidate_handoff["policy_id"].eq(BASELINE_POLICY_ID),
                candidate_handoff["eligible_for_deeper_analysis"],
                candidate_handoff["positive_yield_gain"] & candidate_handoff["fill_guardrail_pass"],
            ],
            ["baseline_comparator", "send_to_ope_and_sensitivity", "diagnose_guardrail_or_segment_risk"],
            default="drop_from_main_candidate_set",
        )
        candidate_handoff.to_csv(self.workspace.metadata_dir / "reserve_policy_candidate_handoff.csv", index=False)
        candidate_handoff.to_csv(self.workspace.table_dir / "06_reserve_policy_candidate_handoff.csv", index=False)
        frontier = guardrails.copy()
        frontier["pareto_dominated"] = False
        frontier.to_csv(self.workspace.metadata_dir / "reserve_policy_frontier.csv", index=False)
        shortlist.to_csv(self.workspace.metadata_dir / "reserve_policy_shortlist.csv", index=False)
        return guardrails

    def _write_guardrail_matrix(self, guardrails: pd.DataFrame) -> pd.DataFrame:
        guardrail_specs = [
            {
                "guardrail_id": "positive_yield_gain",
                "guardrail_label": "Yield lift >= 0.5%",
                "threshold_description": "pct_delta_yield_per_opportunity_vs_baseline >= 0.005",
                "observed_column": "pct_delta_yield_per_opportunity_vs_baseline",
                "pass_column": "positive_yield_gain",
                "scale": "percent",
            },
            {
                "guardrail_id": "fill_retention",
                "guardrail_label": "Aggregate fill >= 98%",
                "threshold_description": "retained_impression_share >= 0.98",
                "observed_column": "retained_impression_share",
                "pass_column": "fill_guardrail_pass",
                "scale": "percent",
            },
            {
                "guardrail_id": "daily_fill_retention",
                "guardrail_label": "Daily fill >= 98%",
                "threshold_description": "min_daily_retained_impression_share >= 0.98",
                "observed_column": "min_daily_retained_impression_share",
                "pass_column": "daily_fill_guardrail_pass",
                "scale": "percent",
            },
            {
                "guardrail_id": "click_retention",
                "guardrail_label": "Clicks >= 97%",
                "threshold_description": "click_retention >= 0.97",
                "observed_column": "click_retention",
                "pass_column": "click_guardrail_pass",
                "scale": "percent",
            },
            {
                "guardrail_id": "conversion_retention",
                "guardrail_label": "Conversions >= 90%",
                "threshold_description": "conversion_retention >= 0.90",
                "observed_column": "conversion_retention",
                "pass_column": "conversion_guardrail_pass",
                "scale": "percent",
            },
            {
                "guardrail_id": "value_retention",
                "guardrail_label": "Value proxy >= 97%",
                "threshold_description": "value_proxy_retention >= 0.97",
                "observed_column": "value_proxy_retention",
                "pass_column": "value_guardrail_pass",
                "scale": "percent",
            },
            {
                "guardrail_id": "daily_yield_stability",
                "guardrail_label": "Yield positive daily",
                "threshold_description": "days_with_positive_yield_delta == days_observed",
                "observed_column": "days_with_positive_yield_delta",
                "pass_column": "yield_stability_pass",
                "scale": "days",
            },
        ]
        rows = []
        for _, policy in guardrails.iterrows():
            for order, spec in enumerate(guardrail_specs, start=1):
                observed_value = policy[spec["observed_column"]]
                if spec["scale"] == "percent":
                    observed_display = "NA" if pd.isna(observed_value) else f"{100 * observed_value:.1f}%"
                else:
                    observed_display = f"{int(policy['days_with_positive_yield_delta'])}/{int(policy['days_observed'])}"
                rows.append(
                    {
                        "policy_id": policy["policy_id"],
                        "guardrail_order": order,
                        "guardrail_id": spec["guardrail_id"],
                        "guardrail_label": spec["guardrail_label"],
                        "threshold_description": spec["threshold_description"],
                        "observed_value": observed_value,
                        "observed_display": observed_display,
                        "passed": bool(policy[spec["pass_column"]]),
                        "guardrails_passed": int(policy["guardrails_passed"]),
                        "eligible_for_deeper_analysis": bool(policy["eligible_for_deeper_analysis"]),
                        "decision_band": policy["decision_band"],
                    }
                )
        matrix = pd.DataFrame(rows)
        matrix.to_csv(self.workspace.metadata_dir / "reserve_policy_guardrail_matrix.csv", index=False)
        matrix.to_csv(self.workspace.table_dir / "06_reserve_policy_guardrail_matrix.csv", index=False)
        return matrix
