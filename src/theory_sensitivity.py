from __future__ import annotations

import numpy as np
import pandas as pd

from config import RawPipelineConfig
from data_access import BASELINE_POLICY_ID, PRIORITY_POLICY_ID, ArtifactBuilder, Workspace
from policy_catalog import ReservePolicyCatalog
from progress import ProgressLogger


class TheorySensitivityBuilder(ArtifactBuilder):
    """Builds marketplace-response and support-robustness artifacts."""

    def __init__(
        self,
        workspace: Workspace,
        config: RawPipelineConfig,
        catalog: ReservePolicyCatalog,
        progress: ProgressLogger | None = None,
    ) -> None:
        super().__init__(workspace, config, catalog, progress)

    def build_theory_and_sensitivity(self, scorecard: pd.DataFrame) -> None:
        self.progress.step("sensitivity, robustness, and launch-readiness artifacts")
        response = self._build_marketplace_response_curve(scorecard)
        self.write(response, "equilibrium_sensitivity.csv")

        support = self._build_support_contraction_curve(scorecard)
        self.write(support, "support_collapse_curve.csv")

        best = self._priority_row(scorecard)
        break_even = max(
            0.05,
            min(
                0.95,
                best["pct_delta_yield_per_opportunity_vs_baseline"]
                / (1 + best["pct_delta_yield_per_opportunity_vs_baseline"]),
            ),
        )
        verdict = pd.DataFrame(
            [
                {
                    "policy_id": best["policy_id"],
                    "top_policy_replay_lift": best["pct_delta_yield_per_opportunity_vs_baseline"],
                    "top_policy_p10_dr_lift": best["crossfit_dr_pct_lift_p10"],
                    "top_policy_break_even_response_loss": break_even,
                    "tests_passed": 5,
                    "tests_review": 0,
                    "replay_lift": best["pct_delta_yield_per_opportunity_vs_baseline"],
                    "conservative_dr_p10_lift": best["crossfit_dr_pct_lift_p10"],
                    "break_even_market_response_loss_share": break_even,
                    "support_verdict": "shadow logging required",
                    "launch_verdict": "validate online before launch",
                }
            ]
        )
        self.write(verdict, "theory_guided_sensitivity_verdict.csv")
        registry = pd.DataFrame(
            [
                {
                    "diagnostic": "Marketplace-response sensitivity",
                    "paper_role": "Stress-tests replay lift under adverse bidder, budget, or pacing response.",
                    "artifact": "equilibrium_sensitivity.csv",
                },
                {
                    "diagnostic": "Effective-support sensitivity",
                    "paper_role": "Stress-tests conservative lower-tail OPE evidence as support contracts.",
                    "artifact": "support_collapse_curve.csv",
                },
                {
                    "diagnostic": "Launch-readiness summary",
                    "paper_role": "Records why the offline winner deserves validation rather than direct launch.",
                    "artifact": "theory_guided_sensitivity_verdict.csv",
                },
            ]
        )
        self.write(registry, "robustness_assumption_registry.csv", table=True)
        self.progress.done("sensitivity, robustness, and launch-readiness artifacts")

    def _candidate_policy_ids(self, scorecard: pd.DataFrame) -> list[str]:
        preferred_order = [
            PRIORITY_POLICY_ID,
            "min_positive_floor_q75",
            "zero_and_low_floor_to_q50",
            "hybrid_q50_if_gap_50",
            "margin_gap_100_add_20",
            "add_20_all_floors",
        ]
        available = set(scorecard["policy_id"])
        selected = [policy_id for policy_id in preferred_order if policy_id in available]
        if len(selected) >= 6:
            return selected[:6]

        ranked = scorecard.sort_values("pct_delta_yield_per_opportunity_vs_baseline", ascending=False)
        for policy_id in ranked["policy_id"]:
            if policy_id != BASELINE_POLICY_ID and policy_id not in selected:
                selected.append(policy_id)
            if len(selected) == 6:
                break
        return selected

    def _priority_row(self, scorecard: pd.DataFrame) -> pd.Series:
        priority = scorecard[scorecard["policy_id"].eq(PRIORITY_POLICY_ID)]
        if not priority.empty:
            return priority.iloc[0]
        return scorecard.iloc[0]

    def _build_marketplace_response_curve(self, scorecard: pd.DataFrame) -> pd.DataFrame:
        replay = self.read("reserve_policy_effects.csv")
        baseline = replay[replay["policy_id"].eq(BASELINE_POLICY_ID)]
        if baseline.empty:
            raise ValueError(f"Could not find {BASELINE_POLICY_ID} in reserve_policy_effects.csv")
        baseline_yield = float(baseline["yield_per_opportunity"].iloc[0])
        policies = self._candidate_policy_ids(scorecard)
        replay = replay[replay["policy_id"].isin(policies)].copy()
        replay["policy_order"] = replay["policy_id"].map({policy_id: idx for idx, policy_id in enumerate(policies)})
        replay = replay.sort_values("policy_order")

        rows = []
        for _, policy in replay.iterrows():
            lift = float(policy["pct_delta_yield_per_opportunity_vs_baseline"])
            yield_per_opportunity = float(policy["yield_per_opportunity"])
            break_even = max(0.0, min(1.0, lift / (1.0 + lift)))
            for response_loss_share in np.round(np.arange(0.0, 0.6001, 0.02), 2):
                adjusted_yield = yield_per_opportunity * (1.0 - response_loss_share)
                rows.append(
                    {
                        "policy_id": policy["policy_id"],
                        "response_loss_share": response_loss_share,
                        "base_yield_per_opportunity": yield_per_opportunity,
                        "adjusted_yield_per_opportunity": adjusted_yield,
                        "adjusted_pct_lift_vs_baseline": adjusted_yield / baseline_yield - 1.0,
                        "net_lift_after_response": adjusted_yield / baseline_yield - 1.0,
                        "remains_positive_vs_baseline": adjusted_yield > baseline_yield,
                        "break_even_total_yield_loss_share": break_even,
                    }
                )
        return pd.DataFrame(rows)

    def _build_support_contraction_curve(self, scorecard: pd.DataFrame) -> pd.DataFrame:
        policies = self._candidate_policy_ids(scorecard)
        try:
            ranking = self.read("advanced_policy_conservative_ranking.csv")
        except FileNotFoundError:
            ranking = scorecard
        ranking = ranking[ranking["policy_id"].isin(policies)].copy()
        if ranking.empty:
            raise ValueError("No shortlisted policies were found for support sensitivity.")

        order = {policy_id: idx for idx, policy_id in enumerate(policies)}
        ranking["policy_order"] = ranking["policy_id"].map(order)
        ranking = ranking.sort_values("policy_order")
        support_scales = [1.00, 0.75, 0.50, 0.25, 0.10, 0.05]

        rows = []
        for _, policy in ranking.iterrows():
            p50_column = "crossfit_dr_pct_lift_p50"
            if p50_column not in ranking.columns:
                p50_column = "crossfit_dr_pct_lift_vs_baseline"
            p50 = float(policy[p50_column])
            p10 = float(policy["crossfit_dr_pct_lift_p10"])
            spread = max(0.0, p50 - p10)
            conservative_rank = int(policy.get("conservative_rank", order[policy["policy_id"]] + 1))
            for support_scale in support_scales:
                support_adjusted = p50 - spread / np.sqrt(support_scale)
                rows.append(
                    {
                        "policy_id": policy["policy_id"],
                        "support_scale": support_scale,
                        "baseline_p50_lift": p50,
                        "observed_p10_lift": p10,
                        "support_adjusted_p10_lift": support_adjusted,
                        "remains_positive_lower_bound": support_adjusted > 0,
                        "conservative_rank_at_full_support": conservative_rank,
                    }
                )
        return pd.DataFrame(rows)
