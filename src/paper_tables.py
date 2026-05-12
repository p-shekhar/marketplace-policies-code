from __future__ import annotations

import pandas as pd

from config import RawPipelineConfig
from data_access import (  # noqa: F401
    BASELINE_POLICY_ID,
    PRIORITY_POLICY_ID,
    PRIORITY_POLICY_LABEL,
    ArtifactBuilder,
    Workspace,
)
from policy_catalog import ReservePolicyCatalog
from progress import ProgressLogger


class PaperTableBuilder(ArtifactBuilder):
    """Builds manuscript table catalogs and figure/table selection manifests."""

    def __init__(
        self,
        workspace: Workspace,
        config: RawPipelineConfig,
        catalog: ReservePolicyCatalog,
        progress: ProgressLogger | None = None,
    ) -> None:
        super().__init__(workspace, config, catalog, progress)

    def build_static_tables(self) -> None:
        self.progress.step("static table and figure-selection catalogs")
        logging_contract = pd.DataFrame(
            [
                {"field": "bid_id", "role": "join key", "needed_for": "event reconciliation"},
                {"field": "slot_floor_price", "role": "logged treatment state", "needed_for": "reserve/floor replay"},
                {"field": "bid_price", "role": "auction state", "needed_for": "candidate-floor clearing"},
                {"field": "pay_price", "role": "observed outcome", "needed_for": "yield calculation"},
            ]
        )
        logging_contract.to_csv(self.workspace.table_dir / "01_logging_contract.csv", index=False)
        season2_density = self.read("season2_outcome_density.csv")
        season3_daily = self.read("season3_policy_daily_effects.csv")
        season3_baseline = season3_daily.query("policy_id == @BASELINE_POLICY_ID").copy()
        data_windows = pd.DataFrame(
            [
                {
                    "window": "Season two",
                    "role_in_study": "Discovery and main empirical panel",
                    "dates": int(season2_density["event_date"].nunique()),
                    "bid_opportunities": int(season2_density["bid_opportunities"].sum()),
                    "filled_impressions": int(season2_density["filled"].sum()),
                    "fill_rate": float(
                        season2_density["filled"].sum() / season2_density["bid_opportunities"].sum()
                    ),
                    "clicks": int(season2_density["clicks"].sum()),
                    "conversions": int(season2_density["conversions"].sum()),
                },
                {
                    "window": "Season three",
                    "role_in_study": "External frozen-policy replay validation",
                    "dates": int(season3_baseline["event_date"].nunique()),
                    "bid_opportunities": int(season3_baseline["opportunities"].sum()),
                    "filled_impressions": int(season3_baseline["observed_filled_impressions"].sum()),
                    "fill_rate": float(
                        season3_baseline["observed_filled_impressions"].sum()
                        / season3_baseline["opportunities"].sum()
                    ),
                    "clicks": int(season3_baseline["retained_clicks"].sum()),
                    "conversions": int(season3_baseline["retained_conversions"].sum()),
                },
            ]
        )
        self.write(data_windows, "paper_table_data_windows.csv", table=True)
        registry = self.read("reserve_policy_registry.csv").copy()
        registry["reader_facing_policy"] = (
            registry["policy_id"]
            .map({PRIORITY_POLICY_ID: PRIORITY_POLICY_LABEL})
            .fillna(registry["policy_id"].str.replace("_", " ").str.title())
        )
        self.write(
            registry[["policy_number", "reader_facing_policy", "policy_id", "policy_family", "description"]],
            "paper_table_policy_set.csv",
            table=True,
        )
        final = self.read("final_policy_recommendation.csv")
        self.write(
            final[
                [
                    "policy_id",
                    "recommended_action",
                    "direct_launch_ready",
                    "why_not_direct_launch",
                    "replay_yield_lift",
                    "p10_crossfit_dr_lift",
                    "primary_validation_design",
                ]
            ],
            "paper_table_recommendation.csv",
            table=True,
        )
        transfer = self.read("season2_vs_season3_policy_transfer.csv")
        self.write(
            transfer[
                [
                    "policy_id",
                    "policy_label",
                    "season2_pct_yield_lift",
                    "season3_pct_yield_lift",
                    "season2_rank",
                    "season3_rank",
                    "yield_lift_transfer_gap",
                    "season3_retained_impression_share",
                    "season3_value_proxy_retention",
                ]
            ].sort_values("season3_rank"),
            "paper_table_season3_transfer.csv",
            table=True,
        )
        ablation = self.read("decision_rule_ablation_summary.csv")
        self.write(
            ablation[
                [
                    "rule_label",
                    "selection_basis",
                    "selected_policy_id",
                    "recommended_action_under_rule",
                    "direct_launch_overclaim",
                    "unresolved_launch_gate_count",
                ]
            ],
            "paper_table_decision_ablation.csv",
            table=True,
        )
        gates = pd.DataFrame(
            [
                {
                    "gate": "Replay evidence",
                    "evidence_used": "Full-panel mechanical yield and retained-impression share",
                    "decision_role": "Screens candidate policies",
                },
                {
                    "gate": "Out-of-time validation",
                    "evidence_used": "Frozen-policy replay on later market window",
                    "decision_role": "Tests whether the offline recommendation transfers without retuning",
                },
                {
                    "gate": "Support and overlap",
                    "evidence_used": "Effective sample size, exact floor support, clipping sensitivity",
                    "decision_role": "Determines whether OPE estimates are credible",
                },
                {
                    "gate": "Guardrails",
                    "evidence_used": "Fill, advertiser value proxy, click/conversion proxy, segment harm",
                    "decision_role": "Prevents single-metric revenue maximization",
                },
                {
                    "gate": "Theory sensitivity",
                    "evidence_used": "Support collapse, break-even bidder response, placebo checks",
                    "decision_role": "Downgrades claims that rely on fragile assumptions",
                },
                {
                    "gate": "Interference-aware validation",
                    "evidence_used": "Shadow logging, switchback detectability, stop rules",
                    "decision_role": "Converts offline priority into an online test plan",
                },
            ]
        )
        self.write(gates, "paper_table_decision_support_gates.csv", table=True)
        theory_empirical = pd.DataFrame(
            [
                {
                    "formal_result": "Replay identification",
                    "empirical_counterpart": "Replay frontier and daily stability",
                    "decision_implication": (
                        "Replay lift is a fixed-behavior mechanical estimand, not a launch causal effect."
                    ),
                },
                {
                    "formal_result": "Break-even response",
                    "empirical_counterpart": "Marketplace-response sensitivity",
                    "decision_implication": "Replay gains must survive plausible bidder, pacing, or budget response.",
                },
                {
                    "formal_result": "Support collapse",
                    "empirical_counterpart": "Weight diagnostics and support curves",
                    "decision_implication": "Thin effective support downgrades attractive point estimates.",
                },
                {
                    "formal_result": "Conservative ranking",
                    "empirical_counterpart": "Lower-tail policy ranking",
                    "decision_implication": "Validation priority uses downside-aware evidence, not mean lift alone.",
                },
                {
                    "formal_result": "Segment non-harm",
                    "empirical_counterpart": "Segment diagnostics",
                    "decision_implication": "Aggregate yield gains do not remove the need for guardrails.",
                },
                {
                    "formal_result": "Clipping diagnostic",
                    "empirical_counterpart": "Weight-tail and estimator-stability checks",
                    "decision_implication": "Sensitivity to arbitrary caps signals weak overlap.",
                },
                {
                    "formal_result": "Interference",
                    "empirical_counterpart": "Interference map and switchback design",
                    "decision_implication": "Offline evidence should become an interference-aware validation plan.",
                },
                {
                    "formal_result": "Claim-preserving action map",
                    "empirical_counterpart": "Decision-rule ablation",
                    "decision_implication": "Integration changes the action and prevents direct-launch overclaim.",
                },
            ]
        )
        self.write(theory_empirical, "paper_table_theory_empirical_link.csv", table=True)
        dss_artifact = pd.DataFrame(
            [
                {
                    "artifact_field": "Decision question",
                    "output": "Should a reserve/floor policy be launched, validated online, held, or redesigned?",
                },
                {
                    "artifact_field": "Priority candidate",
                    "output": (
                        f"{PRIORITY_POLICY_LABEL}, an auditable hybrid rule that raises floors only when "
                        "bid-floor margins are large."
                    ),
                },
                {
                    "artifact_field": "Evidence package",
                    "output": (
                        "Replay frontier, daily stability, season-three validation, nuisance calibration, OPE "
                        "estimator comparison, weight diagnostics, conservative ranking, segment guardrails, "
                        "support sensitivity, bidder-response sensitivity, and ablation."
                    ),
                },
                {
                    "artifact_field": "Blocking gates",
                    "output": (
                        "Production logging propensities are unobserved; bidder response is not observed; "
                        "row-level no-interference is not credible for live launch."
                    ),
                },
                {"artifact_field": "Decision status", "output": "Validation-ready, not launch-ready."},
                {
                    "artifact_field": "Next action",
                    "output": (
                        "Shadow logging for 7--14 days, then a pre-registered exchange-hour switchback with "
                        "hard guardrails and stop rules."
                    ),
                },
            ]
        )
        self.write(dss_artifact, "paper_table_dss_artifact.csv", table=True)
        figure_selection = pd.DataFrame(
            [
                (
                    "Figure 1",
                    "03_marketplace_interference_dag.png",
                    "main_paper",
                    "Decision setting",
                    "Marketplace interference paths.",
                    "Panel (a) in the framework figure.",
                    True,
                    "figures/03_marketplace_interference_dag.png",
                ),
                (
                    "Figure 1",
                    "05_auction_replay_flow.png",
                    "main_paper",
                    "Decision setting",
                    "Auction replay and launch-readiness workflow.",
                    "Panel (b) in the framework figure.",
                    True,
                    "figures/05_auction_replay_flow.png",
                ),
                (
                    "Figure 2",
                    "01_sample_price_distributions.png",
                    "main_paper",
                    "Empirical foundation",
                    "Price landscape for bids, payments, and floors.",
                    "Motivates reserve/floor interventions.",
                    True,
                    "figures/01_sample_price_distributions.png",
                ),
                (
                    "Figure 3",
                    "05_outcome_density_by_day.png",
                    "main_paper",
                    "Empirical foundation",
                    "Season-two outcome density by day.",
                    "Shows the full analysis panel.",
                    True,
                    "figures/05_outcome_density_by_day.png",
                ),
                (
                    "Figure 4",
                    "04_nuisance_model_calibration.png",
                    "main_paper",
                    "Empirical foundation",
                    "Nuisance-model calibration diagnostics.",
                    "Checks probability and regression nuisance components for model-assisted OPE.",
                    True,
                    "figures/04_nuisance_model_calibration.png",
                ),
                (
                    "Figure 5",
                    "06_reserve_policy_tradeoff_frontier.png",
                    "main_paper",
                    "Replay results",
                    "Yield/fill replay frontier.",
                    "Main replay frontier.",
                    True,
                    "figures/06_reserve_policy_tradeoff_frontier.png",
                ),
                (
                    "Figure 6",
                    "06_shortlist_daily_stability.png",
                    "main_paper",
                    "Replay results",
                    "Shortlist daily stability.",
                    "Shows replay result is not one-day driven.",
                    True,
                    "figures/06_shortlist_daily_stability.png",
                ),
                (
                    "Figure 7",
                    "13_season2_vs_season3_transfer.png",
                    "main_paper",
                    "External validation",
                    "Season-two versus season-three transfer.",
                    "Shows frozen-policy out-of-time validation.",
                    True,
                    "figures/13_season2_vs_season3_transfer.png",
                ),
                (
                    "Figure 8",
                    "13_season3_priority_policy_guardrails.png",
                    "main_paper",
                    "External validation",
                    "Season-three guardrails.",
                    "Checks holdout guardrails for the priority policy.",
                    True,
                    "figures/13_season3_priority_policy_guardrails.png",
                ),
                (
                    "Figure 9",
                    "13_season3_daily_priority_validation.png",
                    "main_paper",
                    "External validation",
                    "Season-three daily priority validation.",
                    "Checks holdout result is not one date.",
                    True,
                    "figures/13_season3_daily_priority_validation.png",
                ),
                (
                    "Figure 10",
                    "14_decision_rule_gate_matrix.png",
                    "main_paper",
                    "Ablation",
                    "Evidence gates used by each decision rule.",
                    "Main decision-rule ablation figure.",
                    True,
                    "figures/14_decision_rule_gate_matrix.png",
                ),
                (
                    "Figure 11",
                    "14_decision_rule_unresolved_gates.png",
                    "main_paper",
                    "Ablation",
                    "Unresolved launch gates under simplified rules.",
                    "Shows why simplified rules overclaim launch readiness.",
                    True,
                    "figures/14_decision_rule_unresolved_gates.png",
                ),
                (
                    "Figure 12",
                    "14_decision_rule_bootstrap_selection.png",
                    "main_paper",
                    "Ablation",
                    "Bootstrap selection stability.",
                    "Shows action disagreement is not policy-selection instability.",
                    True,
                    "figures/14_decision_rule_bootstrap_selection.png",
                ),
                (
                    "Figure 13",
                    "08_ope_estimator_comparison.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Estimator comparison.",
                    "Checks estimator agreement for policy ranking.",
                    True,
                    "figures/08_ope_estimator_comparison.png",
                ),
                (
                    "Figure 14",
                    "08_weight_diagnostics.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Importance-weight diagnostics.",
                    "Shows support and tail behavior.",
                    True,
                    "figures/08_weight_diagnostics.png",
                ),
                (
                    "Figure 15",
                    "08_conservative_lower_bound_ranking.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Conservative lower-bound ranking.",
                    "Ranks policies by downside-aware evidence.",
                    True,
                    "figures/08_conservative_lower_bound_ranking.png",
                ),
                (
                    "Figure 16",
                    "10_equilibrium_sensitivity.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Marketplace-response sensitivity.",
                    "Links replay lift to response risk.",
                    True,
                    "figures/10_equilibrium_sensitivity.png",
                ),
                (
                    "Figure 17",
                    "10_support_collapse_curve.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Support-collapse sensitivity.",
                    "Shows lower-bound deterioration under support loss.",
                    True,
                    "figures/10_support_collapse_curve.png",
                ),
                (
                    "Figure 18",
                    "08_segment_heterogeneity.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Segment heterogeneity diagnostics.",
                    "Checks aggregate gains do not hide segment risk.",
                    True,
                    "figures/08_segment_heterogeneity.png",
                ),
                (
                    "Figure 19",
                    "10_theory_guided_verdict.png",
                    "main_paper",
                    "OPE and sensitivity",
                    "Theory-guided verdict.",
                    "Combines replay, conservative evidence, and sensitivity.",
                    True,
                    "figures/10_theory_guided_verdict.png",
                ),
                (
                    "Figure 20",
                    "11_launch_readiness_checklist.png",
                    "main_paper",
                    "Decision recommendation",
                    "Launch-readiness gates.",
                    "Shows validation-ready but not launch-ready status.",
                    True,
                    "figures/11_launch_readiness_checklist.png",
                ),
                (
                    "Figure 21",
                    "09_scorecard_components.png",
                    "main_paper",
                    "Decision recommendation",
                    "Marketplace scorecard components.",
                    "Decomposes the evidence score.",
                    True,
                    "figures/09_scorecard_components.png",
                ),
                (
                    "Figure 22",
                    "11_decision_waterfall.png",
                    "main_paper",
                    "Decision recommendation",
                    "Decision waterfall.",
                    "Panel (a) in the decision-path figure.",
                    True,
                    "figures/11_decision_waterfall.png",
                ),
                (
                    "Figure 22",
                    "11_validation_sequence.png",
                    "main_paper",
                    "Decision recommendation",
                    "Validation sequence.",
                    "Panel (b) in the decision-path figure.",
                    True,
                    "figures/11_validation_sequence.png",
                ),
                (
                    "Figure 23",
                    "07_design_mde_curves.png",
                    "main_paper",
                    "Decision recommendation",
                    "MDE curves for validation designs.",
                    "Supports switchback design choice.",
                    True,
                    "figures/07_design_mde_curves.png",
                ),
            ]
        )
        figure_selection.columns = [
            "figure_number",
            "filename",
            "role",
            "section",
            "caption",
            "why_selected",
            "exists",
            "relative_path",
        ]
        self.write(figure_selection, "final_figure_selection.csv")
        table_selection = pd.DataFrame(
            [
                (
                    "Table 3",
                    "paper_table_decision_support_gates.csv",
                    "tables",
                    "main_paper",
                    "Theory",
                    "Core decision-support gates.",
                    True,
                    "tables/paper_table_decision_support_gates.csv",
                ),
                (
                    "Table 4",
                    "paper_table_data_windows.csv",
                    "tables",
                    "main_paper",
                    "Empirical study",
                    "Discovery and external-validation data windows.",
                    True,
                    "tables/paper_table_data_windows.csv",
                ),
                (
                    "Table 5",
                    "paper_table_policy_set.csv",
                    "tables",
                    "main_paper",
                    "Empirical study",
                    "Full candidate reserve/floor policy set.",
                    True,
                    "tables/paper_table_policy_set.csv",
                ),
                (
                    "Table 6",
                    "paper_table_recommendation.csv",
                    "tables",
                    "main_paper",
                    "Empirical study",
                    "Priority recommendation and blocking launch gates.",
                    True,
                    "tables/paper_table_recommendation.csv",
                ),
                (
                    "Table 7",
                    "paper_table_season3_transfer.csv",
                    "tables",
                    "main_paper",
                    "External validation",
                    "Season-two to season-three frozen-policy transfer.",
                    True,
                    "tables/paper_table_season3_transfer.csv",
                ),
                (
                    "Table 8",
                    "paper_table_decision_ablation.csv",
                    "tables",
                    "main_paper",
                    "Ablation",
                    "Decision-rule comparison.",
                    True,
                    "tables/paper_table_decision_ablation.csv",
                ),
                (
                    "Table 9",
                    "paper_table_theory_empirical_link.csv",
                    "tables",
                    "main_paper",
                    "Discussion",
                    "How formal results govern empirical evidence.",
                    True,
                    "tables/paper_table_theory_empirical_link.csv",
                ),
                (
                    "Table 10",
                    "paper_table_dss_artifact.csv",
                    "tables",
                    "main_paper",
                    "Decision recommendation",
                    "Launch-readiness artifact.",
                    True,
                    "tables/paper_table_dss_artifact.csv",
                ),
            ]
        )
        table_selection.columns = [
            "table_number",
            "filename",
            "source_folder",
            "role",
            "section",
            "caption",
            "exists",
            "relative_path",
        ]
        self.write(table_selection, "final_table_selection.csv")
        self.progress.done("static table and figure-selection catalogs")
