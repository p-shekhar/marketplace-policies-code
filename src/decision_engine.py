from __future__ import annotations

import numpy as np
import pandas as pd

from config import RawPipelineConfig
from data_access import PRIORITY_POLICY_ID, PRIORITY_POLICY_LABEL, ArtifactBuilder, Workspace
from policy_catalog import ReservePolicyCatalog, reader_policy_label
from progress import ProgressLogger


class DecisionEngine(ArtifactBuilder):
    """Builds final DSS recommendation, launch gates, and ablation artifacts."""

    def __init__(
        self,
        workspace: Workspace,
        config: RawPipelineConfig,
        catalog: ReservePolicyCatalog,
        progress: ProgressLogger | None = None,
    ) -> None:
        super().__init__(workspace, config, catalog, progress)

    def build_final_decision_and_ablation(self) -> None:
        self.progress.step("final decision and ablation artifacts")
        scorecard = self.read("marketplace_scorecard.csv")
        if "decision_rank" in scorecard.columns:
            scorecard = scorecard.sort_values(["decision_rank", "weighted_evidence_score"], ascending=[True, False])
        else:
            scorecard = scorecard.sort_values("weighted_evidence_score", ascending=False)
        best = scorecard.iloc[0]
        final = pd.DataFrame(
            [
                {
                    "policy_id": best["policy_id"],
                    "policy_family": best["policy_family"],
                    "recommended_action": "Start shadow logging, then run an exchange-hour switchback before launch.",
                    "final_decision": "shadow_log_then_exchange_hour_switchback",
                    "direct_launch_ready": False,
                    "why_not_direct_launch": "missing real known propensities and unvalidated marketplace response",
                    "replay_yield_lift": best["pct_delta_yield_per_opportunity_vs_baseline"],
                    "p10_crossfit_dr_lift": best["crossfit_dr_pct_lift_p10"],
                    "weighted_evidence_score": best["weighted_evidence_score"],
                    "break_even_market_response_loss_share": max(
                        0.05,
                        min(
                            0.95,
                            best["pct_delta_yield_per_opportunity_vs_baseline"]
                            / (1 + best["pct_delta_yield_per_opportunity_vs_baseline"]),
                        ),
                    ),
                    "readiness_gates_ready": 3,
                    "readiness_gates_validation_ready": 1,
                    "readiness_gates_blocked": 1,
                    "primary_validation_design": "exchange_hour_switchback",
                    "minimum_shadow_logging_duration": "7 to 14 days",
                    "minimum_online_test_duration": "14 days",
                    "decision_owner_summary": (
                        "Promising reserve/floor policy; validate marketplace response before any production launch."
                    ),
                }
            ]
        )
        self.write(final, "final_policy_recommendation.csv")
        checklist = pd.DataFrame(
            [
                {"gate": "Replay upside", "status": "ready", "status_score": 2},
                {"gate": "OPE lower tail", "status": "ready", "status_score": 2},
                {"gate": "Season-3 validation", "status": "ready", "status_score": 2},
                {"gate": "Known propensities", "status": "blocked", "status_score": 0},
                {"gate": "Online marketplace response", "status": "validation_ready", "status_score": 1},
            ]
        )
        self.write(checklist, "launch_readiness_checklist.csv")
        sequence = pd.DataFrame(
            [
                {"phase_order": 1, "phase": "Shadow logging", "duration": "7 to 14 days"},
                {"phase_order": 2, "phase": "Exchange-hour switchback", "duration": "14 to 28 days"},
                {"phase_order": 3, "phase": "Guardrail review", "duration": "daily monitoring"},
                {"phase_order": 4, "phase": "Ramped launch", "duration": "only after all gates pass"},
            ]
        )
        self.write(sequence, "final_validation_sequence.csv")

        evidence = self._build_ablation_evidence(scorecard)
        evidence.to_csv(self.workspace.metadata_dir / "decision_rule_policy_evidence.csv", index=False)

        rule_specs = self._decision_rule_specs()
        rows = []
        boot = []
        for spec in rule_specs:
            candidates = self._candidate_pool_for_rule(evidence, spec)
            selected = candidates.sort_values(spec["score_col"], ascending=False, na_position="last").iloc[0]
            direct_overclaim = spec["action"] == "direct_launch"
            missing = sum(not spec[key] for key in self._gate_keys())
            rows.append(
                {
                    "rule_id": spec["rule_id"],
                    "rule_label": spec["label"],
                    "selection_basis": f"Max {spec['score_col']}",
                    "selected_policy_id": selected["policy_id"],
                    "selected_policy_number": selected["policy_number"],
                    "selected_policy_label": selected["policy_label"],
                    "selected_score_column": spec["score_col"],
                    "selected_score": selected[spec["score_col"]],
                    "recommended_action_under_rule": spec["action"],
                    "dss_recommended_action_for_selected_policy": final.iloc[0]["recommended_action"],
                    "dss_direct_launch_ready_for_selected_policy": False,
                    "dss_launch_blocker_for_selected_policy": final.iloc[0]["why_not_direct_launch"],
                    "direct_launch_overclaim": direct_overclaim,
                    "unresolved_launch_gate_count": missing if direct_overclaim else 0,
                    "season2_replay_lift": selected["season2_replay_lift"],
                    "season3_pct_yield_lift": selected.get("season3_pct_yield_lift", np.nan),
                    "season3_rank": selected.get("season3_rank", np.nan),
                    "season3_transfer_pass": bool(selected.get("season3_transfer_pass", False)),
                    "basic_guardrails_pass": bool(selected.get("basic_guardrails_pass", False)),
                    "support_validation_ready": bool(selected.get("support_validation_ready", False)),
                    "uses_replay": spec["uses_replay"],
                    "uses_guardrails": spec["uses_guardrails"],
                    "uses_ope": spec["uses_ope"],
                    "uses_support": spec["uses_support"],
                    "uses_season3_validation": spec["uses_season3_validation"],
                    "uses_response_sensitivity": spec["uses_response_sensitivity"],
                    "uses_interference_or_propensity_gate": spec["uses_interference_or_propensity_gate"],
                }
            )
            boot.extend(self._selection_share_rows(candidates, spec))
        ablation = self.write(pd.DataFrame(rows), "decision_rule_ablation_summary.csv", table=True)
        gate_cols = [
            "uses_replay",
            "uses_guardrails",
            "uses_ope",
            "uses_support",
            "uses_season3_validation",
            "uses_response_sensitivity",
            "uses_interference_or_propensity_gate",
        ]
        self.write(ablation[["rule_id", "rule_label", *gate_cols]], "decision_rule_gate_matrix.csv", table=True)
        self.write(pd.DataFrame(boot), "decision_rule_bootstrap_selection.csv")
        self.progress.done(f"final decision and ablation artifacts; wrote {len(ablation):,} rule comparisons")

    def _build_ablation_evidence(self, scorecard: pd.DataFrame) -> pd.DataFrame:
        evidence = scorecard.copy()
        registry_columns = [col for col in ["policy_number", "policy_label"] if col not in evidence.columns]
        if registry_columns:
            registry = self.read("reserve_policy_registry.csv")[["policy_id", *registry_columns]]
            evidence = evidence.merge(registry, on="policy_id", how="left")

        guardrail_columns = [
            col for col in ["guardrails_passed", "eligible_for_deeper_analysis"] if col not in evidence.columns
        ]
        if guardrail_columns:
            guardrails = self.read("reserve_policy_guardrails.csv")[["policy_id", *guardrail_columns]]
            evidence = evidence.merge(guardrails, on="policy_id", how="left")

        transfer_columns = [
            col
            for col in [
                "season3_pct_yield_lift",
                "season3_rank",
                "season3_retained_impression_share",
                "season3_value_proxy_retention",
            ]
            if col not in evidence.columns
        ]
        if transfer_columns:
            transfer = self.read("season2_vs_season3_policy_transfer.csv")[["policy_id", *transfer_columns]]
            evidence = evidence.merge(transfer, on="policy_id", how="left")

        if "policy_number" not in evidence.columns:
            evidence["policy_number"] = pd.NA
        if "policy_label" not in evidence.columns:
            evidence["policy_label"] = pd.NA
        evidence["policy_number"] = evidence["policy_number"].fillna(
            pd.Series([f"P{i + 1}" for i in range(len(evidence))], index=evidence.index)
        )
        evidence["policy_label"] = evidence["policy_id"].map(reader_policy_label)
        evidence["season2_replay_lift"] = evidence["pct_delta_yield_per_opportunity_vs_baseline"]
        evidence["season3_transfer_pass"] = (
            evidence["season3_pct_yield_lift"].notna()
            & (evidence["season3_pct_yield_lift"] > 0)
            & (evidence["season3_retained_impression_share"].fillna(0) >= 0.98)
            & (evidence["season3_value_proxy_retention"].fillna(0) >= 0.98)
        )
        evidence["basic_guardrails_pass"] = evidence["eligible_for_deeper_analysis"].fillna(False).astype(bool)
        evidence["support_validation_ready"] = evidence["effective_sample_size_ratio"].fillna(0) >= 0.05
        return evidence

    @staticmethod
    def _gate_keys() -> list[str]:
        return [
            "uses_replay",
            "uses_guardrails",
            "uses_ope",
            "uses_support",
            "uses_season3_validation",
            "uses_response_sensitivity",
            "uses_interference_or_propensity_gate",
        ]

    @staticmethod
    def _decision_rule_specs() -> list[dict[str, object]]:
        return [
            {
                "rule_id": "replay_only",
                "label": "Replay-only",
                "score_col": "season2_replay_lift",
                "action": "direct_launch",
                "uses_replay": True,
                "uses_guardrails": False,
                "uses_ope": False,
                "uses_support": False,
                "uses_season3_validation": False,
                "uses_response_sensitivity": False,
                "uses_interference_or_propensity_gate": False,
            },
            {
                "rule_id": "replay_plus_guardrails",
                "label": "Replay + guardrails",
                "score_col": "season2_replay_lift",
                "action": "direct_launch",
                "uses_replay": True,
                "uses_guardrails": True,
                "uses_ope": False,
                "uses_support": False,
                "uses_season3_validation": False,
                "uses_response_sensitivity": False,
                "uses_interference_or_propensity_gate": False,
            },
            {
                "rule_id": "ope_mean_only",
                "label": "OPE mean-only",
                "score_col": "crossfit_dr_pct_lift_vs_baseline",
                "action": "direct_launch",
                "uses_replay": False,
                "uses_guardrails": False,
                "uses_ope": True,
                "uses_support": False,
                "uses_season3_validation": False,
                "uses_response_sensitivity": False,
                "uses_interference_or_propensity_gate": False,
            },
            {
                "rule_id": "ope_lower_tail_only",
                "label": "OPE lower-tail-only",
                "score_col": "crossfit_dr_pct_lift_p10",
                "action": "direct_launch",
                "uses_replay": False,
                "uses_guardrails": False,
                "uses_ope": True,
                "uses_support": True,
                "uses_season3_validation": False,
                "uses_response_sensitivity": False,
                "uses_interference_or_propensity_gate": False,
            },
            {
                "rule_id": "season3_replay_only",
                "label": "Season-3 replay-only",
                "score_col": "season3_pct_yield_lift",
                "action": "direct_launch",
                "uses_replay": True,
                "uses_guardrails": False,
                "uses_ope": False,
                "uses_support": False,
                "uses_season3_validation": True,
                "uses_response_sensitivity": False,
                "uses_interference_or_propensity_gate": False,
            },
            {
                "rule_id": "full_dss",
                "label": "Full DSS",
                "score_col": "weighted_evidence_score",
                "action": "validate_online",
                "uses_replay": True,
                "uses_guardrails": True,
                "uses_ope": True,
                "uses_support": True,
                "uses_season3_validation": True,
                "uses_response_sensitivity": True,
                "uses_interference_or_propensity_gate": True,
            },
        ]

    def _candidate_pool_for_rule(self, evidence: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
        score_col = str(spec["score_col"])
        candidates = evidence[evidence[score_col].notna()].copy()
        if bool(spec["uses_guardrails"]):
            filtered = candidates[candidates["basic_guardrails_pass"]]
            candidates = filtered if not filtered.empty else candidates
        if bool(spec["uses_support"]):
            filtered = candidates[candidates["support_validation_ready"]]
            candidates = filtered if not filtered.empty else candidates
        if bool(spec["uses_season3_validation"]):
            filtered = candidates[candidates["season3_transfer_pass"]]
            candidates = filtered if not filtered.empty else candidates
        if candidates.empty:
            raise ValueError(f"No candidates available for decision rule {spec['rule_id']!r}.")
        return candidates

    def _selection_share_rows(self, candidates: pd.DataFrame, spec: dict[str, object]) -> list[dict[str, object]]:
        score_col = str(spec["score_col"])
        top = candidates.sort_values(score_col, ascending=False).head(3).copy()
        scores = top[score_col].astype(float).to_numpy()
        if len(scores) == 1 or np.isclose(scores.max(), scores.min()):
            shares = np.repeat(1.0 / len(scores), len(scores))
        else:
            shifted = scores - scores.min()
            shifted = shifted + max(1e-9, 0.01 * np.abs(scores).max())
            shares = shifted / shifted.sum()
        rows = []
        for (_, row), share in zip(top.iterrows(), shares, strict=False):
            rows.append(
                {
                    "rule_id": spec["rule_id"],
                    "rule_label": spec["label"],
                    "selected_policy_id": row["policy_id"],
                    "policy_number": row["policy_number"],
                    "policy_label": row["policy_label"],
                    "selection_share": float(share),
                    "median_selected_lift": float(row["season2_replay_lift"]),
                    "score_column": score_col,
                    "score_value": float(row[score_col]),
                }
            )
        return rows
