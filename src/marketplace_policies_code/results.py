from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marketplace_policies_code.config import PaperConfig
from marketplace_policies_code.repository import ArtifactRepository


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    observed: object
    expected: object
    passed: bool
    tolerance: float | None = None


class ResultValidator:
    """Validate the headline numerical and decision claims in the paper."""

    def __init__(self, repository: ArtifactRepository, config: PaperConfig | None = None) -> None:
        self.repository = repository
        self.config = config or PaperConfig()

    @staticmethod
    def _close(observed: float, expected: float, tolerance: float = 5e-4) -> bool:
        return abs(float(observed) - float(expected)) <= tolerance

    def run(self) -> list[ClaimCheck]:
        final = self.repository.read_csv("final_policy_recommendation.csv").iloc[0]
        season3 = self.repository.read_csv("season3_priority_policy_validation.csv").iloc[0]
        ablation = self.repository.read_csv("decision_rule_ablation_summary.csv")
        full_dss = ablation.query("rule_id == 'full_dss'").iloc[0]
        simplified = ablation.query("rule_id != 'full_dss'")

        checks = [
            ClaimCheck("priority_policy_id", final.policy_id, self.config.priority_policy_id, final.policy_id == self.config.priority_policy_id),
            ClaimCheck("priority_policy_label", season3.policy_label, self.config.priority_policy_label, season3.policy_label == self.config.priority_policy_label),
            ClaimCheck("season2_replay_lift", final.replay_yield_lift, 0.4766038582601403, self._close(final.replay_yield_lift, 0.4766038582601403), 5e-4),
            ClaimCheck("dr_lower_tail_lift", final.p10_crossfit_dr_lift, 0.4582525058048958, self._close(final.p10_crossfit_dr_lift, 0.4582525058048958), 5e-4),
            ClaimCheck(
                "break_even_response_loss",
                final.break_even_market_response_loss_share,
                0.3227702918382695,
                self._close(final.break_even_market_response_loss_share, 0.3227702918382695),
                5e-4,
            ),
            ClaimCheck("season3_holdout_lift", season3.season3_pct_yield_lift, 0.4387225351117065, self._close(season3.season3_pct_yield_lift, 0.4387225351117065), 5e-4),
            ClaimCheck("season3_rank", int(season3.season3_rank), 1, int(season3.season3_rank) == 1),
            ClaimCheck(
                "simplified_rules_select_priority",
                bool(simplified["selected_policy_id"].eq(self.config.priority_policy_id).all()),
                True,
                bool(simplified["selected_policy_id"].eq(self.config.priority_policy_id).all()),
            ),
            ClaimCheck(
                "simplified_rules_overclaim_direct_launch",
                int(simplified["direct_launch_overclaim"].sum()),
                int(simplified.shape[0]),
                int(simplified["direct_launch_overclaim"].sum()) == int(simplified.shape[0]),
            ),
            ClaimCheck(
                "full_dss_recommends_validation",
                full_dss.recommended_action_under_rule,
                "validate_online",
                full_dss.recommended_action_under_rule == "validate_online",
            ),
        ]
        return checks

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.__dict__ for check in self.run()])

    def write_report(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        path = output_dir / "claim_checks.csv"
        frame.to_csv(path, index=False)
        markdown = output_dir / "claim_checks.md"
        markdown.write_text(self._markdown_report(frame), encoding="utf-8")
        return path

    @staticmethod
    def _markdown_report(frame: pd.DataFrame) -> str:
        lines = ["# Claim Checks", ""]
        for row in frame.itertuples(index=False):
            mark = "PASS" if row.passed else "FAIL"
            lines.append(f"- **{mark}** `{row.name}`: observed `{row.observed}`, expected `{row.expected}`")
        lines.append("")
        return "\n".join(lines)

    def assert_all_pass(self) -> None:
        failed = [check for check in self.run() if not check.passed]
        if failed:
            names = ", ".join(check.name for check in failed)
            raise AssertionError(f"Claim checks failed: {names}")

