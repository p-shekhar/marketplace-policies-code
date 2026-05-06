from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import textwrap

import graphviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from marketplace_policies_code.config import PaperConfig
from marketplace_policies_code.repository import ArtifactRepository


PALETTE = {
    "navy": "#1f4e79",
    "blue": "#3b82f6",
    "teal": "#2a9d8f",
    "green": "#4c956c",
    "gold": "#f2b134",
    "coral": "#e76f51",
    "red": "#c2410c",
    "purple": "#7c3aed",
    "slate": "#475569",
    "light_blue": "#dbeafe",
    "light_teal": "#ccfbf1",
    "light_green": "#dcfce7",
    "light_gold": "#fef3c7",
    "light_red": "#fee2e2",
    "light_purple": "#ede9fe",
    "light_slate": "#f1f5f9",
    "ink": "#172033",
    "grid": "#d9e2ec",
}


POLICY_LABELS = {
    "logged_floor_status_quo": "Logged\nstatus quo",
    "hybrid_q75_if_gap_100": "Q75 Margin-Gated\nFloor",
    "min_positive_floor_q75": "Positive floors\nto Q75",
    "hybrid_q50_if_gap_50": "Q50 Margin-Gated\nFloor",
    "zero_and_low_floor_to_q50": "Low floors\nto Q50",
    "margin_gap_100_add_20": "Gap >= 100\nadd 20",
    "add_20_all_floors": "All floors\nadd 20",
}


def policy_label(policy_id: str, newline: bool = False) -> str:
    label = POLICY_LABELS.get(policy_id, policy_id.replace("_", " ").title())
    if not newline:
        label = label.replace("\n", " ")
    return label


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


@dataclass
class FigureRenderer:
    """Object-oriented renderer for all manuscript figures.

    The manuscript captions carry the figure-level title, so all matplotlib
    figures are saved without embedded titles. Graphviz flowcharts are left as
    diagrams because their text is semantic content, not redundant titles.
    """

    repository: ArtifactRepository
    output_dir: Path
    config: PaperConfig = PaperConfig()

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.set_style()

    @staticmethod
    def set_style() -> None:
        sns.set_theme(style="whitegrid", context="paper")
        mpl.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9.5,
                "axes.labelsize": 10,
                "axes.labelcolor": PALETTE["ink"],
                "text.color": PALETTE["ink"],
                "xtick.color": PALETTE["ink"],
                "ytick.color": PALETTE["ink"],
                "axes.edgecolor": PALETTE["grid"],
                "grid.color": PALETTE["grid"],
                "grid.linewidth": 0.7,
                "legend.frameon": False,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 320,
                "figure.dpi": 140,
            }
        )

    @staticmethod
    def clean_axis(ax: plt.Axes) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["grid"])
        ax.spines["bottom"].set_color(PALETTE["grid"])
        ax.grid(axis="y", alpha=0.85)
        ax.grid(axis="x", alpha=0.15)

    def save(self, fig: plt.Figure, filename: str) -> Path:
        fig.suptitle("")
        for ax in fig.axes:
            ax.set_title("")
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=320)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return path

    def graph_attrs(self, rankdir: str = "LR") -> dict[str, str]:
        return {
            "rankdir": rankdir,
            "bgcolor": "white",
            "pad": "0.35",
            "nodesep": "0.48",
            "ranksep": "0.62",
            "splines": "spline",
            "forcelabels": "true",
            "outputorder": "edgesfirst",
            "fontname": "DejaVu Sans",
            "dpi": "300",
        }

    @staticmethod
    def node_attrs() -> dict[str, str]:
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "DejaVu Sans",
            "fontsize": "17",
            "margin": "0.16,0.10",
            "penwidth": "1.4",
            "color": PALETTE["slate"],
            "fontcolor": PALETTE["ink"],
        }

    @staticmethod
    def edge_attrs() -> dict[str, str]:
        return {
            "fontname": "DejaVu Sans",
            "fontsize": "16",
            "color": PALETTE["slate"],
            "fontcolor": PALETTE["slate"],
            "arrowsize": "0.75",
            "penwidth": "1.5",
        }

    @staticmethod
    def edge_text(label: str, down_shift: int = 3) -> dict[str, str]:
        top_spacer = f"<TR><TD COLSPAN=\"2\" HEIGHT=\"{down_shift}\"> </TD></TR>" if down_shift > 0 else ""
        return {
            "xlabel": (
                "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLPADDING=\"0\">"
                f"{top_spacer}<TR>"
                f"<TD><FONT FACE=\"DejaVu Sans\" POINT-SIZE=\"16\" COLOR=\"{PALETTE['slate']}\">{escape(label)}</FONT></TD>"
                "<TD WIDTH=\"18\"> </TD></TR></TABLE>>"
            )
        }

    def arrow(self, dot: graphviz.Digraph, start: str, end: str, label: str, color: str | None = None) -> None:
        dot.edge(
            start,
            end,
            color=color or PALETTE["slate"],
            fontcolor=PALETTE["slate"],
            arrowsize="0.78",
            penwidth="1.6",
            **self.edge_text(label),
        )

    def render_graph(self, dot: graphviz.Digraph, filename: str) -> Path:
        stem = filename.replace(".png", "")
        for fmt in ["png", "pdf", "svg"]:
            dot.format = fmt
            dot.render(str(self.output_dir / stem), cleanup=True)
        return self.output_dir / filename

    def render_interference_dag(self) -> Path:
        dot = graphviz.Digraph("marketplace_interference")
        dot.attr(**self.graph_attrs("TB"))
        dot.attr("node", **self.node_attrs())
        dot.attr("edge", **self.edge_attrs())
        dot.node("policy", "Candidate\nfloor policy", fillcolor=PALETTE["light_blue"])
        dot.node("replay", "Static replay\nfixed bids", fillcolor=PALETTE["light_gold"])
        dot.node("state", "Shared marketplace\nstate", fillcolor=PALETTE["light_slate"])
        dot.node("mechanical", "Mechanical value\nand guardrails", fillcolor=PALETTE["light_green"])
        dot.node("live", "Live marketplace\neffect", fillcolor=PALETTE["light_red"])
        dot.node("validation", "Interference-aware\nvalidation design", fillcolor=PALETTE["light_purple"])
        dot.edge("policy", "replay", label="observed bids")
        dot.edge("replay", "mechanical", label="identifies")
        dot.edge("policy", "state", label="budget and pacing", style="dashed", color=PALETTE["red"])
        dot.edge("state", "live", label="bidder response", style="dashed", color=PALETTE["red"])
        dot.edge("state", "replay", label="not identified offline", style="dashed")
        dot.edge("mechanical", "validation", label="screen policy")
        dot.edge("live", "validation", label="test online", color=PALETTE["red"], style="dashed")
        return self.render_graph(dot, "03_marketplace_interference_dag.png")

    def render_replay_flow(self) -> Path:
        dot = graphviz.Digraph("auction_replay_flow")
        dot.attr(**self.graph_attrs("TB"))
        dot.attr(nodesep="0.82")
        dot.attr("node", **self.node_attrs())
        dot.attr("edge", **self.edge_attrs())
        nodes = [
            ("data", "Data contract\nfield audit + full panel", PALETTE["light_blue"]),
            ("replay", "Replay engine\ncandidate floors + fixed bids", PALETTE["light_gold"]),
            ("ope", "Evidence layer\nOPE + support + guardrails", PALETTE["light_teal"]),
            ("decision", "Decision artifact\nscorecard + validation plan", PALETTE["light_purple"]),
        ]
        for node, label, color in nodes:
            dot.node(node, label, fillcolor=color)
        self.arrow(dot, "data", "replay", "contexts, bids, floors")
        self.arrow(dot, "replay", "ope", "shortlist and estimates")
        self.arrow(dot, "ope", "decision", "launch-readiness gates")
        return self.render_graph(dot, "05_auction_replay_flow.png")

    def render_validation_sequence(self) -> Path:
        seq = self.repository.read_csv("final_validation_sequence.csv").sort_values("phase_order")
        dot = graphviz.Digraph("validation_sequence")
        dot.attr(**self.graph_attrs("TB"))
        dot.attr(nodesep="0.82")
        dot.attr("node", **self.node_attrs())
        dot.attr("edge", **self.edge_attrs())
        previous = None
        colors = [PALETTE["light_blue"], PALETTE["light_teal"], PALETTE["light_gold"], PALETTE["light_green"]]
        for idx, row in enumerate(seq.itertuples(index=False)):
            node = f"phase_{int(row.phase_order)}"
            phase = "\n".join(textwrap.wrap(str(row.phase), width=28))
            duration = "\n".join(textwrap.wrap(str(row.duration), width=30))
            dot.node(node, f"{int(row.phase_order)}. {phase}\n\n{duration}", fillcolor=colors[idx % len(colors)])
            if previous:
                self.arrow(dot, previous, node, "go/no-go")
            previous = node
        dot.node("launch", "Launch only if\nconservative effect > 0\nand all guardrails pass", fillcolor=PALETTE["light_red"])
        self.arrow(dot, str(previous), "launch", "ramp decision", color=PALETTE["red"])
        return self.render_graph(dot, "11_validation_sequence.png")

    def render_decision_waterfall(self) -> Path:
        rec = self.repository.read_csv("final_policy_recommendation.csv").iloc[0]
        dot = graphviz.Digraph("decision_waterfall")
        dot.attr(**self.graph_attrs("TB"))
        dot.attr(nodesep="0.82")
        dot.attr("node", **self.node_attrs())
        dot.attr("edge", **self.edge_attrs())
        dot.node("candidate", f"Priority candidate\n{policy_label(rec.policy_id, newline=True)}", fillcolor=PALETTE["light_blue"])
        dot.node("evidence", f"Offline evidence\nReplay {pct(rec.replay_yield_lift)}\nP10 DR {pct(rec.p10_crossfit_dr_lift)}", fillcolor=PALETTE["light_green"])
        dot.node("stress", f"Stress tests\nBreak-even response\n{pct(rec.break_even_market_response_loss_share)}", fillcolor=PALETTE["light_teal"])
        dot.node("blocker", "Direct launch blocked\nNo real propensities\nNo live response estimate", fillcolor=PALETTE["light_red"])
        dot.node("action", "Next action\nShadow logging +\nexchange-hour switchback", fillcolor=PALETTE["light_purple"])
        self.arrow(dot, "candidate", "evidence", "screen")
        self.arrow(dot, "evidence", "stress", "survives")
        self.arrow(dot, "stress", "blocker", "not enough for launch", color=PALETTE["red"])
        self.arrow(dot, "blocker", "action", "next evidence")
        return self.render_graph(dot, "11_decision_waterfall.png")

    def plot_price_distributions(self) -> Path:
        df = self.repository.read_parquet(
            "ipinyou_nuisance_prototype_opportunity_panel.parquet",
            columns=["bid_price", "pay_price", "slot_floor_price"],
        )
        df = df.sample(n=min(120_000, len(df)), random_state=42)
        long = []
        for col, label in [("bid_price", "Bid price"), ("slot_floor_price", "Floor price"), ("pay_price", "Pay price")]:
            values = df[col].dropna()
            values = values[(values >= 0) & (values <= values.quantile(0.995))]
            long.append(pd.DataFrame({"price": values, "series": label}))
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        sns.kdeplot(data=pd.concat(long, ignore_index=True), x="price", hue="series", common_norm=False, linewidth=2.0, ax=ax)
        ax.set_xlabel("Price units, truncated at 99.5th percentile")
        ax.set_ylabel("Density")
        self.clean_axis(ax)
        return self.save(fig, "01_sample_price_distributions.png")

    def plot_outcome_density(self) -> Path:
        df = self.repository.read_csv("season2_outcome_density.csv")
        df["event_date"] = pd.to_datetime(df["event_date"])
        fig, ax1 = plt.subplots(figsize=(8.4, 4.2))
        ax1.bar(df["event_date"], df["bid_opportunities"] / 1e6, color=PALETTE["light_blue"], edgecolor=PALETTE["navy"], label="Bid opportunities")
        ax1.bar(df["event_date"], df["filled"] / 1e6, color=PALETTE["teal"], alpha=0.8, label="Filled impressions")
        ax1.set_ylabel("Rows, millions")
        ax1.set_xlabel("Event date")
        ax2 = ax1.twinx()
        ax2.plot(df["event_date"], df["fill_rate"] * 100, color=PALETTE["red"], marker="o", linewidth=2.2, label="Fill rate")
        ax2.set_ylabel("Fill rate (%)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        self.clean_axis(ax1)
        ax2.spines["top"].set_visible(False)
        return self.save(fig, "05_outcome_density_by_day.png")

    def plot_calibration(self) -> Path:
        df = self.repository.read_csv("prototype_calibration_summary.csv").copy()
        df["bin"] = np.arange(1, len(df) + 1)
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.plot(df["bin"], df["predicted_rate"], marker="o", linewidth=2.2, label="Predicted", color=PALETTE["navy"])
        ax.plot(df["bin"], df["observed_rate"], marker="s", linewidth=2.2, label="Observed", color=PALETTE["coral"])
        ax.set_xlabel("Predicted-probability bin")
        ax.set_ylabel("Rate")
        ax.legend(loc="best")
        self.clean_axis(ax)
        return self.save(fig, "04_nuisance_model_calibration.png")

    def plot_frontier(self) -> Path:
        df = self.repository.read_csv("marketplace_scorecard.csv").sort_values("weighted_evidence_score", ascending=False)
        fig, ax = plt.subplots(figsize=(8.1, 5.0))
        colors = np.where(df["policy_id"].eq(self.config.priority_policy_id), PALETTE["red"], PALETTE["blue"])
        sizes = 100 + 90 * df["weighted_evidence_score"]
        ax.scatter(df["floor_changed_share"] * 100, df["pct_delta_yield_per_opportunity_vs_baseline"] * 100, s=sizes, c=colors, edgecolors=PALETTE["ink"], linewidths=0.8, alpha=0.92)
        for policy_id, offset in {
            self.config.priority_policy_id: (-72, 8),
            "min_positive_floor_q75": (8, -5),
            "hybrid_q50_if_gap_50": (8, 8),
            "add_20_all_floors": (-84, -16),
        }.items():
            row = df[df["policy_id"].eq(policy_id)]
            if not row.empty:
                point = row.iloc[0]
                ax.annotate(policy_label(policy_id), (point.floor_changed_share * 100, point.pct_delta_yield_per_opportunity_vs_baseline * 100), xytext=offset, textcoords="offset points", fontsize=8.3, ha="left" if offset[0] >= 0 else "right")
        ax.axhline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlabel("Floor-changed opportunity share (%)")
        ax.set_ylabel("Yield lift vs. baseline (%)")
        self.clean_axis(ax)
        return self.save(fig, "06_reserve_policy_tradeoff_frontier.png")

    def plot_daily_stability(self) -> Path:
        daily = self.repository.read_csv("reserve_policy_daily_effects.csv")
        shortlist = self.repository.read_csv("marketplace_scorecard.csv").sort_values("weighted_evidence_score", ascending=False)["policy_id"].head(4)
        daily = daily[daily["policy_id"].isin(shortlist)].copy()
        daily["event_date"] = pd.to_datetime(daily["event_date"])
        daily["policy"] = daily["policy_id"].map(policy_label)
        fig, ax = plt.subplots(figsize=(9.2, 4.1))
        sns.lineplot(data=daily, x="event_date", y=daily["pct_delta_yield_per_opportunity_vs_daily_baseline"] * 100, hue="policy", marker="o", linewidth=2, ax=ax)
        ax.axhline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlabel("Event date")
        ax.set_ylabel("Daily yield lift (%)")
        ax.legend(title=None, ncol=2, loc="upper left")
        self.clean_axis(ax)
        return self.save(fig, "06_shortlist_daily_stability.png")

    def plot_ope_comparison(self) -> Path:
        df = self.repository.read_csv("conservative_ranking_comparison.csv").sort_values("weighted_evidence_score", ascending=True)
        y = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.barh(y - 0.18, df["pct_delta_yield_per_opportunity_vs_baseline"] * 100, height=0.32, color=PALETTE["light_blue"], edgecolor=PALETTE["navy"], label="Replay mean")
        ax.barh(y + 0.18, df["crossfit_dr_pct_lift_p10"] * 100, height=0.32, color=PALETTE["teal"], label="DR p10 lower tail")
        ax.set_yticks(y)
        ax.set_yticklabels([policy_label(p) for p in df["policy_id"]])
        ax.set_xlabel("Lift vs. baseline (%)")
        ax.legend(loc="lower right")
        self.clean_axis(ax)
        return self.save(fig, "08_ope_estimator_comparison.png")

    def plot_weight_diagnostics(self) -> Path:
        df = self.repository.read_csv("ope_weight_diagnostics.csv")
        df = df[df["policy_id"].ne("logged_floor_status_quo")].sort_values("effective_sample_size_ratio")
        labels = [policy_label(p) for p in df["policy_id"]]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
        axes[0].barh(labels, df["effective_sample_size_ratio"] * 100, color=PALETTE["teal"], edgecolor=PALETTE["ink"], linewidth=0.7)
        axes[0].set_xlabel("ESS ratio (%)")
        axes[1].barh(labels, df["realized_weight_p99"], color=PALETTE["gold"], edgecolor=PALETTE["ink"], linewidth=0.7)
        axes[1].axvline(10, color=PALETTE["red"], linestyle="--", linewidth=1.4)
        axes[1].text(10.15, -0.45, "weight 10", color=PALETTE["red"], fontsize=8.2, va="center")
        axes[1].set_xlabel("99th percentile realized weight")
        for ax in axes:
            self.clean_axis(ax)
        return self.save(fig, "08_weight_diagnostics.png")

    def plot_conservative_ranking(self) -> Path:
        df = self.repository.read_csv("advanced_policy_conservative_ranking.csv")
        df = df[df["policy_id"].ne("logged_floor_status_quo")].sort_values("crossfit_dr_pct_lift_p10")
        y = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.hlines(y, df["crossfit_dr_pct_lift_p05"] * 100, df["crossfit_dr_pct_lift_p90"] * 100, color=PALETTE["grid"], linewidth=7)
        ax.scatter(df["crossfit_dr_pct_lift_p50"] * 100, y, color=PALETTE["navy"], s=60, zorder=3, label="Median")
        ax.scatter(df["crossfit_dr_pct_lift_p10"] * 100, y, color=PALETTE["red"], s=50, zorder=3, label="P10 lower tail")
        ax.set_yticks(y)
        ax.set_yticklabels([policy_label(p) for p in df["policy_id"]])
        ax.axvline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlabel("Estimated lift vs. baseline (%)")
        ax.legend(loc="lower right")
        self.clean_axis(ax)
        return self.save(fig, "08_conservative_lower_bound_ranking.png")

    def plot_equilibrium_sensitivity(self) -> Path:
        df = self.repository.read_csv("equilibrium_sensitivity.csv")
        keep = [self.config.priority_policy_id, "min_positive_floor_q75", "hybrid_q50_if_gap_50"]
        df = df[df["policy_id"].isin(keep)].copy()
        df["policy"] = df["policy_id"].map(policy_label)
        fig, ax = plt.subplots(figsize=(7.7, 4.6))
        sns.lineplot(data=df, x="response_loss_share", y=df["adjusted_pct_lift_vs_baseline"] * 100, hue="policy", linewidth=2.2, ax=ax)
        ax.axhline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlabel("Adverse response loss share")
        ax.set_ylabel("Adjusted lift vs. baseline (%)")
        ax.xaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
        ax.legend(title=None)
        self.clean_axis(ax)
        return self.save(fig, "10_equilibrium_sensitivity.png")

    def plot_support_collapse(self) -> Path:
        df = self.repository.read_csv("support_collapse_curve.csv")
        keep = [self.config.priority_policy_id, "min_positive_floor_q75", "hybrid_q50_if_gap_50"]
        df = df[df["policy_id"].isin(keep)].copy()
        df["policy"] = df["policy_id"].map(policy_label)
        fig, ax = plt.subplots(figsize=(7.7, 4.6))
        sns.lineplot(data=df, x="support_scale", y=df["support_adjusted_p10_lift"] * 100, hue="policy", marker="o", linewidth=2.2, ax=ax)
        ax.axhline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlabel("Effective support scale")
        ax.set_ylabel("Support-adjusted p10 lift (%)")
        ax.xaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
        ax.legend(title=None)
        self.clean_axis(ax)
        return self.save(fig, "10_support_collapse_curve.png")

    def plot_segment_heterogeneity(self) -> Path:
        df = self.repository.read_csv("advanced_policy_heterogeneity.csv")
        plot_df = df[df["policy_id"].eq(self.config.priority_policy_id)].sort_values("segment_rows", ascending=False).head(18).copy()
        plot_df["segment"] = plot_df["segment_type"] + "=" + plot_df["segment_value"].astype(str)
        plot_df = plot_df.sort_values("segment_dr_lift")
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        colors = np.where(plot_df["segment_dr_lift"] >= 0, PALETTE["teal"], PALETTE["coral"])
        ax.barh(plot_df["segment"], plot_df["segment_dr_lift"] * 100, color=colors, edgecolor=PALETTE["ink"], linewidth=0.5)
        ax.axvline(0, color=PALETTE["slate"], linewidth=1)
        ax.set_xlabel("Segment DR lift (%)")
        ax.set_ylabel("Largest observed segments")
        self.clean_axis(ax)
        return self.save(fig, "08_segment_heterogeneity.png")

    def plot_theory_verdict(self) -> Path:
        row = self.repository.read_csv("theory_guided_sensitivity_verdict.csv").iloc[0]
        metrics = pd.DataFrame(
            {
                "metric": ["Replay lift", "P10 DR lift", "Break-even response", "Tests passed"],
                "value": [row.top_policy_replay_lift * 100, row.top_policy_p10_dr_lift * 100, row.top_policy_break_even_response_loss * 100, row.tests_passed],
                "display": [
                    f"{row.top_policy_replay_lift*100:.1f}%",
                    f"{row.top_policy_p10_dr_lift*100:.1f}%",
                    f"{row.top_policy_break_even_response_loss*100:.1f}%",
                    f"{int(row.tests_passed)} / {int(row.tests_passed + row.tests_review)}",
                ],
            }
        )
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        ax.barh(metrics["metric"], metrics["value"], color=[PALETTE["teal"], PALETTE["green"], PALETTE["gold"], PALETTE["navy"]], edgecolor=PALETTE["ink"], linewidth=0.7)
        for y, val, text in zip(metrics["metric"], metrics["value"], metrics["display"], strict=False):
            ax.text(val + max(metrics["value"]) * 0.02, y, text, va="center", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_xlim(0, max(metrics["value"]) * 1.22)
        self.clean_axis(ax)
        return self.save(fig, "10_theory_guided_verdict.png")

    def plot_launch_readiness(self) -> Path:
        df = self.repository.read_csv("launch_readiness_checklist.csv")
        df["status_order"] = df["status"].map({"ready": 2, "validation_ready": 1, "blocked": 0})
        df = df.sort_values("status_order")
        colors = df["status"].map({"ready": PALETTE["teal"], "validation_ready": PALETTE["gold"], "blocked": PALETTE["coral"]})
        fig, ax = plt.subplots(figsize=(8.3, 5.0))
        ax.barh(df["gate"], df["status_score"], color=colors, edgecolor=PALETTE["ink"], linewidth=0.6)
        ax.set_xlim(0, 2.25)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Blocked", "Validate", "Ready"])
        ax.set_xlabel("Gate status")
        self.clean_axis(ax)
        return self.save(fig, "11_launch_readiness_checklist.png")

    def plot_scorecard_components(self) -> Path:
        df = self.repository.read_csv("marketplace_scorecard.csv").sort_values("weighted_evidence_score", ascending=False)
        score_cols = [
            "yield_upside_score",
            "support_score",
            "estimator_agreement_score",
            "conservative_lower_bound_score",
            "downside_risk_score",
            "heterogeneity_score",
            "interference_readiness_score",
        ]
        labels = ["Yield", "Support", "Estimator\nagreement", "Lower\nbound", "Downside", "Heterogeneity", "Interference"]
        heat = df.set_index("policy_id")[score_cols]
        heat.index = [policy_label(p) for p in heat.index]
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        sns.heatmap(heat, cmap=sns.color_palette("YlGnBu", as_cmap=True), vmin=0, vmax=5, annot=True, fmt=".1f", linewidths=0.7, linecolor="white", cbar_kws={"label": "Score"}, ax=ax)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        return self.save(fig, "09_scorecard_components.png")

    def plot_mde_curves(self) -> Path:
        df = self.repository.read_csv("validation_design_detectability.csv").copy()
        df["design"] = df["design_id"].str.replace("_", " ").str.title()
        fig, ax = plt.subplots(figsize=(7.8, 4.7))
        sns.lineplot(data=df, x="experiment_days", y=df["mde_yield_per_opportunity_pct_of_baseline"] * 100, hue="design", marker="o", linewidth=2.2, ax=ax)
        replay = df["priority_replay_lift"].iloc[0] * 100
        p10 = df["priority_p10_dr_lift"].iloc[0] * 100
        ax.axhline(replay, color=PALETTE["teal"], linestyle="--", linewidth=1.5, label="Replay lift")
        ax.axhline(p10, color=PALETTE["red"], linestyle="--", linewidth=1.5, label="P10 DR lift")
        ax.set_xlabel("Experiment days")
        ax.set_ylabel("MDE, percent of baseline yield")
        ax.legend(title=None, ncol=2, loc="upper right")
        self.clean_axis(ax)
        return self.save(fig, "07_design_mde_curves.png")

    def plot_season3_transfer(self) -> Path:
        df = self.repository.read_csv("season2_vs_season3_policy_transfer.csv")
        df["highlight"] = np.where(df["policy_id"].eq(self.config.priority_policy_id), "Priority policy", "Other policies")
        fig, ax = plt.subplots(figsize=(8.8, 6.8))
        sns.scatterplot(data=df, x="season2_pct_yield_lift", y="season3_pct_yield_lift", hue="policy_family", style="highlight", s=110, ax=ax)
        lims = [min(df["season2_pct_yield_lift"].min(), df["season3_pct_yield_lift"].min()) - 0.03, max(df["season2_pct_yield_lift"].max(), df["season3_pct_yield_lift"].max()) + 0.03]
        ax.plot(lims, lims, color=PALETTE["slate"], linestyle="--", linewidth=1.2, label="Equal lift")
        ax.axhline(0, color=PALETTE["grid"], linewidth=1)
        ax.axvline(0, color=PALETTE["grid"], linewidth=1)
        priority_point = df[df["policy_id"].eq(self.config.priority_policy_id)].iloc[0]
        ax.annotate(str(priority_point["policy_label"]), xy=(priority_point["season2_pct_yield_lift"], priority_point["season3_pct_yield_lift"]), xytext=(12, 10), textcoords="offset points", fontsize=9.5, weight="bold", arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "linewidth": 1.1})
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Season-two yield lift vs. logged floor")
        ax.set_ylabel("Season-three yield lift vs. logged floor")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.legend(loc="best", frameon=True, fontsize=8.5)
        self.clean_axis(ax)
        return self.save(fig, "13_season2_vs_season3_transfer.png")

    def plot_season3_guardrails(self) -> Path:
        row = self.repository.read_csv("season3_priority_policy_validation.csv").iloc[0]
        guardrail_values = pd.DataFrame(
            {
                "metric": ["Yield lift", "Impression retention", "Click retention", "Conversion retention", "Value proxy retention"],
                "value": [row.season3_pct_yield_lift, row.season3_retained_impression_share, row.season3_click_retention, row.season3_conversion_retention, row.season3_value_proxy_retention],
                "threshold": [0.0, 0.99, 0.99, 0.99, 0.99],
            }
        )
        fig, ax = plt.subplots(figsize=(9.8, 5.6))
        colors = [PALETTE["blue"] if metric == "Yield lift" else PALETTE["green"] for metric in guardrail_values["metric"]]
        sns.barplot(data=guardrail_values, x="metric", y="value", palette=colors, ax=ax)
        for idx, row in guardrail_values.iterrows():
            ax.hlines(row["threshold"], idx - 0.38, idx + 0.38, color=PALETTE["ink"], linewidth=2)
            ax.text(idx, row["value"] + 0.025, f"{row['value']:.1%}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Rate or lift")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.set_ylim(0, max(1.12, guardrail_values["value"].max() + 0.12))
        ax.tick_params(axis="x", rotation=20)
        self.clean_axis(ax)
        return self.save(fig, "13_season3_priority_policy_guardrails.png")

    def plot_season3_daily_validation(self) -> Path:
        df = self.repository.read_csv("season3_priority_policy_daily_validation.csv")
        df["event_date"] = pd.to_datetime(df["event_date"].astype(str))
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        sns.lineplot(data=df, x="event_date", y="daily_yield_lift_pct", marker="o", color=PALETTE["blue"], linewidth=2.2, ax=ax)
        ax.axhline(0, color=PALETTE["slate"], linestyle="--", linewidth=1.2)
        ax.set_xlabel("Season-three date")
        ax.set_ylabel("Daily yield lift vs. logged floor")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.tick_params(axis="x", rotation=30)
        self.clean_axis(ax)
        return self.save(fig, "13_season3_daily_priority_validation.png")

    def plot_decision_rule_gate_matrix(self) -> Path:
        df = self.repository.read_csv("decision_rule_gate_matrix.csv")
        gate_columns = ["uses_replay", "uses_guardrails", "uses_ope", "uses_support", "uses_season3_validation", "uses_response_sensitivity", "uses_interference_or_propensity_gate"]
        names = {
            "uses_replay": "Replay",
            "uses_guardrails": "Guardrails",
            "uses_ope": "OPE",
            "uses_support": "Support",
            "uses_season3_validation": "Season-3",
            "uses_response_sensitivity": "Response",
            "uses_interference_or_propensity_gate": "Interference /\npropensities",
        }
        matrix = df.set_index("rule_label")[gate_columns].rename(columns=names).astype(int)
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        sns.heatmap(matrix, cmap=sns.color_palette([PALETTE["light_red"], PALETTE["light_green"]], as_cmap=True), cbar=False, linewidths=0.8, linecolor="white", annot=matrix.replace({0: "No", 1: "Yes"}), fmt="", annot_kws={"fontsize": 12.5, "fontweight": "bold"}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25, labelsize=12)
        ax.tick_params(axis="y", rotation=0, labelsize=12)
        return self.save(fig, "14_decision_rule_gate_matrix.png")

    def plot_decision_rule_unresolved_gates(self) -> Path:
        df = self.repository.read_csv("decision_rule_ablation_summary.csv")
        df["action_label"] = np.where(df["recommended_action_under_rule"].eq("direct_launch"), "Direct launch", "Online validation")
        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        sns.barplot(data=df, x="rule_label", y="unresolved_launch_gate_count", hue="action_label", dodge=False, palette={"Direct launch": PALETTE["red"], "Online validation": PALETTE["blue"]}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Unresolved launch gates")
        ax.tick_params(axis="x", rotation=25, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(title="Rule action", loc="upper right", fontsize=11, title_fontsize=12)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3, fontsize=12)
        self.clean_axis(ax)
        return self.save(fig, "14_decision_rule_unresolved_gates.png")

    def plot_decision_rule_bootstrap_selection(self) -> Path:
        df = self.repository.read_csv("decision_rule_bootstrap_selection.csv").groupby("rule_label").head(3).copy()
        df["policy_display"] = df["policy_number"].astype(str) + ": " + df["policy_label"].astype(str)
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        sns.barplot(data=df, x="selection_share", y="rule_label", hue="policy_display", ax=ax)
        ax.set_xlabel("Selection share across resamples")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(title="Selected policy", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=11, title_fontsize=12)
        self.clean_axis(ax)
        return self.save(fig, "14_decision_rule_bootstrap_selection.png")

    def render_all(self) -> list[Path]:
        renderers = [
            self.render_interference_dag,
            self.render_replay_flow,
            self.render_validation_sequence,
            self.render_decision_waterfall,
            self.plot_price_distributions,
            self.plot_outcome_density,
            self.plot_calibration,
            self.plot_frontier,
            self.plot_daily_stability,
            self.plot_ope_comparison,
            self.plot_weight_diagnostics,
            self.plot_conservative_ranking,
            self.plot_equilibrium_sensitivity,
            self.plot_support_collapse,
            self.plot_segment_heterogeneity,
            self.plot_theory_verdict,
            self.plot_launch_readiness,
            self.plot_scorecard_components,
            self.plot_mde_curves,
            self.plot_season3_transfer,
            self.plot_season3_guardrails,
            self.plot_season3_daily_validation,
            self.plot_decision_rule_gate_matrix,
            self.plot_decision_rule_unresolved_gates,
            self.plot_decision_rule_bootstrap_selection,
        ]
        return [renderer() for renderer in renderers]
