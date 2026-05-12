from __future__ import annotations

import numpy as np
import pandas as pd

from data_access import PRIORITY_POLICY_LABEL


READER_POLICY_LABELS = {
    "logged_floor_status_quo": "Logged Status Quo",
    "uniform_raise_05pct": "Uniform +5%",
    "uniform_raise_10pct": "Uniform +10%",
    "uniform_raise_15pct": "Uniform +15%",
    "uniform_raise_20pct": "Uniform +20%",
    "uniform_raise_30pct": "Uniform +30%",
    "add_5_all_floors": "Add 5 To All Floors",
    "add_10_all_floors": "Add 10 To All Floors",
    "add_20_all_floors": "Add 20 To All Floors",
    "min_positive_floor_q25": "Positive Floors To Q25",
    "min_positive_floor_q50": "Positive Floors To Q50",
    "min_positive_floor_q75": "Positive Floors To Q75",
    "zero_and_low_floor_to_q25": "All Low Floors To Q25",
    "zero_and_low_floor_to_q50": "All Low Floors To Q50",
    "margin_gap_25_add_5": "Gap 25 Add 5",
    "margin_gap_50_add_10": "Gap 50 Add 10",
    "margin_gap_100_add_20": "Gap 100 Add 20",
    "hybrid_q50_if_gap_50": "Q50 Margin-Gated Floor",
    "hybrid_q75_if_gap_100": PRIORITY_POLICY_LABEL,
}


def reader_policy_label(policy_id: str) -> str:
    return READER_POLICY_LABELS.get(policy_id, policy_id.replace("_", " ").title())


class ReservePolicyCatalog:
    """Candidate non-decreasing reserve/floor policies."""

    def __init__(self, price_quantiles: dict[str, float]) -> None:
        self.q25 = float(price_quantiles["q25"])
        self.q50 = float(price_quantiles["q50"])
        self.q75 = float(price_quantiles["q75"])

    def registry(self) -> pd.DataFrame:
        rows = [
            ("logged_floor_status_quo", "baseline", "Logged floor exactly as observed.", "logged"),
            ("uniform_raise_05pct", "uniform_percent", "Raise every logged floor by 5 percent.", "logged * 1.05"),
            ("uniform_raise_10pct", "uniform_percent", "Raise every logged floor by 10 percent.", "logged * 1.10"),
            ("uniform_raise_15pct", "uniform_percent", "Raise every logged floor by 15 percent.", "logged * 1.15"),
            ("uniform_raise_20pct", "uniform_percent", "Raise every logged floor by 20 percent.", "logged * 1.20"),
            ("uniform_raise_30pct", "uniform_percent", "Raise every logged floor by 30 percent.", "logged * 1.30"),
            ("add_5_all_floors", "absolute_increment", "Add 5 price units to every floor.", "logged + 5"),
            ("add_10_all_floors", "absolute_increment", "Add 10 price units to every floor.", "logged + 10"),
            ("add_20_all_floors", "absolute_increment", "Add 20 price units to every floor.", "logged + 20"),
            (
                "min_positive_floor_q25",
                "minimum_positive_floor",
                f"Lift positive floors below q25 ({self.q25:.2f}) to q25.",
                f"max(logged, {self.q25:.2f}) if logged > 0",
            ),
            (
                "min_positive_floor_q50",
                "minimum_positive_floor",
                f"Lift positive floors below q50 ({self.q50:.2f}) to q50.",
                f"max(logged, {self.q50:.2f}) if logged > 0",
            ),
            (
                "min_positive_floor_q75",
                "minimum_positive_floor",
                f"Lift positive floors below q75 ({self.q75:.2f}) to q75.",
                f"max(logged, {self.q75:.2f}) if logged > 0",
            ),
            (
                "zero_and_low_floor_to_q25",
                "minimum_all_floor",
                f"Lift all floors below q25 ({self.q25:.2f}) to q25.",
                f"max(logged, {self.q25:.2f})",
            ),
            (
                "zero_and_low_floor_to_q50",
                "minimum_all_floor",
                f"Lift all floors below q50 ({self.q50:.2f}) to q50.",
                f"max(logged, {self.q50:.2f})",
            ),
            (
                "margin_gap_25_add_5",
                "margin_aware_increment",
                "Add 5 only when bid-floor gap is at least 25.",
                "logged + 5 if gap >= 25",
            ),
            (
                "margin_gap_50_add_10",
                "margin_aware_increment",
                "Add 10 only when bid-floor gap is at least 50.",
                "logged + 10 if gap >= 50",
            ),
            (
                "margin_gap_100_add_20",
                "margin_aware_increment",
                "Add 20 only when bid-floor gap is at least 100.",
                "logged + 20 if gap >= 100",
            ),
            (
                "hybrid_q50_if_gap_50",
                "hybrid_minimum_margin",
                f"Lift to q50 ({self.q50:.2f}) only when bid-floor gap is at least 50.",
                f"max(logged, {self.q50:.2f}) if gap >= 50",
            ),
            (
                "hybrid_q75_if_gap_100",
                "hybrid_minimum_margin",
                f"Lift to q75 ({self.q75:.2f}) only when bid-floor gap is at least 100.",
                f"max(logged, {self.q75:.2f}) if gap >= 100",
            ),
        ]
        frame = pd.DataFrame(rows, columns=["policy_id", "policy_family", "description", "floor_rule"])
        frame["supported_by_replay_contract"] = True
        frame["policy_number"] = [f"P{i}" for i in range(len(frame))]
        frame["policy_label"] = frame["policy_id"].map(reader_policy_label)
        return frame

    def floor(self, frame: pd.DataFrame, policy_id: str) -> np.ndarray:
        logged = pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0).clip(lower=0).to_numpy("float64")
        gap = pd.to_numeric(frame["bid_floor_gap"], errors="coerce").fillna(-np.inf).to_numpy("float64")
        if policy_id == "logged_floor_status_quo":
            return logged
        if policy_id.startswith("uniform_raise_"):
            pct = float(policy_id.split("_")[-1].replace("pct", "")) / 100.0
            return logged * (1.0 + pct)
        if policy_id == "add_5_all_floors":
            return logged + 5.0
        if policy_id == "add_10_all_floors":
            return logged + 10.0
        if policy_id == "add_20_all_floors":
            return logged + 20.0
        if policy_id == "min_positive_floor_q25":
            return np.where(logged > 0, np.maximum(logged, self.q25), logged)
        if policy_id == "min_positive_floor_q50":
            return np.where(logged > 0, np.maximum(logged, self.q50), logged)
        if policy_id == "min_positive_floor_q75":
            return np.where(logged > 0, np.maximum(logged, self.q75), logged)
        if policy_id == "zero_and_low_floor_to_q25":
            return np.maximum(logged, self.q25)
        if policy_id == "zero_and_low_floor_to_q50":
            return np.maximum(logged, self.q50)
        if policy_id == "margin_gap_25_add_5":
            return np.where(gap >= 25.0, logged + 5.0, logged)
        if policy_id == "margin_gap_50_add_10":
            return np.where(gap >= 50.0, logged + 10.0, logged)
        if policy_id == "margin_gap_100_add_20":
            return np.where(gap >= 100.0, logged + 20.0, logged)
        if policy_id == "hybrid_q50_if_gap_50":
            return np.where(gap >= 50.0, np.maximum(logged, self.q50), logged)
        if policy_id == "hybrid_q75_if_gap_100":
            return np.where(gap >= 100.0, np.maximum(logged, self.q75), logged)
        raise ValueError(f"Unknown policy_id: {policy_id}")


