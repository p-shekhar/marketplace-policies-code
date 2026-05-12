from __future__ import annotations

import numpy as np
import pandas as pd

from config import RawPipelineConfig
from data_access import Workspace
from progress import ProgressLogger


class NuisanceModelTrainer:
    """LightGBM nuisance-model prototype used by the notebook workflow."""

    categorical_features = [
        "region",
        "city",
        "ad_exchange",
        "advertiser_id",
        "slot_visibility",
        "slot_format",
    ]
    numeric_features = [
        "bid_price",
        "slot_floor_price",
        "bid_floor_gap",
        "floor_to_bid_ratio",
        "slot_width",
        "slot_height",
        "slot_area",
        "hour",
        "day_of_week",
        "user_tag_count",
        "has_user_tags",
    ]
    feature_families = "time; geography; exchange; inventory; advertiser; bid_and_floor; user_tag_count"

    def __init__(self, workspace: Workspace, config: RawPipelineConfig, progress: ProgressLogger | None = None) -> None:
        self.workspace = workspace
        self.config = config
        self.progress = progress or ProgressLogger(enabled=False)

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def feature_label_contract(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "model_name": "fill_probability_model",
                    "target": "filled",
                    "task_type": "binary_classification",
                    "training_population": "all_bid_opportunities",
                    "allowed_feature_families": self.feature_families,
                    "downstream_use": "fill guardrail, reserve-policy replay support, direct-method expected fill",
                    "leakage_rule": "exclude pay_price, clicked, converted, and any post-auction outcome fields",
                },
                {
                    "model_name": "pay_price_model",
                    "target": "pay_price",
                    "task_type": "regression",
                    "training_population": "filled_opportunities_only",
                    "allowed_feature_families": self.feature_families,
                    "downstream_use": (
                        "yield prediction and direct-method expected platform revenue conditional on fill"
                    ),
                    "leakage_rule": "exclude clicked, converted, and post-auction labels; pay_price is the target only",
                },
                {
                    "model_name": "ctr_model",
                    "target": "clicked",
                    "task_type": "binary_classification",
                    "training_population": "filled_opportunities_only",
                    "allowed_feature_families": self.feature_families,
                    "downstream_use": (
                        "advertiser value proxy, user-experience guardrail proxy, direct-method value model"
                    ),
                    "leakage_rule": "exclude conversion and all labels other than clicked",
                },
                {
                    "model_name": "conversion_model",
                    "target": "converted",
                    "task_type": "binary_classification",
                    "training_population": "filled_opportunities_only_or_click_conditioned_when_needed",
                    "allowed_feature_families": self.feature_families,
                    "downstream_use": "advertiser value guardrail and value proxy sensitivity",
                    "leakage_rule": "conversion is extremely sparse; do not overstate prototype results",
                },
                {
                    "model_name": "value_proxy_model",
                    "target": "value_proxy_lambda_10",
                    "task_type": "regression",
                    "training_population": "filled_opportunities_only",
                    "allowed_feature_families": self.feature_families,
                    "downstream_use": "multi-objective scorecard and direct-method expected value proxy",
                    "leakage_rule": "value proxy is a constructed target; report its lambda choice explicitly",
                },
            ]
        )
        frame.to_csv(self.workspace.metadata_dir / "feature_label_contract.csv", index=False)
        frame.to_csv(self.workspace.table_dir / "04_feature_label_contract.csv", index=False)
        return frame

    @staticmethod
    def _build_estimator(task_type: str):
        from lightgbm import LGBMClassifier, LGBMRegressor

        if task_type == "binary_classification":
            return LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=42,
                n_jobs=2,
                verbose=-1,
            )
        return LGBMRegressor(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )

    def _preprocess(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=60, sparse_output=True)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_features),
                ("cat", categorical_pipe, self.categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

    def _evaluate_model(
        self, model_name: str, target: str, task_type: str, data: pd.DataFrame, min_positive: int
    ) -> tuple[dict[str, object], pd.DataFrame | None]:
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline

        working = data.dropna(subset=self.feature_columns + [target]).copy()
        train = working.loc[working["time_split"].eq("train")]
        test = working.loc[working["time_split"].eq("test")]
        record: dict[str, object] = {
            "model_name": model_name,
            "target": target,
            "task_type": task_type,
            "train_rows": len(train),
            "test_rows": len(test),
            "status": "not_trained",
            "reason": "",
        }
        if len(train) == 0 or len(test) == 0:
            record["reason"] = "empty train or test split"
            return record, None

        if task_type == "binary_classification":
            train_positive = int(train[target].sum())
            test_positive = int(test[target].sum())
            record.update(
                {
                    "train_positive": train_positive,
                    "test_positive": test_positive,
                    "train_positive_rate": train[target].mean(),
                    "test_positive_rate": test[target].mean(),
                }
            )
            if train[target].nunique() < 2 or test[target].nunique() < 2 or train_positive < min_positive:
                record["status"] = "skipped"
                record["reason"] = "too few positive examples for a credible prototype classifier"
                return record, None
        else:
            record["train_target_mean"] = train[target].mean()
            record["test_target_mean"] = test[target].mean()
            if train[target].notna().sum() < 500 or test[target].notna().sum() < 100:
                record["status"] = "skipped"
                record["reason"] = "too few regression rows for a credible prototype regressor"
                return record, None

        pipe = Pipeline(steps=[("preprocess", self._preprocess()), ("model", self._build_estimator(task_type))])
        pipe.fit(train[self.feature_columns], train[target])

        if task_type == "binary_classification":
            predicted = pipe.predict_proba(test[self.feature_columns])[:, 1]
            record.update(
                {
                    "status": "trained",
                    "roc_auc": roc_auc_score(test[target], predicted) if test[target].nunique() > 1 else np.nan,
                    "average_precision": average_precision_score(test[target], predicted),
                    "brier_score": brier_score_loss(test[target], predicted),
                    "prediction_mean": float(np.mean(predicted)),
                }
            )
        else:
            predicted = pipe.predict(test[self.feature_columns])
            record.update(
                {
                    "status": "trained",
                    "mae": mean_absolute_error(test[target], predicted),
                    "rmse": mean_squared_error(test[target], predicted) ** 0.5,
                    "r2": r2_score(test[target], predicted),
                    "prediction_mean": float(np.mean(predicted)),
                }
            )

        predictions = (
            test[["bid_id", "time_split", target]].copy()
            if "bid_id" in test.columns
            else test[["time_split", target]].copy()
        )
        predictions["model_name"] = model_name
        predictions["prediction"] = predicted
        return record, predictions

    def run(self, panel: pd.DataFrame) -> None:
        self.progress.step("LightGBM nuisance-model training and calibration")
        self.feature_label_contract()
        if len(panel) > self.config.nuisance_sample_rows:
            model_panel = panel.sample(
                n=self.config.nuisance_sample_rows, random_state=self.config.random_seed + 404
            ).copy()
            model_panel = model_panel.sort_values(["event_time", "bid_id"]).reset_index(drop=True)
            row_position = np.arange(len(model_panel)) / max(len(model_panel) - 1, 1)
            model_panel["time_split"] = np.select(
                [row_position < 0.60, row_position < 0.80], ["train", "validation"], default="test"
            )
        else:
            model_panel = panel.copy()

        for column in self.categorical_features:
            model_panel[column] = model_panel[column].astype("string").fillna("missing")
        model_panel["floor_to_bid_ratio"] = model_panel["floor_to_bid_ratio"].replace([np.inf, -np.inf], np.nan)

        model_specs = [
            ("fill_probability_model", "filled", "binary_classification", model_panel, 50),
            ("pay_price_model", "pay_price", "regression", model_panel.loc[model_panel["filled"].eq(1)].copy(), 20),
            ("ctr_model", "clicked", "binary_classification", model_panel.loc[model_panel["filled"].eq(1)].copy(), 10),
            (
                "conversion_model",
                "converted",
                "binary_classification",
                model_panel.loc[model_panel["filled"].eq(1)].copy(),
                10,
            ),
            (
                "value_proxy_model",
                "value_proxy_lambda_10",
                "regression",
                model_panel.loc[model_panel["filled"].eq(1)].copy(),
                20,
            ),
        ]
        records: list[dict[str, object]] = []
        prediction_frames: list[pd.DataFrame] = []
        for model_name, target, task_type, data, min_positive in model_specs:
            self.progress.log(
                f"Training nuisance model `{model_name}` for target `{target}` on {len(data):,} candidate rows."
            )
            record, predictions = self._evaluate_model(model_name, target, task_type, data, min_positive)
            records.append(record)
            if predictions is not None:
                prediction_frames.append(predictions)
            self.progress.log(f"Model `{model_name}` status: {record.get('status', 'unknown')}.")

        metrics = pd.DataFrame(records)
        metrics.to_csv(self.workspace.metadata_dir / "prototype_model_metrics.csv", index=False)
        metrics.to_csv(self.workspace.table_dir / "04_prototype_model_metrics.csv", index=False)

        if prediction_frames:
            all_predictions = pd.concat(prediction_frames, ignore_index=True)
        else:
            all_predictions = pd.DataFrame()
        calibration_frames = []
        classification_targets = metrics.loc[
            metrics["task_type"].eq("binary_classification") & metrics["status"].eq("trained"),
            ["model_name", "target"],
        ]
        for row in classification_targets.itertuples(index=False):
            pred = all_predictions.loc[
                all_predictions["model_name"].eq(row.model_name), [row.target, "prediction"]
            ].copy()
            if pred.empty or pred["prediction"].nunique() < 2:
                continue
            pred["probability_bin"] = pd.qcut(pred["prediction"], q=10, duplicates="drop")
            cal = (
                pred.groupby("probability_bin", observed=True)
                .agg(
                    rows=(row.target, "size"), predicted_rate=("prediction", "mean"), observed_rate=(row.target, "mean")
                )
                .reset_index()
            )
            cal["model_name"] = row.model_name
            cal["target"] = row.target
            calibration_frames.append(cal)
        calibration = (
            pd.concat(calibration_frames, ignore_index=True)
            if calibration_frames
            else pd.DataFrame(
                columns=["probability_bin", "rows", "predicted_rate", "observed_rate", "model_name", "target"]
            )
        )
        calibration.to_csv(self.workspace.metadata_dir / "prototype_calibration_summary.csv", index=False)

        regression_frames = []
        regression_targets = metrics.loc[
            metrics["task_type"].eq("regression") & metrics["status"].eq("trained"),
            ["model_name", "target"],
        ]
        for row in regression_targets.itertuples(index=False):
            pred = all_predictions.loc[
                all_predictions["model_name"].eq(row.model_name), [row.target, "prediction"]
            ].dropna(subset=[row.target, "prediction"])
            if pred.empty or pred["prediction"].nunique() < 2:
                continue
            pred = pred.copy()
            pred["prediction_bin"] = pd.qcut(pred["prediction"], q=10, duplicates="drop")
            reg_cal = (
                pred.groupby("prediction_bin", observed=True)
                .agg(
                    rows=(row.target, "size"),
                    predicted_mean=("prediction", "mean"),
                    observed_mean=(row.target, "mean"),
                    observed_sd=(row.target, "std"),
                )
                .reset_index()
            )
            reg_cal["residual_mean"] = reg_cal["observed_mean"] - reg_cal["predicted_mean"]
            reg_cal["model_name"] = row.model_name
            reg_cal["target"] = row.target
            regression_frames.append(reg_cal)
        regression_calibration = (
            pd.concat(regression_frames, ignore_index=True)
            if regression_frames
            else pd.DataFrame(
                columns=[
                    "prediction_bin",
                    "rows",
                    "predicted_mean",
                    "observed_mean",
                    "observed_sd",
                    "residual_mean",
                    "model_name",
                    "target",
                ]
            )
        )
        regression_calibration.to_csv(
            self.workspace.metadata_dir / "prototype_regression_calibration_summary.csv", index=False
        )

        status_lookup = metrics.set_index("model_name")["status"].to_dict()
        reason_lookup = metrics.set_index("model_name").get("reason", pd.Series(dtype=str)).to_dict()
        registry = pd.DataFrame(
            [
                {
                    "model_name": "fill_probability_model",
                    "supports_estimands": "E1_short_run_replay; E3_direct_method; E6_constrained_launch_decision",
                    "paper_role": "estimate fill and diagnose how floor changes may reduce delivery",
                    "prototype_status": status_lookup.get("fill_probability_model", "missing"),
                    "main_risk": "offline fill does not capture strategic bidder response under new floors",
                },
                {
                    "model_name": "pay_price_model",
                    "supports_estimands": "E1_short_run_replay; E3_direct_method; E6_constrained_launch_decision",
                    "paper_role": "predict platform yield conditional on fill",
                    "prototype_status": status_lookup.get("pay_price_model", "missing"),
                    "main_risk": "pay price is mechanism-dependent and may shift when reserve policy changes",
                },
                {
                    "model_name": "ctr_model",
                    "supports_estimands": "E3_direct_method; E6_constrained_launch_decision",
                    "paper_role": "proxy advertiser value and user-response quality",
                    "prototype_status": status_lookup.get("ctr_model", "missing"),
                    "main_risk": "clicks are rare, so calibration and uncertainty dominate point estimates",
                },
                {
                    "model_name": "conversion_model",
                    "supports_estimands": "E3_direct_method; E6_constrained_launch_decision",
                    "paper_role": "proxy deeper advertiser value where enough data exist",
                    "prototype_status": status_lookup.get("conversion_model", "missing"),
                    "main_risk": "conversion remains sparse even in the season-level modeling sample",
                },
                {
                    "model_name": "value_proxy_model",
                    "supports_estimands": "E3_direct_method; E6_constrained_launch_decision",
                    "paper_role": "combine click and conversion proxy outcomes for scorecard stress tests",
                    "prototype_status": status_lookup.get("value_proxy_model", "missing"),
                    "main_risk": "proxy value depends on the chosen click/conversion weighting",
                },
            ]
        )
        registry["prototype_note"] = registry["model_name"].map(reason_lookup).fillna("")
        registry.to_csv(self.workspace.metadata_dir / "nuisance_model_registry.csv", index=False)
        registry.to_csv(self.workspace.table_dir / "04_nuisance_model_registry.csv", index=False)
        trained = int(metrics["status"].eq("trained").sum()) if "status" in metrics else 0
        self.progress.done(f"LightGBM nuisance modeling; {trained}/{len(metrics)} models trained")
