from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas import DriftStatus, TaskType


@dataclass
class TrainingResult:
    pipeline: Pipeline
    task: TaskType
    feature_columns: list[str]
    metrics: dict[str, Any]
    drift_profile: dict[str, Any]


def infer_task(y: pd.Series, task_hint: TaskType | None) -> TaskType:
    if task_hint is not None:
        return task_hint
    numeric_target = pd.api.types.is_numeric_dtype(y)
    unique_count = int(y.nunique(dropna=True))
    unique_ratio = unique_count / max(len(y), 1)
    if numeric_target and unique_count > 20 and unique_ratio > 0.05:
        return "regression"
    return "classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable features found for training.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _classification_models() -> list[tuple[str, Any]]:
    return [
        ("logistic_regression", LogisticRegression(max_iter=2000)),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
        ),
    ]


def _regression_models() -> list[tuple[str, Any]]:
    return [
        ("elastic_net", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)),
        (
            "random_forest",
            RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1),
        ),
    ]


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    return {
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _primary_metric(task: TaskType) -> str:
    return "f1_weighted" if task == "classification" else "rmse"


def _selection_score(task: TaskType, metrics: dict[str, float]) -> float:
    if task == "classification":
        return metrics["f1_weighted"]
    return -metrics["rmse"]


def _build_drift_profile(X: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

    for feature in numeric_columns:
        series = pd.to_numeric(X[feature], errors="coerce").dropna()
        if series.empty or series.nunique() < 2:
            continue
        inner_edges = np.unique(np.quantile(series, np.linspace(0.1, 0.9, 9)))
        bins = np.concatenate(([-np.inf], inner_edges, [np.inf]))
        histogram, _ = np.histogram(series, bins=bins)
        total = histogram.sum()
        if total == 0:
            continue
        expected = histogram / total
        profile[feature] = {
            "edges": [float(value) for value in inner_edges.tolist()],
            "expected": [float(value) for value in expected.tolist()],
        }

    return profile


def train_best_model(
    df: pd.DataFrame,
    target_column: str,
    task_hint: TaskType | None,
) -> TrainingResult:
    if df.empty:
        raise ValueError("Training dataset is empty.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the CSV.")
    if len(df) < 8:
        raise ValueError("At least 8 rows are required for train/validation split.")

    clean_df = df.dropna(how="all").copy()
    clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]

    y = clean_df[target_column]
    X = clean_df.drop(columns=[target_column])

    if X.shape[1] == 0:
        raise ValueError("The dataset must include at least one feature column.")
    if y.nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two unique values.")

    task = infer_task(y, task_hint)
    preprocessor = _build_preprocessor(X)

    stratify = y if task == "classification" else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    candidates = _classification_models() if task == "classification" else _regression_models()
    evaluation_rows: list[dict[str, Any]] = []

    for name, estimator in candidates:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_valid)
            metrics = (
                _classification_metrics(y_valid, predictions)
                if task == "classification"
                else _regression_metrics(y_valid, predictions)
            )
            evaluation_rows.append(
                {
                    "name": name,
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "score": _selection_score(task, metrics),
                }
            )
        except Exception:
            continue

    if not evaluation_rows:
        raise ValueError("Model training failed for every candidate algorithm.")

    best = max(evaluation_rows, key=lambda item: item["score"])
    metric_name = _primary_metric(task)
    candidates_payload = [
        {"model": row["name"], "metrics": row["metrics"]}
        for row in sorted(evaluation_rows, key=lambda item: item["score"], reverse=True)
    ]

    return TrainingResult(
        pipeline=best["pipeline"],
        task=task,
        feature_columns=X.columns.tolist(),
        metrics={
            "primary_metric": metric_name,
            "primary_value": float(best["metrics"][metric_name]),
            "candidates": candidates_payload,
        },
        drift_profile=_build_drift_profile(X_train),
    )


def _psi(expected: np.ndarray, observed: np.ndarray, epsilon: float = 1e-9) -> float:
    expected = np.clip(expected, epsilon, None)
    observed = np.clip(observed, epsilon, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _status_from_psi(value: float) -> DriftStatus:
    if value < 0.1:
        return "stable"
    if value < 0.2:
        return "moderate"
    return "significant"


def score_drift(rows: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    feature_results: list[dict[str, Any]] = []
    psi_values: list[float] = []

    for feature, spec in profile.items():
        if feature not in rows.columns:
            continue
        series = pd.to_numeric(rows[feature], errors="coerce").dropna()
        if series.empty:
            continue

        edges = np.array([-np.inf, *spec["edges"], np.inf], dtype=float)
        observed_hist, _ = np.histogram(series, bins=edges)
        observed_total = observed_hist.sum()
        if observed_total == 0:
            continue

        observed = observed_hist / observed_total
        expected = np.array(spec["expected"], dtype=float)
        if len(expected) != len(observed):
            continue

        psi_value = _psi(expected, observed)
        psi_values.append(psi_value)
        feature_results.append(
            {
                "feature": feature,
                "psi": round(float(psi_value), 4),
                "status": _status_from_psi(psi_value),
            }
        )

    overall_psi = float(np.mean(psi_values)) if psi_values else 0.0
    return {
        "overall_psi": round(overall_psi, 4),
        "status": _status_from_psi(overall_psi),
        "features": feature_results,
    }
