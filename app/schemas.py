from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["classification", "regression"]
DriftStatus = Literal["stable", "moderate", "significant"]


class HealthResponse(BaseModel):
    status: str
    models: int


class CandidateResult(BaseModel):
    model: str
    metrics: dict[str, float]


class TrainMetrics(BaseModel):
    primary_metric: str
    primary_value: float
    candidates: list[CandidateResult]


class TrainResponse(BaseModel):
    model_id: str
    created_at: datetime
    model_name: str
    task: TaskType
    target_column: str
    feature_columns: list[str]
    row_count: int
    metrics: TrainMetrics


class ModelSummary(BaseModel):
    model_id: str
    created_at: datetime
    model_name: str
    task: TaskType
    target_column: str
    feature_columns: list[str]
    row_count: int
    metrics: TrainMetrics


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=2000)
    include_probabilities: bool = False


class PredictResponse(BaseModel):
    model_id: str
    task: TaskType
    predictions: list[Any]
    probabilities: list[dict[str, float]] | None = None


class DriftRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=5000)


class DriftFeatureResult(BaseModel):
    feature: str
    psi: float
    status: DriftStatus


class DriftResponse(BaseModel):
    model_id: str
    overall_psi: float
    status: DriftStatus
    features: list[DriftFeatureResult]
