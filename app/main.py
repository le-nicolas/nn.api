from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.ml import score_drift, train_best_model
from app.schemas import (
    DriftRequest,
    DriftResponse,
    HealthResponse,
    ModelSummary,
    PredictRequest,
    PredictResponse,
    TaskType,
    TrainResponse,
)
from app.storage import RegistryStore


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _model_summary(record: dict[str, Any]) -> ModelSummary:
    return ModelSummary(
        model_id=record["model_id"],
        created_at=record["created_at"],
        model_name=record["model_name"],
        task=record["task"],
        target_column=record["target_column"],
        feature_columns=record["feature_columns"],
        row_count=record["row_count"],
        metrics=record["metrics"],
    )


def create_app(data_dir: Path | None = None) -> FastAPI:
    resolved_data_dir = data_dir or Path(os.getenv("NN_API_DATA_DIR", "runtime"))
    store = RegistryStore(resolved_data_dir)

    app = FastAPI(
        title="nn.api",
        version="2.0.0",
        description="Train, serve, and monitor tabular machine learning models.",
    )

    def fetch_record(model_id: str) -> dict[str, Any]:
        record = store.get_record(model_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' was not found.")
        return record

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", models=len(store.list_records()))

    @app.post("/v1/train", response_model=TrainResponse, status_code=201)
    async def train(
        file: Annotated[UploadFile, File(...)],
        target_column: Annotated[str, Form(...)],
        model_name: Annotated[str | None, Form()] = None,
        task: Annotated[TaskType | None, Form()] = None,
    ) -> TrainResponse:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

        try:
            result = train_best_model(df=df, target_column=target_column, task_hint=task)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        created_at = datetime.now(timezone.utc)
        model_id = uuid4().hex
        inferred_name = model_name
        if not inferred_name:
            inferred_name = (
                file.filename.rsplit(".", 1)[0]
                if file.filename
                else f"model-{model_id[:8]}"
            )

        store.save_model(model_id, result.pipeline)
        record = {
            "model_id": model_id,
            "created_at": created_at.isoformat(),
            "model_name": inferred_name,
            "task": result.task,
            "target_column": target_column,
            "feature_columns": result.feature_columns,
            "row_count": int(len(df)),
            "metrics": result.metrics,
            "drift_profile": result.drift_profile,
        }
        store.add_record(record)

        return TrainResponse(
            model_id=model_id,
            created_at=created_at,
            model_name=inferred_name,
            task=result.task,
            target_column=target_column,
            feature_columns=result.feature_columns,
            row_count=int(len(df)),
            metrics=result.metrics,
        )

    @app.get("/v1/models", response_model=list[ModelSummary])
    def list_models() -> list[ModelSummary]:
        records = sorted(
            store.list_records(),
            key=lambda item: item["created_at"],
            reverse=True,
        )
        return [_model_summary(record) for record in records]

    @app.get("/v1/models/{model_id}", response_model=ModelSummary)
    def get_model(model_id: str) -> ModelSummary:
        return _model_summary(fetch_record(model_id))

    @app.post("/v1/models/{model_id}/predict", response_model=PredictResponse)
    def predict(model_id: str, payload: PredictRequest) -> PredictResponse:
        record = fetch_record(model_id)
        try:
            pipeline = store.load_model(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        frame = pd.DataFrame(payload.rows)
        missing_features = [col for col in record["feature_columns"] if col not in frame.columns]
        if missing_features:
            missing_text = ", ".join(missing_features)
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing_text}.",
            )
        frame = frame[record["feature_columns"]]

        try:
            predictions = pipeline.predict(frame)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

        response = PredictResponse(
            model_id=model_id,
            task=record["task"],
            predictions=[_to_builtin(value) for value in predictions.tolist()],
        )

        if payload.include_probabilities and record["task"] == "classification":
            model = pipeline.named_steps.get("model")
            if hasattr(model, "predict_proba"):
                probabilities = pipeline.predict_proba(frame)
                classes = [str(item) for item in getattr(model, "classes_", [])]
                response.probabilities = [
                    {classes[idx]: float(score) for idx, score in enumerate(row)}
                    for row in probabilities
                ]

        return response

    @app.post("/v1/models/{model_id}/drift", response_model=DriftResponse)
    def check_drift(model_id: str, payload: DriftRequest) -> DriftResponse:
        record = fetch_record(model_id)
        profile = record.get("drift_profile", {})
        if not profile:
            raise HTTPException(
                status_code=400,
                detail="No numeric features available for drift checking on this model.",
            )

        frame = pd.DataFrame(payload.rows)
        missing_features = [col for col in record["feature_columns"] if col not in frame.columns]
        if missing_features:
            missing_text = ", ".join(missing_features)
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing_text}.",
            )

        frame = frame[record["feature_columns"]]
        report = score_drift(frame, profile)
        return DriftResponse(model_id=model_id, **report)

    return app


app = create_app()
