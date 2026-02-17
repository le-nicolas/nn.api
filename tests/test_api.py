from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def _training_csv(rows: int = 84) -> bytes:
    plans = ["basic", "pro", "premium"]
    lines = ["age,income,plan,sessions,last_login_days,churn"]

    for idx in range(rows):
        age = 22 + (idx % 35)
        income = 28000 + (idx * 1700)
        plan = plans[idx % 3]
        sessions = 1 + (idx % 11)
        last_login_days = idx % 25
        churn = 1 if ((plan == "basic" and last_login_days > 12) or sessions < 3) else 0
        lines.append(f"{age},{income},{plan},{sessions},{last_login_days},{churn}")

    return "\n".join(lines).encode("utf-8")


def _train(client: TestClient) -> str:
    response = client.post(
        "/v1/train",
        files={"file": ("customers.csv", _training_csv(), "text/csv")},
        data={"target_column": "churn", "model_name": "customer-churn"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["task"] == "classification"
    assert body["metrics"]["primary_metric"] == "f1_weighted"
    return body["model_id"]


def test_train_predict_and_drift(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "runtime"))
    model_id = _train(client)

    prediction_response = client.post(
        f"/v1/models/{model_id}/predict",
        json={
            "rows": [
                {
                    "age": 34,
                    "income": 62000,
                    "plan": "basic",
                    "sessions": 2,
                    "last_login_days": 15,
                },
                {
                    "age": 40,
                    "income": 97000,
                    "plan": "premium",
                    "sessions": 8,
                    "last_login_days": 2,
                },
            ],
            "include_probabilities": True,
        },
    )
    assert prediction_response.status_code == 200, prediction_response.text
    prediction = prediction_response.json()
    assert len(prediction["predictions"]) == 2
    assert prediction["probabilities"] is not None
    assert abs(sum(prediction["probabilities"][0].values()) - 1.0) < 0.001

    drift_response = client.post(
        f"/v1/models/{model_id}/drift",
        json={
            "rows": [
                {
                    "age": 63,
                    "income": 140000,
                    "plan": "basic",
                    "sessions": 1,
                    "last_login_days": 45,
                },
                {
                    "age": 59,
                    "income": 130000,
                    "plan": "basic",
                    "sessions": 1,
                    "last_login_days": 40,
                },
            ]
        },
    )
    assert drift_response.status_code == 200, drift_response.text
    drift = drift_response.json()
    assert "overall_psi" in drift
    assert isinstance(drift["features"], list)


def test_missing_features_return_422(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "runtime"))
    model_id = _train(client)

    response = client.post(
        f"/v1/models/{model_id}/predict",
        json={"rows": [{"age": 31, "income": 50000}]},
    )
    assert response.status_code == 422
    assert "Missing required features" in response.text
