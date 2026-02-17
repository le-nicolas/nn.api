from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import joblib


class RegistryStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.registry_path = self.data_dir / "registry.json"
        self._lock = RLock()
        self._bootstrap()

    def _bootstrap(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({"models": []})

    def _read_registry(self) -> dict[str, Any]:
        with self.registry_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise RuntimeError("registry.json has an invalid format.")
        return data

    def _write_registry(self, payload: dict[str, Any]) -> None:
        tmp_path = self.registry_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.registry_path)

    def list_records(self) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._read_registry()
            return list(payload["models"])

    def get_record(self, model_id: str) -> dict[str, Any] | None:
        records = self.list_records()
        for record in records:
            if record.get("model_id") == model_id:
                return record
        return None

    def add_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            payload = self._read_registry()
            if any(item.get("model_id") == record.get("model_id") for item in payload["models"]):
                raise ValueError(f"Model with id '{record.get('model_id')}' already exists.")
            payload["models"].append(record)
            self._write_registry(payload)

    def save_model(self, model_id: str, model: Any) -> Path:
        path = self.models_dir / f"{model_id}.joblib"
        joblib.dump(model, path)
        return path

    def load_model(self, model_id: str) -> Any:
        path = self.models_dir / f"{model_id}.joblib"
        if not path.exists():
            raise KeyError(f"Model artifact '{model_id}' does not exist.")
        return joblib.load(path)
