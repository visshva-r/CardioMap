from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.explain import explain_patient

MODELS_DIR = ROOT / "models"

app = FastAPI(title="CardioMap API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/schema")
def get_schema() -> dict[str, Any]:
    path = MODELS_DIR / "schema.json"
    if not path.exists():
        raise HTTPException(503, "Models not trained. Run: python -m ml.train")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "Metrics missing. Run: python -m ml.train")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/patients/samples")
def sample_patients() -> list[dict[str, Any]]:
    path = MODELS_DIR / "sample_patients.json"
    if not path.exists():
        raise HTTPException(503, "Sample patients missing. Run: python -m ml.train")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/importance/global")
def global_importance() -> dict[str, Any]:
    path = MODELS_DIR / "global_shap.json"
    if not path.exists():
        raise HTTPException(503, "Run: python -m ml.explain after training")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    if not (MODELS_DIR / "cath.joblib").exists():
        raise HTTPException(503, "Models not trained. Run: python -m ml.train")
    try:
        return explain_patient(body.features)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Prediction failed: {exc}") from exc


WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="frontend")
