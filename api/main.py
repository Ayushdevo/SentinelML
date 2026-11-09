import logging
from functools import lru_cache
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.artifacts import read_json
from src.model_contracts import model_feature_order, positive_class_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
HEALTH_PATH = PROJECT_ROOT / "models" / "model_health_report.json"
METRICS_PATH = PROJECT_ROOT / "models" / "baseline_metrics.json"
BASELINE_METRIC_NAMES = {"accuracy", "precision", "recall", "f1", "roc_auc"}

app = FastAPI(
    title="SentinelML API",
    description="ML prediction and model-health monitoring API",
    version="1.0.0",
)

logger = logging.getLogger(__name__)


_model_lock = RLock()


@lru_cache(maxsize=1)
def _load_model_version(path, signature):
    try:
        model = joblib.load(path)
    except Exception as exc:
        logger.exception("Unable to load the baseline model")
        raise HTTPException(
            status_code=503, detail="Baseline model is unreadable"
        ) from exc
    if not callable(getattr(model, "predict_proba", None)):
        raise HTTPException(
            status_code=503, detail="Baseline model does not support prediction"
        )
    return model


def load_model():
    """Reload on artifact changes while reusing unchanged models within a worker.

    Deploy new artifacts with an atomic replacement. Inode and ctime are included
    so replacements preserving mtime and file size also invalidate the cache.
    """
    with _model_lock:
        path = Path(MODEL_PATH).absolute()
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail="Baseline model is not available"
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail="Baseline model is unreadable"
            ) from exc
        signature = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        return _load_model_version(path, signature)


# Preserve the existing explicit cache-reset hook used by tests and operators.
load_model.cache_clear = _load_model_version.cache_clear


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tenure: float
    monthly_charges: float
    usage_hours: float
    support_calls: float
    late_payments: float
    contract_score: float
    engagement_score: float
    service_count: float


class PredictionBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: list[PredictionRequest] = Field(min_length=1, max_length=100)


class HealthReport(BaseModel):
    """Required top-level contract, shared by health reads and readiness."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    health_score: float = Field(ge=0, le=100)
    health_status: Literal["HEALTHY", "WATCH", "DEGRADED", "CRITICAL"]
    metrics: dict
    drift: dict
    top_high_risk_drivers: dict
    retraining: dict


@app.get("/")
def root():
    return {
        "service": "SentinelML",
        "status": "running",
    }


@app.get("/health")
def model_health():
    if not HEALTH_PATH.exists():
        raise HTTPException(
            status_code=503, detail="Model health report is not available"
        )
    try:
        health = read_json(HEALTH_PATH)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="Model health report is unreadable"
        ) from exc
    try:
        HealthReport.model_validate(health)
    except ValidationError as exc:
        raise HTTPException(
            status_code=503, detail="Invalid model health report"
        ) from exc

    return health


@app.get("/metrics")
def baseline_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503, detail="Baseline metrics are not available"
        )
    try:
        metrics = read_json(METRICS_PATH)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="Baseline metrics are unreadable"
        ) from exc

    if (
        not isinstance(metrics, dict)
        or set(metrics) != BASELINE_METRIC_NAMES
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
            for value in metrics.values()
        )
    ):
        raise HTTPException(status_code=503, detail="Invalid baseline metrics report")
    return metrics


@app.get("/ready")
def readiness():
    missing = [str(path) for path in (MODEL_PATH, HEALTH_PATH) if not path.exists()]
    if missing:
        raise HTTPException(status_code=503, detail={"missing_artifacts": missing})
    model_health()
    load_model()
    return {"status": "ready"}


@app.post("/predict")
def predict(data: PredictionRequest):
    return _predict_records([data])[0]


@app.post("/predict/batch")
def predict_batch(data: PredictionBatchRequest):
    return {"predictions": _predict_records(data.predictions)}


def _predict_records(records: list[PredictionRequest]):
    model = load_model()

    input_df = pd.DataFrame([record.model_dump() for record in records])
    try:
        input_df = input_df[model_feature_order(model, input_df.columns)]
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Predictors may expose class probabilities in a nonstandard order.
    try:
        positive_index = positive_class_index(model)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        probabilities = np.asarray(model.predict_proba(input_df), dtype=float)
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=503, detail="Model prediction failed") from exc
    if (
        probabilities.shape != (len(records), 2)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    ):
        raise HTTPException(
            status_code=503, detail="Model returned invalid probabilities"
        )
    results = []
    for probability in probabilities[:, positive_index]:
        probability = float(probability)
        prediction = int(probability >= 0.5)
        uncertainty = float(1 - abs(probability - 0.5) * 2)
        results.append(
            {
                "prediction": prediction,
                "churn_probability": round(probability, 4),
                "uncertainty": round(uncertainty, 4),
                "prediction_risk": (
                    "HIGH_UNCERTAINTY" if uncertainty >= 0.60 else "NORMAL"
                ),
            }
        )
    return results
