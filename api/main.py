from pathlib import Path
import json
import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from src.model_contracts import model_feature_order, positive_class_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
HEALTH_PATH = PROJECT_ROOT / "models" / "model_health_report.json"

app = FastAPI(
    title="SentinelML API",
    description="ML prediction and model-health monitoring API",
    version="1.0.0",
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Baseline model is not available")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Unable to load the baseline model")
        raise HTTPException(status_code=503, detail="Baseline model is unreadable") from exc
    if not callable(getattr(model, "predict_proba", None)):
        raise HTTPException(status_code=503, detail="Baseline model does not support prediction")
    return model


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


@app.get("/")
def root():
    return {
        "service": "SentinelML",
        "status": "running",
    }


@app.get("/health")
def model_health():
    if not HEALTH_PATH.exists():
        raise HTTPException(status_code=503, detail="Model health report is not available")
    try:
        with open(HEALTH_PATH, "r", encoding="utf-8") as file:
            health = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503, detail="Model health report is unreadable"
        ) from exc

    return health


@app.get("/ready")
def readiness():
    missing = [str(path) for path in (MODEL_PATH, HEALTH_PATH) if not path.exists()]
    if missing:
        raise HTTPException(status_code=503, detail={"missing_artifacts": missing})
    try:
        with open(HEALTH_PATH, "r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503, detail="Model health report is unreadable"
        ) from exc
    if not isinstance(report, dict):
        raise HTTPException(status_code=503, detail="Invalid model health report")
    load_model()
    return {"status": "ready"}


@app.post("/predict")
def predict(data: PredictionRequest):
    model = load_model()

    input_df = pd.DataFrame(
        [
            {
                "tenure": data.tenure,
                "monthly_charges": data.monthly_charges,
                "usage_hours": data.usage_hours,
                "support_calls": data.support_calls,
                "late_payments": data.late_payments,
                "contract_score": data.contract_score,
                "engagement_score": data.engagement_score,
                "service_count": data.service_count,
            }
        ]
    )
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
        probabilities.shape != (1, 2)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    ):
        raise HTTPException(
            status_code=503, detail="Model returned invalid probabilities"
        )
    probability = float(probabilities[0, positive_index])

    prediction = int(
        probability >= 0.5
    )

    uncertainty = float(
        1 - abs(probability - 0.5) * 2
    )

    if uncertainty >= 0.60:
        risk = "HIGH_UNCERTAINTY"
    else:
        risk = "NORMAL"

    return {
        "prediction": prediction,
        "churn_probability": round(
            probability,
            4,
        ),
        "uncertainty": round(
            uncertainty,
            4,
        ),
        "prediction_risk": risk,
    }
