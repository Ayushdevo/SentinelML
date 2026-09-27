from pathlib import Path
import json
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


MODEL_PATH = Path("models/baseline_model.joblib")
HEALTH_PATH = Path("models/model_health_report.json")

app = FastAPI(
    title="SentinelML API",
    description="ML prediction and model-health monitoring API",
    version="1.0.0",
)

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Baseline model is not available")
    return joblib.load(MODEL_PATH)


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
    with open(
        HEALTH_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        health = json.load(file)

    return health


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

    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

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
