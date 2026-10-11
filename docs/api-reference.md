# API Reference

SentinelML exposes a FastAPI service from `api/main.py`.

## `GET /`

Basic liveness endpoint.

Example response:

```json
{
  "service": "SentinelML",
  "status": "running"
}
```

## `GET /ready`

Readiness check for required runtime artifacts:

- `models/baseline_model.joblib`
- `models/model_health_report.json`

Returns HTTP 503 with the missing artifact paths if either file is unavailable.
Readiness also loads the model and checks that it supports probability
prediction. Corrupt or incompatible artifacts return HTTP 503.

## `GET /health`

Returns the current model-health report. The endpoint returns HTTP 503 when the report has not been generated.

## `POST /predict`

Required numeric fields:

- `tenure`
- `monthly_charges`
- `usage_hours`
- `support_calls`
- `late_payments`
- `contract_score`
- `engagement_score`
- `service_count`

Unknown fields are rejected. NaN and infinite values are rejected by request validation.

The response contains the binary prediction, churn probability, uncertainty score, and a prediction-risk label.

## Operational note

Fitted feature names must match the request schema exactly; valid names are
reordered into the fitted model's order. Incompatible feature names or churn
class labels return HTTP 503. Churn classes must be numeric 0 and 1, and the
positive probability is selected by class label rather than column position.
