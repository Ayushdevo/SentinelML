# Operations Runbook

## Standard pipeline

Run from the repository root:

```bash
python -m src.run_pipeline
```

Runtime artifact paths are resolved from the project files, so the API and
dashboard continue to find model and report files when launched from another
working directory.

The pipeline performs:

1. dataset generation when both source datasets are absent
2. baseline training when the model artifact is absent
3. feature drift detection
4. anomaly and prediction-uncertainty analysis
5. SHAP root-cause analysis
6. model-health calculation

## Expected outputs

Key generated artifacts include:

- `models/drift_report.json`
- `models/production_analysis.csv`
- `models/root_cause_report.json`
- `models/model_health_report.json`

The dashboard's HIGH-risk observations section can export the displayed
observations as CSV for offline review. The download is disabled when there
are no HIGH-risk rows.

## Failure handling

### Only one dataset exists

The pipeline deliberately stops if only one of `data/reference.csv` or `data/production.csv` exists. Restore the matching file or remove both before regeneration.

### Model unavailable

The API prediction endpoint returns HTTP 503 until the baseline model exists.

### Health report unavailable

The health endpoint returns HTTP 503 until the monitoring pipeline has produced a health report.

### Schema mismatch

Do not coerce or silently drop unexpected features. Reconcile the data/model feature contract and rerun validation.

## Retraining

Treat a retraining recommendation as an escalation for review. Validate candidate models against labeled holdout data and regression checks before any deployment decision.
