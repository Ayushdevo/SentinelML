# Monitoring Contract

This document defines the current meaning of SentinelML monitoring outputs.

## Drift

Feature drift combines Population Stability Index (PSI) with a Kolmogorov-Smirnov test.

Current PSI interpretation:

- `PSI < 0.10`: stable unless the KS test indicates a statistically significant shift
- `0.10 <= PSI < 0.25`: moderate drift
- `PSI >= 0.25`: high drift

A missing-value-rate shift of at least 10 percentage points upgrades otherwise stable/low drift to moderate.

## Prediction-level risk

Production rows are scored for anomaly status and prediction uncertainty.

- anomaly: Isolation Forest returns `-1`
- uncertainty: `1 - 2 * abs(p - 0.5)`
- high uncertainty: uncertainty is at least `0.60`

Risk labels:

- `HIGH`: anomaly and high uncertainty
- `MEDIUM`: anomaly or high uncertainty
- `LOW`: neither condition

## Model health

The health score is a heuristic monitoring score, not a direct estimate of real-world model accuracy. It applies bounded penalties for drift, anomalies, uncertainty, and high-risk predictions.

A retraining recommendation means the model should be reviewed with labeled validation data. It does not authorize automatic deployment of a replacement model.

## Compatibility

Changes to thresholds, report field names, or risk semantics should be documented and covered by tests because dashboards and downstream consumers may rely on them.
