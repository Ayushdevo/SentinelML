# Monitoring Contract

This document defines the current meaning of SentinelML monitoring outputs.
Also thier non-existable answer behaviour that takes large data sets to minimize the operations.

## Drift

Feature drift combines Population Stability Index (PSI) with a Kolmogorov-Smirnov test.
KS p-values are corrected across all monitored features with the Holm
family-wise error procedure. Reports retain both the raw `p_value` and
`adjusted_p_value`; drift classification uses the adjusted value.

Current PSI interpretation:

- `PSI < 0.10`: stable unless the KS test indicates a statistically significant shift
- `0.10 <= PSI < 0.25`: moderate drift
- `PSI >= 0.25`: high drift

A missing-value-rate shift of at least 10 percentage points upgrades otherwise stable/low drift to moderate.
Production data must contain every reference feature and may contain the
target column, but any other unexpected columns are rejected before a report
is written.

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

Health generation rejects unknown drift statuses or conflicting drift summaries.
Root-cause contributions must be finite percentages from 0 to 100. When a
root-cause observation count is present, it must match the production HIGH-risk
row count. The five displayed drivers are sorted by contribution, independent
of the input JSON key order. Invalid reports do not overwrite the current health
report.

## Compatibility

Changes to thresholds, report field names, or risk semantics should be documented and covered by tests because dashboards and downstream consumers may rely on them.


## Standalone numeric drift utilities

The `sentinelml.drift` helpers accept one-dimensional numeric array-like
samples. They remove NaN and infinite observations independently from each
sample and reject samples with no finite observations. This differs from the
pipeline's `src.drift_detector`, which rejects nonfinite values after handling
missing data. Filtering does not mutate the caller's input.

Standalone PSI requires at least two integer buckets and a finite smoothing
`epsilon` strictly between zero and one. Boolean configuration values are
rejected rather than interpreted as numbers.

PSI removes repeated reference quantile boundaries. When the reference has too
few distinct quantiles, it uses equal-width boundaries over both samples;
identical constant populations return zero. Smoothed histogram probabilities
are renormalized before scoring. Shifts between different constant populations
therefore remain detectable instead of silently returning zero.

The standalone `compute_ks_drift` interprets `threshold` as a significance level
(default 0.05), not a minimum ECDF distance. It returns SciPy's two-sided KS
`statistic` and `p_value`, and flags drift only when `p_value < threshold`.
`critical_value` is an alpha-dependent asymptotic diagnostic; the decision uses
the p-value. This corrects earlier false alarms on small samples. The pipeline's
Holm-adjusted drift calculation is unchanged.
