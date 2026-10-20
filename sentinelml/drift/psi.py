import numpy as np
from sentinelml.drift.sanitizer import DataSanitizer

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """Calculates the Population Stability Index between baseline and current data distributions."""
    baseline, current = DataSanitizer.sanitize_arrays(baseline, current)

    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(baseline, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bins)
    current_counts, _ = np.histogram(current, bins=bins)

    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    baseline_pct = np.clip(baseline_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi_value = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi_value)
