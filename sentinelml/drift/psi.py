from numbers import Integral, Real

import numpy as np
from sentinelml.drift.sanitizer import DataSanitizer

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """Calculates the Population Stability Index between baseline and current data distributions."""
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, Integral) or num_buckets < 2:
        raise ValueError("num_buckets must be an integer of at least two")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, Real)
            or not np.isfinite(epsilon) or not 0 < epsilon < 1):
        raise ValueError("epsilon must be finite and strictly between zero and one")
    baseline, current = DataSanitizer.sanitize_arrays(baseline, current)

    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.unique(np.percentile(baseline, quantiles))
    if len(bins) < 3:
        low = min(baseline.min(), current.min())
        high = max(baseline.max(), current.max())
        if low == high:
            return 0.0
        bins = np.linspace(low, high, num_buckets + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bins)
    current_counts, _ = np.histogram(current, bins=bins)

    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    baseline_pct = np.clip(baseline_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)
    baseline_pct /= baseline_pct.sum()
    current_pct /= current_pct.sum()

    psi_value = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi_value)
