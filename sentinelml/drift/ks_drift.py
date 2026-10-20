from numbers import Real

import numpy as np
from scipy.stats import ks_2samp

from sentinelml.drift.sanitizer import DataSanitizer


def compute_ks_drift(baseline: np.ndarray, current: np.ndarray, threshold: float = 0.05) -> dict:
    """Run a two-sided KS test, using threshold as the significance level."""
    if (isinstance(threshold, bool) or not isinstance(threshold, Real)
            or not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("threshold must be finite and strictly between zero and one")
    base, curr = DataSanitizer.sanitize_arrays(baseline, current)
    result = ks_2samp(base, curr, alternative="two-sided", method="auto")
    critical_value = np.sqrt(-0.5 * np.log(threshold / 2)) * np.sqrt(
        (len(base) + len(curr)) / (len(base) * len(curr))
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "critical_value": float(critical_value),
        "is_drifted": bool(result.pvalue < threshold),
    }
