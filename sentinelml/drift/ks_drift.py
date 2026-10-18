import numpy as np
from sentinelml.drift.sanitizer import DataSanitizer

def compute_ks_drift(baseline: np.ndarray, current: np.ndarray, threshold: float = 0.05) -> dict:
    """
    Computes the two-sample Kolmogorov-Smirnov statistic to detect continuous distribution drift.
    Returns the maximum vertical ECDF distance (D-statistic) and a drift flag.
    """
    base, curr = DataSanitizer.sanitize_arrays(baseline, current)
    
    n_base, n_curr = len(base), len(curr)
    all_values = np.sort(np.unique(np.concatenate([base, curr])))
    
    # Calculate empirical CDFs over unified evaluation points
    cdf_base = np.searchsorted(np.sort(base), all_values, side='right') / n_base
    cdf_curr = np.searchsorted(np.sort(curr), all_values, side='right') / n_curr
    
    # Kolmogorov-Smirnov statistic D is the supremum difference
    d_stat = float(np.max(np.abs(cdf_base - cdf_curr)))
    
    # Asymptotic critical value approximation at alpha = 0.05
    critical_val = 1.36 * np.sqrt((n_base + n_curr) / (n_base * n_curr))
    is_drifted = bool(d_stat > critical_val or d_stat > threshold)
    
    return {
        "statistic": d_stat,
        "critical_value": float(critical_val),
        "is_drifted": is_drifted
    }
