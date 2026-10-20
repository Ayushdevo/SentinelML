import numpy as np
from typing import Tuple

class DataSanitizer:
    """Pre-cleans and validates numeric feature distributions before drift computation."""
    
    @staticmethod
    def sanitize_arrays(baseline: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Removes NaN and infinite values, ensuring non-empty clean arrays."""
        baseline = np.asarray(baseline, dtype=float)
        current = np.asarray(current, dtype=float)
        if baseline.ndim != 1 or current.ndim != 1:
            raise ValueError("Distributions must be one-dimensional numeric samples")
        clean_base = baseline[np.isfinite(baseline)]
        clean_curr = current[np.isfinite(current)]
        
        if len(clean_base) == 0 or len(clean_curr) == 0:
            raise ValueError("Distribution contains no valid finite samples.")
            
        return clean_base, clean_curr

    @staticmethod
    def is_constant_feature(data: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Flags degenerate zero-variance features to prevent division by zero in bucket calculations."""
        return bool(np.std(data) < tolerance)
