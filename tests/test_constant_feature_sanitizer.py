import numpy as np
import pytest

from sentinelml.drift.sanitizer import DataSanitizer


def test_constant_feature_handles_extreme_finite_values():
    with np.errstate(over="raise", invalid="raise"):
        assert DataSanitizer.is_constant_feature([1e308, 1e308])
        assert DataSanitizer.is_constant_feature([-1e308, 1e308], tolerance=1.1e308)
        assert not DataSanitizer.is_constant_feature([-1e308, 1e308], tolerance=0.9e308)


def test_constant_feature_ignores_nonfinite_values_but_requires_finite_sample():
    assert DataSanitizer.is_constant_feature([np.nan, 7., np.inf, 7.])
    with pytest.raises(ValueError, match="finite samples"):
        DataSanitizer.is_constant_feature([np.nan, np.inf])


@pytest.mark.parametrize("tolerance", [True, -1., np.nan, np.inf, "bad"])
def test_constant_feature_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        DataSanitizer.is_constant_feature([1., 1.], tolerance=tolerance)
