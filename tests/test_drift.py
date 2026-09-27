import numpy as np
import pytest

from src.drift_detector import calculate_psi


def test_identical_constant_populations_have_zero_drift():
    assert calculate_psi([7] * 30, [7] * 20) == 0


def test_constant_population_shift_is_detected():
    assert calculate_psi([7] * 30, [9] * 20) > 0.25


@pytest.mark.parametrize("reference, production", [([], [1]), ([1], []), ([1, np.inf], [1])])
def test_invalid_psi_samples_fail_clearly(reference, production):
    with pytest.raises(ValueError):
        calculate_psi(reference, production)
