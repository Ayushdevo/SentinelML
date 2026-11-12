import numpy as np
import pytest

from sentinelml.drift.psi import calculate_psi


def test_psi_distinguishes_extreme_opposite_constants():
    reference = np.full(5, 1e308)
    current = np.full(5, -1e308)
    with np.errstate(over="raise", invalid="raise"):
        value = calculate_psi(reference, current)
    assert value > 1.0
    assert np.isfinite(value)


def test_psi_extreme_magnitudes_match_rescaled_distributions():
    baseline = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    current = np.array([-0.8, -0.1, 0.2, 0.7, 0.9])
    expected = calculate_psi(baseline, current, num_buckets=4)
    with np.errstate(over="raise", invalid="raise"):
        actual = calculate_psi(baseline * 1e308, current * 1e308, num_buckets=4)
    assert actual == pytest.approx(expected)


def test_pipeline_psi_distinguishes_extreme_opposite_constants():
    from src.drift_detector import calculate_psi as pipeline_psi

    reference = np.full(5, 1e308)
    current = np.full(5, -1e308)
    with np.errstate(over="raise", invalid="raise"):
        value = pipeline_psi(reference, current)
    assert value > 1.0
    assert np.isfinite(value)


def test_pipeline_psi_matches_rescaled_distributions():
    from src.drift_detector import calculate_psi as pipeline_psi

    baseline = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    current = np.array([-0.8, -0.1, 0.2, 0.7, 0.9])
    expected = pipeline_psi(baseline, current, bins=4)
    with np.errstate(over="raise", invalid="raise"):
        actual = pipeline_psi(baseline * 1e308, current * 1e308, bins=4)
    assert actual == pytest.approx(expected)
