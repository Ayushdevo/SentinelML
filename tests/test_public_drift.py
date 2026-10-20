
import numpy as np
import pytest
from scipy.stats import ks_2samp

from sentinelml.drift.sanitizer import DataSanitizer
from sentinelml.drift.psi import calculate_psi
from sentinelml.drift.ks_drift import compute_ks_drift


def test_sanitizer_accepts_numeric_array_likes_without_mutation():
    reference = [1.0, np.nan, 2.0]
    current = (3.0, np.inf, 4.0)
    first, second = DataSanitizer.sanitize_arrays(reference, current)
    np.testing.assert_array_equal(first, [1.0, 2.0])
    np.testing.assert_array_equal(second, [3.0, 4.0])
    assert len(reference) == 3 and np.isnan(reference[1])


@pytest.mark.parametrize("bad", [1.0, [[1.0, 2.0]], [[1.0], [2.0]]])
def test_sanitizer_rejects_nonvector_samples(bad):
    with pytest.raises(ValueError, match="one-dimensional"):
        DataSanitizer.sanitize_arrays(bad, [1.0, 2.0])


def test_public_psi_filters_nonfinite_samples_consistently():
    clean = calculate_psi([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    dirty = calculate_psi([0.0, np.nan, 1.0, 2.0], [0.0, 1.0, np.inf, 2.0])
    assert dirty == pytest.approx(clean)


@pytest.mark.parametrize("bad", [[], [np.nan, np.inf]])
def test_public_psi_rejects_distributions_without_finite_samples(bad):
    with pytest.raises(ValueError, match="finite samples"):
        calculate_psi(bad, [1.0, 2.0])


@pytest.mark.parametrize("buckets", [True, 0, 1, 2.5, "10"])
def test_public_psi_requires_at_least_two_integer_buckets(buckets):
    with pytest.raises(ValueError, match="num_buckets"):
        calculate_psi([1.0, 2.0], [2.0, 3.0], num_buckets=buckets)


@pytest.mark.parametrize("epsilon", [True, 0, -0.1, 1.0, float("nan"), float("inf"), "small"])
def test_public_psi_rejects_invalid_smoothing(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        calculate_psi([1.0, 2.0], [2.0, 3.0], epsilon=epsilon)


def test_public_psi_detects_constant_population_shifts():
    assert calculate_psi([0.0] * 20, [1.0] * 10) > 0.25
    assert calculate_psi([7.0] * 20, [7.0] * 10) == 0


def test_public_psi_handles_repeated_quantiles_and_sample_sizes():
    reference = [0.0] * 90 + [1.0] * 10
    assert calculate_psi(reference, reference * 2) == pytest.approx(0)
    score = calculate_psi(reference, [0.0] * 10 + [1.0] * 90)
    assert np.isfinite(score) and score > 0.25


def test_public_ks_uses_significance_instead_of_a_fixed_distance_cutoff():
    baseline = np.arange(10, dtype=float)
    current = baseline + 0.1
    result = compute_ks_drift(baseline, current)
    expected = ks_2samp(baseline, current)
    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["is_drifted"] == bool(expected.pvalue < 0.05)
    assert not result["is_drifted"]


@pytest.mark.parametrize("threshold", [True, 0, 1, -0.1, np.nan, np.inf, "0.05"])
def test_public_ks_rejects_invalid_significance_levels(threshold):
    with pytest.raises(ValueError, match="threshold"):
        compute_ks_drift([0.0, 1.0], [1.0, 2.0], threshold=threshold)


def test_public_ks_reports_alpha_dependent_critical_values():
    reference = np.arange(30, dtype=float)
    strict = compute_ks_drift(reference, reference + 0.1, threshold=0.01)
    loose = compute_ks_drift(reference, reference + 0.1, threshold=0.10)
    assert strict["critical_value"] > loose["critical_value"]
