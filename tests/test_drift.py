import numpy as np
import pytest
import pandas as pd

from src import drift_detector
from src.drift_detector import calculate_psi


def test_identical_constant_populations_have_zero_drift():
    assert calculate_psi([7] * 30, [7] * 20) == 0


def test_constant_population_shift_is_detected():
    assert calculate_psi([7] * 30, [9] * 20) > 0.25


@pytest.mark.parametrize("reference, production", [([], [1]), ([1], []), ([1, np.inf], [1])])
def test_invalid_psi_samples_fail_clearly(reference, production):
    with pytest.raises(ValueError):
        calculate_psi(reference, production)


def test_missingness_shift_is_reported_even_with_stable_values(tmp_path, monkeypatch):
    reference = pd.DataFrame({"signal": [1.0] * 30, "churn": [0, 1] * 15})
    production = pd.DataFrame({"signal": [1.0] * 20 + [np.nan] * 10})
    reference.to_csv(tmp_path / "reference.csv", index=False)
    production.to_csv(tmp_path / "production.csv", index=False)
    monkeypatch.setattr(drift_detector, "REFERENCE_PATH", tmp_path / "reference.csv")
    monkeypatch.setattr(drift_detector, "PRODUCTION_PATH", tmp_path / "production.csv")
    monkeypatch.setattr(drift_detector, "OUTPUT_PATH", tmp_path / "drift.json")

    drift_detector.detect_drift()

    import json
    report = json.loads((tmp_path / "drift.json").read_text())
    assert report["overall_drift_detected"] is True
    assert report["features"]["signal"]["status"] == "MODERATE"
    assert report["features"]["signal"]["production_count"] == 20


@pytest.mark.parametrize("bins", [0, 1, 2.5, True, "10"])
def test_psi_rejects_noninteger_or_too_few_bins(bins):
    with pytest.raises(ValueError, match="PSI bins"):
        calculate_psi([1.0, 2.0], [2.0, 3.0], bins=bins)


def test_drift_fails_clearly_when_feature_type_changes(tmp_path, monkeypatch):
    reference = pd.DataFrame({"signal": [1.0, 2.0], "churn": [0, 1]})
    production = pd.DataFrame({"signal": ["bad", "values"]})
    reference.to_csv(tmp_path / "reference.csv", index=False)
    production.to_csv(tmp_path / "production.csv", index=False)
    monkeypatch.setattr(drift_detector, "REFERENCE_PATH", tmp_path / "reference.csv")
    monkeypatch.setattr(drift_detector, "PRODUCTION_PATH", tmp_path / "production.csv")
    monkeypatch.setattr(drift_detector, "OUTPUT_PATH", tmp_path / "drift.json")
    with pytest.raises(ValueError, match="signal.*numeric"):
        drift_detector.detect_drift()
    assert not (tmp_path / "drift.json").exists()
