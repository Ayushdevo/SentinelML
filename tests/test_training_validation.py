"""Fail before model fitting when source features contain invalid values."""
import numpy as np
import pandas as pd
import pytest

from src import train as training


def test_training_rejects_infinite_features(tmp_path, monkeypatch):
    path = tmp_path / "reference.csv"
    pd.DataFrame({
        "signal": [1., 2., 3., 4., 5., np.inf, 7., 8., 9., 10., 11., 12.],
        "churn": [0, 1] * 6,
    }).to_csv(path, index=False)
    monkeypatch.setattr(training, "DATA_PATH", path)
    with pytest.raises(ValueError, match="finite"):
        training.train()


def test_training_rejects_non_numeric_features(tmp_path, monkeypatch):
    path = tmp_path / "reference.csv"
    pd.DataFrame({
        "signal": ["a", "b"] * 6,
        "churn": [0, 1] * 6,
    }).to_csv(path, index=False)
    monkeypatch.setattr(training, "DATA_PATH", path)
    with pytest.raises(ValueError, match="numeric"):
        training.train()
