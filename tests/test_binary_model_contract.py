from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main
from src import anomaly_detector, root_cause
from src.model_contracts import positive_class_index


INVALID_CLASSES = [[2, 1], [1, 1], [False, True],
    [np.bool_(False), np.bool_(True)], [0, 1, 2], ["0", "1"], None]


@pytest.mark.parametrize("classes", INVALID_CLASSES)
def test_invalid_class_labels_fail_in_api_and_monitoring(tmp_path, monkeypatch, classes):
    frame = pd.DataFrame({"signal": [1., 2.], "prediction_risk": ["LOW", "HIGH"]})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    model = SimpleNamespace(classes_=classes)
    monkeypatch.setattr(main, "load_model", lambda: model)
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    response = TestClient(main.app).post("/predict", json=payload)
    assert response.status_code == 503
    assert "binary churn classes" in response.json()["detail"]
    for module, run in ((anomaly_detector, anomaly_detector.analyze_production),
                        (root_cause, root_cause.analyze_root_causes)):
        monkeypatch.setattr(module, "PRODUCTION_PATH", path)
        if module is anomaly_detector:
            monkeypatch.setattr(module, "REFERENCE_PATH", path)
        monkeypatch.setattr(module, "OUTPUT_PATH", tmp_path / "output.json")
        monkeypatch.setattr(module.joblib, "load", lambda _: model)
        with pytest.raises(ValueError, match="binary churn classes"):
            run()
        assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("classes, index", [([0, 1], 1), ([1, 0], 0),
    ([np.int64(1), np.int64(0)], 0)])
def test_positive_class_index_preserves_numeric_class_order(classes, index):
    assert positive_class_index(SimpleNamespace(classes_=classes)) == index
