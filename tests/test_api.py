import pytest
from fastapi.testclient import TestClient

from api import main


def test_missing_model_returns_503_without_preventing_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.joblib")
    main.load_model.cache_clear()
    client = TestClient(main.app)
    assert client.get("/").status_code == 200
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def test_invalid_payload_is_rejected_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.joblib")
    main.load_model.cache_clear()
    client = TestClient(main.app)
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    payload["extra_feature"] = 2
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_health_report_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "HEALTH_PATH", tmp_path / "missing.json")
    assert TestClient(main.app).get("/health").status_code == 503


def test_corrupt_health_report_returns_503(tmp_path, monkeypatch):
    report = tmp_path / "health.json"
    report.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(main, "HEALTH_PATH", report)
    response = TestClient(main.app).get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model health report is unreadable"


@pytest.mark.parametrize("contents", ["[]", '"healthy"', "42", "null"])
def test_nonobject_health_report_returns_503(tmp_path, monkeypatch, contents):
    report = tmp_path / "health.json"
    report.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(main, "HEALTH_PATH", report)

    response = TestClient(main.app).get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Invalid model health report"


def test_predict_uses_label_one_when_model_class_order_is_reversed(monkeypatch):
    import numpy as np

    class ReversedModel:
        classes_ = np.array([1, 0])
        feature_names_in_ = list(main.PredictionRequest.model_fields)

        def predict_proba(self, frame):
            assert list(frame.columns) == list(self.feature_names_in_)
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr(main, "load_model", lambda: ReversedModel())
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    response = TestClient(main.app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.8
    assert response.json()["prediction"] == 1


@pytest.mark.parametrize("probs", [[[float("nan"), 0.5]], [[1.5, -0.5]], [[0.5]]])
def test_predict_rejects_invalid_model_probabilities(probs, monkeypatch):
    class InvalidModel:
        classes_ = [0, 1]

        def predict_proba(self, frame):
            return probs

    monkeypatch.setattr(main, "load_model", lambda: InvalidModel())
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    response = TestClient(main.app).post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model returned invalid probabilities"


def test_prediction_runtime_failure_returns_503_and_is_logged(monkeypatch, caplog):
    import logging

    class FailingModel:
        classes_ = [0, 1]

        def predict_proba(self, frame):
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr(main, "load_model", lambda: FailingModel())
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)

    with caplog.at_level(logging.ERROR, logger=main.__name__):
        response = TestClient(main.app).post("/predict", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model prediction failed"
    assert "Model prediction failed" in caplog.text
