"""The prediction endpoint should reject internally inconsistent model outputs."""
from fastapi.testclient import TestClient

from api import main


def test_prediction_rejects_probabilities_that_do_not_sum_to_one(monkeypatch):
    class InvalidModel:
        classes_ = [0, 1]

        def predict_proba(self, frame):
            return [[0.4, 0.4]]

    monkeypatch.setattr(main, "load_model", lambda: InvalidModel())
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    response = TestClient(main.app).post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model returned invalid probabilities"
