import pytest
from fastapi.testclient import TestClient

from api import main


FEATURES = list(main.PredictionRequest.model_fields)


@pytest.mark.parametrize("names", [
    FEATURES[:-1], FEATURES + ["unexpected"], [FEATURES[0]] * len(FEATURES),
    ["unexpected"] + FEATURES[1:], None, "tenure", [],
])
def test_invalid_model_feature_metadata_returns_503(monkeypatch, names):
    class Model:
        feature_names_in_ = names

        def predict_proba(self, frame):
            raise AssertionError("Invalid metadata must not reach model prediction")

    monkeypatch.setattr(main, "load_model", lambda: Model())
    response = TestClient(main.app).post("/predict", json=dict.fromkeys(FEATURES, 1.0))
    assert response.status_code == 503
    assert response.json()["detail"] == "Model feature names do not match prediction schema"


def test_prediction_reorders_valid_model_features(monkeypatch):
    payload = {name: float(index + 1) for index, name in enumerate(FEATURES)}

    class Model:
        feature_names_in_ = FEATURES[::-1]

        def predict_proba(self, frame):
            assert list(frame.columns) == self.feature_names_in_
            assert frame.iloc[0].tolist() == list(payload.values())[::-1]
            return [[0.2, 0.8]]

    monkeypatch.setattr(main, "load_model", lambda: Model())
    response = TestClient(main.app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.8
