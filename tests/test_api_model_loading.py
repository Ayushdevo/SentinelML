import joblib
import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture(autouse=True)
def clear_model_cache():
    main.load_model.cache_clear()
    yield
    main.load_model.cache_clear()


@pytest.mark.parametrize("path_kind, detail", [
    ("corrupt", "Baseline model is unreadable"),
    ("directory", "Baseline model is unreadable"),
    ("wrong_object", "Baseline model does not support prediction"),
])
def test_invalid_artifacts_return_503_for_prediction_and_readiness(
    tmp_path, monkeypatch, path_kind, detail
):
    model_path = tmp_path / "model.joblib"
    if path_kind == "directory":
        model_path.mkdir()
    elif path_kind == "wrong_object":
        joblib.dump({"not": "a model"}, model_path)
    else:
        model_path.write_bytes(b"corrupt artifact")
    report = tmp_path / "health.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "HEALTH_PATH", report)
    client = TestClient(main.app)
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    for response in (client.post("/predict", json=payload), client.get("/ready")):
        assert response.status_code == 503
        assert response.json()["detail"] == detail
