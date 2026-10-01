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
