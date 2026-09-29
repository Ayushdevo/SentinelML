from fastapi.testclient import TestClient

from api import main


def test_readiness_lists_missing_artifacts(tmp_path, monkeypatch):
    model_path = tmp_path / "missing-model.joblib"
    health_path = tmp_path / "missing-health.json"
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "HEALTH_PATH", health_path)

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["missing_artifacts"] == [
        str(model_path),
        str(health_path),
    ]


def test_readiness_succeeds_when_required_artifacts_exist(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    health_path = tmp_path / "health.json"
    model_path.write_bytes(b"placeholder")
    health_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "HEALTH_PATH", health_path)

    response = TestClient(main.app).get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
