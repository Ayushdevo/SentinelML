
from fastapi.testclient import TestClient
import pytest

from api import main


@pytest.mark.parametrize("route,attribute", [("/health", "HEALTH_PATH"), ("/metrics", "METRICS_PATH"), ("/ready", "HEALTH_PATH")])
def test_report_endpoints_handle_invalid_utf8(route, attribute, tmp_path, monkeypatch):
    report = tmp_path / "invalid.json"
    report.write_bytes(b'{"status":"\xff"}')
    monkeypatch.setattr(main, attribute, report)
    model = tmp_path / "model.joblib"
    model.write_bytes(b"placeholder")
    monkeypatch.setattr(main, "MODEL_PATH", model)
    response = TestClient(main.app).get(route)
    assert response.status_code == 503
    assert "unreadable" in response.json()["detail"]
