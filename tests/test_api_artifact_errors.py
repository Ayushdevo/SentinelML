
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


@pytest.mark.parametrize("route,attribute", [("/health", "HEALTH_PATH"), ("/metrics", "METRICS_PATH"), ("/ready", "HEALTH_PATH")])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_report_endpoints_reject_nonfinite_json_numbers(route, attribute, constant, tmp_path, monkeypatch):
    report = tmp_path / "nonfinite.json"
    report.write_text('{"score":' + constant + '}', encoding="utf-8")
    monkeypatch.setattr(main, attribute, report)
    model = tmp_path / "model.joblib"
    model.write_bytes(b"placeholder")
    monkeypatch.setattr(main, "MODEL_PATH", model)
    monkeypatch.setattr(main, "load_model", lambda: object())
    response = TestClient(main.app).get(route)
    assert response.status_code == 503
    assert "unreadable" in response.json()["detail"]


def test_strict_report_reader_preserves_strings_and_finite_values(tmp_path):
    from src.artifacts import read_json
    report = tmp_path / "finite.json"
    report.write_text('{"score":0.75,"note":"NaN","items":[1,null,true]}', encoding="utf-8")
    assert read_json(report) == {"score":0.75,"note":"NaN","items":[1,None,True]}
