import json

import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def valid_report():
    return {
        "health_score": 90.0,
        "health_status": "HEALTHY",
        "metrics": {},
        "drift": {},
        "top_high_risk_drivers": {},
        "retraining": {},
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(main, "MODEL_PATH", model)
    monkeypatch.setattr(main, "HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(main, "load_model", lambda: object())
    return TestClient(main.app)


@pytest.mark.parametrize(
    "field",
    [
        "health_score",
        "health_status",
        "metrics",
        "drift",
        "top_high_risk_drivers",
        "retraining",
    ],
)
@pytest.mark.parametrize("route", ["/health", "/ready"])
def test_missing_health_fields_are_not_ready(client, valid_report, field, route):
    del valid_report[field]
    main.HEALTH_PATH.write_text(json.dumps(valid_report), encoding="utf-8")
    assert client.get(route).status_code == 503


@pytest.mark.parametrize(
    "field,value",
    [
        ("health_score", True),
        ("health_score", "90"),
        ("health_score", -1),
        ("health_score", 101),
        ("health_status", "UNKNOWN"),
        ("health_status", []),
        ("metrics", []),
        ("drift", None),
        ("top_high_risk_drivers", 3),
        ("retraining", False),
    ],
)
@pytest.mark.parametrize("route", ["/health", "/ready"])
def test_invalid_health_field_types_are_rejected(
    client, valid_report, field, value, route
):
    valid_report[field] = value
    main.HEALTH_PATH.write_text(json.dumps(valid_report), encoding="utf-8")
    assert client.get(route).status_code == 503


def test_valid_health_report_keeps_optional_fields(client, valid_report):
    valid_report["scoring_note"] = "Review recommended"
    main.HEALTH_PATH.write_text(json.dumps(valid_report), encoding="utf-8")
    assert client.get("/health").json() == valid_report
    assert client.get("/ready").json() == {"status": "ready"}
