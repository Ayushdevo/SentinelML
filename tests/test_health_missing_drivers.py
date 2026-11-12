import json

import pytest

from src import retraining_engine as engine
from test_monitoring_contracts import setup_health


def test_health_rejects_missing_high_risk_driver_analysis(tmp_path, monkeypatch):
    setup_health(tmp_path, monkeypatch, {
        "is_anomaly": 1, "high_uncertainty": 1, "prediction_risk": "HIGH",
    })
    (tmp_path / "causes.json").write_text(json.dumps({
        "high_risk_observations": 1, "high_risk_root_causes": {},
    }), encoding="utf-8")
    output = tmp_path / "out.json"
    output.write_text('{"previous": "report"}', encoding="utf-8")
    with pytest.raises(ValueError, match="high-risk.*root-cause"):
        engine.calculate_health()
    assert json.loads(output.read_text()) == {"previous": "report"}


def test_health_allows_empty_drivers_when_no_high_risk_observations(tmp_path, monkeypatch):
    setup_health(tmp_path, monkeypatch, {
        "is_anomaly": 0, "high_uncertainty": 0, "prediction_risk": "LOW",
    })
    (tmp_path / "causes.json").write_text(json.dumps({
        "high_risk_observations": 0, "high_risk_root_causes": {},
    }), encoding="utf-8")
    engine.calculate_health()
    assert json.loads((tmp_path / "out.json").read_text())["top_high_risk_drivers"] == {}
