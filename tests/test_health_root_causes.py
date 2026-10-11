import json

import pytest

from src import retraining_engine as engine
from test_monitoring_contracts import setup_health


def root_cause_inputs(tmp_path, monkeypatch, causes, count=1):
    setup_health(tmp_path, monkeypatch,
        {"is_anomaly": 1, "high_uncertainty": 1, "prediction_risk": "HIGH"})
    (tmp_path / "causes.json").write_text(json.dumps({
        "high_risk_root_causes": causes, "high_risk_observations": count,
    }), encoding="utf-8")


def test_top_drivers_are_sorted_by_contribution(tmp_path, monkeypatch):
    root_cause_inputs(tmp_path, monkeypatch,
        {"weak": 1, "strong": 50, "medium": 20, "another": 15, "small": 10, "rest": 4})
    engine.calculate_health()
    report = json.loads((tmp_path / "out.json").read_text())
    assert list(report["top_high_risk_drivers"]) == ["strong", "medium", "another", "small", "rest"]


@pytest.mark.parametrize("percentage", [None, True, "10", float("nan"), float("inf"), -1, 101, 10**400])
def test_invalid_percentages_do_not_overwrite_report(tmp_path, monkeypatch, percentage):
    root_cause_inputs(tmp_path, monkeypatch, {"signal": percentage})
    output = tmp_path / "out.json"
    output.write_text('{"previous": "report"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Root-cause contributions"):
        engine.calculate_health()
    assert json.loads(output.read_text()) == {"previous": "report"}


@pytest.mark.parametrize("count", [0, 2, True, "1"])
def test_stale_observation_counts_are_rejected(tmp_path, monkeypatch, count):
    root_cause_inputs(tmp_path, monkeypatch, {"signal": 100}, count)
    with pytest.raises(ValueError, match="observation count"):
        engine.calculate_health()
    assert not (tmp_path / "out.json").exists()
