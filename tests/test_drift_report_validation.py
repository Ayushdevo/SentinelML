import json

import pandas as pd
import pytest

from src import retraining_engine as engine


def health_inputs(tmp_path, monkeypatch, drift):
    for name, data in (("drift", drift), ("causes", {"high_risk_root_causes": {}})):
        (tmp_path / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    pd.DataFrame([{"is_anomaly": 0, "high_uncertainty": 0, "prediction_risk": "LOW"}]).to_csv(
        tmp_path / "production.csv", index=False
    )
    for attribute, name in (("DRIFT_PATH", "drift.json"), ("ROOT_CAUSE_PATH", "causes.json"),
                            ("PRODUCTION_PATH", "production.csv"), ("OUTPUT_PATH", "health.json")):
        monkeypatch.setattr(engine, attribute, tmp_path / name)


@pytest.mark.parametrize("feature", [None, {}, {"status": "UNKNOWN"}, {"status": "stable"}, {"status": []}])
def test_invalid_feature_status_cannot_produce_healthy_report(tmp_path, monkeypatch, feature):
    health_inputs(tmp_path, monkeypatch, {"features": {"signal": feature},
        "drifted_features": [], "number_of_drifted_features": 0})
    with pytest.raises(ValueError, match="Drift report"):
        engine.calculate_health()
    assert not (tmp_path / "health.json").exists()


@pytest.mark.parametrize("overall", [False, "true", 1])
def test_overall_flag_must_agree_with_feature_statuses(tmp_path, monkeypatch, overall):
    health_inputs(tmp_path, monkeypatch, {"features": {"signal": {"status": "HIGH"}},
        "drifted_features": ["signal"], "number_of_drifted_features": 1,
        "overall_drift_detected": overall})
    with pytest.raises(ValueError, match="inconsistent"):
        engine.calculate_health()
    assert not (tmp_path / "health.json").exists()


@pytest.mark.parametrize("status", ["STABLE", "LOW", "MODERATE", "HIGH"])
def test_documented_statuses_remain_supported(tmp_path, monkeypatch, status):
    drifted = ["signal"] if status in {"MODERATE", "HIGH"} else []
    health_inputs(tmp_path, monkeypatch, {"features": {"signal": {"status": status}},
        "drifted_features": drifted, "number_of_drifted_features": len(drifted)})
    engine.calculate_health()
    report = json.loads((tmp_path / "health.json").read_text())
    assert report["metrics"]["drift_ratio"] == (1 if drifted else 0)
