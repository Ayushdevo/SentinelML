import os

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api import main


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", path)
    main.load_model.cache_clear()
    yield path, TestClient(main.app)
    main.load_model.cache_clear()


def save_model(path, prediction):
    model = DummyClassifier(strategy="constant", constant=prediction)
    model.fit([[0] * 8, [1] * 8], [0, 1])
    joblib.dump(model, path)


def test_predictions_reload_after_atomic_replacement(runtime, monkeypatch):
    path, client = runtime
    save_model(path, 0)
    original_load = main.joblib.load
    loads = []

    def counted_load(*args, **kwargs):
        loads.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(main.joblib, "load", counted_load)
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    for _ in range(2):
        assert client.post("/predict", json=payload).json()["prediction"] == 0
    assert len(loads) == 1

    replacement = path.with_name("replacement.joblib")
    save_model(replacement, 1)
    # Replacement must be detected even when deployment preserves timestamps.
    timestamp = path.stat().st_mtime_ns
    os.utime(replacement, ns=(timestamp, timestamp))
    os.replace(replacement, path)
    for _ in range(2):
        assert client.post("/predict", json=payload).json()["prediction"] == 1
    assert len(loads) == 2


@pytest.mark.parametrize("replacement", ["corrupt", "deleted"])
def test_cached_model_does_not_hide_invalid_replacement(runtime, replacement):
    path, client = runtime
    save_model(path, 0)
    payload = dict.fromkeys(main.PredictionRequest.model_fields, 1.0)
    assert client.post("/predict", json=payload).status_code == 200
    if replacement == "corrupt":
        path.write_bytes(b"broken model")
    else:
        path.unlink()
    assert client.post("/predict", json=payload).status_code == 503
    save_model(path, 1)
    assert client.post("/predict", json=payload).json()["prediction"] == 1


def test_cache_identity_includes_model_path(runtime, monkeypatch):
    path, _ = runtime
    save_model(path, 0)
    first = main.load_model()
    other = path.with_name("other.joblib")
    save_model(other, 1)
    monkeypatch.setattr(main, "MODEL_PATH", other)
    assert main.load_model() is not first
