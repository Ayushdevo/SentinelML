from pathlib import Path

from api import main
from dashboard import app as dashboard
from src import anomaly_detector, drift_detector, generate_data, retraining_engine, root_cause, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_artifact_paths_are_anchored_to_the_project():
    paths = [
        main.MODEL_PATH,
        main.HEALTH_PATH,
        dashboard.HEALTH_PATH,
        dashboard.DRIFT_PATH,
        dashboard.ROOT_CAUSE_PATH,
        dashboard.PRODUCTION_PATH,
        anomaly_detector.REFERENCE_PATH,
        anomaly_detector.PRODUCTION_PATH,
        anomaly_detector.MODEL_PATH,
        anomaly_detector.OUTPUT_PATH,
        drift_detector.REFERENCE_PATH,
        drift_detector.PRODUCTION_PATH,
        drift_detector.OUTPUT_PATH,
        generate_data.DATA_DIR,
        retraining_engine.DRIFT_PATH,
        retraining_engine.ROOT_CAUSE_PATH,
        retraining_engine.PRODUCTION_PATH,
        retraining_engine.OUTPUT_PATH,
        root_cause.MODEL_PATH,
        root_cause.PRODUCTION_PATH,
        root_cause.OUTPUT_PATH,
        train.DATA_PATH,
        train.MODEL_DIR,
    ]

    assert all(path.is_absolute() for path in paths)
    assert all(path.is_relative_to(PROJECT_ROOT) for path in paths)
