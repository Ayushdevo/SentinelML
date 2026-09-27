from src.generate_data import build_dataset, create_production_data
from src.drift_detector import detect_drift
from src.anomaly_detector import analyze_production
from src.root_cause import analyze_root_causes
from src.retraining_engine import calculate_health
from src.train import train

import pandas as pd
from pathlib import Path


def main():

    print("\n" + "=" * 60)
    print("SENTINELML — MONITORING PIPELINE")
    print("=" * 60)

    # Only regenerate data if files do not exist
    reference_path = Path("data/reference.csv")
    production_path = Path("data/production.csv")
    model_path = Path("models/baseline_model.joblib")

    if reference_path.exists() != production_path.exists():
        raise FileNotFoundError(
            "Only one dataset exists; restore its matching dataset or remove both to regenerate"
        )

    if not reference_path.exists():
        reference_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = build_dataset()

        reference = dataset.iloc[:7500].copy()

        production = create_production_data(
            dataset.iloc[7500:]
        )

        reference.to_csv(
            reference_path,
            index=False,
        )

        production.to_csv(
            production_path,
            index=False,
        )

        print("Datasets generated.")

    if not model_path.exists():
        print("\n[setup] Training baseline model...")
        train()

    print("\n[1/4] Detecting drift...")
    detect_drift()

    print("\n[2/4] Detecting anomalies...")
    analyze_production()

    print("\n[3/4] Running SHAP analysis...")
    analyze_root_causes()

    print("\n[4/4] Calculating model health...")
    calculate_health()

    print("\n" + "=" * 60)
    print("SENTINELML PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
