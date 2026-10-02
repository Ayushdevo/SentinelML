from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


REFERENCE_PATH = Path("data/reference.csv")
PRODUCTION_PATH = Path("data/production.csv")
MODEL_PATH = Path("models/baseline_model.joblib")
OUTPUT_PATH = Path("models/production_analysis.csv")

TARGET = "churn"


def analyze_production():
    reference = pd.read_csv(REFERENCE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)
    model = joblib.load(MODEL_PATH)

    features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [
        column for column in reference.columns if column != TARGET
    ]
    for name, frame in (("Reference", reference), ("Production", production)):
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data is missing model features: {', '.join(missing)}")
        if frame.empty:
            raise ValueError(f"{name} data has no rows")

    X_reference = reference[features]
    X_production = production[features]
    for name, frame in (("Reference", X_reference), ("Production", X_production)):
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError(f"{name} features must be numeric")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} features contain missing or infinite values")

    # ----------------------------
    # 1. Scale features
    # ----------------------------
    scaler = StandardScaler()

    X_ref_scaled = scaler.fit_transform(X_reference)
    X_prod_scaled = scaler.transform(X_production)

    # ----------------------------
    # 2. Isolation Forest
    # ----------------------------
    detector = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )

    detector.fit(X_ref_scaled)

    anomaly_prediction = detector.predict(X_prod_scaled)

    anomaly_score = detector.decision_function(
        X_prod_scaled
    )

    # IsolationForest:
    # -1 = anomaly
    #  1 = normal

    production["is_anomaly"] = (
        anomaly_prediction == -1
    ).astype(int)

    production["anomaly_score"] = anomaly_score

    # ----------------------------
    # 3. Model predictions
    # ----------------------------
    classes = list(getattr(model, "classes_", (0, 1)))
    if len(classes) != 2 or 1 not in classes:
        raise ValueError("Churn model must have a positive class labeled 1")
    class_probabilities = np.asarray(model.predict_proba(X_production), dtype=float)
    if (
        class_probabilities.shape != (len(production), len(classes))
        or not np.isfinite(class_probabilities).all()
        or np.any((class_probabilities < 0) | (class_probabilities > 1))
        or not np.allclose(class_probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    ):
        raise ValueError("Model returned invalid class probabilities")
    probabilities = class_probabilities[:, classes.index(1)]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    production["prediction"] = predictions
    production["churn_probability"] = probabilities

    # ----------------------------
    # 4. Prediction uncertainty
    # ----------------------------
    # Highest uncertainty occurs near probability = 0.5.
    #
    # uncertainty:
    # 0 = very confident
    # 1 = maximally uncertain

    uncertainty = 1 - (
        np.abs(probabilities - 0.5) * 2
    )

    production["uncertainty"] = uncertainty

    production["high_uncertainty"] = (
        uncertainty >= 0.60
    ).astype(int)

    # ----------------------------
    # 5. Combined risk
    # ----------------------------

    production["prediction_risk"] = np.select(
        [
            (
                (production["is_anomaly"] == 1)
                & (production["high_uncertainty"] == 1)
            ),
            (
                (production["is_anomaly"] == 1)
                | (production["high_uncertainty"] == 1)
            ),
        ],
        [
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    # ----------------------------
    # Save results
    # ----------------------------

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    production.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ----------------------------
    # Summary
    # ----------------------------

    total = len(production)

    anomalies = int(
        production["is_anomaly"].sum()
    )

    uncertain = int(
        production["high_uncertainty"].sum()
    )

    risk_counts = (
        production["prediction_risk"]
        .value_counts()
        .to_dict()
    )

    print("\nSENTINELML — PRODUCTION RISK ANALYSIS")
    print("=" * 55)

    print(f"Production observations : {total}")
    print(
        f"Anomalies detected      : "
        f"{anomalies} ({anomalies / total:.2%})"
    )

    print(
        f"Uncertain predictions   : "
        f"{uncertain} ({uncertain / total:.2%})"
    )

    print("\nPrediction Risk")

    print(
        f"HIGH   : {risk_counts.get('HIGH', 0)}"
    )

    print(
        f"MEDIUM : {risk_counts.get('MEDIUM', 0)}"
    )

    print(
        f"LOW    : {risk_counts.get('LOW', 0)}"
    )

    print("=" * 55)

    print(
        "\nSaved →",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    analyze_production()
