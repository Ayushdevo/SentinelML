from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("models/baseline_model.joblib")
PRODUCTION_PATH = Path("models/production_analysis.csv")
OUTPUT_PATH = Path("models/root_cause_report.json")

TARGET = "churn"

NON_FEATURE_COLUMNS = {
    TARGET,
    "prediction",
    "churn_probability",
    "uncertainty",
    "high_uncertainty",
    "is_anomaly",
    "anomaly_score",
    "prediction_risk",
}


def analyze_root_causes():

    print("\nSENTINELML — ROOT CAUSE ANALYSIS")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    features = [
        column
        for column in production.columns
        if column not in NON_FEATURE_COLUMNS
    ]
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    missing = sorted(set(features) - set(production.columns))
    if missing or production.empty:
        raise ValueError(f"Production analysis is empty or missing features: {missing}")

    X = production[features]

    # --------------------------------
    # 1. SHAP Tree Explainer
    # --------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Compatibility with different SHAP versions
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # Multiclass explainers return (rows, features, classes).
        shap_values = shap_values[:, :, 1]
    if shap_values.shape != (len(X), len(features)):
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; expected {X.shape}"
        )
    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP attributions must contain only finite values")

    # --------------------------------
    # 2. Global feature importance
    # --------------------------------

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    total_importance = (
        importance["mean_abs_shap"].sum()
    )

    if total_importance > 0:
        importance["importance_percent"] = (
            importance["mean_abs_shap"]
            / total_importance
            * 100
        )
    else:
        importance["importance_percent"] = 0

    # --------------------------------
    # 3. Analyze HIGH-risk predictions
    # --------------------------------

    high_risk_mask = (
        production["prediction_risk"] == "HIGH"
    ).to_numpy()

    high_risk_count = int(
        high_risk_mask.sum()
    )

    high_risk_causes = {}

    if high_risk_count > 0:

        high_risk_shap = np.abs(
            shap_values[high_risk_mask]
        )

        high_risk_importance = (
            high_risk_shap.mean(axis=0)
        )

        total_high = (
            high_risk_importance.sum()
        )

        for feature, value in zip(
            features,
            high_risk_importance,
        ):

            percent = (
                float(value / total_high * 100)
                if total_high > 0
                else 0.0
            )

            high_risk_causes[feature] = percent

        high_risk_causes = dict(
            sorted(
                high_risk_causes.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # --------------------------------
    # 4. Console report
    # --------------------------------

    print(
        f"{'Feature':22}"
        f"{'SHAP Importance':>18}"
        f"{'Contribution':>16}"
    )

    print("-" * 60)

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:22}"
            f"{row['mean_abs_shap']:>18.4f}"
            f"{row['importance_percent']:>15.2f}%"
        )

    print("\n" + "=" * 60)

    print(
        f"HIGH-risk observations analysed: "
        f"{high_risk_count}"
    )

    if high_risk_count:

        print("\nTop causes for HIGH-risk predictions:")

        for feature, percent in list(
            high_risk_causes.items()
        )[:5]:

            print(
                f"  {feature:22} "
                f"{percent:6.2f}%"
            )

    # --------------------------------
    # 5. Save report
    # --------------------------------

    report = {
        "high_risk_observations": high_risk_count,
        "global_feature_importance": {
            row["feature"]: round(
                float(row["importance_percent"]),
                4,
            )
            for _, row in importance.iterrows()
        },
        "high_risk_root_causes": {
            feature: round(percent, 4)
            for feature, percent
            in high_risk_causes.items()
        },
    }

    OUTPUT_PATH.parent.mkdir(
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "\nReport saved →",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    analyze_root_causes()
