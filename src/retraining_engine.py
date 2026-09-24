import json
from pathlib import Path

import pandas as pd


DRIFT_PATH = Path("models/drift_report.json")
ROOT_CAUSE_PATH = Path("models/root_cause_report.json")
PRODUCTION_PATH = Path("models/production_analysis.csv")
OUTPUT_PATH = Path("models/model_health_report.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_health():

    drift_report = load_json(DRIFT_PATH)
    root_cause_report = load_json(ROOT_CAUSE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    total = len(production)

    anomaly_rate = (
        production["is_anomaly"].mean()
    )

    uncertainty_rate = (
        production["high_uncertainty"].mean()
    )

    high_risk_rate = (
        production["prediction_risk"]
        .eq("HIGH")
        .mean()
    )

    medium_risk_rate = (
        production["prediction_risk"]
        .eq("MEDIUM")
        .mean()
    )

    drifted_features = (
        drift_report["number_of_drifted_features"]
    )

    total_features = len(
        drift_report["features"]
    )

    drift_ratio = (
        drifted_features / total_features
        if total_features
        else 0
    )

    # ---------------------------------
    # Health penalty calculation
    # ---------------------------------

    drift_penalty = min(
        drift_ratio * 40,
        40
    )

    anomaly_penalty = min(
        anomaly_rate * 100 * 0.25,
        20
    )

    uncertainty_penalty = min(
        uncertainty_rate * 100 * 0.30,
        25
    )

    high_risk_penalty = min(
        high_risk_rate * 100 * 1.50,
        15
    )

    total_penalty = (
        drift_penalty
        + anomaly_penalty
        + uncertainty_penalty
        + high_risk_penalty
    )

    health_score = max(
        0,
        100 - total_penalty
    )

    # ---------------------------------
    # Health classification
    # ---------------------------------

    if health_score >= 85:
        health_status = "HEALTHY"

    elif health_score >= 70:
        health_status = "WATCH"

    elif health_score >= 50:
        health_status = "DEGRADED"

    else:
        health_status = "CRITICAL"

    # ---------------------------------
    # Retraining rules
    # ---------------------------------

    reasons = []

    if drift_ratio >= 0.40:
        reasons.append(
            "Significant drift detected across "
            "a large proportion of input features."
        )

    if anomaly_rate >= 0.10:
        reasons.append(
            "Production anomaly rate exceeded 10%."
        )

    if uncertainty_rate >= 0.10:
        reasons.append(
            "High prediction uncertainty exceeded 10%."
        )

    if high_risk_rate >= 0.02:
        reasons.append(
            "High-risk prediction rate exceeded 2%."
        )

    if health_score < 70:
        reasons.append(
            "Overall model health score fell below 70."
        )

    retraining_recommended = (
        len(reasons) > 0
    )

    # ---------------------------------
    # Root-cause summary
    # ---------------------------------

    root_causes = (
        root_cause_report
        .get("high_risk_root_causes", {})
    )

    top_root_causes = dict(
        list(root_causes.items())[:5]
    )

    # ---------------------------------
    # Final report
    # ---------------------------------

    report = {
        "health_score": round(
            health_score,
            2,
        ),
        "health_status": health_status,

        "metrics": {
            "drift_ratio": round(
                drift_ratio,
                4,
            ),
            "anomaly_rate": round(
                anomaly_rate,
                4,
            ),
            "uncertainty_rate": round(
                uncertainty_rate,
                4,
            ),
            "high_risk_rate": round(
                high_risk_rate,
                4,
            ),
            "medium_risk_rate": round(
                medium_risk_rate,
                4,
            ),
        },

        "drift": {
            "drifted_features":
                drift_report[
                    "drifted_features"
                ],

            "number_of_drifted_features":
                drifted_features,

            "total_features":
                total_features,
        },

        "top_high_risk_drivers":
            top_root_causes,

        "retraining": {
            "recommended":
                retraining_recommended,

            "reasons":
                reasons,
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

    # ---------------------------------
    # Console output
    # ---------------------------------

    print("\nSENTINELML — MODEL HEALTH")
    print("=" * 60)

    print(
        f"Health Score        : "
        f"{health_score:.2f}/100"
    )

    print(
        f"Health Status       : "
        f"{health_status}"
    )

    print(
        f"Drifted Features    : "
        f"{drifted_features}/{total_features}"
    )

    print(
        f"Anomaly Rate        : "
        f"{anomaly_rate:.2%}"
    )

    print(
        f"Uncertainty Rate    : "
        f"{uncertainty_rate:.2%}"
    )

    print(
        f"HIGH Risk Rate      : "
        f"{high_risk_rate:.2%}"
    )

    print(
        f"MEDIUM Risk Rate    : "
        f"{medium_risk_rate:.2%}"
    )

    print("-" * 60)

    if retraining_recommended:

        print(
            "RETRAINING RECOMMENDATION: "
            "REVIEW / RETRAIN"
        )

        print("\nReasons:")

        for reason in reasons:
            print(f"  - {reason}")

    else:

        print(
            "RETRAINING RECOMMENDATION: "
            "NOT REQUIRED"
        )

    print("\nTop HIGH-risk drivers:")

    for feature, contribution in (
        top_root_causes.items()
    ):

        print(
            f"  {feature:22} "
            f"{contribution:6.2f}%"
        )

    print("=" * 60)

    print(
        "\nSaved →",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    calculate_health()