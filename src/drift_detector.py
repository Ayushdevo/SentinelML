import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


REFERENCE_PATH = Path("data/reference.csv")
PRODUCTION_PATH = Path("data/production.csv")
OUTPUT_PATH = Path("models/drift_report.json")

TARGET = "churn"


def calculate_psi(expected, actual, bins=10):
    """
    Population Stability Index (PSI).

    PSI < 0.10  -> Stable
    PSI 0.10-0.25 -> Moderate drift
    PSI >= 0.25 -> Significant drift
    """

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if not len(expected) or not len(actual):
        raise ValueError("PSI requires nonempty reference and production samples")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("PSI requires finite numeric values")
    if bins < 2:
        raise ValueError("PSI requires at least two bins")
    if np.min(expected) == np.max(expected) == np.min(actual) == np.max(actual):
        return 0.0

    # Quantile-based bins from reference distribution
    breakpoints = np.unique(
        np.quantile(
            expected,
            np.linspace(0, 1, bins + 1),
        )
    )

    # Handle features with too few unique values
    if len(breakpoints) < 3:
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1,
        )
        if breakpoints[0] == breakpoints[-1]:
            breakpoints = np.array([breakpoints[0] - 0.5, breakpoints[0] + 0.5])

    # Make sure all production values are captured
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints,
    )

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Prevent division by zero
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def classify_drift(psi, p_value):
    """
    Combine PSI magnitude with KS statistical significance.
    """

    if psi >= 0.25:
        return "HIGH"

    if psi >= 0.10:
        return "MODERATE"

    if p_value < 0.05:
        return "LOW"

    return "STABLE"


def detect_drift():
    reference = pd.read_csv(REFERENCE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)
    if TARGET not in reference.columns:
        raise ValueError(f"Reference data is missing target column: {TARGET}")

    features = [
        column
        for column in reference.columns
        if column != TARGET
    ]
    if not features:
        raise ValueError("Reference data has no feature columns")
    missing = sorted(set(features) - set(production.columns))
    if missing:
        raise ValueError(f"Production data is missing features: {', '.join(missing)}")

    report = {}

    print("\nSENTINELML — DRIFT ANALYSIS")
    print("=" * 72)

    print(
        f"{'Feature':22}"
        f"{'PSI':>10}"
        f"{'KS':>10}"
        f"{'p-value':>14}"
        f"{'Status':>14}"
    )

    print("-" * 72)

    for feature in features:

        ref_values = reference[feature].dropna()
        prod_values = production[feature].dropna()
        if ref_values.empty or prod_values.empty:
            raise ValueError(f"Feature {feature!r} has no usable samples")

        psi = calculate_psi(
            ref_values,
            prod_values,
        )

        ks_statistic, p_value = ks_2samp(
            ref_values,
            prod_values,
        )

        status = classify_drift(
            psi,
            p_value,
        )

        report[feature] = {
            "reference_count": int(len(ref_values)),
            "production_count": int(len(prod_values)),
            "reference_missing_rate": round(float(reference[feature].isna().mean()), 6),
            "production_missing_rate": round(float(production[feature].isna().mean()), 6),
            "psi": round(float(psi), 6),
            "ks_statistic": round(
                float(ks_statistic), 6
            ),
            "p_value": float(p_value),
            "status": status,
        }

        print(
            f"{feature:22}"
            f"{psi:>10.4f}"
            f"{ks_statistic:>10.4f}"
            f"{p_value:>14.6f}"
            f"{status:>14}"
        )

    drifted_features = [
        feature
        for feature, values in report.items()
        if values["status"] in {"MODERATE", "HIGH"}
    ]

    overall_drift = len(drifted_features) > 0

    final_report = {
        "overall_drift_detected": overall_drift,
        "drifted_features": drifted_features,
        "number_of_drifted_features": len(
            drifted_features
        ),
        "features": report,
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_report,
            file,
            indent=4,
        )

    print("=" * 72)

    if overall_drift:
        print("\n⚠ DATA DRIFT DETECTED")
        print(
            "Drifted features:",
            ", ".join(drifted_features),
        )
    else:
        print("\n✓ No significant data drift detected.")

    print(
        "\nReport saved →",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    detect_drift()
