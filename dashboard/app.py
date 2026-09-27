import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SentinelML",
    page_icon="🛡️",
    layout="wide",
)

HEALTH_PATH = Path("models/model_health_report.json")
DRIFT_PATH = Path("models/drift_report.json")
ROOT_CAUSE_PATH = Path("models/root_cause_report.json")
PRODUCTION_PATH = Path("models/production_analysis.csv")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_data():
    required = (HEALTH_PATH, DRIFT_PATH, ROOT_CAUSE_PATH, PRODUCTION_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Generate monitoring reports with `python -m src.run_pipeline` first. "
            f"Missing: {', '.join(missing)}"
        )
    health = load_json(HEALTH_PATH)
    drift = load_json(DRIFT_PATH)
    root_cause = load_json(ROOT_CAUSE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    return health, drift, root_cause, production


try:
    health, drift, root_cause, production = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()


st.title("🛡️ SentinelML")
st.caption(
    "Adaptive ML Failure Detection & Root-Cause Monitoring Engine"
)

st.divider()


# ------------------------------------------------
# Model Health
# ------------------------------------------------

score = health["health_score"]
status = health["health_status"]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Model Health",
    f"{score:.1f}/100",
)

col2.metric(
    "Status",
    status,
)

col3.metric(
    "Drifted Features",
    f"{health['drift']['number_of_drifted_features']}/"
    f"{health['drift']['total_features']}",
)

col4.metric(
    "Anomaly Rate",
    f"{health['metrics']['anomaly_rate']:.2%}",
)

col5.metric(
    "Uncertainty Rate",
    f"{health['metrics']['uncertainty_rate']:.2%}",
)


st.divider()


# ------------------------------------------------
# Retraining recommendation
# ------------------------------------------------

st.subheader("Retraining Decision")

if health["retraining"]["recommended"]:
    st.error("⚠️ REVIEW / RETRAIN RECOMMENDED")

    for reason in health["retraining"]["reasons"]:
        st.write(f"• {reason}")
else:
    st.success("✓ Retraining currently not required")


st.divider()


# ------------------------------------------------
# Drift
# ------------------------------------------------

st.subheader("Feature Drift Analysis")

drift_rows = []

for feature, values in drift["features"].items():

    drift_rows.append(
        {
            "Feature": feature,
            "PSI": values["psi"],
            "KS Statistic": values["ks_statistic"],
            "p-value": values["p_value"],
            "Status": values["status"],
        }
    )

drift_df = pd.DataFrame(drift_rows)

st.dataframe(
    drift_df,
    use_container_width=True,
    hide_index=True,
)


# PSI chart

psi_chart = (
    drift_df[
        ["Feature", "PSI"]
    ]
    .set_index("Feature")
)

st.subheader("PSI by Feature")

st.bar_chart(psi_chart)


st.divider()


# ------------------------------------------------
# Prediction Risk
# ------------------------------------------------

st.subheader("Production Prediction Risk")

risk_counts = (
    production["prediction_risk"]
    .value_counts()
    .rename_axis("Risk")
    .reset_index(name="Count")
)

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric(
    "HIGH Risk",
    int(
        (
            production["prediction_risk"]
            == "HIGH"
        ).sum()
    ),
)

risk_col2.metric(
    "MEDIUM Risk",
    int(
        (
            production["prediction_risk"]
            == "MEDIUM"
        ).sum()
    ),
)

risk_col3.metric(
    "LOW Risk",
    int(
        (
            production["prediction_risk"]
            == "LOW"
        ).sum()
    ),
)

st.bar_chart(
    risk_counts.set_index("Risk")
)


st.divider()


# ------------------------------------------------
# Root cause analysis
# ------------------------------------------------

st.subheader("Root-Cause Analysis")

root_causes = health["top_high_risk_drivers"]

root_df = pd.DataFrame(
    {
        "Feature": list(root_causes.keys()),
        "Contribution (%)": list(root_causes.values()),
    }
)

root_df = root_df.sort_values(
    "Contribution (%)",
    ascending=False,
)

st.bar_chart(
    root_df.set_index("Feature")
)

st.dataframe(
    root_df,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ------------------------------------------------
# Risky observations
# ------------------------------------------------

st.subheader("HIGH-Risk Production Observations")

high_risk = production[
    production["prediction_risk"] == "HIGH"
].copy()

display_columns = [
    "prediction",
    "churn_probability",
    "uncertainty",
    "anomaly_score",
    "support_calls",
    "tenure",
    "monthly_charges",
    "usage_hours",
    "engagement_score",
]

available_columns = [
    column
    for column in display_columns
    if column in high_risk.columns
]

st.dataframe(
    high_risk[available_columns],
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ------------------------------------------------
# Model explanation
# ------------------------------------------------

with st.expander("How SentinelML works"):

    st.markdown(
        """
        **SentinelML monitors deployed ML models using four layers:**

        1. **Distribution Drift**
           - Population Stability Index (PSI)
           - Kolmogorov-Smirnov testing

        2. **Anomaly Detection**
           - Isolation Forest detects unusual production observations.

        3. **Prediction Uncertainty**
           - Predictions close to the classifier decision boundary
             receive higher uncertainty scores.

        4. **Root-Cause Analysis**
           - SHAP attributes model predictions to input features.

        These signals are combined into a transparent model-health
        score and a rules-based retraining recommendation.
        """
    )
