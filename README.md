# 🛡️ SentinelML

**Adaptive ML Failure Detection & Root-Cause Monitoring Engine**

SentinelML is an end-to-end machine learning monitoring prototype designed to detect when a deployed ML model begins behaving unreliably because of changing production data.

Instead of only measuring model accuracy during training, SentinelML monitors production behavior using **data drift detection, anomaly detection, uncertainty estimation, SHAP explanations, model-health scoring, and automated retraining recommendations**.

---

## 🚀 Project Motivation

Machine learning models can perform extremely well during training and still degrade after deployment.

Common causes include:

- Changes in user behavior
- Feature distribution shifts
- Previously unseen observations
- Increased prediction uncertainty
- Changes in relationships between features and targets
- Data quality issues

SentinelML provides a monitoring layer around a deployed ML model to detect these issues.

---

# 🧠 System Architecture

```text
                     SENTINELML
                         │
                         ▼
                  Reference Dataset
                         │
                         ▼
                  Baseline XGBoost
                         │
                         ▼
                   Production Data
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
      PSI + KS      Isolation Forest   Predictions
         │               │                │
         ▼               ▼                ▼
   Feature Drift      Anomalies       Probability
                                          │
                                          ▼
                                     Uncertainty
                                          │
                                          ▼
                                         SHAP
                                          │
                                          ▼
                                  Root-Cause Analysis
                                          │
                                          ▼
                                  Model Health Engine
                                          │
                                          ▼
                              Retraining Recommendation
                                          │
                                          ▼
                                 Streamlit Dashboard
```

---

# ✨ Features

SentinelML currently provides:

- XGBoost classification model
- Production prediction monitoring
- Population Stability Index (PSI)
- Kolmogorov-Smirnov statistical drift testing
- Isolation Forest anomaly detection
- Prediction uncertainty estimation
- SHAP-based model attribution analysis
- HIGH / MEDIUM / LOW prediction-risk classification
- Model-health scoring
- Rules-based retraining recommendation
- Interactive Streamlit monitoring dashboard

---

# 📊 Baseline Model Performance

The baseline XGBoost model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 98.00% |
| Precision | 98.77% |
| Recall | 94.12% |
| F1 Score | 96.39% |
| ROC-AUC | 99.10% |

These values represent held-out classification performance of the baseline model.

They should not be interpreted as drift-detection accuracy.

---

# 🔍 Production Monitoring Results

During the simulated production monitoring experiment:

| Monitoring Metric | Result |
|---|---:|
| Model Health Score | 74.30 / 100 |
| Health Status | WATCH |
| Drifted Features | 4 / 8 |
| Anomaly Rate | 10.72% |
| High-Uncertainty Predictions | 8.08% |
| HIGH-Risk Predictions | 0.40% |
| MEDIUM-Risk Predictions | 18.00% |
| Recommendation | REVIEW / RETRAIN |

---

# 📈 Data Drift Detection

SentinelML combines two complementary methods.

## Population Stability Index

PSI measures the magnitude of distribution change between reference and production data.

```text
PSI < 0.10       → Stable
PSI 0.10–0.25    → Moderate Drift
PSI >= 0.25      → High Drift
```

## Kolmogorov-Smirnov Test

The two-sample KS test determines whether reference and production observations appear to originate from the same underlying distribution.

Using both methods helps distinguish statistically detectable changes from operationally significant drift.

---

# ⚠️ Detected Feature Drift

Significant drift was detected in:

| Feature | PSI | Status |
|---|---:|---|
| monthly_charges | 0.4186 | HIGH |
| usage_hours | 0.2934 | HIGH |
| support_calls | 0.5135 | HIGH |
| engagement_score | 0.3483 | HIGH |

Stable features included:

- tenure
- late_payments
- contract_score
- service_count

---

# 🚨 Anomaly Detection

SentinelML uses **Isolation Forest** to detect production observations that differ significantly from the reference population.

Current experiment:

```text
Production Samples : 2500
Anomaly Rate       : 10.72%
```

Anomaly detection is evaluated independently from data drift.

This allows SentinelML to detect individual unusual observations even when the overall population appears stable.

---

# 🎯 Prediction Uncertainty

The classifier's probability output is converted into an uncertainty score.

Predictions close to:

```text
P(y = 1) = 0.5
```

are treated as more uncertain than predictions close to 0 or 1.

The current monitoring run detected:

```text
202 uncertain predictions
8.08% uncertainty rate
```

---

# 🚦 Prediction Risk

SentinelML combines anomaly detection and prediction uncertainty to classify individual production observations.

```text
HIGH
Anomalous + High Uncertainty

MEDIUM
Anomalous OR High Uncertainty

LOW
Neither condition detected
```

Current results:

```text
HIGH   : 10
MEDIUM : 450
LOW    : 2040
```

---

# 🔬 SHAP Root-Cause Analysis

SentinelML uses **SHAP** to identify which model features contribute most strongly to risky model predictions.

For HIGH-risk observations, the strongest model-attributed drivers were:

| Feature | Contribution |
|---|---:|
| support_calls | 30.11% |
| tenure | 28.75% |
| monthly_charges | 12.61% |
| usage_hours | 9.75% |
| contract_score | 8.04% |

An important observation is that `tenure` remained stable at the population level while still becoming an important driver for HIGH-risk individual predictions.

This demonstrates the distinction between:

```text
Population-level drift
```

and

```text
Observation-level model behavior
```

SHAP values are interpreted as model-attributed drivers rather than proof of real-world causality.

---

# 🩺 Model Health Engine

SentinelML combines monitoring signals into a transparent model-health score.

Current result:

```text
Health Score  : 74.30 / 100
Status        : WATCH
```

Signals include:

- Percentage of drifted features
- Production anomaly rate
- Prediction uncertainty rate
- HIGH-risk prediction rate

The current health score is a transparent heuristic rather than a learned metric.

---

# 🔄 Retraining Recommendation Engine

SentinelML independently evaluates hard monitoring thresholds.

Current recommendation:

```text
REVIEW / RETRAIN
```

Reasons:

```text
- Significant drift detected across a large proportion of input features.
- Production anomaly rate exceeded 10%.
```

This allows SentinelML to recommend review even when the aggregate model-health score has not reached a critical state.

---

# 🖥️ Streamlit Dashboard

The interactive monitoring dashboard provides:

- Model health score
- Current health status
- Drifted feature count
- Anomaly rate
- Prediction uncertainty
- Retraining recommendation
- PSI feature chart
- Drift statistics
- Prediction-risk distribution
- SHAP root-cause visualization
- HIGH-risk production observations

Start the dashboard with:

```bash
python -m streamlit run dashboard/app.py
```

---

# 📁 Project Structure

```text
sentinel-ml/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── reference.csv
│   └── production.csv
│
├── models/
│   ├── baseline_model.joblib
│   ├── baseline_metrics.json
│   ├── drift_report.json
│   ├── production_analysis.csv
│   ├── root_cause_report.json
│   └── model_health_report.json
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── train.py
│   ├── drift_detector.py
│   ├── anomaly_detector.py
│   ├── root_cause.py
│   ├── retraining_engine.py
│   └── run_pipeline.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Ayushdevo/SentinelML.git
cd SentinelML
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 🏗️ Generate Dataset

```bash
python src/generate_data.py
```

This generates:

```text
data/reference.csv
data/production.csv
```

Production data intentionally contains feature-distribution shifts to test SentinelML's monitoring capabilities.

---

# 🤖 Train Baseline Model

```bash
python src/train.py
```

The trained model and baseline metrics are stored inside:

```text
models/
```

---

# 🔎 Run Drift Detection

```bash
python src/drift_detector.py
```

The resulting monitoring report is written to:

```text
models/drift_report.json
```

---

# 🚨 Run Anomaly & Uncertainty Analysis

```bash
python src/anomaly_detector.py
```

Results are stored in:

```text
models/production_analysis.csv
```

---

# 🔬 Run Root-Cause Analysis

```bash
python src/root_cause.py
```

The SHAP report is stored in:

```text
models/root_cause_report.json
```

---

# 🩺 Calculate Model Health

```bash
python src/retraining_engine.py
```

The model-health report is stored in:

```text
models/model_health_report.json
```

---

# ⚡ Run Complete Monitoring Pipeline

Instead of executing each monitoring component manually:

```bash
python -m src.run_pipeline
```

Run from the repository root. The pipeline generates both sample datasets when
they are absent and trains the baseline model when its artifact is absent.
Existing datasets and model artifacts are reused. To retrain after changing
the reference data, run `python -m src.train` first.

Run regression checks with `python -m pip install -r requirements-dev.txt`
and `python -m pytest -q`.

Pipeline:

```text
Drift Detection
       ↓
Anomaly Detection
       ↓
Prediction Uncertainty
       ↓
SHAP Analysis
       ↓
Model Health
       ↓
Retraining Decision
```

---

# 🖥️ Launch Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

# 🛠️ Technology Stack

### Machine Learning

- Python
- XGBoost
- Scikit-learn
- Isolation Forest

### Model Monitoring

- Population Stability Index
- Kolmogorov-Smirnov Test
- Prediction uncertainty
- SHAP

### Data

- Pandas
- NumPy
- SciPy

### Visualization

- Streamlit

### Model Serialization

- Joblib

---

# 🎯 What SentinelML Demonstrates

This project demonstrates more than model training.

It covers the ML lifecycle:

```text
Data
 ↓
Model Training
 ↓
Evaluation
 ↓
Deployment Simulation
 ↓
Production Monitoring
 ↓
Drift Detection
 ↓
Anomaly Detection
 ↓
Prediction Uncertainty
 ↓
Explainability
 ↓
Model Health
 ↓
Retraining Decision
```

The objective is to demonstrate practical **ML engineering and model-observability concepts**, rather than optimizing a single classification metric.

---

# 🔮 Future Improvements

Planned extensions include:

- FastAPI inference service
- MLflow experiment tracking
- Docker deployment
- Automated monitoring schedules
- Historical drift tracking
- Concept-drift detection
- Performance degradation tracking when labels become available
- Automated model retraining
- Model version comparison
- Alerting integrations
- Prometheus/Grafana monitoring
- Cloud deployment
- CI/CD testing

---

# ⚠️ Experimental Scope

SentinelML V1 uses synthetic production drift to demonstrate ML-monitoring techniques.

The project intentionally separates:

```text
Covariate Drift
Model Uncertainty
Anomaly Detection
Model Attribution
Retraining Decisions
```

Future versions will evaluate model performance degradation against delayed production labels and introduce explicit concept-drift simulations.

---

# 👨‍💻 Author

**Ayush Tiwari**

Machine Learning • Data Science • Generative AI • ML Engineering

GitHub: **Ayushdevo**

Portfolio: **www.ayushtiwari.tech**

---

## ⭐ SentinelML

> Detect drift. Identify risky predictions. Understand why. Know when to retrain.
