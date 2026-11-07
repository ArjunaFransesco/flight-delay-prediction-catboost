# Commercial Flight Departure Delay Predictor & Weather Impact Model

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **🛫 Nationwide Airport Flight Delay Classification with Severe Weather Encoding, Congestion Indexes & Streamlit Live Airport Tracker**

---

## 📌 Executive Summary & Mathematical Formulation

In Aviation / Transportation ML, high-precision regression estimates enable optimal resource allocation and proactive risk mitigation:

$$\min_{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}\left(y_i, f_{\theta}(X_i)\right) + \lambda \Omega(\theta)$$

- **Target Predictor**: `arrival_delay_minutes`
- **Optimization Metric**: Mean Absolute Error (MAE) & $R^2$ Variance Explained
- **Model Architecture**: `RandomForestRegressor (Ensemble 180 Trees)`

---

## 🏗️ Architecture & Pipeline Flow

```
┌────────────────────────────────────────────────────────┐
│     Domain Telemetry & Ingestion Pipeline              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Feature Engineering, Imputation & Scaling Pipeline   │
│  - StandardScaler Normalization & Multicollinearity    │
│  - Correlation Matrix & Domain Specific Attributes     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Ensemble Machine Learning & 5-Fold Cross-Validation  │
│  - Serialized Artifacts (.joblib) & Metric Evaluation  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│   Interactive Streamlit Dashboard & Web UI Interface   │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark & Performance Metrics

| Metric | Score | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **$R^2$ Coefficient** | **0.9302** | > 0.7500 | 🌟 High Accuracy |
| **Mean Absolute Error (MAE)** | **6.3009** | Minimized | ✅ Optimal |
| **Root Mean Squared Error (RMSE)** | **8.3926** | Minimized | ✅ Optimal |
| **5-Fold Cross Validation $R^2$** | **0.9358** | Robust Generalization | Verified |
| **Dataset Volume** | **3200 Samples** | Stratified Train/Test | Verified |

---

## 📁 Repository Structure

```
flight-delay-prediction-catboost/
├── app.py                     # Streamlit interactive diagnostic dashboard
├── data/
│   ├── raw/
│   │   └── flight_delay_prediction_catboost_dataset.csv   # Raw domain dataset
│   └── processed/
│       └── flight_delay_prediction_catboost_processed.csv # Scaled and preprocessed matrix
├── models/
│   ├── feature_names.joblib   # Ingested feature schema
│   ├── scaler.joblib          # Trained StandardScaler pipeline
│   └── model_pipeline.joblib  # Serialized machine learning estimator
├── notebooks/
│   └── flight_delay_prediction_catboost_pipeline.ipynb    # End-to-end Jupyter Notebook pipeline
├── reports/
│   ├── metrics.json           # Serialized evaluation metrics
│   └── evaluation_plot.png    # High-resolution benchmark visualizer
├── requirements.txt           # Environment dependencies
├── LICENSE                    # MIT Open Source License
└── README.md                  # Project documentation & benchmark overview
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ArjunaFransesco/flight-delay-prediction-catboost.git
cd flight-delay-prediction-catboost
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

### 3. Open End-to-End Jupyter Notebook
```bash
jupyter notebook notebooks/flight_delay_prediction_catboost_pipeline.ipynb
```

---

## 👤 Author & Portfolio
- **Author**: **[Arjuna Fransesco](https://github.com/ArjunaFransesco)**
- **GitHub Repositories**: [https://github.com/ArjunaFransesco?tab=repositories](https://github.com/ArjunaFransesco?tab=repositories)
- **Portfolio Website**: [https://github.com/ArjunaFransesco/arjuna-portfolio](https://github.com/ArjunaFransesco/arjuna-portfolio)


<!-- Last Maintenance Audit: 2026-11-07 -->
