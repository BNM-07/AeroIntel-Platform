# AeroIntel Platform

AI-powered Airline Analytics Dashboard for predicting flight delays and optimizing ticket pricing.

---

## Features

* Flight Delay Prediction (High Recall Model)
* Ticket Pricing Prediction (Regression Model)
* Threshold Tuning for Business Trade-offs
* Model Comparison Dashboard
* Explainable AI (Reason for Delay)

---

## Tech Stack

* Python, Streamlit
* LightGBM, XGBoost, Scikit-learn
* Pandas, NumPy
* Plotly, SHAP

---

## Project Structure

app.py → Main dashboard
pages/ → Multi-page UI
models/ → Trained models
data/ → Sample datasets
scripts/ → Training & validation

---

## Run Locally

pip install -r requirements.txt
streamlit run app.py

---

## Highlights

* Optimized recall (~93–95%) for delay prediction
* Designed business-aware threshold tuning
* Built production-style dashboard UI

---

## Use Case

Helps airlines:

* Predict delays early
* Optimize pricing strategies
* Reduce operational risk
