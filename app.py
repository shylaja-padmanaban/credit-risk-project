import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict, risk_band

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦")

st.title("🏦 Credit Risk Default Predictor")
st.markdown("""
Predicts whether a customer is likely to default on a loan based on their
financial profile. Built using Logistic Regression with SHAP explainability.
**Test ROC-AUC: 0.685 | Test Recall: 0.62**
""")

st.sidebar.header("Customer Financial Profile")

income = st.sidebar.number_input(
    "Annual Income (₹)", min_value=0, max_value=10000000,
    value=85000, step=5000
)
savings = st.sidebar.number_input(
    "Total Savings (₹)", min_value=0, max_value=10000000,
    value=273000, step=10000
)
debt = st.sidebar.number_input(
    "Total Debt (₹)", min_value=0, max_value=10000000,
    value=395000, step=10000
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Derived ratios (auto-computed):**")
debt_to_income = round(debt / income, 3) if income > 0 else 0
savings_to_income = round(savings / income, 3) if income > 0 else 0
st.sidebar.write(f"Debt-to-Income ratio: **{debt_to_income}**")
st.sidebar.write(f"Savings-to-Income ratio: **{savings_to_income}**")

if st.button("Predict Default Risk", type="primary"):
    row = {"INCOME": income, "SAVINGS": savings, "DEBT": debt}
    result = predict(row)

    prob = result["default_probability"]
    band = result["risk_band"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Default Probability", f"{prob:.1%}")
    with col2:
        st.metric("Risk Decision", band)

    if "Approve" in band:
        st.success(f"✅ {band} — Low default risk.")
        st.progress(prob)
    elif "Manual" in band:
        st.warning(f"⚠️ {band} — Moderate risk. Recommend further review.")
        st.progress(prob)
    else:
        st.error(f"🚫 {band} — High default risk.")
        st.progress(prob)

    st.markdown("---")
    st.markdown("### What drove this prediction?")
    st.markdown(f"""
- **Debt-to-Income ratio = {debt_to_income}** — strongest SHAP driver.
  Higher values push default risk up.
- **Savings-to-Income ratio = {savings_to_income}** — higher savings
  relative to income reduces risk.
- Remaining ~70 spend-category features use median values from training data.
    """)

st.markdown("---")
st.markdown("""
**Model:** Logistic Regression, `class_weight='balanced'`, 752 training rows.
Top SHAP features: `R_DEBT_INCOME`, `R_SAVINGS_INCOME`, `R_TAX_DEBT`.
[View on GitHub](https://github.com/shylaja-padmanaban/credit-risk-project)
""")