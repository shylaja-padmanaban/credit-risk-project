import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict, risk_band

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2d6a9f;
        margin: 0.5rem 0;
    }
    .approve { border-left-color: #28a745; background: #f0fff4; }
    .review  { border-left-color: #ffc107; background: #fffdf0; }
    .decline { border-left-color: #dc3545; background: #fff5f5; }
    .footer  { text-align: center; color: #888; font-size: 0.85rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 Credit Risk Default Predictor</h1>
    <p style="margin:0; opacity:0.9; font-size:1.1rem;">
        Predicts loan default probability using Logistic Regression + SHAP explainability
    </p>
    <p style="margin:0.5rem 0 0 0; opacity:0.7; font-size:0.9rem;">
        Built by Shylaja P &nbsp;|&nbsp; ROC-AUC: 0.685 &nbsp;|&nbsp; Recall: 0.62
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("## 📋 Customer Profile")
st.sidebar.markdown("Enter the customer's financial details:")

income = st.sidebar.number_input(
    "💰 Annual Income (₹)",
    min_value=0, max_value=10000000,
    value=85000, step=5000,
    help="Customer's total annual income"
)
savings = st.sidebar.number_input(
    "🏦 Total Savings (₹)",
    min_value=0, max_value=10000000,
    value=273000, step=10000,
    help="Customer's total savings balance"
)
debt = st.sidebar.number_input(
    "💳 Total Debt (₹)",
    min_value=0, max_value=10000000,
    value=395000, step=10000,
    help="Customer's total outstanding debt"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Derived Ratios")
debt_to_income = round(debt / income, 3) if income > 0 else 0
savings_to_income = round(savings / income, 3) if income > 0 else 0
debt_to_savings = round(debt / savings, 3) if savings > 0 else 0

col_a, col_b = st.sidebar.columns(2)
col_a.metric("Debt/Income", debt_to_income,
             delta="High risk" if debt_to_income > 5 else "Normal",
             delta_color="inverse")
col_b.metric("Savings/Income", savings_to_income,
             delta="Good" if savings_to_income > 2 else "Low")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This model was trained on **752 customers** after outlier removal.
Missing spend-category features are imputed using training data medians.

[📂 GitHub Repo](https://github.com/shylaja-padmanaban/credit-risk-project)
""")

# Main panel
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(
        "🔍 Predict Default Risk",
        type="primary",
        use_container_width=True
    )

if predict_btn:
    row = {"INCOME": income, "SAVINGS": savings, "DEBT": debt}
    result = predict(row)
    prob = result["default_probability"]
    band = result["risk_band"]

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob:.1%}")
    col2.metric("Risk Decision", band.split(" - ")[0])
    col3.metric("Debt-to-Income", debt_to_income,
                delta="Above threshold" if debt_to_income > 5 else "Within range",
                delta_color="inverse")

    st.markdown("")

    if "Approve" in band:
        st.markdown(f"""
        <div class="metric-card approve">
            <h3>✅ Decision: Approve</h3>
            <p>This customer shows a <strong>low default risk</strong> of {prob:.1%}.
            Financial profile appears stable and within acceptable lending parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    elif "Manual" in band:
        st.markdown(f"""
        <div class="metric-card review">
            <h3>⚠️ Decision: Manual Review</h3>
            <p>This customer shows a <strong>moderate default risk</strong> of {prob:.1%}.
            Recommend further review by a credit officer before approving.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card decline">
            <h3>🚫 Decision: Decline</h3>
            <p>This customer shows a <strong>high default risk</strong> of {prob:.1%}.
            Financial profile indicates significant risk of non-repayment.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.progress(float(prob))

    st.markdown("---")
    st.markdown("## 🔍 What drove this prediction?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Risk Factors (Global)")
        st.markdown(f"""
        | Feature | Value | Impact |
        |---------|-------|--------|
        | Debt-to-Income | {debt_to_income} | {'🔴 High risk' if debt_to_income > 5 else '🟢 Normal'} |
        | Savings-to-Income | {savings_to_income} | {'🟢 Protective' if savings_to_income > 2 else '🔴 Low savings'} |
        | Debt-to-Savings | {debt_to_savings} | {'🔴 High burden' if debt_to_savings > 3 else '🟢 Manageable'} |
        """)
    with col2:
        st.markdown("### Model Insights")
        st.markdown("""
        Based on SHAP analysis across all customers:
        - **R_DEBT_INCOME** is the #1 default driver
        - **R_SAVINGS_INCOME** is the top protective factor
        - **R_TAX_DEBT** is the #3 driver
        - Higher debt-to-income = higher default risk
        """)

    st.markdown("---")
    st.markdown("### 📈 Risk Band Reference")
    band_col1, band_col2, band_col3 = st.columns(3)
    band_col1.success("✅ **Approve** (< 30%)\nActual default rate: 8.2%")
    band_col2.warning("⚠️ **Manual Review** (30–55%)\nActual default rate: 29.9%")
    band_col3.error("🚫 **Decline** (> 55%)\nActual default rate: 39.2%")

# Footer
st.markdown("""
<div class="footer">
    Built by <strong>Shylaja P</strong> &nbsp;|&nbsp;
    Logistic Regression + SHAP &nbsp;|&nbsp;
    <a href="https://github.com/shylaja-padmanaban/credit-risk-project">GitHub</a>
    &nbsp;|&nbsp; Test ROC-AUC: 0.685
</div>
""", unsafe_allow_html=True)