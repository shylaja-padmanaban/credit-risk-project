import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Credit Risk Analyser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f5f7fa; }

    .hero-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563a8 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(30,58,95,0.3);
    }
    .hero-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .hero-header p  { font-size: 1.1rem; opacity: 0.85; margin: 0.5rem 0 0 0; }

    .step-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }
    .step-title {
        color: #1e3a5f;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .step-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    .progress-bar-container {
        background: white;
        border-radius: 50px;
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .score-card {
        background: linear-gradient(135deg, #1e3a5f, #2563a8);
        color: white;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(30,58,95,0.3);
    }
    .score-number {
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        margin: 1rem 0;
    }
    .score-label { font-size: 1.2rem; opacity: 0.85; }

    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
        border-left: 5px solid #2563a8;
    }
    .result-card.green  { border-left-color: #10b981; }
    .result-card.amber  { border-left-color: #f59e0b; }
    .result-card.red    { border-left-color: #ef4444; }

    .tip-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .tip-card h4 { color: #0369a1; margin: 0 0 0.3rem 0; font-size: 0.95rem; }
    .tip-card p  { color: #374151; margin: 0; font-size: 0.88rem; }

    .band-pill {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .band-poor      { background: #fee2e2; color: #dc2626; }
    .band-fair      { background: #ffedd5; color: #ea580c; }
    .band-good      { background: #fef9c3; color: #ca8a04; }
    .band-verygood  { background: #dcfce7; color: #16a34a; }
    .band-excellent { background: #d1fae5; color: #059669; }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1e3a5f, #2563a8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] button:hover { opacity: 0.9; }

    .stSelectbox label, .stNumberInput label,
    .stRadio label, .stSlider label { color: #374151; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# Session state init
for key in ['page','name','age','employment','experience',
            'income','savings','monthly_savings',
            'debt','monthly_emi',
            'food','shopping','entertainment','travel',
            'bills','rent','education','medical',
            'active_loans','credit_cards','cc_usage',
            'past_default','late_payments','repayment_history',
            'own_house','vehicle','investments',
            'avg_balance','monthly_txn','cash_withdrawal']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'page' not in st.session_state or st.session_state['page'] is None:
    st.session_state['page'] = 1


def show_progress(current, total=3):
    steps = ["Personal Details", "Financial Info", "Credit History"]
    cols = st.columns(total)
    for i, (col, step) in enumerate(zip(cols, steps)):
        num = i + 1
        if num < current:
            col.markdown(f"✅ **{step}**")
        elif num == current:
            col.markdown(f"🔵 **{step}** ← *You are here*")
        else:
            col.markdown(f"⬜ {step}")
    st.progress(current / total)
    st.markdown("")


def page_welcome():
    st.markdown("""
    <div class="hero-header">
        <h1>🏦 Credit Risk Analyser</h1>
        <p>Get your predicted credit score and default risk in under 2 minutes</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Welcome! Let\'s get started</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="step-subtitle">Tell us a little about yourself first</div>',
                    unsafe_allow_html=True)

        name = st.text_input("Your Full Name", placeholder="e.g. Shylaja P",
                             value=st.session_state['name'] or "")
        age  = st.number_input("Age", min_value=18, max_value=80,
                               value=int(st.session_state['age']) if st.session_state['age'] else 25)
        employment = st.selectbox("Employment Type",
            ["Student", "Salaried", "Self-employed", "Business owner"],
            index=["Student","Salaried","Self-employed","Business owner"].index(
                st.session_state['employment']) if st.session_state['employment'] else 1)
        experience = st.slider("Work Experience (years)", 0, 40,
                               int(st.session_state['experience']) if st.session_state['experience'] else 2)

        st.markdown("---")
        if st.button("Continue to Financial Details →"):
            if not name.strip():
                st.error("Please enter your name to continue.")
            else:
                st.session_state.update({
                    'name': name, 'age': age,
                    'employment': employment, 'experience': experience,
                    'page': 2
                })
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def page_financial():
    st.markdown(f"### Hello, {st.session_state['name']} 👋")
    show_progress(1)

    # --- Income & Savings ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">💰 Income & Savings</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Basic financial capacity</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    income = c1.number_input("Monthly Income (₹)", 0, 10000000,
        int(st.session_state['income']) if st.session_state['income'] else 50000, step=1000)
    savings = c2.number_input("Total Savings (₹)", 0, 50000000,
        int(st.session_state['savings']) if st.session_state['savings'] else 100000, step=5000)
    monthly_savings = st.number_input("Monthly Savings (₹)", 0, 1000000,
        int(st.session_state['monthly_savings']) if st.session_state['monthly_savings'] else 5000, step=500)

    if income > 0:
        sr = round(monthly_savings / income * 100, 1)
        color = "#10b981" if sr >= 20 else "#f59e0b" if sr >= 10 else "#ef4444"
        st.markdown(f"📊 Savings Rate: <span style='color:{color};font-weight:600'>{sr}%</span> "
                    f"({'Good ✅' if sr >= 20 else 'Average ⚠️' if sr >= 10 else 'Low 🔴'})",
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Debt ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">💳 Debt Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Outstanding obligations</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    debt = c1.number_input("Total Debt (₹)", 0, 50000000,
        int(st.session_state['debt']) if st.session_state['debt'] else 200000, step=5000)
    monthly_emi = c2.number_input("Monthly EMI (₹)", 0, 1000000,
        int(st.session_state['monthly_emi']) if st.session_state['monthly_emi'] else 10000, step=500)

    if income > 0:
        dr = round(monthly_emi / income * 100, 1)
        color = "#10b981" if dr <= 30 else "#f59e0b" if dr <= 50 else "#ef4444"
        st.markdown(f"📊 EMI-to-Income Ratio: <span style='color:{color};font-weight:600'>{dr}%</span> "
                    f"({'Safe ✅' if dr <= 30 else 'Caution ⚠️' if dr <= 50 else 'High Risk 🔴'})",
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Spending ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🛒 Monthly Spending (₹)</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Approximate monthly spend per category</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    food          = c1.number_input("🍔 Food",          0, 500000, int(st.session_state['food'] or 8000), step=500)
    shopping      = c2.number_input("👗 Shopping",      0, 500000, int(st.session_state['shopping'] or 5000), step=500)
    entertainment = c3.number_input("🎬 Entertainment", 0, 500000, int(st.session_state['entertainment'] or 3000), step=500)
    travel        = c4.number_input("✈️ Travel",        0, 500000, int(st.session_state['travel'] or 4000), step=500)

    c1, c2, c3, c4 = st.columns(4)
    bills     = c1.number_input("💡 Bills",     0, 500000, int(st.session_state['bills'] or 5000), step=500)
    rent      = c2.number_input("🏠 Rent",      0, 500000, int(st.session_state['rent'] or 15000), step=500)
    education = c3.number_input("📚 Education", 0, 500000, int(st.session_state['education'] or 2000), step=500)
    medical   = c4.number_input("🏥 Medical",   0, 500000, int(st.session_state['medical'] or 1000), step=500)

    total_expenses = food + shopping + entertainment + travel + bills + rent + education + medical
    if income > 0:
        er = round(total_expenses / income * 100, 1)
        color = "#10b981" if er <= 50 else "#f59e0b" if er <= 75 else "#ef4444"
        st.markdown(f"📊 Total Expenses: ₹{total_expenses:,} &nbsp;|&nbsp; "
                    f"Expense Ratio: <span style='color:{color};font-weight:600'>{er}%</span> "
                    f"({'Healthy ✅' if er <= 50 else 'Watch out ⚠️' if er <= 75 else 'Overspending 🔴'})",
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back"):
            st.session_state['page'] = 1
            st.rerun()
    with c2:
        if st.button("Continue to Credit History →"):
            st.session_state.update({
                'income': income, 'savings': savings,
                'monthly_savings': monthly_savings,
                'debt': debt, 'monthly_emi': monthly_emi,
                'food': food, 'shopping': shopping,
                'entertainment': entertainment, 'travel': travel,
                'bills': bills, 'rent': rent,
                'education': education, 'medical': medical,
                'page': 3
            })
            st.rerun()


def page_credit_history():
    st.markdown(f"### Almost there, {st.session_state['name']}! 🎯")
    show_progress(2)

    # --- Credit history ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">📋 Credit History</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Past behavior predicts future behavior</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    active_loans = c1.number_input("Active Loans", 0, 20,
        int(st.session_state['active_loans'] or 1))
    credit_cards = c2.number_input("Credit Cards", 0, 20,
        int(st.session_state['credit_cards'] or 1))
    cc_usage = c3.slider("Credit Card Usage (%)", 0, 100,
        int(st.session_state['cc_usage'] or 30))

    c1, c2 = st.columns(2)
    past_default = c1.radio("Past Loan Defaults?", ["No", "Yes"],
        index=["No","Yes"].index(st.session_state['past_default'] or "No"),
        horizontal=True)
    late_payments = c2.number_input("Late EMI Payments (count)", 0, 50,
        int(st.session_state['late_payments'] or 0))

    repayment_history = st.select_slider(
        "Loan Repayment History",
        options=["Poor", "Average", "Good", "Excellent"],
        value=st.session_state['repayment_history'] or "Good"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Assets ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🏠 Assets (Optional)</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Assets reduce perceived financial risk</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    own_house   = c1.radio("Own House?", ["No", "Yes"],
        index=["No","Yes"].index(st.session_state['own_house'] or "No"), horizontal=True)
    vehicle     = c2.radio("Vehicle Owned?", ["No", "Yes"],
        index=["No","Yes"].index(st.session_state['vehicle'] or "No"), horizontal=True)
    investments = c3.number_input("Investments (₹)", 0, 50000000,
        int(st.session_state['investments'] or 0), step=5000)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Banking behaviour ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🏧 Banking Behaviour</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Transaction patterns signal financial health</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    avg_balance      = c1.number_input("Avg Bank Balance (₹)", 0, 10000000,
        int(st.session_state['avg_balance'] or 50000), step=1000)
    monthly_txn      = c2.number_input("Monthly Transactions", 0, 500,
        int(st.session_state['monthly_txn'] or 20))
    cash_withdrawal  = c3.selectbox("Cash Withdrawal Frequency",
        ["Rarely", "Sometimes", "Often", "Very Often"],
        index=["Rarely","Sometimes","Often","Very Often"].index(
            st.session_state['cash_withdrawal'] or "Sometimes"))
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back"):
            st.session_state['page'] = 2
            st.rerun()
    with c2:
        if st.button("🔍 Analyse My Credit Risk"):
            st.session_state.update({
                'active_loans': active_loans,
                'credit_cards': credit_cards,
                'cc_usage': cc_usage,
                'past_default': past_default,
                'late_payments': late_payments,
                'repayment_history': repayment_history,
                'own_house': own_house,
                'vehicle': vehicle,
                'investments': investments,
                'avg_balance': avg_balance,
                'monthly_txn': monthly_txn,
                'cash_withdrawal': cash_withdrawal,
                'page': 4
            })
            st.rerun()


def compute_score_and_risk():
    """
    Compute a rule-based credit score (300-900) and default risk
    using the trained pipeline for the base prediction, then adjusting
    with the extra user-provided features that weren't in training data.
    """
    s = st.session_state

    income   = float(s['income'] or 50000)
    savings  = float(s['savings'] or 100000)
    debt     = float(s['debt'] or 200000)
    monthly_savings = float(s['monthly_savings'] or 5000)
    monthly_emi     = float(s['monthly_emi'] or 10000)

    total_expenses = sum([
        float(s[k] or 0) for k in
        ['food','shopping','entertainment','travel','bills','rent','education','medical']
    ])

    # Load trained pipeline
    try:
        pipeline = joblib.load("models/default_model_pipeline.pkl")
        medians  = pd.read_csv("models/train_medians.csv", index_col=0)["0"]
        expected = pipeline.named_steps['scaler'].feature_names_in_
        base_row = medians.reindex(expected).to_dict()
        base_row["INCOME"]          = income * 12
        base_row["SAVINGS"]         = savings
        base_row["DEBT"]            = debt
        base_row["R_SAVINGS_INCOME"]= savings / (income * 12) if income > 0 else 0
        base_row["R_DEBT_INCOME"]   = debt / (income * 12) if income > 0 else 0
        base_row["R_DEBT_SAVINGS"]  = debt / savings if savings > 0 else 0
        row_df = pd.DataFrame([base_row]).reindex(columns=expected, fill_value=0)
        default_proba = float(pipeline.predict_proba(row_df)[:, 1][0])
    except Exception:
        default_proba = 0.35

    # --- Rule-based credit score (300–900) ---
    score = 650

    # Income factor
    if   income >= 200000: score += 40
    elif income >= 100000: score += 25
    elif income >= 50000:  score += 10
    else:                  score -= 20

    # Savings rate
    sr = monthly_savings / income if income > 0 else 0
    if   sr >= 0.30: score += 35
    elif sr >= 0.20: score += 20
    elif sr >= 0.10: score += 5
    else:            score -= 15

    # Debt ratio
    dr = monthly_emi / income if income > 0 else 0
    if   dr <= 0.20: score += 40
    elif dr <= 0.30: score += 20
    elif dr <= 0.50: score -= 10
    else:            score -= 40

    # Expense ratio
    er = total_expenses / income if income > 0 else 0
    if   er <= 0.50: score += 20
    elif er <= 0.70: score += 5
    else:            score -= 25

    # Credit history
    rh = s['repayment_history'] or "Good"
    score += {"Excellent": 60, "Good": 30, "Average": 0, "Poor": -50}[rh]

    # Past default
    if s['past_default'] == "Yes": score -= 80
    late = int(s['late_payments'] or 0)
    score -= min(late * 10, 50)

    # Credit card usage
    cc = int(s['cc_usage'] or 30)
    if   cc <= 30: score += 20
    elif cc <= 60: score += 5
    else:          score -= 25

    # Active loans
    al = int(s['active_loans'] or 1)
    if   al == 0: score += 10
    elif al <= 2: score += 5
    elif al >  5: score -= 20

    # Assets
    if s['own_house'] == "Yes": score += 25
    if s['vehicle'] == "Yes":   score += 10
    inv = float(s['investments'] or 0)
    if inv >= 500000: score += 20
    elif inv >= 100000: score += 10

    # Banking behavior
    cw = s['cash_withdrawal'] or "Sometimes"
    score += {"Rarely": 10, "Sometimes": 0, "Often": -10, "Very Often": -20}[cw]

    # Employment
    emp = s['employment'] or "Salaried"
    score += {"Salaried": 20, "Business owner": 15,
              "Self-employed": 5, "Student": -10}[emp]

    score = max(300, min(900, score))

    # Adjust default proba with extra signals
    if s['past_default'] == "Yes": default_proba = min(default_proba + 0.20, 0.99)
    if rh == "Excellent":           default_proba = max(default_proba - 0.10, 0.01)
    if rh == "Poor":                default_proba = min(default_proba + 0.15, 0.99)
    if dr > 0.50:                   default_proba = min(default_proba + 0.10, 0.99)

    return score, default_proba


def score_band(score):
    if score >= 800: return "Excellent",  "band-excellent", "#059669"
    if score >= 740: return "Very Good",  "band-verygood",  "#16a34a"
    if score >= 670: return "Good",       "band-good",      "#ca8a04"
    if score >= 580: return "Fair",       "band-fair",      "#ea580c"
    return                   "Poor",      "band-poor",      "#dc2626"


def improvement_tips():
    s = st.session_state
    tips = []

    income = float(s['income'] or 1)
    monthly_savings = float(s['monthly_savings'] or 0)
    monthly_emi = float(s['monthly_emi'] or 0)
    total_exp = sum([float(s[k] or 0) for k in
                     ['food','shopping','entertainment','travel','bills','rent','education','medical']])

    sr = monthly_savings / income if income > 0 else 0
    dr = monthly_emi / income if income > 0 else 0
    er = total_exp / income if income > 0 else 0
    cc = int(s['cc_usage'] or 30)
    rh = s['repayment_history'] or "Good"

    if sr < 0.20:
        tips.append(("💰 Increase Your Savings Rate",
            f"You're saving {sr*100:.1f}% of income. Aim for at least 20%. "
            f"Even ₹{int(income*0.05):,} more per month significantly improves your score."))
    if dr > 0.30:
        tips.append(("💳 Reduce Your EMI Burden",
            f"Your EMIs consume {dr*100:.1f}% of income. Target below 30%. "
            "Consider prepaying smaller loans first to free up cash flow."))
    if er > 0.70:
        tips.append(("🛒 Control Monthly Spending",
            f"Expenses are {er*100:.1f}% of income. Reducing discretionary spend "
            "(entertainment, shopping) by 15% can meaningfully improve your ratio."))
    if cc > 30:
        tips.append(("💳 Lower Credit Card Utilisation",
            f"You're using {cc}% of your credit limit. Keep it below 30%. "
            "Pay off balances fully each month if possible."))
    if s['past_default'] == "Yes":
        tips.append(("📋 Address Past Defaults",
            "A past default is the single biggest score reducer. "
            "Ensuring timely payments for the next 12 months consistently improves credit history."))
    if rh in ["Poor", "Average"]:
        tips.append(("📅 Build a Clean Repayment Record",
            "Set up auto-debit for all EMIs and credit card minimums. "
            "12 consecutive on-time payments significantly improve your repayment history score."))
    if float(s['investments'] or 0) < 100000:
        tips.append(("📈 Start Investing",
            "Even small investments (SIP, FD, mutual funds) demonstrate financial planning "
            "and improve your overall financial strength score."))
    if s['own_house'] == "No" and s['vehicle'] == "No":
        tips.append(("🏠 Build Tangible Assets",
            "Owning property or a vehicle signals financial stability. "
            "If buying isn't feasible now, focus on growing investments instead."))

    if not tips:
        tips.append(("🌟 Your profile looks strong!",
            "Keep maintaining your savings rate, low EMI ratio, and clean repayment history. "
            "Consider increasing investments to further strengthen your financial position."))

    return tips


def page_results():
    score, default_proba = compute_score_and_risk()
    band_name, band_class, band_color = score_band(score)
    tips = improvement_tips()

    s = st.session_state

    st.markdown(f"## 📊 Credit Analysis for {s['name']}")
    show_progress(3)

    # --- Score + Risk side by side ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Predicted Credit Score</div>
            <div class="score-number">{score}</div>
            <span class="band-pill {band_class}">{band_name}</span>
            <div style="margin-top:1rem;opacity:0.8;font-size:0.85rem;">
                Range: 300 (Poor) → 900 (Excellent)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk_color  = "#10b981" if default_proba < 0.30 else \
                      "#f59e0b" if default_proba < 0.55 else "#ef4444"
        risk_label  = "Low Risk — Safe Borrower ✅" if default_proba < 0.30 else \
                      "Medium Risk — Review Needed ⚠️" if default_proba < 0.55 else \
                      "High Risk — Likely to Default 🚫"
        card_class  = "green" if default_proba < 0.30 else \
                      "amber" if default_proba < 0.55 else "red"

        st.markdown(f"""
        <div class="result-card {card_class}" style="height:100%">
            <h3 style="color:#1e3a5f;margin-top:0">Loan Default Risk</h3>
            <div style="font-size:3.5rem;font-weight:700;color:{risk_color};line-height:1">
                {default_proba:.1%}
            </div>
            <div style="font-size:1.1rem;margin-top:0.5rem;font-weight:500;color:{risk_color}">
                {risk_label}
            </div>
            <hr style="margin:1rem 0">
            <div style="font-size:0.85rem;color:#6b7280">
                Threshold: &lt;30% Approve | 30–55% Review | &gt;55% Decline
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # --- Score band reference ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">📈 Credit Score Bands</div>', unsafe_allow_html=True)
    b1, b2, b3, b4, b5 = st.columns(5)
    for col, rng, lbl, highlighted in [
        (b1, "300–579", "Poor",      score < 580),
        (b2, "580–669", "Fair",      580 <= score < 670),
        (b3, "670–739", "Good",      670 <= score < 740),
        (b4, "740–799", "Very Good", 740 <= score < 800),
        (b5, "800–900", "Excellent", score >= 800),
    ]:
        bg = "#1e3a5f" if highlighted else "#f3f4f6"
        fg = "white"   if highlighted else "#374151"
        col.markdown(f"""
        <div style="background:{bg};color:{fg};border-radius:10px;padding:0.8rem;
                    text-align:center;font-size:0.85rem">
            <div style="font-weight:700;font-size:1rem">{lbl}</div>
            <div style="opacity:0.8">{rng}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Key financial ratios summary ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">📊 Your Financial Snapshot</div>',
                unsafe_allow_html=True)

    income = float(s['income'] or 1)
    monthly_savings = float(s['monthly_savings'] or 0)
    monthly_emi = float(s['monthly_emi'] or 0)
    total_exp = sum([float(s[k] or 0) for k in
                     ['food','shopping','entertainment','travel','bills','rent','education','medical']])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Monthly Income",    f"₹{int(income):,}")
    r2.metric("Savings Rate",      f"{monthly_savings/income*100:.1f}%",
              delta="Good" if monthly_savings/income >= 0.20 else "Low",
              delta_color="normal" if monthly_savings/income >= 0.20 else "inverse")
    r3.metric("EMI/Income Ratio",  f"{monthly_emi/income*100:.1f}%",
              delta="Safe" if monthly_emi/income <= 0.30 else "High",
              delta_color="normal" if monthly_emi/income <= 0.30 else "inverse")
    r4.metric("Expense Ratio",     f"{total_exp/income*100:.1f}%",
              delta="Healthy" if total_exp/income <= 0.70 else "Overspending",
              delta_color="normal" if total_exp/income <= 0.70 else "inverse")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Improvement tips ---
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🚀 How to Improve Your Score</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Personalised recommendations based on your profile</div>',
                unsafe_allow_html=True)

    for title, body in tips:
        st.markdown(f"""
        <div class="tip-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Start over ---
    st.markdown("")
    if st.button("← Start Over / Analyse Another Customer"):
        for key in list(st.session_state.keys()):
            st.session_state[key] = None
        st.session_state['page'] = 1
        st.rerun()

    st.markdown("""
    <div class="footer">
        Built by <strong>Shylaja P</strong> &nbsp;|&nbsp;
        Logistic Regression + SHAP &nbsp;|&nbsp;
        <a href="https://github.com/shylaja-padmanaban/credit-risk-project">GitHub</a>
        &nbsp;|&nbsp; This is a portfolio project, not financial advice.
    </div>
    """, unsafe_allow_html=True)


# Router
page = st.session_state.get('page', 1)
if   page == 1: page_welcome()
elif page == 2: page_financial()
elif page == 3: page_credit_history()
elif page == 4: page_results()