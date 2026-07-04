"""
Flask backend for Credit Risk Analyser
Run: python app_flask/app.py
"""
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'default_model_pipeline.pkl')
MEDIANS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'train_medians.csv')


def get_default_proba(income_annual, savings, debt):
    try:
        pipeline = joblib.load(MODEL_PATH)
        medians  = pd.read_csv(MEDIANS_PATH, index_col=0)["0"]
        expected = pipeline.named_steps['scaler'].feature_names_in_
        base_row = medians.reindex(expected).to_dict()
        base_row["INCOME"]           = income_annual
        base_row["SAVINGS"]          = savings
        base_row["DEBT"]             = debt
        base_row["R_SAVINGS_INCOME"] = savings / income_annual if income_annual > 0 else 0
        base_row["R_DEBT_INCOME"]    = debt / income_annual if income_annual > 0 else 0
        base_row["R_DEBT_SAVINGS"]   = debt / savings if savings > 0 else 0
        row_df = pd.DataFrame([base_row]).reindex(columns=expected, fill_value=0)
        return float(pipeline.predict_proba(row_df)[:, 1][0])
    except Exception as e:
        print(f"Model error: {e}")
        return 0.35


def compute_credit_score(data):
    income         = float(data.get('income', 0))
    monthly_savings= float(data.get('monthly_savings', 0))
    monthly_emi    = float(data.get('monthly_emi', 0))
    past_default   = data.get('past_default', 'No')
    repayment      = data.get('repayment_history', 'Good')
    cc_usage       = float(data.get('cc_usage', 30))
    active_loans   = int(data.get('active_loans', 1))
    employment     = data.get('employment', 'Salaried')
    own_house      = data.get('own_house', 'No')
    vehicle        = data.get('vehicle', 'No')
    investments    = float(data.get('investments', 0))
    cash_withdrawal= data.get('cash_withdrawal', 'Sometimes')
    late_payments  = int(data.get('late_payments', 0))

    spend_keys = ['food','shopping','entertainment','travel','bills','rent','education','medical']
    total_exp  = sum(float(data.get(k, 0)) for k in spend_keys)

    score = 650

    if   income >= 200000: score += 40
    elif income >= 100000: score += 25
    elif income >= 50000:  score += 10
    else:                  score -= 20

    sr = monthly_savings / income if income > 0 else 0
    if   sr >= 0.30: score += 35
    elif sr >= 0.20: score += 20
    elif sr >= 0.10: score +=  5
    else:            score -= 15

    dr = monthly_emi / income if income > 0 else 0
    if   dr <= 0.20: score += 40
    elif dr <= 0.30: score += 20
    elif dr <= 0.50: score -= 10
    else:            score -= 40

    er = total_exp / income if income > 0 else 0
    if   er <= 0.50: score += 20
    elif er <= 0.70: score +=  5
    else:            score -= 25

    score += {"Excellent": 60, "Good": 30, "Average": 0, "Poor": -50}[repayment]

    if past_default == "Yes":
        score -= 80
    score -= min(late_payments * 10, 50)

    if   cc_usage <= 30: score += 20
    elif cc_usage <= 60: score +=  5
    else:                score -= 25

    if   active_loans == 0: score += 10
    elif active_loans <= 2: score +=  5
    elif active_loans > 5:  score -= 20

    if own_house  == "Yes": score += 25
    if vehicle    == "Yes": score += 10
    if investments >= 500000: score += 20
    elif investments >= 100000: score += 10

    score += {"Rarely": 10, "Sometimes": 0, "Often": -10, "Very Often": -20}[cash_withdrawal]
    score += {"Salaried": 20, "Business owner": 15, "Self-employed": 5, "Student": -10}[employment]

    return max(300, min(900, int(score)))


def get_score_band(score):
    if score >= 800: return "Excellent",  "#10b981"
    if score >= 740: return "Very Good",  "#22c55e"
    if score >= 670: return "Good",       "#eab308"
    if score >= 580: return "Fair",       "#f97316"
    return                   "Poor",      "#ef4444"


def get_risk_band(proba):
    if proba < 0.30: return "Low Risk",    "approve", "#10b981"
    if proba < 0.55: return "Medium Risk", "review",  "#f59e0b"
    return                   "High Risk",  "decline", "#ef4444"


def get_tips(data):
    tips = []
    income          = float(data.get('income', 1)) or 1
    monthly_savings = float(data.get('monthly_savings', 0))
    monthly_emi     = float(data.get('monthly_emi', 0))
    spend_keys = ['food','shopping','entertainment','travel','bills','rent','education','medical']
    total_exp  = sum(float(data.get(k, 0)) for k in spend_keys)
    sr = monthly_savings / income
    dr = monthly_emi / income
    er = total_exp / income
    cc = float(data.get('cc_usage', 30))
    rh = data.get('repayment_history', 'Good')

    if sr < 0.20:
        tips.append({
            "icon": "💰",
            "title": "Increase Your Savings Rate",
            "body": f"You're saving {sr*100:.1f}% of income. Aim for 20%+. "
                    f"An extra ₹{int(income*0.05):,}/month makes a significant difference."
        })
    if dr > 0.30:
        tips.append({
            "icon": "💳",
            "title": "Reduce Your EMI Burden",
            "body": f"EMIs take up {dr*100:.1f}% of your income. Target below 30%. "
                    "Prepay smaller loans first to free up cash flow."
        })
    if er > 0.70:
        tips.append({
            "icon": "🛒",
            "title": "Control Monthly Spending",
            "body": f"Expenses are {er*100:.1f}% of income. Cutting discretionary spend "
                    "(entertainment, shopping) by 15% improves your ratio meaningfully."
        })
    if cc > 30:
        tips.append({
            "icon": "💳",
            "title": "Lower Credit Card Utilisation",
            "body": f"You're using {cc:.0f}% of your credit limit. Keep below 30%. "
                    "Pay balances in full each month where possible."
        })
    if data.get('past_default') == "Yes":
        tips.append({
            "icon": "📋",
            "title": "Address Past Defaults",
            "body": "A past default is the single biggest score reducer. "
                    "12 consecutive on-time payments significantly rebuild credit history."
        })
    if rh in ["Poor", "Average"]:
        tips.append({
            "icon": "📅",
            "title": "Build a Clean Repayment Record",
            "body": "Set up auto-debit for all EMIs and credit card minimums. "
                    "Consistent on-time payments improve your history score fastest."
        })
    if float(data.get('investments', 0)) < 100000:
        tips.append({
            "icon": "📈",
            "title": "Start Investing",
            "body": "Even small SIPs or FDs demonstrate financial planning and "
                    "improve your overall financial strength score."
        })
    if not tips:
        tips.append({
            "icon": "🌟",
            "title": "Your profile looks strong!",
            "body": "Keep maintaining your savings rate, low EMI ratio, and clean "
                    "repayment history. Consider growing investments further."
        })
    return tips


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    income_monthly = float(data.get('income', 0))
    income_annual  = income_monthly * 12
    savings        = float(data.get('savings', 0))
    debt           = float(data.get('debt', 0))

    score          = compute_credit_score(data)
    band_name, band_color = get_score_band(score)

    default_proba  = get_default_proba(income_annual, savings, debt)
    pd_name = data.get('past_default', 'No')
    rh      = data.get('repayment_history', 'Good')
    dr      = float(data.get('monthly_emi', 0)) / (income_monthly or 1)
    if pd_name == "Yes":  default_proba = min(default_proba + 0.20, 0.99)
    if rh == "Excellent": default_proba = max(default_proba - 0.10, 0.01)
    if rh == "Poor":      default_proba = min(default_proba + 0.15, 0.99)
    if dr > 0.50:         default_proba = min(default_proba + 0.10, 0.99)

    risk_name, risk_class, risk_color = get_risk_band(default_proba)
    tips = get_tips(data)

    spend_keys = ['food','shopping','entertainment','travel','bills','rent','education','medical']
    total_exp  = sum(float(data.get(k, 0)) for k in spend_keys)

    return jsonify({
        "name":           data.get('name', 'User'),
        "score":          score,
        "band_name":      band_name,
        "band_color":     band_color,
        "default_proba":  round(default_proba * 100, 1),
        "risk_name":      risk_name,
        "risk_class":     risk_class,
        "risk_color":     risk_color,
        "metrics": {
            "savings_rate": round(float(data.get('monthly_savings',0)) / (income_monthly or 1) * 100, 1),
            "emi_ratio":    round(dr * 100, 1),
            "expense_ratio":round(total_exp / (income_monthly or 1) * 100, 1),
            "debt_income":  round(debt / (income_annual or 1), 2),
        },
        "tips": tips
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)