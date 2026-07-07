from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'default_model_pipeline.pkl')
MEDIANS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'train_medians.csv')


def get_default_proba(income_annual, savings, debt):
    try:
        pipeline = joblib.load(MODEL_PATH)
        medians  = pd.read_csv(MEDIANS_PATH, index_col=0)["0"]
        expected = pipeline.named_steps['scaler'].feature_names_in_
        row      = medians.reindex(expected).to_dict()
        row["INCOME"]           = income_annual
        row["SAVINGS"]          = savings
        row["DEBT"]             = debt
        row["R_SAVINGS_INCOME"] = savings / income_annual if income_annual > 0 else 0
        row["R_DEBT_INCOME"]    = debt / income_annual   if income_annual > 0 else 0
        row["R_DEBT_SAVINGS"]   = debt / savings         if savings > 0 else 0
        df = pd.DataFrame([row]).reindex(columns=expected, fill_value=0)
        return float(pipeline.predict_proba(df)[:, 1][0])
    except Exception as e:
        print("Model error:", e)
        return 0.35


def compute_score(d):
    income  = float(d.get('income', 0))
    ms      = float(d.get('monthly_savings', 0))
    emi     = float(d.get('monthly_emi', 0))
    spend   = sum(float(d.get(k, 0)) for k in
                  ['food', 'shopping', 'entertainment', 'travel', 'bills', 'rent', 'education', 'medical'])
    rh      = d.get('repayment_history', 'Good')
    default = d.get('past_default', 'No')
    cc      = float(d.get('cc_usage', 30))
    loans   = int(d.get('active_loans', 1))
    emp     = d.get('employment', 'Salaried')
    house   = d.get('own_house', 'No')
    vehicle = d.get('vehicle', 'No')
    invest  = float(d.get('investments', 0))
    late    = int(d.get('late_payments', 0))
    cw      = d.get('cash_withdrawal', 'Sometimes')

    score = 650

    if   income >= 200000: score += 40
    elif income >= 100000: score += 25
    elif income >= 50000:  score += 10
    else:                  score -= 20

    sr = ms / income if income > 0 else 0
    if   sr >= 0.30: score += 35
    elif sr >= 0.20: score += 20
    elif sr >= 0.10: score +=  5
    else:            score -= 15

    dr = emi / income if income > 0 else 0
    if   dr <= 0.20: score += 40
    elif dr <= 0.30: score += 20
    elif dr <= 0.50: score -= 10
    else:            score -= 40

    er = spend / income if income > 0 else 0
    if   er <= 0.50: score += 20
    elif er <= 0.70: score +=  5
    else:            score -= 25

    score += {"Excellent": 60, "Good": 30, "Average": 0, "Poor": -50}[rh]

    if default == "Yes": score -= 80
    score -= min(late * 10, 50)

    if   cc <= 30: score += 20
    elif cc <= 60: score +=  5
    else:          score -= 25

    if   loans == 0: score += 10
    elif loans <= 2: score +=  5
    elif loans >  5: score -= 20

    if house   == "Yes": score += 25
    if vehicle == "Yes": score += 10
    if invest >= 500000: score += 20
    elif invest >= 100000: score += 10

    score += {"Rarely": 10, "Sometimes": 0, "Often": -10, "Very Often": -20}[cw]
    score += {"Salaried": 20, "Business owner": 15, "Self-employed": 5, "Student": -10}[emp]

    return max(300, min(900, int(score)))


def score_band(score):
    if score >= 800: return "Excellent",  "#10b981"
    if score >= 740: return "Very Good",  "#3b82f6"
    if score >= 670: return "Good",       "#f59e0b"
    if score >= 580: return "Fair",       "#f97316"
    return                   "Poor",      "#ef4444"


def risk_band(p):
    if p < 0.30: return "Low",    "approve", "#10b981"
    if p < 0.55: return "Medium", "review",  "#f59e0b"
    return              "High",   "decline", "#ef4444"


def build_tips(d):
    tips = []
    income = float(d.get('income', 1)) or 1
    ms     = float(d.get('monthly_savings', 0))
    emi    = float(d.get('monthly_emi', 0))
    spend  = sum(float(d.get(k, 0)) for k in
                 ['food','shopping','entertainment','travel','bills','rent','education','medical'])
    cc     = float(d.get('cc_usage', 30))
    rh     = d.get('repayment_history', 'Good')
    inv    = float(d.get('investments', 0))

    sr = ms / income
    dr = emi / income
    er = spend / income

    if sr < 0.20:
        tips.append({
            "title": "Grow your savings",
            "body": f"You are saving {sr*100:.1f}% of your income. Reaching 20% "
                    f"gives lenders confidence in your financial discipline and directly improves your score."
        })
    if dr > 0.30:
        tips.append({
            "title": "Reduce your loan burden",
            "body": f"Your monthly repayments take up {dr*100:.1f}% of income. "
                    "Clearing smaller loans first can free up cash flow and improve your profile quickly."
        })
    if er > 0.70:
        tips.append({
            "title": "Manage your monthly expenses",
            "body": f"Spending is at {er*100:.1f}% of income. Bringing this below 70% "
                    "signals better financial control to lenders."
        })
    if cc > 30:
        tips.append({
            "title": "Lower your card utilisation",
            "body": f"Using {cc:.0f}% of your credit limit is above the recommended 30%. "
                    "Paying down your balance improves this ratio quickly."
        })
    if rh in ["Poor", "Average"]:
        tips.append({
            "title": "Build a clean repayment record",
            "body": "Setting up automatic payments ensures you never miss a due date. "
                    "Twelve consecutive on-time payments significantly strengthen your history."
        })
    if d.get('past_default') == "Yes":
        tips.append({
            "title": "Address previous defaults",
            "body": "A past default has a strong negative impact. Consistent on-time payments "
                    "over the next 12 months is the fastest way to begin rebuilding trust."
        })
    if inv < 100000:
        tips.append({
            "title": "Start building investments",
            "body": "Even small, regular investments in fixed deposits or mutual funds "
                    "demonstrate long-term financial planning and strengthen your overall profile."
        })
    if not tips:
        tips.append({
            "title": "Your profile is strong",
            "body": "Keep maintaining your current savings discipline, low repayment burden, "
                    "and consistent payment history. Consider increasing investments to further strengthen your position."
        })
    return tips


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    d      = request.json
    income = float(d.get('income', 0))
    savings= float(d.get('savings', 0))
    debt   = float(d.get('debt', 0))

    score           = compute_score(d)
    band_name, band_color = score_band(score)
    proba           = get_default_proba(income * 12, savings, debt)

    rh = d.get('repayment_history', 'Good')
    dr = float(d.get('monthly_emi', 0)) / (income or 1)
    if d.get('past_default') == "Yes": proba = min(proba + 0.20, 0.99)
    if rh == "Excellent":              proba = max(proba - 0.10, 0.01)
    if rh == "Poor":                   proba = min(proba + 0.15, 0.99)
    if dr > 0.50:                      proba = min(proba + 0.10, 0.99)

    risk_name, risk_class, risk_color = risk_band(proba)
    tips = build_tips(d)

    spend = sum(float(d.get(k, 0)) for k in
                ['food','shopping','entertainment','travel','bills','rent','education','medical'])
    ms    = float(d.get('monthly_savings', 0))
    emi   = float(d.get('monthly_emi', 0))

    return jsonify({
        "name":        d.get('name', 'User'),
        "score":       score,
        "band_name":   band_name,
        "band_color":  band_color,
        "proba":       round(proba * 100, 1),
        "risk_name":   risk_name,
        "risk_class":  risk_class,
        "risk_color":  risk_color,
        "tips":        tips,
        "metrics": {
            "savings_rate":  round(ms / income * 100, 1) if income else 0,
            "emi_ratio":     round(emi / income * 100, 1) if income else 0,
            "expense_ratio": round(spend / income * 100, 1) if income else 0,
            "debt_income":   round(debt / (income * 12), 2) if income else 0,
        },
        "spend_breakdown": {
            "Food":           float(d.get('food', 0)),
            "Shopping":       float(d.get('shopping', 0)),
            "Entertainment":  float(d.get('entertainment', 0)),
            "Travel":         float(d.get('travel', 0)),
            "Bills":          float(d.get('bills', 0)),
            "Rent":           float(d.get('rent', 0)),
            "Education":      float(d.get('education', 0)),
            "Medical":        float(d.get('medical', 0)),
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)