 # Credit Risk Default Prediction

![Tests](https://github.com/shylaja-padmanaban/credit-risk-project/actions/workflows/tests.yml/badge.svg)

## 🚀 Live Demo
👉 **[Try the app here](https://credit-risk-project-mbvl6yedog3tfhns2q5fxf.streamlit.app)**

Predicts whether a bank customer is likely to default on a loan based on
their financial profile. Built end-to-end: EDA → modeling → SHAP
explainability → CLI inference → Streamlit demo with CI/CD.

## Results

| Task | Model | Key Metric |
|------|-------|------------|
| Credit Score (Regression) | Linear Regression | Test R² = 0.853 |
| Default Prediction (Classification) | Logistic Regression | ROC-AUC = 0.685, Recall = 0.62 |

**Risk band validation:** "Decline" customers defaulted at 39.2% vs 8.2%
for "Approve" — a 4.8x risk separation.

**Top SHAP drivers:** R_DEBT_INCOME, R_SAVINGS_INCOME, R_TAX_DEBT



## How to run locally
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Predict for a new customer
python src/predict.py --income 85000 --savings 273000 --debt 395000

# Launch demo app
streamlit run app.py

# Run tests
pytest tests/ -v
```

## Key findings
- Linear Regression outperformed tree models for credit score prediction
- Hyperparameter tuning was tested but degraded performance on this small
  dataset — kept default-parameter Logistic Regression (deliberate decision)
- R_GAMBLING_INCOME showed counterintuitive SHAP direction — flagged as
  limitation for further investigation
- Risk bands validated: "Decline" customers were 4.8x more likely to default
  than "Approve" customers (39.2% vs 8.2%)

## Tech stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit, Pytest,
GitHub Actions

## Project Structure
 
 
 Results Summary

**Regression (CREDIT_SCORE):** Linear Regression, Test R² = 0.853

**Classification (DEFAULT):** Logistic Regression (class_weight='balanced'),
Test ROC-AUC = 0.685, Test Recall = 0.62

**Risk bands validated:** customers flagged "Decline" were 4.8x more likely
to actually default than those flagged "Approve" (39.2% vs 8.2%)

**Top risk drivers (SHAP):** R_DEBT_INCOME, R_SAVINGS_INCOME, R_TAX_DEBT

See `notebooks/credit-risk-analysis.ipynb` for full analysis, explainability,
and limitations.