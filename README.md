 Results Summary

**Regression (CREDIT_SCORE):** Linear Regression, Test R² = 0.853

**Classification (DEFAULT):** Logistic Regression (class_weight='balanced'),
Test ROC-AUC = 0.685, Test Recall = 0.62

**Risk bands validated:** customers flagged "Decline" were 4.8x more likely
to actually default than those flagged "Approve" (39.2% vs 8.2%)

**Top risk drivers (SHAP):** R_DEBT_INCOME, R_SAVINGS_INCOME, R_TAX_DEBT

See `notebooks/credit-risk-analysis.ipynb` for full analysis, explainability,
and limitations.