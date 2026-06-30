"""
Predict default risk for a new customer from the command line.
Usage: python src/predict.py --income 50000 --debt 10000 --savings 20000
"""
import argparse
import pandas as pd
import joblib

MODEL_PATH = "models/default_model_pipeline.pkl"
TRAIN_MEDIANS_PATH = "models/train_medians.csv"


def risk_band(p):
    if p < 0.30:
        return "Low risk - Approve"
    elif p < 0.55:
        return "Medium risk - Manual review"
    else:
        return "High risk - Decline"


def predict(row_dict):
    pipeline = joblib.load(MODEL_PATH)
    expected_cols = pipeline.named_steps['scaler'].feature_names_in_

    medians = pd.read_csv(TRAIN_MEDIANS_PATH, index_col=0)["0"]
    base_row = medians.reindex(expected_cols).to_dict()
    base_row.update(row_dict)

    income = base_row.get("INCOME", 1) or 1
    savings = base_row.get("SAVINGS", 0)
    debt = base_row.get("DEBT", 0)
    base_row["R_SAVINGS_INCOME"] = savings / income
    base_row["R_DEBT_INCOME"] = debt / income
    base_row["R_DEBT_SAVINGS"] = debt / savings if savings > 0 else 0

    row_df = pd.DataFrame([base_row]).reindex(columns=expected_cols, fill_value=0)
    proba = pipeline.predict_proba(row_df)[:, 1][0]
    return {
        "default_probability": round(float(proba), 4),
        "risk_band": risk_band(proba)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--income", type=float, required=True)
    parser.add_argument("--savings", type=float, required=True)
    parser.add_argument("--debt", type=float, required=True)
    args = parser.parse_args()

    row = {"INCOME": args.income, "SAVINGS": args.savings, "DEBT": args.debt}
    result = predict(row)
    print(result)


if __name__ == "__main__":
    main()