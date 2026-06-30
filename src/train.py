"""
Train the credit default classification model and save the pipeline.
Run from project root: python src/train.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import joblib

RANDOM_STATE = 42
DATA_PATH = "data/credit_score.csv"
MODEL_OUTPUT_PATH = "models/default_model_pipeline.pkl"


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.drop(columns=["CUST_ID"])

    gambling_order = ["No", "Low", "High"]
    df["CAT_GAMBLING"] = pd.Categorical(
        df["CAT_GAMBLING"], categories=gambling_order, ordered=True
    ).codes

    redundant_6_month_cols = [
        col for col in df.columns
        if col.endswith('_6') and col.replace('_6', '_12') in df.columns
    ]
    df = df.drop(columns=redundant_6_month_cols)
    return df


def split_and_clean_outliers(df, test_size=0.2):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RANDOM_STATE, stratify=df['DEFAULT']
    )

    outlier_check_cols = ['INCOME', 'SAVINGS', 'DEBT']
    Q1 = train_df[outlier_check_cols].quantile(0.25)
    Q3 = train_df[outlier_check_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    train_mask = ((train_df[outlier_check_cols] < lower_bound) |
                  (train_df[outlier_check_cols] > upper_bound)).any(axis=1)
    train_df = train_df[~train_mask]

    test_mask = ((test_df[outlier_check_cols] < lower_bound) |
                 (test_df[outlier_check_cols] > upper_bound)).any(axis=1)
    test_df = test_df[~test_mask]

    return train_df, test_df


def train_classifier(train_df, test_df):
    drop_cols = ["CREDIT_SCORE", "DEFAULT"]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["DEFAULT"]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["DEFAULT"]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"Test Recall: {recall_score(y_test, pred):.3f}")
    print(f"Test Precision: {precision_score(y_test, pred):.3f}")

    return pipeline


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    df = load_and_clean_data(DATA_PATH)
    train_df, test_df = split_and_clean_outliers(df)
    pipeline = train_classifier(train_df, test_df)
    X_train_for_median = train_df.drop(columns=["CREDIT_SCORE", "DEFAULT"])
    X_train_for_median.median().to_csv("models/train_medians.csv")
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Saved model to {MODEL_OUTPUT_PATH}")