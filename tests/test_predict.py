import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import predict, risk_band


def test_risk_band_boundaries():
    assert risk_band(0.10) == "Low risk - Approve"
    assert risk_band(0.40) == "Medium risk - Manual review"
    assert risk_band(0.80) == "High risk - Decline"


def test_predict_returns_expected_keys():
    sample = {"INCOME": 50000, "SAVINGS": 20000, "DEBT": 10000}
    result = predict(sample)
    assert "default_probability" in result
    assert "risk_band" in result
    assert 0 <= result["default_probability"] <= 1