"""
Unit tests for 32-feature extraction engine and dual-voting feature selection.
"""

import numpy as np
import pytest
from src.features.extraction import (
    ALL_32_FEATURE_NAMES,
    extract_all_32_features,
)
from src.features.selection import dual_voting_feature_selection


def test_extract_all_32_features_keys_and_values():
    segment = np.sin(np.linspace(0, np.pi, 108))
    template = np.sin(np.linspace(0, np.pi, 108))
    pre_rr = 0.85
    post_rr = 0.90
    local_rr = 0.88

    feat_dict = extract_all_32_features(segment, template, pre_rr, post_rr, local_rr)

    assert len(feat_dict) == 32, f"Expected 32 features, got {len(feat_dict)}"
    for name in ALL_32_FEATURE_NAMES:
        assert name in feat_dict, f"Missing feature: {name}"
        val = feat_dict[name]
        assert np.isfinite(val), f"Feature {name} produced non-finite value: {val}"

    # For identical segment and template, SAD should be 0.0, Corr_Coeff should be 1.0
    assert abs(feat_dict["SAD"]) < 1e-6
    assert abs(feat_dict["Corr_Coeff"] - 1.0) < 1e-4
    assert abs(feat_dict["Discordance"]) < 1e-6


def test_flatline_zero_variance_input():
    # Constant amplitude flatline (zero variance, zero derivative)
    segment = np.zeros(108)
    template = np.zeros(108)
    pre_rr = 1.0
    post_rr = 1.0
    local_rr = 1.0

    # Must execute safely without ZeroDivisionError, Inf, or NaN
    feat_dict = extract_all_32_features(segment, template, pre_rr, post_rr, local_rr)
    for name, val in feat_dict.items():
        assert np.isfinite(val), f"Feature {name} is non-finite on flatline: {val}"


def test_feature_mathematical_bounds():
    rng = np.random.default_rng(42)
    segment = rng.standard_normal(108)
    template = rng.standard_normal(108)

    feat_dict = extract_all_32_features(segment, template, pre_rr=0.8, post_rr=0.9, local_rr_avg=0.85)

    # Correlation coefficient must be bounded in [-1, 1]
    assert -1.0 - 1e-6 <= feat_dict["Corr_Coeff"] <= 1.0 + 1e-6
    # Discordance and Residual Energy must be non-negative
    assert feat_dict["Discordance"] >= 0.0
    assert feat_dict["Res_Energy"] >= 0.0
    assert feat_dict["SAD"] >= 0.0
    assert feat_dict["Energy"] >= 0.0


def test_dual_voting_feature_selection():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 32))
    y = np.array([0] * 50 + [1] * 50)

    top_idx, scores = dual_voting_feature_selection(X, y, n_features=12, random_state=42)

    assert len(top_idx) == 12
    assert len(scores["rf_scores"]) == 12
    assert len(scores["xgb_scores"]) == 12
    assert len(scores["all_aggregated_scores"]) == 32
    assert len(np.unique(top_idx)) == 12


def test_dual_voting_validation_errors():
    # Non-finite input
    X_bad = np.array([[np.nan, 1.0], [2.0, 3.0]])
    y_good = np.array([0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        dual_voting_feature_selection(X_bad, y_good, n_features=1)

    # Single class
    X_good = np.ones((10, 5))
    y_single = np.zeros(10)
    with pytest.raises(ValueError, match="at least two classes"):
        dual_voting_feature_selection(X_good, y_single, n_features=2)
