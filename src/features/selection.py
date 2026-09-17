"""
Dual-Voting Feature Selection Module.
Combines Random Forest (Gini impurity) and XGBoost (Information Gain) importance metrics
to identify the top-12 most discriminative features strictly on training data.
"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def dual_voting_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int = 12,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Select the top N features using consensus voting between Random Forest and XGBoost.
    Computed strictly on the training partition to prevent feature selection leakage.

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix of shape (N_train, n_features_total).
    y_train : np.ndarray
        Training binary label vector of shape (N_train,).
    n_features : int
        Number of top features to select (default: 12).
    random_state : int
        Seed for deterministic execution (default: 42).

    Returns:
    --------
    top_indices : np.ndarray
        Array of column indices corresponding to the selected features.
    scores_dict : Dict[str, Any]
        Dictionary containing raw and normalized importance scores.
    """
    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train contains non-finite values (NaN or Inf)")
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train must contain at least two classes to compute feature importance")
    if n_features <= 0 or n_features > X_train.shape[1]:
        raise ValueError(f"n_features must be between 1 and {X_train.shape[1]}, got {n_features}")

    # 1. Random Forest Voter (Mean Decrease in Impurity / Gini)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    # 2. XGBoost Voter (Information Gain)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_importances = xgb_model.feature_importances_

    # 3. Min-Max Normalization (0-1 scale) & Consensus Aggregation
    rf_max = float(np.max(rf_importances)) if np.max(rf_importances) > 0 else 1.0
    xgb_max = float(np.max(xgb_importances)) if np.max(xgb_importances) > 0 else 1.0

    rf_norm = rf_importances / (rf_max + 1e-9)
    xgb_norm = xgb_importances / (xgb_max + 1e-9)
    aggregated = (rf_norm + xgb_norm) / 2.0

    # 4. Top N selection
    top_indices = np.argsort(aggregated)[-n_features:]

    scores_dict = {
        "rf_scores": rf_importances[top_indices].tolist(),
        "xgb_scores": xgb_importances[top_indices].tolist(),
        "aggregated_scores": aggregated[top_indices].tolist(),
        "all_aggregated_scores": aggregated.tolist()
    }

    return top_indices, scores_dict
