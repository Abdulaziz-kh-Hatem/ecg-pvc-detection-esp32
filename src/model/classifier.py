"""
Decision Tree Classifier for ECG Premature Ventricular Contraction (PVC) detection.
Implements balanced entropy-criterion decision tree models for lightweight beat classification.
"""

from typing import Optional, Union, Dict
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def create_pvc_classifier(
    max_depth: int = 8,
    criterion: str = "entropy",
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Construct a Decision Tree classifier configured for lightweight PVC classification.

    Parameters:
    -----------
    max_depth : int
        Maximum tree depth to prevent overfitting (default: 8).
    criterion : str
        Split quality criterion (default: 'entropy').
    class_weight : str or dict
        Class weighting strategy to compensate for Normal vs. PVC class imbalance (default: 'balanced').
    random_state : int
        Deterministic random seed (default: 42).

    Returns:
    --------
    clf : DecisionTreeClassifier
        Instantiated, unfitted scikit-learn DecisionTreeClassifier.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
    )


def train_pvc_classifier(
    clf: DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> DecisionTreeClassifier:
    """
    Fit the Decision Tree classifier onto the training partition.

    Parameters:
    -----------
    clf : DecisionTreeClassifier
        Decision tree model instance.
    X_train : np.ndarray
        Training feature matrix of shape (N_train, n_features).
    y_train : np.ndarray
        Training binary target labels (0 = Normal, 1 = PVC).

    Returns:
    --------
    clf : DecisionTreeClassifier
        Fitted DecisionTreeClassifier.
    """
    clf.fit(X_train, y_train)
    return clf


def predict_pvc(clf: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
    """
    Predict binary PVC labels (0 = Normal, 1 = PVC) for an input feature matrix.

    Parameters:
    -----------
    clf : DecisionTreeClassifier
        Fitted DecisionTreeClassifier.
    X : np.ndarray
        Feature matrix of shape (N_samples, n_features).

    Returns:
    --------
    predictions : np.ndarray
        1D array of binary predictions.
    """
    return clf.predict(X)
