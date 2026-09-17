"""
Unit tests for Decision Tree model construction and prediction.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.model.classifier import create_pvc_classifier, train_pvc_classifier, predict_pvc


def test_create_pvc_classifier_hyperparameters():
    clf = create_pvc_classifier(max_depth=8, criterion="entropy", class_weight="balanced", random_state=42)
    assert isinstance(clf, DecisionTreeClassifier)
    assert clf.max_depth == 8
    assert clf.criterion == "entropy"
    assert clf.class_weight == "balanced"
    assert clf.random_state == 42


def test_train_and_predict_deterministic():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 12))
    y_train = np.array([0] * 70 + [1] * 30)

    clf = create_pvc_classifier(max_depth=4, random_state=42)
    clf = train_pvc_classifier(clf, X_train, y_train)

    X_test = rng.standard_normal((20, 12))
    preds_1 = predict_pvc(clf, X_test)
    preds_2 = predict_pvc(clf, X_test)

    assert len(preds_1) == 20
    assert np.array_equal(preds_1, preds_2)
    assert set(preds_1).issubset({0, 1})
