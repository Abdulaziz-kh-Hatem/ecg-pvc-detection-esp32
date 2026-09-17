"""
Cross-validation and statistical uncertainty estimation module for ECG studies.
Implements patient-level grouped splits, Leave-One-Patient-Out (LOPO), and
stratified bootstrap confidence intervals.
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold


def generate_lopo_splits(
    patient_ids: List[str]
) -> List[Tuple[List[str], str]]:
    """
    Generate Leave-One-Patient-Out (LOPO) split partitions across a list of patient IDs.

    Parameters:
    -----------
    patient_ids : List[str]
        List of distinct patient record identifiers.

    Returns:
    --------
    splits : List[Tuple[List[str], str]]
        List of (train_patients, test_patient) pairs.
    """
    if len(patient_ids) < 2:
        raise ValueError("At least 2 patient IDs are required to generate LOPO splits.")

    splits = []
    for test_pid in patient_ids:
        train_pids = [p for p in patient_ids if p != test_pid]
        splits.append((train_pids, test_pid))
    return splits


def generate_grouped_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate grouped K-fold splits ensuring zero patient-group overlap between train and test.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    groups : np.ndarray
        Group/patient identifier for each sample.
    n_splits : int
        Number of folds (default: 5).

    Returns:
    --------
    splits : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) arrays.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Number of groups ({len(unique_groups)}) is less than n_splits ({n_splits}).")

    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert len(train_groups.intersection(test_groups)) == 0, "Group overlap detected in split!"
        splits.append((train_idx, test_idx))
    return splits


def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 95.0,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate the non-parametric percentile bootstrap confidence interval for a metric vector.

    Parameters:
    -----------
    values : np.ndarray
        Array of metric values across patients or folds.
    n_bootstraps : int
        Number of bootstrap iterations (default: 1000).
    ci : float
        Confidence level percentage (default: 95.0, must be between 0 and 100).
    random_state : int
        Random seed (default: 42).

    Returns:
    --------
    mean : float
        Sample mean.
    lower_bound : float
        Lower confidence limit.
    upper_bound : float
        Upper confidence limit.
    """
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        raise ValueError("Cannot compute bootstrap CI on empty array.")
    if not (0.0 < ci < 100.0):
        raise ValueError(f"Confidence level ci must be between 0 and 100, got {ci}.")

    rng = np.random.default_rng(random_state)
    boot_means = []
    n = len(vals)

    for _ in range(n_bootstraps):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    mean_val = float(np.mean(vals))

    return mean_val, lower, upper
