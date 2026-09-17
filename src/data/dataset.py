"""
Patient dataset assembly and post-calibration beat extraction module.
Converts raw continuous ECG recordings and cardiologist annotations into
labeled feature matrices ready for model training and evaluation.
"""

from typing import List, Tuple
import numpy as np

from src.beats.segmentation import segment_beat
from src.features.extraction import (
    ALL_32_FEATURE_NAMES,
    extract_all_32_features,
)


def extract_patient_dataset(
    signal: np.ndarray,
    beats: np.ndarray,
    symbols: np.ndarray,
    template: np.ndarray,
    last_calib_index: int,
    fs: int = 360,
    window_pre: int = 36,
    window_post: int = 72,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract the feature matrix and binary label vector for all post-calibration eligible beats.

    Parameters:
    -----------
    signal : np.ndarray
        Filtered ECG signal.
    beats : np.ndarray
        Sample indices of annotated beats.
    symbols : np.ndarray
        Beat symbol annotations.
    template : np.ndarray
        Patient median template (108 samples).
    last_calib_index : int
        Highest annotation index among the calibration beats.
    fs : int
        Sampling rate in Hz (default: 360).
    window_pre : int
        Samples before R-peak (default: 36, 100 ms).
    window_post : int
        Samples after R-peak (default: 72, 200 ms).

    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (N_beats, 32).
    y : np.ndarray
        Binary label vector (0 = Normal 'N', 1 = PVC 'V') of shape (N_beats,).
    beat_indices : List[int]
        Annotation indices corresponding to extracted rows.
    """
    X_rows: List[List[float]] = []
    y_labels: List[int] = []
    extracted_indices: List[int] = []

    rr_intervals = np.diff(beats) / fs

    for i in range(last_calib_index + 1, len(beats) - 1):
        sym = symbols[i]
        if sym not in ["N", "V"]:
            continue

        seg = segment_beat(signal, beats[i], pre_samples=window_pre, post_samples=window_post)
        if seg is None:
            continue

        pre_rr = float((beats[i] - beats[i - 1]) / fs)
        post_rr = float((beats[i + 1] - beats[i]) / fs)
        local_rr = float(np.mean(rr_intervals[max(0, i - 5):i])) if i > 0 else pre_rr

        feat_dict = extract_all_32_features(
            seg, template, pre_rr, post_rr, local_rr
        )

        row = [feat_dict[fname] for fname in ALL_32_FEATURE_NAMES]
        X_rows.append(row)
        y_labels.append(1 if sym == "V" else 0)
        extracted_indices.append(i)

    return np.array(X_rows, dtype=np.float64), np.array(y_labels, dtype=np.int64), extracted_indices
