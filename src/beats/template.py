"""
Subject-specific calibration and reference template construction module.
Extracts baseline morphological templates using consecutive normal sinus beats.
"""

from typing import Tuple, List, Literal
import numpy as np

from src.beats.segmentation import segment_beat


def build_subject_template(
    signal: np.ndarray,
    beats: np.ndarray,
    symbols: np.ndarray,
    n_beats: int = 50,
    window_pre: int = 36,
    window_post: int = 72,
    method: Literal["median", "mean"] = "median"
) -> Tuple[np.ndarray, int]:
    """
    Construct a subject-specific reference template from the first N normal beats.

    Parameters:
    -----------
    signal : np.ndarray
        Filtered 1D continuous ECG signal.
    beats : np.ndarray
        Array of R-peak sample indices from annotations.
    symbols : np.ndarray
        Array of annotation beat symbols corresponding to `beats`.
    n_beats : int
        Number of normal beats required for calibration (default: 50).
    window_pre : int
        Samples before R-peak (default: 36 samples, 100 ms).
    window_post : int
        Samples after R-peak (default: 72 samples, 200 ms).
    method : {'median', 'mean'}
        Aggregation strategy across calibration beats (default: 'median' for outlier robustness).

    Returns:
    --------
    template : np.ndarray
        1D reference template of length (window_pre + window_post = 108 samples).
    last_calib_index : int
        The highest annotation index among the calibration beats. All subsequent processing
        MUST start after this index to prevent self-calibration data leakage.
    """
    temp_beats: List[np.ndarray] = []
    calib_indices: List[int] = []

    for i in range(1, len(beats) - 1):
        if len(temp_beats) >= n_beats:
            break
        if symbols[i] == "N":
            seg = segment_beat(signal, beats[i], pre_samples=window_pre, post_samples=window_post)
            if seg is not None:
                temp_beats.append(seg)
                calib_indices.append(i)

    if len(temp_beats) < n_beats:
        raise ValueError(
            f"Insufficient normal beats for calibration: found {len(temp_beats)}, expected {n_beats}."
        )

    beats_array = np.array(temp_beats)

    if method == "median":
        template = np.median(beats_array, axis=0)
    elif method == "mean":
        template = np.mean(beats_array, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Choose 'median' or 'mean'.")

    last_calib_index = max(calib_indices)
    return template, last_calib_index
