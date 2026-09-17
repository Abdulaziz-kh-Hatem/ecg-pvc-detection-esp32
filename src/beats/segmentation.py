"""
Beat segmentation parameters and extraction utilities.
Defines asymmetric 108-sample windows (36 samples pre-R, 72 samples post-R)
at 360 Hz sampling rate, corresponding to 100 ms look-back and 200 ms look-ahead.
"""

from typing import Optional
import numpy as np

# Standard window parameters (300 ms total: 100 ms pre-R, 200 ms post-R)
PRE_R_SAMPLES: int = 36    # 100 ms at 360 Hz
POST_R_SAMPLES: int = 72   # 200 ms at 360 Hz
WINDOW_LENGTH: int = PRE_R_SAMPLES + POST_R_SAMPLES  # 108 samples


def segment_beat(
    signal: np.ndarray,
    r_peak_sample: int,
    pre_samples: int = PRE_R_SAMPLES,
    post_samples: int = POST_R_SAMPLES
) -> Optional[np.ndarray]:
    """
    Extract a single heartbeat segment centered around the annotated R-peak.

    Parameters:
    -----------
    signal : np.ndarray
        Continuous 1D filtered ECG signal.
    r_peak_sample : int
        Sample index of the R-peak.
    pre_samples : int
        Number of samples preceding R-peak (default: 36).
    post_samples : int
        Number of samples following R-peak (default: 72).

    Returns:
    --------
    segment : Optional[np.ndarray]
        1D heartbeat window of length (pre_samples + post_samples), or None if boundary exceeded.
    """
    start = r_peak_sample - pre_samples
    end = r_peak_sample + post_samples
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]
