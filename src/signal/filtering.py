"""
Signal preprocessing and digital filtering module for ECG recordings.
Implements Butterworth bandpass filtering matching both causal microcontroller constraints
and zero-phase visualization modes.
"""

from typing import Literal
import numpy as np
from scipy.signal import butter, lfilter, filtfilt


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = 360,
    order: int = 3,
    mode: Literal["causal", "zero_phase"] = "causal"
) -> np.ndarray:
    """
    Apply a digital Butterworth bandpass filter to eliminate baseline wander and EMG noise.

    Parameters:
    -----------
    data : np.ndarray
        1D raw ECG signal.
    lowcut : float
        Lower cutoff frequency in Hz (default: 0.5 Hz, suppresses respiratory baseline drift).
    highcut : float
        Upper cutoff frequency in Hz (default: 40.0 Hz, suppresses powerline and EMG interference).
    fs : int
        Sampling frequency in Hz (default: 360 Hz).
    order : int
        Filter order (default: 3).
    mode : {'causal', 'zero_phase'}
        - 'causal': Uses `scipy.signal.lfilter` (IIR) matching real-time microcontroller firmware.
        - 'zero_phase': Uses `scipy.signal.filtfilt` (bidirectional) for non-distorted offline plots.

    Returns:
    --------
    filtered_signal : np.ndarray
        Filtered ECG signal with identical length.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")

    if mode == "causal":
        return lfilter(b, a, data)
    elif mode == "zero_phase":
        return filtfilt(b, a, data)
    else:
        raise ValueError(f"Unsupported filtering mode: {mode}. Choose 'causal' or 'zero_phase'.")
