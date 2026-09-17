"""
Unit tests for digital Butterworth bandpass filtering.
"""

import numpy as np
import pytest
from src.signal.filtering import butter_bandpass_filter


def test_causal_filter_output_shape_and_finite():
    np.random.seed(42)
    sig = np.random.randn(3600)  # 10 seconds of random noise at 360 Hz
    filtered = butter_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=3, mode="causal")
    assert len(filtered) == len(sig), "Filtered signal length mismatch"
    assert np.all(np.isfinite(filtered)), "Filtered signal contains NaN or Inf"


def test_zero_phase_filter_mode():
    sig = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 360))  # 10 Hz sine wave
    filtered = butter_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=3, mode="zero_phase")
    assert len(filtered) == len(sig)
    assert np.all(np.isfinite(filtered))


def test_dc_offset_suppression():
    # Constant DC offset + 5 Hz sine wave
    t = np.linspace(0, 5, 5 * 360)
    sig = 5.0 + np.sin(2 * np.pi * 5 * t)
    filtered = butter_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=3, mode="causal")
    # Discard transient initial 1 second (360 samples)
    steady_state = filtered[360:]
    assert abs(np.mean(steady_state)) < 0.1, f"DC offset was not removed: mean = {np.mean(steady_state)}"


def test_invalid_filter_mode_raises():
    sig = np.random.randn(100)
    with pytest.raises(ValueError, match="Unsupported filtering mode"):
        butter_bandpass_filter(sig, mode="invalid_mode")  # type: ignore
