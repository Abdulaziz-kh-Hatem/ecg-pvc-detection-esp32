"""
Unit tests for template construction and beat segmentation.
"""

import numpy as np
import pytest
from src.beats.template import build_subject_template
from src.beats.segmentation import segment_beat, PRE_R_SAMPLES, POST_R_SAMPLES, WINDOW_LENGTH


def test_template_building():
    np.random.seed(42)
    sig = np.random.randn(50000)
    # Generate 60 artificial normal beat peaks separated by 360 samples
    beats = np.arange(500, 500 + 60 * 360, 360)
    symbols = np.array(["N"] * 60)

    template, last_idx = build_subject_template(
        sig, beats, symbols, n_beats=50, window_pre=36, window_post=72, method="median"
    )
    assert template.shape == (108,), f"Expected 108 template samples, got {template.shape}"
    assert last_idx == 50, f"Expected last calib index 50, got {last_idx}"
    assert np.all(np.isfinite(template)), "Template contains non-finite values"


def test_insufficient_normal_beats_raises_error():
    sig = np.random.randn(10000)
    # Only 30 normal beats when 50 are required
    beats = np.arange(500, 500 + 30 * 360, 360)
    symbols = np.array(["N"] * 30)

    with pytest.raises(ValueError, match="Insufficient normal beats"):
        build_subject_template(sig, beats, symbols, n_beats=50)


def test_template_mean_method():
    sig = np.random.randn(50000)
    beats = np.arange(500, 500 + 60 * 360, 360)
    symbols = np.array(["N"] * 60)

    template, last_idx = build_subject_template(
        sig, beats, symbols, n_beats=50, method="mean"
    )
    assert template.shape == (108,)
    assert last_idx == 50


def test_segment_beat_bounds():
    sig = np.zeros(1000)
    # Valid beat
    seg = segment_beat(sig, 500, pre_samples=36, post_samples=72)
    assert seg is not None
    assert len(seg) == WINDOW_LENGTH == 108

    # Near left edge (underflow)
    seg_left = segment_beat(sig, 10, pre_samples=36, post_samples=72)
    assert seg_left is None

    # Near right edge (overflow)
    seg_right = segment_beat(sig, 980, pre_samples=36, post_samples=72)
    assert seg_right is None
