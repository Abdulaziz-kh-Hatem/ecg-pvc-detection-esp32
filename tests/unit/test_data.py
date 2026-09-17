"""
Unit tests for data loader utilities.
"""

import pytest
from src.data.loader import load_ecg_record, validate_record_files


def test_load_ecg_record_structure():
    signal, annotation, fs = load_ecg_record("105", data_dir="data/raw/", channel=0)
    assert fs == 360, f"Expected 360 Hz sampling rate, got {fs}"
    assert signal.ndim == 1, "Expected 1D signal array"
    assert len(signal) > 600000, f"Record 105 shorter than expected: {len(signal)} samples"
    assert len(annotation.sample) > 2000, "Expected > 2000 annotated beats"
    assert len(annotation.symbol) == len(annotation.sample), "Annotation samples and symbols length mismatch"


def test_validate_record_files_valid_and_invalid():
    assert validate_record_files("105", data_dir="data/raw/") is True
    assert validate_record_files("non_existent_record", data_dir="data/raw/") is False


def test_missing_header_raises_filenotfound():
    with pytest.raises(FileNotFoundError, match="ECG header not found"):
        load_ecg_record("99999", data_dir="data/raw/")
