"""
Data loading and verification utilities for the MIT-BIH Arrhythmia Database.
Handles PhysioNet record ingestion, annotation parsing, and checksum validation.
"""

import os
import hashlib
from typing import Tuple, Dict
import numpy as np
import wfdb


def validate_record_files(record_id: str, data_dir: str = "data/raw/") -> bool:
    """Check whether required record files (.dat, .hea, .atr) exist on disk."""
    base_path = os.path.join(data_dir, str(record_id))
    for ext in [".dat", ".hea", ".atr"]:
        if not os.path.exists(base_path + ext):
            return False
    return True


def load_ecg_record(
    record_id: str,
    data_dir: str = "data/raw/",
    channel: int = 0
) -> Tuple[np.ndarray, wfdb.Annotation, int]:
    """
    Load ECG signal, reference annotations, and sampling frequency for a patient record.

    Parameters:
    -----------
    record_id : str
        PhysioNet record identifier (e.g., '105', '208').
    data_dir : str
        Directory containing raw database files.
    channel : int
        ECG channel index (0 for MLII).

    Returns:
    --------
    signal : np.ndarray
        1D physical signal amplitude in millivolts.
    annotation : wfdb.Annotation
        Cardiologist annotations object containing beat sample locations and symbols.
    fs : int
        Sampling frequency in Hertz (typically 360 Hz).
    """
    base_path = os.path.join(data_dir, str(record_id))
    if not os.path.exists(base_path + ".hea"):
        raise FileNotFoundError(f"ECG header not found: {base_path}.hea")

    record = wfdb.rdrecord(base_path)
    annotation = wfdb.rdann(base_path, "atr")
    signal = record.p_signal[:, channel]
    fs = record.fs
    return signal, annotation, fs


def verify_checksums(data_dir: str = "data/raw/") -> Dict[str, bool]:
    """
    Verify SHA-256 hashes of all local record files against the official PhysioNet manifest.

    Returns:
    --------
    results : Dict[str, bool]
        Mapping from filename to verification status (True if hash matches).
    """
    manifest_path = os.path.join(data_dir, "SHA256SUMS.txt")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Checksum manifest missing: {manifest_path}")

    expected_hashes: Dict[str, str] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                expected_hashes[parts[1]] = parts[0]

    verification: Dict[str, bool] = {}
    for filename, expected_hash in expected_hashes.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as bf:
                actual_hash = hashlib.sha256(bf.read()).hexdigest()
            verification[filename] = (actual_hash == expected_hash)

    return verification
