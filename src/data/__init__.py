"""
Data management module for ECG recordings from the MIT-BIH Arrhythmia Database.
Handles raw PhysioNet record ingestion, file integrity verification, and dataset extraction.
"""

from src.data.loader import load_ecg_record, validate_record_files, verify_checksums
from src.data.dataset import extract_patient_dataset

__all__ = [
    "load_ecg_record",
    "validate_record_files",
    "verify_checksums",
    "extract_patient_dataset",
]
