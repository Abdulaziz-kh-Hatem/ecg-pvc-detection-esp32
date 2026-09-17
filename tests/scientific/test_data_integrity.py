"""
Scientific data integrity and official PhysioNet checksum verification tests.
"""

from src.data.loader import (
    validate_record_files,
    verify_checksums
)

RECORDS = ["105", "106", "119", "200", "201", "203", "205", "208",
           "210", "213", "215", "219", "221", "228", "233"]


def test_all_15_records_exist_in_raw():
    for r in RECORDS:
        assert validate_record_files(r, data_dir="data/raw/"), f"Record {r} missing files in data/raw/"


def test_official_sha256_checksums():
    results = verify_checksums(data_dir="data/raw/")
    assert len(results) > 0, "No files verified by checksum manifest"
    for r in RECORDS:
        for ext in [".dat", ".hea", ".atr"]:
            fn = r + ext
            assert results.get(fn) is True, f"Checksum verification failed for {fn}"
