"""
Canonical pipeline execution module.
Provides the primary workflow for subject-specific intra-patient ECG classification.
"""

from src.pipeline.train import run_canonical_pipeline

__all__ = ["run_canonical_pipeline"]
