"""
Leave-One-Patient-Out (LOPO) cross-validation evaluation module.
"""

from validation.lopo.pipeline import run_inter_patient_lopo_pipeline

__all__ = ["run_inter_patient_lopo_pipeline"]
