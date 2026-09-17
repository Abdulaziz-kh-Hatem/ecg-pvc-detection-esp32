"""
Beat segmentation and subject-specific template extraction module.
"""

from src.beats.template import build_subject_template
from src.beats.segmentation import segment_beat, PRE_R_SAMPLES, POST_R_SAMPLES, WINDOW_LENGTH

__all__ = [
    "build_subject_template",
    "segment_beat",
    "PRE_R_SAMPLES",
    "POST_R_SAMPLES",
    "WINDOW_LENGTH",
]
