"""
ECG Feature Extraction Engine.
Extracts 32 physiological and mathematical descriptors per heartbeat across four domains:
1. Temporal (6 features): Pre_RR, Post_RR, RR_Ratio, Local_RR_Avg, RR_Diff, Heart_Rate
2. Morphological (13 features): Mean, Std_Dev, Max_Val, Min_Val, Peak_to_Peak, RMS,
   Area, Energy, Line_Length, Steepness, Pulse_Index, Crest_Factor, Form_Factor
3. Statistical (6 features): Skewness, Kurtosis, Hjorth_Activity, Hjorth_Mobility,
   Hjorth_Complexity, ZCR
4. Template-Matching (7 features): SAD, Corr_Coeff, Max_Dev, Energy_Ratio,
   Temp_Energy_Ratio, Discordance, Res_Energy
"""

from typing import Dict, List
import numpy as np
from scipy.stats import skew, kurtosis

ALL_32_FEATURE_NAMES: List[str] = [
    "Pre_RR", "Post_RR", "RR_Ratio", "Local_RR_Avg", "RR_Diff", "Heart_Rate",
    "Mean", "Std_Dev", "Max_Val", "Min_Val", "Peak_to_Peak", "RMS",
    "Area", "Energy", "Line_Length", "Steepness", "Pulse_Index",
    "Crest_Factor", "Form_Factor",
    "Skewness", "Kurtosis", "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "SAD", "Corr_Coeff", "Max_Dev", "Energy_Ratio",
    "Temp_Energy_Ratio", "Discordance", "Res_Energy", "ZCR"
]


def extract_all_32_features(
    segment: np.ndarray,
    template: np.ndarray,
    pre_rr: float,
    post_rr: float,
    local_rr_avg: float
) -> Dict[str, float]:
    """
    Extract the complete 32-dimensional feature dictionary for a single heartbeat segment.

    Parameters:
    -----------
    segment : np.ndarray
        1D heartbeat window (108 samples).
    template : np.ndarray
        1D subject-specific normal reference template (108 samples).
    pre_rr : float
        Time interval before current R-peak in seconds.
    post_rr : float
        Time interval after current R-peak in seconds (look-ahead buffer).
    local_rr_avg : float
        Mean of preceding 5 RR intervals in seconds.

    Returns:
    --------
    features : Dict[str, float]
        Dictionary mapping feature name to floating-point value.
    """
    # 1. Temporal Features (6)
    rr_ratio = pre_rr / (post_rr + 1e-6)
    rr_diff = pre_rr - local_rr_avg
    heart_rate = 60.0 / (pre_rr + 1e-6)

    # 2. Morphological Features (13)
    max_val = float(np.max(segment))
    min_val = float(np.min(segment))
    mean_val = float(np.mean(segment))
    std_val = float(np.std(segment))
    p2p = max_val - min_val
    rms = float(np.sqrt(np.mean(segment**2)))
    area = float(np.sum(np.abs(segment)))
    energy = float(np.sum(segment**2))

    mean_abs = float(np.mean(np.abs(segment)))
    pulse_index = max_val / (mean_abs + 1e-6)
    crest_factor = max_val / (rms + 1e-6)

    diff1 = np.diff(segment)
    diff2 = np.diff(diff1)
    form_factor = float((np.std(diff2) + 1e-6) / (np.std(diff1) + 1e-6))
    line_length = float(np.sum(np.abs(diff1)))
    steepness = float(np.max(np.abs(diff1)) / (p2p + 1e-6))

    # 3. Statistical Features (6)
    raw_skew = float(skew(segment))
    raw_kurt = float(kurtosis(segment))
    skew_val = 0.0 if np.isnan(raw_skew) else raw_skew
    kurt_val = 0.0 if np.isnan(raw_kurt) else raw_kurt

    var_zero = float(np.var(segment))
    var_d1 = float(np.var(diff1))
    var_d2 = float(np.var(diff2))

    hjorth_activity = var_zero
    hjorth_mobility = float(np.sqrt(var_d1 / (var_zero + 1e-6)))
    hjorth_complexity = float(np.sqrt(var_d2 / (var_d1 + 1e-6)) / (hjorth_mobility + 1e-6))

    zcr = float(np.sum((segment[:-1] * segment[1:]) < 0))

    # 4. Template-Matching Features (7)
    min_len = min(len(segment), len(template))
    seg_aligned = segment[:min_len]
    tmp_aligned = template[:min_len]
    diff_vec = seg_aligned - tmp_aligned

    sad = float(np.sum(np.abs(diff_vec)))

    num = float(np.dot(seg_aligned, tmp_aligned))
    den = float(np.sqrt(np.dot(seg_aligned, seg_aligned) * np.dot(tmp_aligned, tmp_aligned))) + 1e-9
    corr_coeff = float(num / den)

    max_dev = float(np.max(np.abs(diff_vec)))

    seg_energy = float(np.sum(seg_aligned**2)) + 1e-9
    tmp_energy = float(np.sum(tmp_aligned**2)) + 1e-9
    energy_ratio = float(seg_energy / tmp_energy)
    temp_energy_ratio = float(tmp_energy / seg_energy)

    discordance = float(sad * (1.0 - corr_coeff))
    res_energy = float(np.sum(diff_vec**2))

    return {
        "Pre_RR": pre_rr, "Post_RR": post_rr, "RR_Ratio": rr_ratio,
        "Local_RR_Avg": local_rr_avg, "RR_Diff": rr_diff, "Heart_Rate": heart_rate,
        "Mean": mean_val, "Std_Dev": std_val, "Max_Val": max_val,
        "Min_Val": min_val, "Peak_to_Peak": p2p, "RMS": rms,
        "Area": area, "Energy": energy, "Line_Length": line_length,
        "Steepness": steepness, "Pulse_Index": pulse_index,
        "Crest_Factor": crest_factor, "Form_Factor": form_factor,
        "Skewness": skew_val, "Kurtosis": kurt_val,
        "Hjorth_Activity": hjorth_activity, "Hjorth_Mobility": hjorth_mobility,
        "Hjorth_Complexity": hjorth_complexity,
        "SAD": sad, "Corr_Coeff": corr_coeff, "Max_Dev": max_dev,
        "Energy_Ratio": energy_ratio, "Temp_Energy_Ratio": temp_energy_ratio,
        "Discordance": discordance, "Res_Energy": res_energy, "ZCR": zcr
    }
