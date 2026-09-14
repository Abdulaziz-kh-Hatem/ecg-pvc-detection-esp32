# Subject-Specific Premature Ventricular Contraction Detection Using Dual-Voting Feature Selection and Lightweight Decision Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Paper: iJOE 2026](https://img.shields.io/badge/Journal-iJOE%202026-purple.svg)]()
[![Target Architecture: Microcontroller (Theoretical)](https://img.shields.io/badge/Target%20Architecture-Microcontroller%20(Theoretical)-lightgrey.svg)]()
[![Dataset: PhysioNet MIT-BIH](https://img.shields.io/badge/Dataset-PhysioNet%20MIT--BIH-orange.svg)](https://physionet.org/content/mitdb/1.0.0/)

**Author:** Abdulaziz K. A. Hatem, B.Eng.  
**Affiliation:** Department of Biomedical Engineering, University of Science and Technology (UST), Aden, Yemen  
**Reference Paper:** *"Subject-Specific Premature Ventricular Contraction Detection Using Dual-Voting Feature Selection and Lightweight Decision Trees"* (iJOE, 2026)

---

## 1. Project Overview & Scope

This repository provides the official Python implementation and algorithmic pipeline for the research paper submitted to *iJOE* (2026). The project develops an offline, subject-specific machine learning framework for detecting **Premature Ventricular Contractions (PVCs)** from single-lead electrocardiogram (ECG) signals, specifically designed to address cardiac arrhythmia screening challenges in low-resource and infrastructure-limited healthcare settings.

> **Important Scope Note Regarding Hardware:**  
> All signal processing, feature extraction, dual-voting feature selection, model training, and performance evaluations in this study were conducted **offline in Python** using retrospective data from the MIT-BIH Arrhythmia Database.  
> Microcontroller deployment (such as on the ESP32) was analyzed as a **theoretical feasibility and complexity study** (evaluating memory bounds of ~3 KB and microsecond execution latency). Physical on-hardware deployment and live patient acquisition were **not** performed in this study and are designated as future work in the paper's three-phase development roadmap. An automated C++ transpilation script (`src/export_to_cpp.py`) is provided as a proof-of-concept for generating microcontroller-ready decision logic.

---

## 2. Software Tools & Python Libraries

The entire experimental framework was developed using **Python 3.11**. The following standard scientific and machine learning libraries are utilized:

| Library / Tool | Version | Role in Pipeline |
| :--- | :--- | :--- |
| **`wfdb`** | `4.3.0` | Native reading of PhysioNet MIT-BIH Arrhythmia Database signals (`.dat`), headers (`.hea`), and cardiologist-verified beat annotations (`.atr`). |
| **`scipy.signal`** | `1.17.0` | Digital filtering: 3rd-order causal Butterworth bandpass filter (0.5–40.0 Hz) using `butter` and `lfilter` to eliminate baseline wander and high-frequency noise without look-ahead distortion. |
| **`scipy.stats`** | `1.17.0` | Higher-order statistical feature calculation (sample skewness and kurtosis for waveform asymmetry and peakedness). |
| **`numpy`** | `2.4.1` | Vectorized mathematical operations, signal difference vectors, energy ratios, template dot products, and sample median calculations. |
| **`pandas`** | `2.3.3` | Dataset structuring, chronological 70/30 train/test index management, and per-patient performance metric logging. |
| **`scikit-learn`** | `1.8.0` | **Feature Selection:** `RandomForestClassifier` (100 estimators, Gini impurity ranking).<br>**Classification:** `DecisionTreeClassifier` (CART algorithm, entropy criterion, `max_depth=8`, `class_weight='balanced'`).<br>**Metrics:** `accuracy_score`, `confusion_matrix`, `roc_auc_score`. |
| **`xgboost`** | `3.1.3` | `XGBClassifier` (100 estimators, max_depth=6, learning_rate=0.1) for Information Gain importance ranking in the consensus Dual-Voting mechanism. |
| **`matplotlib` & `seaborn`** | `3.10.8` / `0.13.2` | Generation of scientific figures: raw vs filtered ECG plots, subject-specific templates, feature importance rankings, confusion matrices, and cohort distributions. |

---

## 3. System Architecture & Methodology

The proposed pipeline operates in four sequential stages:

```text
[ Raw ECG Record (MIT-BIH, 360 Hz) ]
                 |
                 v
[ Stage 1: Causal Filtering & Segmentation ]
  • 3rd-order Butterworth bandpass (0.5 - 40 Hz) via SciPy
  • Asymmetric 300 ms beat-centered window (100 ms pre-R, 200 ms post-R = 108 samples)
                 |
                 v
[ Stage 2: Subject-Specific Calibration ]
  • First 50 consecutive normal sinus beats ('N')
  • Sample-by-sample median template (outlier-resistant)
  • Calibration beats strictly isolated from training/testing sets
                 |
                 v
[ Stage 3: Feature Engineering & Dual-Voting Selection ]
  • 32 candidate features extracted (temporal, morphological, statistical, template)
  • Dual-Voting consensus: Random Forest (Gini) + XGBoost (Gain) on 70% train split
  • Dimensionality reduced by 62.5% -> Top 12 personalized features selected
                 |
                 v
[ Stage 4: Lightweight Decision Tree Classification ]
  • CART Decision Tree (Entropy, max_depth <= 8, class_weight='balanced')
  • Evaluated on chronological 30% test split (near real-time look-ahead buffer)
                 |
                 v
[ Automated Transpilation to Standalone C++ Logic ]
  • Exported nested if/else logic for future microcontroller firmware
```

### 3.1 Signal Preprocessing & Segmentation
* **Filtering:** Raw single-channel ECG is filtered using a 3rd-order causal Butterworth bandpass filter (0.5 - 40.0 Hz). The 0.5 Hz lower cutoff removes respiration-induced baseline wander; the 40.0 Hz upper cutoff attenuates powerline interference and electromyographic (EMG) muscle artifact.
* **Segmentation:** Each beat is segmented using annotated R-peaks into an asymmetric 300 ms window (108 samples at 360 Hz): 100 ms before the R-peak and 200 ms after. This captures the complete QRS complex, ST segment, and T wave.

### 3.2 Subject-Specific Calibration
* For each patient, a baseline reference template is formed from the **first 50 consecutive normal beats** (AAMI class 'N').
* A **sample-by-sample median** is computed rather than an arithmetic mean to resist ectopic outliers.
* These initial 50 beats are permanently excluded from both training and test sets to prevent temporal data leakage.

### 3.3 Feature Extraction (32 Candidate Descriptors)
The pipeline computes 32 candidate mathematical descriptors across four functional categories:

| Category | Count | Features | Description |
| :--- | :---: | :--- | :--- |
| **Temporal** | 6 | `Pre_RR`, `Post_RR`, `RR_Ratio`, `Local_RR_Avg`, `RR_Diff`, `Heart_Rate` | Inter-beat timing intervals, compensatory pauses, and local rhythm dynamics. |
| **Morphological** | 13 | `Mean`, `Std_Dev`, `Max_Val`, `Min_Val`, `Peak_to_Peak`, `RMS`, `Area`, `Energy`, `Line_Length`, `Steepness`, `Pulse_Index`, `Crest_Factor`, `Form_Factor` | Waveform amplitude, QRS width, area, signal complexity, and derivative ratios. |
| **Statistical** | 6 | `Skewness`, `Kurtosis`, `Hjorth_Activity`, `Hjorth_Mobility`, `Hjorth_Complexity`, `ZCR` | Higher-order statistical moments, Hjorth spectral parameters, and zero-crossing rate. |
| **Template-Matching** | 7 | `SAD`, `Corr_Coeff`, `Max_Dev`, `Energy_Ratio`, `Temp_Energy_Ratio`, `Discordance`, `Res_Energy` | Deviation from patient's calibrated normal template, Pearson correlation, and residual energy. |

### 3.4 Dual-Voting Feature Selection
To avoid overfitting and select robust features, a consensus dual-voting scheme is executed strictly on the training partition (70%):
1. **Random Forest (100 trees):** Evaluates feature importance via mean decrease in Gini impurity.
2. **XGBoost (100 estimators):** Evaluates feature importance via average information gain across splits.
3. **Consensus Aggregation:** Both importance vectors are normalized to [0, 1] and averaged element-wise:
   $$\text{Score}_i = \frac{1}{2} \left( \frac{\text{RF}_i}{\max(\text{RF})} + \frac{\text{XGB}_i}{\max(\text{XGB})} \right)$$
4. The **top 12 features** with the highest aggregated scores are selected per patient, reducing feature space by **62.5%**.

### 3.5 Classification Model
* **Algorithm:** Decision Tree using CART with **Entropy** criterion and balanced class weights to address severe class imbalance (PVC prevalence: 1.6% to 38.5%).
* **Depth Constraint:** Maximum tree depth is constrained to $\le 8$ levels to guarantee bounded computational complexity.

---

## 4. Experimental Results

The framework was evaluated on **15 selected patient records** from the MIT-BIH Arrhythmia Database, totaling 36,701 analyzed heartbeats under a strict **70/30 chronological train-test split**.

### 4.1 Cohort Performance Summary (Table 3 from Paper)

| Patient Record | Training Beats (70%) | Testing Beats (30%) | Accuracy (%) | Sensitivity (%) | Specificity (%) | Decision Tree Depth |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **105** | 1,754 | 752 | 100.00% | 100.00% | 100.00% | 5 |
| **106** | 1,380 | 592 | 100.00% | 100.00% | 100.00% | 2 |
| **119** | 1,338 | 574 | 100.00% | 100.00% | 100.00% | 2 |
| **200** | 1,737 | 745 | 98.80% | 97.04% | 99.79% | 6 |
| **201** | 1,242 | 533 | 100.00% | 100.00% | 100.00% | 1 |
| **203** | 2,036 | 874 | 98.52% | 93.33% | 99.22% | 8 |
| **205** | 1,808 | 776 | 100.00% | 100.00% | 100.00% | 1 |
| **208** | 1,749 | 751 | 99.60% | 98.83% | 100.00% | 2 |
| **210** | 1,788 | 767 | 100.00% | 100.00% | 100.00% | 6 |
| **213** | 1,960 | 841 | 99.76% | 96.88% | 100.00% | 3 |
| **215** | 2,305 | 989 | 99.80% | 96.15% | 100.00% | 2 |
| **219** | 1,461 | 627 | 99.84% | 95.24% | 100.00% | 3 |
| **221** | 1,653 | 709 | 100.00% | 100.00% | 100.00% | 1 |
| **228** | 1,385 | 594 | 99.33% | 96.58% | 100.00% | 3 |
| **233** | 2,086 | 895 | 99.67% | 98.86% | 100.00% | 4 |
| **Mean ± SD** | — | — | **99.69% ± 0.47%** | **98.19% ± 2.17%** | **99.93% ± 0.20%** | **3.3 ± 2.0** |

### 4.2 Key Performance Insights
* **High Specificity (99.93%):** 13 out of 15 patients achieved a perfect 100% specificity (zero false positive alarms), critical for preventing alarm fatigue in long-term cardiac monitoring.
* **Robust Sensitivity (98.19%):** 14 out of 15 patients exceeded the 95% clinical sensitivity benchmark. Patient 203 recorded 93.33% due to complex multiform/polymorphic PVCs that deviated from a single template.
* **Compact Model Depth:** Average tree depth across the cohort was only **3.3 levels**. Patients with distinct compensatory pauses (e.g., Records 201, 205, 221) achieved 100% classification with single-split (depth-1) decision stumps.

---

## 5. Theoretical Microcontroller Compatibility (ESP32 Analysis)

While the empirical evaluation in this study was conducted offline in Python, the mathematical compactness of the resulting decision trees enables a rigorous theoretical analysis of microcontroller compatibility:

* **RAM Overhead:** Storing a personalized decision tree of depth 1–8 requires **< 3 KB of memory**, easily fitting into the internal SRAM of standard microcontrollers (e.g., 520 KB on the ESP32).
* **Execution Latency:** On a 240 MHz dual-core Xtensa LX6 microcontroller, evaluating a depth-1 to depth-8 decision tree requires **< 1 µs** (with peak branch traversal estimated at **~20 µs**), leaving substantial processor headroom for system sleep and telemetry.
* **Decision Timing Buffer:** Temporal features (`Post_RR`) require registering the subsequent R-peak. This introduces a physiological buffer delay of approximately 500 - 1200 ms (typically ~600 ms at resting heart rate), which represents a natural cardiac event buffer rather than computational latency.
* **Estimated Hardware Bill of Materials (BOM):** A theoretical standalone telemetry node (ESP32 MCU + AD8232 AFE sensor + LiPo battery) is projected at **$8 – $13**, significantly lowering capital costs compared to commercial Holter devices.

### Automated C++ Transpilation
To facilitate future firmware porting, `src/export_to_cpp.py` automatically converts trained Scikit-learn decision tree nodes into nested C++ `if/else` structures:

```cpp
// Sample transpiled C++ decision logic (Patient 208)
inline int classify_pvc_beat(const BeatFeatures& f) {
    if (f.Post_RR <= 0.655556f) {
        if (f.Pre_RR <= 0.444444f) {
            return 1; // Ectopic Premature Beat (PVC)
        } else {
            return 0; // Normal Beat
        }
    } else {
        if (f.Corr_Coeff <= 0.825000f) {
            return 1; // Morphology Deviation (PVC)
        } else {
            return 0; // Normal Beat with pause
        }
    }
}
```

---

## 6. Limitations & Future Roadmap

As detailed in Section 5 of the reference paper, four practical engineering limitations remain to be addressed in future work:
1. **Unsupervised "Cold Start":** Current calibration relies on gold-standard cardiologist annotations to identify the initial 50 normal beats. Future work will introduce a 30-second physician-assisted or automated signal quality check.
2. **Polymorphic PVCs:** Multi-shape ectopic beats (observed in Patient 203) require multi-template clustering (e.g., K-means) during initialization.
3. **Automated R-Peak Detection:** Offline evaluation used reference annotations; embedded deployment will integrate an on-chip real-time Pan-Tompkins or wavelet detector.

### Three-Phase Roadmap:
* **Phase 1 (Firmware Porting):** Porting feature extraction and decision tree traversal to optimized fixed-point C/C++ firmware.
* **Phase 2 (Lab Bench Testing):** Stress-testing physical hardware with synthesized ECG signal generators to validate projected battery life (5–7 days).
* **Phase 3 (Clinical Pilot in Yemen):** Validating store-and-forward telemetry in collaboration with local healthcare centers.

---

## 7. Repository Structure

```text
ecg-pvc-detection-esp32/
|-- README.md                      # Comprehensive academic documentation
|-- LICENSE                        # MIT License
|-- requirements.txt               # Python package dependencies
|-- environment.yml                # Conda environment definition
|-- src/
|   |-- pvc_detection_pipeline.py  # Complete Python training & evaluation pipeline
|   |-- export_to_cpp.py           # Automated C++ decision tree transpiler
|-- results/
|   |-- patient_results.csv        # Detailed per-patient classification metrics
|   |-- feature_selection.json     # Selected 12-feature subsets per patient
|-- docs/
|   |-- figures/                   # Diagnostic plots, templates, confusion matrices
|-- firmware/
|   |-- ESP32_Scientific_Firmware.cpp  # Sample transpiled C++ header (Patient 208)
```

---

## 8. Installation & Usage

### 8.1 Setup Python Environment

```bash
git clone https://github.com/Abdulaziz-kh-Hatem/ecg-pvc-detection-esp32.git
cd ecg-pvc-detection-esp32
pip install -r requirements.txt
```

### 8.2 Run the Scientific Pipeline

```bash
python src/pvc_detection_pipeline.py
```

### 8.3 Transpile Decision Tree to C++

```bash
python src/export_to_cpp.py
```

---

## 9. Citation

If you reference this work or utilize the pipeline, please cite:

```bibtex
@article{hatem2026pvc,
  title={Subject-Specific Premature Ventricular Contraction Detection Using Dual-Voting Feature Selection and Lightweight Decision Trees},
  author={Hatem, Abdulaziz K. A. and AL-Audi, Nasr Kaid Ali},
  journal={International Journal of Online and Biomedical Engineering (iJOE)},
  year={2026},
  note={Under Review}
}
```