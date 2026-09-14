# Subject-Specific Premature Ventricular Contraction Detection on ESP32

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target: ESP32](https://img.shields.io/badge/Target-ESP32%20Microcontroller-red.svg)]()
[![Dataset: MIT-BIH](https://img.shields.io/badge/Dataset-PhysioNet%20MIT--BIH-orange.svg)](https://physionet.org/content/mitdb/1.0.0/)

**Author:** Abdulaziz K. A. Hatem, B.Eng.  
**Department:** Biomedical Engineering, University of Science and Technology, Aden, Yemen  

---

## 1. Project Overview

Developed a patient-specific machine learning pipeline for Premature Ventricular Contraction (PVC) detection on resource-constrained embedded microcontrollers (ESP32). The system addresses computational limitations in Holter monitoring by shifting inference directly to the edge, operating completely offline with minimal memory overhead.

---

## 2. System Architecture

The detection pipeline consists of four sequential stages designed for edge deployment:

\\	ext
[ Raw ECG Stream (360 Hz) ]
             |
             v
[ 1. Filtering & Segmentation ] ---> 3rd-order Butterworth (0.5 - 40 Hz), 300 ms Window
             |
             v
[ 2. Patient Calibration ] ---> 50-beat patient-specific normal sinus median template
             |
             v
[ 3. Feature Extraction ] ---> 32 morphological, temporal, and Hjorth features
             |
             v
[ 4. Feature Selection ] ---> Dual-voting (Random Forest & XGBoost), top 12 features
             |
             v
[ 5. Decision Tree Classifier ] ---> Lightweight tree (Depth 1-8)
             |
             v
[ Automated C++ Code Export for ESP32 Firmware ]
\
### 2.1 Filtering and Segmentation
* Applied a 3rd-order Butterworth bandpass filter (0.5-40 Hz) to eliminate baseline wander and high-frequency artifacts.
* Extracted 300 ms beat-centric windows centered around R-peaks.

### 2.2 Subject-Specific Calibration
* Constructed a personalized reference template using the median waveform of the first 50 normal beats per patient.

![Personalized Template](docs/figures/Figure_3_Template.png)
*Figure 1: Subject-specific reference template.*

### 2.3 Feature Engineering & Selection
* Extracted 32 features, encompassing temporal metrics (Pre-RR, Post-RR), morphological characteristics, cross-correlation with the personalized template, and Hjorth complexity parameters.
* Applied a dual-voting feature selection methodology combining Random Forest and XGBoost to reduce feature dimensionality by 62.5%, isolating the 12 most discriminative predictors.

![Dual Voting Feature Selection](docs/figures/Figure_5_Hybrid_Importance.png)
*Figure 2: Feature importance rankings.*

### 2.4 Classification & Evaluation
* Implemented Decision Trees constrained to maximum depths of 1-8.
* Evaluated models utilizing a strict 70/30 chronological train-test split per patient to prevent data leakage and simulate real-world sequential monitoring.

---

## 3. Evaluation Metrics

Tested on 15 records from the MIT-BIH Arrhythmia Database.

| Metric | Performance |
| :--- | :--- |
| **Accuracy** | 99.69% |
| **Sensitivity** | 98.19% |
| **Specificity** | 99.93% |
| **Average Tree Depth** | 3.3 |

*(Full per-patient breakdown available in results/patient_results.csv)*

![Accuracy Plot](docs/figures/Figure_6_Accuracy.png)
*Figure 3: Testing accuracy across patients.*

![Confusion Matrix](docs/figures/Figure_8_Confusion_Matrix.png)
*Figure 4: Aggregate confusion matrix.*

---

## 4. Hardware Deployment (ESP32)

The trained Decision Trees are transpiled into nested C++ conditional structures via a custom Python exporter for direct ESP32 deployment.

\\cpp
// Example C++ transpilation for Patient 208
int classify_beat(Features f) {
    if (f.Post_RR <= 0.655556f) {
        if (f.Pre_RR <= 0.444444f) {
            return 1; // PVC
        } else {
            return 0; // Normal
        }
    } else {
        return 1; // PVC
    }
}
\
**Hardware Benchmarks:**
* **Memory Footprint:** <3 KB RAM.
* **Inference Time:** ~20 us per beat (excluding initial feature extraction cycles).
* **Network Requirement:** Zero (100% offline edge inference).

---

## 5. Repository Structure

\\	ext
ecg-pvc-detection-esp32/
|-- README.md                      
|-- LICENSE                        
|-- requirements.txt               
|-- environment.yml                
|-- src/
|   |-- pvc_detection_pipeline.py  # Training and testing pipeline
|   |-- export_to_cpp.py           # C++ transpilation utility
|-- results/
|   |-- patient_results.csv        
|   |-- feature_selection.json     
|-- docs/
    |-- figures/                   
    |-- firmware/
        |-- ESP32_Scientific_Firmware.cpp 
\
---

## 6. Execution

\\ash
git clone https://github.com/Abdulaziz-kh-Hatem/ecg-pvc-detection-esp32.git
cd ecg-pvc-detection-esp32
pip install -r requirements.txt
python src/pvc_detection_pipeline.py
\