# Subject-Specific Premature Ventricular Contraction Detection on ESP32

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target: ESP32](https://img.shields.io/badge/Target-ESP32%20Microcontroller-red.svg)]()
[![Dataset: MIT-BIH](https://img.shields.io/badge/Dataset-PhysioNet%20MIT--BIH-orange.svg)](https://physionet.org/content/mitdb/1.0.0/)

**Author:** Abdulaziz K. A. Hatem, B.Eng.  
**Department:** Biomedical Engineering, University of Science and Technology, Aden, Yemen  

---

## 1. Introduction

Premature Ventricular Contractions (PVCs) are early, abnormal heartbeats. Doctors need to count how many PVCs happen in a day to understand a patient's heart risk. Usually, patients wear a Holter monitor to record this.

In Yemen, hospitals and clinics face many challenges, including power cuts and lack of expensive equipment. Cloud-based AI and expensive Holter monitors are not easy to use here. We need a low-cost, battery-powered device that works completely offline. 

This project solves this by using a cheap ESP32 microcontroller (which costs about $4) to detect PVCs. Because every person's heart signal looks different, the model learns the normal heartbeat of each specific patient first, making it very accurate without needing a huge neural network.

---

## 2. System Methodology

The system uses a four-step pipeline designed to run fast on small microcontrollers:

```
[ Raw ECG Stream (360 Hz) ]
             |
             v
[ 1. Filtering & Segmentation ] ---> 3rd-order Butterworth (0.5 - 40 Hz), 300 ms Window
             |
             v
[ 2. Patient Calibration ] ---> Uses the median of the first 50 normal beats to make a template
             |
             v
[ 3. Feature Selection ] ---> Calculates 32 features, selects the top 12 using Random Forest + XGBoost
             |
             v
[ 4. Decision Tree Classifier ] ---> Very small tree (Depth 1 to 8) to classify beats
             |
             v
[ Automated C++ Code Export for ESP32 Firmware ]
```

### Stage 1: Filtering and Segmentation
The raw ECG signal from the MIT-BIH Arrhythmia Database is filtered to remove baseline wander and high-frequency noise. We extract a 300 millisecond window around every R-peak (the main spike of the heartbeat).

### Stage 2: Patient Calibration (Template Matching)
Every patient has a unique ECG shape. We take the first 50 normal beats and calculate the median shape. This becomes the "normal template" for that specific patient. Later beats are compared to this template.

![Personalized Template](docs/figures/Figure_3_Template.png)
*Figure 1: Subject-specific reference template.*

### Stage 3: Feature Selection
We calculate 32 features for each heartbeat, including:
1. **Timing:** How fast the beat happened compared to the previous one (Pre_RR, Post_RR).
2. **Shape Metrics:** Maximum value, area, steepness.
3. **Hjorth Complexity:** Mathematical measures of signal change.
4. **Template Comparison:** How different the beat is from the patient's normal template.

We use Random Forest and XGBoost to find the most important features. We keep only the top 12 features to save memory on the ESP32.

![Dual Voting Feature Selection](docs/figures/Figure_5_Hybrid_Importance.png)
*Figure 2: Feature importance rankings.*

### Stage 4: Decision Tree and Testing
We use a simple Decision Tree because it is very fast and uses almost no memory.
We test the model carefully: we train on the first 70% of a patient's record and test on the last 30%. This makes sure the model works on future, unseen beats.

---

## 3. Results on 15 MIT-BIH Patients

We tested the code on 15 patient records from the MIT-BIH database. Some patients had very few PVCs, and some had many.

| Patient Record | Training Beats | Testing Beats | Accuracy (%) | Sensitivity (%) | Specificity (%) | Decision Tree Depth |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **105** | 1,759 | 754 | 100.0% | 100.0% | 100.0% | 5 |
| **Average** | **1,716** | **736** | **99.69%** | **98.19%** | **99.93%** | **3.3** |
*(Note: Full table is in the repository results CSV)*

![Accuracy Plot](docs/figures/Figure_6_Accuracy.png)
*Figure 3: Testing accuracy across patients.*

![Confusion Matrix](docs/figures/Figure_8_Confusion_Matrix.png)
*Figure 4: Confusion matrix showing very few mistakes.*

---

## 4. Running on the ESP32

To run this on the ESP32, we convert the trained Decision Tree into simple C++ `if/else` statements using a Python script.

```cpp
// Example C++ Code Generated for Patient 208
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
```

**Hardware Limits:**
- **RAM Used:** Less than 3 KB.
- **Speed:** The decision tree takes about 20 microseconds per beat. *(Note: Calculating the 12 features takes extra time, which we will test on physical hardware in the future).*
- **No Internet Needed:** Everything runs directly on the ESP32.

---

## 5. Repository Files

```text
ecg-pvc-detection-esp32/
├── README.md                      
├── LICENSE                        
├── requirements.txt               
├── environment.yml                
├── src/
│   ├── pvc_detection_pipeline.py  # Training and testing code
│   └── export_to_cpp.py           # Converts decision tree to C++
├── results/
│   ├── patient_results.csv        
│   └── feature_selection.json     
└── docs/
    ├── figures/                   
    └── firmware/
        └── ESP32_Scientific_Firmware.cpp 
```

---

## 6. How to Run

```bash
git clone https://github.com/Abdulaziz-kh-Hatem/ecg-pvc-detection-esp32.git
cd ecg-pvc-detection-esp32
pip install -r requirements.txt
python src/pvc_detection_pipeline.py
```

---

## 7. Limitations & Future Work

- **Manual Labeling Need:** The system needs the first 50 beats to be normal to create the template. Right now, a doctor has to check these first 50 beats. In the future, we want to automate this step.
- **Full Hardware Test:** We have tested the decision tree speed, but we still need to test the total time required to extract features on the actual ESP32 chip.
- **Real Patients:** We tested using the public MIT-BIH database. The next step is to test it on real patients in a clinic.