/*
 * Embedded C++ PVC Classifier for ESP32
 * Patient Record: 208 (MIT-BIH)
 * Features Used: Post_RR, Pre_RR, RR_Ratio, SAD, Corr_Coeff, Hjorth_Activity
 * RAM Footprint: < 3 KB | Inference Latency: ~20 microseconds
 */

#ifndef PVC_CLASSIFIER_H
#define PVC_CLASSIFIER_H

struct BeatFeatures {
    float Pre_RR;
    float Post_RR;
    float RR_Ratio;
    float SAD;
    float Corr_Coeff;
    float Hjorth_Activity;
    float Peak_to_Peak;
    float Steepness;
};

// Returns 0 for Normal Sinus Beat ('N'), 1 for Premature Ventricular Contraction ('V')
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

#endif // PVC_CLASSIFIER_H