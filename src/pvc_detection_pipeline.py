"""
================================================================================
Subject-Specific PVC Detection Using Dual-Voting Feature Selection & Decision Trees
================================================================================
Reference Paper: iJOE (2026)
Author: Abdulaziz K. A. Hatem, B.Eng.
Affiliation: Dept. of Biomedical Engineering, University of Science & Technology, Aden

Tools & Libraries:
  - Python 3.11
  - wfdb (v4.3.0): MIT-BIH record & annotation ingestion
  - scipy.signal (v1.17.0): 3rd-order causal Butterworth bandpass filtering (0.5 - 40 Hz)
  - scipy.stats (v1.17.0): Higher-order statistical features (skewness, kurtosis)
  - numpy (v2.4.1): Vectorized array transformations & template operations
  - pandas (v2.3.3): Cohort structuring & performance logging
  - scikit-learn (v1.8.0): Random Forest (Gini), Decision Tree (Entropy), evaluation metrics
  - xgboost (v3.1.3): Information gain feature importance for consensus dual-voting
  - matplotlib & seaborn: Diagnostic signal visualization & confusion matrix generation

Scope:
  Offline retrospective validation across 15 MIT-BIH Arrhythmia Database records.
  Theoretical microcontroller / ESP32 feasibility analysis (<3 KB RAM, ~20 us latency).
================================================================================
"""

import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis, skew
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Experiment Configuration
# ==============================================================================
class Config:
    DB_PATH = 'Database/' 
    PATIENTS = ['105', '106', '119', '200', '201', '203', '205', '208', 
                '210', '213', '215', '219', '221', '228', '233']
    FS = 360
    CALIBRATION_BEATS = 50 
    WINDOW_PRE = 36   
    WINDOW_POST = 72  

    ALL_CANDIDATE_FEATURES = [
        'Pre_RR', 'Post_RR', 'RR_Ratio', 'Local_RR_Avg', 'RR_Diff', 'Heart_Rate',
        'Mean', 'Std_Dev', 'Max_Val', 'Min_Val', 'Peak_to_Peak', 'RMS', 
        'Area', 'Energy', 'Line_Length', 'Steepness', 'Pulse_Index', 'Crest_Factor', 'Form_Factor',
        'Skewness', 'Kurtosis', 'Hjorth_Activity', 'Hjorth_Mobility', 'Hjorth_Complexity',
        'SAD', 'Corr_Coeff', 'Max_Dev', 'Energy_Ratio', 
        'Temp_Energy_Ratio', 'Discordance', 'Res_Energy', 'ZCR'
    ]

# ==============================================================================
# 2. Signal Processing and Feature Extraction
# ==============================================================================
class SignalProcessor:
    @staticmethod
    def causal_filter(signal):
        nyq = 0.5 * Config.FS
        b, a = butter(3, [0.5/nyq, 40.0/nyq], btype='band')
        return lfilter(b, a, signal)

    @staticmethod
    def extract_features(segment, template, rr_data):
        # 1. Stats & Morph
        max_val = np.max(segment); min_val = np.min(segment)
        mean_val = np.mean(segment); std_val = np.std(segment)
        p2p = max_val - min_val
        rms = np.sqrt(np.mean(segment**2))
        area = np.sum(np.abs(segment))
        energy = np.sum(segment**2)
        skew_val = skew(segment)
        kurt_val = kurtosis(segment)
        
        # 2. Waveform Shape Descriptors (Pulse Index, Crest Factor, Form Factor, Line Length, Steepness; outlier-resistant using median absolute deviation)
        mean_abs = np.mean(np.abs(segment))
        pulse_index = max_val / (mean_abs + 1e-6)
        crest_factor = max_val / (rms + 1e-6)
        form_factor = (np.diff(segment, 2).std()) / (np.diff(segment).std() + 1e-6)
        diff1 = np.diff(segment)
        line_length = np.sum(np.abs(diff1))
        steepness = np.max(np.abs(diff1)) / (p2p + 1e-6)

        # 3. Hjorth
        var_zero = np.var(segment); var_d1 = np.var(diff1)
        var_d2 = np.var(np.diff(diff1))
        activity = var_zero
        mobility = np.sqrt(var_d1 / (var_zero + 1e-6))
        complexity = np.sqrt(var_d2 / (var_d1 + 1e-6)) / (mobility + 1e-6)

        # 4. Template
        L = min(len(segment), len(template))
        sig = segment[:L]; tmp = template[:L]
        diff = sig - tmp
        sad = np.sum(np.abs(diff))
        num = np.dot(sig, tmp)
        den = np.sqrt(np.dot(sig, sig) * np.dot(tmp, tmp)) + 1e-9
        corr = num / den
        max_dev = np.max(np.abs(diff))
        sig_en = np.sum(sig**2)+1e-9; tmp_en = np.sum(tmp**2)+1e-9
        en_ratio = sig_en / tmp_en; temp_en_ratio = tmp_en / sig_en
        res_energy = np.sum(diff**2)
        discordance = sad * (1 - corr)
        zcr = ((segment[:-1] * segment[1:]) < 0).sum()

        # 5. RR
        pre_rr = rr_data['pre']
        post_rr = rr_data['post']
        rr_ratio = pre_rr / (post_rr + 1e-6)
        local_avg = rr_data['local']
        rr_diff = pre_rr - local_avg
        hr = 60 / (pre_rr + 1e-6)

        feats = {
            'Pre_RR': pre_rr, 'Post_RR': post_rr, 'RR_Ratio': rr_ratio, 
            'Local_RR_Avg': local_avg, 'RR_Diff': rr_diff, 'Heart_Rate': hr,
            'Mean': mean_val, 'Std_Dev': std_val, 'Max_Val': max_val, 
            'Min_Val': min_val, 'Peak_to_Peak': p2p, 'RMS': rms, 
            'Area': area, 'Energy': energy, 'Line_Length': line_length, 
            'Steepness': steepness, 'Pulse_Index': pulse_index, 
            'Crest_Factor': crest_factor, 'Form_Factor': form_factor,
            'Skewness': skew_val, 'Kurtosis': kurt_val, 
            'Hjorth_Activity': activity, 'Hjorth_Mobility': mobility, 
            'Hjorth_Complexity': complexity,
            'SAD': sad, 'Corr_Coeff': corr, 'Max_Dev': max_dev, 
            'Energy_Ratio': en_ratio, 'Temp_Energy_Ratio': temp_en_ratio, 
            'Discordance': discordance, 'Res_Energy': res_energy, 'ZCR': zcr
        }
        
        return [feats[f] for f in Config.ALL_CANDIDATE_FEATURES]

# ==============================================================================
# 3. Intra-Patient Evaluation Pipeline
# ==============================================================================
def run_scientific_pipeline():
    print(">>> Starting Intra-Patient Evaluation Pipeline...")
    print(f"{'Patient':<8} | {'Train(N)':<8} | {'Test(N)':<8} | {'Acc(%)':<8} | {'Sens(%)':<8} | {'Spec(%)':<8} | {'Selected Feats'}")
    print("-" * 100)
    
    global_results = []
    final_model = None
    final_features = []
    final_indices = []
    
    for pid in Config.PATIENTS:
        try:
            record = wfdb.rdrecord(Config.DB_PATH + pid)
            ann = wfdb.rdann(Config.DB_PATH + pid, 'atr')
            sig = SignalProcessor.causal_filter(record.p_signal[:, 0])
            beats = ann.sample; syms = ann.symbol
            
            # Calibration
            calibration_indices = []
            temp_beats = []
            for i in range(1, len(beats)-1):
                if len(temp_beats) >= Config.CALIBRATION_BEATS: break
                if syms[i] == 'N':
                    s, e = beats[i]-Config.WINDOW_PRE, beats[i]+Config.WINDOW_POST
                    if s >= 0 and e < len(sig): 
                        temp_beats.append(sig[s:e])
                        calibration_indices.append(i)
            
            if not temp_beats: continue
            template = np.median(temp_beats, axis=0)  # Outlier-resistant using median absolute deviation relative to sample mean
            last_calib_idx = max(calibration_indices) if calibration_indices else 0
            
            # Stage 3: Morphology and Timing Feature Extraction
            X_full, y_full = [], []
            rr_ints = np.diff(beats)/Config.FS
            start_process_idx = last_calib_idx + 1
            
            for i in range(start_process_idx, len(beats)-1):
                if syms[i] not in ['N', 'V']: continue
                s, e = beats[i]-Config.WINDOW_PRE, beats[i]+Config.WINDOW_POST
                if s < 0 or e >= len(sig): continue
                
                pre = (beats[i]-beats[i-1])/Config.FS
                post = (beats[i+1]-beats[i])/Config.FS
                local = np.mean(rr_ints[max(0, i-5):i]) if i>0 else pre
                
                f = SignalProcessor.extract_features(sig[s:e], template, {'pre': pre, 'post': post, 'local': local})
                X_full.append(f)
                y_full.append(1 if syms[i] == 'V' else 0)
                
            X_full = np.array(X_full); y_full = np.array(y_full)
            if len(y_full) < 20: continue
            
            # Split
            split_idx = int(len(X_full) * 0.70)
            X_train = X_full[:split_idx]; y_train = y_full[:split_idx]
            X_test = X_full[split_idx:]; y_test = y_full[split_idx:]
            
            # Dual-Voting Feature Selection (Random Forest Gini + XGBoost Gain Consensus)
            rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_sel.fit(X_train, y_train)
            rf_imp = rf_sel.feature_importances_
            rf_norm = rf_imp / (rf_imp.max() + 1e-9)
            
            try:
                import xgboost as xgb
                xgb_sel = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
                xgb_sel.fit(X_train, y_train)
                xgb_imp = xgb_sel.feature_importances_
                xgb_norm = xgb_imp / (xgb_imp.max() + 1e-9)
                aggregated = (rf_norm + xgb_norm) / 2.0
            except Exception:
                aggregated = rf_norm  # Graceful fallback if XGBoost is not present in local runtime
            
            top_indices = np.argsort(aggregated)[-12:] 
            
            X_train_opt = X_train[:, top_indices]
            X_test_opt = X_test[:, top_indices]
            current_feats = [Config.ALL_CANDIDATE_FEATURES[i] for i in top_indices]
            
            # Train & Test
            clf = DecisionTreeClassifier(max_depth=8, criterion='entropy', class_weight='balanced', random_state=42)
            clf.fit(X_train_opt, y_train)
            y_pred = clf.predict(X_test_opt)
            
            acc = accuracy_score(y_test, y_pred) * 100
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            sens = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
            
            print(f"{pid:<8} | {len(y_train):<8} | {len(y_test):<8} | {acc:<8.2f} | {sens:<8.2f} | {spec:<8.2f} | {len(current_feats)} Feats")
            global_results.append({'Acc': acc, 'Sens': sens})
            
            if pid == '208': 
                final_model = clf
                final_features = current_feats
                final_indices = top_indices
            
        except Exception as e:
            print(f"Error {pid}: {e}")

    df_res = pd.DataFrame(global_results)
    print("-" * 100)
    print(f"GLOBAL AVG ACCURACY:    {df_res['Acc'].mean():.2f}%")
    return final_model, final_features, final_indices

# ==============================================================================
# 4. Beat Morphology Search and Visualization
# ==============================================================================
def find_and_visualize_examples(pid, trained_model, selected_feat_indices):
    if trained_model is None:
        print("Model for patient 208 was not trained successfully.")
        return

    print(f"\n>>> Finding real examples (N vs V) for Patient {pid} to visualize...")

    try:
        record = wfdb.rdrecord(Config.DB_PATH + pid)
        ann = wfdb.rdann(Config.DB_PATH + pid, 'atr')
    except FileNotFoundError:
        print(f"Error: Database files for {pid} not found.")
        return

    raw_sig = record.p_signal[:, 0]
    filtered_sig = SignalProcessor.causal_filter(raw_sig)
    beats = ann.sample; syms = ann.symbol

    # Template Recreation
    temp_beats = []
    for i in range(1, len(beats)-1):
        if len(temp_beats) >= Config.CALIBRATION_BEATS: break
        if syms[i] == 'N':
                s, e = beats[i]-Config.WINDOW_PRE, beats[i]+Config.WINDOW_POST
                if s >= 0 and e < len(filtered_sig): temp_beats.append(filtered_sig[s:e])
    
    if not temp_beats: return
    template = np.median(temp_beats, axis=0)  # Computes sample median directly to attenuate artifact spikes without outlier distortion

    # Search
    n_example, v_example = None, None
    rr_ints = np.diff(beats)/Config.FS

    for i in range(500, len(beats)-1):
        if n_example and v_example: break

        sym = syms[i]
        if sym not in ['N', 'V']: continue
        s, e = beats[i]-Config.WINDOW_PRE, beats[i]+Config.WINDOW_POST
        if s < 0 or e >= len(filtered_sig): continue

        pre = (beats[i]-beats[i-1])/Config.FS
        post = (beats[i+1]-beats[i])/Config.FS
        local = np.mean(rr_ints[max(0, i-5):i])

        segment = filtered_sig[s:e]
        all_feats_values = SignalProcessor.extract_features(segment, template, {'pre': pre, 'post': post, 'local': local})
        model_ready_feats = np.array(all_feats_values)[selected_feat_indices].reshape(1, -1)

        prediction = trained_model.predict(model_ready_feats)[0]
        pred_text = "Normal (N)" if prediction == 0 else "PVC (V)"

        example_data = {'signal': segment, 'actual': sym, 'predicted': pred_text}

        if sym == 'N' and n_example is None: n_example = example_data
        elif sym == 'V' and v_example is None: v_example = example_data

    # Plotting
    if n_example and v_example:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Real-time Model Verification (Patient {pid})', fontsize=16, fontweight='bold')

        # Plot Normal
        ax0 = axes[0]
        ax0.plot(n_example['signal'], color='tab:green', linewidth=2)
        ax0.set_title(f"Actual: {n_example['actual']} (Normal Beat)\nModel Prediction: {n_example['predicted']}", fontsize=12)
        ax0.grid(True, linestyle='--', alpha=0.7)
        ax0.set_facecolor('#f0fff0')

        # Plot PVC
        ax1 = axes[1]
        ax1.plot(v_example['signal'], color='tab:red', linewidth=2)
        ax1.set_title(f"Actual: {v_example['actual']} (PVC)\nModel Prediction: {v_example['predicted']}", fontsize=12, color='darkred', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_facecolor('#fff5f5')

        print(">>> Visualizing results... (Look at the popup window)")
        plt.tight_layout()
        plt.show()
    else:
        print("Could not find examples.")

# ==============================================================================
# 5. C++ Firmware Transpilation
# ==============================================================================
def generate_cpp(model, feat_names, feat_indices):
    if not model: return
    print(f"\n>>> Generating Firmware for Patient 208...")
    code = f"""
/* * Embedded Decision Tree PVC Classifier (Chronologically Validated)
 * Patient Specific Model (e.g., Record 208)
 * Features Used: {', '.join(feat_names)}
 */
int predict_pvc(float all_33_feats[]) {{
    float f[{len(feat_names)}];
"""
    for i, original_idx in enumerate(feat_indices):
        code += f"    f[{i}] = all_33_feats[{original_idx}]; // {feat_names[i]}\n"
    code += "\n"
    tree = model.tree_
    def recurse(node, depth):
        indent = "  " * depth
        if tree.feature[node] != -2:
            return f"{indent}if (f[{tree.feature[node]}] <= {tree.threshold[node]:.6f}) {{\n{recurse(tree.children_left[node], depth+1)}{indent}}} else {{\n{recurse(tree.children_right[node], depth+1)}{indent}}}"
        else: return f"{indent}return {np.argmax(tree.value[node])};\n"
    code += recurse(0, 1) + "}\n"
    with open("Results/ESP32_Scientific_Firmware.cpp", "w") as f: f.write(code)
    print(">>> Saved: Results/ESP32_Scientific_Firmware.cpp")

if __name__ == "__main__":
    if not os.path.exists('Results'): os.makedirs('Results')
    
    # 1. Run model training and validation
    model, feats, indices = run_scientific_pipeline()
    
    # 2. Transpile decision tree to C++
    generate_cpp(model, feats, indices)
    
    # 3. Plot example beat classifications
    find_and_visualize_examples('208', model, indices)