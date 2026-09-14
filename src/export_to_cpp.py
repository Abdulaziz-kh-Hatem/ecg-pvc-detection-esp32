import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
import os

# ==============================================================================
# Configuration: Top 16 selected patient records cohort, 12 features, max tree depth 10
# ==============================================================================
DATA_FILE = 'Results/Final_Features_33.csv'
RANKING_FILE = 'Results/Feature_Ranking.csv'

# Decision tree hyperparameters: max_depth=10, criterion='gini', class_weight={0: 1, 1: 5}, random_state=42
TOP_N_FEATURES = 12  # Top 12 ranked features selected for training
TREE_DEPTH = 10      # Maximum decision tree depth (max_depth=10)
PVC_WEIGHT = 5       # Loss penalty weight for PVC class (V): class_weight={0: 1, 1: 5}

# Target MIT-BIH record from the 16 evaluated patient records:
# 106, 119, 124, 200, 201, 203, 205, 208, 210, 213, 215, 219, 221, 223, 228, 233
TARGET_PATIENT = '203'

def tree_to_cpp(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Generate C++ branch conditional
            code = f"{indent}if (features.{name} <= {threshold:.6f}f) {{\n"
            code += recurse(tree_.children_left[node], depth + 1)
            code += f"{indent}}} else {{\n"
            code += recurse(tree_.children_right[node], depth + 1)
            code += f"{indent}}}\n"
            return code
        else:
            # Return leaf classification (0=Normal, 1=PVC)
            val = tree_.value[node][0]
            label = 1 if val[1] > val[0] else 0
            return f"{indent}return {label}; // Class {'PVC' if label == 1 else 'Normal'}\n"

    return recurse(0, 1)

# --- Execution ---
ranking_df = pd.read_csv(RANKING_FILE)
selected_features = ranking_df['Feature'].head(TOP_N_FEATURES).tolist()
df = pd.read_csv(DATA_FILE)

# Train patient-specific decision tree classifier
p_data = df[df['Record_ID'].astype(str) == TARGET_PATIENT]
X = p_data[selected_features]
y = p_data['Label']

clf = DecisionTreeClassifier(max_depth=TREE_DEPTH, class_weight={0: 1, 1: PVC_WEIGHT}, random_state=42)
clf.fit(X, y)

# Transpile decision tree branches to C++ conditionals
cpp_logic = tree_to_cpp(clf, selected_features)

# Write generated C++ code to disk
output_path = f"Results/Logic_Patient_{TARGET_PATIENT}.cpp"
with open(output_path, "w") as f:
    f.write(f"// --- Generated Logic for Patient: {TARGET_PATIENT} ---\n")
    f.write(f"// Sensitivity weight: {PVC_WEIGHT} | Features: {len(selected_features)}\n\n")
    f.write("int classify_beat(Features features) {\n")
    f.write(cpp_logic)
    f.write("}\n")

print(f"Exported C++ logic for patient {TARGET_PATIENT} to: {output_path}")