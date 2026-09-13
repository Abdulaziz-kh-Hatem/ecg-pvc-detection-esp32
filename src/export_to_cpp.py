import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
import os

# ==============================================================================
# إعدادات التصدير النهائي (Elite 16 - 12 Features - Depth 10)
# ==============================================================================
DATA_FILE = 'Results/Final_Features_33.csv'
RANKING_FILE = 'Results/Feature_Ranking.csv'
TOP_N_FEATURES = 12
TREE_DEPTH = 10
PVC_WEIGHT = 5 # لضمان أعلى حساسية

# اختر المريض من القائمة (106, 119, 124, 200, 201, 203, 205, 208, 210, 213, 215, 219, 221, 223, 228, 233)
# في ملف export_to_cpp.py
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
            # توليد شرط C++
            code = f"{indent}if (features.{name} <= {threshold:.6f}f) {{\n"
            code += recurse(tree_.children_left[node], depth + 1)
            code += f"{indent}}} else {{\n"
            code += recurse(tree_.children_right[node], depth + 1)
            code += f"{indent}}}\n"
            return code
        else:
            # إرجاع النتيجة (0=Normal, 1=PVC)
            val = tree_.value[node][0]
            label = 1 if val[1] > val[0] else 0
            return f"{indent}return {label}; // Class {'PVC' if label == 1 else 'Normal'}\n"

    return recurse(0, 1)

# --- بدء التنفيذ ---
ranking_df = pd.read_csv(RANKING_FILE)
selected_features = ranking_df['Feature'].head(TOP_N_FEATURES).tolist()
df = pd.read_csv(DATA_FILE)

# تدريب النموذج لهذا المريض تحديداً
p_data = df[df['Record_ID'].astype(str) == TARGET_PATIENT]
X = p_data[selected_features]
y = p_data['Label']

clf = DecisionTreeClassifier(max_depth=TREE_DEPTH, class_weight={0: 1, 1: PVC_WEIGHT}, random_state=42)
clf.fit(X, y)

# تحويل الشجرة إلى C++
cpp_logic = tree_to_cpp(clf, selected_features)

# حفظ النتيجة
output_path = f"Results/Logic_Patient_{TARGET_PATIENT}.cpp"
with open(output_path, "w") as f:
    f.write(f"// --- Generated Logic for Patient: {TARGET_PATIENT} ---\n")
    f.write(f"// Sensitivity weight: {PVC_WEIGHT} | Features: {len(selected_features)}\n\n")
    f.write("int classify_beat(Features features) {\n")
    f.write(cpp_logic)
    f.write("}\n")

print(f"✅ تم تصدير الكود بنجاح للمريض {TARGET_PATIENT} في المسار: {output_path}")