"""
Integration test verifying the end-to-end execution of the canonical pipeline.
"""

import os
import yaml
import tempfile
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.pipeline.train import run_canonical_pipeline


def test_canonical_pipeline_single_patient_run():
    """Run canonical pipeline on Patient 208 using an isolated temporary directory to verify integration."""
    config_path = "configs/canonical_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        # Test with single patient 208 in isolated environment
        cfg["dataset"]["patients"] = ["208"]
        cfg["paths"]["results_dir"] = os.path.join(tmp_dir, "results")
        cfg["paths"]["models_dir"] = os.path.join(tmp_dir, "models")
        cfg["paths"]["logs_dir"] = os.path.join(tmp_dir, "logs")

        temp_config = os.path.join(tmp_dir, "test_config.yaml")
        with open(temp_config, "w", encoding="utf-8") as tf:
            yaml.dump(cfg, tf)

        df_res, feats, p208_cm, p208_model = run_canonical_pipeline(temp_config)

        assert isinstance(df_res, pd.DataFrame)
        assert len(df_res) == 1
        assert df_res.iloc[0]["Patient"] == "208"
        assert df_res.iloc[0]["Train_N"] == 1750
        assert df_res.iloc[0]["Test_N"] == 751
        assert round(float(df_res.iloc[0]["Accuracy"]), 2) == 99.60
        assert round(float(df_res.iloc[0]["Sensitivity"]), 2) == 98.83
        assert round(float(df_res.iloc[0]["Specificity"]), 2) == 100.00
        assert df_res.iloc[0]["Tree_Depth"] == 2

        assert "208" in feats
        assert len(feats["208"]["selected_features"]) == 12

        assert p208_cm is not None
        assert p208_cm.shape == (2, 2)
        assert p208_cm[0, 0] == 494  # TN
        assert p208_cm[0, 1] == 0    # FP
        assert p208_cm[1, 0] == 3    # FN
        assert p208_cm[1, 1] == 254  # TP

        assert isinstance(p208_model, DecisionTreeClassifier)
