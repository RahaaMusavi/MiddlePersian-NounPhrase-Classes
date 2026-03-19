"""
Main execution script for 'Head Directionality and Dependency Marking in Middle Persian'.
This script handles the full experimental pipeline:
1. Data Loading and Target Encoding
2. Pipeline Optimization (RandomizedSearchCV)
3. 5-Fold Stratified Cross-Validation
4. Metric Reporting (Table 3)
5. SHAP-based Feature Interpretation (Table 6)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# Internal modules
from src.model import EzafeModelPipeline
from src.interpretation import EzafeInterpreter

def main():
    # 1. Configuration and Paths
    DATA_PATH = "data/processed/ezafe_dataset_final.csv"
    ARTIFACT_PATH = "artifacts/"
    RANDOM_STATE = 42
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}. " 
                                "Please run extraction or download the dataset.")

    # 2. Load Dataset (N = 8,019)
    print(f"--- Loading MPCD Annotated Pairs (N=8019) ---")
    df = pd.read_csv(DATA_PATH)
    
    # Define Target: The four-way configuration discussed in Table 1
    # 0_1: No ezafe & Head Initial | 0_2: No ezafe & Head Final
    # 1_1: With ezafe & Head Initial | 1_2: With ezafe & Head Final
    df['joint_label'] = df['ezafe_label'].astype(str) + "_" + df['position'].astype(str)
    
    # 3. Feature/Target Split
    # Exclude technical columns and targets to prevent data leakage
    drop_cols = ['ezafe_label', 'position', 'joint_label', 'source_file', 
                 'nominal_head_form', 'modifier_form']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['joint_label']

    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

    # 4. Initialize and Optimize Pipeline
    print("\n--- Phase 1: Pipeline Optimization & Boruta Selection ---")
    model_wrapper = EzafeModelPipeline(random_state=RANDOM_STATE)
    model_wrapper.build_pipeline(X)
    
    best_params = model_wrapper.train_and_optimize(X, y)
    print(f"Optimization complete. Best Params: {best_params}")

    # 5. Cross-Validation and Metric Generation (Table 3)
    print("\n--- Phase 2: 5-Fold Stratified Evaluation (Table 3) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Perform per-fold training for performance report
    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit on training fold
        model_wrapper.pipeline.fit(X_train, y_train)
        y_pred = model_wrapper.pipeline.predict(X_test)
        
        fold_reports.append(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
        print(f"Fold {fold} evaluated.")

    # Average results for the final report
    print("\n=== Final Classification Report (Composite Tasks) ===")
    final_pipeline = model_wrapper.pipeline.fit(X, y) # Final fit on full data
    y_pred_final = final_pipeline.predict(X)
    
    # Mapping back to paper nomenclature
    label_map = {
        "0_1": "No ezafe & Head Initial",
        "0_2": "No ezafe & Head Final",
        "1_1": "With ezafe & Head Initial",
        "1_2": "With ezafe & Head Final"
    }
    
    report = classification_report(y, y_pred_final, target_names=[label_map[c] for c in sorted(y.unique())])
    print(report)

    # 6. Feature Interpretation (Table 6)
    print("\n--- Phase 3: SHAP Interpretation (Table 6) ---")
    interpreter = EzafeInterpreter(final_pipeline)
    shap_values, X_selected = interpreter.compute_shap_values(X)
    
    df_direction = interpreter.get_directional_effects(shap_values, X_selected)
    
    # Save Table 6 result
    table6_path = os.path.join(ARTIFACT_PATH, "table6_directional_effects.csv")
    df_direction.to_csv(table6_path, index=False)
    print(f"Directional effects saved to {table6_path}")

    # 7. Persist Model and Selected Features
    print("\n--- Phase 4: Saving Artifacts ---")
    model_wrapper.save_assets(ARTIFACT_PATH)
    
    # Save the Boruta selected feature list for manual inspection
    selected_feats = interpreter.get_feature_names()
    with open(os.path.join(ARTIFACT_PATH, "selected_features.txt"), "w") as f:
        f.write("\n".join(selected_feats))
    
    print(f"Artifacts successfully saved to {ARTIFACT_PATH}")
    print("Experiment Complete.")

if __name__ == "__main__":
    main()