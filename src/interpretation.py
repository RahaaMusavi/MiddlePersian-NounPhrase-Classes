import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class EzafeInterpreter:
    """
    Provides helper methods for interpreting Random Forest models in the context 
    of Middle Persian morphosyntax. Focuses on SHAP decomposition and 
    directional feature effects.
    """

    def __init__(self, pipeline, class_names=None):
        self.pipeline = pipeline
        self.clf = pipeline.named_steps['clf']
        self.preprocessor = pipeline.named_steps['pre']
        self.boruta = pipeline.named_steps['boruta']
        self.class_names = class_names or [
            "No ezafe & Head Initial",
            "No ezafe & Head Final",
            "With ezafe & Head Initial",
            "With ezafe & Head Final"
        ]

    def get_feature_names(self):
        """Returns the human-readable names of features selected by Boruta."""
        return [f.replace("remainder__", "").replace("cat__", "") 
                for f in self.boruta.get_feature_names_out()]

    def compute_shap_values(self, X):
        """
        Calculates SHAP values for a given dataset X.
        Note: X should be the raw (non-preprocessed) DataFrame.
        """
        X_transformed = self.preprocessor.transform(X)
        X_selected = self.boruta.transform(X_transformed)
        
        explainer = shap.TreeExplainer(self.clf)
        shap_values = explainer.shap_values(X_selected)
        
        return shap_values, X_selected

    def get_directional_effects(self, shap_values, X_selected):
        """
        Analyzes the directional impact of features (Positive/Negative) 
        per class, as presented in Table 6 of the paper.
        """
        feature_names = self.get_feature_names()
        records = []

        # Iterate through classes
        for class_idx, class_name in enumerate(self.class_names):
            class_shap = shap_values[class_idx]
            
            for f_idx, f_name in enumerate(feature_names):
                # Calculate mean SHAP when the feature is active (greater than 0)
                active_mask = X_selected[:, f_idx] > 0
                if np.any(active_mask):
                    mean_shap_active = np.mean(class_shap[active_mask, f_idx])
                else:
                    mean_shap_active = 0.0

                importance = np.mean(np.abs(class_shap[:, f_idx]))
                
                records.append({
                    "Class": class_name,
                    "Feature": f_name,
                    "Mean_SHAP_Impact": importance,
                    "Direction": "Positive" if mean_shap_active > 0 else "Negative"
                })

        df_results = pd.DataFrame(records)
        return df_results.sort_values(by=["Class", "Mean_SHAP_Impact"], ascending=[True, False])

    def summarize_class_stability(self, shap_values, X_selected):
        """
        Prints a summary of the primary predictors for 'Canonical' vs 'Peripheral' classes.
        Useful for verifying the 'Dual Stability' hypothesis.
        """
        df_dir = self.get_directional_effects(shap_values, X_selected)
        
        print("=== Top Predictors for Dual Stability Model ===")
        for class_name in self.class_names:
            print(f"\nClass: {class_name}")
            top_3 = df_dir[df_dir['Class'] == class_name].head(3)
            for _, row in top_3.iterrows():
                print(f" - {row['Feature']}: {row['Direction']} impact")

    def plot_feature_importance(self, shap_values, X_selected):
        """
        Generates a simplified global importance plot (Mean |SHAP|).
        """
        feature_names = self.get_feature_names()
        shap.summary_plot(
            shap_values, 
            X_selected, 
            feature_names=feature_names, 
            class_names=self.class_names,
            plot_type="bar"
        )