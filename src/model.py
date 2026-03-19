import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from boruta import BorutaPy

class EzafeModelPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.pipeline = None

    def build_pipeline(self, X):
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        self.preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
            remainder='passthrough'
        )

        rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        
        self.pipeline = Pipeline([
            ('pre', self.preprocessor),
            ('ros', RandomOverSampler(random_state=self.random_state)),
            ('boruta', BorutaPy(rf_boruta, n_estimators='auto', random_state=self.random_state)),
            ('clf', RandomForestClassifier(n_jobs=-1, random_state=self.random_state))
        ])

    def train_and_optimize(self, X, y):
        param_dist = {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [None, 10, 20],
            'clf__class_weight': ['balanced', 'balanced_subsample']
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(self.pipeline, param_dist, n_iter=10, cv=cv, scoring='f1_macro', n_jobs=-1)
        search.fit(X, y)
        self.pipeline = search.best_estimator_
        return search.best_params_

    def save_assets(self, path="artifacts/"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.pipeline, f"{path}ezafe_pipeline.joblib")