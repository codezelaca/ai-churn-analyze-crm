import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR  = os.path.join(_ROOT, 'models')
REPORTS_DIR = os.path.join(_ROOT, 'reports')



def package_and_save(
    model,
    feature_names: list,
    threshold: float = 0.36,
    model_name: str = 'rf_churn_model',
) -> str:

    os.makedirs(MODELS_DIR, exist_ok=True)

    artifact = {
        'model'        : model,
        'feature_names': feature_names,
        'threshold'    : threshold,
        'trained_at'   : datetime.now().isoformat(),
        'model_type'   : type(model).__name__,
        'n_features'   : len(feature_names),
    }

    joblib_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    joblib.dump(artifact, joblib_path)

    meta = {k: v for k, v in artifact.items() if k != 'model'}
    meta_path = os.path.join(MODELS_DIR, f'{model_name}_meta.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Model artifact → {joblib_path}")
    print(f"  Metadata JSON  → {meta_path}")
    return joblib_path



def load_model(model_name: str = 'rf_churn_model') -> dict:

    path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at '{path}'. "
            "Run package_and_save() first (notebook cell 14)."
        )
    return joblib.load(path)



def predict(X: pd.DataFrame, model_name: str = 'rf_churn_model') -> pd.DataFrame:

    artifact      = load_model(model_name)
    model         = artifact['model']
    feature_names = artifact['feature_names']
    threshold     = artifact['threshold']

    X_scored = X[feature_names].copy()
    proba    = model.predict_proba(X_scored)[:, 1]

    return pd.DataFrame(
        {
            'churn_proba': proba.round(4),
            'churn_flag' : (proba >= threshold).astype(int),
        },
        index=X.index,
    )



def model_card(model_name: str = 'rf_churn_model') -> dict:
    """Return metadata dict for display in Streamlit dashboard."""
    meta_path = os.path.join(MODELS_DIR, f'{model_name}_meta.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as fh:
        return json.load(fh)
