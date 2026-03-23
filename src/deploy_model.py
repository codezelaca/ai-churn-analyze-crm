import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR  = os.path.join(_ROOT, 'models')
REPORTS_DIR = os.path.join(_ROOT, 'reports')


def _mlflow_ready() -> bool:
    return mlflow is not None


def _configure_mlflow() -> None:
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment = os.environ.get('MLFLOW_EXPERIMENT', 'Telco_Churn_Prediction')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def _set_alias_to_latest(model_name: str) -> None:
    alias = os.environ.get('MLFLOW_MODEL_ALIAS')
    if not alias:
        return

    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(model_name)
    if not latest_versions:
        return

    latest_version = max(latest_versions, key=lambda mv: int(mv.version))
    client.set_registered_model_alias(model_name, alias, latest_version.version)



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

    if _mlflow_ready():
        _configure_mlflow()
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param('threshold', float(threshold))
            mlflow.log_param('n_features', int(len(feature_names)))
            mlflow.log_param('model_type', type(model).__name__)
            mlflow.log_dict({'features': feature_names}, 'feature_names.json')
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path='model',
                registered_model_name=model_name,
            )
            _set_alias_to_latest(model_name)
            print('  Model logged to MLflow')
    else:
        print('  MLflow not available; skipped experiment logging')

    print(f"  Model artifact → {joblib_path}")
    print(f"  Metadata JSON  → {meta_path}")
    return joblib_path



def _load_model_mlflow(model_name: str = 'rf_churn_model') -> dict:
    if not _mlflow_ready():
        raise RuntimeError('MLflow is not installed but MLflow model loading was requested.')

    _configure_mlflow()
    alias = os.environ.get('MLFLOW_MODEL_ALIAS')
    stage = os.environ.get('MLFLOW_MODEL_STAGE')

    if alias:
        model_uri = f'models:/{model_name}@{alias}'
    elif stage:
        model_uri = f'models:/{model_name}/{stage}'
    else:
        model_uri = f'models:/{model_name}/latest'

    model = mlflow.sklearn.load_model(model_uri)

    # Metadata compatibility for existing downstream code.
    return {
        'model': model,
        'feature_names': None,
        'threshold': float(os.environ.get('DEFAULT_PREDICTION_THRESHOLD', '0.36')),
        'trained_at': None,
        'model_type': type(model).__name__,
        'n_features': None,
    }


def load_model(model_name: str = 'rf_churn_model') -> dict:
    model_source = os.environ.get('MODEL_SOURCE', 'local').strip().lower()

    if model_source == 'mlflow':
        try:
            return _load_model_mlflow(model_name=model_name)
        except Exception as exc:
            fallback = os.environ.get('ALLOW_LOCAL_MODEL_FALLBACK', '1') == '1'
            if not fallback:
                raise
            print(f"  MLflow load failed ({exc}); falling back to local artifact")

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

    if feature_names is None:
        X_scored = X.copy()
    else:
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
    if os.environ.get('MODEL_SOURCE', 'local').strip().lower() == 'mlflow':
        return {
            'model_name': model_name,
            'model_source': 'mlflow',
            'threshold': float(os.environ.get('DEFAULT_PREDICTION_THRESHOLD', '0.36')),
        }

    meta_path = os.path.join(MODELS_DIR, f'{model_name}_meta.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as fh:
        return json.load(fh)
