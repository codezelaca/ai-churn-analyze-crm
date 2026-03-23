import argparse
import os
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deploy_model import package_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Log an existing local model artifact (.joblib) to MLflow and register it.'
    )
    parser.add_argument(
        '--model-path',
        default='models/rf_churn_model.joblib',
        help='Path to the existing joblib artifact produced by package_and_save.',
    )
    parser.add_argument(
        '--model-name',
        default='rf_churn_model',
        help='Registered model name in MLflow.',
    )
    parser.add_argument(
        '--tracking-uri',
        default='http://localhost:5000',
        help='MLflow tracking server URI.',
    )
    parser.add_argument(
        '--experiment',
        default='Telco_Churn_Prediction',
        help='MLflow experiment name.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f'Model artifact not found: {model_path}')

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or 'model' not in artifact:
        raise ValueError('Expected artifact dict with keys including: model, feature_names, threshold')

    model = artifact['model']
    feature_names = artifact.get('feature_names')
    threshold = float(artifact.get('threshold', 0.36))

    if not feature_names:
        raise ValueError('feature_names is missing or empty in artifact; cannot safely log model schema.')

    os.environ['MLFLOW_TRACKING_URI'] = args.tracking_uri
    os.environ['MLFLOW_EXPERIMENT'] = args.experiment

    out_path = package_and_save(
        model=model,
        feature_names=feature_names,
        threshold=threshold,
        model_name=args.model_name,
    )

    print('Done: model logged to MLflow.')
    print(f'Local artifact path: {out_path}')
    print(f'Tracking URI: {args.tracking_uri}')
    print(f'Experiment: {args.experiment}')


if __name__ == '__main__':
    main()
