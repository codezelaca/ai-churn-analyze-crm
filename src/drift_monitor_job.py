import argparse
from datetime import datetime, timedelta

import pandas as pd

try:
    from src.drift_monitor import compute_psi, log_drift_to_mlflow, send_retrain_email_alert
except Exception:
    from drift_monitor import compute_psi, log_drift_to_mlflow, send_retrain_email_alert


def run_drift_job(reference_path: str, current_path: str, bins: int = 10) -> dict:
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    numeric_features = reference.select_dtypes(include=['number']).columns.tolist()
    psi_df = compute_psi(reference=reference, current=current, features=numeric_features, bins=bins)

    summary = log_drift_to_mlflow(psi_df=psi_df, run_name='daily_drift_monitor')
    send_retrain_email_alert(summary=summary, psi_df=psi_df)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run drift monitoring and log to MLflow.')
    parser.add_argument('--reference-path', required=True, help='Path to reference dataset CSV')
    parser.add_argument('--current-path', required=True, help='Path to current dataset CSV')
    parser.add_argument('--bins', type=int, default=10, help='Histogram bins for PSI')
    parser.add_argument('--run-date', default=(datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_drift_job(reference_path=args.reference_path, current_path=args.current_path, bins=args.bins)
    print('Drift check complete')
    print(summary)


if __name__ == '__main__':
    main()
