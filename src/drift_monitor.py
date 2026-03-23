import os
import smtplib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from email.mime.text import MIMEText

try:
    import mlflow
except Exception:
    mlflow = None

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(_ROOT, 'reports')

PSI_STABLE  = 0.10
PSI_RETRAIN = 0.25

TIER_COLORS = {
    'Stable' : '#2ecc71',
    'Monitor': '#f39c12',
    'Retrain': '#e74c3c',
}



def _psi_single(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two 1-D numeric arrays using percentile bins."""
    eps    = 1e-6
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks = np.unique(breaks)
    if len(breaks) < 2:
        return 0.0

    exp_count, _ = np.histogram(expected, bins=breaks)
    act_count, _ = np.histogram(actual,   bins=breaks)

    exp_pct = exp_count / max(len(expected), 1) + eps
    act_pct = act_count / max(len(actual),   1) + eps

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_psi(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
    features:  list = None,
    bins:      int  = 10,
) -> pd.DataFrame:

    if features is None:
        features = reference.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        psi_val = _psi_single(ref_vals, cur_vals, bins)
        status  = (
            'Retrain' if psi_val >= PSI_RETRAIN else
            'Monitor' if psi_val >= PSI_STABLE  else
            'Stable'
        )
        rows.append({'feature': feat, 'psi': round(psi_val, 4), 'status': status})

    return (
        pd.DataFrame(rows)
        .sort_values('psi', ascending=False)
        .reset_index(drop=True)
    )



def simulate_drift(reference: pd.DataFrame,
                   drift_fraction: float = 0.3,
                   drift_strength: float = 0.5,
                   seed: int = 99) -> pd.DataFrame:

    rng      = np.random.default_rng(seed)
    drifted  = reference.copy()
    num_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    n_drift  = max(1, int(len(num_cols) * drift_fraction))
    drift_cols = rng.choice(num_cols, size=n_drift, replace=False)

    for col in drift_cols:
        std   = reference[col].std()
        shift = drift_strength * std * rng.choice([-1, 1])
        drifted[col] = drifted[col] + shift + rng.normal(0, std * 0.1, len(drifted))

    return drifted



def plot_drift_heatmap(psi_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Horizontal bar chart colour-coded by drift status.
    Green = Stable  |  Orange = Monitor  |  Red = Retrain
    """
    top = psi_df.head(30).sort_values('psi', ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(5, int(len(top) * 0.38))))
    bars = ax.barh(
        top['feature'],
        top['psi'],
        color=[TIER_COLORS[s] for s in top['status']],
        edgecolor='white',
    )
    ax.axvline(PSI_STABLE,  color='orange', linestyle='--', lw=1.4,
               label=f'Monitor threshold ({PSI_STABLE})')
    ax.axvline(PSI_RETRAIN, color='red',    linestyle='--', lw=1.4,
               label=f'Retrain threshold ({PSI_RETRAIN})')
    ax.set_xlabel('PSI Score', fontsize=11)
    ax.set_title('Feature Drift — Population Stability Index', fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(REPORTS_DIR, 'drift_psi.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def plot_feature_distribution(reference: pd.DataFrame,
                               current: pd.DataFrame,
                               feature: str) -> plt.Figure:
    """Overlay histograms for a single feature: reference vs current."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(reference[feature].dropna(), bins=30, alpha=0.55,
            color='#4e79a7', label='Reference (train)', density=True)
    ax.hist(current[feature].dropna(), bins=30, alpha=0.55,
            color='#e15759', label='Current (production)', density=True)
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution Shift — {feature}', fontsize=12)
    ax.legend()
    plt.tight_layout()
    return fig



def drift_summary(psi_df: pd.DataFrame) -> dict:
    """Return easy-to-display counts and an overall recommendation string."""
    counts  = psi_df['status'].value_counts().to_dict()
    total   = len(psi_df)
    retrain = counts.get('Retrain', 0)
    monitor = counts.get('Monitor', 0)

    if retrain > 0:
        recommendation = f'RETRAIN RECOMMENDED — {retrain} feature(s) drifted significantly'
    elif monitor > 3:
        recommendation = f'MONITOR CLOSELY — {monitor} features showing moderate drift'
    else:
        recommendation = 'Model is stable — no immediate action required'

    return {
        'total_features' : total,
        'stable'         : counts.get('Stable',  0),
        'monitor'        : monitor,
        'retrain'        : retrain,
        'recommendation' : recommendation,
    }


def log_drift_to_mlflow(psi_df: pd.DataFrame, run_name: str = 'drift_monitor') -> dict:
    """Log per-feature PSI and summary metrics to MLflow if available."""
    summary = drift_summary(psi_df)
    if mlflow is None:
        return summary

    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment = os.environ.get('MLFLOW_EXPERIMENT', 'Telco_Churn_Prediction')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        for _, row in psi_df.iterrows():
            mlflow.log_metric(f"psi_{row['feature']}", float(row['psi']))

        mlflow.log_metrics({
            'drift_total_features': int(summary['total_features']),
            'drift_stable_features': int(summary['stable']),
            'drift_monitor_features': int(summary['monitor']),
            'drift_retrain_features': int(summary['retrain']),
        })
        mlflow.log_params({
            'psi_stable_threshold': PSI_STABLE,
            'psi_retrain_threshold': PSI_RETRAIN,
        })
    return summary


def send_retrain_email_alert(summary: dict, psi_df: pd.DataFrame) -> bool:
    """Send an SMTP email alert when retraining is recommended."""
    if summary.get('retrain', 0) <= 0:
        return False

    smtp_host = os.environ.get('ALERT_SMTP_HOST')
    smtp_port = int(os.environ.get('ALERT_SMTP_PORT', '587'))
    smtp_user = os.environ.get('ALERT_SMTP_USER')
    smtp_pass = os.environ.get('ALERT_SMTP_PASSWORD')
    sender = os.environ.get('ALERT_FROM_EMAIL', smtp_user or '')
    recipients_raw = os.environ.get('ALERT_TO_EMAILS', '')
    recipients = [x.strip() for x in recipients_raw.split(',') if x.strip()]

    if not smtp_host or not sender or not recipients:
        return False

    top = psi_df.head(10)[['feature', 'psi', 'status']]
    body = (
        'Churn model drift alert\n\n'
        f"Recommendation: {summary.get('recommendation', 'N/A')}\n"
        f"Stable: {summary.get('stable', 0)} | Monitor: {summary.get('monitor', 0)} | Retrain: {summary.get('retrain', 0)}\n\n"
        'Top drifted features:\n'
        f"{top.to_string(index=False)}\n"
    )

    msg = MIMEText(body)
    msg['Subject'] = '[Churn MLOps] Retrain recommended'
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        server.starttls()
        if smtp_user and smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.sendmail(sender, recipients, msg.as_string())
    return True
