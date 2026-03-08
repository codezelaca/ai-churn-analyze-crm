import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
