import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(_ROOT, 'reports')



def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 500):

    X_sample = X.sample(min(max_samples, len(X)), random_state=42) if len(X) > max_samples else X.copy()

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample


def _churn_class_shap(shap_values):

    if shap_values.values.ndim == 3:        
        return shap_values[:, :, 1]
    return shap_values                      



def plot_summary(shap_values, X_sample: pd.DataFrame,
                 top_n: int = 20, save: bool = True) -> plt.Figure:
    """
    Beeswarm summary plot — global feature impact on churn probability.
    Each dot = one customer; colour = feature value; position = SHAP impact.
    """
    sv = _churn_class_shap(shap_values)

    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(sv, max_display=top_n, show=False)
    plt.title('SHAP Feature Impact — Global View (Churn class)', fontsize=13)
    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(REPORTS_DIR, 'shap_summary.png'),
                    dpi=150, bbox_inches='tight')
    return plt.gcf()


def plot_waterfall(shap_values, idx: int = 0, save: bool = True) -> plt.Figure:

    sv = _churn_class_shap(shap_values)

    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(sv[idx], max_display=15, show=False)
    plt.title(f'SHAP Explanation — Customer Row #{idx}', fontsize=13)
    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(REPORTS_DIR, f'shap_waterfall_{idx}.png'),
                    dpi=150, bbox_inches='tight')
    return plt.gcf()


def plot_bar_importance(shap_values, top_n: int = 20, save: bool = True) -> plt.Figure:
    """Mean |SHAP| bar chart — clean version for non-technical stakeholders."""
    sv     = _churn_class_shap(shap_values)
    vals   = np.abs(sv.values).mean(axis=0)
    feats  = sv.feature_names if sv.feature_names is not None else [f'f{i}' for i in range(len(vals))]

    imp_df = (
        pd.DataFrame({'feature': feats, 'mean_shap': vals})
        .sort_values('mean_shap', ascending=False)
        .head(top_n)
        .sort_values('mean_shap', ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    cmap   = plt.cm.get_cmap('viridis', len(imp_df))
    colors = [cmap(i) for i in range(len(imp_df))]
    ax.barh(imp_df['feature'], imp_df['mean_shap'], color=colors, edgecolor='white')
    ax.set_xlabel('Mean |SHAP value| — Average impact on churn probability')
    ax.set_title(f'Top {top_n} Features by SHAP Importance', fontsize=13)
    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(REPORTS_DIR, 'shap_bar.png'),
                    dpi=150, bbox_inches='tight')
    return fig



def get_shap_df(shap_values, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Return SHAP values as a tidy DataFrame aligned with X_sample."""
    sv = _churn_class_shap(shap_values)
    return pd.DataFrame(sv.values, columns=X_sample.columns, index=X_sample.index)
