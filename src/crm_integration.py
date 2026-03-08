import os
import pandas as pd
import numpy as np
from datetime import datetime

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(_ROOT, 'reports')

TIER_HIGH_THR   = 0.60
TIER_MEDIUM_THR = 0.36

OFFER_MAP = {
    'High'  : 'Priority call + 20% discount + contract upgrade offer',
    'Medium': 'Automated email — loyalty reward + plan upgrade',
    'Low'   : 'Quarterly NPS survey + standard newsletter',
}

TIER_COLORS = {
    'High'  : '#e74c3c',
    'Medium': '#f39c12',
    'Low'   : '#2ecc71',
}



def assign_risk_tier(proba: float,
                     high_thr: float = TIER_HIGH_THR,
                     medium_thr: float = TIER_MEDIUM_THR) -> str:
    if proba >= high_thr:
        return 'High'
    elif proba >= medium_thr:
        return 'Medium'
    return 'Low'



def score_customers(
    df: pd.DataFrame,
    model,
    feature_names: list,
    threshold: float = TIER_MEDIUM_THR,
) -> pd.DataFrame:

    X     = df[feature_names].copy()
    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out['churn_proba']     = proba.round(4)
    out['churn_flag']      = (proba >= threshold).astype(int)
    out['risk_tier']       = [assign_risk_tier(p) for p in proba]
    out['retention_offer'] = out['risk_tier'].map(OFFER_MAP)
    out['priority_rank']   = (
        out['churn_proba']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    out['scored_at'] = datetime.now().strftime('%Y-%m-%d')

    return out.sort_values('churn_proba', ascending=False).reset_index(drop=True)



def export_crm_csv(
    scored_df: pd.DataFrame,
    filename: str = 'crm_churn_scores.csv',
    include_features: bool = False,
) -> str:

    os.makedirs(REPORTS_DIR, exist_ok=True)

    crm_cols = [
        'churn_proba', 'churn_flag', 'risk_tier',
        'retention_offer', 'priority_rank', 'scored_at',
    ]
    if include_features:
        export_df = scored_df
    else:
        export_df = scored_df[[c for c in crm_cols if c in scored_df.columns]]

    out_path = os.path.join(REPORTS_DIR, filename)
    export_df.to_csv(out_path, index=True)
    print(f"CRM export saved → {out_path}  ({len(export_df):,} rows)")
    return out_path



def get_tier_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer counts and avg churn probability per risk tier."""
    return (
        scored_df.groupby('risk_tier', observed=True)
        .agg(
            customers=('churn_proba', 'count'),
            avg_proba=('churn_proba', 'mean'),
            max_proba=('churn_proba', 'max'),
        )
        .round(4)
        .reindex(['High', 'Medium', 'Low'])
        .fillna(0)
    )


def top_at_risk(scored_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top-n highest-risk customers."""
    cols = ['churn_proba', 'risk_tier', 'retention_offer', 'priority_rank']
    available = [c for c in cols if c in scored_df.columns]
    return scored_df[available].head(n)
