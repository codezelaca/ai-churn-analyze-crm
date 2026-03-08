import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORTS_DIR = os.path.join(_ROOT, 'reports')

DEFAULT_CONVERSION_RATE = 0.30



def run_ab_test(
    scored_df: pd.DataFrame,
    true_churn_col:    str   = 'Churn_Numeric',
    churn_proba_col:   str   = 'churn_proba',
    risk_col:          str   = 'risk_tier',
    treatment_size:    int   = None,
    conversion_rate:   float = DEFAULT_CONVERSION_RATE,
    seed:              int   = 42,
) -> dict:

    if true_churn_col not in scored_df.columns:
        raise ValueError(
            f"Column '{true_churn_col}' not found. "
            "The scored DataFrame must include the original churn labels."
        )

    rng = np.random.default_rng(seed)

    high_risk_pool = scored_df[scored_df[risk_col] == 'High'].copy()
    if len(high_risk_pool) < 10:
        raise ValueError("Too few High-risk customers for a valid A/B test (need ≥ 10).")

    n_treat = treatment_size if treatment_size else len(high_risk_pool)
    n_treat = min(n_treat, len(high_risk_pool))

    treatment = high_risk_pool.sample(n_treat, random_state=seed)
    remaining = scored_df.drop(index=treatment.index)
    control   = remaining.sample(min(n_treat, len(remaining)), random_state=seed)

    ctrl_churn_rate_base = control[true_churn_col].mean()
    trt_churn_rate_base  = treatment[true_churn_col].mean()

    is_churner = treatment[true_churn_col] == 1
    saved_mask = is_churner & (rng.random(len(treatment)) < conversion_rate)

    simulated_churn           = treatment[true_churn_col].copy()
    simulated_churn[saved_mask] = 0   # these customers no longer churn

    trt_churn_rate_post = simulated_churn.mean()
    ctrl_churn_rate_post = ctrl_churn_rate_base   # control unchanged

    n_ctrl = len(control)
    n_trt  = len(treatment)
    x_ctrl = int(round(ctrl_churn_rate_post * n_ctrl))
    x_trt  = int(round(trt_churn_rate_post  * n_trt))

    # Two-proportion z-test (one-tailed, H1: p_trt < p_ctrl)
    p_trt  = x_trt  / n_trt
    p_ctrl = x_ctrl / n_ctrl
    p_pool = (x_trt + x_ctrl) / (n_trt + n_ctrl)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1 / n_trt + 1 / n_ctrl))
    z_stat = (p_trt - p_ctrl) / se if se > 0 else 0.0
    p_value = norm.cdf(z_stat)

    abs_lift = ctrl_churn_rate_post - trt_churn_rate_post
    rel_lift = abs_lift / ctrl_churn_rate_post if ctrl_churn_rate_post > 0 else 0.0

    avg_revenue_per_customer = 65
    estimated_monthly_revenue_saved = int(saved_mask.sum()) * avg_revenue_per_customer

    return {
        'n_treatment'                   : n_trt,
        'n_control'                     : n_ctrl,
        'trt_churn_baseline'            : round(trt_churn_rate_base,  4),
        'ctrl_churn_rate'               : round(ctrl_churn_rate_post, 4),
        'trt_churn_rate_post'           : round(trt_churn_rate_post,  4),
        'absolute_lift'                 : round(abs_lift,        4),
        'relative_lift_pct'             : round(rel_lift * 100,  2),
        'customers_saved'               : int(saved_mask.sum()),
        'z_statistic'                   : round(float(z_stat),   4),
        'p_value'                       : round(float(p_value),  4),
        'significant'                   : bool(p_value < 0.05),
        'confidence_level'              : '95%',
        'conversion_rate_used'          : conversion_rate,
        'avg_revenue_per_customer'      : avg_revenue_per_customer,
        'estimated_monthly_revenue'     : estimated_monthly_revenue_saved,
    }



def plot_ab_results(ab_result: dict, save: bool = True) -> plt.Figure:
    """
    Side-by-side chart:
      Left  — churn rate comparison (Control vs Treatment)
      Right — annotated statistics panel
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    groups = ['Control\n(no offer)', 'Treatment\n(retention offer)']
    rates  = [ab_result['ctrl_churn_rate'], ab_result['trt_churn_rate_post']]
    bar_colors = ['#e15759', '#2ecc71']
    bars   = axes[0].bar(groups, rates, color=bar_colors, edgecolor='white', width=0.4)

    for bar, val in zip(bars, rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.1%}', ha='center', fontsize=13, fontweight='bold',
        )

    lift_txt = f'▼ {ab_result["relative_lift_pct"]:.1f}% relative lift'
    axes[0].text(0.5, max(rates) * 1.15, lift_txt,
                 ha='center', fontsize=11, color='#2ecc71', fontweight='bold',
                 transform=axes[0].transAxes)

    axes[0].set_ylim(0, max(rates) * 1.35)
    axes[0].set_ylabel('Churn Rate', fontsize=11)
    axes[0].set_title('Churn Rate — Control vs Treatment', fontsize=13)

    axes[1].axis('off')
    sig_txt   = '✓ SIGNIFICANT'  if ab_result['significant'] else '✗ NOT SIGNIFICANT'
    sig_color = '#2ecc71'        if ab_result['significant'] else '#e74c3c'

    rows = [
        ('Treatment (n)',        f"{ab_result['n_treatment']:,}"),
        ('Control (n)',          f"{ab_result['n_control']:,}"),
        ('Baseline churn (trt)', f"{ab_result['trt_churn_baseline']:.1%}"),
        ('Post-offer churn',     f"{ab_result['trt_churn_rate_post']:.1%}"),
        ('Absolute lift',        f"{ab_result['absolute_lift']:.1%}"),
        ('Relative lift',        f"{ab_result['relative_lift_pct']:.1f}%"),
        ('Customers saved',      f"{ab_result['customers_saved']:,}"),
        ('Z-statistic',          f"{ab_result['z_statistic']:.3f}"),
        ('p-value',              f"{ab_result['p_value']:.4f}"),
        ('Result (95%)',         sig_txt),
        ('Est. revenue saved',   f"£{ab_result['estimated_monthly_revenue']:,}/mo"),
    ]

    y = 0.97
    for label, val in rows:
        is_result = label.startswith('Result')
        color = sig_color if is_result else 'black'
        fw    = 'bold'    if is_result else 'normal'
        axes[1].text(0.02, y, f'{label}:', fontsize=10,
                     transform=axes[1].transAxes, color='grey')
        axes[1].text(0.58, y, val,           fontsize=10,
                     transform=axes[1].transAxes, color=color, fontweight=fw)
        y -= 0.087

    axes[1].set_title('Test Statistics & Business Impact', fontsize=13)

    plt.suptitle(
        'A/B Test — Model-Targeted Retention vs Random Control',
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(REPORTS_DIR, 'ab_test_results.png'),
                    dpi=150, bbox_inches='tight')
    return fig


def power_analysis_table(
    base_rate: float = 0.27,
    effect_sizes: list = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """
    Estimate required sample sizes for various expected lift sizes.
    Useful for planning the next live experiment.
    """
    if effect_sizes is None:
        effect_sizes = [0.05, 0.10, 0.15, 0.20, 0.25]

    rows = []
    for delta in effect_sizes:
        trt_rate = max(0.01, base_rate - delta)
        # Approximate formula: n = (z_a + z_b)^2 * (p1(1-p1)+p2(1-p2)) / (p1-p2)^2
        z_alpha = norm.ppf(1 - alpha)
        z_beta  = norm.ppf(power)
        p1, p2  = base_rate, trt_rate
        n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
        rows.append({
            'Expected absolute lift': f'{delta:.0%}',
            'Control rate':           f'{p1:.0%}',
            'Expected treatment rate':f'{p2:.0%}',
            'Required n per arm':      int(np.ceil(n)),
        })

    return pd.DataFrame(rows)
