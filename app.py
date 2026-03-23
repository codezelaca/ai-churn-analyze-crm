import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from deploy_model    import load_model, model_card
from shap_analysis   import compute_shap_values, plot_waterfall, plot_bar_importance, get_shap_df
from crm_integration import score_customers, export_crm_csv, get_tier_summary, top_at_risk, TIER_COLORS
from drift_monitor   import compute_psi, plot_feature_distribution, drift_summary, simulate_drift
from ab_testing      import run_ab_test, power_analysis_table

ROOT      = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, 'data', 'telco_churn_processed.csv')
API_URL   = os.environ.get('API_URL', 'http://localhost:8000')

AVG_MONTHLY_REVENUE = 65   # £ per customer (MonthlyCharges proxy)

PLAIN_NAMES = {
    'tenure'                        : 'How long they have been a customer',
    'MonthlyCharges'                : 'Their monthly bill',
    'TotalCharges'                  : 'Their total spend to date',
    'AvgMonthlySpend'               : 'Average monthly spending',
    'Contract_Two year'             : 'On a 2-year contract',
    'Contract_One year'             : 'On a 1-year contract',
    'Contract_Month-to-month'       : 'On a month-to-month contract (easy to leave)',
    'InternetService_Fiber optic'   : 'Uses fibre broadband',
    'InternetService_DSL'           : 'Uses DSL broadband',
    'InternetService_No'            : 'No internet service',
    'TechSupport_Yes'               : 'Has tech support',
    'TechSupport_No'                : 'No tech support',
    'OnlineSecurity_Yes'            : 'Has online security add-on',
    'OnlineSecurity_No'             : 'No online security',
    'PaperlessBilling_1'            : 'Uses paperless billing',
    'PaymentMethod_Electronic check': 'Pays by electronic cheque',
    'SeniorCitizen'                 : 'Senior citizen',
    'Dependents_1'                  : 'Has dependents on the account',
    'Partner_1'                     : 'Has a partner on the account',
    'TenureGroup'                   : 'Customer tenure group',
}


def plain(name: str) -> str:
    return PLAIN_NAMES.get(name, name.replace('_', ' ').replace('-', ' ').title())


st.set_page_config(
    page_title='Customer Retention Dashboard',
    page_icon='📡',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg,#1e2130,#2a2f45);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border: 1px solid #2e3450;
}
.kpi-card .label { color:#8892a4; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em; margin:0; }
.kpi-card .value { color:#ffffff; font-size:2rem; font-weight:800; margin:0.2rem 0 0; }
.kpi-card .sub   { color:#8892a4; font-size:0.78rem; }

.action-urgent { background:#fff0f0; border-left:6px solid #e74c3c; padding:1rem 1.3rem; border-radius:8px; margin-bottom:0.6rem; color:#1a1a1a; }
.action-watch  { background:#fffbf0; border-left:6px solid #f39c12; padding:1rem 1.3rem; border-radius:8px; margin-bottom:0.6rem; color:#1a1a1a; }
.action-ok     { background:#f0fff4; border-left:6px solid #2ecc71; padding:1rem 1.3rem; border-radius:8px; margin-bottom:0.6rem; color:#1a1a1a; }
.action-urgent .action-title { font-size:1rem; font-weight:700; margin-bottom:0.3rem; }
.action-watch  .action-title { font-size:1rem; font-weight:700; margin-bottom:0.3rem; }
.action-ok     .action-title { font-size:1rem; font-weight:700; margin-bottom:0.3rem; }
.action-urgent .action-desc, .action-watch .action-desc, .action-ok .action-desc { font-size:0.9rem; color:#333; margin-bottom:0.5rem; }
.action-urgent .action-rev { font-size:1.35rem; font-weight:800; color:#c0392b; }
.action-watch  .action-rev { font-size:1.35rem; font-weight:800; color:#d68910; }
.insight-box   { background:#f4f6fb; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem; line-height:1.8; color:#1a1a1a; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading prediction system…')
def get_model():
    try:
        return load_model('rf_churn_model')
    except Exception:
        # API inference mode can run without local model artifacts.
        return {
            'model': None,
            'feature_names': [],
            'threshold': float(os.environ.get('DEFAULT_PREDICTION_THRESHOLD', '0.36')),
        }


@st.cache_data(show_spinner='Loading customer data…')
def get_data():
    return pd.read_csv(DATA_PATH)


def model_ready() -> bool:
    if use_api_inference():
        return True
    return os.path.exists(os.path.join(ROOT, 'models', 'rf_churn_model.joblib'))


def use_api_inference() -> bool:
    return os.environ.get('USE_API_INFERENCE', '0') == '1'


def show_deploy_prompt():
    st.warning(
        "⚠️  The prediction system has not been set up yet.\n\n"
        "Ask your data team to run **notebook cell 14 (Deploy Model)** "
        "in `notebooks/model_evaluation.ipynb` to activate this dashboard.",
        icon='🔒',
    )
    st.stop()


@st.cache_data(show_spinner='Scoring customers…')
def get_scored(_model, _feature_names, _threshold, _data_hash):
    df = get_data()
    if use_api_inference():
        try:
            response = requests.post(
                f'{API_URL}/predict',
                json={'records': df.to_dict(orient='records')},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            preds = pd.DataFrame(payload.get('predictions', []), index=df.index)
            if preds.empty:
                raise ValueError('No predictions returned by API')
            return pd.concat([df, preds], axis=1)
        except Exception as exc:
            st.warning(f'API scoring unavailable, using local model instead. ({exc})')

    return score_customers(df, _model, _feature_names, _threshold)


with st.sidebar:
    st.markdown('<div style="font-size:2.8rem;margin-bottom:0.1rem">📡</div>', unsafe_allow_html=True)
    st.title('Retention Dashboard')
    st.caption('Powered by predictive analytics')
    st.divider()

    page = st.radio(
        'Navigate',
        options=[
            "🏠  Today's Overview",
            '👥  Customers to Contact',
            '🔍  Why Is a Customer Leaving?',
            '📣  Campaign Results',
            '⚙️  System Health',
        ],
        label_visibility='collapsed',
    )
    st.divider()

    if model_ready():
        card    = model_card('rf_churn_model')
        trained = (card.get('trained_at') or '')[:10] or '—'
        st.caption(f"Last updated: {trained}")
        st.caption('Status: 🟢 Active')
    else:
        st.caption('Status: 🔴 Not set up')


# ── PAGE 1 — Today's Overview ─────────────────────────────────────────────────
if page == "🏠  Today's Overview":
    st.title("🏠 Today's Retention Overview")
    st.markdown(
        "A quick snapshot of which customers are likely to leave "
        "and what it means for your monthly revenue."
    )

    if not model_ready():
        show_deploy_prompt()

    artifact      = get_model()
    model         = artifact['model']
    feature_names = artifact['feature_names']
    threshold     = artifact['threshold']
    df            = get_data()

    scored      = get_scored(model, feature_names, threshold, len(df))
    tier_counts = scored['risk_tier'].value_counts()
    n_high      = tier_counts.get('High', 0)
    n_medium    = tier_counts.get('Medium', 0)
    n_low       = tier_counts.get('Low', 0)
    total       = len(scored)
    rev_urgent  = n_high   * AVG_MONTHLY_REVENUE
    rev_watch   = n_medium * AVG_MONTHLY_REVENUE

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="label">Total Customers</p>
            <p class="value">{total:,}</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="label">⚠️ Urgent — Act Now</p>
            <p class="value" style="color:#e74c3c">{n_high:,}</p>
            <p class="sub">Very likely to leave</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="label">👀 Watch Closely</p>
            <p class="value" style="color:#f39c12">{n_medium:,}</p>
            <p class="sub">Moderate risk</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="label">💰 Revenue at Risk / mo</p>
            <p class="value" style="color:#e74c3c">£{rev_urgent:,}</p>
            <p class="sub">Urgent group only</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader('Customer Risk Breakdown')
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes   = [n_high, n_medium, n_low]
        labels  = [f'Urgent\n{n_high:,}', f'Watch\n{n_medium:,}', f'Safe\n{n_low:,}']
        colors  = ['#e74c3c', '#f39c12', '#2ecc71']
        _, _, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2),
            autopct='%1.0f%%', pctdistance=0.78,
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight('bold')
        ax.set_title('Who needs your attention?', fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        st.subheader('What Should You Do Today?')
        st.markdown(f"""
        <div class="action-urgent">
            <div class="action-title">🔴 Urgent &nbsp;·&nbsp; {n_high:,} customers</div>
            <div class="action-desc">Call personally or send a priority retention offer.
            These customers are very likely to cancel in the next 30 days.</div>
            <div>Monthly revenue at stake:</div>
            <div class="action-rev">£{rev_urgent:,}</div>
        </div>
        <div class="action-watch">
            <div class="action-title">🟡 Watch Closely &nbsp;·&nbsp; {n_medium:,} customers</div>
            <div class="action-desc">Send a loyalty email or plan upgrade offer
            before they move into the urgent group.</div>
            <div>Monthly revenue at stake:</div>
            <div class="action-rev">£{rev_watch:,}</div>
        </div>
        <div class="action-ok">
            <div class="action-title">🟢 Safe &nbsp;·&nbsp; {n_low:,} customers</div>
            <div class="action-desc">No immediate action needed.
            A routine newsletter or quarterly check-in is sufficient.</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader('🔥 Top 5 Customers to Contact Right Now')
    st.caption("These are your highest-risk customers — prioritise these for outreach today.")
    top5 = top_at_risk(scored, n=5)[['churn_proba', 'risk_tier', 'retention_offer', 'priority_rank']]
    top5 = top5.rename(columns={
        'churn_proba'     : 'Risk Score',
        'risk_tier'       : 'Urgency',
        'retention_offer' : 'Recommended Action',
        'priority_rank'   : 'Priority',
    })
    top5['Risk Score'] = top5['Risk Score'].apply(lambda x: f'{x:.0%}')
    st.dataframe(top5, use_container_width=True, hide_index=True)

    st.info(
        "👉 Go to **Customers to Contact** to see the full list and download it for your CRM.",
        icon='💡',
    )


# ── PAGE 2 — Customers to Contact ────────────────────────────────────────────
elif page == '👥  Customers to Contact':
    st.title('👥 Customers to Contact')
    st.markdown(
        "Every customer ranked by how likely they are to leave, "
        "with the recommended action already attached. Filter, review, and export."
    )

    if not model_ready():
        show_deploy_prompt()

    artifact      = get_model()
    model         = artifact['model']
    feature_names = artifact['feature_names']
    threshold     = artifact['threshold']
    df            = get_data()

    scored      = get_scored(model, feature_names, threshold, len(df))
    tier_counts = scored['risk_tier'].value_counts()
    total       = len(scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Customers', f"{total:,}")
    c2.metric('Urgent 🔴',        f"{tier_counts.get('High',   0):,}",
              help='Very likely to leave — contact immediately')
    c3.metric('Watch 🟡',         f"{tier_counts.get('Medium', 0):,}",
              help='Moderate risk — engage proactively')
    c4.metric('Safe 🟢',          f"{tier_counts.get('Low',    0):,}",
              help='Low risk — routine contact only')

    st.divider()

    with st.sidebar:
        st.subheader('Filter List')
        urgency_filter = st.multiselect(
            'Show urgency levels',
            options=['Urgent', 'Watch', 'Safe'],
            default=['Urgent', 'Watch'],
        )
        top_n_show = st.slider('Rows to display', 10, 500, 100, step=10)

    tier_map       = {'Urgent': 'High', 'Watch': 'Medium', 'Safe': 'Low'}
    selected_tiers = [tier_map[u] for u in urgency_filter if u in tier_map]
    filtered       = scored[scored['risk_tier'].isin(selected_tiers)].head(top_n_show)

    disp_cols = [c for c in ['priority_rank','churn_proba','risk_tier','retention_offer','scored_at']
                 if c in filtered.columns]
    disp_df = (
        filtered[disp_cols]
        .rename(columns={
            'priority_rank'  : 'Priority',
            'churn_proba'    : 'Risk Score',
            'risk_tier'      : 'Urgency',
            'retention_offer': 'Recommended Action',
            'scored_at'      : 'Scored On',
        })
        .reset_index(drop=True)
    )
    disp_df.index += 1
    disp_df['Risk Score'] = disp_df['Risk Score'].apply(lambda x: f'{x:.0%}')

    st.dataframe(disp_df, use_container_width=True, height=420)
    st.caption(f"Showing {len(filtered):,} customers · sorted highest risk first")

    st.divider()

    st.subheader('Recommended Actions by Group')
    for tier, label, offer, color in [
        ('High',   '🔴 Urgent',       'Personal phone call + 20% discount + contract upgrade offer', '#e74c3c'),
        ('Medium', '🟡 Watch Closely', 'Loyalty reward email + plan upgrade suggestion',              '#f39c12'),
        ('Low',    '🟢 Safe',          'Quarterly newsletter + standard satisfaction survey',         '#2ecc71'),
    ]:
        n = tier_counts.get(tier, 0)
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:0.6rem 1rem;'
            f'margin-bottom:0.5rem;border-radius:5px;background:#fafafa;">'
            f'<b style="color:{color}">{label} — {n:,} customers</b><br>'
            f'<span style="color:#444">{offer}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader('Download Customer List')
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button('💾  Save to reports folder', type='primary'):
            with st.spinner('Saving…'):
                path = export_crm_csv(scored)
            st.success(f'Saved → `{path}`')
    with col_b:
        csv_out = scored[[c for c in ['churn_proba','risk_tier','retention_offer',
                                       'priority_rank','scored_at']
                           if c in scored.columns]].to_csv().encode()
        st.download_button(
            label='⬇️  Download as spreadsheet',
            data=csv_out,
            file_name='customers_to_contact.csv',
            mime='text/csv',
        )


# ── PAGE 3 — Why Is a Customer Leaving? ──────────────────────────────────────
elif page == '🔍  Why Is a Customer Leaving?':
    st.title('🔍 Why Is a Customer Likely to Leave?')
    st.markdown(
        "Select any customer to see a plain-English breakdown of the key reasons "
        "our system believes they may cancel — and what you can do about it."
    )

    if not model_ready():
        show_deploy_prompt()

    artifact      = get_model()
    model         = artifact['model']
    feature_names = artifact['feature_names']
    df            = get_data()

    if model is None:
        st.info(
            'SHAP explainability requires local model access. '
            'Run with local model artifacts (MODEL_SOURCE=local) to use this page.',
            icon='ℹ️',
        )
        st.stop()

    X = df.drop(columns=['Churn_Numeric'])

    with st.sidebar:
        st.subheader('Settings')
        max_samples = st.slider(
            'Customers to analyse', 100, 800, 400, step=100,
            help='More = slower but covers more customers.',
        )

    with st.spinner('Analysing customer patterns…'):
        _, shap_values, X_sample = compute_shap_values(model, X[feature_names], max_samples=max_samples)

    import shap as _shap

    all_probas   = model.predict_proba(X_sample[feature_names])[:, 1]
    sorted_order = np.argsort(all_probas)[::-1]
    X_sorted     = X_sample.iloc[sorted_order].reset_index(drop=True)
    shap_sorted  = _shap.Explanation(
        values        = shap_values.values[sorted_order],
        base_values   = (shap_values.base_values[sorted_order]
                         if shap_values.base_values.ndim > 0
                         else shap_values.base_values),
        data          = shap_values.data[sorted_order],
        feature_names = shap_values.feature_names,
    )

    n_customers = len(X_sorted)

    rank  = st.slider(
        f'Select customer (1 = highest risk, {n_customers} = lowest risk)',
        min_value=1, max_value=n_customers, value=1,
    )
    idx   = rank - 1
    proba = model.predict_proba(X_sorted.iloc[[idx]][feature_names])[:, 1][0]

    if proba >= 0.60:
        urgency, uc, ubg = 'Urgent', '#e74c3c', '#fff0f0'
        msg = "This customer is very likely to cancel. Contact them as soon as possible."
        action_html = """
        <div class="action-urgent">
            <b>🔴 Immediate action recommended</b><br>
            1. Assign to a retention agent for a personal call within 24 hours.<br>
            2. Offer a personalised discount — start at 15–20%.<br>
            3. Discuss upgrading to a longer-term contract to lock in loyalty benefits.
        </div>"""
    elif proba >= 0.36:
        urgency, uc, ubg = 'Watch Closely', '#f39c12', '#fffbf0'
        msg = "This customer shows signs of risk. A proactive offer could keep them."
        action_html = """
        <div class="action-watch">
            <b>🟡 Proactive engagement recommended</b><br>
            1. Send a personalised loyalty email within the next 3 days.<br>
            2. Highlight add-ons they are missing (tech support, security).<br>
            3. Consider offering a loyalty reward or bill credit.
        </div>"""
    else:
        urgency, uc, ubg = 'Safe', '#2ecc71', '#f0fff4'
        msg = "This customer appears satisfied and is unlikely to leave soon."
        action_html = """
        <div class="action-ok">
            <b>🟢 No urgent action needed</b><br>
            Include in your next quarterly newsletter. Review again at end of quarter.
        </div>"""

    st.markdown(f"""
    <div style="background:{ubg};border-left:5px solid {uc};
                padding:1rem 1.3rem;border-radius:8px;margin-bottom:1rem;">
        <span style="font-size:0.85rem;color:{uc};font-weight:700;text-transform:uppercase">{urgency}</span><br>
        <span style="font-size:1.8rem;font-weight:800;color:{uc}">{proba:.0%} chance of leaving</span><br>
        <span style="color:#555">{msg}</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric('Position in risk ranking', f'#{rank} of {n_customers}',
              help='1 = the customer most likely to leave from the analysed group')
    m2.metric('Suggested action',
              'Priority call' if proba >= 0.60 else
              'Send offer email' if proba >= 0.36 else
              'Routine check-in')

    st.divider()

    shap_df      = get_shap_df(shap_sorted, X_sorted)
    row_shap     = shap_df.iloc[idx]
    row_data     = X_sorted.iloc[idx]
    top_push     = row_shap.nlargest(5)
    top_protect  = row_shap.nsmallest(5)
    max_push_val = abs(top_push.max()) + 1e-9
    max_prot_val = abs(top_protect.min()) + 1e-9

    col_push, col_protect = st.columns(2)

    with col_push:
        st.subheader('⚠️ Reasons this customer may leave')
        st.caption('These factors are increasing their chance of cancelling.')
        for feat, val in top_push.items():
            if val <= 0:
                continue
            bar_pct = min(int(abs(val) / max_push_val * 100), 100)
            st.markdown(
                f'<div class="action-urgent" style="margin-bottom:0.3rem">'
                f'<b>{plain(feat)}</b>'
                f'<div style="background:#e74c3c;height:6px;width:{bar_pct}%;'
                f'border-radius:3px;margin-top:5px"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_protect:
        st.subheader('✅ What is keeping this customer')
        st.caption('These factors are lowering their chance of leaving.')
        for feat, val in top_protect.items():
            if val >= 0:
                continue
            bar_pct = min(int(abs(val) / max_prot_val * 100), 100)
            st.markdown(
                f'<div class="action-ok" style="margin-bottom:0.3rem">'
                f'<b>{plain(feat)}</b>'
                f'<div style="background:#2ecc71;height:6px;width:{bar_pct}%;'
                f'border-radius:3px;margin-top:5px"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader('📋 What to Do')
    st.markdown(action_html, unsafe_allow_html=True)

    with st.expander('📊 See detailed technical chart (for the data team)'):
        st.caption("This chart shows exactly how much each factor shifts this customer's risk score.")
        fig = plot_waterfall(shap_sorted, idx=idx, save=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── PAGE 4 — Campaign Results ─────────────────────────────────────────────────
elif page == '📣  Campaign Results':
    st.title('📣 Retention Campaign Results')
    st.markdown(
        "See how effective a targeted retention offer is compared to doing nothing. "
        "Adjust the settings in the sidebar to match your real campaign data."
    )

    if not model_ready():
        show_deploy_prompt()

    artifact      = get_model()
    model         = artifact['model']
    feature_names = artifact['feature_names']
    threshold     = artifact['threshold']
    df            = get_data()

    with st.sidebar:
        st.subheader('Campaign Settings')
        conversion_rate = st.slider(
            'Offer success rate',
            0.05, 0.60, 0.30, step=0.05,
            help='What % of customers who received the offer stayed as a result? '
                 'Set to match your actual campaign outcome.',
        )
        campaign_size = st.slider(
            'Campaign size (customers)',
            50, 500, 300, step=50,
            help='How many high-risk customers received the retention offer.',
        )

    scored = get_scored(model, feature_names, threshold, len(df))

    try:
        ab = run_ab_test(
            scored,
            true_churn_col  = 'Churn_Numeric',
            conversion_rate = conversion_rate,
            treatment_size  = campaign_size,
            seed            = 42,
        )
    except ValueError as e:
        st.error(f"Not enough high-risk customers for this campaign size. {e}")
        st.stop()

    if ab['significant']:
        st.markdown(
            '<div style="background:#d4edda;border-left:5px solid #2ecc71;'
            'padding:1rem 1.3rem;border-radius:8px;margin-bottom:1rem;">'
            '✅ <b style="font-size:1.1rem">Campaign worked!</b><br>'
            'The targeted offer <b>significantly reduced churn</b> compared to the group '
            'that received nothing. The result is statistically reliable.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#fde0e0;border-left:5px solid #e74c3c;'
            'padding:1rem 1.3rem;border-radius:8px;margin-bottom:1rem;">'
            '⚠️ <b style="font-size:1.1rem">Results are inconclusive</b><br>'
            'The data does not yet show a clear difference. '
            'Try a larger campaign or a more compelling offer.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Customers who got the offer',   f"{ab['n_treatment']:,}")
    c2.metric('Customers kept from leaving',   f"{ab['customers_saved']:,}")
    c3.metric('Churn reduced by',              f"{ab['relative_lift_pct']:.1f}%",
              help='How much lower the leaving rate was in the offer group vs the no-offer group.')
    c4.metric('Monthly revenue recovered',     f"£{ab['estimated_monthly_revenue']:,}")

    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader('Leaving Rate: Offer vs No Offer')
        ctrl_rate = ab['ctrl_churn_rate']
        trt_rate  = ab['trt_churn_rate_post']
        fig, ax   = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(
            ['No offer\n(control)', 'Received offer\n(targeted)'],
            [ctrl_rate, trt_rate],
            color=['#e15759', '#2ecc71'],
            edgecolor='white', width=0.4,
        )
        for bar, val in zip(bars, [ctrl_rate, trt_rate]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.0%}', ha='center', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(ctrl_rate, trt_rate) * 1.4)
        ax.set_ylabel('% of customers who left')
        ax.set_title('Who left: offer group vs no-offer group', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        st.subheader('Campaign Summary')
        roi_annual = ab['estimated_monthly_revenue'] * 12
        st.markdown(f"""
        <div class="insight-box">
            <b>📬 Customers who received the offer:</b> {ab['n_treatment']:,}<br>
            <b>👥 Comparison group (no offer):</b> {ab['n_control']:,}<br>
            <b>📉 Leaving rate — no offer:</b> {ab['ctrl_churn_rate']:.0%}<br>
            <b>📉 Leaving rate — with offer:</b> {ab['trt_churn_rate_post']:.0%}<br>
            <b>🎉 Customers kept:</b> {ab['customers_saved']:,}<br>
            <b>💰 Monthly revenue saved:</b> £{ab['estimated_monthly_revenue']:,}
        </div>
        """, unsafe_allow_html=True)
        st.metric('Projected annual revenue impact', f'£{roi_annual:,}',
                  help='Monthly savings × 12 months')

    with st.expander('📊 See statistical details (for the data team)'):
        stats_show = {
            'Z-statistic'        : f"{ab['z_statistic']:.3f}",
            'p-value'            : f"{ab['p_value']:.4f}",
            'Confidence level'   : ab['confidence_level'],
            'Significant result?': '✅ Yes' if ab['significant'] else '❌ No',
            'Absolute reduction' : f"{ab['absolute_lift']:.1%}",
            'Relative reduction' : f"{ab['relative_lift_pct']:.1f}%",
        }
        st.dataframe(
            pd.DataFrame(stats_show.items(), columns=['Measure', 'Value']),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    st.subheader('📐 How Big Should the Next Campaign Be?')
    st.markdown(
        "This table shows how many customers you need in each group "
        "to reliably detect a given improvement."
    )
    base_rate = df['Churn_Numeric'].mean()
    power_df  = power_analysis_table(base_rate=base_rate).rename(columns={
        'Expected absolute lift'      : 'Target improvement',
        'Control rate'                : 'Current leaving rate',
        'Expected treatment rate'     : 'Target leaving rate (with offer)',
        'Required n per arm'          : 'Min. customers needed per group',
    })
    st.dataframe(power_df, use_container_width=True, hide_index=True)
    st.caption(
        "Example: to confirm a 10‑point reduction in leaving rate, "
        "the table tells you the minimum campaign size required."
    )


# ── PAGE 5 — System Health ────────────────────────────────────────────────────
elif page == '⚙️  System Health':
    st.title('⚙️ Prediction System Health')
    st.markdown(
        "Check whether our customer predictions are still accurate. "
        "Over time, customer behaviour changes and the system may need refreshing."
    )

    df = get_data()

    target       = 'Churn_Numeric'
    feature_cols = [c for c in df.columns if c != target]
    reference    = df[feature_cols].copy()

    with st.sidebar:
        st.subheader('Simulation Controls')
        st.caption('Simulate what happens as customer data patterns change over time.')
        drift_fraction = st.slider(
            'Customer behaviour shift (%)', 0, 100, 30, step=10,
            help='% of data patterns that have shifted since the model was trained.',
        )
        drift_strength = st.slider(
            'Shift severity', 0.0, 2.0, 0.5, step=0.1,
            help='How large the shift is. 1 = one standard deviation.',
        )

    with st.spinner('Checking prediction system…'):
        current  = simulate_drift(reference,
                                  drift_fraction=drift_fraction / 100,
                                  drift_strength=drift_strength)
        psi_df   = compute_psi(reference, current, bins=10)
        summary  = drift_summary(psi_df)

    n_retrain = summary['retrain']
    n_monitor = summary['monitor']
    n_stable  = summary['stable']

    if n_retrain > 0:
        st.markdown(
            f'<div class="action-urgent">'
            f'🔴 <b>Action required — Prediction system needs refreshing</b><br>'
            f'{n_retrain} customer data pattern(s) have changed significantly. '
            f'Ask your data team to retrain the model with recent data.'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif n_monitor > 3:
        st.markdown(
            f'<div class="action-watch">'
            f'🟡 <b>Minor changes detected — Keep an eye on this</b><br>'
            f'{n_monitor} data patterns are drifting slightly. '
            f'Predictions are still reliable but schedule a refresh within the next month.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="action-ok">'
            '🟢 <b>All good — Prediction system is working normally</b><br>'
            'Customer data looks the same as when the system was last trained. No action needed.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric('Data checks passed ✅',      f"{n_stable}",
              help='These patterns match what the model was trained on.')
    c2.metric('Minor changes detected 🟡',  f"{n_monitor}",
              help='Small shifts — worth monitoring.')
    c3.metric('Significant changes 🔴',     f"{n_retrain}",
              help='These have shifted enough to affect prediction accuracy.')

    if model_ready():
        card = model_card('rf_churn_model')
        st.divider()
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader('About the Prediction System')
            st.markdown(f"""
            <div class="insight-box">
                <b>Last refreshed:</b> {(card.get('trained_at') or '')[:10] or '—'}<br>
                <b>Customers used in training:</b> {len(df):,}<br>
                <b>Customer signals analysed:</b> {card.get('n_features','—')} attributes<br>
                <b>Alert level:</b> Customers above
                {float(card.get('threshold', 0.36)):.0%} risk are flagged for action
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.subheader('When Should You Refresh?')
            st.markdown("""
            <div class="insight-box">
                ✅ <b>Every quarter</b> — routine refresh to stay accurate<br>
                🟡 <b>After a price change</b> — customer behaviour may shift<br>
                🟡 <b>After a major product change</b> — new patterns emerge<br>
                🔴 <b>When the red alert triggers</b> — refresh immediately
            </div>
            """, unsafe_allow_html=True)

    drifted_feats = psi_df[psi_df['status'] != 'Stable']['feature'].tolist()
    if drifted_feats:
        st.divider()
        st.subheader('Changed Customer Patterns')
        st.caption('These customer attributes look different from when the system was last trained.')
        selected_feat = st.selectbox(
            'Select an attribute to compare',
            options=drifted_feats,
            format_func=plain,
        )
        fig = plot_feature_distribution(reference, current, selected_feat)
        ax  = fig.axes[0]
        ax.set_xlabel(plain(selected_feat))
        ax.legend(['When model was trained', 'Current customer data'])
        ax.set_title(f'How "{plain(selected_feat)}" has changed over time', fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander('📊 See detailed technical data (for the data team)'):
        st.dataframe(
            psi_df.rename(columns={
                'feature': 'Customer attribute',
                'psi'    : 'Change score',
                'status' : 'Status',
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption('Change score: < 0.10 = Stable  |  0.10–0.25 = Monitor  |  > 0.25 = Needs refresh')
