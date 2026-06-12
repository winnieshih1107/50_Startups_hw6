"""
Startup Profit Predictor — Streamlit App
CRISP-DM Step 4: Feature Selector Comparison via Linear Regression
Data: 50 Startups (github.com/winnieshih1107/50_Startups_hw6)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE,
    SequentialFeatureSelector,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── global background ── */
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; }

/* ── global font size boost ── */
html, body, [class*="css"] { font-size: 17px; }
p, li, span { font-size: 17px; }

/* ── sidebar text ── */
[data-testid="stSidebar"] label  { color: #c9d1d9 !important; font-size: 16px !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9 !important; font-size: 16px !important; }
[data-testid="stSidebar"] h2 { color: #e6edf3 !important; font-size: 20px !important; }
[data-testid="stSidebar"] .stCaption { color: #9bafc4 !important; font-size: 14px !important; }

/* ── KPI card ── */
.kpi-card {
    background: #182236;
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 20px 22px 16px;
    min-height: 125px;
}
.kpi-label {
    color: #7eb8da;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1.0px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    color: #ffffff;
    font-size: 42px;
    font-weight: 700;
    line-height: 1.15;
}
.kpi-highlight {
    color: #f0883e;
    font-size: 32px;
    font-weight: 700;
    line-height: 1.4;
}
.kpi-delta-up   { color: #56d364; font-size: 15px; margin-top: 5px; }
.kpi-delta-down { color: #f85149; font-size: 15px; margin-top: 5px; }

/* ── tab bar ── */
[data-testid="stTabs"] button {
    color: #8b949e !important;
    font-weight: 600;
    font-size: 17px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 3px solid #7b68ee;
}

/* ── section header ── */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 8px;
}

/* ── metric widget ── */
[data-testid="stMetricValue"] { font-size: 32px !important; color: #ffffff !important; }
[data-testid="stMetricLabel"] { font-size: 16px !important; color: #c9d1d9 !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { font-size: 15px; }

/* ── selectbox / slider labels ── */
[data-testid="stWidgetLabel"] { font-size: 16px !important; color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
DATA_URL = (
    "https://raw.githubusercontent.com/winnieshih1107/"
    "50_Startups_hw6/main/50_Startups.csv"
)

@st.cache_data(show_spinner="載入資料…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)

@st.cache_data(show_spinner=False)
def prepare(test_pct: int, seed: int):
    df = load_data()
    df_ohe = pd.get_dummies(df, columns=["State"], drop_first=True)
    for c in df_ohe.columns:
        if df_ohe[c].dtype == bool:
            df_ohe[c] = df_ohe[c].astype(int)
    X = df_ohe.drop("Profit", axis=1).astype(float)
    y = df_ohe["Profit"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_pct / 100, random_state=seed
    )
    return X, y, X_tr, X_te, y_tr, y_te, list(X.columns)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Split & Selection Settings")
test_pct = st.sidebar.slider("Test Split Size (%)", 10, 40, 20, 5)
seed     = int(st.sidebar.number_input("Random Seed", value=42, min_value=0, step=1))
k_feat   = st.sidebar.slider("Target Number of Features (k)", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Model Architecture Selection")
st.sidebar.caption("(Integrates with future regression modeling lessons)")
MODEL_CHOICES = [
    "Linear Regression", "Ridge", "Lasso",
    "Random Forest", "Gradient Boosting",
]
model_name = st.sidebar.selectbox("Choose Evaluator Model:", MODEL_CHOICES)

# ══════════════════════════════════════════════════════════════════════════════
# Feature-selection + evaluation (cached by params)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="執行 10 種特徵選擇演算法…")
def run_all(test_pct: int, seed: int, k: int, model_name: str):
    X, y, X_tr, X_te, y_tr, y_te, feat_names = prepare(test_pct, seed)
    n = len(feat_names)

    sc     = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)
    X_sc   = StandardScaler().fit_transform(X)

    def top_k(scores):
        idx = np.argsort(scores)[::-1][:k]
        m = np.zeros(n, dtype=int); m[idx] = 1
        return m

    # ── 10 methods ────────────────────────────────────────────────────────────
    masks = {}

    # 1 Pearson correlation
    pearson = np.array([abs(np.corrcoef(X_tr.iloc[:, i], y_tr)[0, 1])
                        for i in range(n)])
    masks["Pearson Corr"] = top_k(pearson)

    # 2 Spearman correlation
    spear = np.array([abs(spearmanr(X_tr.iloc[:, i], y_tr)[0]) for i in range(n)])
    masks["Spearman Corr"] = top_k(spear)

    # 3 SelectKBest — F-regression
    skb_f = SelectKBest(f_regression, k=k).fit(X_tr, y_tr)
    masks["F-Regression"] = skb_f.get_support().astype(int)

    # 4 SelectKBest — Mutual Information
    skb_mi = SelectKBest(mutual_info_regression, k=k,
                         ).fit(X_tr, y_tr)
    masks["Mutual Info"] = skb_mi.get_support().astype(int)

    # 5 Sequential Forward Selection
    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select=k,
        direction="forward", cv=3,
    ).fit(X_tr, y_tr)
    masks["Seq. Forward"] = sfs.get_support().astype(int)

    # 6 RFE — Linear Regression
    rfe_lr = RFE(LinearRegression(), n_features_to_select=k).fit(Xtr_sc, y_tr)
    masks["RFE-Linear"] = rfe_lr.support_.astype(int)

    # 7 RFE — Random Forest
    rfe_rf = RFE(
        RandomForestRegressor(n_estimators=50, random_state=seed),
        n_features_to_select=k,
    ).fit(X_tr, y_tr)
    masks["RFE-Forest"] = rfe_rf.support_.astype(int)

    # 8 LassoCV
    lasso = LassoCV(cv=5, max_iter=10000, random_state=seed).fit(Xtr_sc, y_tr)
    lasso_coef = np.abs(lasso.coef_)
    if lasso_coef.sum() == 0:
        lasso_coef[np.argmax(np.abs(lasso.coef_))] = 1.0
    masks["LassoCV"] = top_k(lasso_coef)

    # 9 Random Forest Importance
    rf = RandomForestRegressor(n_estimators=200, random_state=seed).fit(X_tr, y_tr)
    masks["Random Forest"] = top_k(rf.feature_importances_)

    # 10 Gradient Boosting Importance
    gb = GradientBoostingRegressor(n_estimators=100, random_state=seed).fit(X_tr, y_tr)
    masks["Grad Boosting"] = top_k(gb.feature_importances_)

    # ── Consensus ─────────────────────────────────────────────────────────────
    sel_df = pd.DataFrame(masks, index=feat_names)
    sel_df["Votes"] = sel_df.sum(axis=1)

    # ── Evaluate each method ──────────────────────────────────────────────────
    def make_model():
        if model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Ridge":
            return RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5)
        elif model_name == "Lasso":
            return LassoCV(cv=5, max_iter=10000, random_state=seed)
        elif model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=seed)
        else:
            return GradientBoostingRegressor(n_estimators=100, random_state=seed)

    eval_rows = []
    for method, mask in masks.items():
        cols = [f for f, v in zip(feat_names, mask) if v]
        if not cols:
            continue
        scaler = StandardScaler()
        Xtr_m = scaler.fit_transform(X_tr[cols])
        Xte_m = scaler.transform(X_te[cols])
        mdl   = make_model().fit(Xtr_m, y_tr)
        y_hat = mdl.predict(Xte_m)
        eval_rows.append({
            "Method":        method,
            "Features":      cols,
            "Num Features":  len(cols),
            "Test R²":       round(float(r2_score(y_te, y_hat)), 4),
            "Test RMSE":     round(float(np.sqrt(mean_squared_error(y_te, y_hat))), 2),
            "Test MAE":      round(float(mean_absolute_error(y_te, y_hat)), 2),
            "scaler":        scaler,
            "model":         mdl,
        })

    eval_df = pd.DataFrame(eval_rows)
    return sel_df, eval_df, feat_names, X_tr, X_te, y_tr, y_te

sel_df, eval_df, feat_names, X_tr, X_te, y_tr, y_te = run_all(
    test_pct, seed, k_feat, model_name
)

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="color:#e6edf3;font-size:38px;font-weight:800;margin-bottom:2px;">'
    '📈 Startup Profit Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:#9bafc4;font-size:16px;margin-top:0;">'
    f'CRISP-DM Step 4: Comparing Feature Selectors Evaluated via '
    f'<b style="color:#c9d1d9">{model_name}</b></p>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI Row
# ══════════════════════════════════════════════════════════════════════════════
best_r2_row   = eval_df.loc[eval_df["Test R²"].idxmax()]
best_rmse_row = eval_df.loc[eval_df["Test RMSE"].idxmin()]
avg_r2        = eval_df["Test R²"].mean()
avg_rmse      = eval_df["Test RMSE"].mean()
r2_diff       = best_r2_row["Test R²"] - avg_r2
rmse_diff     = best_rmse_row["Test RMSE"] - avg_rmse

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Best Selection Strategy</div>
        <div class="kpi-highlight">{best_r2_row["Method"]}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Max {model_name} R²</div>
        <div class="kpi-value">{best_r2_row["Test R²"]:.4f}</div>
        <div class="kpi-delta-up">↑ {r2_diff:+.4f} vs Avg</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Lowest Test RMSE</div>
        <div class="kpi-value">${best_rmse_row["Test RMSE"]:,.2f}</div>
        <div class="kpi-delta-down">↓ ${rmse_diff:,.2f} vs Avg</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Optimal Feature Count</div>
        <div class="kpi-value">{best_r2_row["Num Features"]}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Performance Analysis — two charts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Performance Analysis</div>',
            unsafe_allow_html=True)

CHART_BG = "#161b22"
GRID_COL = "#2d3748"
TEXT_COL = "#e6edf3"

CHART_LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=TEXT_COL, size=16),
    margin=dict(l=14, r=40, t=56, b=14),
    height=420,
    xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
               tickfont=dict(size=15), title_font=dict(size=16)),
    yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
               tickfont=dict(size=15), title_font=dict(size=16)),
    title_font=dict(size=17),
)

col_l, col_r = st.columns(2)

with col_l:
    # R² bar chart — sorted descending
    ev = eval_df.sort_values("Test R²", ascending=True)
    bar_colors = [
        "#7b68ee" if m == best_r2_row["Method"] else "#4a5568"
        for m in ev["Method"]
    ]
    fig_r2 = go.Figure(go.Bar(
        y=ev["Method"],
        x=ev["Test R²"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"  {v:.4f}" for v in ev["Test R²"]],
        textposition="outside",
        textfont=dict(color=TEXT_COL, size=15),
    ))
    fig_r2.add_vline(x=0.9, line=dict(color="#ff4444", dash="dash", width=3))
    fig_r2.add_annotation(
        x=0.9, y=1.0,
        xref="x", yref="paper",
        text="Target = 0.90",
        showarrow=True, arrowhead=2, arrowwidth=2,
        arrowcolor="#ff4444", ax=30, ay=-30,
        font=dict(color="#ffffff", size=14, family="Arial"),
        bgcolor="#ff4444", borderpad=5, borderwidth=0,
        xanchor="left",
    )
    fig_r2.update_layout(
        title=dict(text=f"Test R² by Selection Method", font=dict(size=13)),
        xaxis=dict(range=[max(0, ev["Test R²"].min() - 0.15), 1.08],
                   gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL),
        **{k: v for k, v in CHART_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    st.plotly_chart(fig_r2, use_container_width=True)

with col_r:
    # Feature Consensus bar chart
    votes_sorted = sel_df["Votes"].sort_values(ascending=True)
    n_methods = len(eval_df)
    vote_colors = [
        "#7b68ee" if v >= n_methods * 0.8 else
        "#5c6bc0" if v >= n_methods * 0.5 else
        "#3a4a6b"
        for v in votes_sorted.values
    ]
    fig_votes = go.Figure(go.Bar(
        y=votes_sorted.index,
        x=votes_sorted.values,
        orientation="h",
        marker=dict(color=vote_colors, line=dict(width=0)),
        text=votes_sorted.values,
        textposition="outside",
        textfont=dict(color=TEXT_COL, size=15),
    ))
    fig_votes.update_layout(
        title=dict(text=f"Feature Selection Consensus (Out of {n_methods} Algorithms)",
                   font=dict(size=13)),
        xaxis=dict(title="Number of Algorithms Selecting Feature",
                   range=[0, n_methods + 1],
                   gridcolor=GRID_COL, zerolinecolor=GRID_COL, dtick=2),
        yaxis=dict(gridcolor=GRID_COL),
        **{k: v for k, v in CHART_LAYOUT.items()
           if k not in ("xaxis", "yaxis")},
    )
    st.plotly_chart(fig_votes, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📋  Complete Comparison Table",
    "🔍  Detailed Method Inspector",
    "💰  Live Profit Simulator",
])

# ─── Tab 1 ───────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"### Performance Metrics Comparison (Evaluated via {model_name})"
    )

    # Display table
    display = eval_df[["Method", "Features", "Num Features",
                        "Test R²", "Test RMSE", "Test MAE"]].copy()
    display["Features"] = display["Features"].apply(lambda lst: ", ".join(lst))
    display["Test RMSE"] = display["Test RMSE"].apply(lambda v: f"${v:,.2f}")
    display["Test MAE"]  = display["Test MAE"].apply(lambda v: f"${v:,.2f}")

    # Highlight best row
    def highlight_best(row):
        is_best = row["Method"] == best_r2_row["Method"]
        return ["background-color: #1f3a1f; color: #3fb950" if is_best
                else "" for _ in row]

    st.dataframe(
        display.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### Feature Selection Matrix (1 = Selected)")

    sel_matrix = sel_df.drop(columns=["Votes"]).T
    sel_matrix.index.name = "Method"
    st.dataframe(
        sel_matrix.style.map(
            lambda v: "background-color:#1f3a1f;color:#3fb950" if v == 1
            else "background-color:#3a1f1f;color:#f85149"
        ),
        use_container_width=True,
    )

# ─── Tab 2 ───────────────────────────────────────────────────────────────────
with tab2:
    chosen = st.selectbox("Select Feature Selection Method:",
                          eval_df["Method"].tolist(), key="inspector_select")
    row = eval_df[eval_df["Method"] == chosen].iloc[0]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Test R²",   f"{row['Test R²']:.4f}")
    mc2.metric("Test RMSE", f"${row['Test RMSE']:,.2f}")
    mc3.metric("Test MAE",  f"${row['Test MAE']:,.2f}")

    st.markdown(
        f"**Selected features ({row['Num Features']}):** "
        f"`{'`,  `'.join(row['Features'])}`"
    )

    # Re-predict for plots
    feats   = row["Features"]
    scaler  = row["scaler"]
    mdl_obj = row["model"]
    y_hat   = mdl_obj.predict(scaler.transform(X_te[feats]))

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        lim = [min(y_te.min(), y_hat.min()) * 0.95,
               max(y_te.max(), y_hat.max()) * 1.05]
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_te.values, y=y_hat, mode="markers",
            marker=dict(color="#7b68ee", size=9, opacity=0.85,
                        line=dict(color="#0d1117", width=1)),
            name="Predictions",
        ))
        fig_avp.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="#f85149", dash="dash", width=1.5),
            name="Perfect Fit",
        ))
        fig_avp.update_layout(
            title=f"Actual vs Predicted — {chosen}",
            xaxis_title="Actual Profit ($)", yaxis_title="Predicted Profit ($)",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with chart_col2:
        residuals = y_te.values - y_hat
        fig_res = go.Figure(go.Histogram(
            x=residuals, nbinsx=8,
            marker=dict(color="#5c6bc0", line=dict(color="#0d1117", width=1)),
        ))
        fig_res.add_vline(x=0, line=dict(color="#f85149", dash="dash"))
        fig_res.update_layout(
            title="Residual Distribution",
            xaxis_title="Residual ($)", yaxis_title="Count",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Coefficient bar chart (if linear model)
    if hasattr(mdl_obj, "coef_"):
        coefs = mdl_obj.coef_
        fig_coef = go.Figure(go.Bar(
            y=feats, x=coefs, orientation="h",
            marker=dict(
                color=["#f85149" if c < 0 else "#7b68ee" for c in coefs],
                line=dict(width=0),
            ),
            text=[f"{c:,.1f}" for c in coefs],
            textposition="outside",
            textfont=dict(color=TEXT_COL, size=15),
        ))
        fig_coef.add_vline(x=0, line=dict(color=TEXT_COL, width=0.8))
        fig_coef.update_layout(
            title="Feature Coefficients (Standardised Input)",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

# ─── Tab 3 ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 💰 Live Profit Simulator")
    st.markdown(
        f"Use the **{best_r2_row['Method']}** feature set "
        f"(best R² = `{best_r2_row['Test R²']:.4f}`) to predict profit."
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        rd_input    = st.number_input(
            "R&D Spend ($)", 0.0, 500_000.0, 150_000.0, 5_000.0,
            format="%.0f", key="sim_rd"
        )
    with s2:
        admin_input = st.number_input(
            "Administration ($)", 0.0, 500_000.0, 120_000.0, 5_000.0,
            format="%.0f", key="sim_admin"
        )
    with s3:
        mktg_input  = st.number_input(
            "Marketing Spend ($)", 0.0, 600_000.0, 300_000.0, 5_000.0,
            format="%.0f", key="sim_mktg"
        )

    state_input = st.selectbox("State", ["California", "Florida", "New York"],
                               key="sim_state")

    if st.button("🔮  Predict Profit", type="primary"):
        best_feats  = best_r2_row["Features"]
        best_scaler = best_r2_row["scaler"]
        best_mdl    = best_r2_row["model"]

        raw = {
            "R&D Spend":       rd_input,
            "Administration":  admin_input,
            "Marketing Spend": mktg_input,
            "State_Florida":   1.0 if state_input == "Florida"  else 0.0,
            "State_New York":  1.0 if state_input == "New York" else 0.0,
        }
        input_row    = pd.DataFrame([{f: raw.get(f, 0.0) for f in best_feats}])
        input_scaled = best_scaler.transform(input_row)
        prediction   = float(best_mdl.predict(input_scaled)[0])

        st.success(f"### Predicted Annual Profit: **${prediction:,.2f}**")

        total = rd_input + admin_input + mktg_input
        if total > 0:
            spend_df = pd.DataFrame({
                "Category": ["R&D Spend", "Administration", "Marketing Spend"],
                "Amount ($)": [rd_input, admin_input, mktg_input],
                "% of Total": [
                    f"{rd_input/total*100:.1f}%",
                    f"{admin_input/total*100:.1f}%",
                    f"{mktg_input/total*100:.1f}%",
                ],
            })
            st.table(spend_df)

        # Context vs training range
        df_raw = load_data()
        st.info(
            f"📊 Training profit range: "
            f"${df_raw['Profit'].min():,.0f} – ${df_raw['Profit'].max():,.0f}  "
            f"|  Training R&D range: $0 – ${df_raw['R&D Spend'].max():,.0f}"
        )
