"""
Feature Selection Comparison — 50 Startups
5 algorithms × k=1..5 features: RMSE, R², selected features at each step.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

SEED  = 42
PATH  = r"D:\wi\50_Startups.csv"
SAVE  = r"D:\wi\feature_selection_figure.png"
CV    = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
df  = pd.read_csv(PATH)
df  = pd.get_dummies(df, columns=["State"])          # keep all 3 dummies
df.columns = [c.replace("State_", "") for c in df.columns]

FEATURES = ["R&D Spend", "Administration", "Marketing Spend",
            "New York", "Florida", "California"]
SHORT    = {"R&D Spend": "R&D Spend",
            "Administration": "Administration",
            "Marketing Spend": "Marketing Spend",
            "New York": "State: New York",
            "Florida": "State: Florida",
            "California": "State: California"}

X = df[FEATURES].astype(float)
y = df["Profit"]
K_MAX = len(FEATURES)   # 6 features

def cv_metrics(features):
    """Return (mean CV RMSE, mean CV R²) for LinearRegression on given features."""
    lr     = LinearRegression()
    r2s    = cross_val_score(lr, X[features], y, cv=CV, scoring="r2")
    rmses  = np.sqrt(-cross_val_score(lr, X[features], y, cv=CV,
                                       scoring="neg_mean_squared_error"))
    return rmses.mean(), r2s.mean()

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: Sequential Forward Selection (SFS)
# Greedy: add the single best feature at each step.
# ─────────────────────────────────────────────────────────────────────────────
def run_sfs():
    selected, remaining = [], list(FEATURES)
    rows = []
    for k in range(1, K_MAX + 1):
        best_r2, best_feat = -np.inf, None
        for feat in remaining:
            _, r2 = cv_metrics(selected + [feat])
            if r2 > best_r2:
                best_r2, best_feat = r2, feat
        selected.append(best_feat)
        remaining.remove(best_feat)
        rmse, r2 = cv_metrics(selected)
        rows.append(dict(k=k, features=selected.copy(), rmse=rmse, r2=r2))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: Recursive Feature Elimination (RFE)
# Start with all features; remove the least important one by one.
# ─────────────────────────────────────────────────────────────────────────────
def run_rfe():
    rows = []
    for k in range(1, K_MAX + 1):
        sel  = RFE(LinearRegression(), n_features_to_select=k)
        sel.fit(X, y)
        feats = [FEATURES[i] for i, s in enumerate(sel.support_) if s]
        rmse, r2 = cv_metrics(feats)
        rows.append(dict(k=k, features=feats, rmse=rmse, r2=r2))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: SelectKBest — F-regression (ANOVA F-test)
# Rank all features by F-statistic; take top k.
# ─────────────────────────────────────────────────────────────────────────────
def run_skb():
    skb_all = SelectKBest(f_regression, k="all").fit(X, y)
    ranking  = np.argsort(skb_all.scores_)[::-1]   # highest F first
    rows = []
    for k in range(1, K_MAX + 1):
        feats = [FEATURES[i] for i in ranking[:k]]
        rmse, r2 = cv_metrics(feats)
        rows.append(dict(k=k, features=feats, rmse=rmse, r2=r2))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4: Lasso (L1 Regularisation) — LassoCV coefficient ranking
# Features with larger |coef| are ranked higher; zero-coef features dropped.
# ─────────────────────────────────────────────────────────────────────────────
def run_lasso():
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    las = LassoCV(cv=5, max_iter=10000, random_state=SEED).fit(Xs, y)
    ranking = np.argsort(np.abs(las.coef_))[::-1]
    rows = []
    for k in range(1, K_MAX + 1):
        feats = [FEATURES[i] for i in ranking[:k]]
        rmse, r2 = cv_metrics(feats)
        rows.append(dict(k=k, features=feats, rmse=rmse, r2=r2))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 5: Random Forest Feature Importance
# Train RF on all features; rank by Gini importance.
# ─────────────────────────────────────────────────────────────────────────────
def run_rf():
    rf      = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(X, y)
    ranking = np.argsort(rf.feature_importances_)[::-1]
    rows = []
    for k in range(1, K_MAX + 1):
        feats = [FEATURES[i] for i in ranking[:k]]
        rmse, r2 = cv_metrics(feats)
        rows.append(dict(k=k, features=feats, rmse=rmse, r2=r2))
    return rows

# ── Run all methods ───────────────────────────────────────────────────────────
print("Running 5 feature selection methods …")
methods = {
    "SFS\n(Sequential Forward Selection)": run_sfs(),
    "RFE\n(Recursive Feature Elimination)": run_rfe(),
    "SelectKBest\n(F-regression)": run_skb(),
    "Lasso\n(L1 Coefficient Rank)": run_lasso(),
    "Random Forest\n(Gini Importance)": run_rf(),
}

for name, rows in methods.items():
    label = name.replace("\n", " ")
    print(f"\n  {label}")
    print(f"  {'k':>2}  {'RMSE':>10}  {'R²':>8}  Features")
    for r in rows:
        feats_str = ", ".join(SHORT[f] for f in r["features"])
        print(f"  {r['k']:>2}  ${r['rmse']:>9,.0f}  {r['r2']:>8.4f}  [{feats_str}]")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# Layout: 2 line charts (top) + 5 method tables (bottom)
# ─────────────────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "SFS\n(Sequential Forward Selection)":    "#4C72B0",
    "RFE\n(Recursive Feature Elimination)":   "#DD8452",
    "SelectKBest\n(F-regression)":            "#55A868",
    "Lasso\n(L1 Coefficient Rank)":           "#C44E52",
    "Random Forest\n(Gini Importance)":       "#9467BD",
}

fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(4, 3, figure=fig,
                        height_ratios=[3, 3, 4, 4],
                        hspace=0.55, wspace=0.35)

ks = list(range(1, K_MAX + 1))

# ── Top-left: RMSE by # features ─────────────────────────────────────────────
ax_rmse = fig.add_subplot(gs[0, :])
for name, rows in methods.items():
    rmses = [r["rmse"] for r in rows]
    ax_rmse.plot(ks, rmses, marker="o", linewidth=2, markersize=7,
                 color=METHOD_COLORS[name], label=name.replace("\n", " "))
ax_rmse.set_xlabel("Number of Features", fontsize=11)
ax_rmse.set_ylabel("RMSE  ($)", fontsize=11)
ax_rmse.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax_rmse.set_title("RMSE by Number of Features — 5 Selection Methods (5-Fold CV)",
                  fontsize=13, fontweight="bold", pad=10)
ax_rmse.set_xticks(ks)
ax_rmse.legend(fontsize=9, ncol=5, loc="upper left",
               framealpha=0.9, edgecolor="#DDDDDD")
ax_rmse.grid(True, alpha=0.4)
ax_rmse.set_facecolor("#FAFAFA")

# ── Second row: R² by # features ─────────────────────────────────────────────
ax_r2 = fig.add_subplot(gs[1, :])
for name, rows in methods.items():
    r2s = [r["r2"] for r in rows]
    ax_r2.plot(ks, r2s, marker="o", linewidth=2, markersize=7,
               color=METHOD_COLORS[name], label=name.replace("\n", " "))
ax_r2.set_xlabel("Number of Features", fontsize=11)
ax_r2.set_ylabel("R-squared", fontsize=11)
ax_r2.set_title("R² by Number of Features — 5 Selection Methods (5-Fold CV)",
                fontsize=13, fontweight="bold", pad=10)
ax_r2.set_xticks(ks)
ax_r2.legend(fontsize=9, ncol=5, loc="lower left",
             framealpha=0.9, edgecolor="#DDDDDD")
ax_r2.axhline(0.90, color="red", linestyle="--", linewidth=1.2, alpha=0.6,
              label="Target 0.90")
ax_r2.grid(True, alpha=0.4)
ax_r2.set_facecolor("#FAFAFA")

# ── Bottom 6 cells: one table per method (3+2 layout, last cell = legend) ────
table_positions = [
    gs[2, 0], gs[2, 1], gs[2, 2],
    gs[3, 0], gs[3, 1],
]

for (name, rows), pos in zip(methods.items(), table_positions):
    ax_t = fig.add_subplot(pos)
    ax_t.axis("off")
    color = METHOD_COLORS[name]

    # Table data
    col_labels = ["k", "Selected Features", "RMSE", "R²"]
    cell_text  = []
    for r in rows:
        feats_disp = "\n".join(f"• {SHORT[f]}" for f in r["features"])
        cell_text.append([
            str(r["k"]),
            feats_disp,
            f"${r['rmse']:,.0f}",
            f"{r['r2']:.4f}",
        ])

    # Column widths (relative)
    col_widths = [0.06, 0.60, 0.17, 0.12]

    tbl = ax_t.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 2.6)

    # Style header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor(color)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        tbl[(0, j)].set_edgecolor("#CCCCCC")

    # Style rows: highlight best k (lowest RMSE)
    best_k = min(range(len(rows)), key=lambda i: rows[i]["rmse"])
    for i, row in enumerate(rows):
        for j in range(len(col_labels)):
            cell = tbl[(i + 1, j)]
            cell.set_edgecolor("#DDDDDD")
            if i == best_k:
                cell.set_facecolor("#EAF4FB")
            elif i % 2 == 0:
                cell.set_facecolor("#F9F9F9")
            else:
                cell.set_facecolor("white")

    short_name = name.split("\n")[0]
    algo_desc  = name.split("\n")[1] if "\n" in name else ""
    ax_t.set_title(f"{short_name}\n{algo_desc}",
                   fontsize=10, fontweight="bold", color=color, pad=8)

# ── Last cell (gs[3,2]): Algorithm summary ────────────────────────────────────
ax_leg = fig.add_subplot(gs[3, 2])
ax_leg.axis("off")
algo_info = [
    ("SFS", "#4C72B0",
     "Sequential Forward Selection",
     "Greedy: adds one feature at a time.\n"
     "At each step, tries all remaining\n"
     "features and keeps the one that\n"
     "maximises CV R²."),
    ("RFE", "#DD8452",
     "Recursive Feature Elimination",
     "Starts with all features; trains a\n"
     "model and removes the feature with\n"
     "the smallest coefficient weight,\n"
     "repeating until k remain."),
    ("SKB", "#55A868",
     "SelectKBest (F-regression)",
     "Computes a univariate F-statistic\n"
     "for each feature vs. target.\n"
     "Selects top k features by score.\n"
     "Fast but ignores interactions."),
    ("Lasso", "#C44E52",
     "Lasso L1 Regularisation",
     "Adds L1 penalty: shrinks small\n"
     "coefficients to exactly zero.\n"
     "Features ranked by |coefficient|\n"
     "at the CV-optimal alpha."),
    ("RF", "#9467BD",
     "Random Forest Importance",
     "Trains 200 trees; computes mean\n"
     "decrease in impurity (Gini) per\n"
     "feature. Non-linear; captures\n"
     "interaction effects."),
]

ax_leg.set_title("Top 5 Feature Selection\nAlgorithms", fontsize=10,
                 fontweight="bold", pad=8, color="#333333")
y_pos = 0.97
for short, color, full, desc in algo_info:
    ax_leg.text(0.04, y_pos, f"● {short}  —  {full}",
                transform=ax_leg.transAxes, fontsize=8.5, fontweight="bold",
                color=color, va="top")
    y_pos -= 0.055
    for line in desc.split("\n"):
        ax_leg.text(0.07, y_pos, line,
                    transform=ax_leg.transAxes, fontsize=7.5,
                    color="#444444", va="top")
        y_pos -= 0.042
    y_pos -= 0.012

ax_leg.set_facecolor("#FAFAFA")
for spine in ax_leg.spines.values():
    spine.set_edgecolor("#DDDDDD")

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "50 Startups — Feature Selection Comparison\n"
    "6 candidate features  ·  5 algorithms  ·  5-Fold CV  ·  LinearRegression estimator",
    fontsize=14, fontweight="bold", y=0.995
)

plt.savefig(SAVE, dpi=130, bbox_inches="tight", facecolor="white")
print(f"\n[Saved] {SAVE}")
