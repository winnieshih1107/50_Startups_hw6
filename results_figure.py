"""
Results figure — built from captured CV output + raw CSV.
No sklearn import: avoids the MemoryError on scipy._resampling.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

SAVE = r"D:\wi\results_figure.png"
PATH = r"D:\wi\50_Startups.csv"

# ── CV results captured from Phase 4 output ──────────────────────────────────
# Source: bcbcy5gmu.output (RepeatedKFold 5×10, baseline feature set)
cv_data = {
    "LinearRegression": dict(r2=0.9315, std=0.0381, rmse=9536),
    "Ridge":            dict(r2=0.9309, std=0.0384, rmse=9578),
    "Lasso":            dict(r2=0.9317, std=0.0380, rmse=9527),
    "ElasticNet":       dict(r2=0.9313, std=0.0381, rmse=9549),
    "RandomForest":     dict(r2=0.9285, std=0.0455, rmse=9923),
    "GradientBoosting": dict(r2=0.9018, std=0.0508, rmse=11495),
}

# Old plain KFold(5) scores from the original script (5 scores, same data)
old_kfold = [0.8931, -0.8112, -0.4193, -0.7012, 0.4304]

# ── Load raw data for scatter / correlation ───────────────────────────────────
df = pd.read_csv(PATH)
numeric = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]

# Simple OLS by hand: Profit ~ R&D Spend  (best single predictor)
x  = df["R&D Spend"].values
y  = df["Profit"].values
m, b = np.polyfit(x, y, 1)
y_hat_ols = m * x + b
ss_res = np.sum((y - y_hat_ols) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2_rd  = 1 - ss_res / ss_tot

# Feature engineering stats
df["Total_Spend"] = df["R&D Spend"] + df["Administration"] + df["Marketing Spend"]
df["RD_Ratio"]    = df["R&D Spend"] / df["Total_Spend"]

# ── Colour palette ────────────────────────────────────────────────────────────
C_BASE  = "#4C72B0"
C_BEST  = "#2ECC71"
C_RED   = "#C44E52"
C_AMBER = "#DD8452"
C_GREY  = "#BDC3C7"
STATE_C = {"New York": "#4C72B0", "California": "#DD8452", "Florida": "#55A868"}

models  = list(cv_data.keys())
r2s     = [cv_data[m]["r2"]  for m in models]
stds    = [cv_data[m]["std"] for m in models]
rmses   = [cv_data[m]["rmse"] for m in models]
best_m  = models[np.argmax(r2s)]          # Lasso
bar_c   = [C_BEST if m == best_m else C_BASE for m in models]

# ── Figure 3×3 ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 15))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# ── ① CV R² bar chart ─────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
xi  = np.arange(len(models))
bars = ax0.bar(xi, r2s, color=bar_c, edgecolor="white", alpha=0.88,
               yerr=stds, capsize=4, error_kw=dict(ecolor="grey", elinewidth=1))
ax0.axhline(0.90, color=C_RED, linestyle="--", lw=1.5, label="Target R²=0.90")
ax0.set_xticks(xi); ax0.set_xticklabels(models, rotation=22, ha="right", fontsize=8)
ax0.set_ylabel("CV R² (mean ± std)")
ax0.set_ylim(0.84, 1.02)
ax0.set_title("① CV R²  per Model\n(RepeatedKFold 5×10 = 50 scores)",
              fontsize=9, fontweight="bold")
ax0.legend(fontsize=8)
for bar, val, sd in zip(bars, r2s, stds):
    ax0.text(bar.get_x() + bar.get_width()/2, val + sd + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7.5,
             fontweight="bold" if val == max(r2s) else "normal")

# ── ② CV R² std (stability) ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
std_c = [C_RED if s > 0.10 else C_BEST for s in stds]
ax1.bar(xi, stds, color=std_c, edgecolor="white", alpha=0.88)
ax1.axhline(0.10, color=C_RED, linestyle="--", lw=1.5, label="Threshold 0.10")
ax1.set_xticks(xi); ax1.set_xticklabels(models, rotation=22, ha="right", fontsize=8)
ax1.set_ylabel("CV R² std  (lower = stable)")
ax1.set_title("② CV Stability\nAll models stable (std < 0.06) ✓",
              fontsize=9, fontweight="bold")
ax1.legend(fontsize=8)
for i, (s, c) in enumerate(zip(stds, std_c)):
    ax1.text(i, s + 0.001, f"{s:.3f}", ha="center", va="bottom", fontsize=7.5)

# ── ③ CV RMSE ────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
rmse_c = [C_BEST if r == min(rmses) else C_BASE for r in rmses]
ax2.bar(xi, rmses, color=rmse_c, edgecolor="white", alpha=0.88)
ax2.axhline(15000, color=C_RED, linestyle="--", lw=1.5, label="Target $15,000")
ax2.set_xticks(xi); ax2.set_xticklabels(models, rotation=22, ha="right", fontsize=8)
ax2.set_ylabel("CV RMSE ($)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax2.set_title("③ CV RMSE per Model\nAll below $15k target ✓",
              fontsize=9, fontweight="bold")
ax2.legend(fontsize=8)
for i, r in enumerate(rmses):
    ax2.text(i, r + 80, f"${r:,.0f}", ha="center", va="bottom", fontsize=7.5)

# ── ④ Old KFold vs New RepeatedKFold ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
xk = np.arange(5)
new_r2  = r2s[:5]
new_std = stds[:5]
ax3.bar(xk - 0.2, old_kfold, 0.38, label="Old: KFold(5)  — 5 scores",
        color=C_RED, alpha=0.75, edgecolor="white")
ax3.bar(xk + 0.2, new_r2, 0.38, yerr=new_std, capsize=3,
        label="New: RepeatedKFold(5×10) — mean of 50",
        color=C_BEST, alpha=0.85, edgecolor="white",
        error_kw=dict(ecolor="grey", elinewidth=1))
ax3.axhline(0, color="black", lw=0.7, ls=":")
ax3.axhline(0.90, color=C_AMBER, linestyle="--", lw=1.2, label="Target 0.90")
ax3.set_xticks(xk); ax3.set_xticklabels(models[:5], rotation=18, ha="right", fontsize=8)
ax3.set_ylabel("R² Score"); ax3.set_ylim(-1.05, 1.05)
ax3.set_title("④ Key Fix: Old KFold vs RepeatedKFold\nSame data, same models — completely different picture",
              fontsize=9, fontweight="bold")
ax3.legend(fontsize=7.5)
# annotate the wild swings
for i, v in enumerate(old_kfold):
    if v < 0:
        ax3.annotate(f"{v:.2f}", (i - 0.2, v - 0.05),
                     ha="center", fontsize=7, color=C_RED)

# ── ⑤ R&D Spend vs Profit scatter ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for state, grp in df.groupby("State"):
    ax4.scatter(grp["R&D Spend"], grp["Profit"],
                label=state, color=STATE_C[state], s=60, alpha=0.85, edgecolors="white")
x_line = np.linspace(x.min(), x.max(), 200)
ax4.plot(x_line, m*x_line + b, color="black", lw=1.5, ls="--", label=f"OLS fit  R²={r2_rd:.3f}")
ax4.set_xlabel("R&D Spend ($)")
ax4.set_ylabel("Profit ($)")
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1e3:.0f}k"))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1e3:.0f}k"))
ax4.set_title(f"⑤ R&D Spend vs Profit\nr = 0.973  |  OLS R²={r2_rd:.3f}",
              fontsize=9, fontweight="bold")
ax4.legend(fontsize=7.5)

# ── ⑥ Correlation heatmap ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
corr = df[numeric].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = LinearSegmentedColormap.from_list("rw", ["#C44E52", "white", "#4C72B0"])
im   = ax5.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ticks = range(len(numeric))
ax5.set_xticks(ticks); ax5.set_yticks(ticks)
ax5.set_xticklabels([c.replace(" ", "\n") for c in numeric], fontsize=8)
ax5.set_yticklabels(numeric, fontsize=8)
for i in range(len(numeric)):
    for j in range(len(numeric)):
        v = corr.values[i, j]
        ax5.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                 color="white" if abs(v) > 0.6 else "black",
                 fontweight="bold" if (i != j and abs(v) > 0.5) else "normal")
plt.colorbar(im, ax=ax5, shrink=0.75)
ax5.set_title("⑥ Correlation Matrix\nR&D Spend dominates (r=0.973)",
              fontsize=9, fontweight="bold")

# ── ⑦ Feature contribution bar (correlation with Profit) ─────────────────────
ax6 = fig.add_subplot(gs[2, 0])
feat_corr = df[["R&D Spend","Administration","Marketing Spend","RD_Ratio"]].corrwith(df["Profit"])
fc_vals   = feat_corr.values
fc_labels = ["R&D Spend", "Administration", "Marketing\nSpend", "RD_Ratio\n(engineered)"]
fc_colors = [C_BEST if v > 0.7 else (C_AMBER if v > 0.3 else C_GREY) for v in fc_vals]
bars6 = ax6.barh(fc_labels, fc_vals, color=fc_colors, edgecolor="white", alpha=0.88)
ax6.axvline(0, color="black", lw=0.8)
ax6.axvline(0.30, color=C_AMBER, lw=1.2, ls="--", label="|r|=0.30 threshold")
ax6.set_xlabel("Pearson r with Profit")
ax6.set_xlim(-0.05, 1.05)
ax6.set_title("⑦ Feature Correlation with Profit\n(green > 0.7 = very strong)",
              fontsize=9, fontweight="bold")
ax6.legend(fontsize=8)
for bar, val in zip(bars6, fc_vals):
    ax6.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"r={val:+.3f}", va="center", fontsize=8.5, fontweight="bold")

# ── ⑧ Profit distribution by State ───────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
state_order = ["Florida", "New York", "California"]
data_by_state = [df[df["State"] == s]["Profit"].values for s in state_order]
bp = ax7.boxplot(data_by_state, labels=state_order, patch_artist=True,
                 medianprops=dict(color="black", linewidth=2),
                 whiskerprops=dict(linewidth=1.2),
                 flierprops=dict(marker="o", markersize=5, alpha=0.5))
colors_box = ["#55A868", "#4C72B0", "#DD8452"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax7.axhline(df["Profit"].mean(), color=C_RED, ls="--", lw=1.3,
            label=f"Grand mean ${df['Profit'].mean():,.0f}")
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1e3:.0f}k"))
ax7.set_ylabel("Profit ($)")
ax7.set_title("⑧ Profit Distribution by State\nOverlapping ranges → State not predictive",
              fontsize=9, fontweight="bold")
ax7.legend(fontsize=8)

# ── ⑨ Summary scorecard ───────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
rows_tab = [
    ["Model",          "CV R²",     "CV std",  "CV RMSE"],
    ["Lasso ★",        "0.9317",    "0.038",   "$9,527"],
    ["LinearReg",      "0.9315",    "0.038",   "$9,536"],
    ["ElasticNet",     "0.9313",    "0.038",   "$9,549"],
    ["Ridge",          "0.9309",    "0.038",   "$9,578"],
    ["RandomForest",   "0.9285",    "0.046",   "$9,923"],
    ["GradBoost",      "0.9018",    "0.051",   "$11,495"],
    ["─────────────",  "────────",  "──────",  "────────"],
    ["Target",         "≥ 0.90",   "< 0.10",  "< $15k"],
    ["Status",         "✓ ALL",     "✓ ALL",   "✓ ALL"],
]
tbl = ax8.table(cellText=rows_tab[1:], colLabels=rows_tab[0],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.15, 1.75)
tbl[(1,0)].set_facecolor("#D5F5E3"); tbl[(1,1)].set_facecolor("#D5F5E3")
tbl[(1,2)].set_facecolor("#D5F5E3"); tbl[(1,3)].set_facecolor("#D5F5E3")
tbl[(9,0)].set_facecolor("#D5F5E3"); tbl[(9,1)].set_facecolor("#D5F5E3")
tbl[(9,2)].set_facecolor("#D5F5E3"); tbl[(9,3)].set_facecolor("#D5F5E3")
ax8.set_title("⑨ Model Leaderboard\n(baseline feature set — all 6 models)",
              fontsize=9, fontweight="bold", pad=72)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "50 Startups — Expert CRISP-DM Results\n"
    "Best model: Lasso (baseline)  |  CV R²=0.9317  |  CV RMSE=$9,527  |  "
    "All 6 models exceed R²≥0.90 target  ✓",
    fontsize=12, fontweight="bold", y=0.995
)

plt.savefig(SAVE, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"[Saved] {SAVE}")
