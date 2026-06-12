"""
gen_charts.py  —  產生工作報告附錄所需的所有圖表 PNG
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

BASE = r"D:\wi\260612"

# ── 資料 ──────────────────────────────────────────────────────────────────────
csv_data = """R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.17,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
114523.61,122616.84,261776.23,New York,129917.04
78013.11,121597.55,264346.06,California,126992.93
94657.16,145077.58,282574.31,New York,125370.37
91749.16,114175.79,294919.57,Florida,124266.9
86419.7,153514.11,0,New York,122776.86
76253.86,113867.3,298664.47,California,118474.03
78389.47,153773.43,299737.29,New York,111313.02
73994.56,122782.75,303319.26,Florida,110352.25
67532.53,105751.03,304768.73,Florida,108733.99
77044.01,99281.34,140574.81,New York,108552.04
64664.71,139553.16,137962.62,California,107404.34
75328.87,144135.98,134050.07,Florida,105733.54
72107.6,127864.55,353183.81,New York,105008.31
66051.52,182645.56,118148.2,Florida,103282.38
65605.48,153032.06,107138.38,New York,101004.64
61994.48,115641.28,91131.24,Florida,99937.59
61136.38,152701.92,88218.23,New York,97483.56
63408.86,129219.61,46085.25,California,97427.84
55493.95,103057.49,214634.81,Florida,96778.92
46426.07,157693.92,210797.67,California,96712.8
46014.02,85047.44,205517.64,New York,96479.51
28663.76,127056.21,201126.82,Florida,90708.19
44069.95,51283.14,197029.42,California,89949.14
20229.59,65947.93,185265.1,New York,81229.06
38558.51,82982.09,174999.3,California,81005.76
28754.33,118546.05,172795.67,California,78239.91
27892.92,84710.77,164470.71,Florida,77798.83
23640.93,96189.63,148001.11,California,71498.49
15505.73,127382.3,35534.17,New York,69758.98
22177.74,154806.14,28334.72,California,65200.33
1000.23,124153.04,1903.93,New York,64926.08
1315.46,115816.21,297114.46,Florida,49490.75
0,135426.92,0,California,42559.73
542.05,51743.15,0,New York,35673.41
0,116983.8,45173.06,California,14681.4"""

import io
df = pd.read_csv(io.StringIO(csv_data))
numeric_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
feat_cols    = ["R&D Spend", "Administration", "Marketing Spend"]

# One-Hot Encoding
df_ohe = pd.get_dummies(df, columns=["State"], drop_first=True)
X_fs = df_ohe.drop("Profit", axis=1).astype(float)
y_fs = df_ohe["Profit"]
feat_names = list(X_fs.columns)
X_scaled_all = StandardScaler().fit_transform(X_fs)

# ══════════════════════════════════════════════════════════════════════════════
# 圖 2a — 分佈圖 + 相關熱力圖
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Phase 2a — 資料分佈與相關性分析", fontsize=14, fontweight="bold")

colors_hist = ["#4C72B0","#DD8452","#55A868","#C44E52"]
for i, col in enumerate(numeric_cols):
    ax = axes[i//3][i%3]
    ax.hist(df[col], bins=12, color=colors_hist[i], edgecolor="white", alpha=0.85)
    ax.set_title(f"{col}  (skew={df[col].skew():.3f})", fontsize=10)
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("頻率", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K" if x>=1000 else f"{x:.0f}"))

state_counts = df["State"].value_counts()
axes[1][1].bar(state_counts.index, state_counts.values,
               color=["#4C72B0","#DD8452","#55A868"], edgecolor="white")
axes[1][1].set_title("State 分佈", fontsize=10)
axes[1][1].set_ylabel("家數", fontsize=9)

corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
            ax=axes[1][2], square=True, linewidths=0.5,
            annot_kws={"size":9})
axes[1][2].set_title("相關係數矩陣", fontsize=10)
axes[1][2].tick_params(labelsize=8)

plt.tight_layout()
out = os.path.join(BASE, "chart_phase2a.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 圖 2b — 散點圖 + 箱型圖
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
fig2.suptitle("Phase 2b — 特徵診斷（散點圖 / 箱型圖 / VIF）", fontsize=14, fontweight="bold")

palette = {"R&D Spend":"#4C72B0","Administration":"#C44E52","Marketing Spend":"#55A868"}
for i, col in enumerate(feat_cols):
    ax = axes2[0][i]
    r = df[[col,"Profit"]].corr().iloc[0,1]
    ax.scatter(df[col], df["Profit"], color=palette[col], alpha=0.7, s=50, edgecolors="white")
    m, b = np.polyfit(df[col], df["Profit"], 1)
    xs = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xs, m*xs+b, "r--", lw=1.8)
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Profit ($)", fontsize=9)
    ax.set_title(f"{col} vs Profit\nr = {r:.4f}", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

for i, col in enumerate(feat_cols):
    ax = axes2[1][i]
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = len(df[(df[col]<Q1-1.5*IQR)|(df[col]>Q3+1.5*IQR)])
    bp = ax.boxplot(df[col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor=palette[col], alpha=0.6),
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", color="red", markersize=5))
    ax.set_title(f"Boxplot: {col}\n異常值: {n_out} 個", fontsize=10)
    ax.set_ylabel(col, fontsize=9)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

plt.tight_layout()
out = os.path.join(BASE, "chart_phase2b.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 圖 3a — 特徵選取矩陣
# ══════════════════════════════════════════════════════════════════════════════
corr_scores = X_fs.corrwith(y_fs).abs()
m1 = (corr_scores >= 0.30).astype(int)
skb = SelectKBest(f_regression, k=3).fit(X_fs, y_fs)
m2  = pd.Series(skb.get_support().astype(int), index=feat_names)
rfe = RFE(LinearRegression(), n_features_to_select=3).fit(X_scaled_all, y_fs)
m3  = pd.Series(rfe.support_.astype(int), index=feat_names)
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_scaled_all, y_fs)
m4  = pd.Series((np.abs(lasso_cv.coef_)>1e-4).astype(int), index=feat_names)
rf  = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_fs, y_fs)
importances = pd.Series(rf.feature_importances_, index=feat_names)
m5  = (importances >= importances.mean()).astype(int)

sel_df = pd.DataFrame({
    "相關性篩選": m1, "SelectKBest": m2,
    "RFE": m3, "Lasso": m4, "隨機森林": m5
}, index=feat_names)
sel_df["投票數"] = sel_df.sum(axis=1)
sel_df["決策"]   = sel_df["投票數"].apply(lambda v: "保留" if v>=3 else "捨棄")

fig3a, axes3a = plt.subplots(1, 2, figsize=(14, 5))
fig3a.suptitle("Phase 3a — 五方法特徵選取共識", fontsize=13, fontweight="bold")

method_cols = ["相關性篩選","SelectKBest","RFE","Lasso","隨機森林"]
sns.heatmap(sel_df[method_cols], annot=True, fmt="d",
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, ax=axes3a[0], cbar=False,
            annot_kws={"size":11, "weight":"bold"})
axes3a[0].set_title("選取矩陣  (1=保留, 0=捨棄)", fontsize=11)
axes3a[0].tick_params(axis="x", rotation=20, labelsize=9)
axes3a[0].tick_params(axis="y", rotation=0, labelsize=9)

vote_colors = ["#55A868" if d=="保留" else "#C44E52" for d in sel_df["決策"]]
bars = axes3a[1].barh(sel_df.index, sel_df["投票數"],
                      color=vote_colors, edgecolor="white", height=0.5)
axes3a[1].axvline(3, color="black", linestyle="--", linewidth=1.5, label="門檻 3/5")
axes3a[1].set_xlabel("票數", fontsize=10)
axes3a[1].set_title("各特徵投票數", fontsize=11)
axes3a[1].set_xlim(0, 6)
for bar, (feat, row) in zip(bars, sel_df.iterrows()):
    axes3a[1].text(bar.get_width()+0.08, bar.get_y()+bar.get_height()/2,
                   f"{row['投票數']}/5  {row['決策']}",
                   va="center", fontsize=9,
                   color="#55A868" if row["決策"]=="保留" else "#C44E52",
                   fontweight="bold")
axes3a[1].legend(fontsize=9)
axes3a[1].tick_params(labelsize=9)

plt.tight_layout()
out = os.path.join(BASE, "chart_phase3a.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 圖 3b — StandardScaler 前後比較
# ══════════════════════════════════════════════════════════════════════════════
selected_features = ["R&D Spend", "Marketing Spend", "Administration"]
X_sel  = df_ohe[selected_features]
y_sel  = df_ohe["Profit"]
X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_sel, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_sc = pd.DataFrame(scaler.transform(X_tr), columns=selected_features)

fig3b, axes3b = plt.subplots(1, 2, figsize=(12, 5))
fig3b.suptitle("Phase 3b — StandardScaler 標準化效果", fontsize=13, fontweight="bold")

axes3b[0].boxplot([X_tr[c] for c in selected_features],
                  tick_labels=selected_features, patch_artist=True,
                  boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                  medianprops=dict(color="black", linewidth=2))
axes3b[0].set_title("標準化前（原始單位）", fontsize=11)
axes3b[0].set_ylabel("數值 (USD)", fontsize=10)
axes3b[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
axes3b[0].tick_params(labelsize=9)

axes3b[1].boxplot([X_tr_sc[c] for c in selected_features],
                  tick_labels=selected_features, patch_artist=True,
                  boxprops=dict(facecolor="#55A868", alpha=0.6),
                  medianprops=dict(color="black", linewidth=2))
axes3b[1].set_title("標準化後 (mean≈0, std≈1)", fontsize=11)
axes3b[1].set_ylabel("標準化值 (z-score)", fontsize=10)
axes3b[1].axhline(0, color="red", linestyle="--", linewidth=1.2, label="mean=0")
axes3b[1].legend(fontsize=9)
axes3b[1].tick_params(labelsize=9)

plt.tight_layout()
out = os.path.join(BASE, "chart_phase3b.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 圖 3c — 各方法效能比較（R² + RMSE）
# ══════════════════════════════════════════════════════════════════════════════
method_labels = ["相關性篩選","SelectKBest","RFE","Lasso","隨機森林","多數決共識"]
method_masks  = [m1, m2, m3, m4, m5,
                 (sel_df["決策"]=="保留").astype(int)]
method_colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#9467BD","#8C564B"]

X_all  = df_ohe.drop("Profit", axis=1).astype(float)
y_all  = df_ohe["Profit"]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
eval_rows = []
for label, mask in zip(method_labels, method_masks):
    cols = [f for f,v in mask.items() if v==1]
    if not cols: continue
    sc   = StandardScaler()
    Xtr  = sc.fit_transform(X_tr2[cols])
    Xte  = sc.transform(X_te2[cols])
    mdl  = LinearRegression().fit(Xtr, y_tr2)
    r2_tr = r2_score(y_tr2, mdl.predict(Xtr))
    r2_te = r2_score(y_te2, mdl.predict(Xte))
    rmse  = np.sqrt(mean_squared_error(y_te2, mdl.predict(Xte)))
    eval_rows.append({"Method":label, "Train_R2":r2_tr, "Test_R2":r2_te, "RMSE":rmse})
eval_df = pd.DataFrame(eval_rows).set_index("Method")

fig3c, axes3c = plt.subplots(1, 2, figsize=(14, 5.5))
fig3c.suptitle("Phase 3c — 特徵選取方法效能比較", fontsize=13, fontweight="bold")

x = np.arange(len(eval_df)); w = 0.38
axes3c[0].bar(x-w/2, eval_df["Train_R2"], w, label="Train R²", color="#4C72B0", alpha=0.85, edgecolor="white")
bars_te = axes3c[0].bar(x+w/2, eval_df["Test_R2"], w, label="Test R²", color="#55A868", alpha=0.85, edgecolor="white")
axes3c[0].axhline(0.90, color="red", linestyle="--", linewidth=2, label="目標 R²=0.90")
axes3c[0].set_xticks(x)
axes3c[0].set_xticklabels(eval_df.index, rotation=22, ha="right", fontsize=9)
axes3c[0].set_ylabel("R²", fontsize=10)
axes3c[0].set_title("R² 比較（訓練集 vs 測試集）", fontsize=11)
axes3c[0].set_ylim(0.80, 1.02)
axes3c[0].legend(fontsize=9)
for bar in bars_te:
    axes3c[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                   f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, color="#1a7a2e", fontweight="bold")

best_i = eval_df["RMSE"].values.argmin()
colors_rmse = [method_colors[i] for i in range(len(eval_df))]
bars_rmse = axes3c[1].bar(eval_df.index, eval_df["RMSE"],
                           color=colors_rmse, alpha=0.85, edgecolor="white")
axes3c[1].set_xticklabels(eval_df.index, rotation=22, ha="right", fontsize=9)
axes3c[1].set_ylabel("RMSE (USD)", fontsize=10)
axes3c[1].set_title("測試集 RMSE", fontsize=11)
axes3c[1].set_ylim(0, eval_df["RMSE"].max()*1.2)
axes3c[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
for j,(bar,rmse) in enumerate(zip(bars_rmse, eval_df["RMSE"])):
    col = "red" if j==best_i else "black"
    axes3c[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                   f"${rmse:,.0f}", ha="center", va="bottom", fontsize=8.5,
                   color=col, fontweight="bold" if j==best_i else "normal")
axes3c[1].annotate("最低 RMSE", xy=(best_i, eval_df["RMSE"].iloc[best_i]+100),
                    xytext=(best_i+0.7, eval_df["RMSE"].min()+1500),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.3))

plt.tight_layout()
out = os.path.join(BASE, "chart_phase3c.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 圖 5 — 評估（Actual vs Predicted / 殘差 / 係數 / CV）
# ══════════════════════════════════════════════════════════════════════════════
preprocessor = ColumnTransformer(
    [("scaler", StandardScaler(), selected_features)],
    remainder="drop"
)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
pipeline.fit(X_tr, y_tr)
model = pipeline.named_steps["model"]
y_pred_tr = pipeline.predict(X_tr)
y_pred_te = pipeline.predict(X_te)
residuals = y_te - y_pred_te

r2_tr  = r2_score(y_tr, y_pred_tr)
r2_te  = r2_score(y_te, y_pred_te)
mae_tr = mean_absolute_error(y_tr, y_pred_tr)
mae_te = mean_absolute_error(y_te, y_pred_te)
rmse_tr= np.sqrt(mean_squared_error(y_tr, y_pred_tr))
rmse_te= np.sqrt(mean_squared_error(y_te, y_pred_te))
mape_te= np.mean(np.abs((y_te - y_pred_te)/y_te))*100

cv_scores = cross_val_score(pipeline, X_sel, y_sel, cv=5, scoring="r2")

coef_df = pd.DataFrame({"特徵":selected_features, "係數":model.coef_}).sort_values("係數", ascending=False)

fig5 = plt.figure(figsize=(16, 10))
fig5.suptitle("Phase 5 — 模型評估結果", fontsize=14, fontweight="bold")
gs5 = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.38, wspace=0.32)

# (1) Actual vs Predicted
ax1 = fig5.add_subplot(gs5[0,0])
ax1.scatter(y_te, y_pred_te, color="steelblue", s=70, edgecolors="white", alpha=0.85, zorder=3)
lim = [min(y_te.min(),y_pred_te.min())*0.95, max(y_te.max(),y_pred_te.max())*1.05]
ax1.plot(lim, lim, "r--", lw=1.8, label="完美預測")
ax1.set_xlabel("實際利潤 ($)", fontsize=9)
ax1.set_ylabel("預測利潤 ($)", fontsize=9)
ax1.set_title(f"實際值 vs 預測值（測試集）\nR² = {r2_te:.4f}", fontsize=10)
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
ax1.text(0.05, 0.93, f"R²={r2_te:.4f}", transform=ax1.transAxes,
         fontsize=11, fontweight="bold", color="green",
         bbox=dict(facecolor="#e6ffe6", edgecolor="green", boxstyle="round,pad=0.3"))

# (2) Residuals vs Predicted
ax2 = fig5.add_subplot(gs5[0,1])
ax2.scatter(y_pred_te, residuals, color="coral", s=70, edgecolors="white", alpha=0.85)
ax2.axhline(0, color="black", lw=1.5, linestyle="--")
ax2.set_xlabel("預測值 ($)", fontsize=9)
ax2.set_ylabel("殘差 ($)", fontsize=9)
ax2.set_title("殘差 vs 預測值", fontsize=10)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

# (3) Residual distribution
ax3 = fig5.add_subplot(gs5[0,2])
ax3.hist(residuals, bins=8, color="mediumseagreen", edgecolor="white", alpha=0.85)
ax3.axvline(0, color="red", linestyle="--", lw=1.5)
ax3.set_xlabel("殘差 ($)", fontsize=9)
ax3.set_ylabel("頻率", fontsize=9)
ax3.set_title("殘差分佈", fontsize=10)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

# (4) Feature coefficients
ax4 = fig5.add_subplot(gs5[1,0])
coef_colors = ["#55A868" if c>=0 else "#C44E52" for c in coef_df["係數"]]
ax4.barh(coef_df["特徵"], coef_df["係數"], color=coef_colors, edgecolor="white")
ax4.axvline(0, color="black", lw=0.8)
ax4.set_xlabel("標準化係數值 ($)", fontsize=9)
ax4.set_title("迴歸係數（標準化特徵）", fontsize=10)
for i,(feat,coef) in enumerate(zip(coef_df["特徵"],coef_df["係數"])):
    ax4.text(coef + (500 if coef>0 else -500), i,
             f"${coef:+,.0f}", va="center", ha="left" if coef>0 else "right",
             fontsize=9, fontweight="bold",
             color="#55A868" if coef>0 else "#C44E52")
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

# (5) Cross-Validation
ax5 = fig5.add_subplot(gs5[1,1])
fold_colors = ["#55A868" if s>=0.9 else "#DD8452" if s>=0 else "#C44E52" for s in cv_scores]
ax5.bar([f"Fold {i+1}" for i in range(5)], cv_scores,
        color=fold_colors, edgecolor="white")
ax5.axhline(cv_scores.mean(), color="red", linestyle="--", lw=1.8,
            label=f"平均={cv_scores.mean():.3f}")
ax5.axhline(0.90, color="green", linestyle=":", lw=1.5, label="目標=0.90")
ax5.set_ylim(-1.1, 1.2)
ax5.set_ylabel("R²", fontsize=9)
ax5.set_title("5 折交叉驗證 R²\n（小樣本不穩定，供參考）", fontsize=10)
ax5.legend(fontsize=8)
ax5.tick_params(labelsize=9)

# (6) Performance table
ax6 = fig5.add_subplot(gs5[1,2])
ax6.axis("off")
tbl_data = [
    ["指標", "訓練集", "測試集"],
    ["R²",     f"{r2_tr:.4f}",  f"{r2_te:.4f}"],
    ["MAE",    f"${mae_tr:,.0f}", f"${mae_te:,.0f}"],
    ["RMSE",   f"${rmse_tr:,.0f}", f"${rmse_te:,.0f}"],
    ["MAPE",   "—",    f"{mape_te:.2f}%"],
]
tbl = ax6.table(cellText=tbl_data[1:], colLabels=tbl_data[0],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1.3, 2.0)
for j in range(3):
    tbl[(0,j)].set_facecolor("#7b68ee")
    tbl[(0,j)].set_text_props(color="white", fontweight="bold")
for i in range(1, 5):
    for j in range(3):
        if i % 2 == 0:
            tbl[(i,j)].set_facecolor("#f0f0f0")
ax6.set_title("效能指標彙總", fontsize=11, fontweight="bold", pad=20)

plt.tight_layout()
out = os.path.join(BASE, "chart_phase5.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# 圖 3d — 隨機森林特徵重要性（補充圖）
# ══════════════════════════════════════════════════════════════════════════════
fig_rf, ax_rf = plt.subplots(figsize=(8, 4))
fig_rf.suptitle("Phase 3d — 隨機森林特徵重要性", fontsize=12, fontweight="bold")

imp_sorted = importances.sort_values(ascending=True)
colors_imp = ["#55A868" if v>=importances.mean() else "#adb5bd" for v in imp_sorted]
bars_imp = ax_rf.barh(imp_sorted.index, imp_sorted.values,
                      color=colors_imp, edgecolor="white", height=0.5)
ax_rf.axvline(importances.mean(), color="red", linestyle="--", lw=1.5,
              label=f"平均值={importances.mean():.4f}")
ax_rf.set_xlabel("特徵重要性", fontsize=10)
ax_rf.set_title("(n_estimators=200, random_state=42)", fontsize=9, pad=4)
ax_rf.legend(fontsize=9)
for bar, val in zip(bars_imp, imp_sorted.values):
    ax_rf.text(val+0.003, bar.get_y()+bar.get_height()/2,
               f"{val:.4f}", va="center", fontsize=9, fontweight="bold",
               color="#1a7a2e" if val>=importances.mean() else "#666")

plt.tight_layout()
out = os.path.join(BASE, "chart_rf_importance.png")
plt.savefig(out, bbox_inches="tight", dpi=130)
plt.close()
print(f"Saved: {out}")

print("\nAll charts generated.")
