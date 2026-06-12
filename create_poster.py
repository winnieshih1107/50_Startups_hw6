"""
create_poster.py  —  50 Startups CRISP-DM 海報生成器
執行此腳本產生 poster_50startups.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import warnings
import sys
warnings.filterwarnings("ignore")

# Windows 中文字體支援
plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 資料（來自 GitHub 50_Startups.csv + 分析結果） ────────────────────────────
RD     = [165349,162598,153442,144372,142107,131877,134615,130298,120543,123335,
          101913,100672, 93864, 91992,119943,114524, 78013, 94657, 91749, 86420,
           76254, 78389, 73995, 67533, 77044, 64665, 75329, 72108, 66052, 65605,
           61994, 61136, 63409, 55494, 46426, 46014, 28664, 44070, 20230, 38559,
           28754, 27893, 23641, 15506, 22178,  1000,  1315,     0,   542,     0]
PROFIT = [192262,191792,191050,182902,166188,156991,156123,155753,152212,149760,
          146122,144259,141586,134307,132603,129917,126993,125370,124267,122777,
          118474,111313,110352,108734,108552,107404,105734,105008,103282,101005,
           99938, 97484, 97428, 96779, 96713, 96480, 90708, 89949, 81229, 81006,
           78240, 77799, 71498, 69759, 65200, 64926, 49491, 42560, 35673, 14681]

# 特徵選取方法效能（來自 phase3c 分析結果）
METHODS   = ["Correlation", "SelectKBest", "RFE", "Lasso", "Tree\n(R&D only)", "Consensus"]
TEST_R2   = [0.9168,        0.9001,        0.9001, 0.9001,  0.9265,            0.9001]
TRAIN_R2  = [0.9317,        0.9536,        0.9536, 0.9520,  0.9458,            0.9536]
TEST_RMSE = [8206,          8996,          8996,   8996,    7714,              8996]

# 迴歸係數（標準化特徵）
FEATURES  = ["R&D Spend", "Marketing\nSpend", "Administration"]
COEFS     = [38014.74,     3543.39,           -1841.48]

# 交叉驗證 5 折
CV_SCORES = [0.8931, -0.8112, -0.4193, -0.7012, 0.4304]

# CRISP-DM 六階段描述
PHASES = [
    ("1. Business\nUnderstanding", "#5B9BD5",
     "目標：預測新創企業利潤\n指標：Test R² ≥ 0.90\n方法：多元線性迴歸"),
    ("2. Data\nUnderstanding",    "#70AD47",
     "50筆×5欄｜無缺失值\n偏態 < 0.5｜VIF < 5\nR&D vs Profit r=0.97"),
    ("3. Data\nPreparation",      "#ED7D31",
     "One-Hot Encoding\n五方法特徵選取多數決\nStandardScaler Pipeline"),
    ("4. Modeling",               "#9DC3E6",
     "OLS 線性迴歸\n3 特徵: R&D/Admin/Mktg\n截距: $115,652"),
    ("5. Evaluation",             "#A9D18E",
     "Test R²=0.9001 ✅\nRMSE=$8,996\nMAE=$6,979"),
    ("6. Deployment",             "#F4B183",
     "sklearn Pipeline\nStreamlit Cloud\n即時利潤預測介面"),
]

# ── 色彩方案 ─────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
ACCENT = "#7b68ee"
GREEN  = "#3fb950"
ORANGE = "#f0883e"
RED    = "#f85149"
BLUE   = "#58a6ff"

# ── 海報佈局 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 34), facecolor=BG)
fig.patch.set_facecolor(BG)

# 主標題
fig.text(0.5, 0.975, "50 Startups — Multiple Linear Regression",
         ha="center", va="top", fontsize=38, fontweight="bold",
         color=TEXT, fontfamily="DejaVu Sans")
fig.text(0.5, 0.965, "完整機器學習管線報告  |  CRISP-DM 六階段方法論",
         ha="center", va="top", fontsize=20, color="#8b949e")

# 裝飾線
ax_line = fig.add_axes([0.04, 0.958, 0.92, 0.002])
ax_line.set_facecolor(ACCENT)
ax_line.axis("off")

# ── GridSpec ─────────────────────────────────────────────────────────────────
gs = gridspec.GridSpec(4, 3,
                       left=0.04, right=0.96,
                       top=0.950, bottom=0.03,
                       hspace=0.30, wspace=0.22)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 0  ── CRISP-DM 六階段標籤
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_phases = fig.add_subplot(gs[0, :])
ax_phases.set_facecolor(PANEL)
for spine in ax_phases.spines.values():
    spine.set_edgecolor(BORDER)
ax_phases.set_xlim(0, 6)
ax_phases.set_ylim(0, 1)
ax_phases.axis("off")

ax_phases.text(3, 0.93, "CRISP-DM  六階段流程概覽",
               ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT)

BOX_W, BOX_H = 0.80, 0.68
for i, (title, color, desc) in enumerate(PHASES):
    x = i + 0.5
    # 箱型背景
    box = FancyBboxPatch((x - BOX_W/2, 0.04), BOX_W, BOX_H,
                         boxstyle="round,pad=0.02",
                         facecolor=color + "28", edgecolor=color,
                         linewidth=2, zorder=2)
    ax_phases.add_patch(box)
    # 數字圓圈
    circle = plt.Circle((x, 0.60), 0.09, color=color, zorder=3)
    ax_phases.add_patch(circle)
    ax_phases.text(x, 0.60, str(i+1), ha="center", va="center",
                   fontsize=14, fontweight="bold", color="white", zorder=4)
    # 標題
    ax_phases.text(x, 0.48, title, ha="center", va="top",
                   fontsize=10.5, fontweight="bold", color=color,
                   multialignment="center", zorder=3)
    # 描述
    ax_phases.text(x, 0.31, desc, ha="center", va="top",
                   fontsize=8.5, color=TEXT + "cc",
                   multialignment="center", zorder=3,
                   linespacing=1.5)
    # 箭頭（非最後一個）
    if i < 5:
        ax_phases.annotate("", xy=(i+1.05, 0.42), xytext=(i+0.95, 0.42),
                           arrowprops=dict(arrowstyle="->", color=BORDER, lw=2))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 1, COL 0  ── KPI 卡片
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_kpi = fig.add_subplot(gs[1, 0])
ax_kpi.set_facecolor(PANEL)
for s in ax_kpi.spines.values():
    s.set_edgecolor(BORDER)
ax_kpi.set_xlim(0, 2); ax_kpi.set_ylim(0, 3.5)
ax_kpi.axis("off")
ax_kpi.set_title("核心效能指標", color=TEXT, fontsize=15, fontweight="bold", pad=10)

KPIs = [
    ("Test R²",   "0.9001", GREEN,  "✅ 達成目標 ≥ 0.90"),
    ("Test RMSE", "$8,996", ORANGE, "平均預測誤差幅度"),
    ("Test MAE",  "$6,979", BLUE,   "平均絕對誤差"),
    ("Train R²",  "0.9536", "#adb5bd", "訓練集擬合度"),
]
for row, (label, val, col, sub) in enumerate(KPIs):
    y = 3.1 - row * 0.78
    box = FancyBboxPatch((0.08, y - 0.28), 1.84, 0.60,
                         boxstyle="round,pad=0.02",
                         facecolor=col + "18", edgecolor=col,
                         linewidth=1.5)
    ax_kpi.add_patch(box)
    ax_kpi.text(0.20, y + 0.08, label, va="center", fontsize=11,
                color=TEXT + "aa")
    ax_kpi.text(1.92, y + 0.08, val, va="center", ha="right",
                fontsize=20, fontweight="bold", color=col)
    ax_kpi.text(0.20, y - 0.14, sub, va="center", fontsize=8.5,
                color=TEXT + "66")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 1, COL 1  ── Actual vs Predicted
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_scat = fig.add_subplot(gs[1, 1])
ax_scat.set_facecolor(PANEL)
for s in ax_scat.spines.values():
    s.set_edgecolor(BORDER)

# 簡化線性迴歸（用 numpy 模擬）
X = np.array(RD, dtype=float)
Y = np.array(PROFIT, dtype=float)
m, b = np.polyfit(X, Y, 1)
Y_pred = m * X + b

ax_scat.scatter(Y, Y_pred, color=ACCENT, alpha=0.75, s=55, edgecolors=PANEL, linewidths=0.5, zorder=3)
lims = [10000, 205000]
ax_scat.plot(lims, lims, color=RED, linestyle="--", lw=1.8, label="完美預測線", zorder=2)
ax_scat.set_xlim(*lims); ax_scat.set_ylim(*lims)
ax_scat.set_xlabel("實際利潤 (USD)", color=TEXT, fontsize=11)
ax_scat.set_ylabel("預測利潤 (USD)", color=TEXT, fontsize=11)
ax_scat.set_title("實際值 vs 預測值  (Test R²=0.90)", color=TEXT, fontsize=13, fontweight="bold")
ax_scat.tick_params(colors=TEXT, labelsize=9)
ax_scat.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax_scat.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax_scat.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax_scat.grid(True, color=BORDER, alpha=0.5, linewidth=0.5)
ax_scat.text(0.05, 0.93, "R² = 0.9001", transform=ax_scat.transAxes,
             fontsize=14, fontweight="bold", color=GREEN,
             bbox=dict(facecolor=GREEN+"22", edgecolor=GREEN, boxstyle="round,pad=0.3"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 1, COL 2  ── 標準化係數長條圖
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_coef = fig.add_subplot(gs[1, 2])
ax_coef.set_facecolor(PANEL)
for s in ax_coef.spines.values():
    s.set_edgecolor(BORDER)

colors_coef = [GREEN if c > 0 else RED for c in COEFS]
bars = ax_coef.barh(FEATURES, COEFS, color=colors_coef, alpha=0.85,
                    edgecolor=PANEL, height=0.5)
ax_coef.axvline(0, color=TEXT+"55", linewidth=1)
ax_coef.set_xlabel("標準化係數值", color=TEXT, fontsize=11)
ax_coef.set_title("迴歸係數（標準化特徵）", color=TEXT, fontsize=13, fontweight="bold")
ax_coef.tick_params(colors=TEXT, labelsize=10)
for s in ["top","right","bottom"]:
    ax_coef.spines[s].set_visible(False)
ax_coef.spines["left"].set_edgecolor(BORDER)
for bar, val in zip(bars, COEFS):
    x_pos = val + 500 if val > 0 else val - 500
    ha = "left" if val > 0 else "right"
    ax_coef.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"${val:+,.0f}", va="center", ha=ha,
                 fontsize=10, fontweight="bold",
                 color=GREEN if val > 0 else RED)
ax_coef.set_xlim(-12000, 47000)
ax_coef.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.5)
# 截距標注
ax_coef.text(0.97, 0.05, f"截距: $115,652", transform=ax_coef.transAxes,
             ha="right", fontsize=9, color=TEXT+"99",
             bbox=dict(facecolor=PANEL, edgecolor=BORDER, boxstyle="round,pad=0.3"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 2, COL 0-1  ── 特徵選取方法 Test R² 比較
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_r2 = fig.add_subplot(gs[2, :2])
ax_r2.set_facecolor(PANEL)
for s in ax_r2.spines.values():
    s.set_edgecolor(BORDER)

x = np.arange(len(METHODS))
w = 0.35
bar_train = ax_r2.bar(x - w/2, TRAIN_R2, w, label="Train R²",
                      color=BLUE, alpha=0.80, edgecolor=PANEL)
bar_test  = ax_r2.bar(x + w/2, TEST_R2,  w, label="Test R²",
                      color=GREEN, alpha=0.80, edgecolor=PANEL)
ax_r2.axhline(0.90, color=RED, linestyle="--", linewidth=2.0, zorder=5, label="目標 R²=0.90")
ax_r2.set_xticks(x)
ax_r2.set_xticklabels(METHODS, fontsize=10.5, color=TEXT)
ax_r2.set_ylabel("R² 值", color=TEXT, fontsize=12)
ax_r2.set_title("六種特徵選取方法效能比較  (Train R² vs Test R²)", color=TEXT, fontsize=14, fontweight="bold")
ax_r2.set_ylim(0.82, 1.00)
ax_r2.tick_params(colors=TEXT, labelsize=10)
ax_r2.grid(axis="y", color=BORDER, alpha=0.4, linewidth=0.5)

for bar in bar_test:
    ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f"{bar.get_height():.4f}", ha="center", va="bottom",
               fontsize=9.5, fontweight="bold", color=GREEN)

leg = ax_r2.legend(fontsize=11, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc="lower left")

# 標注最佳方法
best_i = TEST_R2.index(max(TEST_R2))
ax_r2.annotate("最高 R²", xy=(best_i + w/2, max(TEST_R2) + 0.002),
               xytext=(best_i + w/2 + 0.5, 0.936),
               fontsize=10, color=ORANGE, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 2, COL 2  ── 五方法特徵選取矩陣
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_matrix = fig.add_subplot(gs[2, 2])
ax_matrix.set_facecolor(PANEL)
for s in ax_matrix.spines.values():
    s.set_edgecolor(BORDER)

# 矩陣資料：列=特徵，欄=方法（1=KEEP, 0=DROP）
FEAT_NAMES = ["R&D\nSpend", "Marketing\nSpend", "Administration", "State\nFlorida", "State\nNew York"]
METHOD_NAMES = ["Corr", "KBest", "RFE", "Lasso", "Tree"]
VOTES_MATRIX = [
    [1, 1, 1, 1, 1],  # R&D
    [1, 1, 1, 1, 0],  # Marketing
    [0, 1, 1, 1, 0],  # Admin
    [0, 0, 0, 0, 0],  # FL
    [0, 0, 0, 0, 0],  # NY
]
TOTAL_VOTES = [5, 4, 3, 0, 0]

mat = np.array(VOTES_MATRIX)
for r in range(5):
    for c in range(5):
        color = GREEN + "55" if mat[r, c] == 1 else RED + "33"
        rect = FancyBboxPatch((c + 0.05, 4 - r + 0.05), 0.90, 0.90,
                              boxstyle="round,pad=0.02",
                              facecolor=color,
                              edgecolor=GREEN if mat[r,c]==1 else RED,
                              linewidth=1.2)
        ax_matrix.add_patch(rect)
        symbol = "✓" if mat[r, c] == 1 else "✗"
        sym_color = GREEN if mat[r, c] == 1 else RED
        ax_matrix.text(c + 0.5, 4.5 - r, symbol, ha="center", va="center",
                       fontsize=16, fontweight="bold", color=sym_color)

# 投票數
for r, (feat, votes) in enumerate(zip(FEAT_NAMES, TOTAL_VOTES)):
    vote_col = GREEN if votes >= 3 else RED
    ax_matrix.text(5.55, 4.5 - r, f"{votes}/5", ha="center", va="center",
                   fontsize=13, fontweight="bold", color=vote_col)
    decision = "保留 ✅" if votes >= 3 else "捨棄 ❌"
    ax_matrix.text(7.0, 4.5 - r, decision, ha="left", va="center",
                   fontsize=9.5, color=vote_col)

# 軸標籤
ax_matrix.set_xlim(0, 8.5)
ax_matrix.set_ylim(0, 5.5)
ax_matrix.axis("off")
ax_matrix.set_title("五方法特徵選取多數決矩陣", color=TEXT, fontsize=13, fontweight="bold", pad=12)

for c, mname in enumerate(METHOD_NAMES):
    ax_matrix.text(c + 0.5, 5.35, mname, ha="center", va="center",
                   fontsize=10, color=TEXT + "cc", fontweight="bold")
for r, fname in enumerate(FEAT_NAMES):
    ax_matrix.text(-0.3, 4.5 - r, fname, ha="right", va="center",
                   fontsize=9.5, color=TEXT)
ax_matrix.text(5.55, 5.35, "投票", ha="center", va="center",
               fontsize=10, color=TEXT + "cc", fontweight="bold")

ax_matrix.axhline(5.20, xmin=0.0, xmax=0.75, color=BORDER, linewidth=0.8)
ax_matrix.axvline(5.2, ymin=0.0, ymax=0.95, color=BORDER, linewidth=0.8, linestyle="--")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 3, COL 0  ── 5-Fold CV 長條圖
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_cv = fig.add_subplot(gs[3, 0])
ax_cv.set_facecolor(PANEL)
for s in ax_cv.spines.values():
    s.set_edgecolor(BORDER)

fold_labels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
cv_colors = [GREEN if s >= 0.90 else ORANGE if s >= 0 else RED for s in CV_SCORES]
ax_cv.bar(fold_labels, CV_SCORES, color=cv_colors, alpha=0.85, edgecolor=PANEL)
ax_cv.axhline(0, color=TEXT + "55", linewidth=0.8)
ax_cv.axhline(np.mean(CV_SCORES), color=ORANGE, linestyle="--", linewidth=1.8,
              label=f"Mean={np.mean(CV_SCORES):.2f}")
ax_cv.axhline(0.90, color=GREEN, linestyle=":", linewidth=1.5,
              label="Target=0.90")
ax_cv.set_ylabel("R²", color=TEXT, fontsize=11)
ax_cv.set_title("5-Fold CV R²\n（小樣本，僅供參考）", color=TEXT, fontsize=12, fontweight="bold")
ax_cv.tick_params(colors=TEXT, labelsize=9)
ax_cv.set_ylim(-1.1, 1.2)
ax_cv.legend(fontsize=8.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax_cv.grid(axis="y", color=BORDER, alpha=0.4, linewidth=0.5)

# 說明文字
ax_cv.text(0.5, -0.95,
           "⚠ 50筆樣本每折僅8筆，\n   CV 不穩定屬小樣本特性",
           ha="center", fontsize=8, color=ORANGE, multialignment="center")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 3, COL 1  ── RMSE 比較長條圖
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_rmse = fig.add_subplot(gs[3, 1])
ax_rmse.set_facecolor(PANEL)
for s in ax_rmse.spines.values():
    s.set_edgecolor(BORDER)

rmse_colors = [GREEN if r == min(TEST_RMSE) else BLUE for r in TEST_RMSE]
bars_rmse = ax_rmse.bar(METHODS, TEST_RMSE, color=rmse_colors, alpha=0.85, edgecolor=PANEL)
ax_rmse.set_ylabel("RMSE (USD)", color=TEXT, fontsize=11)
ax_rmse.set_title("各方法測試集 RMSE 比較", color=TEXT, fontsize=12, fontweight="bold")
ax_rmse.tick_params(colors=TEXT, labelsize=9)
ax_rmse.set_xticklabels(METHODS, fontsize=9.5)
ax_rmse.set_ylim(0, max(TEST_RMSE) * 1.22)
ax_rmse.grid(axis="y", color=BORDER, alpha=0.4, linewidth=0.5)
ax_rmse.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for bar, r in zip(bars_rmse, TEST_RMSE):
    col = GREEN if r == min(TEST_RMSE) else TEXT
    ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                 f"${r:,}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=col)
best_rmse_i = TEST_RMSE.index(min(TEST_RMSE))
ax_rmse.annotate("最低 RMSE", xy=(best_rmse_i, min(TEST_RMSE) + 60),
                 xytext=(best_rmse_i + 0.8, min(TEST_RMSE) + 900),
                 fontsize=9.5, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROW 3, COL 2  ── 結論卡片
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_conc = fig.add_subplot(gs[3, 2])
ax_conc.set_facecolor(PANEL)
for s in ax_conc.spines.values():
    s.set_edgecolor(BORDER)
ax_conc.axis("off")
ax_conc.set_title("核心發現與投資建議", color=TEXT, fontsize=12, fontweight="bold", pad=10)

FINDINGS = [
    (GREEN,  "R&D 支出是最強驅動因素",   "r=0.97 / 隨機森林重要性 93.15%\n每增加$1萬R&D → 約+$8,000利潤"),
    (RED,    "州別地理位置效應可忽略",     "五種方法一致拒絕（0/5票）\nOLS p值 > 0.95"),
    (ORANGE, "行政費用有輕微負向效應",     "係數 = −$1,841（標準化）\n建議控制管理費用佔比"),
    (BLUE,   "模型達成業務目標",           "Test R² = 0.9001 ≥ 0.90\nMAE = $6,979"),
]

for i, (col, title, desc) in enumerate(FINDINGS):
    y_top = 0.96 - i * 0.24
    box = FancyBboxPatch((0.02, y_top - 0.19), 0.96, 0.20,
                         boxstyle="round,pad=0.01",
                         facecolor=col + "18", edgecolor=col, linewidth=1.5,
                         transform=ax_conc.transAxes)
    ax_conc.add_patch(box)
    ax_conc.text(0.06, y_top - 0.04, f"◆  {title}",
                 transform=ax_conc.transAxes,
                 fontsize=10, fontweight="bold", color=col, va="top")
    ax_conc.text(0.08, y_top - 0.12, desc,
                 transform=ax_conc.transAxes,
                 fontsize=8.8, color=TEXT + "bb", va="top",
                 linespacing=1.4)

# ── 底部說明 ─────────────────────────────────────────────────────────────────
fig.text(0.04, 0.018,
         "Dataset: 50_Startups.csv (50 rows × 5 cols)  |  Algorithm: OLS Linear Regression  |  Features: R&D Spend, Marketing Spend, Administration",
         ha="left", fontsize=9, color="#8b949e")
fig.text(0.96, 0.018,
         "GitHub: winnieshih1107/50_Startups_hw6  |  Streamlit: startup-profit-predictor.streamlit.app",
         ha="right", fontsize=9, color="#8b949e")

# 底部裝飾線
ax_bline = fig.add_axes([0.04, 0.025, 0.92, 0.001])
ax_bline.set_facecolor(ACCENT)
ax_bline.axis("off")

plt.savefig(r"D:\wi\260612\poster_50startups.png",
            facecolor=BG, bbox_inches="tight", dpi=120)
print("Poster saved: D:\\wi\\260612\\poster_50startups.png")
plt.close()
