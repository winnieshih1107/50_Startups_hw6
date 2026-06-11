"""
Generate two CRISP-DM workflow PNGs:
  hw6_workflow_clean.png      — polished draw.io-like style
  hw6_workflow_excalidraw.png — hand-drawn Excalidraw style (plt.xkcd)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

CLEAN = r"D:\wi\hw6_workflow_clean.png"
EXCAL = r"D:\wi\hw6_workflow_excalidraw.png"

# Coordinate system: 10 wide, rows from y=0 (bottom) upward
CX    = 5.0
BOX_W = 7.6
BOX_H = 1.80
OVL_W = 2.80
OVL_H = 0.85
DIA_W = 4.20
DIA_H = 1.50

Y_START = 20.5
Y_P1    = 18.30
Y_P2    = 16.10
Y_P3    = 13.90
Y_P4    = 11.70
Y_P5    =  9.50
Y_DEC   =  7.30
Y_P6    =  5.10
Y_END   =  3.20

PC = {
    'start': '#2C3E50', 'p1': '#2E86C1', 'p2': '#17A589',
    'p3':    '#229954', 'p4': '#7D3C98', 'p5': '#CA6F1E',
    'dec':   '#C0392B', 'p6': '#943126', 'end': '#2C3E50',
}


def _arrow(ax, x1, y1, x2, y2, color, lw):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=16),
                zorder=4)


def _oval(ax, cx, cy, color, label, ec, lw, fs=14):
    ax.add_patch(mpatches.Ellipse((cx, cy), OVL_W, OVL_H,
                                   facecolor=color, edgecolor=ec,
                                   linewidth=lw, zorder=5))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=6)


def _box(ax, cx, cy, color, title, sub, ec, lw, bsty):
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
        boxstyle=bsty, facecolor=color, edgecolor=ec,
        linewidth=lw, zorder=5))
    if sub:
        ax.text(cx, cy + 0.43, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white', zorder=6)
        ax.text(cx, cy - 0.43, sub, ha='center', va='center',
                fontsize=7.8, color='white', alpha=0.93, zorder=6,
                linespacing=1.45)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white', zorder=6)


def _diamond(ax, cx, cy, color, label, ec, lw):
    pts = np.array([
        [cx, cy + DIA_H/2], [cx + DIA_W/2, cy],
        [cx, cy - DIA_H/2], [cx - DIA_W/2, cy],
    ])
    ax.add_patch(plt.Polygon(pts, facecolor=color, edgecolor=ec,
                              linewidth=lw, zorder=5))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6,
            linespacing=1.3)


def build(xkcd=False):
    bg   = '#FFFBE6' if xkcd else '#F0F3F4'
    ec   = '#333333' if xkcd else 'white'
    ac   = '#333333' if xkcd else '#5D6D7E'
    lw   = 2.5 if xkcd else 1.6
    bsty = "square,pad=0.1" if xkcd else "round,pad=0.15"

    fig, ax = plt.subplots(figsize=(11, 22))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10)
    ax.set_ylim(2.4, 21.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Nodes ──────────────────────────────────────────────────────────────────
    _oval(ax, CX, Y_START, PC['start'], 'START', ec, lw)

    _box(ax, CX, Y_P1, PC['p1'],
         'Phase 1  —  Business Understanding',
         'Goal: Predict startup annual Profit  •  Target: continuous Profit (USD)\n'
         'Success: CV R² ≥ 0.90  |  CV RMSE < $15,000',
         ec, lw, bsty)

    _box(ax, CX, Y_P2, PC['p2'],
         'Phase 2  —  Data Understanding',
         '50 rows x 5 cols  •  No missing values  •  R&D Spend: r = 0.973 (dominant)\n'
         'Administration: p=0.61 (weak)  |  State: p>0.95 (not predictive at n=50)',
         ec, lw, bsty)

    _box(ax, CX, Y_P3, PC['p3'],
         'Phase 3  —  Data Preparation',
         'Engineer: RD_Ratio = R&D/Total_Spend  |  Log_RD = log1p(R&D Spend)\n'
         'Drop State (0/5 votes)  •  StandardScaler in Pipeline  •  80/20 Split',
         ec, lw, bsty)

    _box(ax, CX, Y_P4, PC['p4'],
         'Phase 4  —  Modeling',
         '6 Models: LinearReg / Ridge / Lasso / ElasticNet / RandomForest / GBM\n'
         'RepeatedKFold(n_splits=5, n_repeats=10) = 50 CV scores  •  Alpha auto-tuned',
         ec, lw, bsty)

    _box(ax, CX, Y_P5, PC['p5'],
         'Phase 5  —  Evaluation',
         'Metrics: CV R², RMSE, MAE, MAPE  •  Residual & importance analysis\n'
         'Best: Lasso (baseline)  CV R²=0.9317  CV RMSE=$9,527  Test R²=0.9001',
         ec, lw, bsty)

    _diamond(ax, CX, Y_DEC, PC['dec'],
             'R² >= 0.90\n& RMSE < $15k ?', ec, lw)

    _box(ax, CX, Y_P6, PC['p6'],
         'Phase 6  —  Deployment',
         'Save: 50_startups_best_model.pkl (joblib)  •  API: predict_profit(rd, admin, mkt)\n'
         'Monitoring: RMSE monthly  •  Re-train trigger: RMSE > $12,000',
         ec, lw, bsty)

    _oval(ax, CX, Y_END, PC['end'], 'END', ec, lw)

    # ── Main-flow arrows ────────────────────────────────────────────────────────
    pairs = [
        (CX, Y_START - OVL_H/2, CX, Y_P1 + BOX_H/2),
        (CX, Y_P1 - BOX_H/2,    CX, Y_P2 + BOX_H/2),
        (CX, Y_P2 - BOX_H/2,    CX, Y_P3 + BOX_H/2),
        (CX, Y_P3 - BOX_H/2,    CX, Y_P4 + BOX_H/2),
        (CX, Y_P4 - BOX_H/2,    CX, Y_P5 + BOX_H/2),
        (CX, Y_P5 - BOX_H/2,    CX, Y_DEC + DIA_H/2),
        (CX, Y_P6 - BOX_H/2,    CX, Y_END + OVL_H/2),
    ]
    for (x1, y1, x2, y2) in pairs:
        _arrow(ax, x1, y1, x2, y2, ac, lw)

    # YES path: decision diamond → Phase 6
    _arrow(ax, CX, Y_DEC - DIA_H/2, CX, Y_P6 + BOX_H/2, '#27AE60', lw)
    ax.text(CX + 0.32, (Y_DEC + Y_P6)/2, 'YES ✓',
            fontsize=9, color='#27AE60', fontweight='bold')

    # NO feedback loop: diamond left tip → left side of Phase 4
    dlx  = CX - DIA_W/2          # diamond left tip x = 5 - 2.1 = 2.9
    blx  = CX - BOX_W/2          # box left edge x   = 5 - 3.8 = 1.2
    lx   = 0.55                   # vertical segment x (far left)
    ax.plot([dlx, lx], [Y_DEC, Y_DEC], color='#E74C3C', lw=lw, zorder=4)
    ax.plot([lx,  lx], [Y_DEC, Y_P4],  color='#E74C3C', lw=lw, zorder=4)
    _arrow(ax, lx, Y_P4, blx, Y_P4, '#E74C3C', lw)
    ax.text(dlx - 0.55, Y_DEC + 0.22, 'NO',
            fontsize=9, color='#E74C3C', fontweight='bold')
    mid_y = (Y_DEC + Y_P4) / 2
    ax.text(lx - 0.05, mid_y, 'Re-tune\nModels',
            fontsize=8, color='#E74C3C', fontweight='bold',
            ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC',
                      edgecolor='#E74C3C', alpha=0.9))

    # ── Legend ───────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=PC['p1'], label='Phase 1: Business Understanding'),
        mpatches.Patch(color=PC['p2'], label='Phase 2: Data Understanding'),
        mpatches.Patch(color=PC['p3'], label='Phase 3: Data Preparation'),
        mpatches.Patch(color=PC['p4'], label='Phase 4: Modeling'),
        mpatches.Patch(color=PC['p5'], label='Phase 5: Evaluation'),
        mpatches.Patch(color=PC['p6'], label='Phase 6: Deployment'),
    ]
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 0.0),
              fontsize=8, title='CRISP-DM Phases', title_fontsize=9,
              framealpha=0.93, edgecolor='#BFC9CA', ncol=2)

    style_lbl = 'Excalidraw Style' if xkcd else 'draw.io Style'
    fig.suptitle(f'CRISP-DM Workflow  —  50 Startups  [{style_lbl}]',
                 fontsize=12, fontweight='bold', color='#1C2833', y=0.995)

    return fig


# ── Produce clean PNG ─────────────────────────────────────────────────────────
print("Building clean PNG ...")
fc = build(xkcd=False)
fc.savefig(CLEAN, dpi=150, bbox_inches='tight', facecolor='#F0F3F4')
plt.close(fc)
print(f"[Saved] {CLEAN}")

# ── Produce Excalidraw PNG ────────────────────────────────────────────────────
print("Building Excalidraw PNG ...")
with plt.xkcd(scale=0.8, length=200, randomness=4):
    fe = build(xkcd=True)
    fe.savefig(EXCAL, dpi=150, bbox_inches='tight', facecolor='#FFFBE6')
    plt.close(fe)
print(f"[Saved] {EXCAL}")

print("All done.")
