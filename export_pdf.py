"""
export_pdf.py  —  work_report.md  →  work_report.pdf
使用 Chrome headless 轉換，完整支援繁體中文與表格
"""
import sys, os, subprocess, tempfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import markdown

BASE     = r"D:\wi\260612"
MD_PATH  = os.path.join(BASE, "work_report.md")
PDF_PATH = os.path.join(BASE, "work_report.pdf")
CHROME   = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# ── 讀取 Markdown ────────────────────────────────────────────────────────────
with open(MD_PATH, encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "toc", "fenced_code", "nl2br"],
)

# ── HTML 模板（A4 列印 CSS） ──────────────────────────────────────────────────
FONT = "Microsoft JhengHei, Microsoft YaHei, Arial Unicode MS, sans-serif"

CHART_DIR = BASE

# 附錄圖表清單
APPENDIX_CHARTS = [
    ("chart_phase2a.png", "圖 A-1　Phase 2a — 特徵分佈與相關係數矩陣",
     "四個數值欄位的直方圖（含偏態值）、州別企業數長條圖，以及 Profit 與三個特徵間的 Pearson 相關係數熱力圖。R&D Spend 與 Profit 的相關係數高達 0.973，為最強預測因子。"),
    ("chart_phase2b.png", "圖 A-2　Phase 2b — 特徵診斷（散點圖 / 箱型圖）",
     "左排：三個特徵對 Profit 的散點圖，附迴歸趨勢線與相關係數 r 值。右排：對應的箱型圖，紅點標示以 IQR×1.5 判定的離群值（三特徵均無異常值）。"),
    ("chart_phase3a.png", "圖 A-3　Phase 3a — 五方法特徵選取共識矩陣",
     "左：5×5 選取矩陣（綠=保留，紅=捨棄）。右：各特徵總票數長條圖，虛線為 3/5 共識門檻。R&D Spend 獲得全票（5/5），State 虛擬變數得 0 票一致捨棄。"),
    ("chart_rf_importance.png", "圖 A-4　Phase 3d — 隨機森林特徵重要性",
     "n_estimators=200 的隨機森林計算各特徵的 Gini 重要性。R&D Spend 的重要性高達 0.9315，遠超其他所有特徵的總和（0.0685），紅線為平均值門檻。"),
    ("chart_phase3b.png", "圖 A-5　Phase 3b — StandardScaler 標準化效果",
     "左：原始尺度箱型圖（Marketing Spend std=$122K vs Administration std=$28K，差距懸殊）。右：StandardScaler 後各特徵均值趨近 0、標準差趨近 1，係數可直接比較。"),
    ("chart_phase3c.png", "圖 A-6　Phase 3c — 六種特徵選取方法效能比較",
     "左：Train/Test R² 並排長條圖，紅虛線為目標 R²=0.90。右：測試集 RMSE 比較，隨機森林方法（僅 R&D 特徵）達到最低 RMSE=$7,714。"),
    ("chart_phase5.png", "圖 A-7　Phase 5 — 完整模型評估結果",
     "包含六個子圖：實際值 vs 預測值散點圖（R²=0.9001）、殘差 vs 預測值圖、殘差分佈直方圖、標準化迴歸係數長條圖、五折交叉驗證 R² 及效能指標彙總表。"),
]

# 附錄 HTML 片段
appendix_html = """<hr>
<h2>附錄：完整分析圖表</h2>
<p style="color:#57606a;font-size:10pt;">
以下圖表依 CRISP-DM 各分析階段排列，涵蓋資料理解（Phase 2）、資料準備（Phase 3）及模型評估（Phase 5）的完整視覺化輸出。
</p>
"""
for fname, title, caption in APPENDIX_CHARTS:
    fpath = os.path.join(CHART_DIR, fname)
    if not os.path.exists(fpath):
        continue
    import base64
    with open(fpath, "rb") as img_f:
        b64 = base64.b64encode(img_f.read()).decode()
    appendix_html += f"""
<div style="page-break-inside:avoid; margin-bottom:32px;">
  <h3 style="margin-bottom:6px;">{title}</h3>
  <img src="data:image/png;base64,{b64}"
       style="width:100%; border:1px solid #d0d7de; border-radius:6px;" />
  <p style="font-size:9.5pt; color:#57606a; margin-top:6px; line-height:1.6;">
    {caption}
  </p>
</div>
"""
appendix_html += "</section>"

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<style>
@page {{
  size: A4;
  margin: 2.2cm 2.4cm 2.2cm 2.4cm;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: {FONT};
  font-size: 11pt;
  line-height: 1.85;
  color: #1f2328;
  background: #ffffff;
}}
/* ── 標題 ── */
h1 {{
  font-size: 20pt; font-weight: 700; color: #0d1117;
  border-bottom: 3px solid #7b68ee; padding-bottom: 6px;
  margin-top: 0; page-break-after: avoid;
}}
h2 {{
  font-size: 14pt; font-weight: 700; color: #0d1117;
  border-left: 5px solid #7b68ee; padding-left: 10px;
  margin-top: 26px; page-break-after: avoid;
  background: linear-gradient(90deg, #f0edff 0%, transparent 80%);
  padding: 6px 10px;
}}
h3 {{
  font-size: 12pt; font-weight: 700; color: #24292f;
  margin-top: 18px; page-break-after: avoid;
  border-bottom: 1px solid #d8b4fe; padding-bottom: 3px;
}}
h4 {{
  font-size: 11pt; font-weight: 700; color: #57606a;
  margin-top: 14px;
}}
/* ── 表格 ── */
table {{
  width: 100%; border-collapse: collapse;
  font-size: 10pt; margin: 12px 0 16px;
  page-break-inside: auto;
}}
thead tr {{ background: #7b68ee; color: white; }}
thead th {{
  padding: 8px 10px; text-align: left;
  font-weight: 700; border: 1px solid #6758d4;
}}
tbody tr:nth-child(even) {{ background: #f6f8fa; }}
td {{
  padding: 6px 10px; border: 1px solid #d0d7de;
  vertical-align: top;
}}
/* ── 程式碼 ── */
code {{
  font-family: Consolas, "Courier New", monospace;
  font-size: 9.5pt; background: #f6f8fa;
  padding: 1px 5px; border-radius: 3px;
  border: 1px solid #d0d7de; color: #953800;
}}
pre {{
  background: #f6f8fa; border: 1px solid #d0d7de;
  border-radius: 6px; padding: 12px 14px;
  font-size: 9pt; line-height: 1.5;
  page-break-inside: avoid; overflow-x: auto;
}}
pre code {{
  background: none; border: none; padding: 0; color: #0550ae;
}}
/* ── 引用 ── */
blockquote {{
  margin: 10px 0; padding: 8px 14px;
  border-left: 4px solid #7b68ee; background: #f6f0ff;
  color: #57606a; border-radius: 0 5px 5px 0;
  font-size: 10.5pt;
}}
blockquote p {{ margin: 3px 0; }}
/* ── 其他 ── */
hr {{
  border: none; border-top: 2px solid #d0d7de; margin: 22px 0;
}}
a {{ color: #7b68ee; text-decoration: none; }}
strong {{ color: #0d1117; font-weight: 700; }}
p {{ margin: 5px 0 9px; }}
ul, ol {{ padding-left: 22px; margin: 5px 0 9px; }}
li {{ margin: 3px 0; }}

/* ── 頁首/頁尾（列印用） ── */
.page-header {{
  font-size: 9pt; color: #8b949e; text-align: center;
  margin-bottom: 16px;
  border-bottom: 1px solid #eaecef; padding-bottom: 4px;
}}
</style>
</head>
<body>
<div class="page-header">50 Startups 多元線性迴歸分析工作報告　|　winnieshih1107/50_Startups_hw6</div>
{html_body}
{appendix_html}
</body>
</html>"""

# ── 寫入暫存 HTML ─────────────────────────────────────────────────────────────
tmp_html = os.path.join(BASE, "_tmp_report.html")
with open(tmp_html, "w", encoding="utf-8") as f:
    f.write(HTML_TEMPLATE)

# ── Chrome headless 轉 PDF ───────────────────────────────────────────────────
print("Converting with Chrome headless...")
cmd = [
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=5000",
    f"--print-to-pdf={PDF_PATH}",
    "--print-to-pdf-no-header",
    f"file:///{tmp_html.replace(chr(92), '/')}",
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

# 清理暫存檔
try:
    os.remove(tmp_html)
except Exception:
    pass

if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
    size_kb = os.path.getsize(PDF_PATH) // 1024
    print(f"PDF saved: {PDF_PATH}  ({size_kb} KB)")
else:
    print(f"Error: {result.stderr[:300] if result.stderr else 'unknown'}")
