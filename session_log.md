# 50 Startups HW6 — 操作日誌 (Session Log)

**專案**：winnieshih1107/50_Startups_hw6  
**日期**：2026-06-12  
**工作目錄**：`D:\wi\260612`  
**模型**：Claude Sonnet 4.6 (claude-sonnet-4-6)

---

## Step 1 — 影片動畫效果

### Prompt
```
https://github.com/winnieshih1107/50_Startups_hw6/blob/main/hw6_presentation_pdf.mp4
影片增加動畫
```

### 執行動作
1. 從 GitHub 下載 12 張投影片 PNG（`slide_00.png`–`slide_11.png`）與旁白 MP3（`narr_00.mp3`–`narr_11.mp3`）
2. 撰寫 `encode_animated.py`：
   - 每張投影片套用 Ken Burns 縮放效果（偶數頁縮小→放大，奇數頁放大→縮小，幅度 8%）
   - ffmpeg `zoompan` + `fade` 濾鏡
   - xfade crossfade（0.6s）串接各片段
3. 執行腳本，產出動畫影片

### 輸出
- `hw6_presentation_pdf_animated.mp4`（49.5 MB，9.0 分鐘）
- `encode_animated.py`

---

## Step 2 — 輸出影片檔確認

### Prompt
```
輸出影片檔
```

### 執行動作
- 確認 `hw6_presentation_pdf_animated.mp4` 正確產出
- 回報檔案大小與時長

### 輸出
- 確認：49.5 MB，時長 9.0 分鐘

---

## Step 3 — 互動式 Web 應用程式（Streamlit）

### Prompt
```
用 https://github.com/winnieshih1107/50_Startups_hw6/tree/main
製作這種互動式網頁，最後要部署到 streamlit
[截圖：Startup Profit Predictor 介面]
```

### 執行動作
1. 撰寫 `app.py`（Streamlit + Plotly）：
   - 從 GitHub 載入 `50_Startups.csv`
   - 10 種特徵選取方法（Pearson/Spearman 相關性、SelectKBest F/MI、Sequential Forward、RFE-Linear/Forest、LassoCV、Random Forest、Gradient Boosting）
   - `@st.cache_data` 快取計算結果
   - 暗色主題 CSS（KPI 卡片、Tab 樣式）
   - 三個 Tab：完整比較表 / 方法檢查器 / 即時預測模擬器
2. 建立 `.streamlit/config.toml`（dark theme 設定）
3. 建立 `requirements.txt`
4. Git push 到 GitHub
5. 部署至 Streamlit Cloud

### 輸出
- `app.py`（Streamlit 應用程式）
- `.streamlit/config.toml`
- `requirements.txt`
- 部署網址：https://startup-profit-predictor.streamlit.app/

---

## Step 4 — UI 修正：背景對比度 / 字體大小

### Prompt
```
Connection error → http://localhost:8502/
https://startup-profit-predictor.streamlit.app/
背景太深看不清楚部分文字，文字太小
```

### 執行動作
1. 調整全域字體大小（`font-size: 17px`）
2. 加大 KPI 卡片字體（label: 15px，value: 42px，highlight: 32px）
3. 提高色彩對比度（`#e6edf3` 文字色）
4. Git push 更新

### 輸出
- `app.py`（更新：字體與對比度改善）

---

## Step 5 — UI 修正：字體放大 + Target 紅線

### Prompt
```
字體可以再放大，test r2 圖 target 0.9 紅線不明顯
```

### 執行動作
1. 再次放大全域字體（17px → 更大）
2. 加粗 Target R²=0.90 紅虛線（`width=3`）
3. Git push 更新

### 輸出
- `app.py`（更新：字體與紅線強化）

---

## Step 6 — UI 修正：Target 標注與圖表重疊

### Prompt
```
target 與圖重疊
[截圖：Pearson Corr 長條超過 0.9，標注壓在長條上]
```

### 執行動作
1. 診斷：Pearson Correlation bar 延伸至 ~0.93，超過 0.9，文字標注落在長條上
2. 改用獨立 `add_annotation` + `yref="paper"` + `y=1.0`（標注固定在圖表頂部）
3. 加上箭頭指向紅線，`xanchor="left"` 避免文字被遮蓋
4. Git push 更新

### 輸出
- `app.py`（更新：Target 標注移至圖表頂部）
- Commit：`91cdcdb` — "Fix: move Target 0.90 annotation above chart"

---

## Step 7 — SAM2 動畫影片

### Prompt
```
hw6_presentation_pdf_animated.mp4 → 影片沒有明顯動畫效果，
可以利用 Segment Anything(SAM) 技術，
讓簡報內容圖或文字有動畫效果
```

### 執行動作（第一版：SAM2，速度過慢）
1. 安裝 `ultralytics`、`opencv-python`
2. 撰寫 `sam_animate_video.py`：
   - 載入 SAM2（`sam2_b.pt`，154.4 MB）
   - 對每張投影片執行自動分割（`imgsz=1024`）
   - Spotlight 動畫：暗底 → 各區域逐一亮起 → 邊框閃光
   - ffmpeg rawvideo pipe 串流輸出（避免 temp PNG 磁碟消耗）
3. 執行後發現 SAM2 CPU 推理速度過慢（單張投影片超過 20 分鐘）

### 執行動作（第二版：OpenCV 快速版）
1. 終止 SAM2 任務
2. 撰寫 `fast_animate_video.py`：
   - 改用 OpenCV Canny + Dilate + findContours 偵測視覺區塊（毫秒級）
   - 保留相同的 Spotlight + 邊框閃光動畫效果
   - 加入 Ken Burns 輕微縮放（偶數頁縮小→放大，奇數頁反向）
   - 旁白延遲 0.5s 同步播放
3. 執行中（背景任務 `bhli47znd`，預計 25–30 分鐘）

### 輸出
- `sam_animate_video.py`（SAM2 版，保留供參考）
- `fast_animate_video.py`（OpenCV 快速版，正在執行）
- 預期輸出：`hw6_presentation_sam_animated.mp4`

---

## Step 8 — 工作報告（2000 字）+ 海報圖

### Prompt
```
https://github.com/winnieshih1107/50_Startups_hw6/tree/main
產出超過 2000 字圖表工作報告及入海報圖
[截圖：CRISP-DM 手繪流程圖]
```

### 執行動作
1. 從 GitHub 抓取 `50_Startups.csv`、`50_startups_crisp_dm.py`、`README.md`
2. 撰寫 `work_report.md`（**3,811 個中文字**）：
   - CRISP-DM 六階段完整分析
   - 資料診斷六項檢查（零值/偏態/離群值/尺度/VIF/相關性）
   - 五方法特徵選取多數決詳細表格
   - 迴歸係數解讀與業務意義
   - 投資策略建議與模型限制
3. 撰寫 `create_poster.py`，執行產出 `poster_50startups.png`（556 KB）：
   - CRISP-DM 六階段彩色卡片
   - KPI 卡片（R²=0.9001 / RMSE / MAE）
   - 實際值 vs 預測值散點圖
   - 標準化係數長條圖
   - 六方法 R² 比較圖
   - 五方法特徵選取矩陣
   - 5-Fold CV 圖
   - 核心發現與投資建議

### 輸出
- `work_report.md`（3,811 字，含完整表格與分析）
- `poster_50startups.png`（556 KB，8 個視覺化區塊）
- `create_poster.py`（海報生成腳本）

---

## Step 9 — 工作報告匯出 PDF

### Prompt
```
work_report.md 匯出 pdf
```

### 執行動作
1. 嘗試 WeasyPrint → 失敗（需要 libgobject/GTK，Windows 不支援）
2. 改用 **Chrome headless**（`--headless=new --print-to-pdf`）
3. 撰寫 `export_pdf.py`：
   - `markdown` library 轉換 MD → HTML
   - 嵌入繁體中文字體 CSS（Microsoft JhengHei）
   - Chrome headless 列印為 A4 PDF（含頁眉、表格樣式、程式碼高亮）
4. 輸出 789 KB PDF

### 輸出
- `work_report.pdf`（789 KB，A4 格式，含繁體中文）
- `export_pdf.py`（PDF 匯出腳本）

---

## Step 10 — 附錄圖表加入 PDF

### Prompt
```
work_report.pdf 加上附錄圖表
```

### 執行動作
1. 撰寫 `gen_charts.py`，執行產出 7 張附錄圖表：
   - `chart_phase2a.png`：特徵分佈直方圖 + 相關係數熱力圖
   - `chart_phase2b.png`：散點圖 + 箱型圖
   - `chart_phase3a.png`：五方法選取矩陣 + 投票長條圖
   - `chart_rf_importance.png`：隨機森林特徵重要性
   - `chart_phase3b.png`：StandardScaler 前後比較
   - `chart_phase3c.png`：六方法 R² & RMSE 效能比較
   - `chart_phase5.png`：完整評估（Actual vs Predicted / 殘差 / 係數 / CV）
2. 更新 `export_pdf.py`：將圖表以 base64 嵌入 HTML，追加附錄章節
3. 重新匯出 PDF

### 輸出
- 7 張圖表 PNG（`chart_phase*.png`，共 688 KB）
- `work_report.pdf`（更新：1,519 KB，包含附錄圖表）
- `gen_charts.py`（圖表生成腳本）

---

## Step 11 — 推送所有結果至 GitHub

### Prompt
```
將結果同步 push 到 https://github.com/winnieshih1107/50_Startups_hw6/tree/main
並記錄 log 每個步驟下的 prompt
```

### 執行動作
1. 建立本操作日誌 `session_log.md`
2. 複製所有新增/更新檔案至 `repo_tmp/`
3. `git add` + `git commit` + `git push`

### 推送檔案清單

| 檔案 | 類型 | 說明 |
|------|------|------|
| `work_report.md` | 文件 | 3,811 字 CRISP-DM 完整分析報告 |
| `work_report.pdf` | 文件 | PDF 版本（含附錄圖表，1.52 MB） |
| `poster_50startups.png` | 圖像 | A3 海報（556 KB，8 視覺化區塊） |
| `chart_phase2a.png` | 圖像 | Phase 2a 分佈圖 + 相關熱力圖 |
| `chart_phase2b.png` | 圖像 | Phase 2b 散點圖 + 箱型圖 |
| `chart_phase3a.png` | 圖像 | Phase 3a 特徵選取矩陣 |
| `chart_phase3b.png` | 圖像 | Phase 3b 標準化前後比較 |
| `chart_phase3c.png` | 圖像 | Phase 3c 六方法效能比較 |
| `chart_phase5.png` | 圖像 | Phase 5 完整評估圖 |
| `chart_rf_importance.png` | 圖像 | 隨機森林特徵重要性 |
| `fast_animate_video.py` | 腳本 | OpenCV Spotlight 動畫生成器 |
| `sam_animate_video.py` | 腳本 | SAM2 動畫生成器（備用） |
| `create_poster.py` | 腳本 | 海報生成腳本 |
| `gen_charts.py` | 腳本 | 附錄圖表生成腳本 |
| `export_pdf.py` | 腳本 | Markdown → PDF 轉換腳本 |
| `session_log.md` | 文件 | 本操作日誌 |

---

---

## Step 12 — README 加上 Live Demo 及海報圖

### Prompt
```
README加上live demo
README 加上海報圖（附件：手繪 CRISP-DM infographic）
Project Poster替換附件圖
```

### 執行動作
1. README 頂部加入 Streamlit badge 與 Live Demo callout 連結
2. README 加入 `## Project Poster` 區段，嵌入 `poster_50startups.png`
3. 將手繪 CRISP-DM 海報（`Gemini_Generated_Image_2j3iqz2j3iqz2j3i.png`）複製為 `poster_50startups.png`，取代程式生成版本
4. Git push 三次（commit `97b666a`, `349dd38`, `8b1e479`）

### 輸出
- `README.md`（更新：Live Demo badge + 海報圖區段）
- `poster_50startups.png`（更新：手繪版 CRISP-DM infographic，5.3 MB）

---

## Step 13 — 動畫影片生成（持續進行中）

### Prompt
```
任務完成後會自動通知，產出 hw6_presentation_sam_animated.mp4
```

### 執行動作
1. 原始 SAM2 版本（`sam_animate_video.py`）太慢，已棄用
2. OpenCV 版 `fast_animate_video.py`：
   - **v1**：每幀 Ken Burns resize（1920×1080 float32），GC 壓力過大 → ~8 min/slide
   - **v2**：移除 Ken Burns、in-place ops → 仍有 boolean scatter 問題
   - **v3（現行）**：全面向量化（`np.multiply` + `np.maximum` 取代 boolean gather/scatter） → ~2-3 min/slide（加速 3×）
3. 已完成片段（截至切換電腦時）：
   - `sam_seg_00.mp4`（5.4 MB，完成於 11:39）
   - `sam_seg_01.mp4`（5.4 MB，完成於 12:07）
   - `sam_seg_02.mp4`（1.8 MB，完成於 12:29）
   - `sam_seg_03.mp4`（1.5 MB，完成於 12:33）

### 新電腦繼續執行方式
```bash
# 1. 確認環境
pip install opencv-python imageio-ffmpeg numpy Pillow

# 2. 確認已有的片段（不需要重新渲染）
ls D:\wi\260612\pres_pdf_assets\sam_tmp\

# 3. 執行腳本（已存在的片段會自動跳過）
cd D:\wi\260612
python fast_animate_video.py

# 4. 完成後 push 到 GitHub
cd D:\wi\260612\repo_tmp
cp ..\hw6_presentation_sam_animated.mp4 .
git add hw6_presentation_sam_animated.mp4
git commit -m "Add SAM-style spotlight animation video"
git push origin main
```

### 注意事項
- 片段檔案位於 `D:\wi\260612\pres_pdf_assets\sam_tmp\`，已完成的會自動跳過
- `fast_animate_video.py` 已是 v3 優化版（向量化 spotlight + veryfast h264）
- 若新電腦路徑不同，修改腳本第 16–17 行的 `BASE` 和 `ASSETS` 路徑
- 預計剩餘 8 張投影片 × ~2-3 min = 約 20-25 分鐘

### 預期輸出
- `hw6_presentation_sam_animated.mp4`（預計 ~60 MB，完整 12 張投影片動畫）

---

## 技術堆疊彙整

| 類別 | 工具/套件 |
|------|----------|
| 資料分析 | pandas, numpy, scikit-learn, statsmodels |
| 視覺化 | matplotlib, seaborn, plotly |
| Web 應用 | streamlit, plotly |
| 影片處理 | ffmpeg (rawvideo pipe), imageio-ffmpeg, moviepy |
| 圖像分割 | ultralytics (SAM2), opencv-python |
| PDF 輸出 | markdown, Chrome headless |
| 版本控制 | git, GitHub |
| 部署 | Streamlit Cloud |

---

## GitHub 最新狀態（截至 2026-06-12 12:34）

| Commit | 說明 |
|--------|------|
| `8b1e479` | Update poster: replace with hand-drawn CRISP-DM infographic |
| `349dd38` | README: add project poster image |
| `97b666a` | README: add Live Demo badge and link to Streamlit app |
| `09a45d4` | Add work report, poster, charts, animation scripts, session log |

**Live Demo**：https://startup-profit-predictor.streamlit.app/  
**GitHub**：https://github.com/winnieshih1107/50_Startups_hw6
