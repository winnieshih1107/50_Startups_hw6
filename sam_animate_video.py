"""
sam_animate_video.py
────────────────────────────────────────────────────────────────
SAM2 (Segment Anything Model 2) 驅動的動畫投影片影片生成器

效果：
  • SAM2 自動偵測每張投影片中的視覺區域（圖表、文字塊、圖形）
  • 每個區域依上→下讀取順序逐一「spotlight 亮起」
  • 進場：暗底 → 亮起邊框閃光 → 完全清晰
  • 旁白音訊從動畫啟動後 0.5s 開始（動畫與旁白同步進行）
  • 最後 0.5s 淡出至黑

備用方案（SAM2 無法偵測時）：水平三帶分割逐一亮起
"""

import os, subprocess, numpy as np
from PIL import Image, ImageFilter
import imageio_ffmpeg
from moviepy import AudioFileClip

BASE     = r"D:\wi\260612"
ASSETS   = os.path.join(BASE, "pres_pdf_assets")
TMP_DIR  = os.path.join(ASSETS, "sam_tmp")
OUT_MP4  = os.path.join(BASE, "hw6_presentation_sam_animated.mp4")
FFMPEG   = imageio_ffmpeg.get_ffmpeg_exe()

FPS      = 24
N_SLIDES = 12
TARGET_W, TARGET_H = 1920, 1080

# 動畫時序參數
BG_DUR     = 0.55   # 背景從暗到半亮的時間（秒）
REVEAL_DUR = 0.65   # 每個區域亮起所需時間
GAP        = 0.40   # 每個區域的啟動間隔
NARR_DELAY = 0.50   # 旁白延遲（留給背景亮起）
HOLD_END   = 0.80   # 旁白結束後靜止時間
FADE_OUT   = 0.50   # 淡出時間

os.makedirs(TMP_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SAM2 初始化
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SAM2 動畫影片生成器")
print("=" * 60)

USE_SAM = False
try:
    from ultralytics import SAM
    print("載入 SAM2 模型（sam2_b.pt）…首次使用自動下載 ~80MB")
    _sam_model = SAM("sam2_b.pt")
    USE_SAM = True
    print("  SAM2 就緒 ✓")
except Exception as e:
    print(f"  SAM2 無法載入（{e}），使用備用分割")

# ══════════════════════════════════════════════════════════════════════════════
# 分割引擎
# ══════════════════════════════════════════════════════════════════════════════
def get_masks(img_path: str):
    """回傳 bool masks list，按 y 中心由上到下排序，最多 8 個"""
    H, W = TARGET_H, TARGET_W
    total = H * W

    raw_masks = []

    if USE_SAM:
        try:
            results = _sam_model(img_path, imgsz=1024, verbose=False)
            if results and results[0].masks is not None:
                raw_masks = results[0].masks.data.cpu().numpy()  # (N,H,W)
        except Exception as e:
            print(f"  SAM 推理失敗: {e}")

    # 備用：OpenCV 輪廓偵測
    if len(raw_masks) == 0:
        raw_masks = _contour_masks(img_path, H, W)

    # 最終備用：三條水平帶
    if len(raw_masks) == 0:
        return _horizontal_bands(H, W, 3)

    # 過濾：面積 1%~70%
    filtered = []
    for m in raw_masks:
        area = m.astype(bool).sum() / total
        if 0.01 <= area <= 0.70:
            filtered.append(m.astype(bool))

    if not filtered:
        return _horizontal_bands(H, W, 3)

    # 去重（IoU > 0.8 的取面積較大者）
    filtered = _deduplicate(filtered)

    # 依 y 中心排序（上→下）
    def y_center(m):
        rows = np.where(m.any(axis=1))[0]
        return float(rows.mean()) if len(rows) else 9999.0

    filtered.sort(key=y_center)
    return filtered[:8]


def _contour_masks(img_path, H, W):
    try:
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        img_resized = cv2.resize(img, (W, H))
        edges = cv2.Canny(img_resized, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        for cnt in contours:
            m = np.zeros((H, W), dtype=bool)
            cv2.fillPoly(m.view(np.uint8), [cnt], 1)
            masks.append(m)
        return masks
    except Exception:
        return []


def _horizontal_bands(H, W, n=3):
    masks = []
    band = H // n
    for i in range(n):
        m = np.zeros((H, W), dtype=bool)
        m[i * band:(i + 1) * band, :] = True
        masks.append(m)
    return masks


def _deduplicate(masks, iou_thresh=0.80):
    keep = []
    for i, a in enumerate(masks):
        dominated = False
        for j, b in enumerate(masks):
            if i == j:
                continue
            inter = (a & b).sum()
            union = (a | b).sum()
            if union > 0 and inter / union > iou_thresh and b.sum() > a.sum():
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return keep

# ══════════════════════════════════════════════════════════════════════════════
# Easing 函式
# ══════════════════════════════════════════════════════════════════════════════
def ease_out_quart(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return 1.0 - (1.0 - p) ** 4

def ease_out_cubic(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return 1.0 - (1.0 - p) ** 3

# ══════════════════════════════════════════════════════════════════════════════
# 單幀計算
# ══════════════════════════════════════════════════════════════════════════════
def compute_frame(slide_f32: np.ndarray,
                  masks: list,
                  mask_starts: list,
                  t: float,
                  total_dur: float) -> np.ndarray:
    """
    計算 t 秒時的畫面。
    darkness_map 值域 [0, 1]：0 = 全黑，1 = 原色
    """
    H, W = slide_f32.shape[:2]

    # ── 全域基礎亮度（背景暗區）────────────────────────────────────────────
    bg_p  = ease_out_cubic(min(1.0, t / BG_DUR))
    base_brightness = 0.18 + bg_p * 0.22   # 0.18 → 0.40

    darkness = np.full((H, W), base_brightness, dtype=np.float32)

    # ── 各區域逐一亮起 ────────────────────────────────────────────────────
    for mask, ms in zip(masks, mask_starts):
        if t < ms:
            continue
        p = ease_out_quart(min(1.0, (t - ms) / REVEAL_DUR))
        target = 0.40 + p * 0.60   # 40% → 100%
        darkness[mask] = np.maximum(darkness[mask], target)

        # 邊框閃光（進場後 0.25s 內）
        flash_dur = 0.25
        if t < ms + flash_dur:
            flash_p = 1.0 - (t - ms) / flash_dur
            flash   = ease_out_quart(flash_p) * 0.35
            darkness[mask] = np.minimum(1.0, darkness[mask] + flash)

    # ── 動畫結束後：完全清晰 ──────────────────────────────────────────────
    anim_end = BG_DUR + len(masks) * GAP + REVEAL_DUR
    if t >= anim_end:
        finish_p = ease_out_cubic(min(1.0, (t - anim_end) / 0.40))
        darkness  = darkness + (1.0 - darkness) * finish_p
        darkness  = np.minimum(1.0, darkness)

    # ── 淡出 ──────────────────────────────────────────────────────────────
    if t >= total_dur - FADE_OUT:
        fade = max(0.0, (total_dur - t) / FADE_OUT)
        darkness *= fade

    frame = (slide_f32 * darkness[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    return frame

# ══════════════════════════════════════════════════════════════════════════════
# 每個片段編碼（ffmpeg rawvideo pipe + 音訊）
# ══════════════════════════════════════════════════════════════════════════════
def encode_segment(slide_arr, masks, mp3_path, out_path):
    """使用 ffmpeg rawvideo pipe 編碼動畫片段，同步旁白音訊"""
    with AudioFileClip(mp3_path) as a:
        narr_dur = a.duration

    n_masks  = len(masks)
    anim_end = BG_DUR + n_masks * GAP + REVEAL_DUR
    total_dur = max(NARR_DELAY + narr_dur, anim_end + 0.3) + HOLD_END + FADE_OUT

    total_frames  = int(total_dur * FPS)
    narr_delay_ms = int(NARR_DELAY * 1000)

    # 預計算 mask_starts
    mask_starts = [BG_DUR + i * GAP for i in range(n_masks)]

    slide_f32 = slide_arr.astype(np.float32)

    # ── 靜音影片 via pipe ─────────────────────────────────────────────────
    silent_mp4 = out_path.replace(".mp4", "_silent.mp4")
    cmd_video = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{TARGET_W}x{TARGET_H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        silent_mp4,
    ]
    proc = subprocess.Popen(cmd_video, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    for f_idx in range(total_frames):
        t     = f_idx / FPS
        frame = compute_frame(slide_f32, masks, mask_starts, t, total_dur)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg video encoding failed")

    # ── 合併音訊（延遲 NARR_DELAY 秒啟動）────────────────────────────────
    cmd_audio = [
        FFMPEG, "-y",
        "-i", silent_mp4,
        "-i", mp3_path,
        "-filter_complex",
        (f"[1:a]adelay={narr_delay_ms}|{narr_delay_ms}[a_del];"
         f"[a_del]apad[a_out]"),
        "-map", "0:v",
        "-map", "[a_out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd_audio, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        import shutil
        print(f"    音訊合併警告，改用靜音版\n    {result.stderr[-200:]}")
        shutil.copy(silent_mp4, out_path)

    os.remove(silent_mp4)
    return total_dur

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
slide_pngs  = [os.path.join(ASSETS, f"slide_{i:02d}.png") for i in range(N_SLIDES)]
audio_files = [os.path.join(ASSETS, f"narr_{i:02d}.mp3")  for i in range(N_SLIDES)]

for p in slide_pngs + audio_files:
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到: {p}")

segment_files = []

for i, (png, mp3) in enumerate(zip(slide_pngs, audio_files)):
    seg = os.path.join(TMP_DIR, f"sam_seg_{i:02d}.mp4")

    if os.path.exists(seg):
        print(f"[{i+1:02d}/{N_SLIDES}] 已存在，跳過重算")
        segment_files.append(seg)
        continue

    print(f"\n[{i+1:02d}/{N_SLIDES}] slide_{i:02d}.png")

    # 偵測視覺區域
    print(f"  偵測視覺區域…", end=" ", flush=True)
    masks = get_masks(png)
    print(f"{len(masks)} 個區域")
    for j, m in enumerate(masks):
        area_pct = m.sum() / (TARGET_H * TARGET_W) * 100
        y_rows = np.where(m.any(axis=1))[0]
        y_c = y_rows.mean() if len(y_rows) else 0
        print(f"    區域 {j+1}: 面積 {area_pct:.1f}%  y中心 {y_c:.0f}px")

    # 載入投影片
    slide_img = Image.open(png).convert("RGB").resize((TARGET_W, TARGET_H))
    slide_arr = np.array(slide_img)

    anim_dur = BG_DUR + len(masks) * GAP + REVEAL_DUR
    with AudioFileClip(mp3) as a:
        narr_dur = a.duration
    total_est = max(NARR_DELAY + narr_dur, anim_dur + 0.3) + HOLD_END + FADE_OUT
    total_frames = int(total_est * FPS)

    print(f"  渲染 {total_frames} frames（{total_est:.1f}s）…", end=" ", flush=True)
    dur = encode_segment(slide_arr, masks, mp3, seg)
    size_kb = os.path.getsize(seg) // 1024
    print(f"完成  ({size_kb} KB)")
    segment_files.append(seg)

# ── 串接所有片段 ──────────────────────────────────────────────────────────────
print(f"\n串接 {len(segment_files)} 個片段…")
concat_list = os.path.join(TMP_DIR, "concat_sam.txt")
with open(concat_list, "w", encoding="utf-8") as f:
    for seg in segment_files:
        f.write(f"file '{seg.replace(os.sep, '/')}'\n")

cmd = [FFMPEG, "-y",
       "-f", "concat", "-safe", "0", "-i", concat_list,
       "-c", "copy", "-movflags", "+faststart",
       OUT_MP4]
subprocess.run(cmd, check=True, capture_output=True)

size_mb = os.path.getsize(OUT_MP4) / 1024 / 1024
print(f"\n[完成]")
print(f"  輸出：{OUT_MP4}")
print(f"  大小：{size_mb:.1f} MB")
print(f"\n動畫效果：")
print(f"  • SAM2 自動偵測每張投影片的視覺區域")
print(f"  • 背景從暗淡逐步亮起，各區域依序 spotlight 進場")
print(f"  • 每個區域進場時有邊框閃光提示")
print(f"  • 旁白在動畫啟動 {NARR_DELAY}s 後開始同步播放")
