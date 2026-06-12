"""
fast_animate_video.py
─────────────────────────────────────────────────────────────────
快速 spotlight 動畫影片生成器（OpenCV 區域偵測，無需 SAM2）

效果：
  • 每張投影片從全黑逐漸亮起（Ken Burns 輕微縮放）
  • OpenCV 偵測投影片中的主要視覺區塊（文字段落、圖表、圖形）
  • 各區塊依上→下順序逐一 spotlight 亮起，帶邊框光暈
  • 旁白在動畫開始 0.5s 後同步播放
  • 最後 0.6s 淡出至黑，接下一張

預計總時間：5–10 分鐘（12 張投影片）
"""
import os, sys, subprocess
import numpy as np
from PIL import Image
import imageio_ffmpeg
import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE     = r"D:\wi\260612"
ASSETS   = os.path.join(BASE, "pres_pdf_assets")
TMP_DIR  = os.path.join(ASSETS, "sam_tmp")
OUT_MP4  = os.path.join(BASE, "hw6_presentation_sam_animated.mp4")
FFMPEG   = imageio_ffmpeg.get_ffmpeg_exe()

FPS           = 24
N_SLIDES      = 12
W, H          = 1920, 1080
MAX_REGIONS   = 7    # 每張最多亮起幾個區域

# 動畫時序
BG_DUR     = 0.50   # 背景從暗→半亮
REVEAL_DUR = 0.55   # 每個區域亮起時間
GAP        = 0.35   # 各區域啟動間隔
NARR_DELAY = 0.50   # 旁白延遲（秒）
HOLD_END   = 0.80   # 旁白結束後靜止
FADE_OUT   = 0.55   # 淡出時間

os.makedirs(TMP_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 區域偵測：OpenCV 輪廓 + 固定分割備援
# ─────────────────────────────────────────────────────────────────────────────
def detect_regions(img_bgr):
    """回傳 bool mask list，依 y 中心由上到下排序，最多 MAX_REGIONS 個"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 二值化（Otsu 自動門檻）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形態學膨脹：把附近的字/線合併成一個大區塊
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 18))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = H * W
    masks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 篩掉太小（< 2%）或太大（> 70%）的區域
        area_ratio = (w * h) / total
        if area_ratio < 0.02 or area_ratio > 0.70:
            continue
        # 篩掉太薄（高寬比異常）
        if h < 15 or w < 50:
            continue
        m = np.zeros((H, W), dtype=bool)
        # 用矩形 mask（比輪廓填充更清晰）
        pad = 8
        y1 = max(0, y - pad);  y2 = min(H, y + h + pad)
        x1 = max(0, x - pad);  x2 = min(W, x + w + pad)
        m[y1:y2, x1:x2] = True
        masks.append(m)

    # 若輪廓偵測結果不足，補上水平帶分割
    if len(masks) < 3:
        masks = _horizontal_bands(4)

    # 去重（IoU > 0.6）
    masks = _deduplicate(masks, iou_thresh=0.60)

    # 依 y 中心排序
    def y_center(m):
        rows = np.where(m.any(axis=1))[0]
        return float(rows.mean()) if len(rows) else 9999.0

    masks.sort(key=y_center)
    return masks[:MAX_REGIONS]


def _horizontal_bands(n=4):
    """備援：把投影片切成 n 條水平帶"""
    masks = []
    band = H // n
    for i in range(n):
        m = np.zeros((H, W), dtype=bool)
        m[i*band:(i+1)*band, :] = True
        masks.append(m)
    return masks


def _deduplicate(masks, iou_thresh=0.60):
    keep = []
    for i, a in enumerate(masks):
        skip = False
        for j, b in enumerate(masks):
            if i == j:
                continue
            inter = (a & b).sum()
            union = (a | b).sum()
            if union > 0 and inter/union > iou_thresh and b.sum() > a.sum():
                skip = True
                break
        if not skip:
            keep.append(a)
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Easing
# ─────────────────────────────────────────────────────────────────────────────
def ease_out_cubic(p):
    p = max(0.0, min(1.0, p))
    return 1.0 - (1.0 - p) ** 3

def ease_out_quart(p):
    p = max(0.0, min(1.0, p))
    return 1.0 - (1.0 - p) ** 4


# ─────────────────────────────────────────────────────────────────────────────
# 單幀計算
# ─────────────────────────────────────────────────────────────────────────────
def compute_frame(slide_f32, masks, mask_starts, t, total_dur,
                  zoom_start=1.0, zoom_end=1.06):
    """
    darkness_map: 0=全黑, 1=原色
    輕微 Ken Burns 縮放（zoom_start→zoom_end）
    """
    # ── 縮放（Ken Burns） ──────────────────────────────────────────────────
    zoom_p = min(1.0, t / total_dur)
    scale  = zoom_start + (zoom_end - zoom_start) * zoom_p
    if abs(scale - 1.0) > 0.001:
        new_w = int(W * scale)
        new_h = int(H * scale)
        resized = cv2.resize(
            slide_f32.astype(np.uint8), (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        x0 = (new_w - W) // 2;  y0 = (new_h - H) // 2
        frame_base = resized[y0:y0+H, x0:x0+W]
    else:
        frame_base = slide_f32

    # ── 亮度遮罩 ───────────────────────────────────────────────────────────
    bg_p  = ease_out_cubic(min(1.0, t / BG_DUR))
    base_b = 0.15 + bg_p * 0.25      # 0.15 → 0.40

    darkness = np.full((H, W), base_b, dtype=np.float32)

    for mask, ms in zip(masks, mask_starts):
        if t < ms:
            continue
        p = ease_out_quart(min(1.0, (t - ms) / REVEAL_DUR))
        target = 0.40 + p * 0.60     # 40% → 100%
        darkness[mask] = np.maximum(darkness[mask], target)

        # 邊框閃光（進場 0.20s 內）
        flash_dur = 0.20
        if t < ms + flash_dur:
            flash_p = 1.0 - (t - ms) / flash_dur
            darkness[mask] = np.minimum(1.0, darkness[mask] + ease_out_quart(flash_p) * 0.30)

    # ── 動畫結束後全亮 ─────────────────────────────────────────────────────
    anim_end = BG_DUR + len(masks) * GAP + REVEAL_DUR
    if t >= anim_end:
        finish_p = ease_out_cubic(min(1.0, (t - anim_end) / 0.35))
        darkness  = darkness + (1.0 - darkness) * finish_p

    # ── 淡出 ───────────────────────────────────────────────────────────────
    if t >= total_dur - FADE_OUT:
        fade = max(0.0, (total_dur - t) / FADE_OUT)
        darkness *= fade

    frame = (frame_base * darkness[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 片段編碼（rawvideo pipe）
# ─────────────────────────────────────────────────────────────────────────────
def get_audio_duration(mp3_path):
    """用 ffprobe 取得音訊時長（秒）"""
    cmd = [FFMPEG.replace("ffmpeg", "ffprobe"),
           "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", mp3_path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        # 備援：用 ffmpeg 解碼測量長度
        cmd2 = [FFMPEG, "-i", mp3_path, "-f", "null", "-"]
        r = subprocess.run(cmd2, capture_output=True, text=True)
        for line in r.stderr.split("\n"):
            if "Duration" in line:
                t = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = t.split(":")
                return int(h)*3600 + int(m)*60 + float(s)
        return 30.0


def encode_segment(slide_arr, masks, mp3_path, out_path, slide_idx):
    narr_dur = get_audio_duration(mp3_path)

    n_masks   = len(masks)
    anim_end  = BG_DUR + n_masks * GAP + REVEAL_DUR
    total_dur = max(NARR_DELAY + narr_dur, anim_end + 0.4) + HOLD_END + FADE_OUT
    total_frames   = int(total_dur * FPS)
    narr_delay_ms  = int(NARR_DELAY * 1000)
    mask_starts    = [BG_DUR + i * GAP for i in range(n_masks)]

    # 偶數張：縮小→放大；奇數張：放大→縮小
    zoom_s, zoom_e = (1.0, 1.06) if slide_idx % 2 == 0 else (1.06, 1.0)

    slide_f32 = slide_arr.astype(np.float32)

    # ── 靜音影片 via rawvideo pipe ─────────────────────────────────────────
    silent_mp4 = out_path.replace(".mp4", "_silent.mp4")
    cmd_video = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
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
        frame = compute_frame(slide_f32, masks, mask_starts, t, total_dur,
                              zoom_start=zoom_s, zoom_end=zoom_e)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg video pipe failed (slide {slide_idx})")

    # ── 合併音訊 ───────────────────────────────────────────────────────────
    cmd_audio = [
        FFMPEG, "-y",
        "-i", silent_mp4,
        "-i", mp3_path,
        "-filter_complex",
        f"[1:a]adelay={narr_delay_ms}|{narr_delay_ms}[a_del];"
        f"[a_del]apad[a_out]",
        "-map", "0:v", "-map", "[a_out]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", out_path,
    ]
    r = subprocess.run(cmd_audio, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        import shutil
        shutil.copy(silent_mp4, out_path)

    try:
        os.remove(silent_mp4)
    except Exception:
        pass
    return total_dur


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
slide_pngs  = [os.path.join(ASSETS, f"slide_{i:02d}.png") for i in range(N_SLIDES)]
audio_files = [os.path.join(ASSETS, f"narr_{i:02d}.mp3")  for i in range(N_SLIDES)]

for p in slide_pngs + audio_files:
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到: {p}")

print("=" * 60)
print("快速 Spotlight 動畫影片生成器")
print("=" * 60)

segment_files = []

for i, (png, mp3) in enumerate(zip(slide_pngs, audio_files)):
    seg = os.path.join(TMP_DIR, f"sam_seg_{i:02d}.mp4")

    if os.path.exists(seg) and os.path.getsize(seg) > 50000:
        print(f"[{i+1:02d}/{N_SLIDES}] 已存在，跳過")
        segment_files.append(seg)
        continue

    print(f"\n[{i+1:02d}/{N_SLIDES}] slide_{i:02d}.png", end="  ", flush=True)

    # 載入投影片（BGR for OpenCV, RGB for rendering）
    img_bgr = cv2.imread(png)
    img_bgr = cv2.resize(img_bgr, (W, H))
    slide_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 偵測區域
    masks = detect_regions(img_bgr)
    print(f"偵測到 {len(masks)} 個區域", flush=True)

    # 估算時長
    narr_dur = get_audio_duration(mp3)
    anim_end  = BG_DUR + len(masks) * GAP + REVEAL_DUR
    total_est = max(NARR_DELAY + narr_dur, anim_end + 0.4) + HOLD_END + FADE_OUT
    total_frames = int(total_est * FPS)
    print(f"  渲染 {total_frames} frames（{total_est:.1f}s）…", end=" ", flush=True)

    dur = encode_segment(slide_rgb, masks, mp3, seg, i)
    size_kb = os.path.getsize(seg) // 1024
    print(f"完成  ({size_kb} KB)")
    segment_files.append(seg)

# ── 串接 ──────────────────────────────────────────────────────────────────────
print(f"\n串接 {len(segment_files)} 個片段…", flush=True)
concat_list = os.path.join(TMP_DIR, "concat_fast.txt")
with open(concat_list, "w", encoding="utf-8") as f:
    for seg in segment_files:
        f.write(f"file '{seg.replace(os.sep, '/')}'\n")

cmd_concat = [
    FFMPEG, "-y",
    "-f", "concat", "-safe", "0", "-i", concat_list,
    "-c", "copy", "-movflags", "+faststart",
    OUT_MP4,
]
subprocess.run(cmd_concat, check=True, capture_output=True)

size_mb = os.path.getsize(OUT_MP4) / 1024 / 1024
print(f"\n完成！")
print(f"  輸出：{OUT_MP4}")
print(f"  大小：{size_mb:.1f} MB")
