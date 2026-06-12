"""
fast_animate_video.py  v3
快速 Spotlight 動畫影片生成器（OpenCV 區域偵測）

優化重點：
  • 移除 Ken Burns（消除每幀 25 MB resize 開銷）
  • In-place numpy ops（消除每幀 3×8 MB GC 壓力）
  • 預分配 numpy buffer（frame_f32, frame_u8, tmp）
  • veryfast preset（h264 編碼速度提升 2×）
  • 沿用已存在的 sam_seg_*.mp4（跳過已完成片段）
"""
import os, sys, subprocess
import numpy as np
import imageio_ffmpeg
import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE     = r"D:\wi\260612"
ASSETS   = os.path.join(BASE, "pres_pdf_assets")
TMP_DIR  = os.path.join(ASSETS, "sam_tmp")
OUT_MP4  = os.path.join(BASE, "hw6_presentation_sam_animated.mp4")
FFMPEG   = imageio_ffmpeg.get_ffmpeg_exe()

FPS         = 24
N_SLIDES    = 12
W, H        = 1920, 1080
MAX_REGIONS = 7

BG_DUR     = 0.50
REVEAL_DUR = 0.55
GAP        = 0.35
NARR_DELAY = 0.50
HOLD_END   = 0.80
FADE_OUT   = 0.55

os.makedirs(TMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 區域偵測
# ─────────────────────────────────────────────────────────────────────────────
def detect_regions(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 18))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = H * W
    masks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h) / total < 0.02 or (w * h) / total > 0.70:
            continue
        if h < 15 or w < 50:
            continue
        m = np.zeros((H, W), dtype=bool)
        pad = 8
        m[max(0,y-pad):min(H,y+h+pad), max(0,x-pad):min(W,x+w+pad)] = True
        masks.append(m)
    if len(masks) < 3:
        masks = [np.zeros((H, W), dtype=bool) for _ in range(4)]
        band = H // 4
        for i, m in enumerate(masks):
            m[i*band:(i+1)*band, :] = True
    # 去重
    keep = []
    for i, a in enumerate(masks):
        if any(i != j and (a & b).sum() / max((a | b).sum(), 1) > 0.60 and b.sum() > a.sum()
               for j, b in enumerate(masks)):
            continue
        keep.append(a)
    keep.sort(key=lambda m: float(np.where(m.any(axis=1))[0].mean()) if m.any() else 9999.0)
    return keep[:MAX_REGIONS]


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
# 亮度遮罩（in-place，無臨時陣列分配）
# ─────────────────────────────────────────────────────────────────────────────
def fill_darkness(out, tmp, masks_f32, mask_starts, t, total_dur):
    """
    out       : float32 (H, W) — 輸出亮度遮罩，值域 0→1
    tmp       : float32 (H, W) — 工作暫存（預分配，重複使用）
    masks_f32 : list of float32 (H, W) — 預轉換的 float mask（0.0/1.0）

    全部使用 element-wise vectorized ops，無 boolean gather/scatter。
    """
    bg_p   = ease_out_cubic(min(1.0, t / BG_DUR))
    base_b = 0.15 + bg_p * 0.25        # 0.15 → 0.40

    out[:] = base_b

    for mf, ms in zip(masks_f32, mask_starts):
        if t < ms:
            continue
        p      = ease_out_quart(min(1.0, (t - ms) / REVEAL_DUR))
        target = 0.40 + p * 0.60       # 40% → 100%

        # tmp = mf * target  （unmasked = 0, masked = target）
        # out = max(out, tmp) — unmasked: max(out, 0)=out; masked: max(out, target)
        np.multiply(mf, target, out=tmp)
        np.maximum(out, tmp, out=out)

        # 邊框閃光（進場 0.20s）—— in-place add on masked region
        flash_dur = 0.20
        if t < ms + flash_dur:
            flash_p = 1.0 - (t - ms) / flash_dur
            flash   = ease_out_quart(flash_p) * 0.30
            # tmp = mf * flash, then out += tmp, clip to 1.0
            np.multiply(mf, flash, out=tmp)
            np.add(out, tmp, out=out)
            np.minimum(out, 1.0, out=out)

    # 動畫結束後全亮（in-place，使用 tmp）
    anim_end = BG_DUR + len(masks_f32) * GAP + REVEAL_DUR
    if t >= anim_end:
        finish_p = ease_out_cubic(min(1.0, (t - anim_end) / 0.35))
        np.subtract(1.0, out, out=tmp)
        np.multiply(tmp, finish_p, out=tmp)
        np.add(out, tmp, out=out)

    # 淡出（in-place）
    if t >= total_dur - FADE_OUT:
        fade = max(0.0, (total_dur - t) / FADE_OUT)
        np.multiply(out, fade, out=out)


# ─────────────────────────────────────────────────────────────────────────────
# 音訊時長
# ─────────────────────────────────────────────────────────────────────────────
def get_audio_duration(mp3_path):
    ffprobe = FFMPEG.replace("ffmpeg", "ffprobe")
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", mp3_path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        cmd2 = [FFMPEG, "-i", mp3_path, "-f", "null", "-"]
        r = subprocess.run(cmd2, capture_output=True, text=True)
        for line in r.stderr.split("\n"):
            if "Duration" in line:
                h2, m2, s2 = line.split("Duration:")[1].split(",")[0].strip().split(":")
                return int(h2)*3600 + int(m2)*60 + float(s2)
        return 30.0


# ─────────────────────────────────────────────────────────────────────────────
# 片段編碼
# ─────────────────────────────────────────────────────────────────────────────
def encode_segment(slide_arr, masks, mp3_path, out_path, slide_idx):
    narr_dur      = get_audio_duration(mp3_path)
    n_masks       = len(masks)
    anim_end      = BG_DUR + n_masks * GAP + REVEAL_DUR
    total_dur     = max(NARR_DELAY + narr_dur, anim_end + 0.4) + HOLD_END + FADE_OUT
    total_frames  = int(total_dur * FPS)
    narr_delay_ms = int(NARR_DELAY * 1000)
    mask_starts   = [BG_DUR + i * GAP for i in range(n_masks)]

    slide_f32  = slide_arr.astype(np.float32)           # (H, W, 3) values 0–255
    masks_f32  = [m.astype(np.float32) for m in masks]  # 預轉換，避免 per-frame 轉型

    # 預分配 buffer（整個 slide 只分配一次）
    darkness  = np.empty((H, W),    dtype=np.float32)
    tmp_dark  = np.empty((H, W),    dtype=np.float32)
    frame_f32 = np.empty((H, W, 3), dtype=np.float32)
    frame_u8  = np.empty((H, W, 3), dtype=np.uint8)

    silent_mp4 = out_path.replace(".mp4", "_silent.mp4")
    cmd_video  = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(FPS), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", silent_mp4,
    ]
    proc = subprocess.Popen(cmd_video, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    for f_idx in range(total_frames):
        t = f_idx / FPS

        # 亮度遮罩（in-place，vectorized）
        fill_darkness(darkness, tmp_dark, masks_f32, mask_starts, t, total_dur)

        # 套用遮罩（in-place）
        np.multiply(slide_f32, darkness[:, :, np.newaxis], out=frame_f32)
        np.clip(frame_f32, 0.0, 255.0, out=frame_f32)

        # float32 → uint8（in-place）
        np.copyto(frame_u8, frame_f32, casting="unsafe")

        # 寫入 pipe（使用 memoryview，避免 tobytes() 複製）
        proc.stdin.write(frame_u8.data)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg video encode failed (slide {slide_idx})")

    # 合併音訊
    cmd_audio = [
        FFMPEG, "-y",
        "-i", silent_mp4, "-i", mp3_path,
        "-filter_complex",
        f"[1:a]adelay={narr_delay_ms}|{narr_delay_ms}[a_del];"
        "[a_del]apad[a_out]",
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
print("快速 Spotlight 動畫影片生成器 v3（優化版）")
print("=" * 60)

segment_files = []

for i, (png, mp3) in enumerate(zip(slide_pngs, audio_files)):
    seg = os.path.join(TMP_DIR, f"sam_seg_{i:02d}.mp4")

    if os.path.exists(seg) and os.path.getsize(seg) > 50000:
        print(f"[{i+1:02d}/{N_SLIDES}] 已存在，跳過", flush=True)
        segment_files.append(seg)
        continue

    print(f"\n[{i+1:02d}/{N_SLIDES}] slide_{i:02d}.png", end="  ", flush=True)

    img_bgr  = cv2.imread(png)
    img_bgr  = cv2.resize(img_bgr, (W, H))
    slide_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    masks = detect_regions(img_bgr)
    print(f"偵測到 {len(masks)} 個區域", flush=True)

    narr_dur     = get_audio_duration(mp3)
    anim_end_t   = BG_DUR + len(masks) * GAP + REVEAL_DUR
    total_est    = max(NARR_DELAY + narr_dur, anim_end_t + 0.4) + HOLD_END + FADE_OUT
    total_frames = int(total_est * FPS)
    print(f"  渲染 {total_frames} frames（{total_est:.1f}s）…", end=" ", flush=True)

    dur      = encode_segment(slide_rgb, masks, mp3, seg, i)
    size_kb  = os.path.getsize(seg) // 1024
    print(f"完成  ({size_kb} KB)", flush=True)
    segment_files.append(seg)

# ── 串接所有片段 ──────────────────────────────────────────────────────────────
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
print(f"  大小：{size_mb:.1f} MB", flush=True)
