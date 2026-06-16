"""
flyin_sam_video.py — 全投影片 SAM 飛入動畫影片生成器

demo_flyin_sam.py（單張 demo）的多投影片版：
  • 處理全部 N_SLIDES 張投影片
  • SAM2 box-prompt 像素遮罩飛入（fly_top / fly_left / fly_right / zoom）
  • 近端位移 SLIDE_DIST px + opacity 淡入 → 文字永遠在畫面內，不截斷
  • 各片段串接成完整影片，旁白自動同步
"""
import os, sys, subprocess, time
import numpy as np
import cv2
import imageio_ffmpeg
from moviepy import AudioFileClip

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE     = r"D:\wi\260612"
ASSETS   = os.path.join(BASE, "pres_pdf_assets")
SAM_MODEL = os.path.join(BASE, "sam2_b.pt")
TMP_DIR  = os.path.join(r"D:\wi\50_Startups_repo", "flyin_sam_tmp")
OUT_MP4  = os.path.join(r"D:\wi\50_Startups_repo", "flyin_sam_full.mp4")
FFMPEG   = imageio_ffmpeg.get_ffmpeg_exe()

FPS      = 24
W, H     = 1920, 1080
N_SLIDES = 12

# 動畫時序
ANIM_START       = 0.20
ELEM_GAP         = 0.55
ELEM_DUR         = 0.50
FADE_IN          = 0.30
ZOOM_RANGE       = 0.04
SLIDE_DIST       = 90       # 近端飛入位移（px），不從螢幕外飛
TEXT_SETTLE_FRAC = 0.18     # 前 18% 時間 opacity 達 100%
MAX_ELEM         = 7
GAP_PX           = 10
MAX_BAND_H       = int(H * 0.28)
WIDE_BAND_FRAC   = 0.70

NARR_DELAY  = 0.20   # 旁白在動畫後延遲秒數
HOLD_END    = 0.80   # 旁白結束後靜止時間
FADE_OUT    = 0.45   # 淡出時間

os.makedirs(TMP_DIR, exist_ok=True)


# ── SAM 模型（全域單例，避免重複載入）────────────────────────────────────────

_sam_model = None

def get_sam():
    global _sam_model
    if _sam_model is None:
        print("  載入 SAM2 模型…", end=" ", flush=True)
        from ultralytics import SAM
        _sam_model = SAM(SAM_MODEL)
        print("完成")
    return _sam_model


# ── ease-out cubic ────────────────────────────────────────────────────────────

def ease_out(p):
    return 1 - (1 - float(np.clip(p, 0, 1))) ** 3


# ── 步驟 1：OpenCV 水平投影找粗略 box ────────────────────────────────────────

def detect_boxes(img_bgr):
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners   = [img_bgr[15,15], img_bgr[15,-15], img_bgr[-15,15], img_bgr[-15,-15]]
    bg_bgr    = np.median(corners, axis=0).astype(np.uint8)
    bg_bright = int(bg_bgr.mean()) > 128

    if bg_bright:
        bg_val  = float(np.median(gray[:25, :]))
        content = (np.abs(gray.astype(np.float32) - bg_val) > 10).astype(np.uint8)
    else:
        content = (cv2.Canny(gray, 15, 60) > 0).astype(np.uint8)

    kernel1d   = np.ones(4, dtype=np.float32) / 4
    h_proj     = np.convolve(content.sum(axis=1).astype(np.float32), kernel1d, mode='same')
    is_content = h_proj > (W * 0.012)

    bands, in_band, y0 = [], False, 0
    for y in range(H):
        if is_content[y] and not in_band:
            y0, in_band = y, True
        elif not is_content[y] and in_band:
            if y - y0 > 10:
                bands.append([y0, y])
            in_band = False
    if in_band:
        bands.append([y0, H])

    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < GAP_PX:
            merged[-1][1] = b[1]
        else:
            merged.append(b[:])

    # 強制分割過高的帶
    split = []
    for y1, y2 in merged:
        if y2 - y1 > MAX_BAND_H:
            s = y1 + (y2 - y1) // 4
            e = y1 + 3 * (y2 - y1) // 4
            sub = h_proj[s:e]
            if len(sub) > 0:
                mi = int(sub.argmin()) + s
                sv, pv = h_proj[mi], h_proj[y1:y2].max()
                if sv < pv * 0.60 and mi - y1 > 20 and y2 - mi > 20:
                    split += [[y1, mi], [mi, y2]]
                    continue
        split.append([y1, y2])
    merged = split[:MAX_ELEM]

    PAD, raw_boxes = 15, []
    for y1, y2 in merged:
        strip = content[y1:y2, :]
        xi    = np.where(strip.sum(axis=0) > 0)[0]
        if len(xi) == 0:
            continue
        x1, x2 = int(xi[0]), int(xi[-1] + 1)
        raw_boxes.append((y1, x1, y2, x2))

    # 對寬帶做垂直分割（把標題和圖片分開）
    boxes = []
    for y1, x1, y2, x2 in raw_boxes:
        band_w = x2 - x1
        if band_w > W * WIDE_BAND_FRAC:
            v_proj = content[y1:y2, :].sum(axis=0).astype(np.float32)
            kv     = np.ones(12, np.float32) / 12
            v_proj = np.convolve(v_proj, kv, mode='same')
            s = int(W * 0.28); e = int(W * 0.72)
            sub = v_proj[s:e]
            if len(sub) > 0:
                mi = int(sub.argmin()) + s
                sv, pv = v_proj[mi], v_proj.max()
                if sv < pv * 0.25 and mi - x1 > 40 and x2 - mi > 40:
                    xi_l = np.where(content[y1:y2, :mi].sum(axis=0) > 0)[0]
                    if len(xi_l):
                        boxes.append((max(0,y1-PAD), max(0,int(xi_l[0])-PAD),
                                      min(H,y2+PAD), min(W,mi+PAD)))
                    xi_r = np.where(content[y1:y2, mi:].sum(axis=0) > 0)[0]
                    if len(xi_r):
                        boxes.append((max(0,y1-PAD), max(0,mi-PAD),
                                      min(H,y2+PAD), min(W,int(xi_r[-1])+mi+PAD)))
                    continue
        boxes.append((max(0,y1-PAD), max(0,x1-PAD), min(H,y2+PAD), min(W,x2+PAD)))

    return bg_bgr, boxes[:MAX_ELEM]


# ── 步驟 2：SAM box-prompt 精確遮罩 ──────────────────────────────────────────

def sam_box_masks(png_path, boxes):
    model = get_sam()
    masks = []
    for y1, x1, y2, x2 in boxes:
        results = model(png_path, bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            raw  = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(raw.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask = np.zeros((H, W), dtype=bool)
            mask[y1:y2, x1:x2] = True
        masks.append(mask)
    return masks


# ── 步驟 3：指派飛入效果 ──────────────────────────────────────────────────────

def assign_effects(boxes, masks):
    """預計算 mask bbox crop，幀迴圈用 bbox slicing 取代全圖 fancy indexing"""
    elements, lr = [], 0
    for i, ((y1, x1, y2, x2), mask) in enumerate(zip(boxes, masks)):
        ew, eh   = x2 - x1, y2 - y1
        aspect   = ew / max(1, eh)
        is_title = (i == 0 and y1 < H * 0.30)
        is_image = (0.5 < aspect < 2.5 and mask.sum() > W * H * 0.04 and not is_title)

        if is_title:
            effect = 'fly_top'
        elif is_image:
            effect = 'zoom'
        else:
            effect = 'fly_left' if lr % 2 == 0 else 'fly_right'
            lr += 1

        ys, xs = np.where(mask)
        if len(ys):
            bby1, bby2 = int(ys.min()), int(ys.max()) + 1
            bbx1, bbx2 = int(xs.min()), int(xs.max()) + 1
        else:
            bby1, bby2, bbx1, bbx2 = 0, H, 0, W
        mask_crop = mask[bby1:bby2, bbx1:bbx2]
        elements.append((mask_crop, (bby1, bby2, bbx1, bbx2), (y1, x1, y2, x2), effect))
    return elements


# ── 步驟 4：逐幀動畫 ──────────────────────────────────────────────────────────

def _blit_alpha(canvas_f, src_f, dst_y, dst_x, alpha):
    sh, sw = src_f.shape[:2]
    cy1 = max(0, dst_y);    cx1 = max(0, dst_x)
    cy2 = min(H, dst_y+sh); cx2 = min(W, dst_x+sw)
    if cy2 <= cy1 or cx2 <= cx1:
        return
    ey1 = cy1 - dst_y; ex1 = cx1 - dst_x
    ey2 = ey1+(cy2-cy1); ex2 = ex1+(cx2-cx1)
    canvas_f[cy1:cy2, cx1:cx2] = (canvas_f[cy1:cy2, cx1:cx2] * (1-alpha)
                                   + src_f[ey1:ey2, ex1:ex2] * alpha)


def _blit_mask_alpha(canvas_f, img_f, mask_crop, bbox, off_y, off_x, alpha):
    """bbox slicing：在小 crop 上做 boolean indexing，比全圖 fancy indexing 快 3-5x"""
    bby1, bby2, bbx1, bbx2 = bbox
    dy1, dy2 = bby1 + off_y, bby2 + off_y
    dx1, dx2 = bbx1 + off_x, bbx2 + off_x
    cy1, cx1 = max(0, dy1), max(0, dx1)
    cy2, cx2 = min(H, dy2), min(W, dx2)
    if cy2 <= cy1 or cx2 <= cx1:
        return
    sy1, sx1 = cy1 - dy1, cx1 - dx1
    sy2, sx2 = sy1 + (cy2 - cy1), sx1 + (cx2 - cx1)
    m   = mask_crop[sy1:sy2, sx1:sx2]
    src = img_f[bby1+sy1:bby1+sy2, bbx1+sx1:bbx1+sx2]
    dst = canvas_f[cy1:cy2, cx1:cx2]
    dst[m] = dst[m] * (1-alpha) + src[m] * alpha


def make_frame(img_f, clean_bg_f, elements, t, zoom_f, total_dur):
    fade_out_st = total_dur - FADE_OUT
    if t < FADE_IN:
        global_fade = t / FADE_IN
    elif t > fade_out_st:
        global_fade = max(0.0, (total_dur - t) / FADE_OUT)
    else:
        global_fade = 1.0

    z = zoom_f
    if abs(z - 1.0) > 1e-4:
        nw, nh = int(W/z), int(H/z)
        x0, y0 = (W-nw)//2, (H-nh)//2
        base = cv2.resize(clean_bg_f[y0:y0+nh, x0:x0+nw], (W, H),
                          interpolation=cv2.INTER_LINEAR)
    else:
        base = clean_bg_f.copy()

    result = base * global_fade

    for i, (mask_crop, bbox, (y1, x1, y2, x2), effect) in enumerate(elements):
        t_start = ANIM_START + i * ELEM_GAP
        p = ease_out((t - t_start) / ELEM_DUR)
        if p <= 0:
            continue

        eh = y2 - y1 + 1
        ew = x2 - x1 + 1

        if effect in ('fly_top', 'fly_left', 'fly_right'):
            fast_alpha = min(p / max(TEXT_SETTLE_FRAC, 1e-6), 1.0) * global_fade
            if effect == 'fly_top':
                _blit_mask_alpha(result, img_f, mask_crop, bbox, int(-SLIDE_DIST*(1-p)), 0, fast_alpha)
            elif effect == 'fly_left':
                _blit_mask_alpha(result, img_f, mask_crop, bbox, 0, int(-SLIDE_DIST*(1-p)), fast_alpha)
            else:
                _blit_mask_alpha(result, img_f, mask_crop, bbox, 0, int(SLIDE_DIST*(1-p)), fast_alpha)

        elif effect == 'zoom':
            alpha  = p * global_fade
            scale  = 0.60 + 0.40 * p
            sw = max(1, int(ew * scale)); sh = max(1, int(eh * scale))
            elem_f = img_f[y1:y2, x1:x2]
            scaled = cv2.resize(elem_f, (sw, sh), interpolation=cv2.INTER_LINEAR)
            cy = y1 + (eh-sh)//2; cx = x1 + (ew-sw)//2
            _blit_alpha(result, scaled, cy, cx, alpha)

    return np.clip(result, 0, 255).astype(np.uint8)


# ── 片段編碼（靜音影片 → 合併旁白）─────────────────────────────────────────

def encode_segment(img_rgb, clean_bg, elements, mp3_path, out_path):
    with AudioFileClip(mp3_path) as a:
        narr_dur = a.duration

    n_elem    = len(elements)
    anim_end  = ANIM_START + (n_elem - 1) * ELEM_GAP + ELEM_DUR
    total_dur = max(NARR_DELAY + narr_dur, anim_end + 0.3) + HOLD_END + FADE_OUT
    n_frames  = int(total_dur * FPS)
    zooms     = np.linspace(1.0, 1.0 + ZOOM_RANGE, n_frames, dtype=np.float32)

    # 預轉 float32，幀迴圈內不再重複轉換
    img_f      = img_rgb.astype(np.float32)
    clean_bg_f = clean_bg.astype(np.float32)

    # 預合成「完全就位」圖（所有元素 p=1.0, global_fade=1.0）
    # 動畫結束後的靜止幀只做 Ken Burns，跳過所有元素 blit（省 80%+ 幀數的運算）
    settled_f = clean_bg_f.copy()
    for mask_crop, (bby1, bby2, bbx1, bbx2), (y1, x1, y2, x2), effect in elements:
        if effect != 'zoom':
            src = img_f[bby1:bby2, bbx1:bbx2]
            settled_f[bby1:bby2, bbx1:bbx2][mask_crop] = src[mask_crop]
        else:
            settled_f[y1:y2, x1:x2] = img_f[y1:y2, x1:x2]

    anim_done_fi = int((ANIM_START + (n_elem-1)*ELEM_GAP + ELEM_DUR) * FPS) + 2
    fade_out_fi  = max(0, int((total_dur - FADE_OUT) * FPS))

    # hold 幀預計算成 bytes：靜止期用中間 zoom 值算一次，之後 memcpy 即可
    mid_z = float(zooms[(anim_done_fi + fade_out_fi) // 2]) if anim_done_fi < fade_out_fi else 1.0
    if abs(mid_z - 1.0) > 1e-4:
        nw, nh = int(W/mid_z), int(H/mid_z)
        x0h, y0h = (W-nw)//2, (H-nh)//2
        hold_frame = cv2.resize(settled_f[y0h:y0h+nh, x0h:x0h+nw], (W, H),
                                interpolation=cv2.INTER_LINEAR)
    else:
        hold_frame = settled_f
    hold_bytes = np.clip(hold_frame, 0, 255).astype(np.uint8).tobytes()

    silent = out_path.replace(".mp4", "_silent.mp4")
    cmd_v  = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc", "-preset", "p1",
        "-pix_fmt", "yuv420p",
        silent,
    ]
    proc = subprocess.Popen(cmd_v, stdin=subprocess.PIPE, bufsize=0,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for fi in range(n_frames):
            if anim_done_fi <= fi < fade_out_fi:
                proc.stdin.write(hold_bytes)
            else:
                frame = make_frame(img_f, clean_bg_f, elements,
                                   fi / FPS, float(zooms[fi]), total_dur)
                proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    delay_ms = int(NARR_DELAY * 1000)
    cmd_a    = [
        FFMPEG, "-y",
        "-i", silent, "-i", mp3_path,
        "-filter_complex",
        f"[1:a]adelay={delay_ms}|{delay_ms}[adel];[adel]apad[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", out_path,
    ]
    r = subprocess.run(cmd_a, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        import shutil
        shutil.copy(silent, out_path)
    os.remove(silent)
    return total_dur


# ── 主流程 ────────────────────────────────────────────────────────────────────

segment_files = []

for slide_idx in range(N_SLIDES):
    seg_path = os.path.join(TMP_DIR, f"seg_{slide_idx:02d}.mp4")
    if os.path.exists(seg_path):
        print(f"[{slide_idx+1:02d}/{N_SLIDES}] 已存在，跳過")
        segment_files.append(seg_path)
        continue

    png = os.path.join(ASSETS, f"slide_{slide_idx:02d}.png")
    mp3 = os.path.join(ASSETS, f"narr_{slide_idx:02d}.mp3")
    print(f"\n[{slide_idx+1:02d}/{N_SLIDES}] slide_{slide_idx:02d}.png")

    img_bgr = cv2.imread(png)
    img_bgr = cv2.resize(img_bgr, (W, H))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("  偵測元素…", end=" ", flush=True)
    bg_bgr, boxes = detect_boxes(img_bgr)
    print(f"{len(boxes)} 個 box")

    print("  SAM 分割…", end=" ", flush=True)
    t0 = time.time()
    masks = sam_box_masks(png, boxes)
    print(f"{time.time()-t0:.0f}s")

    elements = assign_effects(boxes, masks)
    for i, (_mc, _bb, (y1, x1, y2, x2), eff) in enumerate(elements):
        print(f"    元素{i+1} [{eff}] y({y1}→{y2}) x({x1}→{x2})")

    bg_rgb_arr = bg_bgr[[2, 1, 0]]
    clean_bg   = np.full((H, W, 3), bg_rgb_arr, dtype=np.uint8)

    print("  渲染…", end=" ", flush=True)
    t0 = time.time()
    dur     = encode_segment(img_rgb, clean_bg, elements, mp3, seg_path)
    size_kb = os.path.getsize(seg_path) // 1024
    print(f"{time.time()-t0:.0f}s  ({size_kb} KB)  總長 {dur:.1f}s")

    segment_files.append(seg_path)

# ── 串接所有片段 ──────────────────────────────────────────────────────────────
print(f"\n串接 {len(segment_files)} 個片段…")
concat_txt = os.path.join(TMP_DIR, "concat.txt")
with open(concat_txt, "w", encoding="utf-8") as f:
    for seg in segment_files:
        f.write(f"file '{seg.replace(os.sep, '/')}'\n")

subprocess.run([
    FFMPEG, "-y",
    "-f", "concat", "-safe", "0", "-i", concat_txt,
    "-c", "copy", "-movflags", "+faststart",
    OUT_MP4,
], check=True, capture_output=True)

size_mb = os.path.getsize(OUT_MP4) / 1024 / 1024
print(f"\n完成 → {OUT_MP4}  ({size_mb:.1f} MB)")
print("效果：SAM 像素遮罩 + 近端 90px 滑入 + opacity 淡入，全 12 張投影片")
