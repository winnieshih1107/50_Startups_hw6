"""
assemble_video_zh.py
====================
從已生成的 PNG + MP3 重新組合成 MP4
使用 moviepy AudioFileClip 取得時長，再以 ffmpeg 逐段編碼後串接。
"""
import os, subprocess, glob
import imageio_ffmpeg

BASE   = r"D:\wi"
ASSETS = os.path.join(BASE, "pres_assets_zh")
OUT    = os.path.join(BASE, "hw6_presentation_zh.mp4")
TMP    = os.path.join(ASSETS, "tmp_segments")
os.makedirs(TMP, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using ffmpeg: {FFMPEG}")

# ── 確認素材 ─────────────────────────────────────────────────────────────────
pngs = sorted(glob.glob(os.path.join(ASSETS, "slide_*.png")))
mp3s = sorted(glob.glob(os.path.join(ASSETS, "narr_*.mp3")))
assert len(pngs) == len(mp3s) == 13, \
    f"Expected 13, got {len(pngs)} PNGs / {len(mp3s)} MP3s"
print(f"Found {len(pngs)} slides and {len(mp3s)} narrations.\n")

# ── 取得 MP3 時長（使用 moviepy） ─────────────────────────────────────────────
from moviepy import AudioFileClip

durations = []
for mp3 in mp3s:
    with AudioFileClip(mp3) as audio:
        d = audio.duration + 1.0  # 1 秒緩衝
    durations.append(d)
    print(f"  {os.path.basename(mp3)}: {d:.1f}s")

total = sum(durations)
print(f"\nTotal duration: {total:.1f}s ({total/60:.1f} min)\n")

# ── 逐段編碼 ─────────────────────────────────────────────────────────────────
segment_files = []
for i, (png, mp3, dur) in enumerate(zip(pngs, mp3s, durations)):
    seg_out = os.path.join(TMP, f"seg_{i:02d}.mp4")
    cmd = [
        FFMPEG, "-y",
        "-loop", "1", "-framerate", "1", "-i", png,
        "-i", mp3,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1920:1080",
        "-r", "24",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(dur),
        seg_out
    ]
    print(f"[{i+1:02d}/13] Encoding segment {os.path.basename(png)}…")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-800:]}")
        raise RuntimeError(f"ffmpeg failed on segment {i}")
    size = os.path.getsize(seg_out) / 1024
    print(f"         -> {seg_out} ({size:.0f} KB)")
    segment_files.append(seg_out)

# ── 串接所有片段 ─────────────────────────────────────────────────────────────
concat_list = os.path.join(TMP, "concat.txt")
with open(concat_list, "w", encoding="utf-8") as f:
    for seg in segment_files:
        safe_path = seg.replace("\\", "/")
        f.write(f"file '{safe_path}'\n")

print(f"\nConcatenating {len(segment_files)} segments…")
cmd = [
    FFMPEG, "-y",
    "-f", "concat", "-safe", "0",
    "-i", concat_list,
    "-c", "copy",
    OUT
]
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if result.returncode != 0:
    print(f"Concat ERROR:\n{result.stderr[-1000:]}")
    raise RuntimeError("ffmpeg concat failed")

# ── Faststart 後處理 ─────────────────────────────────────────────────────────
tmp_faststart = OUT.replace(".mp4", "_tmp_fs.mp4")
os.rename(OUT, tmp_faststart)
cmd = [
    FFMPEG, "-y",
    "-i", tmp_faststart,
    "-c", "copy",
    "-movflags", "+faststart",
    OUT
]
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if result.returncode != 0:
    print(f"Faststart ERROR:\n{result.stderr[-500:]}")
    os.rename(tmp_faststart, OUT)  # fallback: keep without faststart
else:
    os.remove(tmp_faststart)

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"\n[完成] {OUT}")
print(f"       檔案大小: {size_mb:.1f} MB")
print(f"       總時長:   {total/60:.1f} 分鐘")
