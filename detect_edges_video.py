#!/usr/bin/env python3
"""
Video Footer Detector & Cropper v3
===================================
Detects letterbox black bars + news ticker from snapshots, then applies
FIXED crop region to ALL frames (no per-frame wobble). Uses LANCZOS4
for sharp upscaling and H.264 codec for quality.

Usage:
    python detect_edges_video.py video.mp4
    python detect_edges_video.py video.mp4 -m 132 -v
    python detect_edges_video.py video.mp4 -m 132 -o output.mp4
    python detect_edges_video.py video.mp4 -m 132 -n 30 -v
"""

import argparse
import os
import sys
import random
import time
import subprocess
import tempfile
import numpy as np
import cv2


# ────────────────────────────────────────────────────────────────
#  Adaptive threshold (MAD-based)
# ────────────────────────────────────────────────────────────────

def adaptive_threshold(signal):
    """Returns (threshold, median, sigma)."""
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    sigma = max(mad * 1.4826, med * 0.25, 1.0)
    return med + 3.0 * sigma, med, sigma


# ────────────────────────────────────────────────────────────────
#  Letterbox detection — find top/bottom black bars
# ────────────────────────────────────────────────────────────────

def detect_letterbox(frame, black_thresh=15, min_bar=8):
    """
    Fast scan for consecutive black rows from top and bottom edges.
    Returns (content_top, content_bottom) where content lives.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    content_top = 0
    black_count = 0
    for y in range(min(h // 3, 300)):
        if gray[y].mean() <= black_thresh:
            black_count += 1
        else:
            break
    if black_count >= min_bar:
        content_top = black_count

    content_bottom = h
    black_count = 0
    for y in range(h - 1, max(h - h // 3, h - 300), -1):
        if gray[y].mean() <= black_thresh:
            black_count += 1
        else:
            break
    if black_count >= min_bar:
        content_bottom = h - black_count

    return content_top, content_bottom


# ────────────────────────────────────────────────────────────────
#  Footer detection — within content area, ignoring letterbox
# ────────────────────────────────────────────────────────────────

def detect_footer(img, h, w, content_top=0, content_bottom=None,
                   verbose=False):
    """
    Row-to-row colour diff, search only within content area.
    """
    if content_bottom is None:
        content_bottom = h

    diff = np.zeros(h - 1)
    for y in range(1, h):
        diff[y - 1] = np.mean(np.abs(
            img[y].astype(np.float64) - img[y - 1].astype(np.float64)
        ))

    content_h = content_bottom - content_top
    start = content_top + content_h // 2
    end = content_bottom - 1
    if end <= start:
        return 0
    search = diff[start:end]
    if len(search) < 5:
        return 0

    thresh, med, sigma = adaptive_threshold(search)

    if verbose:
        print(f"    Content area   : y=[{content_top}, {content_bottom})")
        print(f"    Signal range   : y={start}..y={end-1}  ({len(search)} rows)")
        print(f"    Median diff    : {med:.2f}")
        print(f"    Robust sigma   : {sigma:.2f}")
        print(f"    Threshold      : {thresh:.2f}")

    for i in range(len(search)):
        if search[i] > thresh:
            footer_top = start + i + 1
            footer_h = content_bottom - footer_top
            if verbose:
                print(f"    First peak     : y={footer_top}  diff={search[i]:.2f}")
                print(f"    Footer height  : {footer_h}px  "
                      f"({footer_h/content_h*100:.1f}% of content)")
            if content_h * 0.02 <= footer_h <= content_h * 0.40:
                footer_region = img[footer_top:content_bottom]
                if footer_region.mean() > 20:
                    return footer_top
                elif verbose:
                    print(f"    REJECTED — footer region is black "
                          f"(mean={footer_region.mean():.1f})")
            return 0

    if verbose:
        print(f"    No significant peak  (max={np.max(search):.2f})")
    return 0


# ────────────────────────────────────────────────────────────────
#  Sample N random frames from video
# ────────────────────────────────────────────────────────────────

def sample_random_frames(cap, n):
    """Seek to N random frame indices and return (frame, frame_number) list."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        return []

    safe_start = max(1, int(total * 0.02))
    safe_end = min(total - 2, int(total * 0.98))
    if safe_end <= safe_start:
        indices = [max(0, total // 2)]
    else:
        indices = sorted(random.sample(range(safe_start, safe_end + 1),
                                       min(n, safe_end - safe_start + 1)))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, idx))
    return frames


# ────────────────────────────────────────────────────────────────
#  Find consensus letterbox + footer from snapshots
# ────────────────────────────────────────────────────────────────

def find_consensus_footer(frames, frame_h, frame_w, verbose=False):
    """
    For each sampled frame:
      1. Detect letterbox
      2. Detect footer within content area
      3. Filter out frames where footer is actually a black bar
    Returns (footer_y, letterbox_top, letterbox_bottom)
    """
    footer_detections = []
    letterbox_tops = []
    letterbox_bottoms = []

    for i, (frame, frame_num) in enumerate(frames):
        fh, fw = frame.shape[:2]
        ct, cb = detect_letterbox(frame)

        has_letterbox = (ct > 0) or (cb < fh)
        lb_info = (f"letterbox=({ct},{cb})" if has_letterbox
                   else "no letterbox")

        ft = detect_footer(frame, fh, fw,
                          content_top=ct, content_bottom=cb,
                          verbose=verbose)
        footer_info = f"y={ft}" if ft > 0 else "NONE"

        print(f"  Snap {i+1:>2d}  (frame {frame_num:>6d})  : "
              f"{lb_info}  footer={footer_info}")

        letterbox_tops.append(ct)
        letterbox_bottoms.append(cb)

        if ft > 0:
            footer_region = frame[ft:cb]
            if footer_region.mean() > 20:
                footer_detections.append(ft)

    lb_top = int(np.median([x for x in letterbox_tops]))
    lb_bot = int(np.median([x for x in letterbox_bottoms]))
    print(f"\n  Letterbox consensus : top={lb_top}  bottom={lb_bot}")

    if not footer_detections:
        print(f"  Footer detections  : NONE")
        return 0, lb_top, lb_bot

    footer_mode = int(np.median(footer_detections))
    unique, counts = np.unique(footer_detections, return_counts=True)
    max_count = counts.max()
    if max_count >= len(footer_detections) * 0.5:
        footer_mode = int(unique[counts.argmax()])

    print(f"  Footer detections  : {footer_detections}")
    print(f"  Footer consensus   : y={footer_mode}  "
          f"({len(footer_detections)}/{len(frames)} frames agreed)")

    return footer_mode, lb_top, lb_bot


# ────────────────────────────────────────────────────────────────
#  Probe original video encoding params (via ffprobe)
# ────────────────────────────────────────────────────────────────

def probe_video_params(video_path, fps):
    """
    Use ffprobe to extract the original video's codec, keyframe interval,
    bitrate, profile, and pixel format. Returns a dict or None.
    """
    try:
        import json

        # Get stream info
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-print_format", "json",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,bit_rate,tags",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=10)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        tags = stream.get("tags", {})

        params = {
            "codec": stream.get("codec_name", "h264"),
            "profile": stream.get("profile", "high"),
            "pix_fmt": stream.get("pix_fmt", "yuv420p"),
            "bit_rate": stream.get("bit_rate", None),
        }

        # Get keyframe interval — only scan first 10 seconds
        if "gop_size" in tags:
            params["gop"] = int(tags["gop_size"])
        else:
            try:
                kf_cmd = [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "v:0",
                    "-print_format", "json",
                    "-show_entries", "frame=pts_time,key_frame",
                    "-read_intervals", "%+#10",
                    video_path
                ]
                kf_result = subprocess.run(kf_cmd, capture_output=True,
                                         text=True, timeout=15)
                if kf_result.returncode == 0:
                    kf_data = json.loads(kf_result.stdout)
                    kf_frames = [f for f in kf_data.get("frames", [])
                                 if f.get("key_frame") == 1]
                    if len(kf_frames) >= 2:
                        kf_times = [float(f["pts_time"]) for f in kf_frames]
                        intervals = [kf_times[i+1] - kf_times[i]
                                     for i in range(min(5, len(kf_times)-1))]
                        avg_interval = sum(intervals) / len(intervals)
                        params["gop"] = max(1, round(avg_interval * fps))
                    else:
                        params["gop"] = int(fps)
                else:
                    params["gop"] = int(fps)
            except Exception:
                params["gop"] = int(fps)

        return params

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


# ────────────────────────────────────────────────────────────────
#  Create video writer — try H.264, fallback to MPEG-4
# ────────────────────────────────────────────────────────────────

def create_writer(out_path, fps, w, h):
    """
    Try H.264 first (best quality on Windows), then MPEG-4.
    Returns (writer, codec_name) or (None, None).
    """
    # Suppress OpenCV codec warning spam on stderr
    import io
    devnull = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = devnull

    result = None, None
    for fourcc_str, name in [('avc1', 'H.264 (avc1)'),
                              ('H264', 'H.264 (H264)'),
                              ('X264', 'H.264 (x264)'),
                              ('mp4v', 'MPEG-4 (mp4v)')]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            result = writer, name
            break

    sys.stderr = old_stderr
    return result


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video footer detector & cropper v3 — fixed crop region, "
                    "LANCZOS4 upscale, best available codec.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed diagnostics")
    parser.add_argument("-o", "--output", default=None,
                        help="Output video path (default: <input>_processed.mp4)")
    parser.add_argument("-m", "--margin", type=int, default=0,
                        help="Fixed crop margin in px from left/right edges "
                             "(e.g. 132)")
    parser.add_argument("-n", "--snapshots", type=int, default=20,
                        help="Number of random frames to sample (default: 20)")
    parser.add_argument("--hq", action="store_true",
                        help="Use ffmpeg H.264 CRF=0 for LOSSLESS quality "
                             "(requires ffmpeg in PATH)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Strip audio from output (default: keep original audio)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}")
        sys.exit(1)

    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if FPS <= 0:
        FPS = 25.0

    orig_ratio = FRAME_W / FRAME_H

    # ── Probe original video encoding params ──
    orig_params = None
    if args.hq:
        print(f"[INFO] Probing original video encoding params ...")
        orig_params = probe_video_params(args.input, FPS)
        if orig_params:
            print(f"[PROBE] Original codec    : {orig_params['codec']} "
                  f"({orig_params.get('profile', '?')})")
            print(f"[PROBE] Original pix_fmt  : {orig_params['pix_fmt']}")
            print(f"[PROBE] Original bitrate  : {orig_params['bit_rate'] or 'N/A'}")
            print(f"[PROBE] Original keyframe : every {orig_params['gop']} frames "
                  f"({orig_params['gop']/FPS:.1f}s)")
        else:
            print(f"[PROBE] Could not probe — will use defaults")
        print()

    print(f"[INFO] {args.input}")
    print(f"[INFO] Resolution      : {FRAME_W}x{FRAME_H}")
    print(f"[INFO] Aspect ratio    : {FRAME_W}:{FRAME_H} = {orig_ratio:.4f}")
    print(f"[INFO] FPS             : {FPS:.2f}")
    print(f"[INFO] Total frames    : {TOTAL}")
    print(f"[INFO] Duration        : {TOTAL/FPS:.1f}s")
    print(f"[INFO] Crop margin     : {args.margin}px from each side")
    print(f"[INFO] Snapshots       : {args.snapshots}")
    print(f"[INFO] HQ mode         : {'ON (ffmpeg LOSSLESS CRF=0)' if args.hq else 'OFF (OpenCV codec)'}")
    print()

    # ── Phase 1: Sample frames, detect letterbox + footer ──
    print("=" * 55)
    print("  PHASE 1: Letterbox + Footer Detection (snapshots)")
    print("=" * 55)
    print()

    random.seed(42)
    samples = sample_random_frames(cap, args.snapshots)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not samples:
        print("[ERROR] Could not read any frames from video")
        cap.release()
        sys.exit(1)

    print(f"  Sampled {len(samples)} frames from {TOTAL} total")
    print()
    print("[DETECTION]")

    footer_top, lb_top, lb_bot = find_consensus_footer(
        samples, FRAME_H, FRAME_W, verbose=args.verbose)

    if footer_top > 0:
        footer_h = lb_bot - footer_top
        print(f"\n  => NEWS TICKER FOUND  y={footer_top}  ({footer_h}px, "
              f"{footer_h/(lb_bot-lb_top)*100:.1f}% of content)")
    else:
        print(f"\n  => NO TICKER DETECTED — cropping full content area")

    print(f"  => LETTERBOX        top={lb_top}px  bottom={lb_bot}px  "
          f"(content: {lb_bot-lb_top}px)")
    if args.margin > 0:
        print(f"  => LEFT/RIGHT crop  x=[{args.margin}, {FRAME_W - args.margin})")

    # ── Phase 2: Pre-compute FIXED crop + zoom params ──
    print()
    print("=" * 55)
    print("  PHASE 2: Processing all frames")
    print("=" * 55)
    print()

    # Fixed crop region (same for ALL frames — no per-frame wobble)
    crop_x1 = args.margin
    crop_x2 = FRAME_W - args.margin
    crop_y1 = lb_top                          # strip top letterbox
    crop_y2 = footer_top if footer_top > 0 else lb_bot  # strip footer
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    # Cover-zoom params — scale so smallest dim matches frame,
    # then center-crop the overflow from the larger dim.
    scale = max(FRAME_W / crop_w, FRAME_H / crop_h)
    zw = max(int(crop_w * scale), FRAME_W)
    zh = max(int(crop_h * scale), FRAME_H)
    # Center-crop offsets (clamp strictly to valid range)
    zx = max(0, min((zw - FRAME_W) // 2, zw - FRAME_W))
    zy = max(0, min((zh - FRAME_H) // 2, zh - FRAME_H))

    print(f"  Crop region   : ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
    print(f"  Crop size     : {crop_w}x{crop_h}")
    print(f"  Cover-zoom    : {crop_w}x{crop_h} -> {zw}x{zh}  "
          f"({scale:.3f}x)")
    print(f"  Center-crop   : offset ({zx}, {zy})  -> {FRAME_W}x{FRAME_H}")
    print(f"  Interpolation : LANCZOS4 (high quality)")
    print(f"  Mode          : FIXED crop region for all frames (no wobble)")
    print()

    if args.hq:
        # ── HQ path: write frames to temp AVI, then ffmpeg to H.264 ──
        out_path = (args.output
                    or os.path.splitext(args.input)[0] + "_processed.mp4")

        # Check ffmpeg available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True,
                          check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("[WARNING] ffmpeg not found — falling back to OpenCV codec")
            args.hq = False

    if args.hq:
        tmp_dir = tempfile.mkdtemp()
        tmp_raw = os.path.join(tmp_dir, "raw_frames.avi")

        # Check if original video has audio
        has_audio = False
        try:
            probe_audio = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-print_format", "json",
                 args.input],
                capture_output=True, text=True, timeout=10)
            audio_data = json.loads(probe_audio.stdout)
            if audio_data.get("streams"):
                has_audio = True
        except Exception:
            pass

        print(f"  Codec         : ffmpeg H.264 CRF=0 (LOSSLESS, matching original)")
        print(f"  Temp file     : {tmp_raw}")
        print(f"  Audio         : {('KEEP (found in source)' if has_audio else 'NONE (no audio in source)') if not args.no_audio else 'STRIPPED (--no-audio)'}")
        print()

        fourcc_raw = cv2.VideoWriter_fourcc(*'HFYU')  # HuffYUV lossless
        writer = cv2.VideoWriter(tmp_raw, fourcc_raw, FPS, (FRAME_W, FRAME_H))

        if not writer.isOpened():
            print("[WARNING] HuffYUV not available, using FFV1")
            fourcc_raw = cv2.VideoWriter_fourcc(*'FFV1')
            writer = cv2.VideoWriter(tmp_raw, fourcc_raw, FPS,
                                     (FRAME_W, FRAME_H))
        if not writer.isOpened():
            print("[WARNING] Lossless codecs not available — "
                  "falling back to OpenCV mp4v")
            args.hq = False

    if not args.hq:
        out_path = (args.output
                    or os.path.splitext(args.input)[0] + "_processed.mp4")
        writer, codec_name = create_writer(out_path, FPS, FRAME_W, FRAME_H)
        if writer is None:
            print("[ERROR] No video codec available")
            cap.release()
            sys.exit(1)
        print(f"  Codec         : {codec_name}")
        print()

    # ── Tight processing loop ──
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Crop (fixed region — numpy slice, zero-copy)
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # 2) Cover-zoom (LANCZOS4 for sharp upscale)
        scaled = cv2.resize(cropped, (zw, zh),
                            interpolation=cv2.INTER_LANCZOS4)

        # 3) Center-crop to final frame (contiguous copy for writer)
        result = scaled[zy:zy + FRAME_H, zx:zx + FRAME_W].copy()
        writer.write(result)

        frame_idx += 1

        if frame_idx % 500 == 0 or frame_idx == TOTAL:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            remaining = ((TOTAL - frame_idx) / fps_proc
                        if fps_proc > 0 else 0)
            pct = frame_idx / TOTAL * 100
            print(f"\r  Progress: {frame_idx}/{TOTAL}  ({pct:.1f}%)  "
                  f"[{fps_proc:.1f} fps]  ETA: {remaining:.0f}s      ",
                  end="", flush=True)

    print()

    writer.release()
    cap.release()
    elapsed = time.time() - start_time

    # ── HQ: ffmpeg re-encode to H.264 CRF=18 ──
    if args.hq:
        print()
        print("  Encoding to H.264 CRF=0 (LOSSLESS) via ffmpeg ...")
        ffmpeg_start = time.time()
        try:
            # Build ffmpeg command matching original video's params
            gop = int(FPS)  # default: 1 keyframe per second
            profile = "high"
            pix_fmt = "yuv420p"
            if orig_params:
                gop = orig_params.get("gop", gop)
                # Map profile names to valid libx264 profiles
                raw_profile = orig_params.get("profile", "high").lower()
                profile_map = {
                    "baseline": "baseline", "simple": "baseline",
                    "main": "main", "high": "high",
                    "high10": "high10", "high422": "high422",
                    "high444": "high444",
                }
                profile = profile_map.get(raw_profile, "high")
                pix_fmt = orig_params.get("pix_fmt", pix_fmt)

            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_raw,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "0",
                "-profile:v", profile,
                "-g", str(gop),
                "-keyint_min", str(gop),
                "-pix_fmt", pix_fmt,
                "-movflags", "+faststart",
            ]

            # Add audio from original video (unless --no-audio)
            if has_audio and not args.no_audio:
                cmd.extend([
                    "-i", args.input,
                    "-map", "0:v:0",      # video from temp file
                    "-map", "1:a:0?",     # audio from original
                    "-c:a", "copy",       # copy audio without re-encoding
                    "-shortest",           # stop when shortest stream ends
                ])

            cmd.append(out_path)

            print(f"  Profile       : {profile}")
            print(f"  Keyframe      : every {gop} frames ({gop/FPS:.1f}s)")
            print(f"  Pixel format  : {pix_fmt}")
            if has_audio and not args.no_audio:
                print(f"  Audio         : copied from original (no re-encode)")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[WARNING] ffmpeg failed:\n{result.stderr[-500:]}")
                print("  Falling back to raw AVI output")
                import shutil
                shutil.move(tmp_raw, out_path)
            else:
                os.remove(tmp_raw)
                os.rmdir(tmp_dir)
                ffmpeg_time = time.time() - ffmpeg_start
                print(f"  ffmpeg encode  : {ffmpeg_time:.1f}s")
        except Exception as e:
            print(f"[WARNING] ffmpeg error: {e}")
            import shutil
            shutil.move(tmp_raw, out_path)

    file_size = os.path.getsize(out_path)
    file_mb = file_size / (1024 * 1024)

    print()
    print("=" * 55)
    print("  DONE")
    print("=" * 55)
    print(f"  Output      : {out_path}")
    print(f"  Size        : {file_mb:.1f} MB")
    print(f"  Resolution  : {FRAME_W}x{FRAME_H}")
    print(f"  Frames      : {frame_idx}")
    print(f"  Duration    : {frame_idx/FPS:.1f}s  @ {FPS:.2f} fps")
    print(f"  Ticker      : {'y=' + str(footer_top) if footer_top else 'NONE'}")
    print(f"  Letterbox   : top={lb_top}px  bottom={lb_bot}px")
    if args.margin > 0:
        print(f"  Margin      : {args.margin}px each side  "
              f"(x={args.margin} to x={FRAME_W - args.margin})")
    print(f"  Crop region : ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2}) "
          f"= {crop_w}x{crop_h}")
    print(f"  Zoom        : {scale:.3f}x  -> {zw}x{zh}  "
          f"center-crop ({zx},{zy})")
    print(f"  Codec       : {'ffmpeg H.264 CRF=0 (LOSSLESS)' if args.hq else codec_name}")
    print(f"  Interp      : LANCZOS4")
    print(f"  Time        : {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
