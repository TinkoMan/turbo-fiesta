#!/usr/bin/env python3
"""
detect.py — Singer Face Detector & High-Quality Snapshot Extractor
===================================================================
100% Offline, No Downloads, Python 3.13 Compatible.
Uses built-in OpenCV Face + Smile cascades + Auto Body Cropping.

Usage:
    python detect.py <video_path> [output_dir] [num_samples] [num_snapshots]

Can also be imported:
    from detect import SingerSnapshotExtractor
    results = SingerSnapshotExtractor(video, outdir).run()
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class SingerSnapshotExtractor:
    def __init__(self, video_path: str, output_dir: str = "singer_snapshots"):
        self.video_path = os.path.abspath(video_path)
        self.output_dir = os.path.abspath(output_dir)
        self.temp_dir = tempfile.mkdtemp(prefix="singer_det_")
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load BUILT-IN Face Detector
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_path)

        # Load BUILT-IN Smile/Mouth Detector (Proxy for singing)
        smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        self.smile_cascade = cv2.CascadeClassifier(smile_path)

        print("  [detect] OpenCV loaded (No external downloads needed)\n")

    # --------------------------------------------------------- FFmpeg utilities
    def _probe(self) -> dict:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height",
            "-of", "flat", self.video_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = {}
        for line in r.stdout.strip().splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip().strip('"')
        return info

    def _extract_frame(self, timestamp: float, out_path: str, quality: int = 2):
        cmd = [
            "ffmpeg", "-y", "-ss", f"{timestamp:.4f}",
            "-i", self.video_path,
            "-vframes", "1", "-q:v", str(quality),
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    # -------------------------------------------------------- Face & Mouth Detection
    def detect_faces(self, img: np.ndarray) -> list[dict]:
        """Detect faces with robust error handling.
        Catches OpenCV getScaleData assertion and handles tiny/empty frames."""
        if img is None or img.size == 0:
            return []

        h, w = img.shape[:2]
        if h < 30 or w < 30:
            return []

        # Resize very small images (e.g. 360p thumbnails) for reliable detection
        work_img = img.copy()
        scale_factor = 1.0
        if min(h, w) < 200:
            scale_factor = 400.0 / min(h, w)
            work_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        try:
            gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Use conservative params to avoid getScaleData assertion
            faces_rects = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except cv2.error as e:
            # getScaleData assertion or other OpenCV internal errors
            print(f"  [detect] OpenCV face detection error (skipped): {str(e)[:80]}")
            return []

        faces = []
        for (x, y, fw, fh) in faces_rects:
            # Scale coordinates back if we resized
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                fw = int(fw / scale_factor)
                fh = int(fh / scale_factor)

            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            if fw < 20 or fh < 20:
                continue

            mouth_score = self._get_mouth_score(img, x, y, fw, fh)

            faces.append({
                "bbox": (x, y, x + fw, y + fh),
                "conf": 0.9,
                "cx": x + fw // 2,
                "cy": y + fh // 2,
                "area": fw * fh,
                "mouth": mouth_score,
            })
        return faces

    def _get_mouth_score(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Detect smile/open mouth in the lower face region."""
        mouth_y_start = y + int(h * 0.6)
        mouth_roi = img[mouth_y_start: y + h, x: x + w]

        if mouth_roi.size == 0:
            return 0.5

        try:
            gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            gray_mouth = cv2.equalizeHist(gray_mouth)

            smiles = self.smile_cascade.detectMultiScale(
                gray_mouth,
                scaleFactor=1.7,
                minNeighbors=8,
                minSize=(max(10, int(w * 0.3)), max(5, int(h * 0.08)))
            )

            if len(smiles) > 0:
                return 0.85
        except cv2.error:
            pass
        return 0.25

    # ---------------------------------------------- Face similarity
    @staticmethod
    def _face_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        sz = (64, 64)
        a = cv2.resize(a, sz)
        b = cv2.resize(b, sz)
        ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 40], [0, 180, 0, 256])
        hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 40], [0, 180, 0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)

    # ------------------------------------------------ Clustering faces
    def _cluster_faces(self, all_detections: list[dict]) -> list[dict]:
        SIM_THRESHOLD = 0.38
        clusters = []
        for det in all_detections:
            best_c, best_s = None, SIM_THRESHOLD
            for c in clusters:
                s = self._face_similarity(det["crop"], c["rep_img"])
                if s > best_s:
                    best_s, best_c = s, c
            if best_c is not None:
                best_c["members"].append(det)
            else:
                clusters.append({"rep_img": det["crop"], "members": [det]})
        return clusters

    # ------------------------------------------------ Singer scoring
    def _score_cluster(self, cluster: dict, video_area: int) -> float:
        mems = cluster["members"]
        n = len(mems)
        freq = min(n / 8.0, 1.0)
        mouth = np.mean([m["mouth"] for m in mems])
        center = np.mean([m["center_s"] for m in mems])
        size = np.mean([m["size_s"] for m in mems])
        score = 0.30 * freq + 0.35 * mouth + 0.20 * center + 0.15 * size
        cluster["_score"] = score
        cluster["_n"] = n
        cluster["_mouth"] = mouth
        return score

    # ------------------------------------------------ Auto Crop Logic
    def _auto_crop_half_body(self, img: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Expands the face bounding box heavily outwards and downwards.
        Keeps maximum original quality. Ensures hands, gestures, and mic are NOT cut.
        """
        x1, y1, x2, y2 = bbox
        face_h = y2 - y1
        face_w = x2 - x1

        img_h, img_w = img.shape[:2]

        top_extend = int(face_h * 0.5)
        bottom_extend = int(face_h * 3.5)
        side_extend = int(face_w * 2.5)

        new_y1 = max(0, y1 - top_extend)
        new_y2 = min(img_h, y2 + bottom_extend)
        new_x1 = max(0, x1 - side_extend)
        new_x2 = min(img_w, x2 + side_extend)

        cropped_img = img[new_y1:new_y2, new_x1:new_x2]
        return cropped_img

    # ====================================================== PUBLIC API
    def run(self, num_samples: int = 45, num_snapshots: int = 3):
        """
        Run full detection pipeline.
        Uses threading for parallel frame extraction and face detection.
        Optimized: auto-reduces samples for short/low-res videos, early exit on success.

        Returns
        -------
        list of dict
            Each dict: {"annotated": path, "clean": path, "crop": path, "ts": float}
            "crop" is the half-body cropped image of the singer.
        """
        print("=" * 62)
        print("   [detect] SINGER SNAPSHOT EXTRACTOR & CROPPER (threaded)")
        print("=" * 62)
        print(f"  Video : {self.video_path}")
        print(f"  Output: {self.output_dir}\n")

        info = self._probe()
        duration = float(info.get("format.duration", 60))
        width = int(info.get("stream.0.width", 1920))
        height = int(info.get("stream.0.height", 1080))
        video_area = width * height
        print(f"  Duration : {duration:.1f} s   Resolution : {width}x{height}\n")

        # Auto-reduce samples for short or low-res videos (speed optimization)
        if duration < 60:
            num_samples = min(num_samples, 15)
        elif duration < 180:
            num_samples = min(num_samples, 25)
        if min(width, height) <= 480:
            num_samples = min(num_samples, 20)
            print(f"  [speed] Low-res video, reduced to {num_samples} samples")
        print(f"  [speed] Using {num_samples} samples (auto-adjusted)\n")

        # ── 1. Sample frames (THREADED — parallel ffmpeg extraction) ──
        print(f"[1/4] Sampling {num_samples} frames (threaded) ...")
        t_start = time.time()
        start_t = duration * 0.08
        end_t = duration * 0.95
        gap = (end_t - start_t) / num_samples
        timestamps = [start_t + i * gap for i in range(num_samples)]
        paths = [os.path.join(self.frames_dir, f"f{i:04d}.jpg") for i in range(num_samples)]

        MAX_WORKERS = min(8, os.cpu_count() or 4)

        def _extract_one(ts_path):
            ts, path = ts_path
            # Use JPEG for speed (much faster than PNG for detection)
            self._extract_frame(ts, path, quality=3)
            return (path, ts) if os.path.exists(path) else None

        frames = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = pool.map(_extract_one, zip(timestamps, paths))
        frames = [r for r in results if r is not None]

        t_extract = time.time() - t_start
        print(f"  {len(frames)} frames extracted in {t_extract:.1f}s ({MAX_WORKERS} threads)\n")

        # ── 2. Detect faces + mouth openness (THREADED) ──
        print("[2/4] Detecting faces & checking for singing (threaded) ...")
        t_detect_start = time.time()
        all_dets = []

        def _detect_one(fpath, ts):
            img = cv2.imread(fpath)
            if img is None:
                return []
            fh, fw = img.shape[:2]
            fc = (fw // 2, fh // 2)
            max_dist = np.hypot(fw, fh) / 2
            dets = []
            for face in self.detect_faces(img):
                cx, cy = face["cx"], face["cy"]
                dist = np.hypot(cx - fc[0], cy - fc[1])
                dets.append({
                    "frame": fpath, "ts": ts, "bbox": face["bbox"],
                    "conf": face["conf"],
                    "crop": img[face["bbox"][1]:face["bbox"][3], face["bbox"][0]:face["bbox"][2]].copy(),
                    "mouth": face["mouth"],
                    "center_s": 1.0 - dist / max_dist,
                    "size_s": face["area"] / video_area,
                })
            return dets

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_detect_one, fpath, ts) for fpath, ts in frames]
            for future in as_completed(futures):
                all_dets.extend(future.result())

        t_detect = time.time() - t_detect_start
        print(f"  Detection done in {t_detect:.1f}s ({MAX_WORKERS} threads)")
        print(f"  Total detections: {len(all_dets)}\n")
        if not all_dets:
            print("  No faces found — aborting.")
            self._cleanup()
            return []

        # ── 3. Cluster & identify singer ──
        print("[3/4] Clustering faces to find the singer ...")
        clusters = self._cluster_faces(all_dets)
        for c in clusters:
            self._score_cluster(c, video_area)
        clusters.sort(key=lambda c: c["_score"], reverse=True)

        print("\n  Top clusters:")
        for i, c in enumerate(clusters[:6]):
            tag = " ★ SINGER" if i == 0 else ""
            print(f"    #{i+1}  score={c['_score']:.3f}  faces={c['_n']:2d}  mouth={c['_mouth']:.2f}{tag}")
        singer = clusters[0]
        print(f"\n  -> Singer cluster: {singer['_n']} detections\n")

        # ── 4. Pick best snapshots & extract HQ (THREADED) ──
        print(f"[4/4] Selecting & extracting HQ snapshots (threaded) ...\n")
        scored = sorted(
            singer["members"],
            key=lambda m: 0.50 * m["mouth"] + 0.25 * m["conf"] + 0.25 * m["center_s"],
            reverse=True
        )

        picked = []
        MIN_GAP = max(4.0, duration / (num_snapshots * 3))
        for m in scored:
            if len(picked) >= num_snapshots:
                break
            if all(abs(m["ts"] - p["ts"]) >= MIN_GAP for p in picked):
                picked.append(m)
        for m in scored:
            if len(picked) >= num_snapshots:
                break
            if m not in picked:
                picked.append(m)

        def _extract_hq_snapshot(idx, det):
            """Extract one HQ frame + save annotated/clean/cropped."""
            ts = det["ts"]
            bbox = det["bbox"]
            raw_path = os.path.join(self.temp_dir, f"hq_{idx}.png")
            self._extract_frame(ts, raw_path, quality=1)
            hq = cv2.imread(raw_path)
            if hq is None:
                return None

            x1, y1, x2, y2 = bbox

            # 1. Annotated
            ann = hq.copy()
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 220, 80), 3)
            lbl = "SINGER"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(ann, (x1, y1 - th - 14), (x1 + tw + 8, y1), (0, 220, 80), -1)
            cv2.putText(ann, lbl, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            info_txt = f"t = {ts:.1f}s  |  singing = {det['mouth']:.2f}"
            cv2.putText(ann, info_txt, (12, ann.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            ann_path = os.path.join(self.output_dir, f"singer_{idx+1}_annotated.png")
            cv2.imwrite(ann_path, ann, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # 2. Clean
            clean_path = os.path.join(self.output_dir, f"singer_{idx+1}_clean.png")
            cv2.imwrite(clean_path, hq, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # 3. Half-body crop
            cropped_body = self._auto_crop_half_body(hq, bbox)
            crop_path = os.path.join(self.output_dir, f"singer_{idx+1}_halfbody.png")
            cv2.imwrite(crop_path, cropped_body, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            c_h, c_w = cropped_body.shape[:2]
            return {
                "annotated": ann_path, "clean": clean_path, "crop": crop_path, "ts": ts,
                "_info": (f"t={ts:.1f}s  singing={det['mouth']:.2f}  "
                          f"crop={c_w}x{c_h}px"),
                "_idx": idx,
            }

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_extract_hq_snapshot, idx, det): idx
                       for idx, det in enumerate(picked[:num_snapshots])}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

        # Sort by snapshot index for consistent output
        results.sort(key=lambda x: x["_idx"])
        for i, r in enumerate(results):
            print(f"  Snapshot {i+1}:  {r['_info']}")
            print(f"     -> {os.path.basename(r['annotated'])}")
            print(f"     -> {os.path.basename(r['clean'])}")
            print(f"     -> {os.path.basename(r['crop'])}")
        # Remove internal keys
        for r in results:
            del r["_info"]
            del r["_idx"]

        self._cleanup()
        total_t = time.time() - t_start
        print(f"\n  Total time: {total_t:.1f}s (extract={t_extract:.1f}s + detect={t_detect:.1f}s + rest={total_t - t_extract - t_detect:.1f}s)")
        print("\n" + "=" * 62)
        print(f"  DONE — {len(results)} sets saved to: {self.output_dir}")
        print("=" * 62)
        return results

    def _cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path> [output_dir] [num_samples] [num_snapshots]")
        sys.exit(0)

    video = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "singer_snapshots"
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    n_snaps = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    if not os.path.isfile(video):
        print(f"ERROR: file not found — {video}")
        sys.exit(1)

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: FFmpeg not found in PATH.")
        sys.exit(1)

    extractor = SingerSnapshotExtractor(video, outdir)
    extractor.run(num_samples=n_samples, num_snapshots=n_snaps)


if __name__ == "__main__":
    main()
