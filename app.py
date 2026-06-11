"""
app.py — Bhajan Video Automation Pipeline REST API
====================================================
Flask application factory + all route handlers.
Designed for WSO2 Choreo deployment (matches docker-rest-user-service template).

Endpoints:
  GET  /healthz                        — Health check (root level, Choreo reads this)
  POST /api/v1/tasks                   — Create a new video processing task
  GET  /api/v1/tasks                   — List all tasks with status
  GET  /api/v1/tasks/{task_id}         — Get task details + logs
  POST /api/v1/tasks/{task_id}/run     — Start pipeline execution
  GET  /api/v1/tasks/{task_id}/download — Download final video
  DELETE /api/v1/tasks/{task_id}       — Delete a task and its files
  GET  /api/v1/                        — API info
"""

from flask import Flask, jsonify, request, send_file
import subprocess
import os
import json
import time
import uuid
import threading
import traceback
from datetime import datetime


def create_app():
    """Flask application factory (template pattern)."""
    app = Flask(__name__)

    # ── Storage ────────────────────────────────────────────────────────
    tasks_dir = os.environ.get("TASKS_DIR", "/tmp/bhajan_tasks")
    os.makedirs(tasks_dir, exist_ok=True)

    _background_tasks = {}
    _task_lock = threading.Lock()

    # ── Helpers ────────────────────────────────────────────────────────
    def task_dir(task_id):
        return os.path.join(tasks_dir, task_id)

    def state_path(task_id):
        return os.path.join(task_dir(task_id), "state.json")

    def load_state(task_id):
        sp = state_path(task_id)
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_state(task_id, state):
        os.makedirs(task_dir(task_id), exist_ok=True)
        with open(state_path(task_id), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def add_log(task_id, msg):
        state = load_state(task_id)
        if state is None:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{ts}] {msg}"
        state["logs"].append(log_line)
        state["updated_at"] = datetime.now().isoformat()
        save_state(task_id, state)
        print(f"[{task_id}] {log_line}")

    def create_task_state(url, start_time, end_time, custom_title=""):
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        os.makedirs(task_dir(task_id), exist_ok=True)
        state = {
            "task_id": task_id,
            "url": url,
            "start_time": start_time,
            "end_time": end_time,
            "custom_title": custom_title or "Bhajan Video",
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "steps_done": [],
            "logs": [],
            "final_video": "",
            "error": "",
        }
        save_state(task_id, state)
        return task_id

    def task_to_response(state, include_logs=False):
        resp = {
            "task_id": state["task_id"],
            "url": state.get("url", ""),
            "start_time": state.get("start_time", ""),
            "end_time": state.get("end_time", ""),
            "custom_title": state.get("custom_title", ""),
            "status": state.get("status", "unknown"),
            "created_at": state.get("created_at", ""),
            "updated_at": state.get("updated_at", ""),
            "steps_done": state.get("steps_done", []),
            "error": state.get("error", ""),
        }
        if include_logs:
            resp["logs"] = state.get("logs", [])
        return resp

    # ── Time Normalizer ────────────────────────────────────────────────
    def normalize_time(ts):
        """Fix common time format mistakes: '0002:50' -> '00:02:50'."""
        ts = ts.strip()
        parts = ts.split(":")
        if len(parts) == 3:
            try:
                h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                return f"{h:02d}:{m:02d}:{int(s):02d}"
            except ValueError:
                pass
        if len(parts) == 2:
            try:
                left, right = int(parts[0]), float(parts[1])
                if right > 59:
                    return f"{left:02d}:{int(right):02d}:00"
                else:
                    return f"00:{left:02d}:{int(right):02d}"
            except ValueError:
                pass
        try:
            total_secs = int(float(ts))
            h, rem = divmod(total_secs, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        except ValueError:
            pass
        return ts

    # ── Video Resolution Check ─────────────────────────────────────────
    def get_video_resolution(path):
        """Quick ffprobe wrapper returning (w, h)."""
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_entries", "stream=width,height", path],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0:
                streams = json.loads(probe.stdout).get("streams", [])
                for s in streams:
                    if s.get("width") and s.get("height"):
                        return s["width"], s["height"]
        except Exception:
            pass
        return 0, 0

    # ── Pipeline Runner (background thread) ───────────────────────────
    def run_pipeline_background(task_id):
        """Execute the full 4-step pipeline in a background thread."""
        state = load_state(task_id)
        if not state:
            return
        url = state["url"]
        start_time = state["start_time"]
        end_time = state["end_time"]

        try:
            state["status"] = "running"
            state["updated_at"] = datetime.now().isoformat()
            save_state(task_id, state)
            add_log(task_id, "Pipeline started")
            add_log(task_id, f"  URL: {url}")
            add_log(task_id, f"  Trim: {start_time} -> {end_time}")

            raw_path = os.path.join(task_dir(task_id), "raw_video.mp4")
            cut_path = os.path.join(task_dir(task_id), "cut_video.mp4")
            output_path = os.path.join(task_dir(task_id), "final_video.mp4")

            # ── Step 1: Download + Trim ──
            if "download" not in state.get("steps_done", []):
                add_log(task_id, "Step 1/4: Downloading from YouTube...")
                result = subprocess.run(
                    ["yt-dlp", "--no-warnings",
                     "-f", "bestvideo[height>=720]+bestaudio/best[height>=720]/bestvideo+bestaudio/best",
                     "--merge-output-format", "mp4",
                     "-o", raw_path, url],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0 or not os.path.exists(raw_path):
                    raise RuntimeError(f"YouTube download failed: {(result.stderr or '')[-200:]}")

                size_mb = os.path.getsize(raw_path) / 1048576
                w, h = get_video_resolution(raw_path)
                add_log(task_id, f"  Downloaded: {size_mb:.1f} MB ({w}x{h})")

                # Quality gate: reject < 480p
                if h > 0 and h < 480:
                    os.remove(raw_path)
                    raise RuntimeError(f"Video quality too low ({h}p), need at least 480p")

                # Trim
                add_log(task_id, f"  Trimming: {start_time} -> {end_time} ...")
                trim_result = subprocess.run(
                    ["ffmpeg", "-y", "-ss", start_time, "-i", raw_path,
                     "-to", end_time, "-c:v", "copy", "-c:a", "copy",
                     "-avoid_negative_ts", "make_zero", cut_path],
                    capture_output=True, text=True, timeout=300,
                )
                if trim_result.returncode != 0 or not os.path.exists(cut_path):
                    raise RuntimeError(f"Trim failed: {(trim_result.stderr or '')[-200:]}")

                add_log(task_id, f"  Trim done: {os.path.getsize(cut_path) / 1048576:.1f} MB")
                state = load_state(task_id)
                state["steps_done"].append("download")
                state["updated_at"] = datetime.now().isoformat()
                save_state(task_id, state)

            # ── Step 2: Singer Detection ──
            if "detect_singer" not in state.get("steps_done", []):
                add_log(task_id, "Step 2/4: Singer detection...")
                snap_dir = os.path.join(task_dir(task_id), "singer_snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                try:
                    from detect import SingerSnapshotExtractor
                    extractor = SingerSnapshotExtractor(cut_path, snap_dir)
                    results = extractor.run(num_samples=20, num_snapshots=3)
                    if results:
                        import shutil, random
                        chosen = random.choice(results)
                        crop_path = os.path.join(task_dir(task_id), "singer_crop.png")
                        shutil.copy2(chosen["crop"], crop_path)
                        add_log(task_id, f"  Singer crop saved (t={chosen['ts']:.1f}s)")
                    else:
                        add_log(task_id, "  No singer faces detected (skipping)")
                except Exception as e:
                    add_log(task_id, f"  Singer detection skipped: {str(e)[:100]}")

                state = load_state(task_id)
                state["steps_done"].append("detect_singer")
                state["updated_at"] = datetime.now().isoformat()
                save_state(task_id, state)

            # ── Step 3: Edge Detection ──
            if "edge_detect" not in state.get("steps_done", []):
                add_log(task_id, "Step 3/4: Edge detection...")
                try:
                    from detect_edges_video import sample_random_frames, find_consensus_footer
                    import cv2
                    cap = cv2.VideoCapture(cut_path)
                    if cap.isOpened():
                        samples = sample_random_frames(cap, 20)
                        cap.release()
                        if samples:
                            h, w = samples[0].shape[:2]
                            footer_top, lb_top, lb_bot = find_consensus_footer(
                                samples, h, w, verbose=False)
                            edge_params = {
                                "lb_top": int(lb_top), "lb_bot": int(lb_bot),
                                "footer_top": int(footer_top), "margin": 0,
                                "frame_w": w, "frame_h": h,
                            }
                            params_path = os.path.join(task_dir(task_id), "edge_params.json")
                            with open(params_path, "w") as f:
                                json.dump(edge_params, f)
                            add_log(task_id, f"  Letterbox: top={lb_top} bottom={lb_bot}")
                        else:
                            add_log(task_id, "  No frames sampled (skipping)")
                    else:
                        add_log(task_id, "  Cannot open video (skipping)")
                except Exception as e:
                    add_log(task_id, f"  Edge detection skipped: {str(e)[:100]}")

                state = load_state(task_id)
                state["steps_done"].append("edge_detect")
                state["updated_at"] = datetime.now().isoformat()
                save_state(task_id, state)

            # ── Step 4: Video Export ──
            if "video_export" not in state.get("steps_done", []):
                add_log(task_id, "Step 4/4: Video export...")

                params_file = os.path.join(task_dir(task_id), "edge_params.json")
                crop_filter = ""
                if os.path.exists(params_file):
                    with open(params_file) as f:
                        ep = json.load(f)
                    lb_top = ep.get("lb_top", 0)
                    lb_bot = ep.get("lb_bot", ep.get("frame_h", 1080))
                    footer_top = ep.get("footer_top", 0)
                    fw = ep.get("frame_w", 1920)
                    fh = ep.get("frame_h", 1080)
                    crop_y = lb_top
                    crop_bottom = footer_top if footer_top > 0 else lb_bot
                    crop_h = crop_bottom - crop_y

                    # Crop safety: validate coordinates fit within actual video
                    actual_w, actual_h = get_video_resolution(cut_path)
                    if actual_w and actual_h:
                        if crop_y + crop_h > actual_h:
                            add_log(task_id, "  Crop out of bounds detected, skipping crop")
                            crop_filter = ""
                        else:
                            if crop_h > 0 and crop_h <= fh:
                                crop_filter = f",crop={fw}:{crop_h}:0:{crop_y},scale={fw}:{fh}"
                    else:
                        if crop_h > 0 and crop_h <= fh:
                            crop_filter = f",crop={fw}:{crop_h}:0:{crop_y},scale={fw}:{fh}"

                if crop_filter:
                    cmd = [
                        "ffmpeg", "-y", "-i", cut_path,
                        "-vf", f"null{crop_filter}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        output_path,
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y", "-i", cut_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        output_path,
                    ]

                export_result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if export_result.returncode != 0 or not os.path.exists(output_path):
                    add_log(task_id, "  Encode failed, trying stream copy fallback...")
                    fallback_cmd = [
                        "ffmpeg", "-y", "-i", cut_path,
                        "-c:v", "copy", "-c:a", "copy",
                        "-movflags", "+faststart",
                        output_path,
                    ]
                    fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)
                    if fallback_result.returncode != 0 or not os.path.exists(output_path):
                        raise RuntimeError(f"Video export failed: {(export_result.stderr or '')[-200:]}")

                size_mb = os.path.getsize(output_path) / 1048576
                add_log(task_id, f"  Export done: {size_mb:.1f} MB")
                state = load_state(task_id)
                state["steps_done"].append("video_export")
                state["status"] = "completed"
                state["final_video"] = output_path
                state["updated_at"] = datetime.now().isoformat()
                save_state(task_id, state)
                add_log(task_id, "Pipeline completed!")
                return

        except Exception as e:
            err_msg = str(e)
            traceback.print_exc()
            state = load_state(task_id)
            if state:
                state["status"] = "failed"
                state["error"] = err_msg
                state["updated_at"] = datetime.now().isoformat()
                save_state(task_id, state)
            add_log(task_id, f"Pipeline failed: {err_msg}")
        finally:
            with _task_lock:
                if task_id in _background_tasks:
                    del _background_tasks[task_id]

    # ================================================================
    # ROUTES
    # ================================================================

    # ── Health Check (root level — Choreo reads this) ────────────────
    @app.route('/healthz', methods=['GET'])
    def healthz():
        return jsonify({
            "message": "Bhajan Video Pipeline is healthy",
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }), 200

    # ── API Info ──────────────────────────────────────────────────────
    @app.route('/api/v1/', methods=['GET'])
    def api_info():
        return jsonify({
            "name": "Bhajan Video Automation Pipeline API",
            "version": "1.0.0",
            "description": "REST API for bhajan video processing - download, detect singer, crop letterbox, and export final video.",
            "endpoints": {
                "POST /api/v1/tasks": "Create a new processing task",
                "GET  /api/v1/tasks": "List all tasks",
                "GET  /api/v1/tasks/{id}": "Get task details + logs",
                "POST /api/v1/tasks/{id}/run": "Start pipeline execution",
                "GET  /api/v1/tasks/{id}/download": "Download final video",
                "DELETE /api/v1/tasks/{id}": "Delete a task",
                "GET  /healthz": "Health check",
            }
        })

    # ── CRUD: Tasks ─────────────────────────────────────────────────
    @app.route('/api/v1/tasks', methods=['GET'])
    def list_tasks():
        """List all tasks. Query param ?status=completed|failed|running to filter."""
        tasks = []
        status_filter = request.args.get('status', '').strip().lower()
        if not os.path.isdir(tasks_dir):
            return jsonify({"tasks": tasks, "count": 0})

        for entry in sorted(os.listdir(tasks_dir), reverse=True):
            state_file = os.path.join(tasks_dir, entry, "state.json")
            if os.path.isfile(state_file):
                try:
                    with open(state_file, "r") as f:
                        state = json.load(f)
                    if status_filter and state.get("status", "").lower() != status_filter:
                        continue
                    tasks.append(task_to_response(state, include_logs=False))
                except Exception:
                    continue

        return jsonify({"tasks": tasks, "count": len(tasks)})

    @app.route('/api/v1/tasks', methods=['POST'])
    def create_task():
        """Create a new video processing task."""
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        url = data.get("url", "").strip()
        start_time = data.get("start_time", "00:00:00").strip()
        end_time = data.get("end_time", "").strip()
        custom_title = data.get("custom_title", "").strip()

        if not url:
            return jsonify({"error": "url is required"}), 400
        if not end_time:
            return jsonify({"error": "end_time is required"}), 400

        # Normalize time formats
        start_time = normalize_time(start_time)
        end_time = normalize_time(end_time)

        task_id = create_task_state(url, start_time, end_time, custom_title)
        add_log(task_id, f"Task created: {url}")

        state = load_state(task_id)
        return jsonify(task_to_response(state)), 201

    @app.route('/api/v1/tasks/<task_id>', methods=['GET'])
    def get_task(task_id):
        """Get task details. Query param ?logs=true to include full logs."""
        state = load_state(task_id)
        if not state:
            return jsonify({"error": "Task not found"}), 404

        include_logs = request.args.get('logs', '').lower() == 'true'
        return jsonify(task_to_response(state, include_logs=include_logs))

    @app.route('/api/v1/tasks/<task_id>', methods=['DELETE'])
    def delete_task(task_id):
        """Delete a task and all its files."""
        td = task_dir(task_id)
        if not os.path.isdir(td):
            return jsonify({"error": "Task not found"}), 404

        import shutil
        shutil.rmtree(td, ignore_errors=True)

        with _task_lock:
            if task_id in _background_tasks:
                del _background_tasks[task_id]

        return '', 204

    # ── Pipeline Execution ─────────────────────────────────────────────
    @app.route('/api/v1/tasks/<task_id>/run', methods=['POST'])
    def run_task(task_id):
        """Start pipeline execution for a task (runs in background)."""
        state = load_state(task_id)
        if not state:
            return jsonify({"error": "Task not found"}), 404

        if state["status"] == "running":
            return jsonify({"error": "Task is already running", "task_id": task_id}), 409
        if state["status"] == "completed":
            return jsonify({"error": "Task already completed. Delete and recreate to run again.", "task_id": task_id}), 409

        with _task_lock:
            if task_id in _background_tasks:
                return jsonify({"error": "Task is already running", "task_id": task_id}), 409

            thread = threading.Thread(
                target=run_pipeline_background,
                args=(task_id,),
                daemon=True,
            )
            _background_tasks[task_id] = thread
            thread.start()

        return jsonify({
            "message": "Pipeline started",
            "task_id": task_id,
            "status": "running",
        }), 202

    # ── Download Final Video ──────────────────────────────────────────
    @app.route('/api/v1/tasks/<task_id>/download', methods=['GET'])
    def download_video(task_id):
        """Download the final processed video."""
        state = load_state(task_id)
        if not state:
            return jsonify({"error": "Task not found"}), 404

        if state["status"] != "completed":
            return jsonify({"error": f"Task not completed (status: {state['status']})", "task_id": task_id}), 409

        video_path = state.get("final_video", "")
        if not video_path or not os.path.isfile(video_path):
            return jsonify({"error": "Final video file not found", "task_id": task_id}), 404

        filename = f"{task_id}_final.mp4"
        return send_file(
            video_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=filename,
        )

    return app
