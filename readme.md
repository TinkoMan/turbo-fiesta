# Bhajan Video Automation Pipeline API

REST API for bhajan video processing — download YouTube videos, detect singer faces, crop letterbox/footer, and export final processed video.

## Deploy on WSO2 Choreo

1. Push this repo to GitHub
2. In Choreo: Create Component → Service → Connect GitHub repo
3. Build Configuration:
   - Build Pack: **Dockerfile**
   - Dockerfile Path: `Dockerfile`
   - Docker Context Path: `.`
4. Click **Build & Deploy**

## API Usage

```bash
# Create a task
curl -X POST https://<choreo-url>/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"url":"https://youtube.com/watch?v=xxx","start_time":"00:02:00","end_time":"00:10:30"}'

# Run the pipeline
curl -X POST https://<choreo-url>/api/v1/tasks/<task_id>/run

# Check status & logs
curl https://<choreo-url>/api/v1/tasks/<task_id>?logs=true

# Download final video
curl -O https://<choreo-url>/api/v1/tasks/<task_id>/download
```

## Pipeline Steps

1. **Download + Trim** — Downloads YouTube video (720p+), trims to specified time range
2. **Singer Detection** — Detects singer face using OpenCV cascade, saves crop
3. **Edge Detection** — Detects letterbox bars and footer overlay boundaries
4. **Video Export** — Crops letterbox/footer, re-encodes with H.264, outputs final MP4
