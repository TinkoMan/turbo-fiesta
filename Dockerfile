FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (ffmpeg + curl for healthcheck)
# Split into separate RUN layers for better caching
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install yt-dlp via pip (always latest)
RUN pip3 install --no-cache-dir --timeout 120 yt-dlp

# Copy requirements and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --timeout 300 -r requirements.txt

# Copy application code
COPY main.py main.py
COPY app.py app.py
COPY detect.py detect.py
COPY detect_edges_video.py detect_edges_video.py

# Create a non-root user with UID 10016 (Choreo requirement)
RUN addgroup --system --gid 10016 choreo && \
    adduser --system --no-create-home --uid 10016 --ingroup choreo choreouser && \
    mkdir -p /tmp/bhajan_tasks && \
    chown -R choreouser:choreo /tmp/bhajan_tasks /app

# Switch to non-root user
USER 10016

# Environment
ENV TASKS_DIR=/tmp/bhajan_tasks
ENV CHANNEL_NAME=Shyam Sunder Studio
ENV PORT=8080

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Start with gunicorn on port 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "main:app"]