FROM python:3.11-slim

WORKDIR /app

# Install system deps (ffmpeg + curl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install all pip packages
RUN pip3 install --no-cache-dir --default-timeout=300 yt-dlp

COPY requirements.txt .
RUN pip3 install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy app code
COPY main.py .
COPY app.py .
COPY detect.py .
COPY detect_edges_video.py .

# Create non-root user (Choreo requirement)
RUN addgroup --system --gid 10016 choreo && \
    adduser --system --no-create-home --uid 10016 --ingroup choreo choreouser && \
    mkdir -p /tmp/bhajan_tasks && \
    chown -R choreouser:choreo /tmp/bhajan_tasks /app

USER 10016

ENV TASKS_DIR=/tmp/bhajan_tasks
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120", "main:app"]
