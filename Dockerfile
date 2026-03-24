# Dockerfile for video_analyzer

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies needed for OpenCV/ffmpeg and typical Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application source and metadata (needed for pip install)
COPY pyproject.toml README.md /app/
COPY . /app

# Install Python package dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Expose API port
EXPOSE 3002

# Default execution (same as `video-analyzer-api` entrypoint script)
CMD ["video-analyzer-api", "--host", "0.0.0.0", "--port", "3002"]
