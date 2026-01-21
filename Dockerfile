# ============================================
# Vision Agents Docker Image
# Lightweight base image - source code mounted via volumes
# ============================================

FROM python:3.12-slim

# Install system dependencies (including cmake for face-recognition/dlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN groupadd --gid 1000 visionagent \
    && useradd --uid 1000 --gid visionagent --shell /bin/bash --create-home visionagent

WORKDIR /app

# Copy dependency files and required directories (for uv sync)
COPY pyproject.toml uv.lock ./
COPY agents-core ./agents-core
COPY plugins ./plugins

# Install Python dependencies
RUN uv sync --frozen

# Install additional backend dependencies
RUN uv pip install \
    fastapi \
    uvicorn[standard] \
    python-dotenv \
    getstream \
    pydantic \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-prometheus \
    prometheus-client \
    face-recognition \
    opencv-python \
    ultralytics \
    tweepy \
    Pillow \
    aiofiles

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/agents-core:/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_LINK_MODE=copy

# Default command
CMD ["bash"]
