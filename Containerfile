# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.12-slim-bookworm

# Copy uv binaries directly from official uv image
COPY --from=uv_bin /uv /uvx /bin/

# Configure uv and Python runtime environment
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal OS dependencies for MediaPipe C++ runtime and pyttsx3 speech engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgles2 \
    libegl1 \
    espeak-ng \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Layer caching: copy dependency manifests first
COPY pyproject.toml uv.lock* ./

# Install dependencies without installing the project itself
RUN uv sync --no-install-project

# Copy application source code
COPY . .

# Complete project sync
RUN uv sync

CMD ["uv", "run", "pytest", "-v"]
