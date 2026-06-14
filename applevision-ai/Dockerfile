# ==============================================================================
# AppleVision AI – Production Dockerfile
# ==============================================================================
# Multi-purpose image for local, staging, and production deployments.
#
#   docker build -t applevision-ai .
#   docker run -p 8000:8000 applevision-ai
# ==============================================================================

FROM python:3.12-slim AS base

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Shashwat-19 <github.com/Shashwat-19>"
LABEL description="AppleVision AI – Production-grade Apple Classification API"
LABEL version="1.0.0"

# ── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ─────────────────────────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (layer caching) ──────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY . .

# ── Create instance directory for SQLite ──────────────────────────────────────
RUN mkdir -p /app/instance && chown -R appuser:appuser /app

# ── Switch to non-root user ──────────────────────────────────────────────────
USER appuser

# ── Environment defaults ─────────────────────────────────────────────────────
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Start Gunicorn ───────────────────────────────────────────────────────────
CMD ["gunicorn", "wsgi:app", \
     "--workers", "2", \
     "--threads", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
