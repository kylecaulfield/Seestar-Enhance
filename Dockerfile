# Multi-stage build: compile the SPA with Node, then drop the static
# assets into the Python image. The backend (FastAPI) serves the built
# SPA from app/static when that directory exists, so one container on
# one port handles both API and UI.

FROM node:20-alpine AS frontend-build
WORKDIR /ui
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

# OCI image metadata — surfaces on ghcr.io, enables "linked repository"
# and the security-advisory workflow. Values are overridden from the
# build workflow via --label so we don't need to hand-edit per release.
LABEL org.opencontainers.image.title="Seestar Enhance" \
      org.opencontainers.image.description="Astrophotography enhancement pipeline for ZWO Seestar S50 FITS files (FastAPI + React SPA, single container)." \
      org.opencontainers.image.source="https://github.com/kylecaulfield/seestar-enhance" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./

# Built SPA lives at /app/app/static so FastAPI's StaticFiles mount finds
# it. app/main.py only enables the mount if this dir exists, so local
# (non-docker) runs still work without a build step.
COPY --from=frontend-build /ui/dist ./app/static

# Drop privileges: the pipeline and web server don't need root. Creating
# the user after COPYing narrows chown scope to just what's needed.
RUN useradd --system --no-create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Baked-in health probe so `docker run` / `podman run` / k8s all get
# liveness signalling out of the box without a sidecar manifest.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
