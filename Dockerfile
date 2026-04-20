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

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
