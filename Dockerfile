# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Runtime config (Cloud Run يضبط PORT)
ENV GUNICORN_WORKERS=2 \
    GUNICORN_THREADS=8 \
    GUNICORN_TIMEOUT=120

EXPOSE 8080

# Non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request; url=f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/ready'; \
try: \
  r=urllib.request.urlopen(url, timeout=3); sys.exit(0 if r.getcode()==200 else 1) \
except Exception: \
  sys.exit(1)"]

# Start server
CMD ["sh","-c","gunicorn \
  -w ${GUNICORN_WORKERS:-2} -k gthread --threads ${GUNICORN_THREADS:-8} \
  --timeout ${GUNICORN_TIMEOUT:-120} \
  -b 0.0.0.0:${PORT:-8080} \
  --access-logfile - --error-logfile - --log-level debug \
  main:app"]
