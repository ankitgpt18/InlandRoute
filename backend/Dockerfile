# =============================================================================
# AIDSTL Project — Backend Dockerfile
# Multi-stage build: builder → runtime
# "Predicting Inland Waterway Navigability Using Satellite Remote Sensing
#  and Deep Learning"
#
# Build:
#   docker build -t aidstl-api:latest .
#
# Run:
#   docker run -p 8000:8000 --env-file .env aidstl-api:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — Builder
# Installs all Python dependencies into a virtual environment.
# The venv is then copied cleanly into the slim runtime image,
# avoiding the need for build tools in production.
# -----------------------------------------------------------------------------

FROM python:3.11-slim AS builder

# Build-time metadata
LABEL stage="builder"
LABEL maintainer="AIDSTL Team"
LABEL project="AIDSTL — Inland Waterway Navigability"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # pip settings
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    # Virtual environment path
    VENV_PATH=/opt/venv

# Install system build dependencies
# - gcc / g++ / build-essential: required by some C-extension packages (e.g. psycopg2, shapely)
# - libpq-dev: PostgreSQL client headers (psycopg2-binary actually bundles these,
#              but psycopg2 source builds need them)
# - libgeos-dev / libproj-dev / libgdal-dev: geospatial libraries for shapely / pyproj
# - libgdal-dev is needed if geopandas uses GDAL bindings
# - curl: used to verify connectivity in CI
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        libpq-dev \
        libgeos-dev \
        libproj-dev \
        libffi-dev \
        libssl-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create the virtual environment
RUN python -m venv ${VENV_PATH}

# Upgrade pip / setuptools / wheel inside the venv first
RUN ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel

# Copy only the requirements file first to leverage Docker layer caching.
# The expensive pip install step is re-run only when requirements.txt changes.
WORKDIR /build
COPY requirements.txt .

# Install all Python dependencies into the venv
# --no-deps is NOT set so transitive dependencies are resolved correctly.
RUN ${VENV_PATH}/bin/pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 — Runtime
# Minimal production image with only the application code and the pre-built venv.
# Runs as a non-root user for security hardening.
# -----------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

# Runtime metadata
LABEL maintainer="AIDSTL Team"
LABEL project="AIDSTL — Inland Waterway Navigability"
LABEL description="FastAPI backend for predicting inland waterway navigability"
LABEL version="1.0.0"

# Python / runtime environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Activate the venv by prepending it to PATH
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    # Application defaults (overridden at runtime via .env or docker-compose)
    APP_HOME=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=INFO \
    ENVIRONMENT=production \
    # Geospatial library paths (required by pyproj / geopandas)
    PROJ_LIB=/usr/share/proj

# Install ONLY the minimal runtime system libraries.
# We do NOT install build tools here — they stay in the builder stage.
# - libpq5: PostgreSQL client runtime (required by asyncpg / psycopg2)
# - libgeos-c1v5: GEOS geometry runtime (required by shapely)
# - libproj25 / proj-data: PROJ runtime + datum grids (required by pyproj)
# - libgomp1: OpenMP runtime (required by LightGBM, XGBoost)
# - ca-certificates: TLS root certificates for outbound HTTPS (GEE, S3, etc.)
# - curl: lightweight health-check tool used in HEALTHCHECK instruction
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgeos-c1v5 \
        libproj25 \
        proj-data \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# ---------------------------------------------------------------------------
# Non-root user setup
# Using a dedicated system account (uid 10001) with no shell and no home
# directory avoids privilege escalation risks in the container.
# ---------------------------------------------------------------------------

RUN groupadd --system --gid 10001 aidstl \
    && useradd --system \
               --uid 10001 \
               --gid aidstl \
               --no-create-home \
               --shell /usr/sbin/nologin \
               aidstl

# ---------------------------------------------------------------------------
# Application directory layout
# ---------------------------------------------------------------------------

WORKDIR ${APP_HOME}

# Create required subdirectories with correct ownership before dropping privileges.
# /app/secrets  — GEE service account key and other secrets (mounted at runtime)
# /app/logs     — application log files (if LOG_FILE is set)
# /app/tmp      — temporary files for processing
RUN mkdir -p \
        ${APP_HOME}/secrets \
        ${APP_HOME}/logs \
        ${APP_HOME}/tmp \
    && chown -R aidstl:aidstl ${APP_HOME}

# Copy application source code
# Copy selectively to keep the image lean and avoid leaking dev files.
COPY --chown=aidstl:aidstl app/ ${APP_HOME}/app/

# Copy any optional static assets or alembic migration scripts
# (comment out if not present in your project)
# COPY --chown=aidstl:aidstl alembic/ ${APP_HOME}/alembic/
# COPY --chown=aidstl:aidstl alembic.ini ${APP_HOME}/alembic.ini

# ---------------------------------------------------------------------------
# Switch to non-root user
# All subsequent instructions (CMD, ENTRYPOINT, HEALTHCHECK) run as this user.
# ---------------------------------------------------------------------------

USER aidstl

# ---------------------------------------------------------------------------
# Expose the application port
# ---------------------------------------------------------------------------

EXPOSE ${PORT}

# ---------------------------------------------------------------------------
# Health check
# Kubernetes / ECS will use /health/live because it does no I/O and
# returns immediately if the Python process is alive.
# The readiness probe (/health/ready) checks DB + Redis + model status
# and should be configured separately in the orchestrator.
# ---------------------------------------------------------------------------

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl --silent --fail "http://localhost:${PORT}/health/live" \
        || exit 1

# ---------------------------------------------------------------------------
# Default command
# Uvicorn is started with:
#   --host / --port        from environment variables
#   --workers              set to WORKERS (should be 1 in Kubernetes pod;
#                          HPA scales pods, not workers)
#   --loop uvloop          high-performance asyncio event loop
#   --http httptools       fast HTTP parser
#   --log-level            from LOG_LEVEL env var
#   --access-log           enable access logging
#   --proxy-headers        trust X-Forwarded-For from the Nginx reverse proxy
#   --forwarded-allow-ips  accept proxy headers only from trusted sources
#
# The app factory string "app.main:create_app()" is used so that the
# FastAPI lifespan context manager fires correctly with multiple workers.
# ---------------------------------------------------------------------------

CMD ["sh", "-c", \
     "exec uvicorn app.main:app \
      --host ${HOST} \
      --port ${PORT} \
      --workers ${WORKERS} \
      --loop uvloop \
      --http httptools \
      --log-level ${LOG_LEVEL} \
      --access-log \
      --proxy-headers \
      --forwarded-allow-ips='*'"]
