# ---------------------------------------------------------------------------
# Stage 1: Build Frontend (Node.js 24)
# ---------------------------------------------------------------------------
FROM node:24-bookworm-slim AS frontend-builder
WORKDIR /app/frontend

# Copy frontend package manifests and install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build static assets
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Runtime Application (Python 3.11 Slim on Debian Bookworm)
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# System dependencies for OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_IO_MAX_IMAGE_PIXELS=25000000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000

# Install runtime dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/artifacts /app/logs && \
    chown -R appuser:appuser /app

# Copy application code and curated data
COPY backend/ ./backend/
COPY core/ ./core/
COPY config/ ./config/
COPY data/samples/ ./data/samples/
COPY contracts/ ./contracts/

# Copy built frontend assets from Stage 1 into public static directory
COPY --from=frontend-builder --chown=appuser:appuser /app/frontend/dist ./frontend/dist

# Switch to non-root user
USER appuser

# Healthcheck targeting the API health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Run single uvicorn worker bound to localhost (reverse-proxied by Caddy)
CMD ["python", "-m", "uvicorn", "backend.app_factory:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
