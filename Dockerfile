# --- Frontend build -----------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Backend runtime ----------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, for layer caching.
COPY backend/pyproject.toml backend/uv.lock backend/README.md backend/.python-version ./backend/
RUN cd backend && uv sync --frozen --no-dev --no-install-project
COPY backend/ ./backend/
RUN cd backend && uv sync --frozen --no-dev

ENV PATH="/app/backend/.venv/bin:$PATH"
COPY sample_data/ ./sample_data/
RUN python sample_data/generate.py
COPY --from=web /web/dist ./frontend/dist

ENV ADSCI_HOST=0.0.0.0 \
    ADSCI_DATA_DIR=/app/data \
    ADSCI_SECRET_KEY_FILE=/app/data/secret.key \
    ADSCI_STATIC_DIR=/app/frontend/dist \
    ADSCI_SAMPLE_DIR=/app/sample_data/out
# The app runs as "adsci"; analysis kernels run as "kernel", which can't read /app/data (credentials, uploads).
RUN apt-get update && apt-get install -y --no-install-recommends sudo && rm -rf /var/lib/apt/lists/* \
 && groupadd adsci-kernel \
 && useradd --create-home --uid 1000 --groups adsci-kernel adsci \
 && useradd --create-home --uid 1001 --groups adsci-kernel kernel \
 && mkdir -p /app/data /tmp/adsci-kernels && chown adsci /app/data && chmod 700 /app/data \
 && chown adsci:adsci-kernel /tmp/adsci-kernels && chmod 2770 /tmp/adsci-kernels \
 && printf '%s\n' \
    'Defaults:adsci !requiretty, env_keep += "ADSCI_INTERNAL_URL ADSCI_SESSION_ID ADSCI_TOKEN ADSCI_PREVIEW_ROWS ADSCI_KERNEL_MEMORY_BYTES"' \
    'adsci ALL=(kernel) NOPASSWD: /app/backend/.venv/bin/python -m ipykernel_launcher -f *' \
    > /etc/sudoers.d/adsci-kernel && chmod 440 /etc/sudoers.d/adsci-kernel
ENV ADSCI_KERNEL_USER=kernel ADSCI_KERNEL_GROUP=adsci-kernel ADSCI_KERNEL_WORK_ROOT=/tmp/adsci-kernels
USER adsci
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["adsci"]
