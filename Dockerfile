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
# The app runs as "adsci". Each running analysis kernel gets its own user (kernel0..kernel15) with a private
# group shared only with the app, so kernels can't read /app/data (credentials) or each other's files/processes.
# Kernel users have no persistent home: each kernel's HOME is its fresh session directory.
RUN apt-get update && apt-get install -y --no-install-recommends sudo procps && rm -rf /var/lib/apt/lists/* \
 && groupadd adsci-kernel \
 && useradd --create-home --uid 1000 adsci \
 && for i in $(seq 0 15); do \
      useradd --no-create-home --home-dir /nonexistent --uid $((2000 + i)) --user-group --groups adsci-kernel kernel$i \
      && usermod --append --groups kernel$i adsci; \
    done \
 && mkdir -p /app/data /tmp/adsci-kernels && chown adsci /app/data /tmp/adsci-kernels \
 && chmod 700 /app/data && chmod 711 /tmp/adsci-kernels \
 && printf '%s\n' \
    'Defaults:adsci !requiretty, env_keep += "ADSCI_INTERNAL_URL ADSCI_SESSION_ID ADSCI_TOKEN ADSCI_PREVIEW_ROWS ADSCI_KERNEL_MEMORY_BYTES HOME IPYTHONDIR MPLCONFIGDIR PYTHONNOUSERSITE"' \
    'adsci ALL=(%adsci-kernel) NOPASSWD: /app/backend/.venv/bin/python -m ipykernel_launcher -f *, /usr/bin/pkill -KILL -U *' \
    > /etc/sudoers.d/adsci-kernel && chmod 440 /etc/sudoers.d/adsci-kernel
ENV ADSCI_KERNEL_USERS='["kernel0","kernel1","kernel2","kernel3","kernel4","kernel5","kernel6","kernel7","kernel8","kernel9","kernel10","kernel11","kernel12","kernel13","kernel14","kernel15"]' \
    ADSCI_KERNEL_WORK_ROOT=/tmp/adsci-kernels
USER adsci
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["adsci"]
