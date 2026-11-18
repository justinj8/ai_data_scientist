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
RUN useradd --create-home --uid 1000 adsci && mkdir -p /app/data && chown -R adsci /app/data
USER adsci
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["adsci"]
