# App image only. The model server (Ollama) runs as a separate container with GPU access;
# see docker-compose.yml. No model weights or data leave the host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CGPRR_ROOT=/app CGPRR_LLM_URL=http://ollama:11434

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY collections ./collections
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 app && mkdir -p indexes logs out .cache data && chown -R app /app
USER app

ENTRYPOINT ["cgprr"]
CMD ["--help"]
