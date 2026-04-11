FROM python:3.12-slim

WORKDIR /app

# install build deps for hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install engram
COPY pyproject.toml .
COPY engram/ engram/
RUN pip install --no-cache-dir -e ".[api]"

# data volume
VOLUME /data

# default config
ENV ENGRAM_DB_PATH=/data/memory.db

EXPOSE 8420

# default: run web UI
CMD ["python", "-m", "engram", "serve", "--web", "--port", "8420"]
