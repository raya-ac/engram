# Docker

run engram as a container with persistent data.

## quick start

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
docker compose up -d
```

web dashboard at `http://localhost:8420`.

## docker compose

```yaml
services:
  engram:
    build: .
    ports:
      - "8420:8420"
    volumes:
      - engram-data:/data
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - ENGRAM_DB_PATH=/data/memory.db
      # - VOYAGE_API_KEY=your-key
      # - OPENAI_API_KEY=your-key
    restart: unless-stopped

volumes:
  engram-data:
```

## configuration

mount your `config.yaml` to `/app/config.yaml`. set API keys via environment variables.

the database lives in the `engram-data` volume at `/data/memory.db`.

## auth

lock down the web dashboard:

```yaml
# config.yaml
web:
  auth_token: "your-secret-token"
```

access with `?token=your-secret-token` in the URL or `Authorization: Bearer your-secret-token` header.

## MCP over SSE

for remote MCP access (not stdio):

```yaml
services:
  engram-mcp:
    build: .
    ports:
      - "8421:8421"
    command: ["python", "-m", "engram", "serve", "--mcp-sse", "--port", "8421"]
    volumes:
      - engram-data:/data
    environment:
      - ENGRAM_DB_PATH=/data/memory.db
```

endpoints:

- `POST /mcp` — JSON-RPC
- `GET /sse` — SSE stream
- `GET /health` — health check

## building the image

```bash
docker build -t engram .
docker run -p 8420:8420 -v engram-data:/data engram
```

the Dockerfile uses `python:3.12-slim`, installs all deps including API backends, and exposes port 8420.
