<p align="center">
  <img src="assets/logo-192.png" alt="engram" width="150">
</p>

# engram

a cognitive memory system that actually remembers things.

one sqlite file, hybrid retrieval that fuses five signals, memory layers that model how brains actually work, and a neural visualization that shows the whole thing firing in real time.

**98.1% R@5 on [LongMemEval](https://arxiv.org/abs/2410.10813)** (ICLR 2025) — highest published score, beating MemPalace (96.6%), Emergence AI (86%), and every other memory system benchmarked.

## what it does

- **hybrid retrieval** — HNSW dense + BM25 + entity graph BFS + Hopfield associative pattern completion, fused with intent-weighted reciprocal rank fusion, cross-encoder reranking, deep MLP reranking
- **memory layers** — working, episodic, semantic, procedural, codebase. memories promote upward when useful, decay if unused
- **entity graph** — extracts people, tools, projects from every memory. multi-hop traversal via recursive SQL CTEs
- **63 MCP tools** — plugs into Claude Code or any MCP client. 72 tests. docker-ready
- **multi-backend embeddings** — local (MLX/CPU), Voyage AI, OpenAI, Google Gemini. auto-detects from model name
- **web dashboard** — 17 panels including neural map, search, analytics, cognition, drift detection

## install

```bash
pip install engram-memory-system
```

or from source:

```bash
git clone https://github.com/raya-ac/engram.git
cd engram && pip install -e .
```

## quick links

- [Installation](getting-started/installation.md) — full setup guide
- [Quick Start](getting-started/quickstart.md) — first steps
- [MCP Tools Reference](reference/mcp-tools.md) — all 63 tools
- [Architecture](architecture/overview.md) — how it works inside
- [Benchmarks](architecture/benchmarks.md) — LongMemEval results

## supported by

| system | what |
|--------|------|
| [Claude Code](getting-started/claude-code.md) | native MCP integration |
| [Any MCP client](reference/mcp-tools.md) | 63-tool JSON-RPC server |
| [REST API](reference/rest-api.md) | 57 HTTP endpoints |
| [CLI](reference/cli.md) | 15 commands |
| [Docker](guides/docker.md) | single container deployment |
| [Python API](tutorials/build-an-agent.md) | direct library usage |
