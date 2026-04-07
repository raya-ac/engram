# engram

a cognitive memory system that actually remembers things. built because flat markdown files don't scale and every "memory" tool i tried was either too simple (just embeddings) or too complex (needs redis + neo4j + a PhD).

engram sits in the middle. one sqlite file, hybrid retrieval that fuses three signals, memory layers that model how brains actually work, and a neural visualization that shows the whole thing firing in real time.

![neural map](https://github.com/user-attachments/assets/placeholder-neural.png)

## what it does

**hybrid retrieval** — most memory tools just do cosine similarity and call it a day. engram runs three retrieval signals in parallel (BM25 keywords, dense embeddings, entity graph traversal), fuses them with reciprocal rank fusion (k=60, from [cormack et al. 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)), then optionally reranks with a cross-encoder. the RRF paper showed this beats any individual signal by 4-5% on average, and that's held up in practice.

**memory layers** — four layers modeled after atkinson-shiffrin: working (ephemeral, current session), episodic (events, experiences), semantic (permanent knowledge), procedural (decisions, error patterns, how-to). memories promote upward when they prove useful and decay if nobody accesses them. 30-day half-life on episodic, infinite on semantic.

**entity graph** — extracts people, tools, projects, dates from every memory. builds a relationship graph with co-occurrence strength. multi-hop traversal via recursive SQL CTEs, no neo4j needed.

**dream cycle** — consolidation pass that clusters similar memories (cosine > 0.8), summarizes the clusters, generates entity "peer cards" (biographical summaries), and archives the low-value old stuff. like sleep for your memory system.

**neural visualization** — force-directed graph of entities organized in concentric rings by memory layer. neurons fire with traveling impulse particles when memories get accessed. polls the database so it works across processes. fire a query from the CLI or MCP server and watch the web UI light up.

**24 MCP tools** — plugs into claude code (or any MCP client) as a tool server. recall, remember, entity lookup, timeline queries, consolidation, the works.

## the retrieval pipeline

four stages, each one filters and reranks:

```
query
  ├── dense cosine similarity (bge-small-en-v1.5, 384-dim)  → top 3k candidates
  ├── BM25 via sqlite FTS5 (content + hypothetical queries)  → top 3k candidates
  └── entity graph traversal (extracted entities → memories)  → top k candidates
           │
           ▼
     reciprocal rank fusion (k=60)
     score = Σ 1/(60 + rank) across all signals
           │
           ▼
     temporal + importance boosting
     ebbinghaus decay, access frequency, date matching
           │
           ▼
     cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
     joint (query, document) scoring — optional, adds ~200ms
           │
           ▼
     final top-k results
```

the hypothetical query part is from [docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery.pdf) — at ingestion time, generate questions each memory might answer, index them alongside the content. fixes the vocabulary mismatch problem where your search terms don't match the stored text.

## install

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

needs python 3.11+. first run will download two small models (~100MB total):
- `BAAI/bge-small-en-v1.5` (33MB) — embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB) — reranking

## quick start

### ingest some files
```bash
engram ingest ~/notes/
engram ingest ~/projects/docs/ ~/journal/
```

supports markdown, plaintext, JSON (claude/chatgpt/slack exports), PDF. extracts atomic facts via LLM, embeds them, indexes in FTS5, extracts entities and relationships.

### search
```bash
engram search "what happened on march 28"
engram search "melee garden architecture" --debug  # shows retrieval breakdown
engram search "apple sandbox bypass" --rerank      # enables cross-encoder (slower, better)
```

### remember something directly
```bash
engram remember "Ari prefers casual tone, swearing when it fits"
engram remember "deploy command: npm run build && rsync" --layer procedural
```

### check status
```bash
engram status
# Engram Memory System
# ========================================
# Database: ~/.local/share/engram/memory.db
# Size: 3.87 MB
#
# Memories:
#   working: 2
#   episodic: 662
#   semantic: 18
#   procedural: 10
#   total: 692
#
# Entities: 694
# Relationships: 1130
```

### run the web dashboard
```bash
engram serve --web
# → http://127.0.0.1:8420
```

### entity lookup
```bash
engram entity Ari --graph
```

### run the dream cycle
```bash
engram consolidate
```

## MCP server

wire it into claude code by adding to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram", "serve", "--mcp"]
    }
  }
}
```

restart claude code. you get 24 tools:

| tool | what it does |
|------|-------------|
| `recall` | hybrid search across all layers |
| `recall_entity` | everything about a person/project/tool |
| `recall_timeline` | memories in a date range |
| `recall_related` | multi-hop graph traversal from an entity |
| `recall_recent` | last N memories |
| `recall_layer` | search within one layer |
| `remember` | store a memory |
| `remember_decision` | decision + rationale → procedural |
| `remember_error` | error + prevention → procedural |
| `remember_interaction` | Q+A pair → episodic |
| `forget` | soft-delete |
| `invalidate` | mark as no longer true |
| `update_entity` | add aliases, change type |
| `consolidate` | run dream cycle |
| `promote` / `demote` | move between layers |
| `compress` | summarize results to token budget |
| `layers` | graduated L0-L3 context for prompt injection |
| `entity_graph` | relationship subgraph as JSON |
| `access_patterns` | most-recalled memories |
| `status` | system stats |
| `ingest` | import files |
| `diary_write` / `diary_read` | session notes |

## web dashboard

full monitoring UI at `http://127.0.0.1:8420`:

- **neural map** — force-directed entity graph with concentric layer rings. neurons glow and fire impulse particles along synapses when memories are accessed. drag nodes, hover for details, click to inspect.
- **search** — hybrid search with debug mode showing all 4 retrieval stages. filter by layer, importance, date, source type.
- **memories** — browse all memories, filter by layer. inline actions: promote/demote, copy, invalidate, forget.
- **entities** — entity chips with memory counts. click to inspect relationships, add aliases, change type.
- **timeline** — date range queries.
- **remember** — tabbed forms: general, decision, error pattern, Q+A interaction.
- **analytics** — layer distribution donut chart, most recalled memories bar chart, top entities, source distribution.
- **context** — L0-L3 graduated context viewer with token counts and copy buttons. useful for prompt engineering.
- **ingest** — file import with recent ingestion log.
- **live events** — real-time feed of all memory reads/writes from any process (MCP, CLI, web).
- **session diary** — quick note-taking in the sidebar.
- **keyboard shortcuts** — `/` search, `n` neural map, `s` search, `r` remember, `a` analytics, `Esc` close inspector.

## architecture

everything lives in one sqlite file (`~/.local/share/engram/memory.db`). no external services.

```
engram/
├── store.py          # sqlite schema, CRUD, FTS5, entity graph (recursive CTEs)
├── embeddings.py     # bge-small-en-v1.5 + ms-marco cross-encoder
├── retrieval.py      # 4-stage hybrid pipeline (dense + BM25 + graph → RRF → boost → rerank)
├── extractor.py      # LLM fact extraction + hypothetical query generation
├── entities.py       # regex entity extraction, relationship graph
├── lifecycle.py      # ebbinghaus forgetting, 7-factor importance scoring, promotion/demotion
├── consolidator.py   # dream cycle (clustering, summarization, peer cards, archival)
├── layers.py         # L0-L3 graduated context retrieval
├── compress.py       # token-budget compression with entity codes
├── formats.py        # parsers for markdown, JSON chat exports, PDF, slack, email
├── llm.py            # claude CLI + mlx backend abstraction
├── mcp_server.py     # 24-tool MCP server (JSON-RPC, stdio)
├── cli.py            # CLI interface
├── config.py         # yaml config with env var overrides
└── web/
    ├── app.py        # fastapi with model warmup
    ├── routes.py     # all API endpoints
    └── templates/
        └── index.html  # single-page htmx dashboard with neural canvas
```

## what informed the design

i studied three existing memory systems and six IR papers before building this. took the best parts from each:

**systems:**
- [cmyui/ai-memory](https://github.com/cmyui/ai-memory) — LLM-extracted atomic facts, three-stage hybrid retrieval with RRF, dream cycle
- [mempalace](https://github.com/milla-jovovich/mempalace) — spatial metaphor, graduated layers, entity registry, exchange-pair chunking
- [neuro-memory](https://github.com/raya-ac/neuro-memory) — atkinson-shiffrin model, ebbinghaus forgetting, importance scoring, procedural memory

**papers:**
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) (Cormack et al. 2009) — the RRF formula and k=60 constant
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) (Hu et al. 2026) — forms/functions/dynamics taxonomy
- [docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery.pdf) (Nogueira & Lin 2019) — document expansion by query prediction
- [ColBERT-PRF](https://arxiv.org/abs/2106.11251) (Wang et al. 2021) — pseudo-relevance feedback for dense retrieval
- [BM25 Query Augmentation](https://arxiv.org/abs/2305.14087) (Chen & Wiseman 2023) — learned query expansion
- [Word Embedding GLM](https://dl.acm.org/doi/10.1145/2766462.2767780) (Ganguly et al. 2015) — embedding-based language model for IR

## config

lives at `config.yaml` or `~/.config/engram/config.yaml`. env vars override everything (`ENGRAM_DB_PATH`, etc.).

```yaml
db_path: ~/.local/share/engram/memory.db
embedding_model: BAAI/bge-small-en-v1.5
cross_encoder_model: cross-encoder/ms-marco-MiniLM-L-6-v2

retrieval:
  top_k: 10
  rrf_k: 60
  min_confidence: 0.60
  rerank_candidates: 20

lifecycle:
  forgetting_half_life_days: 30
  archive_after_days: 90
  promote_importance: 0.7
  promote_accesses: 5
  cluster_threshold: 0.8

web:
  host: 127.0.0.1
  port: 8420
```

## license

MIT
