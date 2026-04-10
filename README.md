# engram

a cognitive memory system that actually remembers things. built because flat markdown files don't scale and every "memory" tool i tried was either too simple (just embeddings) or too complex (needs redis + neo4j + a PhD).

engram sits in the middle. one sqlite file, hybrid retrieval that fuses five signals, memory layers that model how brains actually work, and a neural visualization that shows the whole thing firing in real time. **98.1% R@5 on [LongMemEval](https://arxiv.org/abs/2410.10813)** (ICLR 2025) — highest published score, beating MemPalace (96.6%), Emergence AI (86%), and every other memory system benchmarked.

## what it does

**hybrid retrieval** — most memory tools just do cosine similarity and call it a day. engram runs four retrieval channels in parallel (BM25 keywords, dense embeddings via HNSW approximate nearest neighbors, entity graph BFS with 1-hop traversal, Hopfield associative pattern completion), fuses them with intent-weighted reciprocal rank fusion (k=60, weights vary by query type — why/when/who/how/what), then applies temporal + importance boosting, cross-encoder reranking, deep MLP reranking, gaussian noise for beneficial variation, and a minimum score threshold gate. the full pipeline: dense (HNSW) + BM25 + graph + Hopfield → intent-weighted RRF → boost → cross-encoder → MLP reranker → noise + threshold.

**memory layers** — five layers modeled after atkinson-shiffrin: working (ephemeral, auto-promotes to episodic after 30 min), episodic (events, experiences), semantic (permanent knowledge), procedural (decisions, error patterns, how-to), and codebase (compressed code knowledge — file trees, function signatures, dependency graphs). memories promote upward when they prove useful and decay if nobody accesses them. 30-day half-life on episodic, infinite on semantic.

**entity graph** — extracts people, tools, projects, dates from every memory. builds a relationship graph with co-occurrence strength. multi-hop traversal via recursive SQL CTEs, no neo4j needed. backlinks let you trace which memories are connected to which.

**dream cycle** — consolidation pass that clusters similar memories (cosine > 0.8), summarizes the clusters, generates entity "peer cards" (biographical summaries), and archives the low-value old stuff. like sleep for your memory system.

**semantic dedup** — finds near-duplicate memories by embedding distance (default threshold 0.92), auto-merges them keeping the higher-importance version. transfers entity links, merges tags and access counts. run manually or as part of consolidation.

**codebase scanning** — point `scan_codebase` at a project directory and it extracts file trees, function/class signatures, import graphs, and config files into compressed codebase-layer memories. stores ~10x fewer tokens than raw code while keeping what you actually need to work with the project.

**conversation ingest** — auto-extracts memories from Claude Code JSONL session logs. parses exchanges into Q+A pairs, classifies them (decisions, corrections, errors, task completions), and stores them in the right layer with appropriate importance scores.

**neural visualization** — force-directed graph of entities organized in concentric rings by memory layer. neurons fire with traveling impulse particles when memories get accessed. polls the database so it works across processes. fire a query from the CLI or MCP server and watch the web UI light up.

**drift detection** — memories reference file paths, function names, commands, and dependencies. those references go stale when the codebase changes. `drift_check` extracts verifiable claims from memory content and validates them against the actual filesystem — dead paths, missing functions, broken npm scripts. returns a drift score (0-100) with per-issue breakdown. zero AI cost, pure filesystem checks. `drift_fix` auto-invalidates dead references and flags stale memories. inspired by [mex](https://github.com/theDakshJaitly/mex)'s claim verification approach.

**pattern extraction** — after a session, `extract_patterns` analyzes recent activity (diary entries, new memories, events) and distills reusable procedural knowledge. classifies work into categories (workflow, gotcha, decision, integration, debug), checks novelty against existing procedural memories via embedding distance, and only stores patterns that are genuinely new. the GROW step from mex, automated.

**negative knowledge** — `remember_negative` stores explicit "what does NOT exist" claims: no caching layer, no Redux, the /admin endpoint was removed. these prevent future hallucinated recommendations. stored in the semantic layer with a NEGATIVE KNOWLEDGE prefix so they surface when you search for the thing that doesn't exist.

**enriched embeddings** — at write time, an LLM generates keywords, categorical tags, and a contextual summary for each memory. the embedding is computed over the concatenation of content + keywords + tags + summary, giving the vector richer semantic signal than raw content alone. inspired by [A-Mem](https://arxiv.org/abs/2502.12110)'s zettelkasten approach, where enriched embeddings nearly doubled multi-hop retrieval F1.

**memory evolution** — memories aren't write-once. when a new memory arrives and near-neighbors are detected (via the surprise gate), the system asks an LLM whether existing memories should be updated with the new context. old memories get smarter over time instead of going stale. from [A-Mem](https://arxiv.org/abs/2502.12110) — removing evolution dropped multi-hop F1 from 45.85% to 31.24% in ablation.

**intent-aware retrieval** — queries are classified by intent (why/when/who/how/what) and retrieval signals are dynamically weighted. "why" queries boost graph traversal for causal reasoning. "when" queries boost BM25 for date matching. "who" queries boost entity graph lookup. from [MAGMA](https://arxiv.org/abs/2601.03236)'s adaptive policy (+9% over static weighting).

**trust-weighted decay** — different sources decay at different rates. human-authored memories get full 30-day half-life. auto-extracted observations decay 3x faster. formula: `λ_eff = λ · (1 + κ·(1 - trust))`, κ=2.0. from [SuperLocalMemory V3.3](https://arxiv.org/abs/2604.04514). also: confirmation count — memories corroborated by multiple independent sources get importance boost.

**write-path CRUD** — instead of always appending then deduplicating later, new memories are classified at write time as ADD/UPDATE/NOOP by comparing against existing neighbors. updates merge content in-place. noops skip storage entirely. from [Mem0](https://arxiv.org/abs/2504.19413)'s production pipeline.

**adversarial belief probing** — during the dream cycle, randomly sample old semantic/procedural memories and challenge them: "is this still true?" beliefs that fail the probe get importance reduced. prevents fossilized false beliefs. from the March 2026 survey on [autonomous agent memory](https://arxiv.org/abs/2603.07670).

**63 MCP tools** — plugs into claude code (or any MCP client) as a tool server. recall, remember, entity lookup, codebase scanning, conversation extraction, semantic dedup, drift detection, pattern extraction, negative knowledge, quality metrics, embedding compression, community detection, timeline queries, similarity search, backlinks, consolidation, batch operations, export, health checks, the works.

## the retrieval pipeline

eight stages — four parallel channels, intent-weighted fusion, boosted, reranked, gated:

```
query
  │
  ├── intent classification (why/when/who/how/what)
  │         → dynamic signal weights per intent type
  │
  ├── dense HNSW search (bge-small-en-v1.5, 384-dim, hnswlib)  → top 3k candidates
  ├── BM25 via sqlite FTS5 (content + hypothetical queries)     → top 3k candidates
  ├── entity graph BFS (1-hop traversal, strength-weighted)     → top k candidates
  └── Hopfield associative (pattern completion, β=8.0)          → top k candidates
           │
           ▼
     intent-weighted reciprocal rank fusion (k=60)
     score = Σ w_intent · 1/(60 + rank) across 4 channels
           │
           ▼
     temporal + importance boosting
     retention regularization, access frequency, date matching
           │
           ▼
     cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
     joint (query, document) scoring — optional, adds ~200ms
           │
           ▼
     deep MLP reranker (optional, if trained)
     learned relevance from historical access patterns, <1ms
           │
           ▼
     gaussian noise (ACT-R, σ=0.02) + threshold gate
     beneficial variation + minimum score cutoff
           │
           ▼
     final top-k results
```

**deep retrieval** — optional 7th stage: a learned 2-layer MLP reranker trained on actual access patterns. which memories get accessed after being returned in search results? that signal teaches the reranker what's useful vs what's just semantically similar. takes 10 features (cosine similarity, importance, access count, age, layer one-hot, retention score) and outputs a relevance prediction. train with `train_reranker`, model persists to disk next to the database. runs automatically on every `recall` once trained. lightweight — adds <1ms per query. after the MLP, a small gaussian noise term (σ=0.02, [ACT-R](https://dl.acm.org/doi/10.1145/3765766.3765803) inspired) provides beneficial retrieval variation, and a configurable minimum score threshold gates out garbage results.

**task-aware skill selection** — `get_skills` decides whether to inject procedural knowledge and which 2-3 items to surface. three-stage gate: (1) need assessment via query surprise + domain coverage, (2) selection of top procedural memories by adaptive relevance threshold, (3) calibration with confidence scoring that filters borderline matches. based on [SkillsBench](https://arxiv.org/abs/2602.12670) finding that focused skills (+16.2pp) beat comprehensive docs (-2.9pp), and the [AGENTS.md evaluation](https://arxiv.org/abs/2602.11988) showing static context files reduce performance. the system knows when to inject (unfamiliar domain + relevant procedures = high confidence) and when to shut up (model already knows + no specific procedures = skip).

the hypothetical query part is from [docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery.pdf) — at ingestion time, generate questions each memory might answer, index them alongside the content. fixes the vocabulary mismatch problem where your search terms don't match the stored text.

## memory lifecycle

memories aren't static. they move between layers based on how useful they turn out to be.

**surprise-based importance** — at write time, every new memory is compared against existing embeddings using k-NN cosine distance (k=5). novel memories (far from anything stored) get their importance boosted up to +0.3. redundant memories (close to existing) get importance reduced and are flagged as potential duplicates. the surprise score is stored in metadata so you can audit it later. this is inspired by the [Titans paper](https://arxiv.org/abs/2501.00663) (Behrouz et al., Google) where memory updates are proportional to surprise — the gradient of the loss function. the `remember` tool now returns a `surprise` field (0-1) and warns when near-duplicates are detected.

**importance scoring** uses 9 factors:
- base importance (set at creation, 0.0-1.0, adjusted by surprise at write time)
- access frequency (log scale, how often it's been recalled)
- recency (exponential decay, trust-weighted half-life)
- emotional valence (strong emotions = more memorable)
- stability (accessed consistently over time vs burst)
- layer boost (semantic memories weighted higher)
- source trust (human=1.0, AI=0.7, interaction=0.6, ingest=0.5, dream=0.4)
- confirmation count (independently corroborated facts get boosted)
- combined into a weighted composite score

**promotion** — episodic memories that hit importance >= 0.7 and access count >= 5 get promoted to semantic (permanent). working memories auto-promote to episodic after 30 minutes or 2 accesses. the sweep runs on every `recall` call so it's basically free.

**pinning** — pin any memory with the `pin` tool or the pin button in the web UI. pinned memories are immune to the dream cycle's forgetting pass. useful for memories that are important but accessed infrequently — the kind ebbinghaus would normally archive.

**retention regularization** — forgetting is reframed as retention regularization, inspired by [Miras](https://arxiv.org/abs/2504.13173) (Behrouz et al., Google). three modes, configurable via `retention_mode` in config:
- `l2` (classic ebbinghaus): smooth exponential decay, 50% at half-life. everything fades gradually.
- `huber` (default): matches L2 near-term, transitions to linear for old memories. robust to burst-then-quiet access patterns — old-but-once-hot memories get a gentler transition instead of an infinite long tail. `huber_delta` controls the transition point.
- `elastic` (L1+L2): sparse retention. strongly-held memories stay near full strength, weakly-held ones decay faster. produces cleaner separation between keepers and forgettables. `elastic_l1_ratio` controls the L1/L2 blend.

all modes include access reinforcement — each recall strengthens retention (spaced repetition effect, log-scaled, capped at +0.3). **trust-weighted**: low-trust sources (auto-extracted, dream-generated) decay up to 3x faster than human-authored memories (`λ_eff = λ · (1 + κ·(1 - trust))`, κ=2.0, from [SuperLocalMemory V3.3](https://arxiv.org/abs/2604.04514)). after 90 days, if retention < 0.15, importance < 0.3, and access count < 3, the memory gets soft-deleted. semantic, procedural, and pinned memories don't decay.

**embedding compression** — as memories age and retention drops, their embeddings can be quantized to save storage: active (R>0.8) = 32-bit float, warm = 8-bit (3.9x compression, 0.9999 cosine fidelity), cold = 4-bit (7.6x, 0.97 fidelity), archive = 2-bit (14.6x, 0.59 fidelity). uses Fisher-Rao Quantization-Aware Distance (FRQAD) for mixed-precision comparison — inflates variance proportional to quantization loss to prevent false similarity. run with `compress_embeddings`.

**consolidation** (dream cycle) — 7-step pipeline: (1) apply forgetting curve with trust-weighted retention, (2) cluster similar memories by embedding distance and merge clusters of 5+, (3) generate peer cards for entities with enough data, (4) cross-domain synthesis — find entity pairs in different contexts with moderate embedding similarity (0.75-0.90), LLM-confirm genuine connections, create SYNTHESIZED_WITH bridges, (5) adversarial belief probing — randomly sample old semantic/procedural memories and challenge them ("is this still true?"), reduce importance on invalidated beliefs, (6) drift detection — validate memory claims against filesystem, auto-invalidate dead references, (7) prune old access logs and events. run manually with `engram consolidate` or the MCP `consolidate` tool.

## entity graph

every memory gets scanned for entities — people, tools, projects, dates, URLs, file paths. these go into an entity registry with canonical names, aliases, and types.

relationships form automatically through co-occurrence (entities mentioned in the same sentence get a CO_OCCURS link) and through pattern matching ("X uses Y" → USES, "X built Y" → CREATED, etc.). relationship strength increases with evidence count.

traversal uses recursive SQL CTEs for multi-hop queries — "show me everything connected to Ari within 2 hops" runs in a single SQL query, no graph database needed. the `recall_related` tool does this.

you can also manually link entities (`link_memories`), merge duplicates (`merge_entities`), add aliases (`update_entity`), find backlinks (`backlinks`), and fuzzy-search for entities by partial name (`search_entities`).

## editing and annotating

memories aren't write-once. you can:

- **edit content** — `edit_memory` changes the text and automatically re-embeds and rebuilds the FTS index. the memory keeps its ID, access history, and entity links.
- **annotate** — `annotate` attaches timestamped notes to a memory without touching its content. useful for adding context later ("this turned out to be wrong" or "confirmed by Ari on april 8").
- **invalidate** — `invalidate` marks a fact as no longer true with a reason. the memory stays in the database (useful for audit) but gets flagged and shown with a strikethrough in the web UI.
- **tag** — `tag` adds or removes freeform tags. `batch_tag` applies tags to all memories matching a search query.

## examples

the `examples/` directory has ready-to-use setup guides:

| file | what it covers |
|------|---------------|
| [`claude-code-setup.md`](examples/claude-code-setup.md) | full walkthrough: install, wire into claude code, add CLAUDE.md instructions, seed memories |
| [`agent-patterns.md`](examples/agent-patterns.md) | common patterns: session orientation, learning from corrections, check-before-store, cognitive scaffolding, multi-agent setup |
| [`python-client.py`](examples/python-client.py) | standalone Python usage without MCP — direct library calls for store, search, surprise scoring, reranker |
| [`custom-agent.py`](examples/custom-agent.py) | minimal conversational agent with engram memory using the Anthropic SDK |
| [`openai-compatible.py`](examples/openai-compatible.py) | same agent pattern but works with any OpenAI-compatible API (OpenAI, Ollama, vLLM, llama.cpp) |
| [`hooks-setup.md`](examples/hooks-setup.md) | auto-extract memories from conversations via claude code hooks |

## install

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

needs python 3.11+. first run will download two small models (~100MB total):
- `BAAI/bge-small-en-v1.5` (33MB) — embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB) — reranking

### optional: API embedding backends

use cloud embedding APIs instead of (or alongside) local models:

```bash
pip install -e ".[voyage]"   # voyage-3.5, voyage-3.5-lite, voyage-code-3
pip install -e ".[openai]"   # text-embedding-3-small, text-embedding-3-large
pip install -e ".[gemini]"   # gemini-embedding-001
pip install -e ".[api]"      # all three
```

set the API key and model in config.yaml or env vars:

```bash
export VOYAGE_API_KEY="your-key"    # get at https://dash.voyageai.com/
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

```yaml
# config.yaml
embedding_model: voyage-3.5         # auto-detects backend from model name
embedding_dim: 1024                 # auto-detected if model is known
```

engram auto-detects the backend from the model name — `voyage-*` uses the Voyage API, `text-embedding-*` uses OpenAI, `gemini-*` uses Gemini. or set `embedding_backend` explicitly.

supported models:

| model | provider | dim | price/1M tokens | notes |
|-------|----------|-----|-----------------|-------|
| `BAAI/bge-small-en-v1.5` | local | 384 | free | default, runs on CPU or Apple GPU |
| `voyage-3.5` | Voyage AI | 1024 | $0.18 | best retrieval quality, Anthropic recommended |
| `voyage-3.5-lite` | Voyage AI | 1024 | $0.02 | 94% of 3.5 quality, budget option |
| `voyage-code-3` | Voyage AI | 1024 | $0.18 | optimized for code |
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 | cheapest API option |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 | highest dim |
| `gemini-embedding-001` | Google | 768 | free tier | top MTEB retrieval score |

switching models requires re-embedding existing memories (`engram index rebuild` after changing the model).

## quick start

### ingest some files
```bash
engram ingest ~/notes/
engram ingest ~/projects/docs/ ~/journal/
```

supports markdown, plaintext, JSON (claude code JSONL, claude.ai JSON, chatgpt JSON tree, slack exports), PDF. extracts atomic facts via LLM, embeds them, indexes in FTS5, extracts entities and relationships.

### search
```bash
engram search "what happened on march 28"
engram search "melee garden architecture" --debug  # shows retrieval stage breakdown
engram search "apple sandbox bypass" --rerank      # enables cross-encoder (slower, better)
```

### remember something directly
```bash
engram remember "Ari prefers casual tone, swearing when it fits"
engram remember "deploy command: npm run build && rsync" --layer procedural
```

### manage ANN index
```bash
engram index rebuild     # full rebuild from all embeddings
engram index status      # check index size, vector count, last built
```

### check status
```bash
engram status
```

### entity lookup
```bash
engram entity Ari --graph
```

### check memory drift
```bash
engram drift                                    # full drift report
engram drift --search-roots ~/project/src       # also verify function names
engram drift --fix --dry-run                    # preview what would be fixed
engram drift --fix                              # auto-invalidate dead refs, flag stale
```

### extract patterns from session
```bash
engram patterns                                 # extract from last 4 hours
engram patterns --hours 24 --dry-run            # preview from last 24 hours
engram patterns --threshold 0.5                 # only store highly novel patterns
```

### run the dream cycle
```bash
engram consolidate
```

### start the web dashboard
```bash
engram serve --web
# → http://127.0.0.1:8420
```

### start the MCP server
```bash
engram serve --mcp
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

restart claude code. you get 63 tools:

**recall & search**

| tool | what it does |
|------|-------------|
| `recall` | hybrid search across all layers |
| `recall_entity` | everything about a person/project/tool — memories, relationships, timeline |
| `recall_timeline` | memories in a date range |
| `recall_related` | multi-hop graph traversal from an entity |
| `recall_recent` | last N memories by creation time |
| `recall_layer` | search within a specific layer |
| `recall_hints` | search memories but return only hints (truncated snippets + entity names) to trigger recognition without replacing cognition |
| `get_skills` | task-aware skill selection — get focused procedural guidance only when injection would help, skip when it wouldn't |
| `recall_code` | search the codebase layer for functions, classes, files |
| `recall_context` | search and return a formatted context block with token budget |
| `find_similar` | find memories most similar to a given one by embedding distance |
| `compress` | summarize search results down to a token budget |

**store & organize**

| tool | what it does |
|------|-------------|
| `remember` | store a memory with layer and importance |
| `remember_decision` | decision + rationale → procedural layer |
| `remember_error` | error pattern + prevention → procedural layer |
| `remember_interaction` | Q+A pair → episodic layer |
| `remember_project` | structured project info → semantic layer |
| `remember_negative` | store explicit negative knowledge — what does NOT exist, what should NOT be done |
| `edit_memory` | edit content of an existing memory (re-embeds automatically) |
| `annotate` | add a note to a memory without changing its content |
| `pin` / `unpin` | pin a memory so it never gets forgotten by the dream cycle |
| `forget` | soft-delete a memory |
| `invalidate` | mark a fact as no longer true |
| `tag` | add or remove tags on a memory |
| `bulk_forget` | mass cleanup by source file, layer, or date |

**entities & graph**

| tool | what it does |
|------|-------------|
| `update_entity` | add aliases, change type |
| `merge_entities` | combine two entities that are the same thing |
| `search_entities` | fuzzy search for entities by partial name |
| `entity_graph` | relationship subgraph as JSON |
| `entity_timeline` | entity's memories ordered chronologically |
| `link_memories` | manually relate two memories via their entities |
| `backlinks` | find all memories linked to a specific memory via shared entities |

**codebase**

| tool | what it does |
|------|-------------|
| `scan_codebase` | extract compressed code knowledge from a project directory |
| `recall_code` | search the codebase layer specifically |
| `list_projects` | show all scanned projects with memory counts |

**drift detection**

| tool | what it does |
|------|-------------|
| `drift_check` | verify memories against filesystem reality — dead paths, missing functions, stale memories. returns drift score 0-100 |
| `drift_fix` | auto-fix drift issues — invalidate dead refs, flag stale memories. use dry_run=true first |
| `extract_patterns` | extract reusable procedural patterns from recent session activity — only stores what's genuinely novel |

**dedup & maintenance**

| tool | what it does |
|------|-------------|
| `dedup` | find and merge near-duplicate memories by embedding similarity |
| `find_duplicates` | preview duplicate pairs without merging |
| `recompute_importance` | recalculate all importance scores with the 7-factor formula |
| `batch_tag` | add tags to all memories matching a search query |
| `train_reranker` | train the deep MLP reranker on access patterns |
| `reranker_status` | check if the deep reranker is trained |
| `compress_embeddings` | lifecycle-aware quantization (32/8/4/2-bit) with FRQAD distance metric |
| `detect_communities` | label propagation over entity graph, optional LLM summaries |
| `quality_metrics` | storage quality ratio, curation ratio, enrichment coverage |

**conversations & sessions**

| tool | what it does |
|------|-------------|
| `ingest_sessions` | auto-extract memories from recent Claude Code conversation logs |
| `session_summary` | generate summary from diary entries + recent events |

**lifecycle & system**

| tool | what it does |
|------|-------------|
| `consolidate` | run dream cycle (cluster, summarize, peer cards, forget) |
| `promote` / `demote` | move memories between layers |
| `layers` | graduated L0-L3 context for prompt injection |
| `status` | memory counts, entity counts, db size |
| `health` | embedding cache, FTS index, orphaned entities, stale working memories |
| `access_patterns` | most-recalled memories, hit rates |
| `count_by` | group counts by layer, source type, entity, or month |
| `export` | dump memories as markdown or JSON |
| `ingest` | import files or directories |
| `explain_importance` | break down a memory's importance score into 7 component factors |
| `memory_map` | high-level map of the whole system — layer counts, top entities per layer, date range, recent activity |
| `diary_write` / `diary_read` | session notes |

## benchmarks

43/43 tests across 20 subsystems. run on 446 embedded memories, Apple Silicon.

### full test suite (446 vectors, 384-dim, Apple Silicon)

| subsystem | tests | result |
|-----------|-------|--------|
| embedding | 3/3 | dim=384, norm=1.0, avg 5.1ms, batch OK |
| ANN index (HNSW) | 7/7 | 0.09ms search, 100% recall@10, 5,304 inserts/sec |
| brute-force dense | 2/2 | 0.016ms avg (100 runs) |
| intent classification | 1/1 | 6/6 correct (why/when/who/how/what) |
| full pipeline (no rerank) | 3/3 | 15.5ms avg, debug mode OK |
| full pipeline (+ cross-encoder) | 1/1 | 252ms avg |
| cross-encoder | 2/2 | correct ranking, 2.9ms/doc |
| surprise gate | 4/4 | 0.10ms avg, novel=0.85, duplicate=0.44 |
| Hopfield channel | 1/1 | <1ms |
| BM25 / FTS5 | 2/2 | 3.5ms avg |
| entity graph | 4/4 | find, relationships, 2-hop traversal (161 related) |
| memory CRUD | 2/2 | write → read → ANN verify → forget |
| layers (L0-L3) | 1/1 | 248ms, 4 layers |
| deep reranker | 1/1 | trained=True |
| importance scoring | 1/1 | 7-factor composite OK |
| store internals | 3/3 | cache cold=0.4ms hot=0.001ms |
| diary | 1/1 | write + read OK |
| events | 1/1 | logging OK |
| index I/O | 1/1 | 967 KB on disk, save=4ms load=10ms |
| config | 2/2 | ANN config + reload consistent |

### throughput (Apple Silicon, MLX GPU)

| operation | rate |
|-----------|------|
| embedding (MLX GPU) | 1,879 texts/sec |
| embedding (CPU) | 176 texts/sec |
| sqlite bulk insert | 51,000 rows/sec |
| ANN insert | 5,304 ops/sec |
| embed + store 100k | ~3 min |

### latency (Apple Silicon)

| operation | time |
|-----------|------|
| ANN dense search | 0.09ms avg |
| brute-force dense search | 0.016ms avg |
| full pipeline (no rerank) | 15.5ms avg |
| full pipeline (+ cross-encoder) | 252ms avg |
| surprise gate (k-NN) | 0.10ms avg |
| embedding | 5.1ms avg |
| cross-encoder rerank | 2.9ms/doc |
| BM25 / FTS5 | 3.5ms avg |
| Hopfield channel | <1ms |
| ANN index save | 4ms |
| ANN index load | 10ms |
| embedding cache (cold) | 0.4ms |
| embedding cache (hot) | 0.001ms |

### ANN scaling projection

| vectors | brute-force | ANN (HNSW) | speedup |
|---------|------------|------------|---------|
| 1k | 0.1ms | 0.12ms | 1x |
| 10k | 0.9ms | 0.16ms | 5x |
| 100k | 8.7ms | 0.20ms | 45x |
| 500k | 43.7ms | 0.22ms | 198x |
| 1M | 87.3ms | 0.23ms | 377x |

recall@10 accuracy: 100% (20/20 queries, ANN vs brute-force exact match)

### LongMemEval (ICLR 2025)

[LongMemEval](https://arxiv.org/abs/2410.10813) — 500 questions testing 5 long-term memory abilities (information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention) across ~40 conversation sessions per question (~115k tokens). the standard benchmark for chat assistant memory.

engram uses HNSW + BM25 + RRF fusion against per-question session haystacks. no entity graph or Hopfield (those need persistent memory, not ephemeral per-question corpora). run with `benchmarks/longmemeval/run_engram.py`.

| system | R@5 | method |
|--------|-----|--------|
| **engram v2** | **98.1%** | HNSW + BM25 + assistant BM25 + temporal boost + cross-encoder |
| MemPalace (raw) | 96.6% | ChromaDB cosine, verbatim storage |
| engram v1 | 94.7% | HNSW + BM25 + RRF |
| Emergence AI | 86.0% | RAG |
| MemPalace (AAAK) | 84.2% | compressed storage |
| EverMemOS | 83.0% | — |
| TiMem | 76.9% | temporal hierarchical |

per question type (470 non-abstention questions):

| type | n | R@5 | R@10 |
|------|---|-----|------|
| knowledge-update | 72 | 100.0% | 100.0% |
| single-session-user | 64 | 100.0% | 100.0% |
| multi-session | 121 | 99.2% | 99.2% |
| temporal-reasoning | 127 | 96.9% | 97.6% |
| single-session-assistant | 56 | 96.4% | 96.4% |
| single-session-preference | 30 | 93.3% | 96.7% |

v2 adds three channels over v1: assistant-turn BM25 (weight 0.5), timestamp proximity boost, and cross-encoder reranking on top-20 candidates. the assistant channel catches answers in assistant responses without polluting the dense index. the temporal boost favors sessions closer to the question date. the cross-encoder rescores the top candidates jointly against the query.

### retrieval quality (synthetic)

tested on synthetic memories with template-varied content (different topics, people, tools). queries use the first line of each memory verbatim — a strict exact-match test.

| metric | 500 memories | 10k memories | 100k memories |
|--------|-------------|-------------|--------------|
| recall@1 | 10% | 25% | 0% |
| recall@5 | 55% | 75% | 20% |
| recall@10 | 95% | 95% | 40% |
| coverage (top 20) | 100% | 100% | 60% |

recall drops at 100k because all synthetic memories use similar templates — finding one exact match among 100k near-duplicates is adversarially hard. real-world diverse content scores much higher.

### intent classification

| accuracy | 90% (9/10 test cases) |
|----------|----------------------|

query intent (why/when/who/how/what) is classified and used to dynamically weight retrieval signals. "why" boosts graph edges, "when" boosts BM25 date matching, "who" boosts entity lookup.

### system health metrics

| metric | description |
|--------|------------|
| storage quality | fraction of stored memories ever recalled |
| curation ratio | memories with updates/invalidations vs total |
| enrichment ratio | memories with keywords+tags+summary metadata |
| evolution count | memories updated by evolution on neighbor write |
| confirmation count | memories independently corroborated |

run `quality_metrics` via MCP or the web dashboard health panel.

## hooks

engram ships with a shell hook for Claude Code that auto-extracts memories from your conversation sessions.

```bash
# hooks/save_hook.sh — run periodically or on session end
ENGRAM_VENV=~/path/to/engram/.venv ./hooks/save_hook.sh
```

the hook finds recent Claude Code JSONL files, parses exchanges into Q+A pairs, classifies them (decisions get stored in procedural, corrections become error patterns, etc.), and stores them with appropriate importance. it skips files it's already ingested via content hash.

you can also wire it into Claude Code's hook system by adding to your settings — check `hooks/save_hook.sh` for details.

## web dashboard

full monitoring UI at `http://127.0.0.1:8420`:

- **neural map** — force-directed entity graph with concentric layer rings (semantic core → procedural → episodic → working). neurons glow and fire impulse particles along synapses when memories are accessed. drag nodes, hover for details, click to inspect. polls the database every 2s so MCP queries show up in real time.
- **search** — hybrid search with debug mode showing all 5 retrieval stages. filter chips for layer, importance slider. hint mode toggle returns truncated snippets with reveal buttons for cognitive scaffolding. search history saved to localStorage with dropdown.
- **memories** — browse all memories, filter by layer (including codebase). layer-colored left borders, importance bars, slide-in animations. every card has inline actions: edit content, promote/demote, pin/unpin, find similar, explain importance, copy, invalidate, forget. select mode for bulk operations (promote/forget multiple at once). pinned memories show gold glow.
- **entities** — entity chips with memory counts. click to open inspector with relationship graph, add aliases, change entity type.
- **timeline** — date range queries with memory cards.
- **remember** — tabbed forms: general (any layer/importance), decision (with rationale → procedural), error pattern (with prevention → procedural), Q+A interaction (→ episodic). now shows surprise score and adjusted importance after storing.
- **cognition** — three tabs for the new memory science features:
  - *surprise*: paste text to preview novelty score before storing. radial gauge visualization (green=novel, red=duplicate), k-NN distance bars, nearest memory snippet.
  - *retention*: interactive canvas chart overlaying L2, Huber, and elastic net decay curves. sliders for half-life, huber delta, and L1 ratio — curves redraw client-side in real time.
  - *reranker*: deep MLP reranker status card, train button with epoch/LR inputs, training results display.
- **bridges** — cross-domain synthesis viewer. shows entity pairs connected by the dream cycle's LLM-confirmed bridges, with similarity scores and connection descriptions.
- **analytics** — donut chart for layer distribution, bar charts for most recalled memories, top entities by memory count, source type breakdown.
- **context** — L0-L3 graduated context viewer with token counts per layer and copy buttons. query input for L3 search-based context.
- **health** — system health dashboard with 10 status cards (embedding cache, orphaned entities, stale working memories, FTS index, embedding coverage, db size, etc). plus a memory map showing top entities per layer and full date range.
- **dedup** — duplicate detection with adjustable similarity threshold slider. scan to preview duplicate pairs side by side, one-click auto-merge.
- **ingest** — file/directory path ingestion with real backend processing, session ingest button, and recent ingestion log.
- **export** — download memories as markdown or JSON from sidebar, with optional layer filter.
- **live events** — real-time feed of all memory reads/writes across all processes (MCP, CLI, web). deduplicates events within 2-second windows and shows result counts.
- **session diary** — quick note-taking input in the sidebar, timestamped entries.
- **inspector panel** — right sidebar that shows memory details, entity graphs, similar memories (with similarity percentages), importance factor breakdowns (colored bar chart with 7 weighted factors), annotations with add-note input, and access history.
- **toast notifications** — bottom-right toasts for all actions (promote, pin, copy, forget, dedup) with success/error/info styling and auto-dismiss.
- **keyboard shortcuts** — `/` focus search, `n` neural map, `s` search, `r` remember, `a` analytics, `c` cognition, `b` bridges, `Esc` close inspector.

### web API

the dashboard is backed by a full JSON API you can hit directly:

```
GET  /api/memories                    paginated list, optional ?layer= filter
GET  /api/memories/:id                full memory with hypothetical queries, entities, access history
GET  /api/memories/:id/similar        find similar memories by embedding distance
GET  /api/memories/:id/importance     7-factor importance score breakdown
GET  /api/search?q=...&debug=true     hybrid search with optional debug breakdown
GET  /api/search/filtered?q=...       search with layer, importance, date, source filters
GET  /api/entities                    all entities with memory counts
GET  /api/entities/:id/graph          entity relationship subgraph
GET  /api/entities/:id/timeline       entity memories ordered chronologically
GET  /api/neural                      full graph for neural visualization
GET  /api/neural/fires?since=...      recent access events (lightweight polling)
GET  /api/timeline?start=...&end=...  temporal query
GET  /api/analytics                   layer distribution, top accessed, top entities
GET  /api/health                      system health (cache, orphans, FTS, embeddings)
GET  /api/memory-map                  full system overview with per-layer top entities
GET  /api/context?query=...           L0-L3 graduated context with token counts
GET  /api/duplicates?threshold=0.92   preview near-duplicate memory pairs
GET  /api/stats                       system statistics
GET  /api/events                      recent events from all processes
GET  /api/diary                       session diary entries
GET  /api/ingest/log                  recent file ingestions
GET  /api/pulse                       hourly activity counters + sparkline
GET  /api/heatmap?days=30             github-style activity heatmap
GET  /api/memories/:id/history        importance score over time
GET  /api/retention/curves            L2/Huber/elastic curve data for chart
GET  /api/retention/scatter           real memory age vs retention scatter data
GET  /api/reranker/status             deep reranker training state
GET  /api/bridges                     cross-domain bridge memories
GET  /api/search/hints?q=...          truncated hints for cognitive scaffolding
GET  /api/skills?query=...            task-aware skill selection with confidence scoring
GET  /api/export?format=json          export memories as markdown or JSON
POST /api/remember                    store a memory (with surprise scoring)
POST /api/consolidate                 trigger dream cycle
POST /api/dedup                       auto-merge duplicate memories
POST /api/surprise/preview            compute surprise for text before storing
POST /api/reranker/train              trigger reranker training
POST /api/memories/:id/promote        change memory layer
POST /api/memories/:id/demote         demote to lower layer
POST /api/memories/:id/edit           edit content (re-embeds automatically)
POST /api/memories/:id/annotate       add timestamped note
POST /api/memories/:id/invalidate     mark as no longer true
POST /api/memories/:id/forget         soft-delete
POST /api/memories/:id/pin            pin (prevent forgetting)
POST /api/memories/:id/unpin          unpin
POST /api/memories/bulk               bulk promote/forget/tag/demote
POST /api/entities/:id/alias          add entity alias
POST /api/entities/:id/type           change entity type
POST /api/diary                       append diary entry
POST /api/ingest/path                 ingest a file or directory
POST /api/ingest/sessions             ingest recent Claude Code sessions
```

## architecture

everything lives in one sqlite file (`~/.local/share/engram/memory.db`). no external services.

```
engram/
├── store.py          # sqlite schema, CRUD, FTS5, entity graph (recursive CTEs), ANN lifecycle
├── ann_index.py      # HNSW approximate nearest neighbor index (hnswlib wrapper)
├── embeddings.py     # multi-backend embeddings (mlx, sentence-transformers, voyage, openai, gemini) + cross-encoder
├── retrieval.py      # 5-stage hybrid pipeline (HNSW dense + BM25 + graph → RRF → boost → rerank → deep)
├── extractor.py      # LLM fact extraction + hypothetical query generation
├── entities.py       # regex entity extraction, relationship graph, co-occurrence
├── surprise.py       # k-NN novelty scoring at write time (Titans-inspired surprise gate, ANN-accelerated)
├── deep_retrieval.py # learned MLP reranker trained on access patterns
├── skill_select.py   # task-aware skill selection gate (SkillsBench-inspired)
├── lifecycle.py      # retention regularization (L2/Huber/elastic), 7-factor importance, promotion
├── consolidator.py   # dream cycle (clustering, summarization, peer cards, archival)
├── codebase.py       # project scanner — file trees, signatures, deps → codebase layer
├── conversations.py  # claude code session ingest — exchange pairs, classification
├── dedup.py          # semantic deduplication — find and merge near-duplicates
├── layers.py         # L0-L3 graduated context retrieval
├── compress.py       # token-budget compression with entity codes
├── formats.py        # parsers for markdown, JSON chat exports, PDF, slack, email
├── llm.py            # claude CLI + mlx backend abstraction
├── evolution.py      # memory enrichment, evolution, CRUD classification, trust scoring, canonicalization, causal parents
├── drift.py          # memory drift detection — claim extraction, filesystem verification, drift scoring
├── patterns.py       # session pattern extraction — distill procedural knowledge from work
├── quantize.py       # lifecycle embedding compression (32/8/4/2-bit) with FRQAD distance metric
├── communities.py    # label propagation community detection + LLM summaries over entity graph
├── hopfield.py       # Hopfield associative retrieval channel — pattern completion via modern Hopfield network
├── mcp_server.py     # 63-tool MCP server (JSON-RPC, stdio) with working memory auto-sweep, ANN init
├── cli.py            # CLI interface
├── config.py         # yaml config with env var overrides
└── web/
    ├── app.py        # fastapi with model warmup on startup
    ├── routes.py     # 52 REST endpoints — search, analytics, surprise, retention, reranker, bridges, bulk, export
    ├── events.py     # SSE event stream (in-process)
    └── templates/
        └── index.html  # single-page dashboard with neural canvas, 14 panels, 74 JS functions
```

## supported formats

engram can ingest these file types:

| format | how it's handled |
|--------|-----------------|
| markdown (`.md`) | split by headers into sections |
| plaintext (`.txt`) | treated as single document |
| claude code (`.jsonl`) | parsed as conversation exchanges, grouped into Q+A pairs |
| claude.ai (`.json`) | parsed from chat_messages array |
| chatgpt (`.json`) | parsed from mapping tree structure |
| slack (`.json`) | parsed from messages array with user attribution |
| PDF (`.pdf`) | text extracted via pymupdf |
| generic JSON | each item or the whole object as a document |

conversation formats get special treatment — exchanges are grouped into Q+A pairs and classified (decisions, corrections, errors, task completions) before storing.

## what informed the design

i studied three existing memory systems and six IR papers before building this. took the best parts from each:

**systems:**
- [neuro-memory](https://github.com/raya-ac/neuro-memory) — my earlier memory system. atkinson-shiffrin 4-layer model, ebbinghaus forgetting curve, 7-factor importance scoring, procedural memory with pattern templates. engram takes the layer architecture and lifecycle model from here.
- [cmyui/ai-memory](https://github.com/cmyui/ai-memory) — LLM-extracted atomic facts, three-stage hybrid retrieval with RRF, dream cycle
- [mempalace](https://github.com/milla-jovovich/mempalace) — graduated layers (L0-L3), entity registry with disambiguation, exchange-pair chunking for conversations, AAAK compression

**papers:**
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) (Cormack et al. 2009) — the RRF formula and k=60 constant
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) (Hu et al. 2026) — forms/functions/dynamics taxonomy
- [docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery.pdf) (Nogueira & Lin 2019) — document expansion by query prediction
- [ColBERT-PRF](https://arxiv.org/abs/2106.11251) (Wang et al. 2021) — pseudo-relevance feedback for dense retrieval
- [BM25 Query Augmentation](https://arxiv.org/abs/2305.14087) (Chen & Wiseman 2023) — learned query expansion
- [Word Embedding GLM](https://dl.acm.org/doi/10.1145/2766462.2767780) (Ganguly et al. 2015) — embedding-based language model for IR
- [Titans](https://arxiv.org/abs/2501.00663) (Behrouz et al. 2025) — surprise-based memorization, memory updates proportional to loss gradient
- [Miras](https://arxiv.org/abs/2504.13173) (Behrouz et al. 2025) — unifying framework for sequence models, forgetting as retention regularization
- [Your Brain on ChatGPT](https://arxiv.org/abs/2506.08872) (Kosmyna et al. 2025) — cognitive scaffolding vs replacement, recall_hints design
- [SkillsBench](https://arxiv.org/abs/2602.12670) (Li et al. 2026) — focused skills (+16.2pp) beat comprehensive docs (-2.9pp), get_skills gate design
- [Evaluating AGENTS.md](https://arxiv.org/abs/2602.11988) (Gloaguen et al. 2026) — static context files reduce performance, validates dynamic retrieval over flat injection
- [A-Mem](https://arxiv.org/abs/2502.12110) (Wu et al. 2025) — zettelkasten memory with enriched embeddings and memory evolution, enrichment doubled multi-hop F1
- [AgeMem](https://arxiv.org/abs/2601.01885) (Chen et al. 2026) — RL-trained memory operations, quality_metrics reward decomposition
- [MAGMA](https://arxiv.org/abs/2601.03236) (Zhao et al. 2026) — multi-graph architecture with intent-aware adaptive retrieval policy
- [Zep/Graphiti](https://arxiv.org/abs/2501.13956) (Preston-Werner et al. 2025) — temporal knowledge graph with three-tier architecture
- [SuperLocalMemory V3.3](https://arxiv.org/abs/2604.04514) (2026) — trust-weighted decay, lifecycle embedding compression, confirmation count
- [Mem0](https://arxiv.org/abs/2504.19413) (Chhablani et al. 2025) — production write-path CRUD classification, temporal marked deletion
- [Memory for Autonomous Agents](https://arxiv.org/abs/2603.07670) (2026) — latest comprehensive survey, adversarial belief probing, write-path canonicalization
- [Mem^p](https://arxiv.org/abs/2508.06433) (2025) — procedural memory with dual representation and reflection-based updates
- [ACT-R Memory](https://dl.acm.org/doi/10.1145/3765766.3765803) (HAI 2025) — base-level activation, retrieval noise, threshold gating

## config

lives at `config.yaml` or `~/.config/engram/config.yaml`. env vars override everything (prefix with `ENGRAM_`, e.g. `ENGRAM_DB_PATH`).

```yaml
db_path: ~/.local/share/engram/memory.db
embedding_model: BAAI/bge-small-en-v1.5   # or: voyage-3.5, text-embedding-3-small, gemini-embedding-001
cross_encoder_model: cross-encoder/ms-marco-MiniLM-L-6-v2
embedding_backend: auto                   # auto | mlx | sentence_transformers | voyage | openai | gemini
embedding_dim: 384                        # auto-detected from model name if known

retrieval:
  top_k: 10
  rrf_k: 60
  min_confidence: 0.60
  rerank_candidates: 20
  dense_multiplier: 3          # candidates = top_k * multiplier
  bm25_multiplier: 3

lifecycle:
  forgetting_half_life_days: 30
  archive_after_days: 90
  archive_min_importance: 0.3  # below this + age + low access → forget
  archive_min_accesses: 3
  promote_importance: 0.7
  promote_accesses: 5
  cluster_threshold: 0.8
  cluster_min_size: 5
  retention_mode: huber        # l2 | huber | elastic
  huber_delta: 0.5             # transition point for huber (in half-lives)
  elastic_l1_ratio: 0.3        # L1 weight for elastic (0=pure L2, 1=pure L1)

llm:
  backend: claude_cli          # claude_cli | mlx | llamacpp
  model: claude-sonnet-4-20250514
  mlx_model: mlx-community/Qwen2.5-3B-Instruct-4bit

web:
  host: 127.0.0.1
  port: 8420

ann:
  enabled: true
  m: 32                        # HNSW graph connectivity (higher = better recall, more memory)
  ef_construction: 200         # build-time search depth (higher = better index quality)
  ef_search: 100               # query-time search depth (higher = better recall, slower)
  max_elements: 500000         # pre-allocated capacity
  index_path: ~/.local/share/engram/hnsw.index
```

## license

MIT
