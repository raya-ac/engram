# Memory Lifecycle

memories aren't static. they move between layers, gain or lose importance, and eventually get archived or forgotten.

## layers

| layer | purpose | half-life | example |
|-------|---------|-----------|---------|
| working | ephemeral, auto-promotes after 30 min | — | current task context |
| episodic | events, experiences | 30 days | "deployed v2.1 on march 28" |
| semantic | permanent knowledge | ∞ | "the API uses JWT tokens" |
| procedural | decisions, patterns, how-to | ∞ | "always use real DB for integration tests" |
| codebase | compressed code knowledge | ∞ | file trees, function signatures |

## importance scoring (9 factors)

| factor | weight | description |
|--------|--------|-------------|
| base importance | set at creation | 0.0-1.0, adjusted by surprise |
| access frequency | log scale | how often recalled |
| recency | exponential decay | trust-weighted half-life |
| emotional valence | moderate | strong emotions = more memorable |
| stability | moderate | consistent access vs burst |
| layer boost | moderate | semantic weighted higher |
| source trust | moderate | human=1.0, AI=0.7, ingest=0.5 |
| confirmation count | moderate | independently corroborated facts |
| ACT-R activation | moderate | base-level activation from cognitive architecture |

## surprise-based importance

at write time, every new memory is compared against existing embeddings using k-NN (k=5):

- **novel** (far from stored) → importance boosted up to +0.3
- **redundant** (close to stored) → importance reduced, flagged as duplicate
- surprise score stored in metadata for auditing

```
surprise=0.85 → novel content, high importance
surprise=0.15 → near-duplicate, low importance
```

## retention regularization

three modes (configurable via `retention_mode`):

- **l2** (classic Ebbinghaus): smooth exponential decay, 50% at half-life
- **huber** (default): matches L2 near-term, transitions to linear for old memories. robust to burst-then-quiet patterns
- **elastic** (L1+L2): sparse retention. strongly-held memories stay, weak ones fade faster

all modes include access reinforcement — each recall strengthens retention (spaced repetition, log-scaled, capped at +0.3).

## trust-weighted decay

different sources decay at different rates:

```
λ_eff = λ · (1 + κ·(1 - trust))    where κ=2.0
```

- human-authored: full 30-day half-life
- auto-extracted: 3x faster decay

## promotion and demotion

- **auto-promotion**: episodic → semantic when importance ≥ 0.7 and access_count ≥ 5
- **working → episodic**: auto after 30 minutes or 2 accesses
- **manual**: `promote` / `demote` MCP tools or web UI buttons

## pinning

pin any memory to make it immune to the dream cycle's forgetting pass:

```
pin(memory_id)
unpin(memory_id)
```

useful for memories that are important but rarely accessed.

## dream cycle (consolidation)

7-step pipeline that runs on `consolidate`:

1. apply forgetting curve with trust-weighted retention
2. cluster similar memories (cosine > 0.8), merge clusters of 5+
3. generate peer cards for entities with enough data
4. cross-domain synthesis — find entity pairs in different contexts, LLM-confirm bridges
5. adversarial belief probing — challenge old beliefs, reduce importance on invalidated ones
6. drift detection — validate claims against filesystem, auto-invalidate dead refs
7. prune old access logs and events

## archival

after 90 days, if retention < 0.15, importance < 0.3, and access_count < 3 → soft-delete. semantic, procedural, and pinned memories are exempt.

## embedding compression

as memories age, their embeddings can be quantized:

| retention | precision | compression | cosine fidelity |
|-----------|-----------|-------------|-----------------|
| active (R>0.8) | 32-bit float | 1x | 1.000 |
| warm | 8-bit int | 3.9x | 0.9999 |
| cold | 4-bit int | 7.6x | 0.97 |
| archive | 2-bit int | 14.6x | 0.59 |

uses FRQAD (Fisher-Rao Quantization-Aware Distance) for mixed-precision comparison.
