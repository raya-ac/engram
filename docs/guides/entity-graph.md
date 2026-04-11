# Entity Graph

every memory gets scanned for entities — people, tools, projects, dates, URLs, file paths. these form a relationship graph that powers graph-based retrieval.

## how entities are extracted

regex patterns match:

- **people**: capitalized names, @mentions
- **tools/projects**: known tool names, project references
- **dates**: ISO dates, natural language dates
- **URLs**: http/https links
- **file paths**: `/path/to/file` patterns

entities get canonical names, aliases, and types (person, tool, project, concept, date).

## relationships

relationships form through:

- **co-occurrence**: entities mentioned in the same sentence get a `CO_OCCURS` link
- **pattern matching**: "X uses Y" → `USES`, "X built Y" → `CREATED`, etc.
- **strength**: increases with evidence count (more co-occurrences = stronger link)
- **temporal validity**: relationships can have `valid_from` and `valid_to` timestamps

## traversal

multi-hop queries use recursive SQL CTEs:

```sql
-- "show me everything connected to Ari within 2 hops"
-- runs in a single SQL query, no graph database needed
```

MCP tools:

- `recall_entity(name)` — everything about an entity
- `recall_related(name, max_hops=2)` — graph traversal
- `entity_graph(name)` — relationship subgraph as JSON
- `entity_timeline(name)` — chronological story

## management

- `merge_entities(source, target)` — combine duplicates, moves all links
- `update_entity(name, alias)` — add aliases
- `search_entities(query)` — fuzzy search by partial name
- `link_memories(id1, id2)` — manually relate two memories
- `backlinks(memory_id)` — find all memories linked via shared entities

## community detection

`detect_communities` runs label propagation over the entity graph to discover clusters. optionally generates LLM summaries for each community.

## graph in retrieval

the entity graph is the 3rd retrieval channel. when you search, engram:

1. extracts entity names from your query
2. finds matching entities
3. retrieves their memories (hop 0, score 1.0)
4. traverses 1-hop related entities (score 0.5 * strength)
5. feeds candidates into RRF fusion

for "who" queries, the graph channel gets 1.8x weight boost via intent classification.
