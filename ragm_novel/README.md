# ragm_novel

Novel RAG pipeline with graph-augmented retrieval. Chunks a plain-text novel into ChromaDB, builds a graph of sequential and character-relationship edges, then uses graph traversal to expand retrieval context at query time.

Run all commands from the **repo root** (`ragmemory/`).

---

## Pipeline

### 1. Ingest

```bash
uv run python ragm_novel/ingest_novel.py novels/my_novel.txt --title "My Novel"
```

Splits the novel into ~900-char chunks by chapter, embeds them into ChromaDB (`./chroma_novel`), and writes sequential `next`/`prev` edges to `chroma_novel/graph_edges.jsonl`.

| Flag | Default | Description |
|------|---------|-------------|
| `--db-path` | `./chroma_novel` | ChromaDB output path |
| `--title` | filename stem | Novel title stored in metadata |
| `--target-chars` | `900` | Target chunk size |
| `--overlap-chars` | `120` | Overlap between chunks |
| `--reset-db` | off | Wipe DB before ingesting |
| `--dry-run` | off | Parse only, no writes |

### 2. Build character edges

Calls LM Studio to extract character names from each chunk, then connects chunks that share the same character.

```bash
# Fast version (batched + async)
uv run python ragm_novel/build_character_edges_fast.py

# Options
uv run python ragm_novel/build_character_edges_fast.py --batch-size 10 --concurrency 2
uv run python ragm_novel/build_character_edges_fast.py --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | `8` | Chunks per LLM call. Sweet spot: 8–12 |
| `--concurrency` | `2` | In-flight requests. >3 gives no benefit on local GPU |
| `--top-k` | `5` | Nearest same-character chunks to connect per chunk |
| `--dry-run` | off | Extract + cache only, no edge writes |

Extraction results are cached in `chroma_novel/character_cache.json`. Re-runs skip already-processed chunks.

**Character normalization** — aliases are merged via `ALIAS_MAP` in the script (e.g. `清隆` → `綾小路`). Edit the map to add novel-specific aliases before running.

### 3. Normalize existing edges (optional)

If edges were written with un-normalized names, fix them in-place without re-running LLM:

```bash
uv run python ragm_novel/normalize_character_edges.py --dry-run  # preview
uv run python ragm_novel/normalize_character_edges.py            # apply
```

### 4. Chat

```bash
uv run python ragm_novel/chat_novel.py
```

Requires LM Studio running at `http://localhost:1234`.

At query time:
1. Vector search retrieves top-5 chunks
2. For each hit, `StoryGraph.around()` expands to ±3 sequential neighbors + 2 character-edge neighbors
3. All chunks are passed as context to the LLM

| Env var | Default | Description |
|---------|---------|-------------|
| `LMSTUDIO_MODEL` | `gemma-4-e4b-...` | Model name in LM Studio |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API URL |
| `LMSTUDIO_MAX_TOKENS` | `10000` | Max tokens per response |
| `NOVEL_DB_PATH` | `./chroma_novel` | Novel ChromaDB path |
| `NOVEL_CHAT_DB_PATH` | `./chroma_novel_chat` | Chat history DB path |

---

## Graph structure

All edges live in `chroma_novel/graph_edges.jsonl`. Each line is one directed edge:

```json
{"source": "<chunk_id>", "target": "<chunk_id>", "type": "next|prev|character", "weight": 1.0, "character": "堀北", "source_title": "...", "chapter": "..."}
```

| Edge type | Created by | Connects |
|-----------|-----------|---------|
| `next` / `prev` | `ingest_novel.py` | Adjacent chunks within the same chapter |
| `character` | `build_character_edges_fast.py` | Chunks sharing the same character name |

---

## Visualization

```bash
# Chapter/chunk bar chart
uv run python ragm_novel/visualize_graph.py
open ragm_novel/graph_visualization.html

# Character co-occurrence network
uv run python ragm_novel/visualize_characters.py
open ragm_novel/character_visualization.html
```

---

## Utilities

| Script | Purpose |
|--------|---------|
| `analyze_cache.py` | Print character name frequencies from cache |
| `restore_cache.py` | Reconstruct cache from existing graph edges (lossy) |
