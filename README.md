# RagMemory

RagMemory gives Codex a local memory that survives across turns.

It stores your conversation locally, recalls useful context before a new prompt,
and keeps a generated Obsidian mirror so you can inspect what was remembered.

![Obsidian graph view](obsidian.png)

## What It Does

- Remembers useful chat context across Codex sessions.
- Recalls relevant memory automatically through Codex hooks.
- Saves assistant replies after each turn.
- Extracts durable structured memory such as decisions, preferences, configs,
  code references, and open questions.
- Exports a readable Obsidian vault under `.data/obsidian_memory`.
- Uses decay-aware retrieval so stale memories naturally appear less often.
- Supports manual tombstone removal for wrong, private, or harmful memory.

Raw memory is not deleted during normal forgetting. Forgetting means old,
unused memories rank lower during retrieval.

## Recommended Setup

From the repo root:

```powershell
uv venv
uv pip install -e .
```

Create your local settings file:

```powershell
Copy-Item .\ragmemory.example.ini .\ragmemory.local.ini
```

Edit `ragmemory.local.ini` and add your API key:

```ini
[structured_memory]
api_key = your-nvidia-api-key
model = minimaxai/minimax-m2.7
max_chars = 6000
max_tokens = 900

[llm]
structured_provider = nvidia
compact_provider = nvidia

[llm.nvidia]
api_key = your-nvidia-api-key
base_url = https://integrate.api.nvidia.com/v1
model = minimaxai/minimax-m2.7
api_style = openai_chat

[embedding]
provider = chroma_default

[compact]
enable = true
model = minimaxai/minimax-m2.7
min_chars = 1500
max_chars = 30000
max_tokens = 1200
target_ratio = 0.35
mode = background
```

`ragmemory.local.ini` is ignored by git. Do not commit it.

### Optional: OpenCode Go For Compaction

RagMemory can use different LLM providers for structured extraction and message
compaction. Keep structured extraction on NVIDIA first, and try OpenCode Go for
compaction:

```ini
[llm]
structured_provider = nvidia
compact_provider = opencode_go

[llm.opencode_go]
api_key = your-opencode-go-api-key
base_url = https://opencode.ai/zen/go/v1
model = deepseek-v4-flash
api_style = openai_chat
thinking = disabled

[compact]
enable = true
max_tokens = 4096
```

Smoke-test a provider without writing to the DB:

```powershell
uv run python scripts/test_llm_provider.py --provider opencode_go
```

Only `openai_chat` providers are supported for now. OpenCode Go models that use
`/messages` need a separate adapter later.

### Optional: Better Local Embeddings

RagMemory uses local embeddings for Chroma vector search, then combines those
results with BM25 keyword search. The default is still Chroma's built-in
embedding function because it is the easiest clean install.

For better recall on a local Windows machine, try the small BGE model:

```ini
[embedding]
provider = sentence_transformers
model = BAAI/bge-small-en-v1.5
device = cpu
normalize_embeddings = true
```

Model tradeoffs:

```text
all-MiniLM-L6-v2        fastest and lightest
BAAI/bge-small-en-v1.5  better recall, still CPU-friendly
BAAI/bge-m3             better multilingual recall, too heavy for many CPUs
```

After changing the embedding model, rebuild both chat and structured indexes:

```powershell
uv run python scripts/rebuild_memory_index.py --db-path ./.data/chroma_db
```

RagMemory uses model-specific Chroma collection names for non-default embedding
models, so old vectors are not mixed with new vectors.

## Use With Codex Hooks

Install the Codex hooks from:

```text
ragm_mcp/hooks/README.md
```

After the hooks are installed:

- `UserPromptSubmit` recalls memory and injects it into Codex.
- `UserPromptSubmit` saves your user prompt.
- `Stop` saves the assistant response.
- `Stop` refreshes the Obsidian mirror.
- Long-running structured extraction and compaction jobs are handled by the
  manual worker: `uv run python scripts/run_worker.py`.

On Windows you can also start the worker by double-clicking:

```text
start_worker.bat
```

Keep the worker running while chatting. Stop it before a large manual
`compact_backfill.py` run if you want to avoid duplicate API calls or NVIDIA
rate limits.

You should see injected context like this at the start of a turn:

```text
=== RagMemory Context ===
...
=== End RagMemory Context ===
```

If this context takes too many tokens, tune the hook recall size in
`ragmemory.local.ini`:

```ini
[recall]
context_token_budget = 900
retrieve_top_k = 3
structured_top_k = 2
recent_messages = 4
include_recent = true
include_structured = true
```

## MCP Tools

MCP is optional when hooks are installed.

Recommended local MCP settings:

```ini
[mcp.tools]
enable_recall = false
enable_save = false
enable_tombstone = true
```

With this split:

- Hooks own automatic recall and save.
- MCP recall is disabled to avoid duplicate token usage.
- MCP save is disabled to avoid duplicate writes.
- MCP remains useful for `memory_stats`, `remove_memory_preview`, and
  `remove_memory_confirm`.

See the MCP details in:

```text
ragm_mcp/README.md
```

## Check What Was Remembered

Inspect recent events:

```powershell
uv run python scripts/inspect_events.py --db-path ./.data/chroma_db --limit 20
```

Inspect structured-object add events:

```powershell
uv run python scripts/inspect_events.py --event structured_object_added
```

Check that the Obsidian graph export stays clean:

```powershell
uv run python scripts/check_obsidian_graph.py
```

Create an animated GIF that shows the map forming over time:

```powershell
uv run python scripts/animate_obsidian_graph.py --obsidian ./.data/obsidian_memory --output ./.data/graph_animation/ragmemory-map-formed.gif
```

By default this renders every memory graph node, including raw message notes.
For every Markdown note Obsidian can see, add `--include-navigation`. For a
smaller explainer GIF, add `--exclude-messages --max-nodes 900`.

If the graph shows isolated message dots, hide raw message notes that have no
structured links with this Obsidian graph filter:

```text
-["cssclasses":"memory-unlinked"]
```

File/path hubs are disabled by default because they can clutter the graph. Turn
them on only if you want file-level nodes:

```ini
[obsidian.files]
enable = true
```

Export the Obsidian mirror manually:

```powershell
uv run python scripts/export_obsidian.py --db-path ./.data/chroma_db --output ./.data/obsidian_memory
```

Open this folder in Obsidian:

```text
.data/obsidian_memory
```

### Optional: Topic Groups

The normal Obsidian export keeps leaf topic hubs under `topics/`. If the graph
gets too dense, run the topic regroup step to add an upper layer under
`topic_groups/`:

```powershell
uv run python scripts/regroup_topics.py --run --db-path ./.data/chroma_db
uv run python scripts/export_obsidian.py --db-path ./.data/chroma_db --output ./.data/obsidian_memory --config ragmemory.local.ini
```

This writes `.data/chroma_db/topic_taxonomy.json`. The regroup step does not
delete or rewrite the existing leaf topics; it asks the LLM to create group
notes that link to related leaf topics.

Configure the token budget in `ragmemory.local.ini`:

```ini
[topic_regroup]
enable = true
max_input_topics = 150
min_groups = 10
max_tokens = 6000
thinking = disabled
```

`max_input_topics` limits how many leaf-topic summaries are sent to the LLM for
grouping. Topics outside that limit stay in `topics/`; they are just ungrouped
for that run.

To queue the same work for the worker instead of running it immediately:

```powershell
uv run python scripts/regroup_topics.py --queue --db-path ./.data/chroma_db
uv run python scripts/run_worker.py --once --db-path ./.data/chroma_db
```

Generate non-LLM wiki pages from the current graph:

```powershell
uv run python scripts/generate_wiki.py --obsidian ./.data/obsidian_memory
```

Add cached LLM summaries one page at a time:

```powershell
uv run python scripts/generate_wiki.py --obsidian ./.data/obsidian_memory --config ragmemory.local.ini --llm --llm-limit 1
```

Compacted messages may contain evidence references like:

```text
evidence[text:874b3468c9f7]
```

These are deterministic pointers to exact structured evidence blocks. In the
Obsidian mirror, message notes include an `Evidence References` section that
links each marker to the matching structured note. Structured notes also expose
`content_hash` and `evidence_ref` in frontmatter so the hash is searchable.

## Remove Bad Memory

Use removal only for memory that is wrong, private, harmful, or should not be
used again.

Preview recent records:

```powershell
uv run python scripts/remove_memory.py --recent 20
```

Search for a bad record:

```powershell
uv run python scripts/remove_memory.py --search "wrong remembered detail"
```

Preview specific message IDs:

```powershell
uv run python scripts/remove_memory.py --message-ids 12,13
```

Confirm tombstone removal:

```powershell
uv run python scripts/remove_memory.py --message-ids 12,13 --confirm
```

This is tombstone-only. It hides records from retrieval and moves them to
`forgotten/` in the Obsidian mirror. It does not hard-delete raw storage.

## Backup

Back up the active DB folder:

```powershell
Copy-Item -Recurse .\.data\chroma_db .\.data\backup-chroma_db
```

The DB folder contains:

```text
state.sqlite
chroma.sqlite3
structured_memory.jsonl
ledger.json
events.jsonl
```

## Message Compaction

Raw messages stay as the source of truth. Compaction writes a smaller
`compact_text` beside the raw message in SQLite:

```text
messages.text          raw audit log
messages.compact_text  compact retrieval/export text
messages.compact_status ok | failed | skipped_short | too_long
```

Retrieval and Obsidian prefer `compact_text` only when
`compact_status = ok`; otherwise they fall back to raw text.

Run the worker for new messages:

```powershell
uv run python scripts/run_worker.py
```

Backfill old messages manually:

```powershell
uv run python scripts/compact_backfill.py --limit 20
```

If NVIDIA returns `429 Too Many Requests`, wait a minute and retry with a
smaller limit. Stop the worker during large backfills to reduce overlap.

Reasoning-heavy providers may spend output tokens on internal reasoning before
returning final compact text. For those providers, raise the compact output
limit:

```ini
[compact]
max_tokens = 4096
```

After changing compaction behavior, rebuild the retrieval index:

```powershell
uv run python scripts/rebuild_memory_index.py
```

## Benchmark Retrieval

Run the small tracked smoke benchmark:

```powershell
uv run python scripts/benchmark_retrieval.py
```

For a more realistic local benchmark, generate cases from your current
structured memory. This writes under `.data/`, so private project details stay
out of git:

```powershell
uv run python scripts/make_benchmark_cases.py --db-path ./.data/chroma_db --output ./.data/bench_retrieval/real_cases.json --limit 30
```

Compare embedding models against the same local cases:

```powershell
uv run python scripts/benchmark_retrieval.py --cases ./.data/bench_retrieval/real_cases.json --source-db ./.data/chroma_db --embedding-provider sentence_transformers --embedding-model BAAI/bge-small-en-v1.5 --bench-db ./.data/bench_retrieval/real_bge_small

uv run python scripts/benchmark_retrieval.py --cases ./.data/bench_retrieval/real_cases.json --source-db ./.data/chroma_db --embedding-provider sentence_transformers --embedding-model all-MiniLM-L6-v2 --bench-db ./.data/bench_retrieval/real_minilm

uv run python scripts/benchmark_retrieval.py --cases ./.data/bench_retrieval/real_cases.json --source-db ./.data/chroma_db --embedding-provider chroma_default --embedding-model='' --bench-db ./.data/bench_retrieval/real_chroma_default
```

The local benchmark reports `recall@5`, `recall@10`, MRR, rebuild time, and
query latency. Treat generated cases as a starting point; hand-written
paraphrase cases are still better for final model decisions.

Latest local run on 30 generated real-memory cases:

```text
model                    recall@5  recall@10  MRR    p50 latency
BAAI/bge-small-en-v1.5   96.7%     96.7%      0.894  85.9 ms
all-MiniLM-L6-v2         90.0%     93.3%      0.818  62.8 ms
chroma_default           90.0%     93.3%      0.818  582.3 ms
```

This result is machine- and corpus-specific. Re-run the benchmark after major
memory, embedding, chunking, or retrieval changes.

## Technical Details

The implementation details, architecture, scripts, and tests live in:

```text
docs/technical.md
```

For a copy-paste command cheat sheet, see:

```text
docs/commands.md
```
