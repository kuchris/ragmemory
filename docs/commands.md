# RagMemory Commands

Useful copy-paste commands from the repo root.

## Setup

```powershell
uv venv
uv pip install -e .
Copy-Item .\ragmemory.example.ini .\ragmemory.local.ini
```

Edit `ragmemory.local.ini` before running LLM-backed commands.

## Daily Use

Start the worker:

```powershell
uv run python scripts/run_worker.py
```

Or double-click:

```text
start_worker.bat
```

Export the Obsidian mirror:

```powershell
uv run python scripts/export_obsidian.py --db-path ./.data/chroma_db --output ./.data/obsidian_memory --config ragmemory.local.ini
```

Check the Obsidian graph:

```powershell
uv run python scripts/check_obsidian_graph.py
```

Open this folder in Obsidian:

```text
.data/obsidian_memory
```

## Topic Groups

Regroup topics, then refresh Obsidian:

```powershell
uv run python scripts/regroup_topics.py --run --db-path ./.data/chroma_db
uv run python scripts/export_obsidian.py --db-path ./.data/chroma_db --output ./.data/obsidian_memory --config ragmemory.local.ini
```

Warning: regroup currently rewrites `.data/chroma_db/topic_taxonomy.json`, so old
topic groups can be replaced by the new LLM result.

Disable regroup if you do not want topic groups to change:

```ini
[topic_regroup]
enable = false
```

Require at least 10 groups on the next regroup:

```ini
[topic_regroup]
min_groups = 10
```

## Non-LLM Wiki

Generate wiki pages from the current Obsidian graph. This does not call an LLM:

```powershell
uv run python scripts/generate_wiki.py --obsidian ./.data/obsidian_memory
```

Add cached LLM summaries. Start with one page to control token use:

```powershell
uv run python scripts/generate_wiki.py --obsidian ./.data/obsidian_memory --config ragmemory.local.ini --llm --llm-limit 1
```

Generate or refresh all uncached summaries:

```powershell
uv run python scripts/generate_wiki.py --obsidian ./.data/obsidian_memory --config ragmemory.local.ini --llm
```

Open:

```text
.data/obsidian_memory/wiki/index.md
```

## Inspect Memory

Recent events:

```powershell
uv run python scripts/inspect_events.py --db-path ./.data/chroma_db --limit 20
```

Structured objects added:

```powershell
uv run python scripts/inspect_events.py --event structured_object_added
```

Ask retrieval what it would recall:

```powershell
uv run python scripts/ask_memory.py "your test question"
```

View chunks:

```powershell
uv run python scripts/view_chunks.py --db-path ./.data/chroma_db --limit 20
```

Decay report:

```powershell
uv run python scripts/decay_report.py --db-path ./.data/chroma_db --output decay_report.txt
```

## Compaction And Backfill

Compact old messages manually:

```powershell
uv run python scripts/compact_backfill.py --limit 20
```

Backfill structured extraction for old messages:

```powershell
uv run python scripts/structured_backfill.py --limit 20 --queue
uv run python scripts/run_worker.py --once
```

Rebuild retrieval indexes after changing embeddings or compaction:

```powershell
uv run python scripts/rebuild_memory_index.py --db-path ./.data/chroma_db
```

## Provider Tests

Smoke-test the configured provider:

```powershell
uv run python scripts/test_llm_provider.py --provider opencode_go
```

Check OpenCode reasoning behavior:

```powershell
uv run python scripts/test_opencode_reasoning.py
```

## Remove Bad Memory

Preview recent records:

```powershell
uv run python scripts/remove_memory.py --recent 20
```

Search for a bad memory:

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

## Benchmark

Run the small retrieval benchmark:

```powershell
uv run python scripts/benchmark_retrieval.py
```

Generate local benchmark cases:

```powershell
uv run python scripts/make_benchmark_cases.py --db-path ./.data/chroma_db --output ./.data/bench_retrieval/real_cases.json --limit 30
```

## Graph GIF

Obsidian is better for real inspection, but this makes an optional GIF:

```powershell
uv run python scripts/animate_obsidian_graph.py --obsidian ./.data/obsidian_memory --output ./.data/graph_animation/ragmemory-map-formed.gif
```

Smaller GIF:

```powershell
uv run python scripts/animate_obsidian_graph.py --obsidian ./.data/obsidian_memory --output ./.data/graph_animation/ragmemory-map-small.gif --exclude-messages --max-nodes 900
```

## Backup

Back up the active DB:

```powershell
Copy-Item -Recurse .\.data\chroma_db .\.data\backup-chroma_db
```

## Tests

Run focused tests:

```powershell
uv run python tests/test_export_obsidian.py
uv run python tests/test_topic_regroup.py
uv run python tests/test_topic_regroup_interval.py
```

Compile-check changed scripts:

```powershell
uv run python -m py_compile scripts/export_obsidian.py scripts/regroup_topics.py scripts/animate_obsidian_graph.py
```
