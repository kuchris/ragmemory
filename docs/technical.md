# RagMemory Technical Notes

This document keeps implementation details out of the user-facing README.

## Core Idea

RagMemory stores memory in layers:

| Layer | Purpose |
|------|---------|
| Raw log | Source-of-truth messages in `state.sqlite`. |
| Raw chunks | Searchable Chroma chunks for broad recall. |
| Structured memory | High-signal objects such as decisions, constraints, configs, tables, and open questions. |
| Recent window | Latest messages included directly for short-term continuity. |
| Removal ledger | Retrieved chunks that were dropped because the context budget was full. |

The key rule:

```text
raw chunks preserve recall
structured memory preserves meaning
exact artifacts preserve usable source
```

## Save Pipeline

When saving a message:

```text
message
  -> append raw log
  -> split into chunks
  -> embed chunks in Chroma
  -> extract exact artifacts by code
  -> optionally extract structured memory with NVIDIA API
  -> store structured objects in JSONL + Chroma
  -> log structured_object_added events
```

Structured extraction can run immediately or be queued for background draining.
The Codex Stop hook saves assistant messages, drains up to 3 pending extraction
jobs, and refreshes the Obsidian export.

## Recall Pipeline

When answering a question:

```text
question
  -> retrieve structured memory
  -> retrieve raw chunk candidates with embedding search + BM25
  -> rerank raw chunks by relevance x decay strength
  -> add recent messages
  -> drop overflow chunks into ledger.json
  -> mark injected raw memories as accessed
  -> send context to the model
```

Default retrieval constants live in `src/ragmemory/memory.py`, but the hook path
can override the main recall limits from `ragmemory.local.ini`:

```ini
[recall]
context_token_budget = 900
retrieve_top_k = 3
structured_top_k = 2
recent_messages = 4
include_recent = true
include_structured = true
```

| Constant | Meaning |
|----------|---------|
| `CONTEXT_TOKEN_BUDGET` | Approximate context budget for memory injection. |
| `STRUCTURED_TOP_K` | Number of structured objects retrieved. |
| `RECENT_MESSAGES` | Number of recent raw messages included. |
| `RETRIEVE_TOP_K` | Number of raw chunks kept after retrieval. |

## Forgetting Model

Automatic forgetting is retrieval decay, not deletion.

Raw messages stay in SQLite. Chroma stays a content index. `memory_metadata`
tracks:

- `last_accessed_at`
- `access_count`
- `base_importance`
- `half_life_days`
- `pinned`
- `superseded_by`
- `tombstoned_at`

Old unused memories rank lower over time. Memories injected into context get
their access timestamp refreshed.

Manual removal is separate. It tombstones wrong, private, or harmful records so
they no longer appear in retrieval.

## Structured Memory

Structured objects are stored in:

```text
<db_path>/structured_memory.jsonl
```

Supported object types:

```text
decision
preference
constraint
config
table
code_reference
chart
open_question
identity
```

Exact fenced configs, tables, Mermaid diagrams, and code blocks are detected by
deterministic code when possible. The LLM extractor is for semantic objects such
as decisions, preferences, constraints, and open questions.

Structured object additions are logged to `events.jsonl` as
`structured_object_added` with:

- structured object ID
- type
- tags
- message ID
- role
- importance
- `job_id` when created by background extraction

This event trail is intended for future compaction dry-runs.

## Obsidian Export

The generated vault is a one-way mirror:

```text
.data/obsidian_memory/
```

Main folders:

| Folder | Meaning |
|--------|---------|
| `active/messages/` | Active raw message notes. |
| `active/structured/` | Active structured memory notes. |
| `forgotten/` | Tombstoned records. |
| `topics/` | Generated semantic topic hubs. |
| `files/` | Optional generated file/path hubs from explicit source paths. |
| `profile/` | Generated profile/user hub for preference and identity memory. |
| `maps/` | Timeline and turn navigation notes. |

Chronology is stored as frontmatter, not body wikilinks. Message bodies link to
structured memory objects; structured objects link to semantic hubs.

Timeline and turns pages are marked with:

```text
cssclasses: ["navigation"]
```

Raw message notes with no structured object links are marked with:

```text
cssclasses: ["memory-message", "memory-unlinked"]
```

Useful Obsidian graph filter:

```text
-["cssclasses":"navigation"] -["cssclasses":"memory-unlinked"]
```

Topic hubs are filtered at export time by `[obsidian.topics]` in
`ragmemory.local.ini`:

- denylisted artifact/type/language tags are hidden
- allowlisted tags are always shown
- other tags must recur at least `min_count` times

Raw structured-memory tags stay intact.

File hubs are opt-in. They are disabled by default because path nodes can clutter
the graph:

```ini
[obsidian.files]
enable = false
```

## MCP And Hooks

Hooks are the recommended path for Codex:

- `UserPromptSubmit` calls `build_recall_context(...)` directly.
- `UserPromptSubmit` saves the user message.
- `Stop` saves the assistant message.
- `Stop` drains pending structured extraction jobs.
- `Stop` exports the Obsidian mirror.

MCP is optional. With hooks installed, recommended flags are:

```ini
[mcp.tools]
enable_recall = false
enable_save = false
enable_tombstone = true
```

That keeps MCP recall/save from duplicating hook work while preserving manual
inspection/removal tools.

## Important Scripts

| File | Purpose |
|------|---------|
| `scripts/chat.py` | CLI chat using NVIDIA API and RagMemory context. |
| `scripts/ask_memory.py` | Prints retrieved context for preset questions. |
| `scripts/view_chunks.py` | Shows raw Chroma chunks. |
| `scripts/inspect_events.py` | Filters the JSONL event log. |
| `scripts/export_obsidian.py` | Exports the generated Obsidian mirror. |
| `scripts/check_obsidian_graph.py` | Checks topic count, denylisted hubs, and phantom wikilinks. |
| `scripts/remove_memory.py` | Previews/confirms tombstones for wrong, private, or harmful memory. |

Common commands:

```powershell
uv run python scripts/inspect_events.py --event structured_object_added
uv run python scripts/check_obsidian_graph.py
uv run python scripts/remove_memory.py --recent 20
```

## Tests

Tests are script-style. Run targeted files directly:

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python tests/test_event_log.py
uv run python tests/test_background_extraction.py
uv run python tests/test_ragm_mcp_forget_tools.py
```

Useful test files:

| File | Purpose |
|------|---------|
| `tests/test_memory.py` | Basic raw retrieval behavior. |
| `tests/test_bm25_lazy_rebuild.py` | BM25 lazy rebuild behavior. |
| `tests/test_dedup.py` | Normalized duplicate-message skip. |
| `tests/test_sqlite_state.py` | SQLite state and concurrent writer behavior. |
| `tests/test_background_extraction.py` | Background structured extraction. |
| `tests/test_event_log.py` | JSONL event logging. |
| `tests/test_export_obsidian.py` | Obsidian mirror export. |
| `tests/test_ragm_mcp_root_adapter.py` | MCP/hooks root adapter. |
| `tests/test_ragm_mcp_forget_tools.py` | MCP remove/tombstone wrapper. |
| `tests/test_ragm_mcp_obsidian_export.py` | MCP-triggered Obsidian export. |
| `tests/test_decay_forgetting.py` | Decay-aware ranking and access touch. |

`tests/test_structured_memory.py` uses the configured NVIDIA-backed extractor.

## Future Compaction Plan

Compaction is intentionally not implemented yet.

The next safe step, when real pain appears, is a read-only signal checker or
`scripts/compact.py --dry-run`. It should report candidates without rewriting
memory.

Concrete triggers:

- `topics/` grows above roughly 30 hubs despite topic filtering.
- `structured_object_added` events show repeated objects on the same
  `message_id`.
- At least 3 tag spellings normalize to the same concept.
- At least 5 message IDs have multiple structured objects whose tags overlap by
  more than 50%.
- Retrieval returns several results that are obviously the same memory.

Tier 1 dry-run checks:

- tag variant detection
- exact duplicate structured objects
- likely duplicate structured objects from repeated background extraction jobs
- same `message_id`, different `job_id`, same `type`, overlapping normalized
  tags

Do not add automatic LLM "sleep mode" that rewrites memory. If LLM assistance is
added later, use it only as a bounded judge for specific duplicate,
normalization, or supersession decisions.

Compaction must preserve raw messages, source pointers, tombstone history, and
an undo path.
