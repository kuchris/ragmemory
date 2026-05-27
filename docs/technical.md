# RagMemory Technical Notes

This document explains the technical method behind RagMemory. The README is for
normal use; this file is for understanding how the memory system works.

## Technical Main

RagMemory is a local persistent-memory layer for Codex.

The main idea is:

```text
chat messages -> local memory store -> retrieval context -> next model prompt
```

The model does not receive the full conversation every turn. Instead, RagMemory
stores the conversation locally, searches for relevant memory before each new
prompt, and injects only a small context bundle.

The core design has three separate roles:

| Part | Role |
|------|------|
| RagMemory DB | Source of truth for memory and recall. |
| Codex hooks | Automatic remember and recall path. |
| Obsidian export | Human-readable mirror for inspection and cleanup. |

The Obsidian vault is not the memory engine. It is a generated view of what the
engine already stores.

## Memory Method

RagMemory uses layered memory instead of one single summary.

| Layer | Stored as | Purpose |
|-------|-----------|---------|
| Raw messages | SQLite `messages` table | Source of truth; never summarized away. |
| Raw chunks | Chroma collection + BM25 index | Broad semantic and keyword recall. |
| Structured memory | JSONL + Chroma collection | Durable facts such as decisions and preferences. |
| Recent window | Latest raw messages | Short-term continuity. |
| Removal ledger | `ledger.json` | Overflow chunks dropped from the prompt budget. |
| Metadata | SQLite `memory_metadata` table | Decay, access count, tombstone, pin state. |

The key rule is:

```text
raw messages preserve evidence
raw chunks preserve recall
structured memory preserves meaning
metadata controls ranking
```

RagMemory does not depend on one compressed conversation summary. Raw messages
stay as the recoverable ground truth, while structured memory is only an
additional high-signal layer.

## Save Method

When a message is saved:

```text
message
  -> allocate message_id in SQLite
  -> store raw message in state.sqlite
  -> split text into chunks
  -> embed chunks into Chroma
  -> add chunks to BM25
  -> create memory_metadata row
  -> extract exact artifacts by code
  -> optionally extract structured memory with an LLM
  -> log events to events.jsonl
```

The `message_id` is the stable join key between raw messages, chunks,
structured objects, metadata, events, and Obsidian notes.

Structured extraction can run immediately or as a background job. In the Codex
hook path, user and assistant messages are saved first, then structured
extraction and compaction jobs are queued in SQLite for `scripts/run_worker.py`.

## Recall Method

Before a new Codex prompt, the UserPromptSubmit hook builds a context bundle:

```text
new prompt
  -> retrieve structured memory
  -> retrieve raw chunks with vector search
  -> retrieve raw chunks with BM25 keyword search
  -> merge and rerank candidates
  -> apply decay-aware score
  -> add recent messages
  -> fit raw chunks into token budget
  -> inject context into Codex
  -> mark injected memories as accessed
```

Raw chunk retrieval combines:

- Chroma vector search for semantic similarity.
- BM25 for exact keyword matching.
- Reciprocal-rank fusion to merge both result lists.
- Decay-aware reranking so stale unused memory naturally drops.

The hook recall limits are configurable in `ragmemory.local.ini`:

```ini
[recall]
context_token_budget = 500
retrieve_top_k = 2
structured_top_k = 1
recent_messages = 2
include_recent = true
include_structured = true
```

The default constants still exist in `src/ragmemory/memory.py`, but the hook path
overrides them from local config.

## Forgetting Method

RagMemory has two different concepts that should not be confused:

| Word | Meaning |
|------|---------|
| Forgetting | Automatic decay-aware ranking. |
| Removal | Manual tombstone for wrong, private, or harmful memory. |

Automatic forgetting does not delete data. It changes retrieval priority.

Each memory has metadata:

- `created_at`
- `last_accessed_at`
- `access_count`
- `base_importance`
- `half_life_days`
- `pinned`
- `superseded_by`
- `tombstoned_at`

The retrieval score is adjusted by decay strength. Old unused memory becomes
less likely to appear. If it is retrieved and injected again, its access time is
refreshed, so useful memory can stay active.

Manual removal is tombstone-only. Tombstoned records stop appearing in recall
and move to `forgotten/` in the Obsidian mirror, but the raw data is still kept
for audit and recovery.

## Structured Memory Method

Structured memory is used for durable, high-signal objects.

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

There are two extraction paths:

| Extractor | What it handles |
|-----------|-----------------|
| Deterministic extractor | Exact fenced configs, code blocks, tables, Mermaid diagrams. |
| LLM extractor | Semantic facts such as decisions, preferences, constraints, open questions. |

Structured objects are stored in:

```text
<db_path>/structured_memory.jsonl
```

Each structured object keeps:

- object ID
- type
- summary
- source text
- tags
- importance
- message ID
- role

Structured additions are logged as `structured_object_added` events. The event
includes the background `job_id` when applicable, which is useful for future
duplicate detection.

## Hook Method

Hooks are the normal automation path.

| Hook | Action |
|------|--------|
| `UserPromptSubmit` | Recall memory, inject context, save the user message. |
| `Stop` | Save assistant response, enqueue background jobs, export Obsidian mirror. |

This gives the practical behavior:

```text
before user prompt reaches model -> recall
after assistant finishes -> remember
```

The hook path bypasses MCP tool gates because hooks are the trusted automatic
pipeline.

## MCP Method

MCP is optional when hooks are installed.

Recommended local config:

```ini
[mcp.tools]
enable_recall = false
enable_save = false
enable_tombstone = true
```

The reason is:

- Hooks already own automatic recall and save.
- MCP recall/save can duplicate token usage or writes.
- MCP remains useful for inspection and manual removal tools.

The safe removal surface is:

```text
remove_memory_preview(...)
remove_memory_confirm(message_ids, reason)
```

`remove_memory_confirm` requires explicit message IDs, a reason, a max of three
IDs, and the tombstone flag enabled. It never hard-deletes memory.

## Obsidian Method

The Obsidian vault is generated from the DB.

```text
.data/obsidian_memory/
```

It is a mirror, not a source of truth. Edit the RagMemory data path or removal
tools, not generated Markdown files.

Main generated folders:

| Folder | Meaning |
|--------|---------|
| `active/messages/` | Active raw message notes. |
| `active/structured/` | Active structured memory notes. |
| `forgotten/` | Tombstoned records. |
| `topic_groups/` | Optional LLM-curated upper layer over leaf topics. |
| `topics/` | Generated topic hubs. |
| `files/` | Optional file/path hubs. |
| `profile/` | User profile and identity/preference hub. |
| `maps/` | Timeline and navigation pages. |

The graph is intentionally filtered:

- chronology links are frontmatter, not body wikilinks
- raw messages with no structured links get `memory-unlinked`
- file hubs are off by default
- topic hubs fall back to count-based tags with allowlist, denylist, and
  `min_count`
- `topic_taxonomy.json` can add `topic_groups/` above leaf topics without
  deleting or replacing the leaf `topics/` notes

Useful Obsidian graph filter:

```text
-["cssclasses":"navigation"] -["cssclasses":"memory-unlinked"]
```

## Storage Files

The active DB folder usually contains:

| File | Purpose |
|------|---------|
| `state.sqlite` | Raw messages, metadata, tombstone state, next message ID. |
| `chroma.sqlite3` | Chroma vector collections. |
| `structured_memory.jsonl` | Structured memory objects. |
| `ledger.json` | Chunks dropped from context budget. |
| `events.jsonl` | Audit and observability event log. |
| `hook_debug.jsonl` | Hook execution debug records. |

Local config:

| File | Purpose |
|------|---------|
| `ragmemory.local.ini` | Private local settings and API key. Ignored by git. |
| `ragmemory.example.ini` | Safe template committed to git. |

## Important Scripts

| File | Purpose |
|------|---------|
| `scripts/inspect_events.py` | Inspect event log. |
| `scripts/export_obsidian.py` | Generate Obsidian mirror. |
| `scripts/regroup_topics.py` | Create or queue LLM-curated topic groups. |
| `scripts/generate_wiki.py` | Generate non-LLM wiki pages from topic groups. |
| `scripts/animate_obsidian_graph.py` | Render the Obsidian graph as a growing GIF. |
| `scripts/check_obsidian_graph.py` | Smoke-test generated graph quality. |
| `scripts/remove_memory.py` | Preview/confirm tombstone removal. |
| `scripts/ask_memory.py` | Print retrieved context for test prompts. |
| `scripts/view_chunks.py` | Inspect raw Chroma chunks. |
| `scripts/chat.py` | CLI chat path. |

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
| `tests/test_dedup.py` | Duplicate-message skip. |
| `tests/test_sqlite_state.py` | SQLite state and concurrent writer behavior. |
| `tests/test_background_extraction.py` | Background structured extraction. |
| `tests/test_event_log.py` | JSONL event logging. |
| `tests/test_export_obsidian.py` | Obsidian mirror export. |
| `tests/test_context_bundle.py` | Context budget and bundle behavior. |
| `tests/test_decay_forgetting.py` | Decay-aware ranking and access touch. |
| `tests/test_ragm_mcp_forget_tools.py` | MCP remove/tombstone wrapper. |
| `tests/test_remove_memory_script.py` | CLI tombstone workflow. |

`tests/test_structured_memory.py` uses the configured NVIDIA-backed extractor.

## Future Compaction Method

Compaction is intentionally not implemented yet.

The future method should be read-only first:

```text
scripts/compact.py --dry-run
```

It should report candidates without rewriting memory.

Concrete triggers:

- `topics/` grows above roughly 30 hubs despite topic filtering.
- `structured_object_added` events show repeated objects on the same
  `message_id`.
- at least three tag spellings normalize to the same concept
- at least five message IDs have multiple structured objects whose tags overlap
  by more than 50%
- retrieval returns several results that are obviously the same memory

Safe Tier 1 checks:

- tag variant detection
- exact duplicate structured objects
- likely duplicate structured objects from repeated background extraction jobs
- same `message_id`, different `job_id`, same `type`, overlapping normalized
  tags

Do not add automatic LLM "sleep mode" that rewrites memory. If LLM assistance is
added later, use it only as a bounded judge for specific duplicate,
normalization, or supersession decisions.

Compaction must preserve:

- raw messages
- source pointers
- tombstone history
- event log evidence
- undo path
