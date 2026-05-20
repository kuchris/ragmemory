# RagMemory API v0

This document defines the public library boundary for the next hardening pass.
It is a target contract, not a promise that the current implementation already
matches it.

The goal for v0 is small and boring: callers can save memory, retrieve memory,
search memory, and safely forget memory without parsing formatted prompt text or
depending on internal storage details.

## Public Surface

The public library surface should be:

```python
store.add_message(role, text, *, extract_structured=True) -> AddMessageResult
store.build_context_bundle(query) -> ContextBundle
store.search(query, top_k=None) -> list[SearchResult]
store.forget(*, query=None, before=None, message_ids=None, confirm=False) -> ForgetPreview | ForgetResult
```

Prompt formatting is separate:

```python
format_for_prompt(context: ContextBundle) -> str
```

The old `build_context(query) -> str` wrapper is deprecated. It remains for
compatibility and should call `build_context_bundle`, `commit_drops`, and
`format_for_prompt`.

MCP tools may wrap these methods, but they should not become the internal
library contract.

## Principles

- Return structured data, not preformatted prompt strings.
- Return IDs for anything saved so logs, retrieval, and background jobs can be
  correlated.
- Treat destructive operations as preview-first and recoverable by default.
- Keep storage, embedding, chunking, ranking, and prompt formatting internal.
- Prefer one clear behavior over multiple hidden modes.

## add_message

```python
store.add_message(role, text, *, extract_structured=True) -> AddMessageResult
```

Adds one user, assistant, or document message to memory.

Expected return shape:

```python
@dataclass
class AddMessageResult:
    saved: bool
    deduped: bool
    message_id: int | None
    content_hash: str
    chunk_ids: list[str]
    structured_object_ids: list[str]
    queued_job_ids: list[str]
```

Rules:

- Duplicate detection should use normalized exact text or a content hash, not
  substring matching.
- If a message is deduped, return `saved=False`, `deduped=True`, and the hash.
- The method should return before slow background structured extraction when
  background mode exists.
- Caller-provided metadata is reserved for a later API revision.

## build_context_bundle

```python
store.build_context_bundle(query) -> ContextBundle
```

Builds the memory bundle for a user query.

Expected return shape:

```python
@dataclass
class ContextBundle:
    query: str
    recent: list[MessageRecord]
    structured: list[StructuredMemoryRecord]
    retrieved: list[ChunkRecord]
    ledger_recovered: list[ChunkRecord]
    kept: list[ChunkRecord]
    would_be_dropped: list[ChunkRecord]
    token_budget: int
    tokens_used: int
```

Rules:

- Do not return one giant formatted string as the public contract.
- The bundle should preserve enough metadata to explain why each item appeared.
- Prompt text should be produced by `format_for_prompt(context)`.
- Dropped items should be visible in the bundle and event log.
- Building a bundle should not mutate the ledger. Call `commit_drops(bundle)`
  when the caller wants to persist budget drops.

## search

```python
store.search(query, top_k=None) -> list[SearchResult]
```

Searches memory without constructing a prompt context.

Expected return shape:

```python
@dataclass
class SearchResult:
    item_id: str
    item_type: str
    text: str
    score: float
    message_id: int | None
    source: str
    metadata: dict
```

Rules:

- Return result objects with metadata, not strings.
- `source` should identify whether the hit came from raw chunks, structured
  memory, recent messages, or ledger recovery.
- Ranking details may change internally as long as the returned shape remains
  stable.

## forget

```python
store.forget(*, query=None, before=None, message_ids=None, confirm=False) -> ForgetPreview | ForgetResult
```

Safely removes memory from active retrieval.

Expected default behavior:

- `confirm=False` returns a preview only.
- `confirm=True` tombstones records instead of hard-deleting them.
- Physical deletion is a separate maintenance operation, not v0 behavior.

Expected return shape:

```python
@dataclass
class ForgetPreview:
    messages: list[MessageRecord]
    chunks: list[ChunkRecord]
    structured: list[StructuredMemoryRecord]
    ledger_entries: list[LedgerRecord]
    message_count: int
    chunk_count: int
    structured_count: int
    ledger_count: int
    truncated: bool

@dataclass
class ForgetResult(ForgetPreview):
    tombstoned_count: int
    event_id: str
```

Rules:

- The current implementation supports `message_ids` and `before`.
- `before` means message `created_at` ingest time; records strictly older than
  the cutoff match.
- `query` is a reserved selector.
- Passing multiple selectors together is reserved for a later decision.
- Preview record lists may be capped, but counts should reflect total matches.
- At least one selector is required: `query`, `before`, or `message_ids`.
- Forgetting a message should cascade logically to its chunks, structured
  objects, and ledger entries.
- Tombstoned records should be excluded from normal retrieval.
- Forget operations must be recorded in the event log.

## Internal Surface

These are internal and may change without breaking the v0 API:

- Chroma collection names and metadata layout.
- SQLite table names and indexes.
- BM25 implementation.
- Chunking strategy.
- Importance/ranking policy.
- Structured extraction prompts and models.
- Prompt formatting.
- Ledger storage layout.

## Reserved Later

These ideas are intentionally not part of the current implemented API:

- `filters` on `build_context_bundle` and `search`.
- Caller-provided metadata on `add_message`.
- Semantic `forget(query=...)`.
- Background structured extraction jobs.

## MCP Mapping

The MCP server may keep a simple tool interface:

```text
recall(user_message)
save(summary)
remember_document(text)
memory_stats()
```

Internally, those tools should call the public library API:

- `recall` calls `add_message`, `build_context_bundle`, `commit_drops`, and
  `format_for_prompt`.
- `save` calls `add_message`.
- `remember_document` calls `add_message` or a document-specific wrapper.
- `memory_stats` reads storage/index statistics.

The MCP response can remain a string, but the library should not be string-only.

## Open Questions

- Should `remember_document` be a public fifth method or a thin wrapper around
  `add_message(role="document", ...)`?
- Should background extraction return only job IDs, or also a wait handle?
- Should tombstones be compacted manually only, or by an explicit retention
  policy later?
- What is the first stable storage version number?
