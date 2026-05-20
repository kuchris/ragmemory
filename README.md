# RagMemory

RagMemory is a local prototype for persistent chat memory.

The goal is to keep useful context outside the model prompt, retrieve it when needed, and preserve exact artifacts such as configs, tables, and code blocks.

## Concept

The model should not need the full conversation every turn.

RagMemory stores memory in layers:

| Layer | Purpose |
|------|---------|
| Raw log | Source-of-truth messages in `state.json`. |
| Raw chunks | Searchable Chroma chunks for broad recall. |
| Structured memory | High-signal objects such as decisions, constraints, configs, tables, and open questions. |
| Recent window | Latest messages included directly for short-term continuity. |
| Removal ledger | Retrieved chunks that were dropped because the context budget was full. |

The important rule is:

```text
raw chunks preserve recall
structured memory preserves meaning
exact artifacts preserve usable source
```

## Current Pipeline

When saving a message:

```text
message
  -> append raw log
  -> split into chunks
  -> embed chunks in Chroma
  -> extract exact artifacts by code
  -> optionally extract structured memory with NVIDIA API
  -> store structured objects in JSONL + Chroma
```

When answering a question:

```text
question
  -> retrieve structured memory
  -> retrieve raw chunks with embedding search + BM25
  -> add recent messages
  -> drop overflow chunks into ledger.json
  -> send context to the model
```

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
```

Configs, tables, Mermaid diagrams, and code blocks are detected by code when possible, so their `source_text` stays exact. The LLM is used for semantic objects such as decisions, preferences, constraints, and open questions.

## Files

| File | Purpose |
|------|---------|
| `src/ragmemory/memory.py` | Core memory engine. |
| `scripts/chat.py` | CLI chat using NVIDIA API and RagMemory context. |
| `scripts/ask_memory.py` | Prints retrieved context for preset questions. |
| `scripts/view_chunks.py` | Shows raw Chroma chunks. |
| `tests/test_memory.py` | Basic raw retrieval behavior test. |
| `tests/test_structured_memory.py` | NVIDIA-backed structured extraction test. |
| `tests/test_ledger_drop.py` | Forced context-budget drop test. |
| `ragm_mcp/hooks/` | Codex hook scripts and install notes for automatic memory. |
| `experiments/llamac_plan.md` | Design notes for future llama.cpp / pflash direction. |

Generated folders:

```text
chroma_structured_test/
chroma_test/
chroma_ledger_test/
chroma_db/
```

## Setup

```powershell
uv venv
uv pip install -e .
uv pip install openai chromadb rank-bm25 mcp
```

For NVIDIA-backed extraction/chat:

```powershell
$env:NVIDIA_API_KEY='your-nvidia-api-key'
```

Optional model override:

```powershell
$env:STRUCTURED_MEMORY_MODEL='meta/llama-3.1-8b-instruct'
```

## Run

Chat with the structured-memory test DB:

```powershell
uv run python scripts/chat.py
```

`scripts/chat.py` defaults to:

```text
./chroma_structured_test
```

Use another DB:

```powershell
$env:RAGMEMORY_DB_PATH='./chroma_db'
uv run python scripts/chat.py
```

Inspect retrieval context without calling the LLM:

```powershell
uv run python scripts/ask_memory.py
```

View raw chunks:

```powershell
uv run python scripts/view_chunks.py
```

Test structured extraction:

```powershell
$env:NVIDIA_API_KEY='your-nvidia-api-key'
uv run python tests/test_structured_memory.py
```

Test ledger dropping:

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python tests/test_ledger_drop.py
```

## Notes

- `message_id` is a message sequence number, not a paired user/assistant turn ID.
- Normal chat saves raw chunks but does not run structured extraction every turn.
- Structured extraction is best used for durable information: decisions, preferences, constraints, configs, tables, code references, and open questions.
- Graph memory is intentionally postponed. The current focus is reliable layered memory first.
