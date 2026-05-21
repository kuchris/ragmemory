# ragm-mcp

A RAG memory MCP server that gives MCP-compatible assistants persistent,
recoverable memory across sessions.

## How It Works

The MCP adapter uses the root `ragmemory.MemoryStore` package. That means it
shares the hardened SQLite state, Chroma store, structured context bundle,
event log, ledger, and recoverable tombstone behavior.

```text
user message
    -> recall()         retrieve relevant memory
    -> context injected into prompt
    -> model answers
    -> save()           assistant summary stored
```

## Memory Layers

| Layer | What | When included |
|-------|------|---------------|
| Recent window | Recent messages verbatim | Always |
| Structured memory | Durable facts and exact artifacts | Every user request |
| Retrieved memory | Semantically relevant chunks | Every user request |
| Removal ledger | Chunks dropped from context budget | On missing-context phrases |

Automatic forgetting is decay-aware ranking. Raw memories stay in SQLite and
Chroma; `memory_metadata` lowers stale memories in retrieval over time and
refreshes only memories that are actually injected into context.

## Tools

| Tool | Description |
|------|-------------|
| `recall(user_message)` | Disabled unless `[mcp.tools] enable_recall = true`. Keep disabled when hooks own recall. |
| `save(summary)` | Call at turn end only if `[mcp.tools] enable_save = true`. Stores a short assistant summary and runs one pending extraction job. |
| `remember_document(text)` | Stores a large document or paste only if `[mcp.tools] enable_save = true`. |
| `remove_memory_preview(message_ids, before)` | Shows records that would be tombstoned. `before` is preview-only. |
| `remove_memory_confirm(message_ids, reason)` | Tombstones up to 3 explicit message IDs. Requires `[mcp.tools] enable_tombstone = true`. |
| `forget_preview(...)` / `forget_confirm(...)` | Deprecated aliases kept for compatibility. Prefer `remove_memory_*`. |
| `memory_stats()` | Shows DB path, chunk count, message count, structured count, pending jobs, and ledger size. |

Removal is tombstone-only. Use it only when the current user explicitly asks to
forget, remove, or delete a specific memory. Automatic decay handles stale
memory.

## Setup

This subproject uses the master folder `uv` environment. Run commands from the
master folder unless noted otherwise.

```bash
uv venv
uv pip install -e .
uv pip install chromadb rank-bm25 mcp
```

## Running

From the master folder:

```bash
uv run python ragm_mcp/server.py
```

From inside `ragm_mcp/`:

```bash
uv run --directory .. python ragm_mcp/server.py
```

Data is stored in the root `.data/chroma_db/` directory unless
`RAGMEMORY_DB_PATH` is set:

- `.data/chroma_db/chroma.sqlite3` - Chroma vector store
- `.data/chroma_db/state.sqlite` - raw message log + next message ID
- `.data/chroma_db/ledger.json` - removal ledger
- `.data/chroma_db/structured_memory.jsonl` - structured memory objects
- `.data/chroma_db/events.jsonl` - JSONL event log

Obsidian mirror:

```bash
uv run python scripts/export_obsidian.py --db-path ./.data/chroma_db --output ./.data/obsidian_memory
```

The MCP adapter updates this mirror after `recall`, `save`,
`remember_document`, and `remove_memory_confirm`. The mirror is generated and one-way.
Active records go under `active/`; tombstoned records go under `forgotten/`.
Message notes include previous/next frontmatter, and `maps/` contains
active-only timeline pages plus a simple turns view. Override the output folder
with `RAGMEMORY_OBSIDIAN_PATH`.

The generated graph is semantic-first: chronology lives in frontmatter, maps
are tagged `cssclasses: ["navigation"]`, and structured notes link to generated
hub notes under `topics/`, `files/`, and `profile/`. To hide timeline/turns
navigation from Obsidian graph view, use:

```text
-["cssclasses":"navigation"]
```

Topic hubs are filtered by `[obsidian.topics]` in `ragmemory.local.ini`.
Denylisted artifact/type/language tags are hidden, allowlisted tags are always
shown, and other tags need to recur at least `min_count` times.

## Claude Desktop Config

```json
{
  "mcpServers": {
    "rag-memory": {
      "command": "uv",
      "args": ["run", "python", "/path/to/ragm_mcp/server.py"],
      "cwd": "/path/to/ragm_mcp"
    }
  }
}
```

## LM Studio Config (`%USERPROFILE%\.lmstudio\mcp.json`)

In LM Studio, open the right sidebar `Program` tab, then choose `Install` ->
`Edit mcp.json`. Add this server entry under `mcpServers`:

```json
{
  "mcpServers": {
    "rag-memory": {
      "command": "uv",
      "args": ["run", "python", "ragm_mcp/server.py"],
      "cwd": "C:/path/to/master-folder"
    }
  }
}
```

If LM Studio cannot find `uv`, replace `"command": "uv"` with the full path to
`uv.exe`.

## OpenCode Config (`~/.config/opencode/opencode.json`)

```json
{
  "mcpServers": {
    "rag-memory": {
      "command": "uv",
      "args": ["run", "python", "ragm_mcp/server.py"],
      "cwd": "C:/path/to/master-folder"
    }
  },
  "instructions": ["C:/path/to/ragm_mcp/AGENTS.md"]
}
```

## Codex CLI Config (`~/.codex/config.toml`)

```toml
[mcp_servers.rag-memory]
command = "uv"
args = ["run", "python", "ragm_mcp/server.py"]
cwd = "C:/path/to/master-folder"
```

Codex usually works better with the direct hook setup in
`ragm_mcp/hooks/README.md`, but the MCP server remains available for clients
that prefer explicit memory tools.

The direct hooks save messages and enqueue structured extraction/compaction
jobs. Run `uv run python scripts/run_worker.py` in another terminal to process
those jobs without making the hooks wait on LLM API calls.

## Agent Instructions

Add to your agent's system prompt or use the included `AGENTS.md`:

```text
Every turn:
1. Call recall(user_message=<user message>) before answering only when MCP recall is enabled
2. Call save(summary=<1-2 sentence summary>) after answering only when MCP save is enabled
For large documents: use remember_document(text=<content>) only when MCP save is enabled
For removal: call remove_memory_preview(...) first, then remove_memory_confirm(message_ids=<explicit ids>, reason=<specific reason>) only after explicit user approval
```

## Configuration

Set the DB path with:

```bash
RAGMEMORY_DB_PATH=./.data/chroma_db
```

Set the Obsidian mirror path with:

```bash
RAGMEMORY_OBSIDIAN_PATH=./.data/obsidian_memory
```

Core retrieval constants live in `src/ragmemory/memory.py`.

## Technical Details

- **Embeddings**: ChromaDB default embedding function
- **Vector store**: ChromaDB, persistent and local
- **Raw state**: SQLite
- **Keyword search**: BM25, rebuilt lazily on search
- **Retrieval fusion**: Reciprocal Rank Fusion combining embedding + BM25 scores
- **Forgetting**: SQLite metadata decay reranking; no Chroma deletion
- **Context budgeting**: retrieval-score based
- **Chunking**: paragraph-based with header context injection and sentence fallback
