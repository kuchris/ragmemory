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

## Tools

| Tool | Description |
|------|-------------|
| `recall(user_message)` | Call at turn start. Retrieves context, then stores the user message. |
| `save(summary)` | Call at turn end. Stores a short assistant summary and runs one pending extraction job. |
| `remember_document(text)` | Stores a large document or paste, split into chunks. |
| `memory_stats()` | Shows DB path, chunk count, message count, structured count, pending jobs, and ledger size. |

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

## Agent Instructions

Add to your agent's system prompt or use the included `AGENTS.md`:

```text
Every turn:
1. Call recall(user_message=<user message>) before answering
2. Call save(summary=<1-2 sentence summary>) after answering
For large documents: use remember_document(text=<content>)
```

## Configuration

Set the DB path with:

```bash
RAGMEMORY_DB_PATH=./.data/chroma_db
```

Core retrieval constants live in `src/ragmemory/memory.py`.

## Technical Details

- **Embeddings**: ChromaDB default embedding function
- **Vector store**: ChromaDB, persistent and local
- **Raw state**: SQLite
- **Keyword search**: BM25, rebuilt lazily on search
- **Retrieval fusion**: Reciprocal Rank Fusion combining embedding + BM25 scores
- **Context budgeting**: retrieval-score based
- **Chunking**: paragraph-based with header context injection and sentence fallback
