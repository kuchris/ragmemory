# ragm-mcp

A RAG memory MCP server that gives any MCP-compatible AI assistant persistent, recoverable memory across sessions.

## How it works

Each saved message is chunked, embedded, and stored in a local vector database. On every user request, relevant past context is retrieved via hybrid search (embeddings + BM25) and injected into the prompt. Chunks that exceed the token budget are compressed out but logged to a removal ledger -- nothing is permanently lost.

```
user message
    → recall()         hybrid retrieval (embeddings + BM25 via RRF)
    → context injected into prompt
    → model answers
    → save()           summary stored, chunked, indexed
```

## Memory layers

| Layer | What | When included |
|-------|------|---------------|
| Recent window | Last 12 messages verbatim | Always |
| Retrieved memory | Top-5 semantically relevant chunks | Every user request |
| Removal ledger | Chunks compressed out of budget | On demand (missing context phrases) |

## Tools

| Tool | Description |
|------|-------------|
| `recall(user_message)` | Call at turn start. Stores user message, returns relevant memory context. |
| `save(summary)` | Call at turn end. Store a short summary of your response. |
| `remember_document(text)` | Store a large document or long paste, split into chunks. |
| `memory_stats()` | Show chunk count, message count, ledger size. |

## Setup

This subproject uses the master folder `uv` environment. Run commands from the master folder unless noted otherwise.

If you rebuild the master environment, install the MCP runtime dependencies there:

```bash
uv venv
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

Data is stored in `chroma_db/` next to `server.py`:
- `chroma_db/` — vector store (ChromaDB)
- `chroma_db/state.json` — raw message log + next message ID
- `chroma_db/ledger.json` — removal ledger (persisted across restarts)

## Claude Desktop config

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

## OpenCode config (`~/.config/opencode/opencode.json`)

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

## Codex CLI config (`~/.codex/config.toml`)

```toml
[mcp_servers.rag-memory]
command = "uv"
args = ["run", "python", "ragm_mcp/server.py"]
cwd = "C:/path/to/master-folder"
```

Codex reads `AGENTS.md` from the project root automatically. Copy or symlink `AGENTS.md` into your project directory:

```bash
cp C:/path/to/ragm_mcp/AGENTS.md ./AGENTS.md
```

## Agent instructions (AGENTS.md)

Add to your agent's system prompt or use the included `AGENTS.md`:

```
Every turn:
1. Call recall(user_message=<user message>) before answering
2. Call save(summary=<1-2 sentence summary>) after answering
For large documents: use remember_document(text=<content>)
```

## Configuration

Edit the constants at the top of `server.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_MAX_TOKENS` | 300 | Max tokens per chunk before splitting |
| `CHUNK_MIN_TOKENS` | 80 | Min tokens before merging with neighbor |
| `RECENT_MESSAGES` | 12 | How many recent messages to always include verbatim |
| `RETRIEVE_TOP_K` | 5 | Number of chunks to retrieve per user request |
| `CONTEXT_TOKEN_BUDGET` | 2000 | Max tokens for retrieved memory in prompt |

## Technical details

- **Embeddings**: `all-MiniLM-L6-v2` via ChromaDB's built-in ONNX runtime (no Ollama required)
- **Vector store**: ChromaDB (persistent, local)
- **Keyword search**: BM25 (rank-bm25)
- **Retrieval fusion**: Reciprocal Rank Fusion (RRF) combining embedding + BM25 scores
- **Importance scoring**: Heuristic (length, signal words, code presence, numbers)
- **Chunking**: Paragraph-based with header context injection and sentence-level fallback
