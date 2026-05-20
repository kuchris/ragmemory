"""
Manual test for NVIDIA-backed structured memory extraction.

Run:
    $env:PYTHONIOENCODING='utf-8'
    $env:NVIDIA_API_KEY='your-nvidia-api-key'
    uv run python tests/test_structured_memory.py
"""
import os
import json
import shutil
from pathlib import Path

from ragmemory.memory import MemoryStore

DB_PATH = Path("./chroma_structured_test")
LEDGER_PATH = DB_PATH / "structured_memory.jsonl"

message = """
We decided to keep memory.py separate from ragm_mcp/server.py because memory.py
is the reference playground for llama.cpp and pflash experiments, while
ragm_mcp/server.py should stay stable for MCP clients.

Never merge those two engines until their behavior is stable. I prefer minimal
changes in ragm_mcp because OpenCode and other MCP clients depend on that tool
surface.

Use this OpenCode config when testing the MCP server from the master folder:

```json
{
  "mcp": {
    "rag-memory": {
      "type": "local",
      "command": ["uv", "run", "--directory", "..", "python", "ragm_mcp/server.py"],
      "enabled": true
    }
  }
}
```

| File | Purpose |
|---|---|
| memory.py | llama.cpp and pflash reference experiments |
| ragm_mcp/server.py | stable MCP server |

Important code reference: MemoryStore.add_message() is where raw chunks and
structured memory objects are saved.

Open question: should structured extraction run every message, or only for user
messages and long assistant summaries?
""".strip()


def main():
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("Set NVIDIA_API_KEY before running this test.")

    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    store = MemoryStore(db_path=str(DB_PATH))

    print("=== Add Message ===")
    store.add_message("user", message)

    print("\n=== Structured Objects ===")
    print(f"Count: {len(store.structured)}")
    if LEDGER_PATH.exists():
        structured_text = LEDGER_PATH.read_text(encoding="utf-8")
        print(structured_text)
        objects = [json.loads(line) for line in structured_text.splitlines() if line.strip()]
        exact_configs = [
            obj for obj in objects
            if obj["type"] == "config" and obj["source_text"].startswith("```json")
        ]
        assert exact_configs
        assert len([obj for obj in objects if obj["type"] == "config"]) == 1
        assert '"command": ["uv", "run", "--directory", "..", "python", "ragm_mcp/server.py"]' in exact_configs[0]["source_text"]
    else:
        print("structured_memory.jsonl was not created.")

    print("\n=== Retrieved Context ===")
    ctx = store.build_context("why did we keep memory.py separate?")
    print(ctx)


if __name__ == "__main__":
    main()
