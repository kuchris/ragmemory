"""
Force context-budget drops and verify ledger.json is created.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python test_ledger_drop.py
"""
import json
import shutil
from pathlib import Path

import memory
from memory import MemoryStore

DB_PATH = Path("./chroma_ledger_test")
LEDGER_PATH = DB_PATH / "ledger.json"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

# Keep this test fast and deterministic by forcing a tiny budget.
memory.CONTEXT_TOKEN_BUDGET = 120
memory.RECENT_MESSAGES = 0

store = MemoryStore(db_path=str(DB_PATH))

LONG_MESSAGES = [
    """
    PFlash decision: keep_ratio should stay conservative for code and config
    prompts. The pflash notes explain that aggressive compression can remove
    important file paths, function names, commands, and version numbers.
    """,
    """
    Retrieval design note: structured memory should preserve decisions,
    preferences, constraints, exact config blocks, Markdown tables, code
    references, and open questions. Raw chunks remain the backup.
    """,
    """
    MCP stability note: ragm_mcp/server.py should change slowly because MCP
    clients depend on its tool interface. memory.py is allowed to move faster
    for llama.cpp and pflash experiments.
    """,
    """
    Ledger test payload: this chunk is intentionally long and relevant to
    compression, retrieval, memory, pflash, structured objects, and dropped
    context. It should overflow the tiny test budget and land in ledger.json.
    """,
]

for text in LONG_MESSAGES:
    store.add_message("user", text.strip(), extract_structured=False)

ctx = store.build_context("compression retrieval pflash structured memory ledger")

print("=== Context ===")
print(ctx)

print("\n=== Ledger ===")
if not LEDGER_PATH.exists():
    raise AssertionError("ledger.json was not created")

entries = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
print(json.dumps(entries, ensure_ascii=False, indent=2))
print(f"\nLedger entries: {len(entries)}")

assert len(entries) > 0
