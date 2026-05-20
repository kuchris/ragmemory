"""
Verify context compression uses retrieval score, not keyword importance.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_importance_policy.py
"""
import shutil
from pathlib import Path

import ragmemory.memory as memory
from ragmemory.memory import MemoryStore, RetrievedChunk

DB_PATH = Path("./.data/chroma_importance_policy_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

original_budget = memory.CONTEXT_TOKEN_BUDGET
original_recent = memory.RECENT_MESSAGES
memory.CONTEXT_TOKEN_BUDGET = 12
memory.RECENT_MESSAGES = 0

try:
    store = MemoryStore(db_path=str(DB_PATH))
    store.retrieve_structured = lambda _query: []
    store.retrieve = lambda _query: [
        RetrievedChunk(
            id="low-score",
            text="LOW_SCORE critical warning note.",
            importance=1.0,
            message_id=1,
            score=0.1,
        ),
        RetrievedChunk(
            id="high-score",
            text="HIGH_SCORE plain winner.",
            importance=0.1,
            message_id=2,
            score=0.9,
        ),
    ]

    assert memory.score_importance("critical important warning bug") == 0.5
    assert memory.score_importance("plain note") == 0.5

    bundle = store.build_context_bundle("retrieval winner")
    assert len(store.ledger) == 0
    store.commit_drops(bundle)
    context = memory.format_for_prompt(bundle)
    assert "HIGH_SCORE" in context
    assert "LOW_SCORE" not in context
    assert len(store.ledger) == 1
    assert store.ledger.entries[0].chunk_id == "low-score"
finally:
    memory.CONTEXT_TOKEN_BUDGET = original_budget
    memory.RECENT_MESSAGES = original_recent

print("Importance policy test passed.")
