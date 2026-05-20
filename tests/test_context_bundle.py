"""
Verify structured context bundle behavior and compatibility wrapper.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_context_bundle.py
"""
import shutil
from pathlib import Path

import ragmemory.memory as memory
from ragmemory.memory import MemoryStore, RetrievedChunk, format_for_prompt

DB_PATH = Path("./.data/chroma_context_bundle_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

original_budget = memory.CONTEXT_TOKEN_BUDGET
original_recent = memory.RECENT_MESSAGES
memory.CONTEXT_TOKEN_BUDGET = 10
memory.RECENT_MESSAGES = 0

try:
    store = MemoryStore(db_path=str(DB_PATH))
    store.retrieve_structured = lambda _query: []
    store.retrieve = lambda _query: [
        RetrievedChunk(
            id="keep",
            text="KEEP context bundle winner.",
            importance=0.5,
            message_id=2,
            score=0.9,
        ),
        RetrievedChunk(
            id="drop",
            text="DROP context bundle loser.",
            importance=0.5,
            message_id=1,
            score=0.1,
        ),
    ]

    bundle = store.build_context_bundle("context bundle")
    assert bundle.query == "context bundle"
    assert [chunk.id for chunk in bundle.kept] == ["keep"]
    assert [chunk.id for chunk in bundle.would_be_dropped] == ["drop"]
    assert bundle.token_budget == 10
    assert bundle.tokens_used > 0
    assert len(store.ledger) == 0

    prompt = format_for_prompt(bundle)
    assert "KEEP context bundle winner." in prompt
    assert "DROP context bundle loser." not in prompt

    store.commit_drops(bundle)
    assert len(store.ledger) == 1
    assert store.ledger.entries[0].chunk_id == "drop"

    compat_store = MemoryStore(db_path=str(DB_PATH))
    compat_store.retrieve_structured = lambda _query: []
    compat_store.retrieve = store.retrieve
    compat_bundle = compat_store.build_context_bundle("context bundle")
    assert format_for_prompt(compat_bundle) == compat_store.build_context("context bundle")
finally:
    memory.CONTEXT_TOKEN_BUDGET = original_budget
    memory.RECENT_MESSAGES = original_recent

print("Context bundle test passed.")
