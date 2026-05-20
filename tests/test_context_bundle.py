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
    def fake_retrieve(_query, top_k=memory.RETRIEVE_TOP_K):
        chunks = [
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
        return chunks[:top_k]

    store.retrieve_structured = lambda _query, top_k=memory.STRUCTURED_TOP_K: []
    store.retrieve = fake_retrieve

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

    store.configure_recall(
        retrieve_top_k=0,
        structured_top_k=0,
        context_token_budget=5,
        recent_messages=0,
        include_recent=False,
        include_structured=False,
    )
    small_bundle = store.build_context_bundle("context bundle")
    assert small_bundle.token_budget == 5
    assert small_bundle.recent == []
    assert small_bundle.structured == []
    assert small_bundle.retrieved == []

    compat_store = MemoryStore(db_path=str(DB_PATH))
    compat_store.retrieve_structured = lambda _query, top_k=memory.STRUCTURED_TOP_K: []
    compat_store.retrieve = fake_retrieve
    compat_bundle = compat_store.build_context_bundle("context bundle")
    assert format_for_prompt(compat_bundle) == compat_store.build_context("context bundle")
finally:
    memory.CONTEXT_TOKEN_BUDGET = original_budget
    memory.RECENT_MESSAGES = original_recent

print("Context bundle test passed.")
