"""
Verify structured context bundle behavior and compatibility wrapper.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_context_bundle.py
"""
import shutil
from pathlib import Path

import ragmemory.memory as memory
from ragmemory.memory import (
    ContextBundle,
    MemoryStore,
    MessageRecord,
    RetrievedChunk,
    format_for_prompt,
)

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

    noisy_bundle = ContextBundle(
        query="token cleanup",
        recent=[
            MessageRecord(
                role="user",
                text="# Context from my IDE setup:\n\n"
                "## Active file: ragmemory.local.ini\n\n"
                "## Open tabs:\n"
                "- ragmemory.local.ini: ragmemory.local.ini\n\n"
                "## My request for Codex:\n"
                "remember the SQLite worker plan",
                message_id=10,
                content_hash="recent-1",
            ),
            MessageRecord(
                role="user",
                text="btw",
                message_id=11,
                content_hash="recent-2",
            ),
        ],
        structured=[],
        retrieved=[],
        ledger_recovered=[],
        kept=[
            RetrievedChunk(
                id="noisy",
                text="# Context from my IDE setup: ## Active file: ragmemory.local.ini "
                "## Open tabs:\n"
                "- compact_backfill.py: scripts/compact_backfill.py\n"
                "## My request for Codex:\n"
                "let me ask claude",
                importance=0.5,
                message_id=12,
            ),
            RetrievedChunk(
                id="duplicate",
                text="remember the SQLite worker plan",
                importance=0.5,
                message_id=13,
            ),
        ],
        would_be_dropped=[],
        token_budget=100,
        tokens_used=1,
    )
    noisy_prompt = format_for_prompt(noisy_bundle)
    assert "Context from my IDE setup" not in noisy_prompt
    assert "Open tabs" not in noisy_prompt
    assert "btw" not in noisy_prompt
    assert "let me ask claude" not in noisy_prompt
    assert noisy_prompt.count("remember the SQLite worker plan") == 1

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

    dropped_recent_text = "DROPPED_RECENT_CAN_RETURN from retrieval."
    budget_store = MemoryStore(db_path=str(DB_PATH))
    budget_store.raw_log = [
        {
            "role": "user",
            "text": "ok",
            "message_id": 20,
            "content_hash": "recent-ok",
        },
        {
            "role": "assistant",
            "text": dropped_recent_text,
            "message_id": 21,
            "content_hash": "recent-drop",
        },
    ]
    budget_store.retrieve_structured = lambda _query, top_k=memory.STRUCTURED_TOP_K: []
    budget_store.retrieve = lambda _query, top_k=memory.RETRIEVE_TOP_K: [
        RetrievedChunk(
            id="dropped-recent-returned",
            text=dropped_recent_text,
            importance=0.5,
            message_id=21,
            score=0.9,
        )
    ]
    budget_store.configure_recall(
        retrieve_top_k=1,
        structured_top_k=0,
        context_token_budget=40,
        recent_messages=2,
        include_recent=True,
        include_structured=False,
        recent_token_budget_ratio=0.1,
    )
    budget_bundle = budget_store.build_context_bundle("budget")
    assert [message.text for message in budget_bundle.recent] == ["ok"]
    assert [chunk.id for chunk in budget_bundle.kept] == ["dropped-recent-returned"]

    budget_store.configure_recall(
        retrieve_top_k=1,
        structured_top_k=0,
        context_token_budget=40,
        recent_messages=2,
        include_recent=True,
        include_structured=False,
        recent_token_budget_ratio=0,
    )
    no_recent_bundle = budget_store.build_context_bundle("budget")
    assert no_recent_bundle.recent == []
    assert [chunk.id for chunk in no_recent_bundle.kept] == ["dropped-recent-returned"]

    budget_store.raw_log = [
        {
            "role": "assistant",
            "text": "fits within full budget",
            "message_id": 22,
            "content_hash": "recent-fit",
        }
    ]
    budget_store.retrieve = lambda _query, top_k=memory.RETRIEVE_TOP_K: []
    budget_store.configure_recall(
        retrieve_top_k=0,
        structured_top_k=0,
        context_token_budget=10,
        recent_messages=1,
        include_recent=True,
        include_structured=False,
        recent_token_budget_ratio=9,
    )
    clamped_bundle = budget_store.build_context_bundle("budget")
    assert [message.text for message in clamped_bundle.recent] == [
        "fits within full budget"
    ]
finally:
    memory.CONTEXT_TOKEN_BUDGET = original_budget
    memory.RECENT_MESSAGES = original_recent

print("Context bundle test passed.")
