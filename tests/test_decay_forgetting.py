"""
Verify automatic forgetting uses decay-aware ranking, not deletion.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_decay_forgetting.py
"""
import shutil
import sqlite3
from pathlib import Path

import ragmemory.memory as memory
from ragmemory import MemoryStore
from ragmemory.memory import RetrievedChunk


DB_PATH = Path("./.data/decay_forgetting_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

original_recent = memory.RECENT_MESSAGES
original_budget = memory.CONTEXT_TOKEN_BUDGET
memory.RECENT_MESSAGES = 0

try:
    store = MemoryStore(db_path=str(DB_PATH))
    stale = store.add_message(
        "user",
        "DECAY_TARGET stale memory about the old implementation.",
        extract_structured=False,
    )
    fresh = store.add_message(
        "user",
        "DECAY_TARGET fresh memory about the current implementation.",
        extract_structured=False,
    )
    assert stale.message_id is not None
    assert fresh.message_id is not None

    with sqlite3.connect(store.state_db) as conn:
        rows = conn.execute(
            """
            SELECT message_id, memory_type, access_count, base_importance, half_life_days
            FROM memory_metadata
            ORDER BY message_id
            """
        ).fetchall()
        assert rows == [
            (stale.message_id, "raw_message", 0, 1.0, memory.DEFAULT_HALF_LIFE_DAYS["raw_message"]),
            (fresh.message_id, "raw_message", 0, 1.0, memory.DEFAULT_HALF_LIFE_DAYS["raw_message"]),
        ]
        conn.execute(
            """
            UPDATE memory_metadata
            SET last_accessed_at = '2020-01-01T00:00:00+00:00',
                half_life_days = 1.0
            WHERE message_id = ?
            """,
            (stale.message_id,),
        )

    ranked = store.retrieve("DECAY_TARGET implementation", top_k=2)
    assert ranked[0].message_id == fresh.message_id
    stale_ranked = next(chunk for chunk in ranked if chunk.message_id == stale.message_id)
    fresh_ranked = next(chunk for chunk in ranked if chunk.message_id == fresh.message_id)
    assert stale_ranked.decay_strength < fresh_ranked.decay_strength
    assert stale_ranked.score < fresh_ranked.score

    memory.CONTEXT_TOKEN_BUDGET = 10
    store.retrieve_structured = lambda _query, top_k=memory.STRUCTURED_TOP_K: []
    store.retrieve = lambda _query, top_k=memory.RETRIEVE_TOP_K: [
        RetrievedChunk(
            id="keep",
            text="KEEP_DECAY injected.",
            importance=0.5,
            message_id=fresh.message_id,
            score=0.9,
        ),
        RetrievedChunk(
            id="drop",
            text="DROP_DECAY not injected because budget is tight.",
            importance=0.5,
            message_id=stale.message_id,
            score=0.1,
        ),
    ]
    bundle = store.build_context_bundle("decay touch")
    assert [chunk.id for chunk in bundle.kept] == ["keep"]
    assert [chunk.id for chunk in bundle.would_be_dropped] == ["drop"]

    with sqlite3.connect(store.state_db) as conn:
        conn.execute(
            """
            UPDATE memory_metadata
            SET base_importance = 2.0
            WHERE message_id = ?
            """,
            (fresh.message_id,),
        )
    assert store._decay_strength(
        store._load_memory_metadata({fresh.message_id})[fresh.message_id],
        store._parse_iso_datetime("2026-01-01T00:00:00+00:00"),
    ) == 1.0

    with sqlite3.connect(store.state_db) as conn:
        access = dict(
            conn.execute(
                "SELECT message_id, access_count FROM memory_metadata"
            ).fetchall()
        )
    assert access[fresh.message_id] == 1
    assert access[stale.message_id] == 0

    events = [
        line for line in (DB_PATH / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"event": "memory_accessed"' in line
    ]
    assert '"reason": "context_injected"' in events[-1]
    assert '"citation_evidence": false' in events[-1]
finally:
    memory.RECENT_MESSAGES = original_recent
    memory.CONTEXT_TOKEN_BUDGET = original_budget

print("Decay forgetting test passed.")
