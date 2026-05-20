"""
Verify raw message state is persisted in SQLite and legacy state.json imports.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_sqlite_state.py
"""
import json
import shutil
import sqlite3
from pathlib import Path

from ragmemory.memory import MemoryStore

DB_PATH = Path("./.data/chroma_sqlite_state_test")
MIGRATION_DB_PATH = Path("./.data/chroma_state_migration_test")
UNIQUE_DB_PATH = Path("./.data/chroma_state_unique_migration_test")

for path in (DB_PATH, MIGRATION_DB_PATH, UNIQUE_DB_PATH):
    if path.exists():
        shutil.rmtree(path)

store = MemoryStore(db_path=str(DB_PATH))
first = store.add_message("user", "SQLite should keep raw messages.", extract_structured=False)
second = store.add_message("assistant", "It should reload them after restart.", extract_structured=False)

assert first.message_id == 0
assert second.message_id == 1
assert (DB_PATH / "state.sqlite").exists()
assert not (DB_PATH / "state.json").exists()

reloaded = MemoryStore(db_path=str(DB_PATH))
assert reloaded.message_id == 2
assert len(reloaded.raw_log) == 2
assert reloaded.raw_log[0]["text"] == "SQLite should keep raw messages."

duplicate = reloaded.add_message(
    "user",
    " SQLite   should keep raw messages. ",
    extract_structured=False,
)
assert duplicate.deduped is True
assert duplicate.message_id == 0
assert reloaded.message_id == 2
assert len(reloaded.raw_log) == 2

MIGRATION_DB_PATH.mkdir(parents=True)
(MIGRATION_DB_PATH / "state.json").write_text(
    json.dumps(
        {
            "raw_log": [
                {"role": "user", "text": "Legacy state should import.", "message_id": 0},
                {"role": "assistant", "text": "Without rewriting JSON.", "message_id": 1},
            ],
            "message_id": 2,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

migrated = MemoryStore(db_path=str(MIGRATION_DB_PATH))
assert (MIGRATION_DB_PATH / "state.sqlite").exists()
assert len(migrated.raw_log) == 2
assert migrated.message_id == 2

migrated_duplicate = migrated.add_message(
    "user",
    "Legacy state should import.",
    extract_structured=False,
)
assert migrated_duplicate.deduped is True
assert migrated_duplicate.message_id == 0

UNIQUE_DB_PATH.mkdir(parents=True)
with sqlite3.connect(UNIQUE_DB_PATH / "state.sqlite") as conn:
    conn.execute(
        """
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            tombstoned INTEGER NOT NULL DEFAULT 0,
            UNIQUE(role, content_hash)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO messages(
            message_id, role, text, content_hash, created_at, tombstoned
        ) VALUES (0, 'user', 'old tombstone', 'samehash', '2026-01-01T00:00:00Z', 1)
        """
    )
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta(key, value) VALUES('next_message_id', '1')")

unique_migrated = MemoryStore(db_path=str(UNIQUE_DB_PATH))
with sqlite3.connect(unique_migrated.state_db) as conn:
    conn.execute(
        """
        INSERT INTO messages(
            message_id, role, text, content_hash, created_at, tombstoned
        ) VALUES (1, 'user', 'new active', 'samehash', '2026-01-02T00:00:00Z', 0)
        """
    )

print("SQLite state test passed.")
