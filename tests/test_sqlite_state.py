"""
Verify raw message state is persisted in SQLite and legacy state.json imports.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_sqlite_state.py
"""
import json
import shutil
from pathlib import Path

from ragmemory.memory import MemoryStore

DB_PATH = Path("./.data/chroma_sqlite_state_test")
MIGRATION_DB_PATH = Path("./.data/chroma_state_migration_test")

for path in (DB_PATH, MIGRATION_DB_PATH):
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

print("SQLite state test passed.")
