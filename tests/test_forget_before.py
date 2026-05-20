"""
Verify before= forget selects messages by created_at.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_forget_before.py
"""
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ragmemory import ForgetPreview, ForgetResult, MemoryStore

DB_PATH = Path("./.data/chroma_forget_before_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
old = store.add_message("user", "FORGET_BEFORE_OLD should be tombstoned.", extract_structured=False)
edge = store.add_message("user", "FORGET_BEFORE_EDGE is exactly at cutoff.", extract_structured=False)
new = store.add_message("user", "FORGET_BEFORE_NEW should remain active.", extract_structured=False)

assert old.message_id is not None
assert edge.message_id is not None
assert new.message_id is not None

created_at = {
    old.message_id: "2026-01-01T00:00:00+00:00",
    edge.message_id: "2026-01-02T00:00:00+00:00",
    new.message_id: "2026-01-03T00:00:00+00:00",
}
with sqlite3.connect(store.state_db) as conn:
    conn.executemany(
        "UPDATE messages SET created_at = ? WHERE message_id = ?",
        [(timestamp, message_id) for message_id, timestamp in created_at.items()],
    )
for message in store.raw_log:
    message["created_at"] = created_at[message["message_id"]]

preview = store.forget(before="2026-01-02T00:00:00Z")

assert isinstance(preview, ForgetPreview)
assert [message.message_id for message in preview.messages] == [old.message_id]
assert preview.message_count == 1
assert preview.chunk_count == 1
assert preview.truncated is False

result = store.forget(
    before=datetime(2026, 1, 3, tzinfo=timezone.utc),
    confirm=True,
)

assert isinstance(result, ForgetResult)
assert [message.message_id for message in result.messages] == [old.message_id, edge.message_id]
assert result.message_count == 2
assert result.chunk_count == 2
assert result.tombstoned_count == 2

assert all(chunk.message_id != old.message_id for chunk in store.retrieve("FORGET_BEFORE_OLD"))
assert all(chunk.message_id != edge.message_id for chunk in store.retrieve("FORGET_BEFORE_EDGE"))
assert any(chunk.message_id == new.message_id for chunk in store.retrieve("FORGET_BEFORE_NEW"))

reloaded = MemoryStore(db_path=str(DB_PATH))
active_ids = {message["message_id"] for message in reloaded.raw_log}
assert old.message_id not in active_ids
assert edge.message_id not in active_ids
assert new.message_id in active_ids

try:
    store.forget(message_ids=[new.message_id], before="2026-01-04T00:00:00Z")
    raise AssertionError("mixing selectors should fail for now")
except ValueError:
    pass

print("Forget before test passed.")
