"""
Verify confirm=True tombstones message-id forget targets.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_forget_tombstone.py
"""
import json
import shutil
from pathlib import Path

from ragmemory import ForgetResult, MemoryStore
from ragmemory.memory import RetrievedChunk

DB_PATH = Path("./.data/chroma_forget_tombstone_test")
EVENTS_PATH = DB_PATH / "events.jsonl"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
target_text = """TOMBSTONE_TARGET should disappear from active memory.

```json
{"forget": "TOMBSTONE_TARGET"}
```"""
target = store.add_message(
    "user",
    target_text,
)
keep = store.add_message(
    "user",
    "KEEP_ACTIVE should remain searchable after tombstoning another message.",
    extract_structured=False,
)
assert target.message_id is not None
assert keep.message_id is not None

store.ledger.log(
    RetrievedChunk(
        id="ledger-tombstone-target",
        text="Ledger recovery for TOMBSTONE_TARGET should be hidden.",
        importance=0.5,
        message_id=target.message_id,
    )
)

result = store.forget(message_ids=[target.message_id], confirm=True)

assert isinstance(result, ForgetResult)
assert result.message_count == 1
assert result.chunk_count >= 1
assert result.structured_count == 1
assert result.ledger_count == 1
assert result.tombstoned_count == 1
assert result.event_id

assert all(message["message_id"] != target.message_id for message in store.raw_log)
assert all(chunk.message_id != target.message_id for chunk in store.retrieve("TOMBSTONE_TARGET"))
assert all(item.message_id != target.message_id for item in store.search("TOMBSTONE_TARGET"))
assert all(obj.message_id != target.message_id for obj in store.retrieve_structured("TOMBSTONE_TARGET"))

bundle = store.build_context_bundle("as we said TOMBSTONE_TARGET")
assert all(message.message_id != target.message_id for message in bundle.recent)
assert all(chunk.message_id != target.message_id for chunk in bundle.kept)
assert all(chunk.message_id != target.message_id for chunk in bundle.ledger_recovered)
assert all(obj.message_id != target.message_id for obj in bundle.structured)

assert any(chunk.message_id == keep.message_id for chunk in store.retrieve("KEEP_ACTIVE"))

reloaded = MemoryStore(db_path=str(DB_PATH))
assert all(message["message_id"] != target.message_id for message in reloaded.raw_log)
assert all(chunk.message_id != target.message_id for chunk in reloaded.retrieve("TOMBSTONE_TARGET"))

resaved = reloaded.add_message("user", target_text, extract_structured=False)
assert resaved.saved is True
assert resaved.message_id != target.message_id

events = [
    json.loads(line)
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tombstone_event = next(event for event in events if event["event"] == "memory_tombstoned")
assert tombstone_event["event_id"] == result.event_id
assert tombstone_event["message_ids"] == [target.message_id]
assert tombstone_event["message_count"] == 1

print("Forget tombstone test passed.")
