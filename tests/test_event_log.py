"""
Verify append-only JSONL event logging.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_event_log.py
"""
import json
import shutil
from pathlib import Path

from ragmemory.memory import MemoryStore

DB_PATH = Path("./.data/chroma_event_log_test")
EVENTS_PATH = DB_PATH / "events.jsonl"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
first = store.add_message(
    "user",
    "Event log should record saved messages and retrieval.",
    extract_structured=False,
)
duplicate = store.add_message(
    "user",
    " Event log should record saved messages and retrieval. ",
    extract_structured=False,
)
context = store.build_context("what should event log record?")

assert first.saved is True
assert duplicate.deduped is True
assert context
assert EVENTS_PATH.exists()

events = [
    json.loads(line)
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
event_names = [event["event"] for event in events]

assert "message_saved" in event_names
assert "message_deduped" in event_names
assert "chunks_retrieved" in event_names
assert "context_built" in event_names

saved = next(event for event in events if event["event"] == "message_saved")
assert saved["message_id"] == first.message_id
assert saved["chunk_ids"] == first.chunk_ids

deduped = next(event for event in events if event["event"] == "message_deduped")
assert deduped["message_id"] == first.message_id
assert deduped["content_hash"] == first.content_hash

retrieved = next(event for event in events if event["event"] == "chunks_retrieved")
assert retrieved["result_count"] > 0
assert retrieved["chunk_ids"]
assert retrieved["scores"]

context_event = next(event for event in events if event["event"] == "context_built")
assert context_event["kept_count"] == 0
assert context_event["recent_count"] == 1

print("Event log test passed.")
