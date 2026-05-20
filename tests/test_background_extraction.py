"""
Verify in-process background structured extraction.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_background_extraction.py
"""
import json
import shutil
from pathlib import Path

from ragmemory import MemoryStore

DB_PATH = Path("./.data/chroma_background_extraction_test")
EVENTS_PATH = DB_PATH / "events.jsonl"

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
result = store.add_message(
    "user",
    """Background extraction should queue this exact config.

```json
{"background": true, "kind": "exact-artifact"}
```""",
    extract_structured="background",
)

assert result.saved is True
assert result.chunk_ids
assert result.structured_object_ids == []
assert len(result.queued_job_ids) == 1
assert len(store.structured) == 0
assert store.retrieve("exact config")

assert store.run_pending_extractions(limit=0) == []
assert len(store.structured) == 0

structured_ids = store.run_pending_extractions(limit=1)
assert len(structured_ids) == 1
assert len(store.structured) == 1
assert store.run_pending_extractions() == []

obj = next(iter(store.structured.objects.values()))
assert obj.id == structured_ids[0]
assert obj.message_id == result.message_id
assert obj.type == "config"
assert obj.source_text == """```json
{"background": true, "kind": "exact-artifact"}
```"""

events = [
    json.loads(line)
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
event_names = [event["event"] for event in events]
assert "structured_extraction_queued" in event_names
assert "structured_extraction_completed" in event_names

queued = next(event for event in events if event["event"] == "structured_extraction_queued")
completed = next(event for event in events if event["event"] == "structured_extraction_completed")
assert queued["job_id"] == result.queued_job_ids[0]
assert completed["job_id"] == result.queued_job_ids[0]
assert completed["structured_object_ids"] == structured_ids

print("Background extraction test passed.")
