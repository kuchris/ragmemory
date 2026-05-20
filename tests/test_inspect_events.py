"""
Verify scripts/inspect_events.py filters event logs.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_inspect_events.py
"""
import contextlib
import io
import json
import shutil
import importlib.util
from pathlib import Path

DB_PATH = Path("./.data/chroma_inspect_events_test")
EVENTS_PATH = DB_PATH / "events.jsonl"
SCRIPT_PATH = Path("scripts/inspect_events.py")

spec = importlib.util.spec_from_file_location("inspect_events", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)
DB_PATH.mkdir(parents=True)

events = [
    {"ts": "2026-01-01T00:00:00+00:00", "event": "message_saved", "message_id": 1},
    {"ts": "2026-01-02T00:00:00+00:00", "event": "chunks_retrieved", "message_ids": [1, 2], "result_count": 2},
    {"ts": "2026-01-03T00:00:00+00:00", "event": "message_saved", "message_id": 3},
]
EVENTS_PATH.write_text(
    "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
    encoding="utf-8",
)


def run_inspector(*args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = module.main(["--db-path", str(DB_PATH), *args])
    assert code == 0
    return output.getvalue()


saved = run_inspector("--event", "message_saved")
assert "message_id=1" in saved
assert "message_id=3" in saved
assert "chunks_retrieved" not in saved

message_two = run_inspector("--message-id", "2")
assert "chunks_retrieved" in message_two
assert "result_count=2" in message_two
assert "message_id=1" not in message_two

since = run_inspector("--since", "2026-01-02T00:00:00+00:00")
assert "2026-01-01" not in since
assert "2026-01-02" in since
assert "2026-01-03" in since

as_json = run_inspector("--event", "message_saved", "--limit", "1", "--json")
record = json.loads(as_json)
assert record["message_id"] == 3

print("Inspect events test passed.")
