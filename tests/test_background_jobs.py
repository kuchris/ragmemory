"""
Verify SQLite-backed background jobs.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_background_jobs.py
"""
import shutil
import sqlite3
from pathlib import Path

from ragmemory import MemoryStore
from ragmemory.memory import (
    JOB_STATUS_DONE,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_TYPE_STRUCTURED_EXTRACT,
)

DB_PATH = Path("./.data/chroma_background_jobs_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

store = MemoryStore(db_path=str(DB_PATH))
manual = store.add_message("user", "manual queue target", extract_structured=False)
job_id = store.enqueue_job(JOB_TYPE_STRUCTURED_EXTRACT, manual.message_id)
duplicate_job_id = store.enqueue_job(JOB_TYPE_STRUCTURED_EXTRACT, manual.message_id)
assert job_id is not None
assert duplicate_job_id is None

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    rows = conn.execute(
        "SELECT job_id, status FROM jobs WHERE message_id = ?",
        (manual.message_id,),
    ).fetchall()
assert rows == [(job_id, JOB_STATUS_PENDING)]

claimed = store.claim_next_job()
assert claimed is not None
assert claimed.job_id == job_id
assert claimed.job_type == JOB_TYPE_STRUCTURED_EXTRACT

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    status = conn.execute(
        "SELECT status FROM jobs WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]
assert status == JOB_STATUS_RUNNING

restarted = MemoryStore(db_path=str(DB_PATH))
assert restarted.reset_running_jobs() == 1

queued = restarted.add_message(
    "user",
    """Background worker should recover this exact config.

```json
{"worker": true}
```""",
    extract_structured="background",
)
assert queued.queued_job_ids

structured_ids = restarted.run_pending_extractions(limit=2)
assert structured_ids
assert any(
    obj.type == "config" and '"worker": true' in obj.source_text
    for obj in restarted.structured.objects.values()
)

with sqlite3.connect(DB_PATH / "state.sqlite") as conn:
    done_count = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = ?",
        (JOB_STATUS_DONE,),
    ).fetchone()[0]
assert done_count == 2

print("Background jobs test passed.")
