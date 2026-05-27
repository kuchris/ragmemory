"""
Verify topic regroup jobs are queued on message-count boundaries.

Run:
    uv run python tests/test_topic_regroup_interval.py
"""
import shutil
import sqlite3
import os
from pathlib import Path

import ragmemory.memory as memory_module
from ragmemory.memory import JOB_TYPE_TOPIC_REGROUP, MemoryStore


DB_PATH = Path("./.data/topic_regroup_interval_test")

if DB_PATH.exists():
    shutil.rmtree(DB_PATH, ignore_errors=True)

original_interval = memory_module.TOPIC_REGROUP_MESSAGE_INTERVAL
original_enable = os.environ.get("RAGMEMORY_TOPIC_ENABLE")
memory_module.TOPIC_REGROUP_MESSAGE_INTERVAL = 2
os.environ["RAGMEMORY_TOPIC_ENABLE"] = "true"

try:
    store = MemoryStore(db_path=str(DB_PATH))

    first = store.add_message("user", "first interval message", extract_structured=False)
    assert first.queued_job_ids == []

    second = store.add_message("assistant", "second interval message", extract_structured=False)
    assert len(second.queued_job_ids) == 1

    third = store.add_message("user", "third interval message", extract_structured=False)
    assert third.queued_job_ids == []

    with sqlite3.connect(store.state_db) as conn:
        rows = conn.execute(
            """
            SELECT job_type, message_id, status
            FROM jobs
            WHERE job_type = ?
            """,
            (JOB_TYPE_TOPIC_REGROUP,),
        ).fetchall()
    assert rows == [(JOB_TYPE_TOPIC_REGROUP, 0, "pending")]
finally:
    memory_module.TOPIC_REGROUP_MESSAGE_INTERVAL = original_interval
    if original_enable is None:
        os.environ.pop("RAGMEMORY_TOPIC_ENABLE", None)
    else:
        os.environ["RAGMEMORY_TOPIC_ENABLE"] = original_enable
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH, ignore_errors=True)

print("Topic regroup interval test passed.")
