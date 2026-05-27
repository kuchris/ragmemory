import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragmemory import MemoryStore
from ragmemory.memory import JOB_STATUS_DONE, JOB_TYPE_STRUCTURED_EXTRACT
from scripts.decay_report import format_markdown, load_rows, sort_rows


DB_PATH = Path("./.data/decay_report_test")


def test_decay_report_sorts_stale_first():
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    store = MemoryStore(db_path=str(DB_PATH))
    stale = store.add_message(
        "user",
        "DECAY_REPORT stale memory.",
        extract_structured=False,
    )
    fresh = store.add_message(
        "user",
        "DECAY_REPORT fresh memory.",
        extract_structured=False,
    )
    with sqlite3.connect(store.state_db) as conn:
        conn.execute(
            """
            UPDATE memory_metadata
            SET last_accessed_at = '2020-01-01T00:00:00+00:00',
                half_life_days = 1.0
            WHERE message_id = ?
            """,
            (stale.message_id,),
        )
        conn.execute(
            """
            INSERT INTO jobs(
                job_id, job_type, message_id, status, attempts,
                created_at, started_at, finished_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                "job_decay_report_done",
                JOB_TYPE_STRUCTURED_EXTRACT,
                stale.message_id,
                JOB_STATUS_DONE,
                "2020-01-01T00:00:00+00:00",
                "2020-01-01T00:00:00+00:00",
                "2020-01-01T00:00:01+00:00",
            ),
        )

    rows = load_rows(store)
    ranked = sort_rows(rows, "stale")

    assert ranked[0].message_id == stale.message_id
    assert ranked[0].extraction_status == "extract_done_empty"
    assert ranked[0].review_candidate is True
    assert ranked[0].decay_strength < next(
        row.decay_strength for row in rows if row.message_id == fresh.message_id
    )
    fresh_row = next(row for row in rows if row.message_id == fresh.message_id)
    assert fresh_row.extraction_status == "unknown_or_disabled"
    assert fresh_row.review_candidate is False
    assert ranked[0].retrieval_multiplier >= 0.3
    markdown = format_markdown(ranked, sort="stale", db_path=DB_PATH)
    assert "# RagMemory Decay Report" in markdown
    assert "| message | role | decay | retrieval_x |" in markdown
    assert "Review candidates: `1`" in markdown
    assert "DECAY_REPORT stale memory." in markdown


if __name__ == "__main__":
    test_decay_report_sorts_stale_first()
    print("Decay report test passed.")
