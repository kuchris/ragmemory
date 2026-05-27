"""
Re-run structured extraction for already-saved RagMemory messages.

Run:
    uv run python scripts/structured_backfill.py --message-ids 560-572 --queue
    uv run python scripts/structured_backfill.py --recent-missing 30 --run
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

from ragmemory import MemoryStore
from ragmemory.memory import JOB_TYPE_STRUCTURED_EXTRACT


def parse_message_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def message_ids_with_structured(store: MemoryStore) -> set[int]:
    return {obj.message_id for obj in store.structured.objects.values()}


def recent_missing_structured(store: MemoryStore, limit: int) -> list[int]:
    with sqlite3.connect(store.state_db) as conn:
        rows = conn.execute(
            """
            SELECT message_id
            FROM messages
            WHERE tombstoned = 0
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (max(0, limit),),
        ).fetchall()
    existing = message_ids_with_structured(store)
    return [message_id for (message_id,) in rows if message_id not in existing]


def print_preview(store: MemoryStore, message_ids: list[int]) -> None:
    if not message_ids:
        print("No candidate messages found.")
        return
    existing = message_ids_with_structured(store)
    with sqlite3.connect(store.state_db) as conn:
        placeholders = ",".join("?" for _ in message_ids)
        rows = conn.execute(
            f"""
            SELECT message_id, role, substr(replace(replace(text, char(13), ' '), char(10), ' '), 1, 140)
            FROM messages
            WHERE message_id IN ({placeholders})
            ORDER BY message_id
            """,
            message_ids,
        ).fetchall()
    print(f"Candidate message(s): {len(rows)}")
    for message_id, role, preview in rows:
        status = "has_structured" if message_id in existing else "missing_structured"
        print(f"  {message_id} | {role} | {status} | {preview}")


def queue_messages(store: MemoryStore, message_ids: list[int]) -> list[int]:
    queued = []
    for message_id in message_ids:
        job_id = store.enqueue_job(JOB_TYPE_STRUCTURED_EXTRACT, message_id)
        if job_id:
            queued.append(message_id)
    return queued


def run_messages(store: MemoryStore, message_ids: list[int]) -> dict[int, list[str]]:
    results: dict[int, list[str]] = {}
    for message_id in message_ids:
        row = store._message_for_job(message_id)
        if row is None:
            results[message_id] = []
            continue
        role, text = row
        objects = store._extract_structured_objects(role, text, message_id)
        store.structured.add_many(objects)
        store._log_structured_objects_added(
            objects,
            reason="structured_backfill",
        )
        results[message_id] = [obj.id for obj in objects]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview, queue, or run structured extraction backfill."
    )
    parser.add_argument("--db-path", default="./.data/chroma_db")
    parser.add_argument("--message-ids", default="", help="Comma/space/range list, e.g. 560-572 or 560,561.")
    parser.add_argument("--recent-missing", type=int, help="Use recent active messages with no structured objects.")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Also rerun messages that already have structured objects.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--queue", action="store_true", help="Queue structured_extract jobs for the worker.")
    mode.add_argument("--run", action="store_true", help="Run extraction immediately in this process.")
    args = parser.parse_args()

    store = MemoryStore(db_path=str(Path(args.db_path)))
    if args.message_ids:
        message_ids = parse_message_ids(args.message_ids)
    elif args.recent_missing is not None:
        message_ids = recent_missing_structured(store, args.recent_missing)
    else:
        parser.error("Provide --message-ids or --recent-missing.")

    print_preview(store, message_ids)
    if not args.queue and not args.run:
        print("Preview only. Add --queue or --run to remake structured extraction.")
        return

    if not args.include_existing:
        existing = message_ids_with_structured(store)
        message_ids = [message_id for message_id in message_ids if message_id not in existing]

    if args.queue:
        queued = queue_messages(store, message_ids)
        print(f"Queued {len(queued)} structured extraction job(s): {', '.join(map(str, queued))}")
        return

    results = run_messages(store, message_ids)
    created = sum(len(ids) for ids in results.values())
    print(f"Created {created} structured object(s).")
    for message_id, object_ids in results.items():
        print(f"  {message_id}: {len(object_ids)}")


if __name__ == "__main__":
    main()
