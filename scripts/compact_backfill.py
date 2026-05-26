"""
Compact already-saved RagMemory messages without touching raw text.

Run:
    uv run python scripts/compact_backfill.py --limit 20
"""
import argparse
from pathlib import Path

from ragmemory import MemoryStore


def _progress_line(done: int, total: int, status: str) -> str:
    width = 28
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    status = status[:18]
    line = f"Compacting [{bar}] {done}/{total} last={status:<18}"
    return f"\r{line}\x1b[K"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill messages.compact_text for existing raw messages."
    )
    parser.add_argument(
        "--db-path",
        default="./.data/chroma_db",
        help="RagMemory DB directory containing state.sqlite.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum messages to compact in this run.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Skip messages that already have this many compact attempts.",
    )
    args = parser.parse_args()

    store = MemoryStore(db_path=str(Path(args.db_path)))
    if not store.compaction_options.enabled:
        print("Compact disabled: set [compact] enable = true in ragmemory.local.ini.")
        return

    progress_enabled = True

    def show_progress(done: int, total: int, summary: dict) -> None:
        if not progress_enabled:
            return
        status = str(summary.get("status") or "unknown")
        print(_progress_line(done, total, status), end="", flush=True)

    message_ids = store.compact_existing_messages(
        limit=args.limit,
        max_attempts=args.max_attempts,
        progress_callback=show_progress,
    )
    if store.last_compact_backfill_attempts:
        print()
    attempts = store.last_compact_backfill_attempts
    if attempts:
        joined_attempts = ", ".join(str(item["message_id"]) for item in attempts)
        print(f"Attempted {len(attempts)} message(s): {joined_attempts}")

    if message_ids:
        joined_ids = ", ".join(str(message_id) for message_id in message_ids)
        print(f"Compacted {len(message_ids)} message(s): {joined_ids}")
    else:
        print("Compacted 0 message(s).")

    failed = [item for item in attempts if item["status"] != "ok"]
    if failed:
        print("Failures/skips:")
        for item in failed:
            reason = item["reason"] or item["status"]
            print(f"  {item['message_id']}: {reason}")
            missing_tokens = item.get("missing_tokens") or []
            if missing_tokens:
                preview = ", ".join(str(token) for token in missing_tokens[:5])
                if len(missing_tokens) > 5:
                    preview += ", ..."
                print(f"    missing: {preview}")

    if not attempts:
        print("No eligible messages found.")


if __name__ == "__main__":
    main()
