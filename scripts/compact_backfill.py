"""
Compact already-saved RagMemory messages without touching raw text.

Run:
    uv run python scripts/compact_backfill.py --limit 20
"""
import argparse
from pathlib import Path

from ragmemory import MemoryStore


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
    args = parser.parse_args()

    store = MemoryStore(db_path=str(Path(args.db_path)))
    if not store.compaction_options.enabled:
        print("Compact disabled: set [compact] enable = true in ragmemory.local.ini.")
        return

    message_ids = store.compact_existing_messages(limit=args.limit)
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
