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
    if not message_ids:
        print("No messages compacted.")
        return

    joined_ids = ", ".join(str(message_id) for message_id in message_ids)
    print(f"Compacted {len(message_ids)} message(s): {joined_ids}")


if __name__ == "__main__":
    main()
